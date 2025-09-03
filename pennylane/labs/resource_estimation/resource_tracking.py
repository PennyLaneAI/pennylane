# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""Core resource tracking logic."""
import copy
from collections import defaultdict
from collections.abc import Callable, Iterable
from functools import singledispatch, wraps

from pennylane.labs.resource_estimation.qubit_manager import (
    AllocWires,
    FreeWires,
    QubitManager,
    BorrowWires,
    ReturnWires, 
    NewQubitManager,
    NQMStack,
)
from pennylane.labs.resource_estimation.resource_mapping import map_to_resource_op
from pennylane.labs.resource_estimation.resource_operator import (
    CompressedResourceOp,
    GateCount,
    ResourceOperator,
)
from pennylane.labs.resource_estimation.resources_base import Resources
from pennylane.operation import Operation
from pennylane.queuing import AnnotatedQueue, QueuingManager
from pennylane.wires import Wires

# pylint: disable=dangerous-default-value,protected-access,too-many-arguments

# user-friendly gateset for visual checks and initial compilation
StandardGateSet = {
    "X",
    "Y",
    "Z",
    "Hadamard",
    "SWAP",
    "CNOT",
    "S",
    "T",
    "Adjoint(S)",
    "Adjoint(T)",
    "Toffoli",
    "RX",
    "RY",
    "RZ",
    "PhaseShift",
}

# realistic gateset for useful compilation of circuits
DefaultGateSet = {
    "X",
    "Y",
    "Z",
    "Hadamard",
    "CNOT",
    "S",
    "T",
    "Toffoli",
}

# parameters for further configuration of the decompositions
resource_config = {
    "error_rx": 1e-9,
    "error_ry": 1e-9,
    "error_rz": 1e-9,
    "precision_select_pauli_rot": 1e-9,
    "precision_qubit_unitary": 1e-9,
    "precision_qrom_state_prep": 1e-9,
    "precision_mps_prep": 1e-9,
    "precision_alias_sampling": 1e-9,
    "qubitization_rotation_precision": 15,
    "qubitization_coeff_precision": 15,
}


### NEW TRACKING W/ NEW QM ### -------------------------------------

def new_estimate_resources(
    obj: ResourceOperator | Callable | Resources | list,
    gate_set: set = DefaultGateSet,
    config: dict = resource_config,
    work_wires: int | dict = 0,
    tight_budget: bool = False,
    single_qubit_rotation_error: float | None = None,
    verbose = False,
) -> Resources | Callable:
    r"""Estimate the quantum resources required from a circuit or operation in terms of the gates
    provided in the gateset.

    Args:
        obj (Union[ResourceOperator, Callable, Resources, List]): The quantum circuit or operation
            to obtain resources from.
        gate_set (Set, optional): A set of names (strings) of the fundamental operations to track
            counts for throughout the quantum workflow.
        config (Dict, optional): A dictionary of additional parameters which sets default values
            when they are not specified on the operator.
        single_qubit_rotation_error (Union[float, None]): The acceptable error when decomposing
            single qubit rotations to `T`-gates using a Clifford + T approximation. This value takes
            preference over the values set in the :code:`config`.

    Returns:
        Resources: the quantum resources required to execute the circuit

    Raises:
        TypeError: could not obtain resources for obj of type :code:`type(obj)`

    **Example**

    We can track the resources of a quantum workflow by passing the quantum function defining our
    workflow directly into this function.

    .. code-block:: python

        import pennylane.labs.resource_estimation as plre

        def my_circuit():
            for w in range(2):
                plre.ResourceHadamard(wires=w)

            plre.ResourceCNOT(wires=[0,1])
            plre.ResourceRX(wires=0)
            plre.ResourceRY(wires=1)

            plre.ResourceQFT(num_wires=3, wires=[0, 1, 2])
            return

    Note that we are passing a python function NOT a :class:`~.QNode`. The resources for this
    workflow are then obtained by:

    >>> res = plre.estimate_resources(
    ...     my_circuit,
    ...     gate_set = plre.DefaultGateSet,
    ...     single_qubit_rotation_error = 1e-4,
    ... )()
    ...
    >>> print(res)
    --- Resources: ---
    Total qubits: 3
    Total gates : 279
    Qubit breakdown:
     clean qubits: 0, dirty qubits: 0, algorithmic qubits: 3
    Gate breakdown:
     {'Hadamard': 5, 'CNOT': 10, 'T': 264}

    """

    if single_qubit_rotation_error is not None:
        config = new_update_config_single_qubit_rot_error(config, single_qubit_rotation_error)

    return _new_estimate_resources(obj, gate_set, config, work_wires, tight_budget, verbose)


@singledispatch
def _new_estimate_resources(
    obj: ResourceOperator | Callable | Resources | list,
    gate_set: set = DefaultGateSet,
    config: dict = resource_config,
    work_wires: int | dict = 0,
    tight_budget: bool = False,
    verbose = False,
) -> Resources | Callable:
    r"""Raise error if there is no implementation registered for the object type."""

    raise TypeError(
        f"Could not obtain resources for obj of type {type(obj)}. obj must be one of Resources, Callable or ResourceOperator"
    )


@_new_estimate_resources.register
def new_resources_from_qfunc(
    obj: Callable,
    gate_set: set = DefaultGateSet,
    config: dict = resource_config,
    work_wires=0,
    tight_budget=False,
    verbose = False,
) -> Callable:
    """Get resources from a quantum function which queues operations"""

    @wraps(obj)
    def wrapper(*args, **kwargs):
        with AnnotatedQueue() as q:
            obj(*args, **kwargs)

        qm = NewQubitManager()
        # Get algorithm wires:
        num_algo_qubits = 0
        circuit_wires = []
        for op in q.queue:
            if isinstance(op, (ResourceOperator, Operation)):
                if op.wires:
                    circuit_wires.append(op.wires)
                else:
                    num_algo_qubits = max(num_algo_qubits, op.num_wires)

        num_algo_qubits += len(Wires.all_wires(circuit_wires))
        qm.clean_active_algo = num_algo_qubits  # set the algorithmic qubits in the qubit manager
        if verbose:
            print(qm)

        # Obtain resources in the gate_set
        compressed_res_ops_lst = new_ops_to_compressed_reps(q.queue)

        gate_counts = defaultdict(int)
        for cmp_rep_op in compressed_res_ops_lst:
            if isinstance(cmp_rep_op, BorrowWires):
                if cmp_rep_op.clean:
                    qm.borrow_clean(cmp_rep_op.num_wires)
                else:
                    qm.borrow_dirty(cmp_rep_op.num_wires)
                if verbose:
                    print("-----< qfunc borrow >----")
                    print(qm)

            elif isinstance(cmp_rep_op, ReturnWires):
                if cmp_rep_op.clean:
                    qm.return_clean(cmp_rep_op.num_wires)
                else:
                    qm.return_dirty(cmp_rep_op.num_wires)
                if verbose:
                    print("-----< qfunc return >----")
                    print(qm)
            
            elif isinstance(cmp_rep_op, AllocWires):
                qm.take(cmp_rep_op.num_wires)
                if verbose:
                    print("-----< qfunc allocate >----")
                    print(qm)

            elif isinstance(cmp_rep_op, FreeWires):
                qm.release(cmp_rep_op.num_wires)
                if verbose:
                    print("-----< qfunc free >----")
                    print(qm)

            else:
                if verbose:
                    print(f"-----< qfunc gate: {cmp_rep_op.name} >----")
                    print(qm)
                
                n_wires = cmp_rep_op.params["num_wires"]
                sub_qm = qm.step_into_decomp_simple(num_active_qubits=n_wires)
                
                if verbose:
                    print("|---- step-in >")
                    print(sub_qm)

                new_counts_from_compressed_res_op(
                    cmp_rep_op, gate_counts, qbit_mngr=sub_qm, gate_set=gate_set, config=config, verbose=verbose,
                )

                new_sub_qm = NQMStack.pop()
                assert new_sub_qm is sub_qm
                qm.step_out_decomp_simple(new_sub_qm)
                if verbose:
                    print(f"|---- step-out {cmp_rep_op.name} >")
                    print(qm)

        return Resources(qubit_manager=qm, gate_types=gate_counts)

    return wrapper


# @_new_estimate_resources.register
# def new_resources_from_resource(
#     obj: Resources,
#     gate_set: set = DefaultGateSet,
#     config: dict = resource_config,
#     work_wires=None,
#     tight_budget=None,
# ) -> Resources:
#     """Further process resources from a resources object."""

#     existing_qm = obj.qubit_manager
#     if work_wires is not None:
#         if isinstance(work_wires, dict):
#             clean_wires = work_wires["clean"]
#             dirty_wires = work_wires["dirty"]
#         else:
#             clean_wires = work_wires
#             dirty_wires = 0

#         existing_qm._clean_qubit_counts = max(clean_wires, existing_qm._clean_qubit_counts)
#         existing_qm._dirty_qubit_counts = max(dirty_wires, existing_qm._dirty_qubit_counts)

#     if tight_budget is not None:
#         existing_qm.tight_budget = tight_budget

#     gate_counts = defaultdict(int)
#     for cmpr_rep_op, count in obj.gate_types.items():
#         new_counts_from_compressed_res_op(
#             cmpr_rep_op,
#             gate_counts,
#             qbit_mngr=existing_qm,
#             gate_set=gate_set,
#             scalar=count,
#             config=config,
#         )

#     # Update:
#     return Resources(qubit_manager=existing_qm, gate_types=gate_counts)


# @_new_estimate_resources.register
# def new_resources_from_resource_ops(
#     obj: ResourceOperator,
#     gate_set: set = DefaultGateSet,
#     config: dict = resource_config,
#     work_wires=None,
#     tight_budget=None,
# ) -> Resources:
#     """Extract resources from a resource operator."""
#     if isinstance(obj, Operation):
#         obj = map_to_resource_op(obj)

#     return new_resources_from_resource(
#         1 * obj,
#         gate_set,
#         config,
#         work_wires,
#         tight_budget,
#     )


# @_new_estimate_resources.register
# def new_resources_from_pl_ops(
#     obj: Operation,
#     gate_set: set = DefaultGateSet,
#     config: dict = resource_config,
#     work_wires=None,
#     tight_budget=None,
# ) -> Resources:
#     """Extract resources from a pl operator."""
#     obj = map_to_resource_op(obj)
#     return new_resources_from_resource(
#         1 * obj,
#         gate_set,
#         config,
#         work_wires,
#         tight_budget,
#     )


def new_counts_from_compressed_res_op(
    cp_rep: CompressedResourceOp,
    gate_counts_dict,
    qbit_mngr: NewQubitManager,
    gate_set: set,
    scalar: int = 1,
    config: dict = resource_config,
    verbose = False,
) -> None:
    """Modifies the `gate_counts_dict` argument by adding the (scaled) resources of the operation provided.

    Args:
        cp_rep (CompressedResourceOp): operation in compressed representation to extract resources from
        gate_counts_dict (Dict): base dictionary to modify with the resource counts
        gate_set (Set): the set of operations to track resources with respect to
        scalar (int, optional): optional scalar to multiply the counts. Defaults to 1.
        config (Dict, optional): additional parameters to specify the resources from an operator. Defaults to resource_config.
    """
    ## If op in gate_set add to resources
    if cp_rep.name in gate_set:
        gate_counts_dict[cp_rep] += scalar
        return

    ## Else decompose cp_rep using its resource decomp [cp_rep --> list[GateCounts]] and extract resources
    resource_decomp = cp_rep.op_type.resource_decomp(config=config, **cp_rep.params)

    for action in resource_decomp:
        if isinstance(action, GateCount):
            if verbose:
                print(f"-----< decomp gate: {action.gate.name} >----")
                print(qbit_mngr)

            n_wires = action.gate.params["num_wires"]
            sub_qm = qbit_mngr.step_into_decomp_simple(num_active_qubits=n_wires)
            if verbose:
                print("|---- step-in >")
                print(sub_qm)

            new_counts_from_compressed_res_op(
                action.gate,
                gate_counts_dict,
                qbit_mngr=sub_qm,
                scalar=scalar * action.count,
                gate_set=gate_set,
                config=config,
                verbose=verbose,
            )

            new_sub_qm = NQMStack.pop()
            assert new_sub_qm is sub_qm
            qbit_mngr.step_out_decomp_simple(new_sub_qm)
            if verbose:
                print(f"|---- step-out {action.gate.name} >")
                print(qbit_mngr)

        if isinstance(action, BorrowWires):
            if action.clean:
                qbit_mngr.borrow_clean(action.num_wires)
            else:
                qbit_mngr.borrow_dirty(action.num_wires)
            if verbose:
                print("-----< decomp borrow >----")
                print(qbit_mngr)

        if isinstance(action, ReturnWires):
            if action.clean:
                qbit_mngr.return_clean(action.num_wires)
            else:
                qbit_mngr.return_dirty(action.num_wires)
            if verbose:
                print("-----< decomp return >----")
                print(qbit_mngr)

        if isinstance(action, AllocWires):
            qbit_mngr.take(action.num_wires * scalar)
            if verbose:
                print("-----< decomp alloc >----")
                print(qbit_mngr)
        
        if isinstance(action, FreeWires):
            qbit_mngr.release(action.num_wires * scalar)
            if verbose:
                print("-----< decomp free >----")
                print(qbit_mngr)
    return


def new_sum_allocated_wires(decomp):
    """Sum together the allocated and released wires in a decomposition."""
    s = 0
    for action in decomp:
        if isinstance(action, AllocWires):
            s += action.num_wires
        if isinstance(action, FreeWires):
            s -= action.num_wires
    return s


def new_update_config_single_qubit_rot_error(config, error):
    r"""Create a new config dictionary with the new single qubit
    error threshold.

    Args:
        config (Dict): the configuration dictionary to override
        error (float): the new error threshold to be set

    """
    new_config = copy.copy(config)
    new_config["error_rx"] = error
    new_config["error_ry"] = error
    new_config["error_rz"] = error
    return new_config


@QueuingManager.stop_recording()
def new_ops_to_compressed_reps(
    ops: Iterable[Operation | ResourceOperator],
) -> list[CompressedResourceOp]:
    """Convert the sequence of operations to a list of compressed resource ops.

    Args:
        ops (Iterable[Union[Operation, ResourceOperator]]): set of operations to convert

    Returns:
        List[CompressedResourceOp]: set of converted compressed resource ops
    """
    cmp_rep_ops = []
    for op in ops:  # We are skipping measurement processes here.
        if isinstance(op, (AllocWires, BorrowWires, ReturnWires)):
            cmp_rep_ops.append(op)

        if isinstance(op, ResourceOperator):
            cmp_rep_ops.append(op.resource_rep_from_op())

        if isinstance(op, Operation):  # map: op --> res_op, then: res_op --> cmprsd_res_op
            cmp_rep_ops.append(map_to_resource_op(op).resource_rep_from_op())

    return cmp_rep_ops


### ----------------------------------------------------------------------------------------------


# def estimate_resources(
#     obj: ResourceOperator | Callable | Resources | list,
#     gate_set: set = DefaultGateSet,
#     config: dict = resource_config,
#     work_wires: int | dict = 0,
#     tight_budget: bool = False,
#     single_qubit_rotation_error: float | None = None,
# ) -> Resources | Callable:
#     r"""Estimate the quantum resources required from a circuit or operation in terms of the gates
#     provided in the gateset.

#     Args:
#         obj (Union[ResourceOperator, Callable, Resources, List]): The quantum circuit or operation
#             to obtain resources from.
#         gate_set (Set, optional): A set of names (strings) of the fundamental operations to track
#             counts for throughout the quantum workflow.
#         config (Dict, optional): A dictionary of additional parameters which sets default values
#             when they are not specified on the operator.
#         single_qubit_rotation_error (Union[float, None]): The acceptable error when decomposing
#             single qubit rotations to `T`-gates using a Clifford + T approximation. This value takes
#             preference over the values set in the :code:`config`.

#     Returns:
#         Resources: the quantum resources required to execute the circuit

#     Raises:
#         TypeError: could not obtain resources for obj of type :code:`type(obj)`

#     **Example**

#     We can track the resources of a quantum workflow by passing the quantum function defining our
#     workflow directly into this function.

#     .. code-block:: python

#         import pennylane.labs.resource_estimation as plre

#         def my_circuit():
#             for w in range(2):
#                 plre.ResourceHadamard(wires=w)

#             plre.ResourceCNOT(wires=[0,1])
#             plre.ResourceRX(wires=0)
#             plre.ResourceRY(wires=1)

#             plre.ResourceQFT(num_wires=3, wires=[0, 1, 2])
#             return

#     Note that we are passing a python function NOT a :class:`~.QNode`. The resources for this
#     workflow are then obtained by:

#     >>> res = plre.estimate_resources(
#     ...     my_circuit,
#     ...     gate_set = plre.DefaultGateSet,
#     ...     single_qubit_rotation_error = 1e-4,
#     ... )()
#     ...
#     >>> print(res)
#     --- Resources: ---
#     Total qubits: 3
#     Total gates : 279
#     Qubit breakdown:
#      clean qubits: 0, dirty qubits: 0, algorithmic qubits: 3
#     Gate breakdown:
#      {'Hadamard': 5, 'CNOT': 10, 'T': 264}

#     """

#     if single_qubit_rotation_error is not None:
#         config = _update_config_single_qubit_rot_error(config, single_qubit_rotation_error)

#     return _estimate_resources(obj, gate_set, config, work_wires, tight_budget)


# @singledispatch
# def _estimate_resources(
#     obj: ResourceOperator | Callable | Resources | list,
#     gate_set: set = DefaultGateSet,
#     config: dict = resource_config,
#     work_wires: int | dict = 0,
#     tight_budget: bool = False,
# ) -> Resources | Callable:
#     r"""Raise error if there is no implementation registered for the object type."""

#     raise TypeError(
#         f"Could not obtain resources for obj of type {type(obj)}. obj must be one of Resources, Callable or ResourceOperator"
#     )


# @_estimate_resources.register
# def resources_from_qfunc(
#     obj: Callable,
#     gate_set: set = DefaultGateSet,
#     config: dict = resource_config,
#     work_wires=0,
#     tight_budget=False,
# ) -> Callable:
#     """Get resources from a quantum function which queues operations"""

#     @wraps(obj)
#     def wrapper(*args, **kwargs):
#         with AnnotatedQueue() as q:
#             obj(*args, **kwargs)

#         qm = QubitManager(work_wires, tight_budget)
#         # Get algorithm wires:
#         num_algo_qubits = 0
#         circuit_wires = []
#         for op in q.queue:
#             if isinstance(op, (ResourceOperator, Operation)):
#                 if op.wires:
#                     circuit_wires.append(op.wires)
#                 else:
#                     num_algo_qubits = max(num_algo_qubits, op.num_wires)

#         num_algo_qubits += len(Wires.all_wires(circuit_wires))
#         qm.algo_qubits = num_algo_qubits  # set the algorithmic qubits in the qubit manager

#         # Obtain resources in the gate_set
#         compressed_res_ops_lst = _ops_to_compressed_reps(q.queue)

#         gate_counts = defaultdict(int)
#         for cmp_rep_op in compressed_res_ops_lst:
#             _counts_from_compressed_res_op(
#                 cmp_rep_op, gate_counts, qbit_mngr=qm, gate_set=gate_set, config=config
#             )

#         return Resources(qubit_manager=qm, gate_types=gate_counts)

#     return wrapper


# @_estimate_resources.register
# def resources_from_resource(
#     obj: Resources,
#     gate_set: set = DefaultGateSet,
#     config: dict = resource_config,
#     work_wires=None,
#     tight_budget=None,
# ) -> Resources:
#     """Further process resources from a resources object."""

#     existing_qm = obj.qubit_manager
#     if work_wires is not None:
#         if isinstance(work_wires, dict):
#             clean_wires = work_wires["clean"]
#             dirty_wires = work_wires["dirty"]
#         else:
#             clean_wires = work_wires
#             dirty_wires = 0

#         existing_qm._clean_qubit_counts = max(clean_wires, existing_qm._clean_qubit_counts)
#         existing_qm._dirty_qubit_counts = max(dirty_wires, existing_qm._dirty_qubit_counts)

#     if tight_budget is not None:
#         existing_qm.tight_budget = tight_budget

#     gate_counts = defaultdict(int)
#     for cmpr_rep_op, count in obj.gate_types.items():
#         _counts_from_compressed_res_op(
#             cmpr_rep_op,
#             gate_counts,
#             qbit_mngr=existing_qm,
#             gate_set=gate_set,
#             scalar=count,
#             config=config,
#         )

#     # Update:
#     return Resources(qubit_manager=existing_qm, gate_types=gate_counts)


# @_estimate_resources.register
# def resources_from_resource_ops(
#     obj: ResourceOperator,
#     gate_set: set = DefaultGateSet,
#     config: dict = resource_config,
#     work_wires=None,
#     tight_budget=None,
# ) -> Resources:
#     """Extract resources from a resource operator."""
#     if isinstance(obj, Operation):
#         obj = map_to_resource_op(obj)

#     return resources_from_resource(
#         1 * obj,
#         gate_set,
#         config,
#         work_wires,
#         tight_budget,
#     )


# @_estimate_resources.register
# def resources_from_pl_ops(
#     obj: Operation,
#     gate_set: set = DefaultGateSet,
#     config: dict = resource_config,
#     work_wires=None,
#     tight_budget=None,
# ) -> Resources:
#     """Extract resources from a pl operator."""
#     obj = map_to_resource_op(obj)
#     return resources_from_resource(
#         1 * obj,
#         gate_set,
#         config,
#         work_wires,
#         tight_budget,
#     )


# def _counts_from_compressed_res_op(
#     cp_rep: CompressedResourceOp,
#     gate_counts_dict,
#     qbit_mngr,
#     gate_set: set,
#     scalar: int = 1,
#     config: dict = resource_config,
# ) -> None:
#     """Modifies the `gate_counts_dict` argument by adding the (scaled) resources of the operation provided.

#     Args:
#         cp_rep (CompressedResourceOp): operation in compressed representation to extract resources from
#         gate_counts_dict (Dict): base dictionary to modify with the resource counts
#         gate_set (Set): the set of operations to track resources with respect to
#         scalar (int, optional): optional scalar to multiply the counts. Defaults to 1.
#         config (Dict, optional): additional parameters to specify the resources from an operator. Defaults to resource_config.
#     """
#     ## If op in gate_set add to resources
#     if cp_rep.name in gate_set:
#         gate_counts_dict[cp_rep] += scalar
#         return

#     ## Else decompose cp_rep using its resource decomp [cp_rep --> list[GateCounts]] and extract resources
#     resource_decomp = cp_rep.op_type.resource_decomp(config=config, **cp_rep.params)
#     qubit_alloc_sum = _sum_allocated_wires(resource_decomp)

#     for action in resource_decomp:
#         if isinstance(action, GateCount):
#             _counts_from_compressed_res_op(
#                 action.gate,
#                 gate_counts_dict,
#                 qbit_mngr=qbit_mngr,
#                 scalar=scalar * action.count,
#                 gate_set=gate_set,
#                 config=config,
#             )
#             continue

#         if isinstance(action, AllocWires):
#             if qubit_alloc_sum != 0 and scalar > 1:
#                 qbit_mngr.grab_clean_qubits(action.num_wires * scalar)
#             else:
#                 qbit_mngr.grab_clean_qubits(action.num_wires)
#         if isinstance(action, FreeWires):
#             if qubit_alloc_sum != 0 and scalar > 1:
#                 qbit_mngr.free_qubits(action.num_wires * scalar)
#             else:
#                 qbit_mngr.free_qubits(action.num_wires)

#     return


# def _sum_allocated_wires(decomp):
#     """Sum together the allocated and released wires in a decomposition."""
#     s = 0
#     for action in decomp:
#         if isinstance(action, AllocWires):
#             s += action.num_wires
#         if isinstance(action, FreeWires):
#             s -= action.num_wires
#     return s


# def _update_config_single_qubit_rot_error(config, error):
#     r"""Create a new config dictionary with the new single qubit
#     error threshold.

#     Args:
#         config (Dict): the configuration dictionary to override
#         error (float): the new error threshold to be set

#     """
#     new_config = copy.copy(config)
#     new_config["error_rx"] = error
#     new_config["error_ry"] = error
#     new_config["error_rz"] = error
#     return new_config


# @QueuingManager.stop_recording()
# def _ops_to_compressed_reps(
#     ops: Iterable[Operation | ResourceOperator],
# ) -> list[CompressedResourceOp]:
#     """Convert the sequence of operations to a list of compressed resource ops.

#     Args:
#         ops (Iterable[Union[Operation, ResourceOperator]]): set of operations to convert

#     Returns:
#         List[CompressedResourceOp]: set of converted compressed resource ops
#     """
#     cmp_rep_ops = []
#     for op in ops:  # We are skipping measurement processes here.
#         if isinstance(op, ResourceOperator):
#             cmp_rep_ops.append(op.resource_rep_from_op())

#         if isinstance(op, Operation):  # map: op --> res_op, then: res_op --> cmprsd_res_op
#             cmp_rep_ops.append(map_to_resource_op(op).resource_rep_from_op())

#     return cmp_rep_ops
