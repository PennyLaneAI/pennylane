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
from collections import defaultdict
from collections.abc import Callable, Iterable
from functools import singledispatch, wraps

from pennylane.labs.resource_estimation.ops.op_math.symbolic import (
    ResourceAdjoint,
    ResourceControlled,
    ResourcePow,
)
from pennylane.labs.resource_estimation.qubit_manager import AllocWires, FreeWires, QubitManager
from pennylane.labs.resource_estimation.resource_config import ResourceConfig
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

# pylent: disable=protected-access,too-many-arguments

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


@singledispatch
def estimate_resources(
    obj: ResourceOperator | Callable | Resources,
    gate_set: set = None,
    config: ResourceConfig = ResourceConfig(),
    work_wires: int | dict = 0,
    tight_budget: bool = False,
) -> Resources | Callable:
    r"""Estimate the quantum resources required for a circuit or operation.

    The resources are estimated in terms of a fundamental gate set. This function can be used to
    track resources for quantum functions, individual operators, or existing resource summaries.

    Args:
        obj (ResourceOperator | Callable | Resources): The quantum circuit, operation, or
            existing resource summary to analyze.
        gate_set (set[str], optional): A set of operation names to be considered fundamental.
            Counts will be tracked for these gates. Defaults to ``DefaultGateSet``.
        config (ResourceConfig, optional): A configuration object used to specify default
            parameters, such as error tolerances for rotations, and to register custom
            decomposition logic for operators.
        work_wires (int | dict, optional): The number of available clean and/or dirty ancilla
            qubits. If an integer is provided, it specifies the number of clean ancillas. If a
            dictionary is provided, it should have the keys ``"clean"`` and ``"dirty"``.
            Defaults to 0.
        tight_budget (bool, optional): If ``True``, the qubit manager will prioritize reusing
            qubits over allocating new ones, even if it means cleaning dirty qubits.
            Defaults to ``False``.

    Returns:
        Resources | Callable: If the input is an operator or a ``Resources`` object, this function
        returns a new ``Resources`` object summarizing the gate counts and qubit requirements. If the
        input is a callable quantum function, it returns a wrapped version of that function which,
        when executed, will return the ``Resources`` object.

    Raises:
        TypeError: If the input object ``obj`` is of a type that is not supported.

    **Example**

    We can track the resources of a quantum function by passing it directly into this function.
    Note that we are passing a Python function, not a QNode.

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

    >>> res_fn = plre.estimate_resources(
    ...     my_circuit,
    ...     gate_set=plre.DefaultGateSet
    ... )
    >>> resources = res_fn()
    >>> print(resources)
    --- Resources: ---
    Total qubits: 3
    Total gates : 280
    Qubit breakdown:
     clean qubits: 0, dirty qubits: 0, algorithmic qubits: 3
    Gate breakdown:
     {'Hadamard': 6, 'CNOT': 10, 'T': 264}

    """
    raise TypeError(
        f"Could not obtain resources for obj of type {type(obj)}. obj must be one of Resources, Callable or ResourceOperator"
    )


@estimate_resources.register
def resources_from_qfunc(
    obj: Callable,
    gate_set: set = None,
    config: ResourceConfig = ResourceConfig(),
    work_wires: int | dict = 0,
    tight_budget: bool = False,
) -> Callable:
    """Get resources from a quantum function that queues operations.

    Args:
        obj (ResourceOperator | Callable | Resources): The quantum circuit, operation, or
            existing resource summary to analyze.
        gate_set (set[str], optional): A set of operation names to be considered fundamental.
            Counts will be tracked for these gates. Defaults to ``DefaultGateSet``.
        config (ResourceConfig, optional): A configuration object used to specify default
            parameters, such as error tolerances for rotations, and to register custom
            decomposition logic for operators.
        work_wires (int | dict, optional): The number of available clean and/or dirty ancilla
            qubits. If an integer is provided, it specifies the number of clean ancillas. If a
            dictionary is provided, it should have the keys ``"clean"`` and ``"dirty"``.
            Defaults to 0.
        tight_budget (bool, optional): If ``True``, the qubit manager will prioritize reusing
            qubits over allocating new ones, even if it means cleaning dirty qubits.
            Defaults to ``False``.

    Returns:
        Callable: A wrapped function that, when called, returns the resource summary.
    """

    @wraps(obj)
    def wrapper(*args, **kwargs):
        with AnnotatedQueue() as q:
            obj(*args, **kwargs)

        qm = QubitManager(work_wires, tight_budget)
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
        qm.algo_qubits = num_algo_qubits  # set the algorithmic qubits in the qubit manager

        # Obtain resources in the gate_set
        compressed_res_ops_lst = _ops_to_compressed_reps(q.queue)

        gate_counts = defaultdict(int)
        for cmp_rep_op in compressed_res_ops_lst:
            _update_counts_from_compressed_res_op(
                cmp_rep_op, gate_counts, qbit_mngr=qm, gate_set=gate_set, config=config
            )

        return Resources(qubit_manager=qm, gate_types=gate_counts)

    return wrapper


@estimate_resources.register
def resources_from_resource(
    obj: Resources,
    gate_set: set = None,
    config: ResourceConfig = ResourceConfig(),
    work_wires: int | dict = None,
    tight_budget: bool = None,
) -> Resources:
    """Further process and decompose an existing Resources object.

    Args:
        obj (ResourceOperator | Callable | Resources): The quantum circuit, operation, or
            existing resource summary to analyze.
        gate_set (set[str], optional): A set of operation names to be considered fundamental.
            Counts will be tracked for these gates. Defaults to ``DefaultGateSet``.
        config (ResourceConfig, optional): A configuration object used to specify default
            parameters, such as error tolerances for rotations, and to register custom
            decomposition logic for operators.
        work_wires (int | dict, optional): The number of available clean and/or dirty ancilla
            qubits. If an integer is provided, it specifies the number of clean ancillas. If a
            dictionary is provided, it should have the keys ``"clean"`` and ``"dirty"``.
            Defaults to 0.
        tight_budget (bool, optional): If ``True``, the qubit manager will prioritize reusing
            qubits over allocating new ones, even if it means cleaning dirty qubits.
            Defaults to ``False``.

    Returns:
        Resources: A new, updated resource summary.
    """
    existing_qm = obj.qubit_manager
    if work_wires is not None:
        if isinstance(work_wires, dict):
            clean_wires = work_wires.get("clean", 0)
            dirty_wires = work_wires.get("dirty", 0)
        else:
            clean_wires = work_wires
            dirty_wires = 0

        existing_qm._clean_qubit_counts = max(clean_wires, existing_qm._clean_qubit_counts)
        existing_qm._dirty_qubit_counts = max(dirty_wires, existing_qm._dirty_qubit_counts)

    if tight_budget is not None:
        existing_qm.tight_budget = tight_budget

    gate_counts = defaultdict(int)
    for cmpr_rep_op, count in obj.gate_types.items():
        _update_counts_from_compressed_res_op(
            cmpr_rep_op,
            gate_counts,
            qbit_mngr=existing_qm,
            gate_set=gate_set,
            scalar=count,
            config=config,
        )

    return Resources(qubit_manager=existing_qm, gate_types=gate_counts)


@estimate_resources.register
def resources_from_resource_ops(
    obj: ResourceOperator,
    gate_set: set = None,
    config: ResourceConfig = ResourceConfig(),
    work_wires: int | dict = None,
    tight_budget: bool = None,
) -> Resources:
    """Extract resources from a single resource operator.

    Args:
        obj (ResourceOperator | Callable | Resources): The quantum circuit, operation, or
            existing resource summary to analyze.
        gate_set (set[str], optional): A set of operation names to be considered fundamental.
            Counts will be tracked for these gates. Defaults to ``DefaultGateSet``.
        config (ResourceConfig, optional): A configuration object used to specify default
            parameters, such as error tolerances for rotations, and to register custom
            decomposition logic for operators.
        work_wires (int | dict, optional): The number of available clean and/or dirty ancilla
            qubits. If an integer is provided, it specifies the number of clean ancillas. If a
            dictionary is provided, it should have the keys ``"clean"`` and ``"dirty"``.
            Defaults to 0.
        tight_budget (bool, optional): If ``True``, the qubit manager will prioritize reusing
            qubits over allocating new ones, even if it means cleaning dirty qubits.
            Defaults to ``False``.

    Returns:
        Resources: A resource summary for the operator.
    """
    if isinstance(obj, Operation):
        obj = map_to_resource_op(obj)

    return resources_from_resource(
        1 * obj,
        gate_set,
        config,
        work_wires,
        tight_budget,
    )


@estimate_resources.register
def resources_from_pl_ops(
    obj: Operation,
    gate_set: set = None,
    config: ResourceConfig = ResourceConfig(),
    work_wires: int | dict = None,
    tight_budget: bool = None,
) -> Resources:
    """Extract resources from a single PennyLane operator.

    Args:
        obj (ResourceOperator | Callable | Resources): The quantum circuit, operation, or
            existing resource summary to analyze.
        gate_set (set[str], optional): A set of operation names to be considered fundamental.
            Counts will be tracked for these gates. Defaults to ``DefaultGateSet``.
        config (ResourceConfig, optional): A configuration object used to specify default
            parameters, such as error tolerances for rotations, and to register custom
            decomposition logic for operators.
        work_wires (int | dict, optional): The number of available clean and/or dirty ancilla
            qubits. If an integer is provided, it specifies the number of clean ancillas. If a
            dictionary is provided, it should have the keys ``"clean"`` and ``"dirty"``.
            Defaults to 0.
        tight_budget (bool, optional): If ``True``, the qubit manager will prioritize reusing
            qubits over allocating new ones, even if it means cleaning dirty qubits.
            Defaults to ``False``.

    Returns:
        Resources: A resource summary for the operator.
    """
    obj = map_to_resource_op(obj)
    return resources_from_resource(
        1 * obj,
        gate_set,
        config,
        work_wires,
        tight_budget,
    )


def _update_counts_from_compressed_res_op(
    cp_rep: CompressedResourceOp,
    gate_counts_dict: defaultdict[CompressedResourceOp, int],
    qbit_mngr: QubitManager,
    gate_set: set = None,
    scalar: int = 1,
    config: ResourceConfig = ResourceConfig(),
) -> None:
    """Recursively update a gate count dictionary from a compressed resource operator.

    This function modifies the ``gate_counts_dict`` in place. If the operator is in the target
    gate set, its count is updated. Otherwise, the function recursively calls itself on the
    operator's decomposition.

    Args:
        cp_rep (CompressedResourceOp): the compressed representation of the operator
        gate_counts_dict (defaultdict): the dictionary to modify with resource counts
        qbit_mngr (QubitManager): the qubit manager for tracking allocations
        gate_set (set, optional): the set of fundamental operations to track.
        scalar (int, optional): A scalar to multiply the counts by. Defaults to 1.
        config (ResourceConfig, optional): configuration for specifying default parameters like
            precision and custom operator decompositions
    """
    if gate_set is None:
        gate_set = DefaultGateSet

    if cp_rep.name in gate_set:
        gate_counts_dict[cp_rep] += scalar
        return

    # Else, decompose cp_rep using its resource decomposition and extract resources.
    kwargs = config.conf.get(cp_rep.op_type, {})
    if cp_rep.op_type in (ResourceAdjoint, ResourceControlled, ResourcePow):
        base_op_type = cp_rep.params["base_cmpr_op"].op_type
        kwargs = config.conf.get(base_op_type, {})

    params = {key: value for key, value in cp_rep.params.items() if value is not None}
    filtered_kwargs = {key: value for key, value in kwargs.items() if key not in params}

    resource_decomp = cp_rep.op_type.default_resource_decomp(**params, **filtered_kwargs)
    qubit_alloc_sum = _sum_allocated_wires(resource_decomp)

    for action in resource_decomp:
        if isinstance(action, GateCount):
            _update_counts_from_compressed_res_op(
                action.gate,
                gate_counts_dict,
                qbit_mngr=qbit_mngr,
                scalar=scalar * action.count,
                gate_set=gate_set,
                config=config,
            )
            continue

        if isinstance(action, AllocWires):
            num_to_alloc = action.num_wires * scalar if qubit_alloc_sum != 0 and scalar > 1 else action.num_wires
            qbit_mngr.grab_clean_qubits(num_to_alloc)
        if isinstance(action, FreeWires):
            num_to_free = action.num_wires * scalar if qubit_alloc_sum != 0 and scalar > 1 else action.num_wires
            qbit_mngr.free_qubits(num_to_free)


def _sum_allocated_wires(decomp: list) -> int:
    """Sum the allocated and released wires in a decomposition."""
    return sum(
        action.num_wires if isinstance(action, AllocWires) else -action.num_wires
        for action in decomp
        if isinstance(action, (AllocWires, FreeWires))
    )


@QueuingManager.stop_recording()
def _ops_to_compressed_reps(
    ops: Iterable[Operation | ResourceOperator],
) -> list[CompressedResourceOp]:
    """Convert a sequence of operations to a list of compressed resource representations.

    Args:
        ops (Iterable[Operation | ResourceOperator]): the operations to convert

    Returns:
        list[CompressedResourceOp]: A list of the converted compressed resource operators.
    """
    cmp_rep_ops = []
    for op in ops:
        if isinstance(op, ResourceOperator):
            cmp_rep_ops.append(op.resource_rep_from_op())
        elif isinstance(op, Operation):
            res_op = map_to_resource_op(op)
            cmp_rep_ops.append(res_op.resource_rep_from_op())
    return cmp_rep_ops
