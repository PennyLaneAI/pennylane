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
from collections.abc import Callable
from functools import singledispatch, wraps
from typing import Dict, Iterable, List, Set, Union

from pennylane.operation import Operation
from pennylane.queuing import AnnotatedQueue, QueuingManager
from pennylane.wires import Wires

from pennylane.labs.resource_estimation.qubit_manager import (
    QubitManager,
    AllocWires,
    FreeWires,
)
from pennylane.labs.resource_estimation.resources_base import Resources
from pennylane.labs.resource_estimation.resource_mapping import map_to_resource_op
from pennylane.labs.resource_estimation.resource_operator import (
    GateCount,
    ResourceOperator,
    CompressedResourceOp,
)

# pylint: disable=dangerous-default-value,protected-access

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

# practical/realistic gateset for useful compilation of circuits
DefaultGateSet = {
    "X",
    "Y",
    "Z",
    "Hadamard",
    "CNOT",
    "S",
    "T",
    "Adjoint(T)",
    "Adjoint(S)",
    "Toffoli",
}

# parameters for further configuration of the decompositions
resource_config = {
    "error_rx": 1e-3,
    "error_ry": 1e-3,
    "error_rz": 1e-3,
    "precision_multiplexer": 1e-3,
    "precision_qrom_state_prep": 1e-3,
}


@singledispatch
def estimate_resources(
    obj,
    gate_set: Set = DefaultGateSet,
    config: Dict = resource_config,
    work_wires=0,
    tight_budget=False,
    single_qubit_rotation_error=None,
) -> Union[Resources, Callable]:
    r"""Obtain the resources from a quantum circuit or operation in terms of the gates provided
    in the gate_set.

    Args:
        obj (Union[Operation, Callable, QuantumScript]): the quantum circuit or operation to obtain resources from
        gate_set (Set, optional): python set of strings specifying the names of operations to track
        config (Dict, optional): dictionary of additiona; configurations that specify how resources are computed

    Returns:
        Resources: the total resources of the quantum circuit

    Raises:
        TypeError: could not obtain resources for obj of type `type(obj)`

    **Example**

    We can track the resources of a quantum workflow by passing the quantum function defining our workflow directly
    into this function.

    .. code-block:: python

        import copy
        import pennylane.labs.resource_estimation as re

        def my_circuit():
            for w in range(2):
                re.ResourceHadamard(w)

            re.ResourceCNOT([0, 1])
            re.ResourceRX(1.23, 0)
            re.ResourceRY(-4.56, 1)

            re.ResourceQFT(wires=[0, 1, 2])
            return qml.expval(re.ResourceHadamard(2))

    Note that we are passing a python function NOT a :class:`~.QNode`. The resources for this workflow are then obtained by:

    >>> res = re.get_resources(my_circuit)()
    >>> print(res)
    wires: 3
    gates: 202
    gate_types:
    {'Hadamard': 5, 'CNOT': 10, 'T': 187}

    .. details::
        :title: Usage Details

        Users can provide custom gatesets to track resources with. Consider :code:`my_circuit()` from above:

        >>> my_gateset = {"Hadamard", "RX", "RY", "QFT(3)", "CNOT"}
        >>> print(re.get_resources(my_circuit, gate_set = my_gateset)())
        wires: 3
        gates: 6
        gate_types:
        {'Hadamard': 2, 'CNOT': 1, 'RX': 1, 'RY': 1, 'QFT(3)': 1}

        We can also obtain resources for individual operations and quantum tapes in a similar manner:

        >>> op = re.ResourceRX(1.23, 0)
        >>> print(re.get_resources(op))
        wires: 1
        gates: 17
        gate_types:
        {'T': 17}

        Finally, we can modify the config values listed in the global :code:`labs.resource_estimation.resource_config`
        dictionary to have finegrain control of how the resources are computed.

        >>> re.resource_config
        {'error_rx': 0.01, 'error_ry': 0.01, 'error_rz': 0.01}
        >>>
        >>> my_config = copy.copy(re.resource_config)
        >>> my_config["error_rx"] = 0.001
        >>>
        >>> print(re.get_resources(op, config=my_config))
        wires: 1
        gates: 21
        gate_types:
        {'T': 21}

    """

    raise TypeError(
        f"Could not obtain resources for obj of type {type(obj)}. obj must be one of Resources, Callable or ResourceOperator"
    )


@estimate_resources.register
def resources_from_qfunc(
    obj: Callable,
    gate_set: Set = DefaultGateSet,
    config: Dict = resource_config,
    work_wires=0,
    tight_budget=False,
    single_qubit_rotation_error=None,
) -> Callable:
    """Get resources from a quantum function which queues operations"""

    if single_qubit_rotation_error is not None:
        _update_config_single_qubit_rot_error(config, single_qubit_rotation_error)

    @wraps(obj)
    def wrapper(*args, **kwargs):
        with AnnotatedQueue() as q:
            obj(*args, **kwargs)
        
        qm = QubitManager(work_wires, tight_budget)
        # Get algorithm wires:
        num_algo_qubits = 0
        circuit_wires = []
        for op in q.queue:
            if op._queue_category in ["_ops", "_resource_op"]:
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
            _counts_from_compressed_res_op(
                cmp_rep_op, gate_counts, qbit_mngr=qm, gate_set=gate_set, config=config
            )

        return Resources(qubit_manager=qm, gate_types=gate_counts)

    return wrapper


@estimate_resources.register
def resources_from_resource(
    obj: Resources,
    gate_set: Set = DefaultGateSet,
    config: Dict = resource_config,
    work_wires=None,
    tight_budget=None,
    single_qubit_rotation_error=None,
) -> Callable:
    
    if single_qubit_rotation_error is not None:
        _update_config_single_qubit_rot_error(config, single_qubit_rotation_error)

    existing_qm = obj.qubit_manager
    if work_wires is not None:
        if isinstance(work_wires, dict):
            clean_wires = work_wires["clean"]
            dirty_wires = work_wires["dirty"]
        else:
            clean_wires = work_wires
            dirty_wires = 0
    
        existing_qm._clean_qubit_counts = max(clean_wires, existing_qm._clean_qubit_counts)
        existing_qm._dirty_qubit_counts = max(dirty_wires, existing_qm._dirty_qubit_counts)

    if tight_budget is not None:
        existing_qm.tight_budget = tight_budget
    
    gate_counts = defaultdict(int)
    for cmpr_rep_op, count in obj.gate_types.items():
        _counts_from_compressed_res_op(
            cmpr_rep_op, gate_counts, qbit_mngr=existing_qm, gate_set=gate_set, scalar=count, config=config
        )

    # Update:
    return Resources(qubit_manager=existing_qm, gate_types=gate_counts)


@estimate_resources.register
def resources_from_resource_ops(
    obj: ResourceOperator,
    gate_set: Set = DefaultGateSet,
    config: Dict = resource_config,
    work_wires=None,
    tight_budget=None,
    single_qubit_rotation_error=None,
) -> Callable:

    return estimate_resources(
        1*obj, gate_set, config, work_wires, tight_budget, single_qubit_rotation_error
    )


def _counts_from_compressed_res_op(
    cp_rep: CompressedResourceOp,
    gate_counts_dict,
    qbit_mngr,
    gate_set: Set,
    scalar: int = 1,
    config: Dict = resource_config,
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
    if cp_rep._name in gate_set:
        gate_counts_dict[cp_rep] += scalar
        return

    ## Else decompose cp_rep using its resource decomp [cp_rep --> dict[cp_rep: counts]] and extract resources
    resource_decomp = cp_rep.op_type.resource_decomp(config=config, **cp_rep.params)
    qubit_alloc_sum = _sum_allocated_wires(resource_decomp)

    for action in resource_decomp:
        if isinstance(action, GateCount):
            _counts_from_compressed_res_op(
                action.gate,
                gate_counts_dict,
                qbit_mngr=qbit_mngr,
                scalar=scalar * action.count,
                gate_set=gate_set,
                config=config,
            )
            continue

        if isinstance(action, AllocWires):
            if qubit_alloc_sum !=0 and scalar > 1:
                qbit_mngr.grab_clean_qubits(action.num_wires * scalar)
            else:
                qbit_mngr.grab_clean_qubits(action.num_wires)
        if isinstance(action, FreeWires):
            if qubit_alloc_sum !=0 and scalar > 1:
                qbit_mngr.free_qubits(action.num_wires * scalar)
            else:
                qbit_mngr.free_qubits(action.num_wires)

    return


def _sum_allocated_wires(decomp):
    s = 0
    for action in decomp:
        if isinstance(action, AllocWires):
            s += action.num_wires
        if isinstance(action, FreeWires):
            s -= action.num_wires
    return s


def _update_config_single_qubit_rot_error(config, error):
    config["error_rx"] = error
    config["error_ry"] = error
    config["error_rz"] = error
    return


@QueuingManager.stop_recording()
def _ops_to_compressed_reps(
    ops: Iterable[Union[Operation, ResourceOperator]],
) -> List[CompressedResourceOp]:
    """Convert the sequence of operations to a list of compressed resource ops.

    Args:
        ops (Iterable[Operation]): set of operations to convert

    Returns:
        List[CompressedResourceOp]: set of converted compressed resource ops
    """
    cmp_rep_ops = []
    for op in ops:
        if op._queue_category == "_resource_op":
            cmp_rep_ops.append(op.resource_rep_from_op())

        elif op._queue_category == "_ops":
            cmp_rep_ops.append(map_to_resource_op(op))

    return cmp_rep_ops
