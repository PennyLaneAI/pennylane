# Copyright 2024 Xanadu Quantum Technologies Inc.

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
from typing import Dict, Iterable, List, Set

import pennylane as qml
from pennylane.measurements import MeasurementProcess
from pennylane.operation import Operation
from pennylane.queuing import AnnotatedQueue
from pennylane.tape import QuantumScript
from pennylane.wires import Wires

from .resource_constructor import ResourceConstructor
from .resource_container import CompressedResourceOp, Resources

# pylint: disable=dangerous-default-value,protected-access

_StandardGateSet = {
    "PauliX",
    "PauliY",
    "PauliZ",
    "Hadamard",
    "SWAP",
    "CNOT",
    "S",
    "T",
    "Toffoli",
    "RX",
    "RY",
    "RZ",
    "PhaseShift",
}


DefaultGateSet = {
    "Hadamard",
    "CNOT",
    "S",
    "T",
    "Toffoli",
}


resource_config = {
    "error_rx": 10e-3,
    "error_ry": 10e-3,
    "error_rz": 10e-3,
}


@singledispatch
def get_resources(obj, gate_set: Set = DefaultGateSet, config: Dict = resource_config) -> Resources:
    """Obtain the resources from a quantum circuit or operation in terms of the gates provided
    in the gate_set.

    Args:
        obj (Union[Operation, Callable, QuantumScript]): The quantum circuit or operation to obtain resources from.
        gate_set (Set, optional): A set (str) specifying the names of opertions to track. Defaults to DefaultGateSet.
        config (Dict, optional): Additional configurations to specify how resources are tracked. Defaults to resource_config.

    Returns:
        Resources: The total resources of the quantum circuit.

    Rasies:
        TypeError: "Could not obtain resources for obj of type (type(obj)).
    """

    raise TypeError(
        f"Could not obtain resources for obj of type {type(obj)}. obj must be one of Operation, Callable or QuantumScript"
    )


@get_resources.register
def resources_from_operation(
    obj: Operation, gate_set: Set = DefaultGateSet, config: Dict = resource_config
) -> Resources:
    """Get resources from an operation"""

    if isinstance(obj, ResourceConstructor):
        cp_rep = obj.resource_rep_from_op()
        gate_counts_dict = defaultdict(int)
        _counts_from_compressed_res_op(cp_rep, gate_counts_dict, gate_set=gate_set, config=config)
        return Resources(gate_types=gate_counts_dict)

    res = Resources()  # TODO: Add implementation here!
    return res


@get_resources.register
def resources_from_qfunc(
    obj: Callable, gate_set: Set = DefaultGateSet, config: Dict = resource_config
) -> Resources:
    """Get resources from a quantum function which queues operations!"""

    @wraps(obj)
    def wrapper(*args, **kwargs):
        with AnnotatedQueue() as q:
            obj(*args, **kwargs)

        operations = tuple(op for op in q.queue if not isinstance(op, MeasurementProcess))
        compressed_res_ops_lst = _operations_to_compressed_reps(operations)

        gate_counts_dict = defaultdict(int)
        for cmp_rep_op in compressed_res_ops_lst:
            _counts_from_compressed_res_op(
                cmp_rep_op, gate_counts_dict, gate_set=_StandardGateSet, config=config
            )

        # Validation:
        condensed_gate_counts = defaultdict(int)
        for sub_cmp_rep, counts in gate_counts_dict.items():
            _counts_from_compressed_res_op(
                sub_cmp_rep, condensed_gate_counts, scalar=counts, gate_set=gate_set, config=config
            )

        clean_gate_counts = _clean_gate_counts(condensed_gate_counts)
        num_gates = sum(clean_gate_counts.values())
        num_wires = len(Wires.shared_wires(tuple(op.wires for op in operations)))
        return Resources(num_wires=num_wires, num_gates=num_gates, gate_types=clean_gate_counts)

    return wrapper


@get_resources.register
def resources_from_tape(
    obj: QuantumScript, gate_set: Set = DefaultGateSet, config: Dict = resource_config
) -> Resources:
    """Get resources from a quantum tape"""
    num_wires = obj.num_wires
    operations = obj.operations
    compressed_res_ops_lst = _operations_to_compressed_reps(operations)

    gate_counts_dict = defaultdict(int)
    for cmp_rep_op in compressed_res_ops_lst:
        _counts_from_compressed_res_op(
            cmp_rep_op, gate_counts_dict, gate_set=_StandardGateSet, config=config
        )

    # Validation:
    condensed_gate_counts = defaultdict(int)
    for sub_cmp_rep, counts in gate_counts_dict.items():
        _counts_from_compressed_res_op(
            sub_cmp_rep, condensed_gate_counts, scalar=counts, gate_set=gate_set, config=config
        )

    clean_gate_counts = _clean_gate_counts(condensed_gate_counts)
    num_gates = sum(clean_gate_counts.values())

    return Resources(num_wires=num_wires, num_gates=num_gates, gate_types=clean_gate_counts)


def _counts_from_compressed_res_op(
    cp_rep: CompressedResourceOp,
    gate_counts_dict,
    gate_set: Set,
    scalar: int = 1,
    config: Dict = resource_config,
) -> None:
    """Modifies the `gate_counts_dict` argument by adding the (scaled) resources of the operation provided.

    Args:
        cp_rep (CompressedResourceOp): operation in compressed representation to extract resources from
        gate_counts_dict (_type_): base dictionary to modify with the resource counts
        gate_set (Set): the set of operations to track resources with respect too
        scalar (int, optional): optional scalar to multiply the counts. Defaults to 1.
        config (Dict, optional): additional parameters to specify the resources from an operator. Defaults to resource_config.
    """
    ## If op in gate_set add to resources
    if cp_rep._name in gate_set:
        gate_counts_dict[cp_rep] += scalar
        return

    ## Else decompose cp_rep using its resource decomp [cp_rep --> dict[cp_rep: counts]] and extract resources
    resource_decomp = cp_rep.op_type.resources(config=config, **cp_rep.params)

    for sub_cp_rep, counts in resource_decomp.items():
        _counts_from_compressed_res_op(
            sub_cp_rep, gate_counts_dict, scalar=scalar * counts, gate_set=gate_set, config=config
        )

    return


def _temp_map_func(op: Operation) -> ResourceConstructor:
    """Temp map function"""
    raise NotImplementedError


def _clean_gate_counts(gate_counts: Dict[CompressedResourceOp, int]) -> Dict[str, int]:
    """Map resources with gate_types made from CompressedResourceOps
    into one which tracks just strings of operations!

    Args:
        gate_counts (Dict[CompressedResourceOp, int]): gate counts in terms of compressed resource ops

    Returns:
        Dict[str, int]: gate counts in terms of names of operations
    """
    clean_gate_counts = defaultdict(int)

    for cmp_res_op, counts in gate_counts.items():
        clean_gate_counts[cmp_res_op._name] += counts

    return clean_gate_counts


@qml.QueuingManager.stop_recording()
def _operations_to_compressed_reps(ops: Iterable[Operation]) -> List[CompressedResourceOp]:
    """Convert the sequence of operations to a list of compressed resource ops.

    Args:
        ops (Iterable[Operation]): set of operations to convert.

    Returns:
        List[CompressedResourceOp]: set of converted compressed resource ops.
    """
    cmp_rep_ops = []
    for op in ops:
        if isinstance(op, ResourceConstructor):
            cmp_rep_ops.append(op.resource_rep_from_op())

        else:
            try:
                cmp_rep_ops.append(_temp_map_func(op).resource_rep_from_op())

            except NotImplementedError:
                decomp = op.decomposition()
                cmp_rep_ops.extend(_operations_to_compressed_reps(decomp))

    return cmp_rep_ops
