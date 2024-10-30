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
from functools import singledispatch, wraps
from typing import Callable, Dict, Iterable, List, Set, Union

import pennylane as qml
from pennylane.measurements import MeasurementProcess
from pennylane.operation import Operation
from pennylane.tape import AnnotatedQueue, QuantumScript

from .resource_constructor import ResourceConstructor
from .resource_container import CompressedResourceOp, Resources, mul_in_series

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
    res = Resources()
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

        operations = (op for op in q.queue if not isinstance(op, MeasurementProcess))
        compressed_res_ops_lst = _operations_to_compressed_reps(operations)

        raw_resources = Resources()
        for cmp_rep_op in compressed_res_ops_lst:
            raw_resources += _resources_from_compressed_res_op(
                cmp_rep_op, gate_set=_StandardGateSet, config=config
            )

        # Validation:
        condensed_resources = _resources_from_dict_compressed_res_ops(
            raw_resources.gate_types, gate_set=gate_set, config=config
        )
        clean_resources = _clean_resources(condensed_resources)

        num_gates = sum(clean_resources.values())
        num_wires = len(set.union((op.wires.toset() for op in operations)))

        clean_resources.num_wires = num_wires
        clean_resources.num_gates = num_gates
        return clean_resources

    return wrapper


@get_resources.register
def resources_from_tape(
    obj: QuantumScript, gate_set: Set = DefaultGateSet, config: Dict = resource_config
) -> Resources:
    """Get resources from a quantum tape"""
    num_wires = obj.num_wires
    operations = obj.operations

    compressed_res_ops_lst = _operations_to_compressed_reps(operations)

    raw_resources = Resources()
    for cmp_rep_op in compressed_res_ops_lst:
        raw_resources += _resources_from_compressed_res_op(
            cmp_rep_op, gate_set=_StandardGateSet, config=config
        )

    # Validation:
    condensed_resources = _resources_from_dict_compressed_res_ops(
        raw_resources.gate_types, gate_set=gate_set, config=config
    )
    clean_resources = _clean_resources(condensed_resources)

    num_gates = sum(clean_resources.values())

    clean_resources.num_wires = num_wires
    clean_resources.num_gates = num_gates
    return clean_resources


def get_resources(
    obj: Union[Callable, QuantumScript, Operation],
    gate_set: Set = DefaultGateSet,
    config: Dict = resource_config,
) -> Resources:
    ## Instantiate empty 'res' Resources        # TODO (Jay): Try passing the resources object around and appending
    ## Extract num_wires from object
    res = Resources()

    ## Iterate over operators:
    ## a. If in gate_set, add to res  <IDEA: use general gate_set and do fine grain conversion in validation!>
    ## b. If ResourceConstructor, extract resources from compressed rep [cp_rep --> res function]
    ## c. Else decompose and repeat  ^^

    ## Validation (convert Resources of CompressedResourceOps to strings!)
    ## a. Use a more general gate set for tracking above, and perform fine grain conversions here!
    return res


def _resources_from_compressed_res_op(
    cp_rep: CompressedResourceOp, gate_set: Set, config: Dict = resource_config
) -> Resources:
    ## If op in gate_set add to resources
    if cp_rep._name in gate_set:
        return Resources(gate_types=defaultdict({cp_rep: 1}))

    ## Else decompose cp_rep using its resource decomp [cp_rep --> dict[cp_rep: counts]] and extract resources
    resource_decomp = cp_rep.op_type.resources(config=config, **cp_rep.params)
    return _resources_from_dict_compressed_res_ops(
        resource_decomp, gate_set=gate_set, config=config
    )


def _resources_from_dict_compressed_res_ops(
    dict_cm_reps: Dict[CompressedResourceOp, int], gate_set: Set, config: Dict = resource_config
) -> Resources:

    ## For each cm_rep obtain resources and scale according to the relavent counts
    res = Resources()
    for cp_rep, counts in dict_cm_reps.items():
        res += mul_in_series(
            _resources_from_compressed_res_op(cp_rep, gate_set=gate_set, config=config),
            counts,
            in_place=True,
        )

    return res


def _temp_map_func(op: Operation) -> ResourceConstructor:
    """Temp map function"""
    raise NotImplementedError


def _clean_resources(res: Resources) -> Resources:
    """Map resources with gate_types made from CompressedResourceOps
    into one which tracks just strings of operations!

    Args:
        res (Resources): _description_

    Returns:
        Resources: _description_
    """
    clean_resources = Resources()

    for cmp_res_op, counts in res.gate_types.items():
        clean_resources.gate_types[cmp_res_op._name] += counts

    return clean_resources


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
