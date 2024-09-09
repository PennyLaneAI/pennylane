# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Stores classes and logic to aggregate all the resource information from a quantum workflow.
"""
from collections import defaultdict
from copy import copy
from dataclasses import dataclass, field
from functools import wraps
from typing import Callable

import pennylane as qml
from pennylane.measurements import MeasurementProcess, Shots
from pennylane.operation import DecompositionUndefinedError, Operator, ResourcesOperation
from pennylane.queuing import AnnotatedQueue


class ResourceConfig(dict):
    """A class to contains the keyword arguments
    for all resource methods."""

    def __missing__(self, key):
        return dict()


__resource_kwarg_config = ResourceConfig(
    {
        "RX": {"epsilon": 1e-3},
        "RY": {"epsilon": 1e-3},
        "RZ": {"epsilon": 1e-3},
        "TrotterProduct": {"estimate": True},
        "TrotterizedQfunc": {"estimate": True},
    }
)


def set_resource_kwargs(op_name, kwargs_dict):
    """Set the keyword arguments for a given operation type."""
    global __resource_kwarg_config
    __resource_kwarg_config[op_name] = kwargs_dict


def get_resource_kwargs(op_name):
    """Return a copy of the keyword arguments used for a given operation type."""
    global __resource_kwarg_config
    return copy(__resource_kwarg_config[op_name])


@dataclass(frozen=True)
class Resources:
    r"""Contains attributes which store key resources such as number of gates, number of wires, shots,
    depth and gate types.

    Args:
        num_wires (int): number of qubits
        num_gates (int): number of gates
        gate_types (dict): dictionary storing operation names (str) as keys
            and the number of times they are used in the circuit (int) as values
        gate_sizes (dict): dictionary storing the number of :math:`n` qubit gates in the circuit
            as a key-value pair where :math:`n` is the key and the number of occurances is the value
        depth (int): the depth of the circuit defined as the maximum number of non-parallel operations
        shots (Shots): number of samples to generate

    .. details::

        The resources being tracked can be accessed as class attributes.
        Additionally, the :code:`Resources` instance can be nicely displayed in the console.

        **Example**

        >>> r = Resources(num_wires=2, num_gates=2, gate_types={'Hadamard': 1, 'CNOT':1}, gate_sizes={1: 1, 2: 1}, depth=2)
        >>> print(r)
        wires: 2
        gates: 2
        depth: 2
        shots: Shots(total=None)
        gate_types:
        {'Hadamard': 1, 'CNOT': 1}
        gate_sizes:
        {1: 1, 2: 1}
    """

    num_wires: int = 0
    num_gates: int = 0
    gate_types: dict = field(default_factory=dict)
    gate_sizes: dict = field(default_factory=dict)
    depth: int = 0
    shots: Shots = field(default_factory=Shots)

    def __str__(self):
        keys = ["wires", "gates", "depth"]
        vals = [self.num_wires, self.num_gates, self.depth]
        items = "\n".join([str(i) for i in zip(keys, vals)])
        items = items.replace("('", "")
        items = items.replace("',", ":")
        items = items.replace(")", "")

        items += f"\nshots: {str(self.shots)}"

        gate_type_str = ", ".join(
            [f"'{gate_name}': {count}" for gate_name, count in self.gate_types.items()]
        )
        items += "\ngate_types:\n{" + gate_type_str + "}"

        gate_size_str = ", ".join(
            [f"{n_gate}: {count}" for n_gate, count in self.gate_sizes.items()]
        )
        items += "\ngate_sizes:\n{" + gate_size_str + "}"
        return items

    def _ipython_display_(self):
        """Displays __str__ in ipython instead of __repr__"""
        print(str(self))

    def __mul__(self, other):
        if isinstance(other, int):
            new_num_gates = self.num_gates * other
            new_gate_types = copy(self.gate_types)
            new_gate_sizes = copy(self.gate_sizes)

            for k, v in new_gate_types.items():
                new_gate_types[k] = v * other

            for k, v in new_gate_sizes.items():
                new_gate_sizes[k] = v * other

            return Resources(
                num_gates=new_num_gates, gate_types=new_gate_types, gate_sizes=new_gate_sizes
            )

        raise NotImplementedError

    __rmul__ = __mul__

    def __add__(self, other):
        if isinstance(other, self.__class__):
            new_gate_types = copy(self.gate_types)
            new_gate_sizes = copy(self.gate_sizes)

            for gate, count in other.gate_types.items():
                new_gate_types[gate] += count

            for size, count in other.gate_sizes.items():
                new_gate_sizes[size] += count

            return Resources(
                num_gates=self.num_gates + other.num_gates,
                gate_types=new_gate_types,
                gate_sizes=new_gate_sizes,
            )

        raise NotImplementedError


def substitute(primary_resources, gate_info, replacement_resources):
    """Replace a certain gate with some fixed resources."""
    gate_str, num_wires = gate_info

    if primary_resources.gate_types[gate_str] == 0:
        return primary_resources

    num_gates = primary_resources.num_gates
    gate_sizes = copy(primary_resources.gate_sizes)
    gate_types = copy(primary_resources.gate_types)

    counts_of_gate_to_replace = gate_types.pop(gate_str)

    num_gates -= counts_of_gate_to_replace
    gate_sizes[num_wires] -= counts_of_gate_to_replace

    # Replace
    scaled_replacement_resources = replacement_resources * counts_of_gate_to_replace

    num_gates = num_gates + scaled_replacement_resources.num_gates
    _combine_dicts(gate_types, scaled_replacement_resources.gate_types)  # update in place
    _combine_dicts(gate_sizes, scaled_replacement_resources.gate_sizes)  # update in place

    return Resources(num_gates=num_gates, gate_sizes=gate_sizes, gate_types=gate_types)


def _count_resources(tape) -> Resources:
    """Given a quantum circuit (tape), this function
     counts the resources used by standard PennyLane operations.

    Args:
        tape (.QuantumTape): The quantum circuit for which we count resources

    Returns:
        (.Resources): The total resources used in the workflow
    """
    num_wires = len(tape.wires)
    shots = tape.shots
    depth = tape.graph.get_depth()

    num_gates = 0
    gate_types = defaultdict(int)
    gate_sizes = defaultdict(int)
    for op in tape.operations:
        if isinstance(op, ResourcesOperation):
            op_resource = op.resources()
            for d in op_resource.gate_types:
                gate_types[d] += op_resource.gate_types[d]

            for n in op_resource.gate_sizes:
                gate_sizes[n] += op_resource.gate_sizes[n]

            num_gates += sum(op_resource.gate_types.values())

        else:
            gate_types[op.name] += 1
            gate_sizes[len(op.wires)] += 1
            num_gates += 1

    return Resources(num_wires, num_gates, gate_types, gate_sizes, depth, shots)


def _combine_dicts(base_dict, other_dict):
    for k, v in other_dict.items():
        base_dict[k] += v


StandardGateSet = {
    "PauliX",
    "PauliY",
    "PauliZ",
    "Hadamard",
    "SWAP",
    "CNOT",
    "S",
    "Adjoint(S)",  # <-- Clifford Gates
    "T",
    "Adjoint(T)",
    "Toffoli",  # <-- Non-Clifford Gates
    "RX",
    "RY",
    "RZ",
}


def get_resources(obj, gate_set=StandardGateSet):
    if isinstance(obj, Callable):

        @wraps(obj)
        def wrapper(*args, **kwargs):
            with AnnotatedQueue() as q:
                obj(*args, **kwargs)

            res = resources_from_sequence_ops(q.queue, gate_set)
            return validate_resources(res, gate_set)

        return wrapper

    if isinstance(obj, Operator):
        res = resources_from_op(obj, gate_set)
        return validate_resources(res, gate_set)

    res = resources_from_sequence_ops(obj, gate_set)
    return validate_resources(res, gate_set)


def resources_from_op(op, gate_set) -> Resources:
    """Compute the resources for a single operator

    Args:
        op (.Operator): The operation for which we must compute resoruces
        gate_set (set, optional): _description_. Defaults to StandardGateSet.

    Raises:
        ValueError: Cannot obtain resources for the operator in the target gate_set

    Returns:
        Resources:
    """
    if isinstance(op, ResourcesOperation):
        op_kwargs = get_resource_kwargs(op.name)
        op_resources = op.resources(gate_set, **op_kwargs)
        return op_resources

    if op.name in gate_set:
        return Resources(
            num_gates=1,
            gate_types=defaultdict(int, {op.name: 1}),
            gate_sizes=defaultdict(int, {len(op.wires): 1}),
        )

    else:
        try:
            return resources_from_sequence_ops(op.decomposition(), gate_set)
        except DecompositionUndefinedError as e:
            raise ValueError(
                f"Cannot obtain the resources for type {type(op)} in terms of the gate-set:\n {gate_set}"
            ) from e


def resources_from_sequence_ops(ops_lst, gate_set):
    num_gates = 0
    gate_types = defaultdict(int)
    gate_sizes = defaultdict(int)

    for op in ops_lst:
        if isinstance(op, MeasurementProcess):
            continue

        if op.name in gate_set:
            gate_types[op.name] += 1
            gate_sizes[len(op.wires)] += 1
            num_gates += 1

        else:
            op_resources = resources_from_op(op, gate_set)
            num_gates += op_resources.num_gates
            _combine_dicts(gate_types, op_resources.gate_types)  # update in place
            _combine_dicts(gate_sizes, op_resources.gate_sizes)  # update in place

    return Resources(num_gates=num_gates, gate_types=gate_types, gate_sizes=gate_sizes)


def validate_resources(resources, gate_set):
    """Ensure that the resources generated respect the gate_set provided.

    If so, return the resources provided. If not, attempt to fix it and
    return the modified resources, otherwise raise an error.

    Args:
        resources (.Resources): The resources to validate.
        gate_set (set): The gateset the resources should come from.
    """

    modified_resources = resources
    original_gate_types = resources.gate_types

    if "RX" in original_gate_types and "RX" not in gate_set:
        modified_resources = substitute(
            modified_resources, ("RX", 1), qml.RX(1.23, 0).resources(gate_set)
        )

    if "RY" in original_gate_types and "RY" not in gate_set:
        modified_resources = substitute(
            modified_resources, ("RY", 1), qml.RY(1.23, 0).resources(gate_set)
        )

    if "RZ" in original_gate_types and "RZ" not in gate_set:
        modified_resources = substitute(
            modified_resources, ("RZ", 1), qml.RZ(1.23, 0).resources(gate_set)
        )

    for gate in modified_resources.gate_types:
        if gate not in gate_set:
            raise ValueError(f"{gate} is not a valid gate in the gateset.")

    return modified_resources
