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
from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field

from pennylane.operation import Operation
from pennylane.measurements import Shots


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


class ResourcesOperation(Operation):
    r"""Base class that represents quantum gates or channels applied to quantum
    states and stores the resource requirements of the quantum gate.

    .. note::
        Child classes must implement the :func:`~.ResourcesOperation.resources` method which computes
        the resource requirements of the operation.
    """

    @abstractmethod
    def resources(self) -> Resources:
        r"""Compute the resources required for this operation.

        Returns:
            Resources: The resources required by this operation.

        **Examples**

        >>> class CustomOp(ResourcesOperation):
        ...     num_wires = 2
        ...     def resources(self):
        ...         return Resources(num_wires=self.num_wires, num_gates=3, depth=2)
        ...
        >>> op = CustomOp(wires=[0, 1])
        >>> print(op.resources())
        wires: 2
        gates: 3
        depth: 2
        shots: Shots(total=None)
        gate_types:
        {}
        gate_sizes:
        {}
        """


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
