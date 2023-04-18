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
from dataclasses import dataclass, field

from pennylane.tape import QuantumTape


@dataclass(frozen=True)
class Resources:
    r"""Contains attributes which store key resources such as number of gates, number of wires, shots,
    depth and gate types.

    Args:
        num_wires (int): number of qubits
        num_gates (int): number of gates
        gate_types (dict): dictionary storing operation names (str) as keys
            and the number of times they are used in the circuit (int) as values
        depth (int): the depth of the circuit defined as the maximum number of non-parallel operations
        shots (int): number of samples to generate

    Raises:
        TypeError: If the attributes provided are not of the correct type.
        ValueError: If an attribute provided has value less than 0.
        AttributeError: If an attribute is set after initialization.

    .. details::

        The resources being tracked can be accessed and set as class attributes.
        Additionally, the :code:`Resources` instance can be nicely displayed in the console.

        **Example**

        >>> r = Resources(num_wires=2, num_gates=2, gate_types={'Hadamard': 1, 'CNOT':1}, depth=2)
        >>> print(r)
        wires: 2
        gates: 2
        depth: 2
        shots: 0
        gate_types: {'Hadamard': 1, 'CNOT': 1}
    """
    num_wires: int = 0
    num_gates: int = 0
    gate_types: dict[int] = field(default_factory=dict)
    depth: int = 0
    shots: int = 0

    def __str__(self):
        keys = ["wires", "gates", "depth", "shots", "gate_types"]
        vals = [self.num_wires, self.num_gates, self.depth, self.shots, self.gate_types]
        items = "\n".join([str(i) for i in zip(keys, vals)])
        items = items.replace("('", "")
        items = items.replace("',", ":")
        items = items.replace(")", "")
        items = items.replace("{", "\n{")
        return items

    def _ipython_display_(self):
        """Displays __str__ in ipython instead of __repr__"""
        print(str(self))


def count_resources(tape: QuantumTape, shots: int) -> Resources:
    """Given a quantum circuit (tape) and number of samples, this function
     counts the resources used by both standard pennylane operations and
     custom user-defined operations.

    Args:
        tape (.QuantumTape): The quantum circuit for which we count resources
        shots (int): The number of samples or shots to execute

    Returns:
        (.Resources): The total resources used in the workflow
    """
    num_wires = len(tape.wires)
    depth = tape.graph.get_depth()

    num_gates = 0
    gate_types = defaultdict(int)
    for op in tape.operations:
        if hasattr(op, "resources"):
            op_resource = op.resources()
            for d in op_resource.gate_types:
                gate_types[d] += op_resource.gate_types[d]
            num_gates += sum(op_resource.gate_types.values())
        else:
            gate_types[op.name] += 1
            num_gates += 1

    return Resources(num_wires, num_gates, gate_types, depth, shots)
