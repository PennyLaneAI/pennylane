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
r"""Base classes for the resources object."""
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class Resources:
    r"""Contains attributes which store key resources such as number of gates, number of wires, and gate types.

    Args:
        num_wires (int): number of qubits
        num_gates (int): number of gates
        gate_types (dict): dictionary storing operations (~.CompressedResourceOp) 
            as keys and the number of times they are used in the circuit (int) as values

    .. details::

        The resources being tracked can be accessed as class attributes.
        Additionally, the :code:`Resources` instance can be nicely displayed in the console.

        **Example**

        >>> r = Resources(
        ...             num_wires=2,
        ...             num_gates=2,
        ...             gate_types={
        ...                 OpTypeWithParams("Hadamard", (("num_wires", 1),)): 1,
        ...                 OpTypeWithParams("CNOT", (("num_wires", 2),)):1,
        ...                 }
        ...             )
        >>> print(r)
        wires: 2
        gates: 2
        gate_types:
        {'Hadamard(num_wires=1)': 1, 'CNOT(num_wires=2)': 1}
    """
    num_wires: int = 0
    num_gates: int = 0
    gate_types: defaultdict = field(default_factory=lambda: defaultdict(int))

    def __str__(self):
        keys = ["wires", "gates"]
        vals = [self.num_wires, self.num_gates]
        items = "\n".join([str(i) for i in zip(keys, vals)])
        items = items.replace("('", "")
        items = items.replace("',", ":")
        items = items.replace(")", "")

        gate_type_str = ", ".join(
            [f"'{str(gate_name)}': {count}" for gate_name, count in self.gate_types.items()]
        )
        items += "\ngate_types:\n{" + gate_type_str + "}"
        return items

    def _ipython_display_(self):
        """Displays __str__ in ipython instead of __repr__"""
        print(str(self))