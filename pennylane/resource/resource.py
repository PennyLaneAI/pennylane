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
The data class which will aggregate all the resource information from a quantum workflow.
"""
from copy import copy
from typing import Iterable
from functools import reduce
from collections import defaultdict


class Resources:
    r"""Contains attributes which store key resources (such as number of gates, number of wires, etc.)
    tracked over a quantum workflow.

    This container object tracks the following resources as attributes:
    - num_wires (int): number of qubits
    - num_gates (int): number of gates
    - depth (int): the depth of the circuit (max number of non-parallel operations)
    - shots (int): number of sampels to measure
    - gate_types (defaultdict(int)): dictionary with keys are operation names (str) and
        values are the number of times the operation is repeated in the circuit (int)

    .. details::

        The resources being tracked can be accessed and set as class attributes.
        Additionally, the :code:`Resources` instance can be nicely displayed in the console.

        **Example**

        >>> r = Resources()
        >>> r.num_wires = 2
        >>> r.num_gates = 2
        >>> r.depth = 2
        >>> r.gate_types = {'Hadamard': 1, 'CNOT':1}
        >>> print(r)
        wires: 2
        gates: 2
        depth: 2
        shots: 0
        gate_types: {'Hadamard': 1, 'CNOT': 1}

        There is also functionality to combine instances of :code:`Resources` intuatively. There
        are two basic ways to combine the resources of quantum workflows together. The series method
        assumes the two workflows occur sequentially on the same qubit register while the parallel
        method assumes they occur simultaneously on two seperate qubit registers.

        **Example**

        >>> r_in_series = r + r
        >>> print(r_in_series)
        wires: 2
        gates: 4
        depth: 4
        shots: 0
        gate_types: {'Hadamard': 2, 'CNOT': 2}
        >>>
        >>> r_in_parallel = r @ r
        wires: 4
        gates: 4
        depth: 2
        shots: 0
        gate_types: {'Hadamard': 2, 'CNOT': 2}
    """

    def __init__(self):
        self.num_wires = 0
        self.num_gates = 0
        self.gate_types = defaultdict(int)
        self.depth = 0
        self.shots = 0

    def __str__(self):
        keys = ["wires", "gates", "depth", "shots", "gate_types"]
        vals = [self.num_wires, self.num_gates, self.depth, self.shots, self.gate_types]
        items = "\n".join([str(i) for i in zip(keys, vals)])
        items = items.replace("('", "")
        items = items.replace("',", ":")
        items = items.replace(")", "")
        items = items.replace("defaultdict(<class 'int'>, ", "\n")
        return items

    def __repr__(self):
        return (
            f"<Resource: wires={self.num_wires}, gates={self.num_gates}, "
            f"depth={self.depth}, shots={self.shots}, gate_types={self.gate_types}>"
        )

    def _ipython_display_(self):
        """Displays __str__ in ipython instead of __repr__"""
        print(str(self))

    def __add__(self, other: "Resources") -> "Resources":
        """Simple add method to combine resources linearly in series."""
        if not isinstance(other, Resources):
            raise ValueError(
                f"Can only combine with another instance of `Resources`, got {type(other)}"
            )

        total_resources = Resources()

        total_resources.num_wires = max(
            self.num_wires, other.num_wires
        )  # qubits in series is just the max of each
        total_resources.num_gates = self.num_gates + other.num_gates
        total_resources.depth = self.depth + other.depth
        total_resources.shots = max(
            self.shots, other.shots
        )  # shots in series is just the max of each algorithm

        total_resources.gate_types = copy(self.gate_types)
        for gate, count in other.gate_types.items():
            total_resources.gate_types[gate] += count

        return total_resources

    def __matmul__(self, other: "Resources") -> "Resources":
        """Simple add method to combine resources in parallel."""
        if not isinstance(other, Resources):
            raise ValueError(
                f"Can only combine with another instance of `Resources`, got {type(other)}"
            )

        total_resources = Resources()

        total_resources.num_wires = (
            self.num_wires + other.num_wires
        )  # qubits in parallel is the sum of each algorithm
        total_resources.num_gates = self.num_gates + other.num_gates
        total_resources.depth = max(self.depth, other.depth)
        total_resources.shots = max(
            self.shots, other.shots
        )  # shots in parallel is just the max of each algorithm

        total_resources.gate_types = copy(self.gate_types)
        for gate, count in other.gate_types.items():
            total_resources.gate_types[gate] += count

        return total_resources

    def __eq__(self, other: "Resources") -> bool:
        return all(
            (
                self.num_wires == other.num_wires,
                self.num_gates == other.num_gates,
                self.depth == other.depth,
                self.shots == other.shots,
                self.gate_types == other.gate_types,
            ),
        )

    @staticmethod
    def combine(lst_resources: Iterable["Resources"], in_parallel=False):
        """A wrapper function to combine a sequence of Resource objects."""
        if in_parallel:
            return reduce(lambda a, b: a @ b, lst_resources)
        return reduce(lambda a, b: a + b, lst_resources)
