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
    r"""Create a resource object for storing quantum resource information."""

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

    def __add__(self: Resources, other: Resources) -> Resources:
        """Simple add method to combine resources linearly in series."""
        if not isinstance(other, Resources):
            raise ValueError("Can only combine with instances of Resource")

        total_resources = Resources()

        total_resources.num_wires = max(self.num_wires, other.num_wires)  # qubits in series is just the max of each
        total_resources.num_gates = self.num_gates + other.num_gates
        total_resources.depth = self.depth + other.depth
        total_resources.shots = max(self.shots, other.shots)  # shots in series is just the max of each algorithm

        total_resources.gate_types = copy(self.gate_types)
        for gate, count in other.gate_types.items():
            total_resources.gate_types[gate] += count

        return total_resources

    def __matmul__(self: Resources, other: Resources) -> Resources:
        """Simple add method to combine resources in parallel."""
        if not isinstance(other, Resources):
            raise ValueError("Can only combine with instances of Resource")

        total_resources = Resources()

        total_resources.num_wires = self.num_wires + other.num_wires  # qubits in parallel is the sum of each algorithm
        total_resources.num_gates = max(self.num_gates, other.num_gates)
        total_resources.depth = max(self.depth, other.depth)
        total_resources.shots = max(self.shots, other.shots)  # shots in parallel is just the max of each algorithm

        total_resources.gate_types = copy(self.gate_types)
        for gate, count in other.gate_types.items():
            total_resources.gate_types[gate] += count

        return total_resources

    @staticmethod
    def combine(lst_resources: Iterable[Resources], in_parallel=False):
        """A wrapper function to combine a sequence of Resource objects."""
        if in_parallel:
            return reduce(lambda a, b: a @ b, lst_resources)
        return sum(lst_resources)
