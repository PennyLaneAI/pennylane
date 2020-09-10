# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

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
This module contains the CircuitGraph class which is used to generate a DAG (directed acyclic graph)
representation of a quantum circuit from an operator and observable queue.
"""
import networkx as nx

from pennylane import CircuitGraph


class NewCircuitGraph(CircuitGraph):
    """New circuit graph object. This will eventually grow to replace
    the existing CircuitGraph; for now, we simply inherit from the
    current CircuitGraph, and modify the instantiation so that it
    can be created via the quantum tape."""

    def __init__(self, ops, obs, wires):
        self._operations = ops
        self._observables = obs
        super().__init__(ops + obs, variable_deps={}, wires=wires)

    @property
    def operations(self):
        """Operations in the circuit."""
        return self._operations

    @property
    def observables(self):
        """Observables in the circuit."""
        return self._observables

    def has_path(self, a, b):
        """Checks if a path exists between the two given nodes.

        Args:
            a (Operator): initial node
            b (Operator): final node

        Returns:
            bool: returns ``True`` if a path exists
        """
        return nx.has_path(self._graph, a, b)
