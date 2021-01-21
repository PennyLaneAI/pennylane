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
# pylint: disable=too-many-arguments
import networkx as nx

import pennylane as qml
from pennylane.circuit_graph import CircuitGraph, Layer


class TapeCircuitGraph(CircuitGraph):
    """Represents a quantum circuit as a directed acyclic graph.

    This will eventually grow to replace the existing ``CircuitGraph``; for now, we simply inherit
    from the current ``CircuitGraph``, and modify the instantiation so that it can be created via
    the quantum tape.

    In this representation the :class:`~.Operator` instances are the nodes of the graph,
    and each directed edge represent a subsystem (or a group of subsystems) on which the two
    Operators act subsequently. This representation can describe the causal relationships
    between arbitrary quantum channels and measurements, not just unitary gates.

    Args:
        ops (Iterable[.Operator]): quantum operators constituting the circuit, in temporal order
        obs (Iterable[.MeasurementProcess]): terminal measurements, in temporal order
        wires (.Wires): The addressable wire register of the device that will be executing this graph
        par_info (dict[int, dict[str, .Operation or int]]): Parameter information. Keys are
            parameter indices (in the order they appear on the tape), and values are a
            dictionary containing the corresponding operation and operation parameter index.
        trainable_params (set[int]): A set containing the indices of parameters that support
            differentiability. The indices provided match the order of appearence in the
            quantum circuit.
    """

    def __init__(self, ops, obs, wires, par_info=None, trainable_params=None):
        self._operations = ops
        self._observables = obs
        self.par_info = par_info
        self.trainable_params = trainable_params

        self._depth = None

        for m in self._observables:
            if m.return_type is qml.operation.State:
                # state measurements are applied to all device wires
                m._wires = wires  # pylint: disable=protected-access

        super().__init__(ops + obs, variable_deps={}, wires=wires)

        # For computing depth; want only a graph with the operations, not
        # including the observables
        self._operation_graph = None

    @property
    def operations(self):
        """Operations in the circuit."""
        return self._operations

    @property
    def observables(self):
        """Observables in the circuit."""
        return self._observables

    def update_node(self, old, new):
        super().update_node(old, new)
        self._operations = self.operations_in_order
        self._observables = self.observables_in_order

    @property
    def parametrized_layers(self):
        """Identify the parametrized layer structure of the circuit.

        Returns:
            list[Layer]: layers of the circuit
        """
        # FIXME maybe layering should be greedier, for example [a0 b0 c1 d1] should layer as [a0
        # c1], [b0, d1] and not [a0], [b0 c1], [d1] keep track of the current layer
        current = Layer([], [])
        layers = [current]

        for idx, info in self.par_info.items():
            if idx in self.trainable_params:
                op = info["op"]

                # get all predecessor ops of the op
                sub = self.ancestors((op,))

                # check if any of the dependents are in the
                # currently assembled layer
                if set(current.ops) & sub:
                    # operator depends on current layer, start a new layer
                    current = Layer([], [])
                    layers.append(current)

                # store the parameters and ops indices for the layer
                current.ops.append(op)
                current.param_inds.append(idx)

        return layers

    def get_depth(self):
        """Depth of the quantum circuit (longest path in the DAG)."""
        # If there are no operations in the circuit, the depth is 0
        if not self.operations:
            self._depth = 0

        # If there are operations but depth is uncomputed, compute the truncated graph
        # with only the operations, and return the longest path + 1 (since the path is
        # expressed in terms of edges, and we want it in terms of nodes).
        if self._depth is None and self.operations:
            if self._operation_graph is None:
                self._operation_graph = self.graph.subgraph(self.operations)
                self._depth = nx.dag_longest_path_length(self._operation_graph) + 1

        return self._depth

    def has_path(self, a, b):
        """Checks if a path exists between the two given nodes.

        Args:
            a (Operator): initial node
            b (Operator): final node

        Returns:
            bool: returns ``True`` if a path exists
        """
        return nx.has_path(self._graph, a, b)
