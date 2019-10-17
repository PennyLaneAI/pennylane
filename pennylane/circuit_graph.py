# Copyright 2019 Xanadu Quantum Technologies Inc.

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
based on operation and observable queues.
"""
from collections import namedtuple
import networkx as nx

from .utils import _flatten


def _by_idx(x):
    """Sorting key for Operations: queue index aka temporal order."""
    return x.queue_idx


Layer = namedtuple("Layer", ["ops", "param_inds"])
"""Parametrized layer of the circuit.

Args:

    ops (list[Operation]): parametrized operations in the layer
    param_inds (tuple[int]): corresponding free parameter indices
"""
#TODO define what a layer is

LayerData = namedtuple("LayerData", ["pre_ops", "ops", "param_inds", "post_ops"])
"""Parametrized layer of the circuit.

Args:
    pre_ops (list[Operation]): operations that precede the layer
    ops (list[Operation]): parametrized operations in the layer
    param_inds (tuple[int]): corresponding free parameter indices
    post_ops (list[Operation]): operations that succeed the layer
"""

class CircuitGraph:
    """Represents a quantum circuit as a directed acyclic graph.

    Args:
        ops (Iterable[Operation]): quantum operations constituting the circuit, in temporal order
        variable_deps (dict[int, list[ParDep]]): Free parameters of the quantum circuit.
            The dictionary key is the parameter index.
            The first element of the ParDep namedtuple is the operation index,
            the second the index of the parameter within the operation.
    """
    def __init__(self, ops, variable_deps):
        self.variable_deps = variable_deps

        self._grid = {}
        """dict[int, list[Operation]]: dictionary representing the quantum circuit as a grid.
        Here, the key is the wire number, and the value is a list containing the operations on that wire.
        """
        for k, op in enumerate(ops):
            op.queue_idx = k  # store the queue index in the Operation
            for w in set(_flatten(op.wires)):  # flatten the nested wires lists of Tensor observables
                # Add op to the grid, to the end of wire w
                self._grid.setdefault(w, []).append(op)

        self._graph = nx.DiGraph()  #: nx.DiGraph: DAG representation of the quantum circuit
        # Iterate over each (populated) wire in the grid
        for wire in self._grid.values():
            # Add the first operation on the wire to the graph
            # This operation does not depend on any others
            self._graph.add_node(wire[0])

            for i in range(1, len(wire)):
                # For subsequent operations on the wire:
                if wire[i] not in self._graph:
                    # Add them to the graph if they are not already
                    # in the graph (multi-qubit operations might already have been placed)
                    self._graph.add_node(wire[i])

                # Create an edge between this operation and the previous operation
                self._graph.add_edge(wire[i - 1], wire[i])

    @property
    def observables(self):
        """Observables in the circuit, sorted by queue index.

        Returns:
            list[Observable]: observables
        """
        nodes = [node for node in self._graph.nodes if getattr(node, 'return_type', None) is not None]
        return sorted(nodes, key=_by_idx)

    @property
    def operations(self):
        """Non-observable Operations in the circuit, sorted by queue index.

        The complement of :meth:`QNode.observables`. Together they return every :class:`Operation`
        instance in the circuit.

        Returns:
            list[Operation]: operations
        """
        nodes = [node for node in self._graph.nodes if getattr(node, 'return_type', None) is None]
        return sorted(nodes, key=_by_idx)

    @property
    def graph(self):
        """The graph representation of the quantum circuit.

        The graph has nodes representing :class:`.Operation` instances,
        and directed edges pointing from nodes to their immediate dependents/successors.

        Returns:
            networkx.DiGraph: the directed acyclic graph representing the quantum circuit
        """
        return self._graph

    def wire_indices(self, wire):
        """Operation indices on the given wire.

        Args:
            wire (int): wire to examine

        Returns:
            list[int]: indices of operations on the wire, in temporal order
        """
        return [op.queue_idx for op in self._grid[wire]]

    def ancestors(self, ops):
        """Ancestor operations of a given set of operations.

        Args:
            ops (Iterable[Operation]): set of operations in the circuit

        Returns:
            set[Operation]: ancestors of the given operations
        """
        return set().union(*(nx.dag.ancestors(self._graph, o) for o in ops)) - set(ops)

    def descendants(self, ops):
        """Descendant operations of a given set of operations.

        Args:
            ops (Iterable[Operation]): set of operations in the circuit

        Returns:
            set[Operation]: descendants of the given operations
        """
        return set().union(*(nx.dag.descendants(self._graph, o) for o in ops)) - set(ops)

    def _in_topological_order(self, ops):
        """Sorts a set of operations in the circuit in a topological order.

        Args:
            ops (Iterable[Operation]): set of operations in the circuit

        Returns:
            Iterable[Operation]: same set of operations, topologically ordered
        """
        G = nx.DiGraph(self._graph.subgraph(ops))
        return nx.dag.topological_sort(G)

    def ancestors_in_order(self, ops):
        """Operation ancestors in a topological order.

        Currently the topological order is determined by the queue index.

        Args:
            ops (Iterable[Operation]): set of operations in the circuit

        Returns:
            list[Operation]: ancestors of the given operations, topologically ordered
        """
        #return self._in_topological_order(self.ancestors(ops))  # an abitrary topological order
        return sorted(self.ancestors(ops), key=_by_idx)

    def descendants_in_order(self, ops):
        """Operation descendants in a topological order.

        Currently the topological order is determined by the queue index.

        Args:
            ops (Iterable[Operation]): set of operations in the circuit

        Returns:
            list[Operation]: descendants of the given operations, topologically ordered
        """
        return sorted(self.descendants(ops), key=_by_idx)

    @property
    def layers(self):
        """Identify the parametrized layer structure of the circuit.

        Returns:
            list[Layer]: layers of the circuit
        """
        # FIXME maybe layering should be greedier, for example [a0 b0 c1 d1] should layer as [a0 c1], [b0, d1] and not [a0], [b0 c1], [d1]
        # keep track of the current layer
        current = Layer([], [])
        layers = [current]

        # sort vars by first occurrence of the var in the ops queue
        variable_ops_sorted = sorted(self.variable_deps.items(), key=lambda x: x[1][0].op.queue_idx)

        # iterate over all parameters
        for param_idx, gate_param_tuple in variable_ops_sorted:
            # iterate over ops depending on that param
            for op, _ in gate_param_tuple:
                # get all predecessor ops of the op
                sub = self.ancestors((op,))

                # check if any of the dependents are in the
                # currently assembled layer
                if set(current.ops) & sub:
                    # operation depends on current layer, start a new layer
                    current = Layer([], [])
                    layers.append(current)

                # store the parameters and ops indices for the layer
                current.ops.append(op)
                current.param_inds.append(param_idx)

        return layers

    def iterate_layers(self):
        """Parametrized layers of the circuit.

        Returns:
            Iterable[LayerData]: layers with extra metadata
        """
        # iterate through each layer
        for ops, param_inds in self.layers:
            pre_queue = self.ancestors_in_order(ops)
            post_queue = self.descendants_in_order(ops)
            yield LayerData(pre_queue, ops, tuple(param_inds), post_queue)

    def update_node(self, old, new):
        """Replaces the given circuit graph node with a new one.

        Args:
            old (Operation): node to replace
            new (Operation): replacement
        """
        # TODO Does alter the graph edges in any way. variable_deps is not changed, _grid is not changed. Dangerous, do we need this?
        if new.wires != old.wires:
            raise ValueError('The new Operation must act on the same wires as the old one.')
        nx.relabel_nodes(self._graph, {old: new}, copy=False)  # change the graph in place
        new.queue_idx = old.queue_idx
