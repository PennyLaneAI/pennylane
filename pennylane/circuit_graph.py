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
CircuitGraph
============

**Module name:** :mod:`pennylane.circuit_graph`

.. currentmodule:: pennylane.circuit_graph

This module contains the CircuitGraph class which is used to generate a DAG (directed acyclic graph)
based on operation and observable queues.

.. raw:: html

    <h3>Summary</h3>

.. autosummary::
    CircuitGraph

.. raw:: html

    <h3>Code details</h3>
"""
from collections import namedtuple
import networkx as nx


Command = namedtuple("Command", ["name", "op", "return_type", "idx"])
Layer = namedtuple("Layer", ["op_idx", "param_idx"])


class CircuitGraph:
    """Represents a queue of operations and observables
    as a directed acyclic graph.

    Args:
        queue (list[Operation]): the quantum operations to apply
        observables (list[Observable]): the quantum observables to measure
        parameters (dict[int->list[(int, int)]]): Specifies the free parameters
            of the quantum circuit. The dictionary key is the parameter index.
            The first element of the value tuple is the operation index,
            the second the index of the parameter within the operation.
    """

    def __init__(self, queue, observables, parameters=None):
        self._queue = queue
        self._observables = observables
        self.parameters = parameters or {}

        self._grid = {}
        """dict[int, list[Command]]: dictionary representing the quantum circuit
        as a grid. Here, the key is the wire number, and the value is a list
        containing the Command object containing the operation/observable itself.
        """

        for idx, op in enumerate(queue + observables):
            cmd = Command(
                name=op.name, op=op, return_type=getattr(op, "return_type", None), idx=idx
            )

            for w in set(op.wires):
                if w not in self._grid:
                    # wire is not already in the grid;
                    # add the corresponding wire to the grid
                    self._grid[w] = []

                # Add the operation to the grid, to the end of the specified wire
                self._grid[w].append(cmd)

        self._graph = nx.DiGraph()

        # Iterate over each wire in the grid
        for cmds in self._grid.values():
            if cmds:
                # Add the first operation on the wire to the graph
                # This operation does not depend on any others
                self._graph.add_node(cmds[0].idx, **cmds[0]._asdict())

            for i in range(1, len(cmds)):
                # For subsequent operations on the wire:
                if cmds[i].idx not in self._graph:
                    # Add them to the graph if they are not already
                    # in the graph (multi-qubit operations might already have been placed)
                    self._graph.add_node(cmds[i].idx, **cmds[i]._asdict())

                # Create an edge between this operation and the
                # previous operation
                self._graph.add_edge(cmds[i - 1].idx, cmds[i].idx)

    @property
    def observables(self):
        """
        Return a list of operations that have a return type.
        """
        return [node["op"] for node in self.observable_nodes]

    @property
    def observable_nodes(self):
        """
        Return a list of nodes of operations that have a return type, sorted by "idx".
        """
        nodes = [node for node in self.graph.nodes.values() if node["return_type"]]
        return sorted(nodes, key=lambda node: node["idx"])

    @property
    def operations(self):
        """
        Return a list of operations that do not have a return type.
        """
        return [node["op"] for node in self.operation_nodes]

    @property
    def operation_nodes(self):
        """
        Return a list of nodes of operations that do not have a return type, sorted by "idx".
        """
        nodes = [node for node in self.graph.nodes.values() if not node["return_type"]]
        return sorted(nodes, key=lambda node: node["idx"])

    @property
    def graph(self):
        """The graph representation of the quantum program.

        The resulting graph has nodes representing quantum operations,
        and edges representing dependent/successor operations.

        Each node is labelled by an integer corresponding to the position
        in the queue; node attributes are used to store information about the node:

        * ``'name'`` *(str)*: name of the quantum operation (e.g., ``'PauliZ'``)

        * ``'op'`` *(Operation or Observable)*: the quantum operation/observable object

        * ``'return_type'`` *(pennylane.operation.ObservableReturnTypes)*: The observable
          return type. If an operation, the return type is simply ``None``.

        Returns:
            networkx.DiGraph: the directed acyclic graph representing
            the quantum program
        """
        return self._graph

    def get_wire_indices(self, wire):
        """The operation indices on the given wire.

        Args:
            wire (int): the wire to examine

        Returns:
            list (int): all operation indices on the wire,
            in temporal order
        """
        return [op.idx for op in self._grid[wire]]

    def ancestors(self, ops):
        """Returns all ancestor operations of a given set of operations.

        Args:
            ops (Iterable[int]): a given set of operations labelled by integer
                position in the queue

        Returns:
            set[int]: integer position of all operations
            in the queue that are ancestors of the given operations
        """
        subGs = [self.graph.subgraph(nx.dag.ancestors(self.graph, o)) for o in ops]
        return set().union(*[set(subG.nodes()) for subG in subGs]) - set(ops)

    def descendants(self, ops):
        """Returns all descendant operations of a given set of operations.

        Args:
            ops (Iterable[int]): a given set of operations labelled by integer
                position in the queue

        Returns:
            set[int]: integer position of all operations
            in the queue that are descendants of the given operations
        """
        subGs = [self.graph.subgraph(nx.dag.descendants(self.graph, o)) for o in ops]
        return set().union(*[set(subG.nodes()) for subG in subGs]) - set(ops)

    def get_nodes(self, ops):
        """Given a set of operation indices, return the nodes corresponding to the indices.

        Args:
            ops (Iterable[int]): a given set of operations labelled by integer
                position in the queue

        Returns:
            List[Node]: nodes corresponding to given integer positions in the queue
        """
        return [self.graph.nodes[i] for i in ops]

    def get_ops(self, ops):
        """Given a set of operation indices, return the operation objects.

        Args:
            ops (Iterable[int]): a given set of operations labelled by integer
                position in the queue

        Returns:
            List[Operation, Observable]: operations or observables
            corresponding to given integer positions in the queue
        """
        return [node["op"] for node in self.get_nodes(ops)]

    def get_names(self, ops):
        """Given a set of operation indices, return the operation names.

        Args:
            ops (Iterable[int]): a given set of operations labelled by integer
                position in the queue

        Returns:
            List[str]: operations or observables
            corresponding to given integer positions in the queue
        """
        return [self.graph.nodes(data="name")[i] for i in ops]

    @property
    def layers(self):
        """Identifies and returns a metadata list describing the
        layer structure of the circuit.

        Each layer is a named tuple containing:

        * ``op_idx`` *(list[int])*: the list of operation indices in the layer

        * ``param_idx`` *(list[int])*: the list of parameter indices used within the layer

        Returns:
            list[Layer]: a list of layers
        """
        # keep track of the layer number
        layer = 0
        layer_ops = {0: ([], [])}

        variable_ops_sorted = sorted(list(self.parameters.items()), key=lambda x: x[1][0][0])

        for param_idx, gate_param_tuple in variable_ops_sorted:
            # iterate over all parameters
            for op_idx, _ in gate_param_tuple:
                # get all dependents of the existing parameter
                sub = set(
                    nx.dag.topological_sort(
                        self.graph.subgraph(nx.dag.ancestors(self.graph, op_idx)).copy()
                    )
                )

                # check if any of the dependents are in the
                # existing layer
                if set(layer_ops[layer][0]) & sub:
                    # operation depends on previous layer,
                    # start a new layer count
                    layer += 1

                # store the parameters and ops indices for the layer
                layer_ops.setdefault(layer, ([], []))
                layer_ops[layer][0].append(op_idx)
                layer_ops[layer][1].append(param_idx)

        return [Layer(*k) for _, k in sorted(list(layer_ops.items()))]

    def iterate_layers(self):
        """Identifies and returns an iterable containing
        the parametrized layers.

        Returns:
            Iterable[tuple[list, list, tuple, list]]: an iterable that returns a tuple
            ``(pre_queue, layer, param_idx, post_queue)`` at each iteration.

            * ``pre_queue`` (*list[Operation]*): all operations that precede the layer

            * ``layer`` (*list[Operation]*): the parametrized gates in the layer

            * ``param_idx`` (*tuple[int]*): The integer indices corresponding
              to the free parameters of this layer, in the order they appear in
              this layer.

            * ``post_queue`` (*list[Operation, Observable]*): all operations that succeed the layer
        """
        # iterate through each layer
        for ops, param_idx in self.layers:

            # get the ops in this layer
            layer = self.get_ops(ops)
            pre_queue = self.get_ops(self.ancestors(ops))
            post_queue = self.get_ops(self.descendants(ops))

            yield pre_queue, layer, tuple(param_idx), post_queue

    def update_node(self, node, op):
        """
        Updates a given node with a new operation, op.

        Args:
            op (Operation): an operation to update the given node with
            node (dict): the node to update
        """
        cmd = Command(
            name=op.name, op=op, return_type=getattr(op, "return_type", None), idx=node["idx"]
        )
        nx.set_node_attributes(self._graph, {node["idx"]: {**cmd._asdict()}})
