# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

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
representation of a quantum circuit from an Operator queue.
"""
# pylint: disable=too-many-branches,too-many-arguments,too-many-instance-attributes
from collections import namedtuple

import rustworkx as rx
import numpy as np

import pennylane as qml

from pennylane.wires import Wires


def _by_idx(x):
    """Sorting key for Operators: queue index aka temporal order.

    Args:
        x (Operator): node in the circuit graph
    Returns:
        int: sorting key for the node
    """
    return x.queue_idx


def _is_observable(x):
    """Predicate for deciding if an Operator instance is an observable.

    .. note::
       Currently some :class:`Observable` instances are not observables in this sense,
       since they can be used as gates as well.

    Args:
        x (Operator): node in the circuit graph
    Returns:
        bool: True iff x is an observable
    """
    return getattr(x, "return_type", None) is not None


Layer = namedtuple("Layer", ["ops", "param_inds"])
"""Parametrized layer of the circuit.

Args:

    ops (list[Operator]): parametrized operators in the layer
    param_inds (list[int]): corresponding free parameter indices
"""
# TODO define what a layer is

LayerData = namedtuple("LayerData", ["pre_ops", "ops", "param_inds", "post_ops"])
"""Parametrized layer of the circuit.

Args:
    pre_ops (list[Operator]): operators that precede the layer
    ops (list[Operator]): parametrized operators in the layer
    param_inds (tuple[int]): corresponding free parameter indices
    post_ops (list[Operator]): operators that succeed the layer
"""


class CircuitGraph:
    """Represents a quantum circuit as a directed acyclic graph.

    In this representation the :class:`~.Operator` instances are the nodes of the graph,
    and each directed edge represent a subsystem (or a group of subsystems) on which the two
    Operators act subsequently. This representation can describe the causal relationships
    between arbitrary quantum channels and measurements, not just unitary gates.

    Args:
        ops (Iterable[.Operator]): quantum operators constituting the circuit, in temporal order
        obs (Iterable[.MeasurementProcess]): terminal measurements, in temporal order
        wires (.Wires): The addressable wire registers of the device that will be executing this graph
        par_info (dict[int, dict[str, .Operation or int]]): Parameter information. Keys are
            parameter indices (in the order they appear on the tape), and values are a
            dictionary containing the corresponding operation and operation parameter index.
        trainable_params (set[int]): A set containing the indices of parameters that support
            differentiability. The indices provided match the order of appearence in the
            quantum circuit.
    """

    # pylint: disable=too-many-public-methods

    def __init__(self, ops, obs, wires, par_info=None, trainable_params=None):
        self._operations = ops
        self._observables = obs
        self.par_info = par_info
        self.trainable_params = trainable_params

        queue = ops + obs

        self._depth = None

        self._grid = {}
        """dict[int, list[Operator]]: dictionary representing the quantum circuit as a grid.
        Here, the key is the wire number, and the value is a list containing the operators on that wire.
        """
        self.wires = wires
        """Wires: wires that are addressed in the operations.
        Required to translate between wires and indices of the wires on the device."""
        self.num_wires = len(wires)
        """int: number of wires the circuit contains"""
        for k, op in enumerate(queue):
            op.queue_idx = k  # store the queue index in the Operator

            if hasattr(op, "return_type"):
                if op.return_type is qml.measurements.State:
                    # State measurements contain no wires by default, but wires are
                    # required for the circuit drawer, so we recreate the state
                    # measurement with all wires
                    op = qml.measurements.MeasurementProcess(qml.measurements.State, wires=wires)

                elif op.return_type is qml.measurements.Sample and op.wires == Wires([]):
                    # Sampling without specifying wires is treated as sampling all wires
                    op = qml.measurements.MeasurementProcess(qml.measurements.Sample, wires=wires)

                op.queue_idx = k

            for w in op.wires:
                # get the index of the wire on the device
                wire = wires.index(w)
                # add op to the grid, to the end of wire w
                self._grid.setdefault(wire, []).append(op)

        # TODO: State preparations demolish the incoming state entirely, and therefore should have no incoming edges.

        self._graph = rx.PyDiGraph(
            multigraph=False
        )  #: rx.PyDiGraph: DAG representation of the quantum circuit

        # Iterate over each (populated) wire in the grid
        for wire in self._grid.values():
            # Add the first operator on the wire to the graph
            # This operator does not depend on any others

            # Check if wire[0] in self._grid.values()
            # is already added to the graph; this
            # condition avoids adding new nodes with
            # the same value but different indexes
            if wire[0] not in self._graph.nodes():
                self._graph.add_node(wire[0])

            for i in range(1, len(wire)):
                # For subsequent operators on the wire:
                if wire[i] not in self._graph.nodes():
                    # Add them to the graph if they are not already
                    # in the graph (multi-qubit operators might already have been placed)
                    self._graph.add_node(wire[i])

                # Create an edge between this and the previous operator
                # There isn't any default value for the edge-data in
                # rx.PyDiGraph.add_edge(); this is set to an empty string
                self._graph.add_edge(
                    self._graph.nodes().index(wire[i - 1]), self._graph.nodes().index(wire[i]), ""
                )

        # For computing depth; want only a graph with the operations, not
        # including the observables
        self._operation_graph = None

        # Required to keep track if we need to handle multiple returned
        # observables per wire
        self._max_simultaneous_measurements = None

    def print_contents(self):
        """Prints the contents of the quantum circuit."""

        print("Operations")
        print("==========")
        for op in self.operations:
            print(repr(op))

        print("\nObservables")
        print("===========")
        for op in self.observables:
            print(repr(op))

    def serialize(self):
        """Serialize the quantum circuit graph based on the operations and
        observables in the circuit graph and the index of the variables
        used by them.

        The string that is produced can be later hashed to assign a unique value to the circuit graph.

        Returns:
            string: serialized quantum circuit graph
        """
        serialization_string = ""
        delimiter = "!"

        for op in self.operations_in_order:
            serialization_string += op.name

            for param in op.data:
                serialization_string += delimiter
                serialization_string += str(param)
                serialization_string += delimiter

            serialization_string += str(op.wires.tolist())

        # Adding a distinct separating string that could not occur by any combination of the
        # name of the operation and wires
        serialization_string += "|||"

        for obs in self.observables_in_order:
            serialization_string += str(obs.return_type)
            serialization_string += delimiter
            serialization_string += str(obs.name)
            for param in obs.data:
                serialization_string += delimiter
                serialization_string += str(param)
                serialization_string += delimiter

            serialization_string += str(obs.wires.tolist())
        return serialization_string

    @property
    def hash(self):
        """Creating a hash for the circuit graph based on the string generated by serialize.

        Returns:
            int: the hash of the serialized quantum circuit graph
        """
        return hash(self.serialize())

    @property
    def observables_in_order(self):
        """Observables in the circuit, in a fixed topological order.

        The topological order used by this method is guaranteed to be the same
        as the order in which the measured observables are returned by the quantum function.
        Currently the topological order is determined by the queue index.

        Returns:
            list[Observable]: observables
        """
        nodes = [node for node in self._graph.nodes() if _is_observable(node)]
        return sorted(nodes, key=_by_idx)

    @property
    def observables(self):
        """Observables in the circuit."""
        return self._observables

    @property
    def operations_in_order(self):
        """Operations in the circuit, in a fixed topological order.

        Currently the topological order is determined by the queue index.

        The complement of :meth:`QNode.observables`. Together they return every :class:`Operator`
        instance in the circuit.

        Returns:
            list[Operation]: operations
        """
        nodes = [node for node in self._graph.nodes() if not _is_observable(node)]
        return sorted(nodes, key=_by_idx)

    @property
    def operations(self):
        """Operations in the circuit."""
        return self._operations

    @property
    def graph(self):
        """The graph representation of the quantum circuit.

        The graph has nodes representing :class:`.Operator` instances,
        and directed edges pointing from nodes to their immediate dependents/successors.

        Returns:
            rustworkx.PyDiGraph: the directed acyclic graph representing the quantum circuit
        """
        return self._graph

    def wire_indices(self, wire):
        """Operator indices on the given wire.

        Args:
            wire (int): wire to examine

        Returns:
            list[int]: indices of operators on the wire, in temporal order
        """
        return [op.queue_idx for op in self._grid[wire]]

    def ancestors(self, ops):
        """Ancestors of a given set of operators.

        Args:
            ops (Iterable[Operator]): set of operators in the circuit

        Returns:
            set[Operator]: ancestors of the given operators
        """
        anc = set(
            self._graph.get_node_data(n)
            for n in set().union(
                # rx.ancestors() returns node indexes instead of node-values
                *(rx.ancestors(self._graph, self._graph.nodes().index(o)) for o in ops)
            )
        )
        return anc - set(ops)

    def descendants(self, ops):
        """Descendants of a given set of operators.

        Args:
            ops (Iterable[Operator]): set of operators in the circuit

        Returns:
            set[Operator]: descendants of the given operators
        """
        des = set(
            self._graph.get_node_data(n)
            for n in set().union(
                # rx.descendants() returns node indexes instead of node-values
                *(rx.descendants(self._graph, self._graph.nodes().index(o)) for o in ops)
            )
        )
        return des - set(ops)

    def _in_topological_order(self, ops):
        """Sorts a set of operators in the circuit in a topological order.

        Args:
            ops (Iterable[Operator]): set of operators in the circuit

        Returns:
            Iterable[Operator]: same set of operators, topologically ordered
        """
        G = self._graph.subgraph(list(self._graph.nodes().index(o) for o in ops))
        indexes = rx.topological_sort(G)
        return list(G[x] for x in indexes)

    def ancestors_in_order(self, ops):
        """Operator ancestors in a topological order.

        Currently the topological order is determined by the queue index.

        Args:
            ops (Iterable[Operator]): set of operators in the circuit

        Returns:
            list[Operator]: ancestors of the given operators, topologically ordered
        """
        return sorted(self.ancestors(ops), key=_by_idx)  # an abitrary topological order

    def descendants_in_order(self, ops):
        """Operator descendants in a topological order.

        Currently the topological order is determined by the queue index.

        Args:
            ops (Iterable[Operator]): set of operators in the circuit

        Returns:
            list[Operator]: descendants of the given operators, topologically ordered
        """
        return sorted(self.descendants(ops), key=_by_idx)

    def nodes_between(self, a, b):
        r"""Nodes on all the directed paths between the two given nodes.

        Returns the set of all nodes ``s`` that fulfill :math:`a \le s \le b`.
        There is a directed path from ``a`` via ``s`` to ``b`` iff the set is nonempty.
        The endpoints belong to the path.

        Args:
            a (Operator): initial node
            b (Operator): final node

        Returns:
            set[Operator]: nodes on all the directed paths between a and b
        """
        A = self.descendants([a])
        A.add(a)
        B = self.ancestors([b])
        B.add(b)
        return A & B

    def invisible_operations(self):
        """Operations that cannot affect the circuit output.

        An :class:`Operation` instance in a quantum circuit is *invisible* if is not an ancestor
        of an observable. Such an operation cannot affect the circuit output, and usually indicates
        there is something wrong with the circuit.

        Returns:
            set[Operator]: operations that cannot affect the output
        """
        visible = self.ancestors(self.observables)
        invisible = set(self.operations) - visible
        return invisible

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

    def iterate_parametrized_layers(self):
        """Parametrized layers of the circuit.

        Returns:
            Iterable[LayerData]: layers with extra metadata
        """
        # iterate through each layer
        for ops, param_inds in self.parametrized_layers:
            pre_queue = self.ancestors_in_order(ops)
            post_queue = self.descendants_in_order(ops)
            yield LayerData(pre_queue, ops, tuple(param_inds), post_queue)

    def update_node(self, old, new):
        """Replaces the given circuit graph node with a new one.

        Args:
            old (Operator): node to replace
            new (Operator): replacement

        Raises:
            ValueError: if the new :class:`~.Operator` does not act on the same wires as the old one
        """
        # NOTE Does not alter the graph edges in any way. variable_deps is not changed, _grid is not changed. Dangerous!
        if new.wires != old.wires:
            raise ValueError("The new Operator must act on the same wires as the old one.")

        new.queue_idx = old.queue_idx
        self._graph[self._graph.nodes().index(old)] = new

        self._operations = self.operations_in_order
        self._observables = self.observables_in_order

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
                self._operation_graph = self._graph.subgraph(
                    list(self._graph.nodes().index(node) for node in self.operations)
                )
                self._depth = rx.dag_longest_path_length(self._operation_graph) + 1
        return self._depth

    def has_path(self, a, b):
        """Checks if a path exists between the two given nodes.

        Args:
            a (Operator): initial node
            b (Operator): final node

        Returns:
            bool: returns ``True`` if a path exists
        """
        if a == b:
            return True

        return (
            len(
                rx.digraph_dijkstra_shortest_paths(
                    self._graph,
                    self._graph.nodes().index(a),
                    self._graph.nodes().index(b),
                    weight_fn=None,
                    default_weight=1.0,
                    as_undirected=False,
                )
            )
            != 0
        )

    @property
    def max_simultaneous_measurements(self):
        """Returns the maximum number of measurements on any wire in the circuit graph.

        This method counts the number of measurements for each wire and returns
        the maximum.

        **Examples**


        >>> dev = qml.device('default.qubit', wires=3)
        >>> def circuit_measure_max_once():
        ...     return qml.expval(qml.PauliX(wires=0))
        >>> qnode = qml.QNode(circuit_measure_max_once, dev)
        >>> qnode()
        >>> qnode.qtape.graph.max_simultaneous_measurements
        1
        >>> def circuit_measure_max_twice():
        ...     return qml.expval(qml.PauliX(wires=0)), qml.probs(wires=0)
        >>> qnode = qml.QNode(circuit_measure_max_twice, dev)
        >>> qnode()
        >>> qnode.qtape.graph.max_simultaneous_measurements
        2

        Returns:
            int: the maximum number of measurements
        """
        if self._max_simultaneous_measurements is None:
            all_wires = []

            for obs in self.observables:
                all_wires.extend(obs.wires.tolist())

            a = np.array(all_wires)
            _, counts = np.unique(a, return_counts=True)
            self._max_simultaneous_measurements = (
                counts.max() if counts.size != 0 else 1
            )  # qml.state() will result in an empty array
        return self._max_simultaneous_measurements
