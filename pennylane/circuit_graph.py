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
representation of a quantum circuit from an Operator queue.
"""
from collections import namedtuple, Counter, OrderedDict
import networkx as nx

import numpy as np
from .utils import _flatten
import pennylane as qml


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
        ops (Iterable[Operator]): quantum operators constituting the circuit, in temporal order
        variable_deps (dict[int, list[ParameterDependency]]): Free parameters of the quantum circuit.
            The dictionary key is the parameter index.
    """

    def __init__(self, ops, variable_deps):
        self.variable_deps = variable_deps

        self._grid = {}
        """dict[int, list[Operator]]: dictionary representing the quantum circuit as a grid.
        Here, the key is the wire number, and the value is a list containing the operators on that wire.
        """
        for k, op in enumerate(ops):
            op.queue_idx = k  # store the queue index in the Operator
            for w in set(
                _flatten(op.wires)
            ):  # flatten the nested wires lists of Tensor observables
                # Add op to the grid, to the end of wire w
                self._grid.setdefault(w, []).append(op)

        # TODO: State preparations demolish the incoming state entirely, and therefore should have no incoming edges.

        self._graph = nx.DiGraph()  #: nx.DiGraph: DAG representation of the quantum circuit
        # Iterate over each (populated) wire in the grid
        for wire in self._grid.values():
            # Add the first operator on the wire to the graph
            # This operator does not depend on any others
            self._graph.add_node(wire[0])

            for i in range(1, len(wire)):
                # For subsequent operators on the wire:
                if wire[i] not in self._graph:
                    # Add them to the graph if they are not already
                    # in the graph (multi-qubit operators might already have been placed)
                    self._graph.add_node(wire[i])

                # Create an edge between this and the previous operator
                self._graph.add_edge(wire[i - 1], wire[i])

    @property
    def observables_in_order(self):
        """Observables in the circuit, in a fixed topological order.

        The topological order used by this method is guaranteed to be the same
        as the order in which the measured observables are returned by the quantum function.
        Currently the topological order is determined by the queue index.

        Returns:
            list[Observable]: observables
        """
        nodes = [node for node in self._graph.nodes if _is_observable(node)]
        return sorted(nodes, key=_by_idx)

    observables = observables_in_order

    @property
    def operations_in_order(self):
        """Operations in the circuit, in a fixed topological order.

        Currently the topological order is determined by the queue index.

        The complement of :meth:`QNode.observables`. Together they return every :class:`Operator`
        instance in the circuit.

        Returns:
            list[Operation]: operations
        """
        nodes = [node for node in self._graph.nodes if not _is_observable(node)]
        return sorted(nodes, key=_by_idx)

    operations = operations_in_order

    @property
    def graph(self):
        """The graph representation of the quantum circuit.

        The graph has nodes representing :class:`.Operator` instances,
        and directed edges pointing from nodes to their immediate dependents/successors.

        Returns:
            networkx.DiGraph: the directed acyclic graph representing the quantum circuit
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
        return set().union(*(nx.dag.ancestors(self._graph, o) for o in ops)) - set(ops)

    def descendants(self, ops):
        """Descendants of a given set of operators.

        Args:
            ops (Iterable[Operator]): set of operators in the circuit

        Returns:
            set[Operator]: descendants of the given operators
        """
        return set().union(*(nx.dag.descendants(self._graph, o) for o in ops)) - set(ops)

    def _in_topological_order(self, ops):
        """Sorts a set of operators in the circuit in a topological order.

        Args:
            ops (Iterable[Operator]): set of operators in the circuit

        Returns:
            Iterable[Operator]: same set of operators, topologically ordered
        """
        G = nx.DiGraph(self._graph.subgraph(ops))
        return nx.dag.topological_sort(G)

    def ancestors_in_order(self, ops):
        """Operator ancestors in a topological order.

        Currently the topological order is determined by the queue index.

        Args:
            ops (Iterable[Operator]): set of operators in the circuit

        Returns:
            list[Operator]: ancestors of the given operators, topologically ordered
        """
        # return self._in_topological_order(self.ancestors(ops))  # an abitrary topological order
        return sorted(self.ancestors(ops), key=_by_idx)

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
                    # operator depends on current layer, start a new layer
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

    @staticmethod
    def list_at_index_or_none(list, idx):
        if len(list) > idx:
            return list[idx]
        else:
            return None

    @staticmethod
    def empty_list_to_none(list):
        if list:
            return list
        else:
            return [None]

    def greedy_layers(self):
        l = 0

        greedy_grid = OrderedDict()
        for key in sorted(self._grid):
            greedy_grid[key] = self._grid[key]

        greedy_grid = {
            wire: list(
                filter(
                    lambda op: not (
                        isinstance(op, qml.operation.Observable) and op.return_type is not None
                    ),
                    greedy_grid[wire],
                )
            )
            for wire in greedy_grid
        }

        while True:
            layer_ops = {
                wire: CircuitGraph.list_at_index_or_none(greedy_grid[wire], l)
                for wire in greedy_grid
            }
            num_ops = Counter(layer_ops.values())

            if None in num_ops and num_ops[None] == len(greedy_grid):
                break

            for (wire, op) in layer_ops.items():
                if op is None:
                    greedy_grid[wire].append(None)
                    continue

                # push back to next layer if not all args wires are there yet
                if op.num_wires > num_ops[op]:
                    greedy_grid[wire].insert(l, None)

            l += 1

        observables = OrderedDict()
        for wire in sorted(self._grid):
            observables[wire] = CircuitGraph.empty_list_to_none(
                list(
                    filter(
                        lambda op: isinstance(op, qml.operation.Observable)
                        and op.return_type is not None,
                        self._grid[wire],
                    )
                )
            )

        return (
            [greedy_grid[wire] for wire in greedy_grid],
            [observables[wire] for wire in observables],
        )

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
        nx.relabel_nodes(self._graph, {old: new}, copy=False)  # change the graph in place

    def render(self):
        grid, obs = self.greedy_layers()

        drawer = CircuitDrawer(grid, obs)

        drawer.render()


class Grid:
    def __init__(self, raw_grid=[]):
        self.raw_grid = raw_grid
        self.raw_grid_transpose = list(map(list, zip(*raw_grid)))

    def insert_layer(self, idx, layer):
        self.raw_grid_transpose.insert(idx, layer)
        self.raw_grid = list(map(list, zip(*self.raw_grid_transpose)))

    def append_layer(self, layer):
        self.raw_grid_transpose.append(layer)
        self.raw_grid = list(map(list, zip(*self.raw_grid_transpose)))

    def replace_layer(self, idx, layer):
        self.raw_grid_transpose[idx] = layer
        self.raw_grid = list(map(list, zip(*self.raw_grid_transpose)))

    def insert_wire(self, idx, wire):
        self.raw_grid.insert(idx, wire)
        self.raw_grid_transpose = list(map(list, zip(*self.raw_grid)))

    def append_wire(self, wire):
        self.raw_grid.append(wire)
        self.raw_grid_transpose = list(map(list, zip(*self.raw_grid)))

    @property
    def num_layers(self):
        return len(self.raw_grid_transpose)

    def layer(self, idx):
        return self.raw_grid_transpose[idx]

    @property
    def num_wires(self):
        return len(self.raw_grid)

    def wire(self, idx):
        return self.raw_grid[idx]

    def copy(self):
        return Grid(self.raw_grid.copy())

    def append_grid_by_layers(self, other_grid):
        for i in range(other_grid.num_layers):
            self.append_layer(other_grid.layer(i))

    def __str__(self):
        ret = ""
        for wire in self.raw_grid:
            ret += str(wire)
            ret += "\n"

        return ret


class UnicodeCharSet:
    WIRE = "─"
    MEASUREMENT = "┤"
    TOP_MULTI_LINE_GATE_CONNECTOR = "╭"
    MIDDLE_MULTI_LINE_GATE_CONNECTOR = "├"
    BOTTOM_MULTI_LINE_GATE_CONNECTOR = "╰"
    EMPTY_MULTI_LINE_GATE_CONNECTOR = "│"
    CONTROL = "C"
    LANGLE = "⟨"
    RANGLE = "⟩"


class RepresentationResolver:
    resolution_dict = {
        "PauliX": "X",
        "CNOT": "X",
        "Toffoli": "X",
        "PauliY": "Y",
        "PauliZ": "Z",
        "CZ": "Z",
        "Identity": "I",
        "Hadamard": "H",
        "CRX": "RX",
        "CRY": "RY",
        "CRZ": "RZ",
        "QubitUnitary" : "U"
    }

    def __init__(self, charset=UnicodeCharSet):
        self.charset = charset
        self._matrix_cache = []

    def render_parameter(self, par):
        if isinstance(par, qml.variable.Variable):
            return par.render()

        if isinstance(par, np.ndarray):
            if not par in self._matrix_cache:
                self._matrix_cache.append(par)

            return "M{}".format(self._matrix_cache.index(par))

        return str(par)

    def operation_representation(self, op, wire):
        name = op.name

        if name in RepresentationResolver.resolution_dict:
            name = RepresentationResolver.resolution_dict[name]

        if op.num_wires > 1 and wire != op.wires[-1]:
            return self.charset.CONTROL

        if op.num_params == 0:
            return name

        return "{}({})".format(name, ",".join([self.render_parameter(par) for par in op.params]))

    def observable_representation(self, obs, wire):
        if obs.return_type == qml.operation.Expectation:
            return (
                self.charset.LANGLE
                + "{}".format(self.operation_representation(obs, wire))
                + self.charset.RANGLE
            )
        elif obs.return_type == qml.operation.Variance:
            return "Var[{}]".format(self.operation_representation(obs, wire))
        elif obs.return_type == qml.operation.Sample:
            return "Sample[{}]".format(self.operation_representation(obs, wire))

    def operator_representation(self, op, wire):
        if op is None:
            return ""
        elif isinstance(op, str):
            return op
        elif isinstance(op, qml.operation.Observable) and op.return_type is not None:
            return self.observable_representation(op, wire)
        else:
            return self.operation_representation(op, wire)


class CircuitDrawer:
    def resolve_representation(self, grid, representation_grid):
        for i in range(grid.num_layers):
            representation_layer = [""] * grid.num_wires

            for wire, operator in enumerate(grid.layer(i)):
                representation_layer[wire] = self.representation_resolver.operator_representation(
                    operator, wire
                )

            representation_grid.append_layer(representation_layer)

    def resolve_decorations(self, grid, representation_grid, decoration_indices):
        print("Resolving decorations: ")
        print("grid = ", grid)
        print("representation_grid = ", representation_grid)
        print("decoration_indices = ", decoration_indices)

        j = 0
        for i in range(grid.num_layers):
            layer_operators = set(grid.layer(i))

            for op in layer_operators:
                if op is None:
                    continue
                
                print("op = ", op)
                print("op.num_wires = ", op.num_wires)
                if len(op.wires) > 1:
                    decoration_layer = [""] * grid.num_wires
                    sorted_wires = op.wires.copy()
                    sorted_wires.sort()

                    decoration_layer[sorted_wires[0]] = self.charset.TOP_MULTI_LINE_GATE_CONNECTOR
                    decoration_layer[
                        sorted_wires[-1]
                    ] = self.charset.BOTTOM_MULTI_LINE_GATE_CONNECTOR
                    for k in range(sorted_wires[0] + 1, sorted_wires[-1]):
                        if k in sorted_wires:
                            decoration_layer[k] = self.charset.MIDDLE_MULTI_LINE_GATE_CONNECTOR
                        else:
                            decoration_layer[k] = self.charset.EMPTY_MULTI_LINE_GATE_CONNECTOR

                    representation_grid.insert_layer(i + j, decoration_layer)
                    decoration_indices.append(i + j)
                    j += 1

    def justify_and_prepend(self, x, prepend_str, suffix_str, max_width, pad_str):
        return prepend_str + str.ljust(x, max_width, pad_str) + suffix_str

    def resolve_padding(self, representation_grid, pad_str, skip_prepend_pad_str, prepend_str, suffix_str, skip_prepend_idx):
        for i in range(representation_grid.num_layers):
            layer = representation_grid.layer(i)
            max_width = max(map(len, layer))

            if i in skip_prepend_idx:
                representation_grid.replace_layer(
                    i, list(map(lambda x: self.justify_and_prepend(x, "", "", max_width, skip_prepend_pad_str), layer))
                )
            else:
                representation_grid.replace_layer(
                    i, list(map(lambda x: self.justify_and_prepend(x, prepend_str, suffix_str, max_width, pad_str), layer))
                )

    def __init__(self, raw_operator_grid, raw_observable_grid, charset=UnicodeCharSet):
        self.charset = charset
        self.representation_resolver = RepresentationResolver(charset)
        self.operator_grid = Grid(raw_operator_grid)
        self.observable_grid = Grid(raw_observable_grid)
        self.operator_representation_grid = Grid()
        self.observable_representation_grid = Grid()
        self.operator_decoration_indices = []
        self.observable_decoration_indices = []

        # Move intertwined multi-wire gates
        # for i in range(self.operator_grid.num_layers):
        #     layer_ops = set(self.operator_grid.layer(i))

        #     for j in range(len(layer_ops)):
        #         op = layer_ops[j]

        #         if op is None:
        #             continue
        
        #         if len(op.wires) > 1:
        #             sorted_wires = op.wires.copy()
        #             sorted_wires.sort()

        #             candidate_wires = range(sorted_wires[0] + 1, sorted_wires[-1])

        #             for wire in candidate_wires:


        # Resolve operator names
        self.resolve_representation(self.operator_grid, self.operator_representation_grid)
        self.resolve_representation(self.observable_grid, self.observable_representation_grid)

        # Add multi-wire gate lines
        self.resolve_decorations(
            self.operator_grid, self.operator_representation_grid, self.operator_decoration_indices
        )
        self.resolve_decorations(
            self.observable_grid,
            self.observable_representation_grid,
            self.observable_decoration_indices,
        )

        self.resolve_padding(
            self.operator_representation_grid,
            charset.WIRE,
            charset.WIRE,
            "",
            2 * charset.WIRE,
            self.operator_decoration_indices,
        )
        self.resolve_padding(
            self.observable_representation_grid,
            " ",
            charset.WIRE,
            charset.MEASUREMENT + " ",
            " ",
            self.observable_decoration_indices,
        )

        self.full_representation_grid = self.operator_representation_grid.copy()
        self.full_representation_grid.append_grid_by_layers(self.observable_representation_grid)

    def render(self):
        for i in range(self.full_representation_grid.num_wires):
            wire = self.full_representation_grid.wire(i)

            print("{:2d}: {}".format(i, 2*self.charset.WIRE), end="")

            for s in wire:
                print(s, end="")

            print()

        for idx, matrix in enumerate(self.representation_resolver._matrix_cache):
            print("M{} =\n{}\n".format(idx, matrix))


# TODO:
# * Move crossing multi-wire gates to different layers
# * Support multi-wire measurements
# * Move to QNode, enable printing of unevaluated QNodes
