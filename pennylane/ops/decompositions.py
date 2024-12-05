# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Defines a library of decompositions and implements graph-based gate set mapping."""

from functools import cached_property

import rustworkx as rx
import sympy as sp
from rustworkx.visit import DijkstraVisitor, StopSearch, PruneSearch
from sympy.utilities.lambdify import lambdify

import pennylane as qml
from pennylane.wires import WiresLike
from pennylane.typing import TensorLike
from pennylane.operation import Operator


class _Decomposition:
    """Base class for a decomposition rule."""

    @property
    def decomposed_ops(self) -> list[Operator]:
        """The list of decomposed operations."""
        raise NotImplementedError

    def bind(
        self,
        *params: TensorLike,
        wires: WiresLike = None,
    ) -> list[Operator]:
        """Binds input parameters and wires to the decomposed operations."""
        raise NotImplementedError


class _StaticDecomposition(_Decomposition):
    """A decomposition that does not depend on the operation instance.

    A static decomposition returns the same sequence of operations for all instances of the
    same operation class, which is the case for most operations.

    """

    def __init__(self, op_class: type[Operator], decomposer: callable):
        self.op_class = op_class
        self.decomposer = decomposer
        self._sym_params = sp.symbols(f"x0:{self.op_class.num_params}")  # symbolic parameters

    @cached_property
    def decomposed_ops(self) -> list[Operator]:
        """The list of decomposed operations with symbolic data."""
        return self.decomposer(*self._sym_params, wires=range(self.op_class.num_wires))

    def bind(
        self,
        *params: TensorLike,
        wires: WiresLike = None,
    ) -> list[Operator]:
        """Binds input parameters and wires to the decomposed operations."""
        wire_map = dict(zip(range(self.op_class.num_wires), wires))
        operations = [qml.map_wires(op, wire_map) for op in self.decomposed_ops]
        data, struct = qml.pytrees.flatten(operations)
        new_data = [self._bind_params_to_expr(params, expr) for expr in data]
        return qml.pytrees.unflatten(new_data, struct)

    def _bind_params_to_expr(self, params, expression):
        """Binds concrete input data to a sympy expression."""
        if qml.math.get_interface(expression) == "sympy":
            evaluator = lambdify(self._sym_params, expression)
            return evaluator(*params)
        return expression

    def __hash__(self):
        return hash((self.op_class, self.decomposer))

    def __eq__(self, other):
        return hash(self) == hash(other)


class _DynamicDecomposition(_Decomposition):
    """A decomposition that depends on the operation instance.

    A dynamic decomposition returns different sequences of operations which may depend on the
    data of the operation instance. This includes most symbolic operations such as Exp whose
    decomposition depends on the coefficients.

    TODO: implement this class and the logic associated with it. For the scope of this
        prototype, assume all operations have a static decomposition.

    """


class DecompositionLibrary:
    """A library for all possible decompositions of each operation."""

    def __init__(self):
        self._library: dict[type, list[_Decomposition]] = {}

    def register_static_decomposition(self, op_class: type[Operator], decomposer: callable) -> None:
        """Registers a static decomposition for a given operation."""
        self._library.setdefault(op_class, []).append(_StaticDecomposition(op_class, decomposer))

    def get_decompositions(self, op_class: type[Operator]) -> list[_Decomposition]:
        """Returns all registered decompositions for a given operation."""
        return self._library.get(op_class, [])


class _Node:
    """A node in the decomposition graph."""


class _OpNode(_Node):
    """A node that represents an operation."""

    def __init__(self, op: Operator):
        self.op_class = op.__class__

    def __hash__(self):
        return hash(self.op_class)

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.op_class == other.op_class

    def __repr__(self):
        return f"OpNode({self.op_class.__name__})"


class _DecompositionNode(_Node):
    """A node that represents a decomposition rule."""

    def __init__(self, decomposition: _Decomposition):
        self.decomposition = decomposition

    def __hash__(self):
        return hash(self.decomposition)

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.decomposition == other.decomposition

    def __repr__(self):
        assert isinstance(self.decomposition, _StaticDecomposition)
        return f"DecompositionNode({self.decomposition.op_class.__name__})"


class _Edge:
    """An edge in the decomposition graph."""

    def weight(self, visitor: "_DecompositionSearchVisitor") -> float:
        raise NotImplementedError


class _EdgeToOp(_Edge):
    """An edge that connects a decomposition to an operation."""

    def weight(self, _) -> float:
        return 0.0


class _EdgeToDecomposition(_Edge):
    """An edge that connects an operation to the decomposition that contains it."""

    def __init__(
        self, source_node: _OpNode, target_node: _DecompositionNode, source_nodes: list[_OpNode]
    ):
        self.source_node = source_node
        self.target_node = target_node
        self.source_nodes = source_nodes
        self.num_unique_sources = len(set(source_nodes))

    def weight(self, visitor: "_DecompositionSearchVisitor") -> float:
        return sum(visitor.d[src] for src in self.source_nodes) - visitor.d[self.source_node]


class DecompositionGraph:
    """A graph that models a gate set mapping problem.

    Args:
        operations (list[Operator]): The list of operations to decompose.
        target_gate_set (set[str]): The set of supported operator names.
        library (DecompositionLibrary): The decomposition library.

    """

    def __init__(
        self, operations: list[Operator], target_gate_set: set[str], library: DecompositionLibrary
    ):
        self._operations = operations
        self._objective_node_indices = set()
        self._target_gate_set = target_gate_set
        self._library = library
        self._graph = rx.PyDiGraph()
        self._node_indices: dict[_Node, int] = {}
        self._source_node_indices: list[int] = []
        self._construct_graph()
        self._visitor = None

    def _construct_graph(self):
        """Constructs the decomposition graph."""
        for op in self._operations:
            node = _OpNode(op)
            idx = self._recursively_add_op_node(node)
            self._objective_node_indices.add(idx)

    def _recursively_add_op_node(self, node: _OpNode) -> int:
        """Recursively adds an operation node to the graph."""

        if node in self._node_indices:
            return self._node_indices[node]

        op_node_idx = self._graph.add_node(node)
        self._node_indices[node] = op_node_idx
        if node.op_class.__name__ in self._target_gate_set:
            self._source_node_indices.append(op_node_idx)
            return op_node_idx

        for decomposition in self._library.get_decompositions(node.op_class):
            decomposition_node = _DecompositionNode(decomposition)
            d_node_idx = self._recursively_add_decomposition_node(decomposition_node)
            self._graph.add_edge(d_node_idx, op_node_idx, _EdgeToOp())

        return op_node_idx

    def _recursively_add_decomposition_node(self, node: _DecompositionNode) -> int:
        """Recursively adds a decomposition node to the graph."""

        node_idx = self._graph.add_node(node)
        source_nodes = []
        source_node_indices = []
        for op in node.decomposition.decomposed_ops:
            op_node = _OpNode(op)
            op_node_idx = self._recursively_add_op_node(op_node)
            source_nodes.append(op_node)
            source_node_indices.append(op_node_idx)
        unique_sources = set(zip(source_node_indices, source_nodes))
        self._graph.add_edges_from(
            [
                (op_node_idx, node_idx, _EdgeToDecomposition(op_node, node, source_nodes))
                for op_node_idx, op_node in unique_sources
            ]
        )
        return node_idx

    def solve(self, lazy=True):
        """Solves the graph using the dijkstra search algorithm.

        Args:
            lazy (bool): If True, the dijkstra search will stop once optimal decompositions are
                found for all operations that we originally want to decompose. Otherwise, the
                entire graph will be explored.

        """
        self._visitor = _DecompositionSearchVisitor(self._graph, self._objective_node_indices, lazy)
        dummy_node = self._graph.add_node(_Node())
        self._graph.add_edges_from(
            [(dummy_node, op_node_idx, None) for op_node_idx in self._source_node_indices]
        )
        rx.dijkstra_search(
            self._graph,
            source=[dummy_node],
            weight_fn=self._visitor.edge_weight,
            visitor=self._visitor,
        )
        self._graph.remove_node(dummy_node)

    def decompose(self, operator: Operator) -> list[Operator]:
        """Decomposes the given operator using the optimal decomposition."""

        if self._visitor is None:
            self.solve()

        return list(self._recursive_decomposition_gen(operator))

    def _recursive_decomposition_gen(self, op: Operator) -> list[Operator]:
        """Recursively find the optimal decomposition for the given operation."""

        if op.name in self._target_gate_set:
            yield op

        else:
            op_node = _OpNode(op)
            predecessor = self._visitor.p[op_node]
            decomposed_ops = predecessor.decomposition.bind(*op.data, wires=op.wires)
            for op in decomposed_ops:
                yield from self._recursive_decomposition_gen(op)


class _DecompositionSearchVisitor(DijkstraVisitor):
    """The visitor used in the dijkstra search for the optimal decomposition."""

    def __init__(self, graph: rx.PyDiGraph, objective_nodes_remaining: set[int], lazy: bool = True):
        self._graph = graph
        self._objective_nodes_remaining = objective_nodes_remaining.copy()
        self._num_src_ops_unvisited_for_decomposition: dict[_DecompositionNode, int] = {}
        _checked_nodes: set[_DecompositionNode] = set()
        for edge in self._graph.edges():
            if not isinstance(edge, _EdgeToDecomposition):
                continue
            target = edge.target_node
            if target in _checked_nodes:
                continue
            _checked_nodes.add(target)
            self._num_src_ops_unvisited_for_decomposition[target] = edge.num_unique_sources
        self.d: dict[_Node, int] = {}
        self.p: dict[_OpNode, _DecompositionNode] = {}
        self._lazy = lazy

    def edge_weight(self, edge):
        """Returns the weight of an edge."""
        if edge is None:
            # This corresponds to the edge from the dummy node to an OpNode
            # representing an operator in the target gate set.
            return 1
        return edge.weight(self)

    def discover_vertex(self, v, score):
        """Triggered when a vertex is visited during the dijkstra search."""
        node = self._graph[v]
        self.d[node] = int(score)
        self._objective_nodes_remaining.discard(v)
        if not self._objective_nodes_remaining and self._lazy:
            raise StopSearch

    def examine_edge(self, edge):
        """Triggered when an edge is examined during the dijkstra search."""
        _, target, _ = edge
        target_node = self._graph[target]
        if isinstance(target_node, _OpNode):
            return
        self._num_src_ops_unvisited_for_decomposition[target_node] -= 1
        if self._num_src_ops_unvisited_for_decomposition[target_node] > 0:
            raise PruneSearch

    def edge_relaxed(self, edge):
        """Triggered when an edge is relaxed during the dijkstra search."""
        src, target, edge_obj = edge
        target_node = self._graph[target]
        if isinstance(edge_obj, _EdgeToOp):
            self.p[target_node] = self._graph[src]
