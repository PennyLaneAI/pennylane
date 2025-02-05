# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implements the graph-based decomposition algorithm."""

from __future__ import annotations

import rustworkx as rx
from rustworkx.visit import DijkstraVisitor, StopSearch, PruneSearch

from .compressed_op import CompressedResourceOp
from .decomposition_rule import DecompositionRule
from .resources import Resources

from pennylane.operation import Operator


class DecompositionGraph:
    """A graph that models a gate set mapping problem.

    Args:
        operations (list[Operator]): The list of operations to decompose.
        target_gate_set (set[str]): The set of supported operator names.

    """

    def __init__(self, operations: list[Operator], target_gate_set: set[str]):
        self._original_ops = operations
        self._original_ops_indices: set[int] = set()
        self._target_gate_set = target_gate_set
        self._target_gate_indices: list[int] = []
        self._graph = rx.PyDiGraph()
        self._node_indices: dict[object, int] = {}
        self._construct_graph()
        self._visitor = None

    def _construct_graph(self):
        """Constructs the decomposition graph."""
        for op in self._original_ops:
            node = CompressedResourceOp(op, op.resource_params)
            idx = self._recursively_add_op_node(node)
            self._original_ops_indices.add(idx)

    def _recursively_add_op_node(self, node: CompressedResourceOp) -> int:
        """Recursively adds an operation node to the graph."""

        if node in self._node_indices:
            return self._node_indices[node]

        op_node_idx = self._graph.add_node(node)
        self._node_indices[node] = op_node_idx
        if node.op_type.__name__ in self._target_gate_set:
            self._target_gate_indices.append(op_node_idx)
            return op_node_idx

        for decomposition in node.op_type.decompositions:
            d_node_idx = self._recursively_add_decomposition_node(decomposition, node.params)
            self._graph.add_edge(d_node_idx, op_node_idx, 0)

        return op_node_idx

    def _recursively_add_decomposition_node(self, node: DecompositionRule, params: dict) -> int:
        """Recursively adds a decomposition node to the graph."""

        d_node_idx = self._graph.add_node(node)
        source_nodes = []
        source_node_indices = []
        resource_decomp = node.compute_resources(**params)
        for op in resource_decomp.gate_counts:
            op_node_idx = self._recursively_add_op_node(op)
            source_nodes.append(op)
            source_node_indices.append(op_node_idx)
        self._graph.add_edges_from(
            [
                (op_node_idx, d_node_idx, (op_node_idx, d_node_idx))
                for op_node_idx in source_node_indices
            ]
        )
        return d_node_idx

    def solve(self, lazy=True):
        """Solves the graph using the dijkstra search algorithm.

        Args:
            lazy (bool): If True, the dijkstra search will stop once optimal decompositions are
                found for all operations that we originally want to decompose. Otherwise, the
                entire graph will be explored.

        """

        self._visitor = _DecompositionSearchVisitor(self._graph, self._original_ops_indices, lazy)
        dummy_node = self._graph.add_node("dummy")
        self._graph.add_edges_from(
            [(dummy_node, op_node_idx, 1) for op_node_idx in self._target_gate_indices]
        )
        rx.dijkstra_search(
            self._graph,
            source=[dummy_node],
            weight_fn=self._visitor.edge_weight,
            visitor=self._visitor,
        )
        self._graph.remove_node(dummy_node)


class _DecompositionSearchVisitor(DijkstraVisitor):
    """The visitor used in the dijkstra search for the optimal decomposition."""

    def __init__(self, graph: rx.PyDiGraph, original_op_indices: set[int], lazy: bool = True):
        self._graph = graph
        self._original_op_indices = original_op_indices.copy()
        self.d: dict[CompressedResourceOp | DecompositionRule, Resources] = {}
        self.p: dict[CompressedResourceOp, DecompositionRule] = {}
        self._num_edges_examined: dict[DecompositionRule, int] = {}
        self._lazy = lazy

    def edge_weight(self, edge_obj):
        """Calculates the weight of an edge."""
        if not isinstance(edge_obj, tuple):
            return edge_obj
        op_node_idx, d_node_idx = edge_obj
        op_node, d_node = self._graph[op_node_idx], self._graph[d_node_idx]
        return self.d[d_node].num_gates - self.d[op_node].num_gates

    def discover_vertex(self, v, score):
        """Triggered when a vertex is about to be explored during the dijkstra search."""
        self._original_op_indices.discard(v)
        if not self._original_op_indices and self._lazy:
            raise StopSearch

    def examine_edge(self, edge):
        """Triggered when an edge is examined during the dijkstra search."""
        src, target, obj = edge
        src_node = self._graph[src]
        target_node = self._graph[target]
        if not isinstance(target_node, DecompositionRule):
            return  # nothing is to be done for edges leading to an operator node
        self.d[target_node] = self.d.get(target_node, Resources()) + self.d[src_node]
        self._num_edges_examined[target_node] += 1
        if self._num_edges_examined[target_node] < obj[0].num_gate_types:
            raise PruneSearch

    def edge_relaxed(self, edge):
        """Triggered when an edge is relaxed during the dijkstra search."""
        src, target, obj = edge
        target_node = self._graph[target]
        if isinstance(target_node, CompressedResourceOp):
            self.p[target_node] = self._graph[src]
