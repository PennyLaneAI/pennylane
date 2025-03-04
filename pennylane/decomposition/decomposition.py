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

"""Implements the DecompositionGraph

This module implements a graph-based decomposition algorithm that constructs a graph of operators
connected by decomposition rules, and then traverses it using Dijkstra's algorithm to find the best
decomposition pathways.

The architecture of this module utilizes design patterns similar to those present in Qiskit's
implementation of the basis translator, the Boost Graph library, and RustworkX.

"""

from __future__ import annotations

from dataclasses import dataclass

import rustworkx as rx
from rustworkx.visit import DijkstraVisitor, PruneSearch, StopSearch

import pennylane as qml

from .controlled_decomposition import ControlledDecompositionRule
from .decomposition_rule import DecompositionRule, get_decompositions
from .resources import CompressedResourceOp, Resources, resource_rep


class DecompositionError(Exception):
    """Base class for decomposition errors."""


@dataclass(frozen=True)
class _DecompositionNode:
    """A node that represents a decomposition rule."""

    rule: DecompositionRule
    resource_decomp: Resources

    def count(self, op: CompressedResourceOp):
        """Find the number of occurrences of an operator in the decomposition."""
        return self.resource_decomp.gate_counts.get(op, 0)


class DecompositionGraph:  # pylint: disable=too-many-instance-attributes
    """A graph that models a decomposition problem.

    Args:
        operations (list[Operator]): The list of operations to find decompositions for.
        target_gate_set (set[str]): The names of the gates in the target gate set.
        fixed_decomps (dict): A dictionary mapping operator names to fixed decompositions.
        alt_decomps (dict): A dictionary mapping operator names to alternative decompositions.

    """

    def __init__(
        self,
        operations,
        target_gate_set: set[str],
        fixed_decomps: dict = None,
        alt_decomps: dict = None,
    ):
        self._original_ops = operations
        self._target_gate_set = target_gate_set
        self._original_ops_indices: set[int] = set()
        self._target_gate_indices: set[int] = set()
        self._graph = rx.PyDiGraph()
        self._op_node_indices: dict[CompressedResourceOp, int] = {}
        self._fixed_decomps = fixed_decomps or {}
        self._alt_decomps = alt_decomps or {}
        self._construct_graph()
        self._visitor = None

    def _get_decompositions(self, op_type) -> list[DecompositionRule]:
        """Helper function to get a list of decomposition rules."""
        if op_type in self._fixed_decomps:
            return self._fixed_decomps[op_type]
        if op_type in self._alt_decomps:
            return self._alt_decomps[op_type] + get_decompositions(op_type)
        return get_decompositions(op_type)

    def _construct_graph(self):
        """Constructs the decomposition graph."""
        for op in self._original_ops:
            op_node = resource_rep(type(op), **op.resource_params)
            idx = self._recursively_add_op_node(op_node)
            self._original_ops_indices.add(idx)

    def _recursively_add_op_node(self, op_node: CompressedResourceOp) -> int:
        """Recursively adds an operation node to the graph.

        An operator node is uniquely defined by its operator type and resource parameters, which
        are conveniently wrapped in a ``CompressedResourceOp``.

        """

        if op_node in self._op_node_indices:
            return self._op_node_indices[op_node]

        op_node_idx = self._graph.add_node(op_node)
        self._op_node_indices[op_node] = op_node_idx

        if op_node.op_type.__name__ in self._target_gate_set:
            self._target_gate_indices.add(op_node_idx)
            return op_node_idx

        if issubclass(op_node.op_type, qml.ops.Controlled):
            base_class = op_node.params["base_class"]
            for base_decomposition in base_class.decompositions:
                rule = ControlledDecompositionRule(base_decomposition)
                resource_decomp = rule.compute_resources(**op_node.params)
                d_node_idx = self._recursively_add_decomposition_node(rule, resource_decomp)
                self._graph.add_edge(d_node_idx, op_node_idx, 0)

        for decomposition in self._get_decompositions(op_node.op_type):
            resource_decomp = decomposition.compute_resources(**op_node.params)
            d_node_idx = self._recursively_add_decomposition_node(decomposition, resource_decomp)
            self._graph.add_edge(d_node_idx, op_node_idx, 0)

        return op_node_idx

    def _recursively_add_decomposition_node(
        self, rule: DecompositionRule, resource_decomp: Resources
    ) -> int:
        """Recursively adds a decomposition node to the graph.

        A decomposition node is defined by a decomposition rule and a first-order resource estimate
        of this decomposition as computed with resource params passed from the operator node.

        """

        d_node = _DecompositionNode(rule, resource_decomp)
        d_node_idx = self._graph.add_node(d_node)
        for op in resource_decomp.gate_counts:
            op_node_idx = self._recursively_add_op_node(op)
            self._graph.add_edge(op_node_idx, d_node_idx, (op_node_idx, d_node_idx))
        return d_node_idx

    def solve(self, lazy=True):
        """Solves the graph using the dijkstra search algorithm.

        Args:
            lazy (bool): If True, the dijkstra search will stop once optimal decompositions are
                found for all operations that we originally want to decompose. Otherwise, the
                entire graph will be explored.

        """
        self._visitor = _DecompositionSearchVisitor(self._graph, self._original_ops_indices, lazy)
        start = self._graph.add_node("dummy")
        self._graph.add_edges_from(
            [(start, op_node_idx, 1) for op_node_idx in self._target_gate_indices]
        )
        rx.dijkstra_search(
            self._graph,
            source=[start],
            weight_fn=self._visitor.edge_weight,
            visitor=self._visitor,
        )
        self._graph.remove_node(start)
        if self._visitor.unsolved_op_indices:
            unsolved_ops = [self._graph[op_idx] for op_idx in self._visitor.unsolved_op_indices]
            op_names = set(op.op_type.__name__ for op in unsolved_ops)
            raise DecompositionError(
                f"Decomposition not found for {op_names} to the gate set {self._target_gate_set}"
            )

    def resource_estimates(self, op) -> Resources:
        """Returns the resource estimates for a given operator.

        Args:
            op (Operator): The operator for which to return the resource estimates.

        Returns:
            Resources: The resource estimates.

        """
        op_node = resource_rep(type(op), **op.resource_params)
        op_node_idx = self._op_node_indices[op_node]
        return self._visitor.d[op_node_idx]

    def decomposition(self, op) -> DecompositionRule:
        """Returns the optimal decomposition rule for a given operator.

        Args:
            op (Operator): The operator for which to return the optimal decomposition.

        Returns:
            DecompositionRule: The optimal decomposition.

        """
        op_node = resource_rep(type(op), **op.resource_params)
        op_node_idx = self._op_node_indices[op_node]
        d_node_idx = self._visitor.p[op_node_idx]
        return self._graph[d_node_idx].rule


class _DecompositionSearchVisitor(DijkstraVisitor):
    """The visitor used in the dijkstra search for the optimal decomposition."""

    def __init__(self, graph: rx.PyDiGraph, original_op_indices: set[int], lazy: bool = True):
        self._graph = graph
        self._lazy = lazy
        self.d: dict[int, Resources] = {}  # maps node indices to the optimal resource estimates
        self.p: dict[int, int] = {}  # maps operator nodes to the optimal decomposition nodes
        self.unsolved_op_indices = original_op_indices.copy()
        self._num_edges_examined: dict[int, int] = {}  # keys are decomposition node indices

    def edge_weight(self, edge_obj):
        """Calculates the weight of an edge."""
        if not isinstance(edge_obj, tuple):
            return float(edge_obj)
        op_node_idx, d_node_idx = edge_obj
        return self.d[d_node_idx].num_gates - self.d[op_node_idx].num_gates

    def discover_vertex(self, v, _):
        """Triggered when a vertex is about to be explored during the dijkstra search."""
        self.unsolved_op_indices.discard(v)
        if not self.unsolved_op_indices and self._lazy:
            raise StopSearch

    def examine_edge(self, edge):
        """Triggered when an edge is examined during the dijkstra search."""
        src_idx, target_idx, _ = edge
        src_node = self._graph[src_idx]
        target_node = self._graph[target_idx]
        if not isinstance(target_node, _DecompositionNode):
            return  # nothing is to be done for edges leading to an operator node
        if target_idx not in self.d:
            self.d[target_idx] = Resources()  # initialize with empty resource
        self.d[target_idx] += self.d[src_idx] * target_node.count(src_node)
        if target_idx not in self._num_edges_examined:
            self._num_edges_examined[target_idx] = 0
        self._num_edges_examined[target_idx] += 1
        if self._num_edges_examined[target_idx] < len(target_node.resource_decomp.gate_counts):
            raise PruneSearch

    def edge_relaxed(self, edge):
        """Triggered when an edge is relaxed during the dijkstra search."""
        src_idx, target_idx, _ = edge
        target_node = self._graph[target_idx]
        if self._graph[src_idx] == "dummy":
            self.d[target_idx] = Resources(1, {target_node: 1})
        elif isinstance(target_node, CompressedResourceOp):
            self.p[target_idx] = src_idx
            self.d[target_idx] = self.d[src_idx]
