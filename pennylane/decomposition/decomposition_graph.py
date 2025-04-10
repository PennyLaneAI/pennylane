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

"""Implements the class DecompositionGraph

This module implements a graph-based decomposition algorithm that constructs a graph of operators
connected by decomposition rules, and then traverses it using Dijkstra's algorithm to find the best
decomposition for every operator.

The architecture of this module utilizes design patterns similar to those present in Qiskit's
implementation of the basis translator, the Boost Graph library, and RustworkX.

"""

from __future__ import annotations

from dataclasses import dataclass

import rustworkx as rx
from rustworkx.visit import DijkstraVisitor, PruneSearch, StopSearch

import pennylane as qml
from pennylane.operation import Operator

from .controlled_decomposition import (
    ControlledBaseDecomposition,
    CustomControlledDecomposition,
    base_to_custom_ctrl_op,
    controlled_global_phase_decomp,
    controlled_x_decomp,
)
from .decomposition_rule import DecompositionRule, list_decomps
from .resources import CompressedResourceOp, Resources, resource_rep
from .symbolic_decomposition import (
    AdjointDecomp,
    adjoint_adjoint_decomp,
    adjoint_controlled_decomp,
    adjoint_pow_decomp,
    pow_decomp,
    pow_pow_decomp,
    same_type_adjoint_decomp,
    same_type_adjoint_ops,
)
from .utils import DecompositionError


class DecompositionGraph:  # pylint: disable=too-many-instance-attributes
    """A graph that models a decomposition problem.

    The decomposition graph contains two types of nodes: operator nodes and decomposition nodes.
    Each decomposition node is a :class:`pennylane.decomposition.DecompositionRule`, and each
    operator node is a :class:`~pennylane.decomposition.resources.CompressedResourceOp` which
    contains an operator type and any additional parameters that affects the resource requirements
    of the operator. Essentially, two instances of the same operator type are represented by the
    same node in the graph if they're expected to have the same decompositions.

    There are also two types of directed edges: edges that connect operators to the decomposition
    rules that contain them, and edges that connect decomposition rules to the operators that they
    decompose. The edge weights represent the difference in the gate count between the two states.
    Edges that connect decomposition rules to operators have a weight of 0 because an operator can
    be replaced with its decomposition at no additional cost.

    On the other hand, edges that connect operators to the decomposition rule that contains them
    will have a weight that is the total resource estimate of the decomposition minus the resource
    estimate of the operator. For example the edge that connects a ``CNOT`` to the following
    decomposition rule:

    .. code-block:: python

        import pennylane as qml

        @qml.register_resources({qml.H: 2, qml.CNOT: 1})
        def my_cz(wires):
            qml.H(wires=wires[1])
            qml.CNOT(wires=wires)
            qml.H(wires=wires[1])

    will have a weight of 2, because the decomposition rule contains 2 additional ``H`` gates.
    Note that this gate count is in terms of gates in the target gate set. If ``H`` isn't supported
    and is in turn decomposed to two ``RZ`` gates and a ``RX`` gate, the weight of this edge
    becomes 2 * 3 = 6. This way, the total distance from a basis gate to a high-level gate is
    conveniently the total number of basis gates required to decompose this high-level gate, which
    allows us to use Dijkstra's algorithm to find the most efficient decomposition.

    Args:
        operations (list[Operator or CompressedResourceOp]): The list of operations to decompose.
        target_gate_set (set[str]): The names of the gates in the target gate set.
        fixed_decomps (dict): A dictionary mapping operator names to fixed decompositions.
        alt_decomps (dict): A dictionary mapping operator names to alternative decompositions.

    **Example**

    .. code-block:: python

        op = qml.CRX(0.5, wires=[0, 1])
        graph = DecompositionGraph(
            operations=[op],
            target_gate_set={"RZ", "RX", "CNOT", "GlobalPhase"},
        )
        graph.solve()

    >>> with qml.queuing.AnnotatedQueue() as q:
    ...     graph.decomposition(op)(0.5, wires=[0, 1])
    >>> q.queue
    [RZ(1.5707963267948966, wires=[1]),
     RY(0.25, wires=[1]),
     CNOT(wires=[0, 1]),
     RY(-0.25, wires=[1]),
     CNOT(wires=[0, 1]),
     RZ(-1.5707963267948966, wires=[1])]
    >>> graph.resource_estimate(op)
    <num_gates=10, gate_counts={RZ: 6, CNOT: 2, RX: 2}>

    """

    def __init__(
        self,
        operations: list[Operator | CompressedResourceOp],
        target_gate_set: set[str],
        fixed_decomps: dict = None,
        alt_decomps: dict = None,
    ):
        self._original_ops = operations
        self._target_gate_set = target_gate_set
        self._original_ops_indices: set[int] = set()
        self._target_gate_indices: set[int] = set()
        self._op_node_indices: dict[CompressedResourceOp, int] = {}
        self._fixed_decomps = fixed_decomps or {}
        self._alt_decomps = alt_decomps or {}
        self._graph = rx.PyDiGraph()
        self._visitor = None

        # Construct the decomposition graph
        self._construct_graph()

    def _get_decompositions(self, op_type) -> list[DecompositionRule]:
        """Helper function to get a list of decomposition rules."""
        if op_type in self._fixed_decomps:
            return [self._fixed_decomps[op_type]]
        return self._alt_decomps.get(op_type, []) + list_decomps(op_type)

    def _construct_graph(self):
        """Constructs the decomposition graph."""
        for op in self._original_ops:
            if isinstance(op, Operator):
                op = resource_rep(type(op), **op.resource_params)
            idx = self._recursively_add_op_node(op)
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

        if op_node.op_type in (qml.ops.Controlled, qml.ops.ControlledOp):
            # This branch only applies to general controlled operators
            return self._add_controlled_decomp_node(op_node, op_node_idx)

        if issubclass(op_node.op_type, qml.ops.Adjoint):
            return self._add_adjoint_decomp_node(op_node, op_node_idx)

        if issubclass(op_node.op_type, qml.ops.Pow):
            return self._add_pow_decomp_node(op_node, op_node_idx)

        for decomposition in self._get_decompositions(op_node.op_type):
            decomp_resource = decomposition.compute_resources(**op_node.params)
            d_node_idx = self._recursively_add_decomposition_node(decomposition, decomp_resource)
            self._graph.add_edge(d_node_idx, op_node_idx, 0)

        return op_node_idx

    def _add_special_decomp_rule_to_op(
        self, rule: DecompositionRule, op_node: CompressedResourceOp, op_node_idx: int
    ):
        """Adds a special decomposition rule to the graph."""
        decomp_resource = rule.compute_resources(**op_node.params)
        d_node_idx = self._recursively_add_decomposition_node(rule, decomp_resource)
        self._graph.add_edge(d_node_idx, op_node_idx, 0)

    def _add_adjoint_decomp_node(self, op_node: CompressedResourceOp, op_node_idx: int) -> int:
        """Adds an adjoint decomposition node."""

        base_class, base_params = op_node.params["base_class"], op_node.params["base_params"]

        if issubclass(base_class, qml.ops.Adjoint):
            rule = adjoint_adjoint_decomp
            self._add_special_decomp_rule_to_op(rule, op_node, op_node_idx)
            return op_node_idx

        if (
            issubclass(base_class, qml.ops.Pow)
            and base_params["base_class"] in same_type_adjoint_ops()
        ):
            rule = adjoint_pow_decomp
            self._add_special_decomp_rule_to_op(rule, op_node, op_node_idx)
            return op_node_idx

        if base_class in same_type_adjoint_ops():
            rule = same_type_adjoint_decomp
            self._add_special_decomp_rule_to_op(rule, op_node, op_node_idx)
            return op_node_idx

        if (
            issubclass(base_class, qml.ops.Controlled)
            and base_params["base_class"] in same_type_adjoint_ops()
        ):
            rule = adjoint_controlled_decomp
            self._add_special_decomp_rule_to_op(rule, op_node, op_node_idx)
            return op_node_idx

        for base_decomposition in self._get_decompositions(base_class):
            rule = AdjointDecomp(base_decomposition)
            self._add_special_decomp_rule_to_op(rule, op_node, op_node_idx)

        return op_node_idx

    def _add_pow_decomp_node(self, op_node: CompressedResourceOp, op_node_idx: int) -> int:
        """Adds a power decomposition node to the graph."""

        base_class = op_node.params["base_class"]

        if issubclass(base_class, qml.ops.Pow):
            rule = pow_pow_decomp
            self._add_special_decomp_rule_to_op(rule, op_node, op_node_idx)
            return op_node_idx

        rule = pow_decomp
        self._add_special_decomp_rule_to_op(rule, op_node, op_node_idx)
        return op_node_idx

    def _add_controlled_decomp_node(self, op_node: CompressedResourceOp, op_node_idx: int) -> int:
        """Adds a controlled decomposition node to the graph."""

        base_class = op_node.params["base_class"]
        num_control_wires = op_node.params["num_control_wires"]

        # Handle controlled global phase
        if base_class is qml.GlobalPhase:
            rule = controlled_global_phase_decomp
            self._add_special_decomp_rule_to_op(rule, op_node, op_node_idx)
            return op_node_idx

        # Handle controlled-X gates
        if base_class is qml.X:
            rule = controlled_x_decomp
            self._add_special_decomp_rule_to_op(rule, op_node, op_node_idx)
            return op_node_idx

        # Handle custom controlled ops
        if (base_class, num_control_wires) in base_to_custom_ctrl_op():
            custom_op_type = base_to_custom_ctrl_op()[(base_class, num_control_wires)]
            rule = CustomControlledDecomposition(custom_op_type)
            self._add_special_decomp_rule_to_op(rule, op_node, op_node_idx)
            return op_node_idx

        # General case
        for base_decomposition in self._get_decompositions(base_class):
            rule = ControlledBaseDecomposition(base_decomposition)
            self._add_special_decomp_rule_to_op(rule, op_node, op_node_idx)

        return op_node_idx

    def _recursively_add_decomposition_node(
        self, rule: DecompositionRule, decomp_resource: Resources
    ) -> int:
        """Recursively adds a decomposition node to the graph.

        A decomposition node is defined by a decomposition rule and a first-order resource estimate
        of this decomposition as computed with resource params passed from the operator node.

        """

        d_node = _DecompositionNode(rule, decomp_resource)
        d_node_idx = self._graph.add_node(d_node)
        for op in decomp_resource.gate_counts:
            op_node_idx = self._recursively_add_op_node(op)
            self._graph.add_edge(op_node_idx, d_node_idx, (op_node_idx, d_node_idx))
        return d_node_idx

    def solve(self, lazy=True):
        """Solves the graph using the Dijkstra search algorithm.

        Args:
            lazy (bool): If True, the Dijkstra search will stop once optimal decompositions are
                found for all operations that the graph was initialized with. Otherwise, the
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

    def is_solved_for(self, op):
        """Tests whether the decomposition graph is solved for a given operator."""
        op_node = resource_rep(type(op), **op.resource_params)
        return (
            op_node in self._op_node_indices
            and self._op_node_indices[op_node] in self._visitor.distances
        )

    def resource_estimate(self, op) -> Resources:
        """Returns the resource estimate for a given operator.

        Args:
            op (Operator): The operator for which to return the resource estimates.

        Returns:
            Resources: The resource estimate.

        **Example**

        The resource estimate is a gate count in terms of the target gate set, not the immediate
        set of gates that the operator decomposes to.

        .. code-block:: python

            op = qml.CRX(0.5, wires=[0, 1])
            graph = DecompositionGraph(
                operations=[op],
                target_gate_set={"RZ", "RX", "CNOT", "GlobalPhase"},
            )
            graph.solve()

        >>> with qml.queuing.AnnotatedQueue() as q:
        ...     graph.decomposition(op)(0.5, wires=[0, 1])
        >>> q.queue
        [RZ(1.5707963267948966, wires=[1]),
         RY(0.25, wires=[1]),
         CNOT(wires=[0, 1]),
         RY(-0.25, wires=[1]),
         CNOT(wires=[0, 1]),
         RZ(-1.5707963267948966, wires=[1])]
        >>> graph.resource_estimate(op)
        <num_gates=10, gate_counts={RZ: 6, CNOT: 2, RX: 2}>

        """
        if not self.is_solved_for(op):
            raise DecompositionError(f"Operator {op} is unsolved in this decomposition graph.")

        op_node = resource_rep(type(op), **op.resource_params)
        op_node_idx = self._op_node_indices[op_node]
        return self._visitor.distances[op_node_idx]

    def decomposition(self, op: Operator) -> DecompositionRule:
        """Returns the optimal decomposition rule for a given operator.

        Args:
            op (Operator): The operator for which to return the optimal decomposition.

        Returns:
            DecompositionRule: The optimal decomposition.

        **Example**

        The decomposition rule is a quantum function that takes ``(*op.parameters, wires=op.wires, **op.hyperparameters)``
        as arguments.

        .. code-block:: python

            op = qml.CRX(0.5, wires=[0, 1])
            graph = DecompositionGraph(
                operations=[op],
                target_gate_set={"RZ", "RX", "CNOT", "GlobalPhase"},
            )
            graph.solve()
            rule = graph.decomposition(op)

        >>> with qml.queuing.AnnotatedQueue() as q:
        ...     rule(*op.parameters, wires=op.wires, **op.hyperparameters)
        >>> q.queue
        [RZ(1.5707963267948966, wires=[1]),
         RY(0.25, wires=[1]),
         CNOT(wires=[0, 1]),
         RY(-0.25, wires=[1]),
         CNOT(wires=[0, 1]),
         RZ(-1.5707963267948966, wires=[1])]

        """
        if not self.is_solved_for(op):
            raise DecompositionError(f"Operator {op} is unsolved in this decomposition graph.")

        op_node = resource_rep(type(op), **op.resource_params)
        op_node_idx = self._op_node_indices[op_node]
        d_node_idx = self._visitor.predecessors[op_node_idx]
        return self._graph[d_node_idx].rule


class _DecompositionSearchVisitor(DijkstraVisitor):
    """The visitor used in the Dijkstra search for the optimal decomposition."""

    def __init__(self, graph: rx.PyDiGraph, original_op_indices: set[int], lazy: bool = True):
        self._graph = graph
        self._lazy = lazy
        # maps node indices to the optimal resource estimates
        self.distances: dict[int, Resources] = {}
        # maps operator nodes to the optimal decomposition nodes
        self.predecessors: dict[int, int] = {}
        self.unsolved_op_indices = original_op_indices.copy()
        self._num_edges_examined: dict[int, int] = {}  # keys are decomposition node indices

    def edge_weight(self, edge_obj):
        """Calculates the weight of an edge."""
        if not isinstance(edge_obj, tuple):
            return float(edge_obj)
        op_node_idx, d_node_idx = edge_obj
        return self.distances[d_node_idx].num_gates - self.distances[op_node_idx].num_gates

    def discover_vertex(self, v, _):
        """Triggered when a vertex is about to be explored during the Dijkstra search."""
        self.unsolved_op_indices.discard(v)
        if not self.unsolved_op_indices and self._lazy:
            raise StopSearch

    def examine_edge(self, edge):
        """Triggered when an edge is examined during the Dijkstra search."""
        src_idx, target_idx, _ = edge
        src_node = self._graph[src_idx]
        target_node = self._graph[target_idx]
        if not isinstance(target_node, _DecompositionNode):
            return  # nothing is to be done for edges leading to an operator node
        if target_idx not in self.distances:
            self.distances[target_idx] = Resources()  # initialize with empty resource
        self.distances[target_idx] += self.distances[src_idx] * target_node.count(src_node)
        if target_idx not in self._num_edges_examined:
            self._num_edges_examined[target_idx] = 0
        self._num_edges_examined[target_idx] += 1
        if self._num_edges_examined[target_idx] < len(target_node.decomp_resource.gate_counts):
            # Typically in Dijkstra's search, a vertex is discovered from any of its incoming
            # edges. However, for a decomposition node, it requires all incoming edges to be
            # examined before it can be discovered (each incoming edge represents a different
            # operator that this decomposition depends on).
            raise PruneSearch

    def edge_relaxed(self, edge):
        """Triggered when an edge is relaxed during the Dijkstra search."""
        src_idx, target_idx, _ = edge
        target_node = self._graph[target_idx]
        if self._graph[src_idx] == "dummy":
            self.distances[target_idx] = Resources({target_node: 1})
        elif isinstance(target_node, CompressedResourceOp):
            self.predecessors[target_idx] = src_idx
            self.distances[target_idx] = self.distances[src_idx]


@dataclass(frozen=True)
class _DecompositionNode:
    """A node that represents a decomposition rule."""

    rule: DecompositionRule
    decomp_resource: Resources

    def count(self, op: CompressedResourceOp):
        """Find the number of occurrences of an operator in the decomposition."""
        return self.decomp_resource.gate_counts.get(op, 0)
