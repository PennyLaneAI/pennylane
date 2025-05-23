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

from .decomposition_rule import DecompositionRule, list_decomps, null_decomp
from .resources import CompressedResourceOp, Resources, resource_rep
from .symbolic_decomposition import (
    adjoint_rotation,
    cancel_adjoint,
    decompose_to_base,
    flip_control_adjoint,
    flip_pow_adjoint,
    make_adjoint_decomp,
    make_controlled_decomp,
    merge_powers,
    pow_involutory,
    pow_rotation,
    repeat_pow_base,
    self_adjoint,
)
from .utils import DecompositionError, translate_op_alias


class DecompositionGraph:  # pylint: disable=too-many-instance-attributes
    """A graph that models a decomposition problem.

    The decomposition graph contains two types of nodes: operator nodes and decomposition nodes.
    Each decomposition node is a :class:`pennylane.decomposition.DecompositionRule`, and each
    operator node is a :class:`~pennylane.decomposition.resources.CompressedResourceOp` which
    contains an operator type and any additional parameters that affect the resource requirements
    of the operator. Essentially, two instances of the same operator type are represented by the
    same node in the graph if they're expected to have the same decompositions.

    There are also two types of directed edges: edges that connect operators to the decomposition
    rules that contain them, and edges that connect decomposition rules to the operators that they
    decompose. The edge weights represent the difference in the total weights of the gates between
    the two states. Edges that connect decomposition rules to operators have a weight of 0 because
    an operator can be replaced with its decomposition at no additional cost.

    On the other hand, edges that connect operators to the decomposition rule that contains them
    will have a weight that is the total resource estimate of the decomposition minus the resource
    estimate of the operator. Edges that connect an operator node to a decomposition node have a weight
    calculated by the difference of the sum of the gate counts multiplied by their respective gate
    weights in the decomposition, minus the weight of the operator of the operator node.

    For example, if the graph was initialized with ``{qml.CNOT: 10.0, qml.H: 1.0}`` as the gate set, the edge that connects a ``CNOT`` to the following
    decomposition rule:

    .. code-block:: python

        import pennylane as qml

        @qml.register_resources({qml.H: 2, qml.CNOT: 1})
        def my_cz(wires):
            qml.H(wires=wires[1])
            qml.CNOT(wires=wires)
            qml.H(wires=wires[1])

    will have a weight of (10.0 + 2 * 1.0) - 10.0 = 2, because the decomposition rule contains 2 additional
    ``H`` gates. Note that this gate count is in terms of gates in the target gate set. If ``H`` isn't
    supported and is in turn decomposed to two ``RZ`` gates and one ``RX`` gate, the weight of this edge
    becomes 2 * 3 = 6, if ``RZ`` and ``RX`` have weights of 1.0 (the default). This way, the total distance
    from the basis gate set to a high-level gate is by default the total number of basis gates required to
    decompose this high-level gate, which allows us to use Dijkstra's algorithm to find the most efficient
    decomposition. By specifying weights in the target gate set, the total distance calculation involves
    a sum of weighted gate counts, which can represent the relative cost of executing a particular element
    of the target gate set on the target hardware i.e. a ``T`` gate.

    Args:
        operations (list[Operator or CompressedResourceOp]): The list of operations to decompose.
        gate_set (set[str | type] | dict[type | str, float]): A set of gates in the target gate set or a dictionary mapping gates in the target gate set to their respective weights. All weights must be positive.
        fixed_decomps (dict): A dictionary mapping operator names to fixed decompositions.
        alt_decomps (dict): A dictionary mapping operator names to alternative decompositions.

    **Example**

    .. code-block:: python

        op = qml.CRX(0.5, wires=[0, 1])
        graph = DecompositionGraph(
            operations=[op],
            gate_set={"RZ", "RX", "CNOT", "GlobalPhase"},
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
        gate_set: set[type | str] | dict[type | str, float],
        fixed_decomps: dict = None,
        alt_decomps: dict = None,
    ):
        if isinstance(gate_set, set):
            # The names of the gates in the target gate set.
            self._weights = {_to_name(gate): 1.0 for gate in gate_set}
        else:
            # the gate_set is a dict
            self._weights = {_to_name(gate): weight for gate, weight in gate_set.items()}

        # Tracks the node indices of various operators.
        self._original_ops_indices: set[int] = set()
        self._all_op_indices: dict[CompressedResourceOp, int] = {}

        # Stores the library of custom decomposition rules
        fixed_decomps = fixed_decomps or {}
        alt_decomps = alt_decomps or {}
        self._fixed_decomps = {_to_name(k): v for k, v in fixed_decomps.items()}
        self._alt_decomps = {_to_name(k): v for k, v in alt_decomps.items()}

        # Initializes the graph.
        self._graph = rx.PyDiGraph()
        self._visitor = None

        # Construct the decomposition graph
        self._start = self._graph.add_node(None)
        self._construct_graph(operations)

    def _get_decompositions(self, op_node: CompressedResourceOp) -> list[DecompositionRule]:
        """Helper function to get a list of decomposition rules."""

        op_name = _to_name(op_node)

        if op_name in self._fixed_decomps:
            return [self._fixed_decomps[op_name]]

        decomps = self._alt_decomps.get(op_name, []) + list_decomps(op_name)

        if (
            issubclass(op_node.op_type, qml.ops.Adjoint)
            and self_adjoint not in decomps
            and adjoint_rotation not in decomps
        ):
            # In general, we decompose the adjoint of an operator by applying adjoint to the
            # decompositions of the operator. However, this is not necessary if the operator
            # is self-adjoint or if it has a single rotation angle which can be trivially
            # inverted to obtain its adjoint. In this case, `self_adjoint` or `adjoint_rotation`
            # would've already been retrieved as a potential decomposition rule for this
            # operator, so there is no need to consider the general case.
            decomps.extend(self._get_adjoint_decompositions(op_node))

        elif (
            issubclass(op_node.op_type, qml.ops.Pow)
            and pow_rotation not in decomps
            and pow_involutory not in decomps
        ):
            # Similar to the adjoint case, the `_get_pow_decompositions` contains the general
            # approach we take to decompose powers of operators. However, if the operator is
            # involutory or if it has a single rotation angle that can be trivially multiplied
            # with the power, we would've already retrieved `pow_involutory` or `pow_rotation`
            # as a potential decomposition rule for this operator, so there is no need to consider
            # the general case.
            decomps.extend(self._get_pow_decompositions(op_node))

        elif op_node.op_type in (qml.ops.Controlled, qml.ops.ControlledOp):
            decomps.extend(self._get_controlled_decompositions(op_node))

        return decomps

    def _construct_graph(self, operations):
        """Constructs the decomposition graph."""
        for op in operations:
            if isinstance(op, Operator):
                op = resource_rep(type(op), **op.resource_params)
            idx = self._add_op_node(op)
            self._original_ops_indices.add(idx)

    def _add_op_node(self, op_node: CompressedResourceOp) -> int:
        """Recursively adds an operation node to the graph.

        An operator node is uniquely defined by its operator type and resource parameters, which
        are conveniently wrapped in a ``CompressedResourceOp``.

        """

        if op_node in self._all_op_indices:
            return self._all_op_indices[op_node]

        op_node_idx = self._graph.add_node(op_node)
        self._all_op_indices[op_node] = op_node_idx

        if op_node.name in self._weights:
            self._graph.add_edge(self._start, op_node_idx, self._weights[op_node.name])
            return op_node_idx

        for decomposition in self._get_decompositions(op_node):
            self._add_decomp(decomposition, op_node, op_node_idx)

        return op_node_idx

    def _add_decomp(self, rule: DecompositionRule, op_node: CompressedResourceOp, op_idx: int):
        """Adds a decomposition rule to the graph."""
        if not rule.is_applicable(**op_node.params):
            return  # skip the decomposition rule if it is not applicable
        decomp_resource = rule.compute_resources(**op_node.params)
        d_node = _DecompositionNode(rule, decomp_resource)
        d_node_idx = self._graph.add_node(d_node)
        if not decomp_resource.gate_counts:
            # If an operator decomposes to nothing (e.g., a Hadamard raised to a
            # power of 2), we must still connect something to this decomposition
            # node so that it is accounted for.
            self._graph.add_edge(self._start, d_node_idx, 0)
        for op in decomp_resource.gate_counts:
            op_node_idx = self._add_op_node(op)
            self._graph.add_edge(op_node_idx, d_node_idx, (op_node_idx, d_node_idx))
        self._graph.add_edge(d_node_idx, op_idx, 0)

    def _get_adjoint_decompositions(self, op_node: CompressedResourceOp) -> list[DecompositionRule]:
        """Gets the decomposition rules for the adjoint of an operator."""

        base_class, base_params = (op_node.params["base_class"], op_node.params["base_params"])

        # Special case: adjoint of an adjoint cancels out
        if issubclass(base_class, qml.ops.Adjoint):
            return [cancel_adjoint]

        # General case: apply adjoint to each of the base op's decomposition rules.
        base = resource_rep(base_class, **base_params)
        return [make_adjoint_decomp(base_decomp) for base_decomp in self._get_decompositions(base)]

    @staticmethod
    def _get_pow_decompositions(op_node: CompressedResourceOp) -> list[DecompositionRule]:
        """Gets the decomposition rules for the power of an operator."""

        base_class = op_node.params["base_class"]

        # Special case: power of zero
        if op_node.params["z"] == 0:
            return [null_decomp]

        if op_node.params["z"] == 1:
            return [decompose_to_base]

        # Special case: power of a power
        if issubclass(base_class, qml.ops.Pow):
            return [merge_powers]

        # Special case: power of an adjoint
        if issubclass(base_class, qml.ops.Adjoint):
            return [flip_pow_adjoint]

        # General case: repeat the operator z times
        return [repeat_pow_base]

    def _get_controlled_decompositions(
        self, op_node: CompressedResourceOp
    ) -> list[DecompositionRule]:
        """Adds a controlled decomposition node to the graph."""

        base_class, base_params = op_node.params["base_class"], op_node.params["base_params"]

        # Special case: control of an adjoint
        if issubclass(base_class, qml.ops.Adjoint):
            return [flip_control_adjoint]

        # General case: apply control to the base op's decomposition rules.
        base = resource_rep(base_class, **base_params)
        return [make_controlled_decomp(decomp) for decomp in self._get_decompositions(base)]

    def solve(self, lazy=True):
        """Solves the graph using the Dijkstra search algorithm.

        Args:
            lazy (bool): If True, the Dijkstra search will stop once optimal decompositions are
                found for all operations that the graph was initialized with. Otherwise, the
                entire graph will be explored.

        """
        self._visitor = _DecompositionSearchVisitor(
            self._graph,
            self._weights,
            self._original_ops_indices,
            lazy,
        )
        rx.dijkstra_search(
            self._graph,
            source=[self._start],
            weight_fn=self._visitor.edge_weight,
            visitor=self._visitor,
        )
        if self._visitor.unsolved_op_indices:
            unsolved_ops = [self._graph[op_idx] for op_idx in self._visitor.unsolved_op_indices]
            op_names = set(op.name for op in unsolved_ops)
            raise DecompositionError(
                f"Decomposition not found for {op_names} to the gate set {set(self._weights)}"
            )

    def is_solved_for(self, op):
        """Tests whether the decomposition graph is solved for a given operator."""
        op_node = resource_rep(type(op), **op.resource_params)
        return (
            op_node in self._all_op_indices
            and self._all_op_indices[op_node] in self._visitor.distances
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
                gate_set={"RZ", "RX", "CNOT", "GlobalPhase"},
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
        op_node_idx = self._all_op_indices[op_node]
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
                gate_set={"RZ", "RX", "CNOT", "GlobalPhase"},
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
        op_node_idx = self._all_op_indices[op_node]
        d_node_idx = self._visitor.predecessors[op_node_idx]
        return self._graph[d_node_idx].rule


class _DecompositionSearchVisitor(DijkstraVisitor):
    """The visitor used in the Dijkstra search for the optimal decomposition."""

    def __init__(
        self,
        graph: rx.PyDiGraph,
        gate_set: dict,
        original_op_indices: set[int],
        lazy: bool = True,
    ):
        self._graph = graph
        self._lazy = lazy
        # maps node indices to the optimal resource estimates
        self.distances: dict[int, Resources] = {}
        # maps operator nodes to the optimal decomposition nodes
        self.predecessors: dict[int, int] = {}
        self.unsolved_op_indices = original_op_indices.copy()
        self._num_edges_examined: dict[int, int] = {}  # keys are decomposition node indices
        self._gate_weights = gate_set

    def edge_weight(self, edge_obj):
        """Calculates the weight of an edge."""
        if not isinstance(edge_obj, tuple):
            return float(edge_obj)

        op_node_idx, d_node_idx = edge_obj
        return self.distances[d_node_idx].weighted_cost - self.distances[op_node_idx].weighted_cost

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
        if src_node is None:
            return  # special case for when the decomposition produces nothing
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
        if self._graph[src_idx] is None and not isinstance(target_node, _DecompositionNode):
            self.distances[target_idx] = Resources(
                {target_node: 1}, self._gate_weights[_to_name(target_node)]
            )
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


def _to_name(op):
    if isinstance(op, type):
        return op.__name__
    if isinstance(op, CompressedResourceOp):
        return op.name
    assert isinstance(op, str)
    return translate_op_alias(op)
