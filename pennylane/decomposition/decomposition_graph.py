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

import warnings
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, replace

import rustworkx as rx
from rustworkx.visit import DijkstraVisitor, PruneSearch, StopSearch

import pennylane as qml
from pennylane.exceptions import DecompositionError
from pennylane.operation import Operator

from .decomposition_rule import DecompositionRule, WorkWireSpec, list_decomps, null_decomp
from .resources import CompressedResourceOp, Resources, resource_rep
from .symbolic_decomposition import (
    adjoint_rotation,
    cancel_adjoint,
    ctrl_single_work_wire,
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
    to_controlled_qubit_unitary,
)
from .utils import translate_op_alias


@dataclass(frozen=True)
class _OperatorNode:
    """A node that represents an operator type."""

    op: CompressedResourceOp
    """The resource rep of the operator."""

    num_work_wire_not_available: int
    """The number of work wires NOT available to the decomposition of this operator

    The choice of decomposition rule for an operator depends on how many work wires is available,
    which can be can be calculated by subtracting this value from the total number of work wires.
    This convention allows the graph to be constructed without specifying the total number of work
    wires, so that the same graph can be solved multiple times with different values for the max
    number of work wires.

    """

    work_wire_dependent: bool = False
    """Whether the decomposition of this operator depend on work wires.

    This is true if any of the decomposition rules for this operator directly uses work wires or
    if any of the operators produced down the line has decomposition rules that use work wires.

    """

    def __hash__(self) -> int:
        # If the decomposition of an operator does not depend on the availability of work wires
        # at all, we don't need to have multiple nodes representing the same operator with
        # different work wire budgets. Therefore, we override the __hash__ and __eq__ of a node
        # so that the num_work_wires_not_available is taken into account only when the operator
        # depends on work wires. Since we keep track of all existing operator nodes in a set,
        # this allows us to quickly find an existing operator node for another instance of the
        # same operator with a different work wire budget that doesn't ultimately matter.
        if self.work_wire_dependent:
            return hash((self.op, self.num_work_wire_not_available))
        return hash(self.op)

    def __eq__(self, other) -> bool:  # pragma: no cover
        if not isinstance(other, _OperatorNode):
            return False
        if self.work_wire_dependent:
            return (
                self.op == other.op
                and self.num_work_wire_not_available == other.num_work_wire_not_available
            )
        return self.op == other.op


@dataclass
class _DecompositionNode:
    """A node that represents a decomposition rule."""

    rule: DecompositionRule
    decomp_resource: Resources
    work_wire_spec: WorkWireSpec
    num_work_wire_not_available: int
    work_wire_dependent: bool = False

    def count(self, op: CompressedResourceOp):
        """Find the number of occurrences of an operator in the decomposition."""
        return self.decomp_resource.gate_counts.get(op, 0)

    def is_feasible(self, num_work_wires: int | None):
        """Checks whether this decomposition is feasible under a work wire constraint"""
        if num_work_wires is None:
            return True
        return num_work_wires - self.num_work_wire_not_available >= self.work_wire_spec.total


class DecompositionGraph:  # pylint: disable=too-many-instance-attributes,too-few-public-methods
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

    For example, if the graph was initialized with ``{qml.CNOT: 10.0, qml.H: 1.0}`` as the gate set,
    the edge that connects a ``CNOT`` to the following decomposition rule:

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
        gate_set (set[str | type] | dict[type | str, float]): A set of gates in the target gate set or a dictionary
            mapping gates in the target gate set to their respective weights. All weights must be positive.
        fixed_decomps (dict): A dictionary mapping operator names to fixed decompositions.
        alt_decomps (dict): A dictionary mapping operator names to alternative decompositions.

    **Example**

    .. code-block:: python

        from pennylane.decomposition import DecompositionGraph

        op = qml.CRX(0.5, wires=[0, 1])
        graph = DecompositionGraph(
            operations=[op],
            gate_set={"RZ", "RX", "CNOT", "GlobalPhase"},
        )
        solution = graph.solve()

    >>> with qml.queuing.AnnotatedQueue() as q:
    ...     solution.decomposition(op)(0.5, wires=[0, 1])
    >>> q.queue
    [RZ(1.5707963267948966, wires=[1]),
     RY(0.25, wires=[1]),
     CNOT(wires=[0, 1]),
     RY(-0.25, wires=[1]),
     CNOT(wires=[0, 1]),
     RZ(-1.5707963267948966, wires=[1])]
    >>> solution.resource_estimate(op)
    <num_gates=10, gate_counts={RZ: 6, CNOT: 2, RX: 2}, weighted_cost=10.0>

    """

    def __init__(
        self,
        operations: list[Operator | CompressedResourceOp],
        gate_set: set[type | str] | dict[type | str, float],
        fixed_decomps: dict | None = None,
        alt_decomps: dict | None = None,
    ):
        self._gate_set_weights: dict[str, float]
        if isinstance(gate_set, dict):
            # the gate_set is a dict
            self._gate_set_weights = {_to_name(gate): weight for gate, weight in gate_set.items()}
        else:
            # The names of the gates in the target gate set.
            self._gate_set_weights = {_to_name(gate): 1.0 for gate in gate_set}

        # The list of operator indices for every op in the original list of operators that the
        # graph is initialized with. This is used to check whether we have found a decomposition
        # pathway for every operator we care about, so that we can stop the graph traversal
        # early when solve() is called with lazy=True.
        self._original_ops_indices: set[int] = set()

        # Maps operator nodes to their indices in the graph.
        self._all_op_indices: dict[_OperatorNode, int] = {}

        # Keeps track of all operators that depend on work wires.
        self._work_wire_dependent_ops: set[CompressedResourceOp] = set()

        # Maps operators to operator nodes. There might be multiple operator nodes mapped to
        # the same operator, but each with a different work wire budget.
        self._op_to_op_nodes: dict[CompressedResourceOp, set[_OperatorNode]] = defaultdict(set)

        # Stores the library of custom decomposition rules
        fixed_decomps = fixed_decomps or {}
        alt_decomps = alt_decomps or {}
        self._fixed_decomps = {_to_name(k): v for k, v in fixed_decomps.items()}
        self._alt_decomps = {_to_name(k): v for k, v in alt_decomps.items()}

        # Initializes the graph.
        self._graph = rx.PyDiGraph()

        # Construct the decomposition graph
        self._start = self._graph.add_node(None)
        self._construct_graph(operations)

    def _construct_graph(self, operations: Iterable[Operator | CompressedResourceOp]):
        """Constructs the decomposition graph."""
        for op in operations:
            if isinstance(op, Operator):
                op = resource_rep(type(op), **op.resource_params)
            idx = self._add_op_node(op, 0)
            self._original_ops_indices.add(idx)

    def _add_op_node(self, op: CompressedResourceOp, num_used_work_wires: int) -> int:
        """Recursively adds an operation node to the graph.

        An operator node is uniquely defined by its operator type and resource parameters, which
        are conveniently wrapped in a ``CompressedResourceOp``.

        Args:
            op: The operator to add to the graph.
            num_used_work_wires: The number of wires taken from the overall budget.

        """

        # If an operator has already been added to the graph, we return the existing node
        # instead of creating a new one. Now when we see an operator with a different work
        # wire budget from the one already in the graph, whether we need to create a new
        # node for this operator is determined by whether this operator's decomposition is
        # work-wire dependent. We have overriden __hash__ and __eq__ of the node class so
        # that when we have a work-wire-independent operator with a different work wire
        # budget from the existing one in the graph, the difference is ignored.
        known_work_wire_dependent = op in self._work_wire_dependent_ops
        op_node = _OperatorNode(op, num_used_work_wires, known_work_wire_dependent)

        if op_node in self._all_op_indices:
            return self._all_op_indices[op_node]

        op_node_idx = self._graph.add_node(op_node)
        self._all_op_indices[op_node] = op_node_idx
        self._op_to_op_nodes[op].add(op_node)

        if op.name in self._gate_set_weights:
            self._graph.add_edge(self._start, op_node_idx, self._gate_set_weights[op.name])
            return op_node_idx

        update_op_to_work_wire_dependent = False
        for decomposition in self._get_decompositions(op):
            d_node = self._add_decomp(decomposition, op_node, op_node_idx, num_used_work_wires)
            # If any of the operator's decompositions depend on work wires, this operator
            # should also depend on work wires.
            if d_node and d_node.work_wire_dependent and not known_work_wire_dependent:
                update_op_to_work_wire_dependent = True

        # If we found that this operator depends on work wires, but it's currently recorded
        # as independent of work wires, we must replace every record of this operator node
        # with a new node with `work_wire_dependent` set to `True`.
        if update_op_to_work_wire_dependent:
            new_op_node = replace(op_node, work_wire_dependent=True)
            self._all_op_indices[new_op_node] = self._all_op_indices.pop(op_node)
            self._graph[op_node_idx] = new_op_node
            self._op_to_op_nodes[op].remove(op_node)
            self._op_to_op_nodes[op].add(new_op_node)
            # Also record that this operator type depends on work wires, so in the future
            # when we encounter other instances of the same operator type, we correctly
            # identify it as work-wire dependent.
            self._work_wire_dependent_ops.add(op_node.op)

        return op_node_idx

    def _add_decomp(
        self,
        rule: DecompositionRule,
        op_node: _OperatorNode,
        op_idx: int,
        num_used_work_wires: int,
    ) -> _DecompositionNode | None:
        """Adds a decomposition rule to the graph and returns whether it depends on work wires."""

        if not rule.is_applicable(**op_node.op.params):
            return None  # skip the decomposition rule if it is not applicable

        decomp_resource = rule.compute_resources(**op_node.op.params)
        work_wire_spec = rule.get_work_wire_spec(**op_node.op.params)

        d_node = _DecompositionNode(rule, decomp_resource, work_wire_spec, num_used_work_wires)
        d_node_idx = self._graph.add_node(d_node)
        if not decomp_resource.gate_counts:
            # If an operator decomposes to nothing (e.g., a Hadamard raised to a
            # power of 2), we must still connect something to this decomposition
            # node so that it is accounted for.
            self._graph.add_edge(self._start, d_node_idx, 0)

        if work_wire_spec.total:
            d_node.work_wire_dependent = True

        for op in decomp_resource.gate_counts:
            op_node_idx = self._add_op_node(op, num_used_work_wires + work_wire_spec.total)
            self._graph.add_edge(op_node_idx, d_node_idx, (op_node_idx, d_node_idx))
            # If any of the operators in the decomposition depends on work wires, this
            # decomposition is also dependent on work wires, even it itself does not use
            # any work wires.
            if self._graph[op_node_idx].work_wire_dependent:
                d_node.work_wire_dependent = True

        self._graph.add_edge(d_node_idx, op_idx, 0)
        return d_node

    def _get_decompositions(self, op: CompressedResourceOp) -> list[DecompositionRule]:
        """Helper function to get a list of decomposition rules."""

        op_name = _to_name(op)

        if op_name in self._fixed_decomps:
            return [self._fixed_decomps[op_name]]

        decomps = self._alt_decomps.get(op_name, []) + list_decomps(op_name)

        if (
            issubclass(op.op_type, qml.ops.Adjoint)
            and self_adjoint not in decomps
            and adjoint_rotation not in decomps
        ):
            # In general, we decompose the adjoint of an operator by applying adjoint to the
            # decompositions of the operator. However, this is not necessary if the operator
            # is self-adjoint or if it has a single rotation angle which can be trivially
            # inverted to obtain its adjoint. In this case, `self_adjoint` or `adjoint_rotation`
            # would've already been retrieved as a potential decomposition rule for this
            # operator, so there is no need to consider the general case.
            decomps.extend(self._get_adjoint_decompositions(op))

        elif (
            issubclass(op.op_type, qml.ops.Pow)
            and pow_rotation not in decomps
            and pow_involutory not in decomps
        ):
            # Similar to the adjoint case, the `_get_pow_decompositions` contains the general
            # approach we take to decompose powers of operators. However, if the operator is
            # involutory or if it has a single rotation angle that can be trivially multiplied
            # with the power, we would've already retrieved `pow_involutory` or `pow_rotation`
            # as a potential decomposition rule for this operator, so there is no need to consider
            # the general case.
            decomps.extend(self._get_pow_decompositions(op))

        elif op.op_type in (qml.ops.Controlled, qml.ops.ControlledOp):
            decomps.extend(self._get_controlled_decompositions(op))

        return decomps

    def _get_adjoint_decompositions(self, op: CompressedResourceOp) -> list[DecompositionRule]:
        """Gets the decomposition rules for the adjoint of an operator."""

        base_class, base_params = (op.params["base_class"], op.params["base_params"])

        # Special case: adjoint of an adjoint cancels out
        if issubclass(base_class, qml.ops.Adjoint):
            return [cancel_adjoint]

        # General case: apply adjoint to each of the base op's decomposition rules.
        base = resource_rep(base_class, **base_params)
        return [make_adjoint_decomp(base_decomp) for base_decomp in self._get_decompositions(base)]

    @staticmethod
    def _get_pow_decompositions(op: CompressedResourceOp) -> list[DecompositionRule]:
        """Gets the decomposition rules for the power of an operator."""

        base_class = op.params["base_class"]

        # Special case: power of zero
        if op.params["z"] == 0:
            return [null_decomp]

        if op.params["z"] == 1:
            return [decompose_to_base]

        # Special case: power of a power
        if issubclass(base_class, qml.ops.Pow):
            return [merge_powers]

        # Special case: power of an adjoint
        if issubclass(base_class, qml.ops.Adjoint):
            return [flip_pow_adjoint]

        # General case: repeat the operator z times
        return [repeat_pow_base]

    def _get_controlled_decompositions(self, op: CompressedResourceOp) -> list[DecompositionRule]:
        """Adds a controlled decomposition node to the graph."""

        base_class, base_params = op.params["base_class"], op.params["base_params"]

        # Special case: control of an adjoint
        if issubclass(base_class, qml.ops.Adjoint):
            return [flip_control_adjoint]

        # Special case: when the base is GlobalPhase, none of the following automatically
        # generated decomposition rules apply.
        if base_class is qml.GlobalPhase:
            return []

        # General case: apply control to the base op's decomposition rules.
        base = resource_rep(base_class, **base_params)
        rules = [make_controlled_decomp(decomp) for decomp in self._get_decompositions(base)]

        # There's always the option of turning the controlled operator into a controlled
        # qubit unitary if the base operator has a matrix form.
        rules.append(to_controlled_qubit_unitary)

        # There's always Lemma 7.11 from https://arxiv.org/abs/quant-ph/9503016.
        rules.append(ctrl_single_work_wire)

        return rules

    def solve(self, num_work_wires: int | None = 0, lazy=True) -> DecompGraphSolution:
        """Solves the graph using the Dijkstra search algorithm.

        Args:
            num_work_wires (int, optional): The total number of available work wires. Set this
                to ``None`` if there is an unlimited number of work wires.
            lazy (bool): If True, the Dijkstra search will stop once optimal decompositions are
                found for all operations that the graph was initialized with. Otherwise, the
                entire graph will be explored.

        Returns:
            DecompGraphSolution

        """
        visitor = DecompositionSearchVisitor(
            self._graph,
            self._gate_set_weights,
            self._original_ops_indices,
            num_work_wires,
            lazy,
        )
        rx.dijkstra_search(
            self._graph,
            source=[self._start],
            weight_fn=visitor.edge_weight,
            visitor=visitor,
        )
        if visitor.unsolved_op_indices:
            unsolved_ops = [self._graph[op_idx] for op_idx in visitor.unsolved_op_indices]
            op_names = {op_node.op.name for op_node in unsolved_ops}
            warnings.warn(
                f"The graph-based decomposition system is unable to find a decomposition for "
                f"{op_names} to the target gate set {set(self._gate_set_weights)}.",
                UserWarning,
            )
        return DecompGraphSolution(visitor, self._all_op_indices, self._op_to_op_nodes)


class DecompGraphSolution:
    """A solution to a decomposition graph.

    An instance of this class is returned from :meth:`DecompositionGraph.solve`

    **Example**

    .. code-block:: python

        from pennylane.decomposition import DecompositionGraph

        op = qml.CRX(0.5, wires=[0, 1])
        graph = DecompositionGraph(
            operations=[op],
            gate_set={"RZ", "RX", "CNOT", "GlobalPhase"},
        )
        solution = graph.solve()

    >>> with qml.queuing.AnnotatedQueue() as q:
    ...     solution.decomposition(op)(0.5, wires=[0, 1])
    >>> q.queue
    [RZ(1.5707963267948966, wires=[1]),
     RY(0.25, wires=[1]),
     CNOT(wires=[0, 1]),
     RY(-0.25, wires=[1]),
     CNOT(wires=[0, 1]),
     RZ(-1.5707963267948966, wires=[1])]
    >>> solution.resource_estimate(op)
    <num_gates=10, gate_counts={RZ: 6, CNOT: 2, RX: 2}, weighted_cost=10.0>

    """

    def __init__(
        self,
        visitor: DecompositionSearchVisitor,
        all_op_indices: dict[_OperatorNode, int],
        op_to_op_nodes: dict[CompressedResourceOp, set[_OperatorNode]],
    ) -> None:
        self._visitor = visitor
        self._graph = visitor._graph
        self._op_to_op_nodes = op_to_op_nodes
        self._all_op_indices = all_op_indices

    def _all_solutions(
        self, visitor: DecompositionSearchVisitor, op: Operator, num_work_wires: int | None
    ) -> Iterable[_OperatorNode]:
        """Returns all valid solutions for an operator and a work wire constraint."""

        op_rep = resource_rep(type(op), **op.resource_params)
        if op_rep not in self._op_to_op_nodes:
            return []

        def _is_solved(op_node: _OperatorNode):
            return (
                op_node in self._all_op_indices
                and self._all_op_indices[op_node] in visitor.distances
            )

        def _is_feasible(op_node: _OperatorNode):
            if visitor.num_available_work_wires is None or num_work_wires is None:
                return True
            op_node_idx = self._all_op_indices[op_node]
            return num_work_wires >= visitor.num_work_wires_used[op_node_idx]

        return filter(_is_feasible, filter(_is_solved, self._op_to_op_nodes[op_rep]))

    def is_solved_for(self, op: Operator, num_work_wires: int | None = 0):
        """Tests whether the decomposition graph is solved for a given operator.

        Args:
            op (Operator): The operator to check.
            num_work_wires (int): The number of available work wires to decompose this operator.

        """
        return any(self._all_solutions(self._visitor, op, num_work_wires))

    def _get_best_solution(
        self, visitor: DecompositionSearchVisitor, op: Operator, num_work_wires: int | None
    ) -> int:
        """Finds the best solution for an operator in terms of resource efficiency."""

        def _resource(node: _OperatorNode):
            op_node_idx = self._all_op_indices[node]
            return (
                visitor.distances[op_node_idx].weighted_cost,
                visitor.num_work_wires_used[op_node_idx],
            )

        all_solutions = self._all_solutions(visitor, op, num_work_wires)
        solution = min(all_solutions, key=_resource, default=None)

        if not solution:
            raise DecompositionError(f"Operator {op} is unsolved in this decomposition graph.")

        op_node_idx = self._all_op_indices[solution]
        return op_node_idx

    def resource_estimate(self, op: Operator, num_work_wires: int | None = 0) -> Resources:
        """Returns the resource estimate for a given operator.

        Args:
            op (Operator): The operator for which to return the resource estimates.
            num_work_wires (int): The number of work wires available to decompose this operator.

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
            solution = graph.solve()

        >>> with qml.queuing.AnnotatedQueue() as q:
        ...     solution.decomposition(op)(0.5, wires=[0, 1])
        >>> q.queue
        [RZ(1.5707963267948966, wires=[1]),
         RY(0.25, wires=[1]),
         CNOT(wires=[0, 1]),
         RY(-0.25, wires=[1]),
         CNOT(wires=[0, 1]),
         RZ(-1.5707963267948966, wires=[1])]
        >>> graph.resource_estimate(op)
        <num_gates=10, gate_counts={RZ: 6, CNOT: 2, RX: 2}, weighted_cost=10.0>

        """
        op_node_idx = self._get_best_solution(self._visitor, op, num_work_wires)
        return self._visitor.distances[op_node_idx]

    def decomposition(self, op: Operator, num_work_wires: int | None = 0) -> DecompositionRule:
        """Returns the optimal decomposition rule for a given operator.

        Args:
            op (Operator): The operator for which to return the optimal decomposition.
            num_work_wires (int): The number of work wires available to decompose this operator.

        Returns:
            DecompositionRule: The optimal decomposition.

        **Example**

        The decomposition rule is a quantum function that takes ``(*op.parameters, wires=op.wires, **op.hyperparameters)``
        as arguments.

        .. code-block:: python

            op = qml.CRY(0.2, wires=[0, 2])
            graph = DecompositionGraph(
                operations=[op],
                gate_set={"RZ", "RX", "CNOT", "GlobalPhase"},
            )
            solution = graph.solve()
            rule = solution.decomposition(op)

        >>> with qml.queuing.AnnotatedQueue() as q:
        ...     rule(*op.parameters, wires=op.wires, **op.hyperparameters)
        >>> q.queue
        [RY(0.1, wires=[2]),
         CNOT(wires=[0, 2]),
         RY(-0.1, wires=[2]),
         CNOT(wires=[0, 2])]

        """
        op_node_idx = self._get_best_solution(self._visitor, op, num_work_wires)
        d_node_idx = self._visitor.predecessors[op_node_idx]
        return self._graph[d_node_idx].rule


class DecompositionSearchVisitor(DijkstraVisitor):  # pylint: disable=too-many-instance-attributes
    """The visitor used in the Dijkstra search for the optimal decomposition."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        graph: rx.PyDiGraph,
        gate_set: dict,
        original_op_indices: set[int],
        num_available_work_wires: int | None = None,
        lazy: bool = True,
    ):
        self._graph = graph
        self._lazy = lazy
        # maps node indices to the optimal resource estimates
        self.distances: dict[int, Resources] = {}
        # maps operator nodes to the optimal decomposition nodes
        self.predecessors: dict[int, int] = {}
        self.unsolved_op_indices = original_op_indices.copy()
        # keys are decomposition node indices
        self._n_edges_examined: dict[int, int] = defaultdict(int)
        self._gate_weights = gate_set
        # work wire related attributes
        self.num_available_work_wires = num_available_work_wires
        # the minimum number of work wires consumed along the path that
        # reaches each node in the graph.
        self.num_work_wires_used: dict[int, int] = defaultdict(int)

    def edge_weight(self, edge_obj):
        """Calculates the weight of an edge."""
        if not isinstance(edge_obj, tuple):
            return float(edge_obj)
        op_node_idx, d_node_idx = edge_obj
        return self.distances[d_node_idx].weighted_cost - self.distances[op_node_idx].weighted_cost

    def discover_vertex(self, v, score):  # pylint: disable=unused-argument
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

        # Check if this decomposition is feasible under the work wire constraint
        if not target_node.is_feasible(self.num_available_work_wires):
            raise PruneSearch

        self.distances[target_idx] += self.distances[src_idx] * target_node.count(src_node.op)
        self._n_edges_examined[target_idx] += 1

        # Update the number of work wires required for this decomposition to be valid
        # with the maximum of the number of work wires required for each of its operators.
        self.num_work_wires_used[target_idx] = max(
            self.num_work_wires_used[target_idx], self.num_work_wires_used[src_idx]
        )

        if self._n_edges_examined[target_idx] < len(target_node.decomp_resource.gate_counts):
            # Typically in Dijkstra's search, a vertex is discovered from any of its incoming
            # edges. However, for a decomposition node, it requires all incoming edges to be
            # examined before it can be discovered (each incoming edge represents a different
            # operator that this decomposition depends on).
            raise PruneSearch

    def edge_relaxed(self, edge):
        """Triggered when an edge is relaxed during the Dijkstra search."""
        src_idx, target_idx, _ = edge
        target_node = self._graph[target_idx]
        if self._graph[src_idx] is None and isinstance(target_node, _OperatorNode):
            # This branch applies to operators in the target gate set.
            weight = self._gate_weights[_to_name(target_node.op)]
            self.distances[target_idx] = Resources({target_node.op: 1}, weight)
            self.num_work_wires_used[target_idx] = 0
        elif isinstance(target_node, _DecompositionNode):
            self.num_work_wires_used[target_idx] += target_node.work_wire_spec.total
        elif isinstance(target_node, _OperatorNode):
            self.predecessors[target_idx] = src_idx
            self.distances[target_idx] = self.distances[src_idx]
            self.num_work_wires_used[target_idx] = self.num_work_wires_used[src_idx]


def _to_name(op):
    if isinstance(op, type):
        return op.__name__
    if isinstance(op, CompressedResourceOp):
        return op.name
    assert isinstance(op, str)
    return translate_op_alias(op)
