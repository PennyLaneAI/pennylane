# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Defines the decomp_inspector transform."""

from functools import partial

from typing_extensions import override

from pennylane.decomposition import DecompGraphSolution, DecompositionGraph, enabled_graph
from pennylane.decomposition.decomposition_graph import _DecompositionNode, _OperatorNode
from pennylane.decomposition.decomposition_rule import _DecompInfo
from pennylane.decomposition.resources import resource_rep
from pennylane.operation import Operator
from pennylane.tape import QuantumScript, QuantumScriptBatch
from pennylane.transforms.core import transform
from pennylane.typing import PostprocessingFn

from .decompose import _resolve_gate_set


# pylint: disable=protected-access
class _DecompInGraphInfo(_DecompInfo):
    """Information about a decomposition rule in a graph for inspection."""

    def __init__(
        self,
        op: Operator,
        decomp_node_idx: int,
        num_work_wires: int | None,
        solution: DecompGraphSolution,
    ) -> None:

        decomp_node = solution._graph[decomp_node_idx]
        assert isinstance(decomp_node, _DecompositionNode)

        super().__init__(op, decomp_node.rule, num_work_wires)
        self._decomp_node_idx = decomp_node_idx
        self._decomp_node = decomp_node
        self._solution = solution
        self._graph = solution._graph

    def __str__(self) -> str:
        result = super().__str__()
        if not self.is_usable:
            return result
        if not self.is_reachable:
            return result + "\n" + self.missing_ops
        return result + "\n" + self.basis_resources

    @property
    def is_reachable(self) -> bool:
        """Whether this decomposition rule is reachable from the target gate set."""
        return self._decomp_node_idx in self._solution._visitor.distances

    @property
    def basis_resources(self) -> str:
        """The gate count and weighted cost in terms of the target gate set."""
        assert self.is_reachable
        basis_resource = self._solution._visitor.distances[self._decomp_node_idx]
        gate_counts = basis_resource.gate_counts
        weighted_cost = basis_resource.weighted_cost
        return f"Full Expansion Gates: {gate_counts}\nWeighted Cost: {weighted_cost}"

    @property
    def missing_ops(self) -> str:
        """Report on any unsolved ops required for this decomposition rule."""
        all_op_node_indices = self._graph.predecessor_indices(self._decomp_node_idx)
        distances = self._solution._visitor.distances
        unsolved_indices = filter(lambda idx: idx not in distances, all_op_node_indices)
        unsolved_ops = set(map(lambda idx: self._graph[idx].op, unsolved_indices))
        return f"Missing Ops: {unsolved_ops}" if unsolved_ops else ""

    @override
    def _get_gate_count_str(self, estimated_count, actual_count) -> str:
        estimated_count = {k: v for k, v in estimated_count.items() if v > 0}
        if estimated_count == actual_count:
            return f"First Expansion Gates: {estimated_count}"
        return (
            f"Estimated First Expansion Gates: {estimated_count}\n"
            f"Actual First Expansion Gates: {actual_count}"
        )


# pylint: disable=protected-access,too-few-public-methods
class DecompGraphInspector:
    """Interactive object that queries a solved decomposition graph.

    .. seealso::

        See the documentation for the :func:`~pennylane.transforms.decomp_inspector`
        transform for how this object is used.

    """

    def __init__(self, decomp_graph: DecompositionGraph, solution: DecompGraphSolution) -> None:
        self._decomp_graph = decomp_graph
        self._raw_graph = decomp_graph._graph
        self._solution = solution
        self._visitor = solution._visitor
        self._max_work_wires = self._solution.num_work_wires

    def inspect_decomps(self, op: Operator, num_work_wires: int | None = 0) -> str:
        """Display all decomposition rules considered for a given operator.

        Args:
            op (Operator): the operator instance to inspect the decomposition rules for.
            num_work_wires (int, optional): the number of work wires available for dynamic allocation.

        Returns:
            str: a printable string with information about the rules considered by the graph.

        """

        op_node = self._find_op_node(op, num_work_wires)

        if isinstance(op_node, str):
            return op_node

        op_node_idx = self._decomp_graph._all_op_indices[op_node]
        decomp_indices = self._raw_graph.predecessor_indices(op_node_idx)

        decomp_strings = []
        for i, d_node_idx in enumerate(decomp_indices):
            rule_info = _DecompInGraphInfo(op, d_node_idx, num_work_wires, self._solution)
            decomp_str = f"Decomposition {i} (name: {rule_info.name})\n{rule_info}"
            if self._visitor.predecessors.get(op_node_idx, -1) == d_node_idx:
                decomp_str = "CHOSEN: " + decomp_str
            decomp_strings.append(decomp_str)
        return "\n\n".join(decomp_strings)

    def _find_op_node(self, op: Operator, num_work_wires: int | None = 0) -> _OperatorNode | str:

        if isinstance(op, type) and issubclass(op, Operator):
            raise TypeError(
                "The inspect_decomps function takes a concrete operator instance as its "
                "first argument, not an operator type."
            )

        op_rep = resource_rep(op.__class__, **op.resource_params)
        if op_rep not in self._decomp_graph._op_to_op_nodes:
            return (
                "This operator is not found in the decomposition graph! This typically "
                "means that this operator was not part of the original circuit, nor is it "
                "produced by any of the operators' decomposition rules."
            )

        op_nodes = self._decomp_graph._op_to_op_nodes[op_rep]

        if self._max_work_wires is None or not next(iter(op_nodes)).work_wire_dependent:
            return min(op_nodes, key=lambda node: node.num_work_wire_not_available)

        # If the graph was solved with a maximum work wire budget constraint, certain
        # decomposition rules will not have been explored because they were not feasible
        # under the work wire constraint specified at the time of solving the graph.
        # We should let the user know if they've specified a work wire budget (via the
        # num_work_wires argument) that does not match what was assumed when the grah
        # was being solved.

        def filter_fn(node):
            return num_work_wires == self._max_work_wires - node.num_work_wire_not_available

        op_node = next(filter(filter_fn, op_nodes), None)

        if op_node is None:
            budget = "None (unlimited)" if num_work_wires is None else num_work_wires
            return (
                f"The decomposition graph was solved with {self._max_work_wires} work wires "
                "available for dynamic allocation at the top level. There is not a point where "
                f"a {op_rep} is decomposed with a dynamic allocation budget of {budget}."
            )

        return op_node


@partial(transform, is_informative=True)
def decomp_inspector(  # pylint: disable=too-many-arguments
    tape: QuantumScript,
    *,
    gate_set=None,
    num_work_wires: int | None = 0,
    minimize_work_wires: bool = False,
    fixed_decomps: dict | None = None,
    alt_decomps: dict | None = None,
) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    """Inspect the decomposition graph solved with a given circuit.

    .. note::

        This function is only relevant when the new experimental graph-based decomposition system
        (introduced in v0.41) is enabled via :func:`~pennylane.decomposition.enable_graph`. This new
        way of doing decompositions is generally more resource efficient and accommodates multiple
        alternative decomposition rules for an operator.

    Args:
        tape (QuantumScript or QNode or Callable): a quantum circuit.
        gate_set (Iterable[str or type], Dict[type or str, float]): The target gate set specified
            as either (1) a sequence of operator types and/or names, (2) a dictionary mapping
            operator types and/or names to their respective costs, in which case the graph will
            try to minimize the total cost.
        num_work_wires (int): The maximum number of work wires that can be simultaneously allocated.
            If ``None``, assume an infinite number of work wires. Defaults to ``0``.
        minimize_work_wires (bool): If ``True``, minimize the number of work wires simultaneously
            allocated throughout the circuit. Defaults to ``False``.
        fixed_decomps (Dict[Type[Operator], DecompositionRule]): a dictionary mapping operator types
            to custom decomposition rules. A decomposition rule is a quantum function decorated with
            :func:`~pennylane.register_resources`. The custom decomposition rules specified here
            will be used in place of the existing decomposition rules defined for this operator.
        alt_decomps (Dict[Type[Operator], List[DecompositionRule]]): a dictionary mapping operator
            types to lists of alternative custom decomposition rules. A decomposition rule is a
            quantum function decorated with :func:`~pennylane.register_resources`. The custom
            decomposition rules specified here will be considered as alternatives to the existing
            decomposition rules defined for this operator, and one of them may be chosen if they
            lead to a more resource-efficient decomposition.

    Returns:
        DecompGraphInspector: an interactive object that can be used to inspect the graph.


    **Examples**

    When the ``decomp_inspector`` transform is applied on a circuit, the circuit will return a
    :class:`DecompGraphInspector` object constructed from operators in the circuit.

    .. code-block:: python

        qp.decomposition.enable_graph()

        @decomp_inspector(gate_set=qp.gate_sets.ROTATIONS_PLUS_CNOT, num_work_wires=2)
        @qp.qnode(qp.device("default.qubit"))
        def circuit():
            qp.ctrl(qp.MultiRZ(0.5, [0, 1]), control=[3, 4, 5])
            return qp.probs()

        inspector = circuit()

    The inspector object provides an :meth:`~DecompGraphInspector.inspect_decomps` method that shows
    the decomposition rules considered for a given operator.

    >>> print(inspector.inspect_decomps(qp.ctrl(qp.MultiRZ(0.5, [0, 1]), control=[3, 4, 5]), num_work_wires=2))
    CHOSEN: Decomposition 0 (name: flip_zero_ctrl_values(_ctrl_single_work_wire))
    <DynamicWire>: ──Allocate─╭X─╭●─────────────╭X──Deallocate─┤
                3: ───────────├●─│──────────────├●─────────────┤
                4: ───────────├●─│──────────────├●─────────────┤
                5: ───────────╰●─│──────────────╰●─────────────┤
                0: ──────────────├MultiRZ(0.50)────────────────┤
                1: ──────────────╰MultiRZ(0.50)────────────────┤
    First Expansion Gates: {MultiControlledX(num_control_wires=3, num_work_wires=0, num_zero_control_values=0, work_wire_type=borrowed): 2, Controlled(MultiRZ(num_wires=2), num_control_wires=1, num_work_wires=0, num_zero_control_values=0, work_wire_type=borrowed): 1}
    Wire Allocations: {'zero': 1}
    Full Expansion Gates: {RZ: 58, CNOT: 34, GlobalPhase: 64, RY: 18, RX: 8, MidMeasure: 2}
    Weighted Cost: 120.0
    <BLANKLINE>
    Decomposition 1 (name: to_controlled_qubit_unitary)
    Not applicable (provided operator instance does not meet all conditions for this rule).
    <BLANKLINE>
    Decomposition 2 (name: controlled(_multi_rz_decomposition))
    0: ─╭X─╭RZ(0.50)─╭X─┤
    1: ─├●─│─────────├●─┤
    3: ─├●─├●────────├●─┤
    4: ─├●─├●────────├●─┤
    5: ─╰●─╰●────────╰●─┤
    First Expansion Gates: {Controlled(RZ, num_control_wires=3, num_work_wires=0, num_zero_control_values=0, work_wire_type=borrowed): 1, MultiControlledX(num_control_wires=4, num_work_wires=0, num_zero_control_values=0, work_wire_type=borrowed): 2}
    Full Expansion Gates: {GlobalPhase: 76, RX: 16, MidMeasure: 4, RY: 24, RZ: 80, CNOT: 72}
    Weighted Cost: 196.0

    In addition to the operators at the top level of the circuit, we can also inspect the graph
    for how intermediate operators (such as the single-controlled ``MultiRZ`` produced in the
    decomposition of the controlled ``MultiRZ``) are decomposed (notice how ``num_work_wires``
    is set to 1 here, since decomposition of the top-level operator already used one of the work
    wires in the budget, so this inner operator has one fewer work wire available to it):

    >>> print(inspector.inspect_decomps(qp.ctrl(qp.MultiRZ(0.5, [0, 1]), control=2), num_work_wires=1))
    Decomposition 0 (name: flip_zero_ctrl_values(_ctrl_single_work_wire))
    Not applicable (provided operator instance does not meet all conditions for this rule).
    <BLANKLINE>
    Decomposition 1 (name: to_controlled_qubit_unitary)
    Not applicable (provided operator instance does not meet all conditions for this rule).
    <BLANKLINE>
    CHOSEN: Decomposition 2 (name: controlled(_multi_rz_decomposition))
    0: ─╭X─╭RZ(0.50)─╭X─┤
    1: ─├●─│─────────├●─┤
    2: ─╰●─╰●────────╰●─┤
    First Expansion Gates: {CRZ: 1, Toffoli: 2}
    Full Expansion Gates: {RZ: 20, CNOT: 14, GlobalPhase: 18, RY: 4}
    Weighted Cost: 38.0

    This can be useful to find out why a circuit cannot be decomposed:

    .. code-block:: python

        qp.decomposition.enable_graph()

        @decomp_inspector(gate_set={"RZ", "RX", "CNOT"}, num_work_wires=2)
        @qp.qnode(qp.device("default.qubit"))
        def circuit():
            qp.PauliRot(0.5, "XYZ", [0, 1, 2])
            return qp.probs()

        inspector = circuit()

    >>> print(inspector.inspect_decomps(qp.PauliRot(0.5, "XYZ", [0, 1, 2])))
    Decomposition 0 (name: _pauli_rot_decomposition)
    0: ──H────────╭MultiRZ(0.50)──H─────────┤
    1: ──RX(1.57)─├MultiRZ(0.50)──RX(-1.57)─┤
    2: ───────────╰MultiRZ(0.50)────────────┤
    First Expansion Gates: {Hadamard: 2, RX: 2, MultiRZ(num_wires=3): 1}
    Missing Ops: {Hadamard}

    The message suggests that the ``PauliRot`` could not be decomposed because the graph was unable
    to find a decomposition for ``Hadamard``. We can investigate further:

    >>> print(inspector.inspect_decomps(qp.Hadamard(0)))
    Decomposition 0 (name: _hadamard_to_rz_ry)
    0: ──RZ(3.14)──RY(1.57)──GlobalPhase(-1.57)─┤
    First Expansion Gates: {RZ: 1, RY: 1, GlobalPhase: 1}
    Missing Ops: {GlobalPhase}
    <BLANKLINE>
    Decomposition 1 (name: _hadamard_to_rz_rx)
    0: ──RZ(1.57)──RX(1.57)──RZ(1.57)──GlobalPhase(-1.57)─┤
    First Expansion Gates: {RZ: 2, RX: 1, GlobalPhase: 1}
    Missing Ops: {GlobalPhase}

    Now it's finally clear that the reason why ``PauliRot`` could not be decomposed was that
    ``GlobalPhase`` is missing from the target gate set.

    .. details::
        :title: Working with a dynamic work wire allocation budget

        Some decomposition rules make use of dynamically allocated work wires. For example:

        >>> print(qp.inspect_decomps(qp.MultiControlledX([0, 1, 2, 3]), "one_zeroed_worker"))
        Name: one_zeroed_worker
        <DynamicWire>: ──Allocate─╭⊕─╭●──⊕╮──Deallocate─┤
                    0: ───────────├●─│───●┤─────────────┤
                    1: ───────────╰●─│───●╯─────────────┤
                    2: ──────────────├●─────────────────┤
                    3: ──────────────╰X─────────────────┤
        Gate Count: {Toffoli: 1, TemporaryAND: 1, Adjoint(TemporaryAND): 1}
        Wire Allocations: {'zero': 1}

        These rules can only be selected if there are enough unused wires left on the device
        for allocation. In order to account for this, the :func:`~pennylane.transforms.decompose`
        transform takes a ``num_work_wires`` argument which acts as a work wire budget. It
        ensures that at no more than ``num_work_wires`` number of work wires can be simultaneously
        allocated during the decomposition. Consider the following circuit:

        .. code-block:: python

            @decomp_inspector(gate_set=qp.gate_sets.ROTATIONS_PLUS_CNOT, num_work_wires=2)
            @qp.qnode(qp.device("default.qubit"))
            def circuit():
                qp.ctrl(qp.MultiRZ(0.5, [0, 1]), control=[3, 4, 5, 6])
                qp.MultiControlledX([2, 3, 4, 5, 6])
                return qp.probs()

            inspector = circuit()

        When calling ``inspect_decomps``, we also need to provide the work wire allocation budget:

        >>> op = qp.ctrl(qp.MultiRZ(0.5, [0, 1]), control=[3, 4, 5, 6])
        >>> print(inspector.inspect_decomps(op, num_work_wires=2))
        CHOSEN: Decomposition 0 (name: flip_zero_ctrl_values(_ctrl_single_work_wire))
        <DynamicWire>: ──Allocate─╭X─╭●─────────────╭X──Deallocate─┤
                    3: ───────────├●─│──────────────├●─────────────┤
                    4: ───────────├●─│──────────────├●─────────────┤
                    5: ───────────├●─│──────────────├●─────────────┤
                    6: ───────────╰●─│──────────────╰●─────────────┤
                    0: ──────────────├MultiRZ(0.50)────────────────┤
                    1: ──────────────╰MultiRZ(0.50)────────────────┤
        First Expansion Gates: {MultiControlledX(num_control_wires=4, num_work_wires=0, num_zero_control_values=0, work_wire_type=borrowed): 2, Controlled(MultiRZ(num_wires=2), num_control_wires=1, num_work_wires=0, num_zero_control_values=0, work_wire_type=borrowed): 1}
        Wire Allocations: {'zero': 1}
        Full Expansion Gates: {RZ: 94, CNOT: 58, GlobalPhase: 104, RY: 26, RX: 12, MidMeasure: 2}
        Weighted Cost: 192.0
        <BLANKLINE>
        Decomposition 1 (name: to_controlled_qubit_unitary)
        Not applicable (provided operator instance does not meet all conditions for this rule).
        <BLANKLINE>
        Decomposition 2 (name: controlled(_multi_rz_decomposition))
        0: ─╭X─╭RZ(0.50)─╭X─┤
        1: ─├●─│─────────├●─┤
        3: ─├●─├●────────├●─┤
        4: ─├●─├●────────├●─┤
        5: ─├●─├●────────├●─┤
        6: ─╰●─╰●────────╰●─┤
        First Expansion Gates: {Controlled(RZ, num_control_wires=4, num_work_wires=0, num_zero_control_values=0, work_wire_type=borrowed): 1, MultiControlledX(num_control_wires=5, num_work_wires=0, num_zero_control_values=0, work_wire_type=borrowed): 2}
        Full Expansion Gates: {GlobalPhase: 200, RX: 32, MidMeasure: 6, RY: 54, RZ: 170, CNOT: 96}
        Weighted Cost: 358.0

        Similarly, for the ``MultiControlledX`` in the circuit:

        >>> op = qp.MultiControlledX([2, 3, 4, 5, 6])
        >>> print(inspector.inspect_decomps(op, num_work_wires=2))
        Decomposition 0 (name: flip_zero_ctrl_values(_2cx_elbow_explicit))
        Not applicable (provided operator instance does not meet all conditions for this rule).
        <BLANKLINE>
        Decomposition 1 (name: no_workers)
        2: ────╭●───────────────────╭●──────────────────────╭●──────────────────┤
        3: ────├●───────────────────├●──────────────────────├●──────────────────┤
        4: ────│─────────╭●─────────│─────────╭●────────────├●──────────────────┤
        5: ────│─────────├●─────────│─────────├●────────────├●──────────────────┤
        6: ──H─╰X──U(M0)─╰X──U(M0)†─╰X──U(M0)─╰X──U(M0)†──H─╰GlobalPhase(-1.57)─┤
        M0 =
        [[ 9.23879533e-01+0.38268343j -5.34910791e-34+0.j        ]
         [ 5.34910791e-34+0.j          9.23879533e-01-0.38268343j]]
        First Expansion Gates: {Hadamard: 2, QubitUnitary(num_wires=1): 2, MultiControlledX(num_control_wires=2, num_work_wires=2, num_zero_control_values=0, work_wire_type=borrowed): 4, Adjoint(QubitUnitary(num_wires=1)): 2, Controlled(GlobalPhase, num_control_wires=4, num_work_wires=0, num_zero_control_values=0, work_wire_type=borrowed): 1}
        Full Expansion Gates: {GlobalPhase: 43, RY: 14, RZ: 57, RX: 4, CNOT: 58}
        Weighted Cost: 133.0
        <BLANKLINE>
        Decomposition 2 (name: one_zeroed_worker)
        <DynamicWire>: ──Allocate─╭⊕───────╭●────────⊕╮──Deallocate─┤
                    2: ───────────├●───────│─────────●┤─────────────┤
                    3: ───────────╰●─╭X──X─├●──X─╭X──●╯─────────────┤
                    4: ──────────────├●────│─────├●─────────────────┤
                    5: ──────────────╰●────│─────╰●─────────────────┤
                    6: ────────────────────╰X───────────────────────┤
        First Expansion Gates: {Toffoli: 3, TemporaryAND: 1, Adjoint(TemporaryAND): 1, PauliX: 2}
        Wire Allocations: {'zero': 1}
        Full Expansion Gates: {GlobalPhase: 43, RX: 6, MidMeasure: 1, RY: 11, RZ: 37, CNOT: 22}
        Weighted Cost: 77.0
        <BLANKLINE>
        Decomposition 3 (name: one_borrowed_worker)
        <DynamicWire>: ──Allocate─╭X───────╭●───────╭X───────╭●──Deallocate────┤
                    2: ───────────├●───────│────────├●───────│─────────────────┤
                    3: ───────────╰●─╭X──X─├●──X─╭X─╰●─╭X──X─├●──X──────────╭X─┤
                    4: ──────────────├●────│─────├●────├●────│──────────────├●─┤
                    5: ──────────────╰●────│─────╰●────╰●────│──────────────╰●─┤
                    6: ────────────────────╰X────────────────╰X────────────────┤
        First Expansion Gates: {Toffoli: 8, PauliX: 4}
        Wire Allocations: {'any': 1}
        Full Expansion Gates: {GlobalPhase: 76, RX: 4, CNOT: 48, RZ: 72, RY: 16}
        Weighted Cost: 140.0
        <BLANKLINE>
        Decomposition 4 (name: one_explicit_worker)
        Not applicable (provided operator instance does not meet all conditions for this rule).
        <BLANKLINE>
        Decomposition 5 (name: two_zeroed_workers)
        Not applicable (provided operator instance does not meet all conditions for this rule).
        <BLANKLINE>
        Decomposition 6 (name: two_borrowed_workers)
        Not applicable (provided operator instance does not meet all conditions for this rule).
        <BLANKLINE>
        Decomposition 7 (name: two_explicit_workers)
        Not applicable (provided operator instance does not meet all conditions for this rule).
        <BLANKLINE>
        CHOSEN: Decomposition 8 (name: many_zeroed_workers)
        <DynamicWire>: ─╭Allocate────╭⊕─╭●──⊕╮─────╭Deallocate─┤
        <DynamicWire>: ─╰Allocate─╭⊕─├●─│───●┤──⊕╮─╰Deallocate─┤
                    5: ───────────├●─│──│────│──●┤─────────────┤
                    4: ───────────╰●─│──│────│──●╯─────────────┤
                    3: ──────────────╰●─│───●╯─────────────────┤
                    2: ─────────────────├●─────────────────────┤
                    6: ─────────────────╰X─────────────────────┤
        First Expansion Gates: {TemporaryAND: 2, Adjoint(TemporaryAND): 2, Toffoli: 1}
        Wire Allocations: {'zero': 2}
        Full Expansion Gates: {GlobalPhase: 37, RX: 8, MidMeasure: 2, RY: 12, RZ: 29, CNOT: 14}
        Weighted Cost: 65.0
        <BLANKLINE>
        Decomposition 9 (name: many_borrowed_workers)
        <DynamicWire>: ─╭Allocate─╭●─╭X────╭X─╭●─╭X────╭X─╭Deallocate─┤
        <DynamicWire>: ─╰Allocate─│──├●─╭X─├●─│──├●─╭X─├●─╰Deallocate─┤
                    2: ───────────├●─│──│──│──├●─│──│──│──────────────┤
                    6: ───────────╰X─│──│──│──╰X─│──│──│──────────────┤
                    3: ──────────────╰●─│──╰●────╰●─│──╰●─────────────┤
                    5: ─────────────────├●──────────├●────────────────┤
                    4: ─────────────────╰●──────────╰●────────────────┤
        First Expansion Gates: {Toffoli: 8}
        Wire Allocations: {'any': 2}
        Full Expansion Gates: {CNOT: 48, GlobalPhase: 72, RZ: 72, RY: 16}
        Weighted Cost: 136.0
        <BLANKLINE>
        Decomposition 10 (name: many_explicit_workers)
        Not applicable (provided operator instance does not meet all conditions for this rule).
        <BLANKLINE>
        Decomposition 11 (name: _mcx_to_cnot_or_toffoli)
        Not applicable (provided operator instance does not meet all conditions for this rule).

        We can see that the chosen decomposition rule for the ``MultiControlledX`` uses two work
        wires. However, not every ``MultiControlledX`` in the circuit can be decomposed the same
        way. For example, notice that the chosen decomposition rule for the controlled ``MultiRZ``
        takes a work wire from the dynamic allocation budget, therefore, within the region of the
        decomposition rule, the ``MultiControlledX`` has one fewer work wire available to it. We
        can inspect how the graph chose a decomposition rule for the inner ``MultiControlledX``
        by changing the ``num_work_wires`` argument:

        >>> op = qp.MultiControlledX([2, 3, 4, 5, 6])  # concrete wire labels don't matter
        >>> print(inspector.inspect_decomps(op, num_work_wires=1))
        Decomposition 0 (name: flip_zero_ctrl_values(_2cx_elbow_explicit))
        Not applicable (provided operator instance does not meet all conditions for this rule).
        <BLANKLINE>
        Decomposition 1 (name: no_workers)
        2: ────╭●───────────────────╭●──────────────────────╭●──────────────────┤
        3: ────├●───────────────────├●──────────────────────├●──────────────────┤
        4: ────│─────────╭●─────────│─────────╭●────────────├●──────────────────┤
        5: ────│─────────├●─────────│─────────├●────────────├●──────────────────┤
        6: ──H─╰X──U(M0)─╰X──U(M0)†─╰X──U(M0)─╰X──U(M0)†──H─╰GlobalPhase(-1.57)─┤
        M0 =
        [[ 9.23879533e-01+0.38268343j -5.34910791e-34+0.j        ]
         [ 5.34910791e-34+0.j          9.23879533e-01-0.38268343j]]
        First Expansion Gates: {Hadamard: 2, QubitUnitary(num_wires=1): 2, MultiControlledX(num_control_wires=2, num_work_wires=2, num_zero_control_values=0, work_wire_type=borrowed): 4, Adjoint(QubitUnitary(num_wires=1)): 2, Controlled(GlobalPhase, num_control_wires=4, num_work_wires=0, num_zero_control_values=0, work_wire_type=borrowed): 1}
        Full Expansion Gates: {GlobalPhase: 43, RY: 14, RZ: 57, RX: 4, CNOT: 58}
        Weighted Cost: 133.0
        <BLANKLINE>
        CHOSEN: Decomposition 2 (name: one_zeroed_worker)
        <DynamicWire>: ──Allocate─╭⊕───────╭●────────⊕╮──Deallocate─┤
                    2: ───────────├●───────│─────────●┤─────────────┤
                    3: ───────────╰●─╭X──X─├●──X─╭X──●╯─────────────┤
                    4: ──────────────├●────│─────├●─────────────────┤
                    5: ──────────────╰●────│─────╰●─────────────────┤
                    6: ────────────────────╰X───────────────────────┤
        First Expansion Gates: {Toffoli: 3, TemporaryAND: 1, Adjoint(TemporaryAND): 1, PauliX: 2}
        Wire Allocations: {'zero': 1}
        Full Expansion Gates: {GlobalPhase: 43, RX: 6, MidMeasure: 1, RY: 11, RZ: 37, CNOT: 22}
        Weighted Cost: 77.0
        <BLANKLINE>
        Decomposition 3 (name: one_borrowed_worker)
        <DynamicWire>: ──Allocate─╭X───────╭●───────╭X───────╭●──Deallocate────┤
                    2: ───────────├●───────│────────├●───────│─────────────────┤
                    3: ───────────╰●─╭X──X─├●──X─╭X─╰●─╭X──X─├●──X──────────╭X─┤
                    4: ──────────────├●────│─────├●────├●────│──────────────├●─┤
                    5: ──────────────╰●────│─────╰●────╰●────│──────────────╰●─┤
                    6: ────────────────────╰X────────────────╰X────────────────┤
        First Expansion Gates: {Toffoli: 8, PauliX: 4}
        Wire Allocations: {'any': 1}
        Full Expansion Gates: {GlobalPhase: 76, RX: 4, CNOT: 48, RZ: 72, RY: 16}
        Weighted Cost: 140.0
        <BLANKLINE>
        Decomposition 4 (name: one_explicit_worker)
        Not applicable (provided operator instance does not meet all conditions for this rule).
        <BLANKLINE>
        Decomposition 5 (name: two_zeroed_workers)
        Not applicable (provided operator instance does not meet all conditions for this rule).
        <BLANKLINE>
        Decomposition 6 (name: two_borrowed_workers)
        Not applicable (provided operator instance does not meet all conditions for this rule).
        <BLANKLINE>
        Decomposition 7 (name: two_explicit_workers)
        Not applicable (provided operator instance does not meet all conditions for this rule).
        <BLANKLINE>
        Decomposition 8 (name: many_zeroed_workers)
        Insufficient work wires: requires 2 but only 1 available.
        <BLANKLINE>
        Decomposition 9 (name: many_borrowed_workers)
        Insufficient work wires: requires 2 but only 1 available.
        <BLANKLINE>
        Decomposition 10 (name: many_explicit_workers)
        Not applicable (provided operator instance does not meet all conditions for this rule).
        <BLANKLINE>
        Decomposition 11 (name: _mcx_to_cnot_or_toffoli)
        Not applicable (provided operator instance does not meet all conditions for this rule).

    """

    if not enabled_graph():
        raise ValueError(
            "The decomp_inspector transform is only relevant with the new "
            "graph-based decomposition system. Use qp.decomposition.enable_graph() "
            "to enable the new system."
        )

    gate_set, _ = _resolve_gate_set(gate_set)
    graph = DecompositionGraph(
        tape.operations,
        gate_set,
        fixed_decomps=fixed_decomps,
        alt_decomps=alt_decomps,
    )
    solution = graph.solve(
        num_work_wires=num_work_wires,
        minimize_work_wires=minimize_work_wires,
        lazy=False,
    )

    def postprocessing(_):
        return DecompGraphInspector(graph, solution)

    return [tape], postprocessing
