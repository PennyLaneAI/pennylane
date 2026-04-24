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

"""Defines the inspect_decomp_graph transform."""

from functools import partial
from typing import override

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
    def get_gate_count_str(self, estimated_count, actual_count) -> str:
        estimated_count = {k: v for k, v in estimated_count.items() if v > 0}
        if estimated_count == actual_count:
            return f"First Expansion Gates: {estimated_count}"
        return (
            f"Estimated First Expansion Gates: {estimated_count}\n"
            f"Actual First Expansion Gates: {actual_count}"
        )


# pylint: disable=protected-access,too-few-public-methods
class DecompGraphInspector:
    """Allows inspectability into the decomposition graph."""

    def __init__(self, decomp_graph: DecompositionGraph, solution: DecompGraphSolution) -> None:
        self._decomp_graph = decomp_graph
        self._raw_graph = decomp_graph._graph
        self._solution = solution
        self._visitor = solution._visitor
        self._max_work_wires = self._solution.num_work_wires

    def inspect_decomps(self, op: Operator, num_work_wires: int | None = 0) -> str:
        """Display all decomposition rules considered for a given operator."""

        op_node = self._find_op_node(op, num_work_wires)

        if isinstance(op_node, str):
            return op_node

        op_node_idx = self._decomp_graph._all_op_indices[op_node]
        decomp_indices = self._raw_graph.predecessor_indices(op_node_idx)

        decomp_strings = []
        for i, node_idx in enumerate(decomp_indices):
            rule_info = _DecompInGraphInfo(op, node_idx, num_work_wires, self._solution)
            decomp_str = f"Decomposition {i} (name: {rule_info.name})\n{rule_info}"
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
def inspect_decomp_graph(  # pylint: disable=too-many-arguments
    tape: QuantumScript,
    *,
    gate_set=None,
    num_work_wires: int | None = 0,
    minimize_work_wires: bool = False,
    fixed_decomps: dict | None = None,
    alt_decomps: dict | None = None,
) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    """Inspect the decomposition graph."""

    if not enabled_graph():
        raise ValueError(
            "The inspect_decomp_graph transform is only relevant with the new "
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
