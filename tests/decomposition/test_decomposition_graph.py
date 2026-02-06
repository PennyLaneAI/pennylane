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

"""Unit tests for the decomposition graph."""
# pylint: disable=protected-access,no-name-in-module

import warnings
from unittest.mock import patch

import numpy as np
import pytest

import pennylane as qp
from conftest import decompositions, to_resources
from pennylane.decomposition import (
    DecompositionGraph,
    adjoint_resource_rep,
    controlled_resource_rep,
    pow_resource_rep,
    resource_rep,
)
from pennylane.decomposition.decomposition_graph import _to_name
from pennylane.exceptions import DecompositionError
from pennylane.operation import Operation

# pylint: disable=protected-access,no-name-in-module


@pytest.mark.unit
@patch(
    "pennylane.decomposition.decomposition_graph.list_decomps",
    side_effect=lambda x: decompositions[_to_name(x)],
)
class TestDecompositionGraph:
    def test_weighted_graph_solve(self, _):
        """Tests solving a simple graph for the optimal decompositions with weighted gates."""

        op = qp.CRX(2.5, wires=[0, 1])

        # the RZ CZ RX CZ decomp is chosen when the RZ and CNOT weights are large.
        gate_weights = {
            "RX": 1.0,
            "RY": 3.0,
            "RZ": 10.0,
            "GlobalPhase": 1.0,
            "CNOT": 20.0,
            "CZ": 1.0,
        }

        graph = DecompositionGraph(
            operations=[op],
            gate_set=gate_weights,
        )
        solution = graph.solve()

        expected_resource = to_resources({qp.CZ: 2, qp.RX: 2})
        assert solution.resource_estimate(op) == expected_resource

        # the RZ CZ RX CZ decomp is avoided when the CZ weight is large.
        gate_weights = {
            "RX": 1.0,
            "RY": 1.0,
            "RZ": 1.0,
            "GlobalPhase": 1.0,
            "CNOT": 1.0,
            "CZ": 100.0,
        }

        graph = DecompositionGraph(
            operations=[op],
            gate_set=gate_weights,
        )
        solution = graph.solve()

        expected_resource = to_resources(
            {qp.RX: 2, qp.CNOT: 2, qp.RY: 4, qp.GlobalPhase: 4, qp.RZ: 4}
        )
        assert solution.resource_estimate(op) == expected_resource

    def test_get_decomp_rule(self, _):
        """Tests the internal method that gets the decomposition rules for an operator."""

        @qp.register_resources({qp.PhaseShift: 2, qp.RX: 1})
        def custom_hadamard(wires):
            qp.PhaseShift(np.pi / 2, wires=wires)
            qp.RX(np.pi / 2, wires=wires)
            qp.PhaseShift(np.pi / 2, wires=wires)

        @qp.register_resources({qp.PhaseShift: 1, qp.RY: 1})
        def custom_hadamard_2(wires):
            qp.PhaseShift(np.pi / 2, wires=wires)
            qp.RY(np.pi / 2, wires=wires)

        graph = DecompositionGraph(operations=[qp.Hadamard(0)], gate_set={"RX", "RY", "RZ"})
        assert graph._get_decompositions(resource_rep(qp.H)) == decompositions["Hadamard"]

        graph = DecompositionGraph(
            operations=[qp.Hadamard(0)],
            gate_set={"RX", "RY", "RZ"},
            fixed_decomps={qp.Hadamard: custom_hadamard},
        )
        assert graph._get_decompositions(resource_rep(qp.H)) == [custom_hadamard]

        alt_dec = [custom_hadamard, custom_hadamard_2]
        graph = DecompositionGraph(
            operations=[qp.Hadamard(0)],
            gate_set={"RX", "RY", "RZ"},
            alt_decomps={qp.Hadamard: alt_dec},
        )
        exp_dec = alt_dec + decompositions["Hadamard"]
        assert graph._get_decompositions(resource_rep(qp.H)) == exp_dec

        graph = DecompositionGraph(
            operations=[qp.Hadamard(0)],
            gate_set={"RX", "RY", "RZ"},
            alt_decomps={qp.Hadamard: alt_dec},
            fixed_decomps={qp.Hadamard: custom_hadamard},
        )
        assert graph._get_decompositions(resource_rep(qp.H)) == [custom_hadamard]

    def test_graph_construction(self, _):
        """Tests constructing a graph from a single Hadamard."""

        op = qp.Hadamard(wires=[0])

        graph = DecompositionGraph(
            operations=[op], gate_set={"RX": 1.0, "RZ": 1.0, "GlobalPhase": 1.0}
        )
        # 5 ops and 3 decompositions (2 for Hadamard and 1 for RY) and 1 dummy starting node
        assert len(graph._graph.nodes()) == 9
        # 8 edges from ops to decompositions, 3 from decompositions to ops, and 3 from the
        # dummy starting node to the target gate set.
        assert len(graph._graph.edges()) == 14

        # Check that graph construction stops at gates in the target gate set.
        graph2 = DecompositionGraph(operations=[op], gate_set={"RY", "RZ", "GlobalPhase"})
        # 5 ops and 2 decompositions (RY is in the target gate set now), and the dummy starting node
        assert len(graph2._graph.nodes()) == 8
        # 6 edges from ops to decompositions and 2 from decompositions to ops,
        # and 3 from the dummy starting node to the target gate set.
        assert len(graph2._graph.edges()) == 11

    def test_graph_construction_non_applicable_rules(self, _):
        """Tests rules which are not applicable are skipped."""

        class CustomOp(Operation):  # pylint: disable=too-few-public-methods
            """A custom op"""

            resource_keys = {"num_wires"}

            @property
            def resource_params(self):
                return {"num_wires": len(self.wires)}

        @qp.register_condition(lambda num_wires: num_wires == 1)
        @qp.register_resources({qp.RZ: 1, qp.CNOT: 1})
        def some_rule(*_, **__):
            raise NotImplementedError

        def _some_other_resource(num_wires):
            return {qp.RZ: 1, qp.CNOT: num_wires - 1}

        @qp.register_condition(lambda num_wires: num_wires >= 2)
        @qp.register_resources(_some_other_resource)
        def some_other_rule(*_, **__):
            raise NotImplementedError

        graph = DecompositionGraph(
            [CustomOp(wires=[0, 1])],
            gate_set={"CNOT", "RZ"},
            alt_decomps={CustomOp: [some_rule, some_other_rule]},
        )
        # 3 ops (CustomOp, CNOT, RZ) and 1 decompositions (only some_other_rule),
        # and the dummy starting node
        assert len(graph._graph.nodes()) == 5
        # 2 edges from ops to decompositions, 1 from decompositions to ops,
        # and 2 from the dummy starting node to the target gate set
        assert len(graph._graph.edges()) == 5

    def test_gate_set(self, _):
        """Tests that graph construction stops at the target gate set."""

        class CustomOp(Operation):  # pylint: disable=too-few-public-methods
            """A custom operation."""

            resource_keys = set()

            @property
            def resource_params(self):
                return {}

        @qp.register_resources(
            {
                qp.RX: 1,
                qp.X: 1,
                adjoint_resource_rep(qp.RY): 1,
                controlled_resource_rep(qp.T, {}, num_control_wires=2): 1,
                pow_resource_rep(qp.Z, {}, z=2): 1,
            }
        )
        def custom_decomp(wires):
            qp.RX(0.1, wires=wires[0])
            qp.X(wires=wires[1])
            qp.adjoint(qp.RY(0.1, wires=wires[2]))
            qp.ctrl(qp.T(0), control=[1, 2])
            qp.pow(qp.Z(0), 2)

        op = CustomOp(wires=[0, 1, 2])
        graph = DecompositionGraph(
            [op],
            gate_set={"RX", "X", "Adjoint(RY)", "C(T)", "Pow(Z)"},
            fixed_decomps={CustomOp: custom_decomp},
        )

        # 1 node for CustomOp, 1 decomposition node, 5 for the ops in the decomposition,
        # and the dummy starting node.
        assert len(graph._graph.nodes()) == 8
        # 5 edges from ops to decompositions, 1 edge from decompositions to ops, and 5
        # edges from the dummy starting node to the target gate set.
        assert len(graph._graph.edges()) == 11

        solution = graph.solve()

        # verify that is_solved_for returns False for gates in the gate set
        assert not solution.is_solved_for(qp.RX(0.5, wires=0))

        # verify that the correct error is raised
        with pytest.raises(DecompositionError, match="is unsolved in this decomposition graph."):
            solution.decomposition(qp.RX(0.5, wires=0))

    def test_graph_solve(self, _):
        """Tests solving a simple graph for the optimal decompositions."""

        op = qp.Hadamard(wires=[0])
        graph = DecompositionGraph(
            operations=[op],
            gate_set={"RX", "RY", "RZ", "GlobalPhase"},
        )
        solution = graph.solve()

        # verify that the better decomposition rule is chosen when both are valid.
        assert solution.resource_estimate(op) == to_resources(
            {qp.RY: 1, qp.GlobalPhase: 1, qp.RZ: 1},
        )
        assert solution.decomposition(op).compute_resources() == to_resources(
            {qp.RY: 1, qp.GlobalPhase: 1, qp.RZ: 1},
        )

        # verify that is_solved_for returns False for non-existent operators
        assert not solution.is_solved_for(qp.Toffoli(wires=[0, 1, 2]))

    def test_decomposition_not_found_warning(self, _):
        """Tests that the correct warning is raised if a decomposition isn't found."""

        op = qp.Hadamard(wires=[0])
        graph = DecompositionGraph(operations=[op], gate_set={"RX", "RY", "GlobalPhase"})
        with pytest.warns(UserWarning, match="unable to find a decomposition for {'Hadamard'}"):
            graph.solve()

    @pytest.mark.parametrize(
        "op", [qp.allocation.Allocate(1), qp.allocation.Deallocate(qp.allocation.DynamicWire())]
    )
    def test_decomposition_not_found_ignored_op_no_warning(self, _, op):
        """Tests that no warning is raised if a decomposition isn't found but the unsolved
        operator type is among specific operators, like Allocate and Deallocate."""

        graph = DecompositionGraph(operations=[op], gate_set={"RX", "RY", "GlobalPhase"})
        with warnings.catch_warnings(record=True) as record:
            graph.solve()
        assert len(record) == 0

    def test_lazy_solve(self, _):
        """Tests the lazy keyword argument."""

        class CustomOp(Operation):  # pylint: disable=too-few-public-methods
            """A custom operation."""

            resource_keys = set()

            @property
            def resource_params(self):
                return {}

        class AnotherOp(Operation):  # pylint: disable=too-few-public-methods
            """Another custom operation."""

            resource_keys = set()

            @property
            def resource_params(self):
                return {}

        @qp.register_resources({qp.RZ: 1, qp.CNOT: 1})
        def _custom_decomp(*_, **__):
            raise NotImplementedError

        @qp.register_resources({AnotherOp: 1, qp.CNOT: 1})
        def _custom_decomp_2(*_, **__):
            raise NotImplementedError

        @qp.register_resources({qp.RZ: 2, qp.CNOT: 2})
        def _another_decomp(*_, **__):
            raise NotImplementedError

        graph = DecompositionGraph(
            operations=[CustomOp(wires=[0])],
            gate_set={"CNOT", "RZ"},
            alt_decomps={
                CustomOp: [_custom_decomp, _custom_decomp_2],
                AnotherOp: [_another_decomp],
            },
        )
        solution = graph.solve(lazy=True)
        assert not solution.is_solved_for(AnotherOp(wires=[0, 1]))

        with pytest.raises(DecompositionError, match="is unsolved in this decomposition graph."):
            solution.resource_estimate(AnotherOp(wires=[0, 1]))

        with pytest.raises(DecompositionError, match="is unsolved in this decomposition graph."):
            solution.decomposition(AnotherOp(wires=[0, 1]))

        solution = graph.solve(lazy=False)
        assert solution.is_solved_for(AnotherOp(wires=[0, 1]))

    def test_decomposition_with_resource_params(self, _):
        """Tests operators with non-empty resource params."""

        class CustomOp(Operation):  # pylint: disable=too-few-public-methods
            """A custom operation."""

            resource_keys = {"num_wires"}

            @property
            def resource_params(self):
                return {"num_wires": len(self.wires)}

        def _custom_resource(num_wires):
            return {
                qp.resource_rep(qp.MultiRZ, num_wires=num_wires): 1,
                qp.resource_rep(qp.MultiRZ, num_wires=num_wires - 1): 2,
            }

        @qp.register_resources(_custom_resource)
        def _custom_decomp(*_, **__):
            raise NotImplementedError

        op = CustomOp(wires=[0, 1, 2, 3])
        graph = DecompositionGraph(
            operations=[op],
            gate_set={"RX", "RZ", "CZ", "GlobalPhase"},
            alt_decomps={CustomOp: [_custom_decomp]},
        )
        # 10 ops (CustomOp, MultiRZ(4), MultiRZ(3), CNOT, CZ, RX, RY, RZ, Hadamard, GlobalPhase)
        # 7 decompositions (1 for CustomOp, 1 for each of the two MultiRZs, 1 for CNOT, 2 for Hadamard, and 1 for RY)
        # and the dummy starting node
        assert len(graph._graph.nodes()) == 18
        # 16 edges from ops to decompositions and 7 from decompositions to ops,
        # and 4 edges from the dummy starting node to the target gate set
        assert len(graph._graph.edges()) == 27

        solution = graph.solve()
        assert solution.resource_estimate(op) == to_resources(
            {qp.CZ: 14, qp.RZ: 59, qp.RX: 28, qp.GlobalPhase: 28},
        )
        assert solution.decomposition(op).compute_resources(**op.resource_params) == to_resources(
            {
                qp.resource_rep(qp.MultiRZ, num_wires=4): 1,
                qp.resource_rep(qp.MultiRZ, num_wires=3): 2,
            },
        )
        assert solution.decomposition(qp.Hadamard(wires=[0])).compute_resources() == to_resources(
            {qp.RZ: 2, qp.RX: 1, qp.GlobalPhase: 1},
        )

    def test_work_wire_requirement(self, _):
        """Tests that the graph respects the work wire requirement."""

        class CustomOp(Operation):  # pylint: disable=too-few-public-methods
            """A custom operation."""

            resource_keys = set()

            @property
            def resource_params(self):
                return {}

        @qp.register_resources({qp.Toffoli: 2, qp.CRot: 1}, work_wires={"zeroed": 1})
        def _decomp_with_work_wire(*_, **__):
            raise NotImplementedError

        @qp.register_resources({qp.Toffoli: 2, qp.CRot: 3})
        def _decomp_without_work_wire(*_, **__):
            raise NotImplementedError

        graph = DecompositionGraph(
            [CustomOp(wires=[0, 1, 2])],
            gate_set={qp.Toffoli, qp.CRot},
            alt_decomps={CustomOp: [_decomp_without_work_wire, _decomp_with_work_wire]},
        )

        solution = graph.solve(num_work_wires=0)
        assert solution.decomposition(CustomOp(wires=[0, 1, 2])) is _decomp_without_work_wire

        solution = graph.solve(num_work_wires=1)
        assert (
            solution.decomposition(CustomOp(wires=[0, 1, 2]), num_work_wires=1)
            is _decomp_with_work_wire
        )

    def test_multiple_nodes_with_different_work_wire_budget(self, _):
        """Tests that the same operator produced under different work wire budgets
        are stored as different nodes in the graph, and results can be queried."""

        class CustomOp(Operation):  # pylint: disable=too-few-public-methods
            """A custom operation."""

            resource_keys = set()

            @property
            def resource_params(self):
                return {}

        @qp.register_resources({qp.Toffoli: 2, qp.CRot: 1}, work_wires={"zeroed": 2})
        def _decomp_with_work_wire(*_, **__):
            raise NotImplementedError

        @qp.register_resources({qp.Toffoli: 4, qp.CRot: 3})
        def _decomp_without_work_wire(*_, **__):
            raise NotImplementedError

        class LargeOp(Operation):  # pylint: disable=too-few-public-methods
            """A larger custom operation."""

            resource_keys = set()

            @property
            def resource_params(self):
                return {}

        @qp.register_resources({qp.Toffoli: 2, CustomOp: 2}, work_wires={"zeroed": 1})
        def _decomp2_with_work_wire(*_, **__):
            raise NotImplementedError

        @qp.register_resources({qp.Toffoli: 4, CustomOp: 2})
        def _decomp2_without_work_wire(*_, **__):
            raise NotImplementedError

        op = LargeOp(wires=[0, 1, 2, 3])
        small_op = CustomOp(wires=[0, 1, 2])

        graph = DecompositionGraph(
            [op, small_op],
            gate_set={qp.Toffoli, qp.RZ, qp.RY, qp.CNOT},
            alt_decomps={
                CustomOp: [_decomp_without_work_wire, _decomp_with_work_wire],
                LargeOp: [_decomp2_without_work_wire, _decomp2_with_work_wire],
            },
        )

        # 1 node for LargerOp, 2 nodes for CustomOp, 1 for Toffoli, 1 for CRot, 1 for RZ,
        # 1 for RY, 1 for CNOT, and 1 dummy starting node, 1 decomposition from CRot,
        # node, 2 decomposition nodes from LargerOp, 2 decompositions from each CustomOp
        assert len(graph._graph.nodes()) == 16
        assert len(graph._graph.edges()) == 26

        solution = graph.solve(num_work_wires=0)
        assert solution.decomposition(op) is _decomp2_without_work_wire
        assert solution.decomposition(small_op) is _decomp_without_work_wire

        solution = graph.solve(num_work_wires=1)
        assert solution.decomposition(op, num_work_wires=1) is _decomp2_with_work_wire
        assert solution.decomposition(small_op, num_work_wires=0) is _decomp_without_work_wire

        solution = graph.solve(num_work_wires=2)
        # When there are only 2 work wires available, by construction, it is more
        # resource efficient to use them on the CustomOp, so even where there are
        # enough work wires to use the more efficient decomposition for the LargeOp,
        # we should still choose the less efficient one to achieve better overall
        # resource efficiency. Because if we use one of the work wires to decompose
        # the LargeOp, there won't be enough work wires left to further decompose
        # the 2 CustomOp and it would result in significantly more gates.
        assert solution.decomposition(op, num_work_wires=2) is _decomp2_without_work_wire
        assert solution.decomposition(small_op, num_work_wires=2) is _decomp_with_work_wire

        solution = graph.solve(num_work_wires=3)
        assert solution.decomposition(op, num_work_wires=3) is _decomp2_with_work_wire
        assert solution.decomposition(small_op, num_work_wires=2) is _decomp_with_work_wire
        assert solution.decomposition(small_op, num_work_wires=3) is _decomp_with_work_wire

        solution = graph.solve(num_work_wires=None)
        assert solution.decomposition(op, num_work_wires=None) is _decomp2_with_work_wire
        assert solution.decomposition(small_op, num_work_wires=None) is _decomp_with_work_wire

    def test_non_work_wire_dependent_ops_reused(self, _):
        """Tests that ops that are not work-wire dependent are not affected by work-wire
        dependent decomposition rules upstream."""

        class SimpleOp(Operation):  # pylint: disable=too-few-public-methods
            """A simple operation that does not depend on work wires."""

        @qp.register_resources({qp.X: 1})
        def _simple_decomp(_):
            raise NotImplementedError

        class CustomOp(Operation):  # pylint: disable=too-few-public-methods
            """Another operation."""

        @qp.register_resources({SimpleOp: 1}, work_wires={"zeroed": 1})
        def _custom_decomp(_):
            raise NotImplementedError

        @qp.register_resources({qp.X: 1})
        def _another_decomp(_):
            raise NotImplementedError

        graph = DecompositionGraph(
            [CustomOp(0), SimpleOp(0)],
            gate_set={qp.X},
            alt_decomps={SimpleOp: [_simple_decomp], CustomOp: [_custom_decomp, _another_decomp]},
        )
        solution = graph.solve()
        assert solution.is_solved_for(SimpleOp(0))

    def test_min_work_wires(self, _):
        """Tests that the graph tracks the minimum number of work wires."""

        class SimpleOp(Operation):  # pylint: disable=too-few-public-methods
            """A simple operation that does not depend on work wires."""

        @qp.register_resources({qp.X: 4})
        def _simple_decomp(_):
            raise NotImplementedError

        class CustomOp(Operation):  # pylint: disable=too-few-public-methods
            """Another operation."""

        @qp.register_resources({SimpleOp: 1, qp.CNOT: 4}, work_wires={"zeroed": 2})
        def _custom_decomp(_):
            raise NotImplementedError

        class AnotherOp(Operation):  # pylint: disable=too-few-public-methods
            """Some other op."""

        @qp.register_resources({CustomOp: 1, qp.CNOT: 4}, work_wires={"zeroed": 2})
        def _another_decomp(_):
            raise NotImplementedError

        @qp.register_resources({SimpleOp: 3, qp.CNOT: 4}, work_wires={"zeroed": 3})
        def _yet_another_decomp(_):
            raise NotImplementedError

        graph = DecompositionGraph(
            [AnotherOp(0)],
            gate_set={qp.X, qp.CNOT},
            alt_decomps={
                SimpleOp: [_simple_decomp],
                CustomOp: [_custom_decomp],
                AnotherOp: [_another_decomp, _yet_another_decomp],
            },
        )
        assert graph._min_work_wires == 3
        with pytest.raises(DecompositionError, match="at least 3 work wires"):
            graph.solve(num_work_wires=2)

        solution = graph.solve(num_work_wires=None)
        assert solution.decomposition(AnotherOp(0)) == _another_decomp

        solution = graph.solve(num_work_wires=None, minimize_work_wires=True)
        assert solution.decomposition(AnotherOp(0), num_work_wires=None) == _yet_another_decomp


@pytest.mark.unit
@patch(
    "pennylane.decomposition.decomposition_graph.list_decomps",
    side_effect=lambda x: decompositions[x],
)
class TestControlledDecompositions:
    """Tests that the decomposition graph can handle controlled decompositions."""

    def test_controlled_global_phase(self, _):
        """Tests that a controlled global phase can be decomposed."""

        op1 = qp.ctrl(qp.GlobalPhase(0.5), control=[1])
        op2 = qp.ctrl(qp.GlobalPhase(0.5), control=[1, 2])
        graph = DecompositionGraph([op1, op2], gate_set={"ControlledPhaseShift", "PhaseShift"})
        # 4 op nodes and 2 decomposition nodes, and 1 dummy starting node.
        assert len(graph._graph.nodes()) == 7
        # 2 edges from decompositions to ops and 2 edges from ops to decompositions,
        # and 2 edges from the dummy starting node to the target gate set.
        assert len(graph._graph.edges()) == 6

        # Verify the decompositions
        solution = graph.solve()
        with qp.queuing.AnnotatedQueue() as q:
            solution.decomposition(op1)(*op1.parameters, wires=op1.wires, **op1.hyperparameters)
            solution.decomposition(op2)(*op2.parameters, wires=op2.wires, **op2.hyperparameters)

        assert q.queue == [
            qp.PhaseShift(-0.5, wires=[1]),
            qp.ControlledPhaseShift(-0.5, wires=[1, 2]),
        ]

    def test_custom_controlled_op(self, _):
        """Tests that a general controlled op can be decomposed into a custom op if applicable."""

        op1 = qp.ops.Controlled(qp.X(0), control_wires=[1])
        op2 = qp.ops.Controlled(qp.H(0), control_wires=[1])
        graph = DecompositionGraph(
            operations=[op1, op2],
            gate_set={"CNOT", "CH"},
        )
        assert len(graph._graph.nodes()) == 32
        assert len(graph._graph.edges()) == 49

        # Verify the decompositions
        solution = graph.solve()
        with qp.queuing.AnnotatedQueue() as q:
            solution.decomposition(op1)(*op1.parameters, wires=op1.wires, **op1.hyperparameters)
            solution.decomposition(op2)(*op2.parameters, wires=op2.wires, **op2.hyperparameters)

        assert q.queue == [qp.CNOT(wires=[1, 0]), qp.CH(wires=[1, 0])]

    def test_controlled_base_decomposition(self, _):
        """Tests applying control on the decomposition of the target operator."""

        class CustomOp(Operation):  # pylint: disable=too-few-public-methods
            """A custom operation."""

            resource_keys = set()

            @property
            def resource_params(self):
                return {}

        @qp.register_resources({qp.X: 1, qp.GlobalPhase: 1})
        def custom_decomp(wires):
            qp.X(wires[0])
            qp.GlobalPhase(np.pi, wires=wires)

        @qp.register_resources({qp.Z: 1, qp.GlobalPhase: 1})
        def second_decomp(wires):
            qp.Z(wires=wires[0])
            qp.GlobalPhase(np.pi / 2, wires=wires)

        class CustomControlledOp(Operation):  # pylint: disable=too-few-public-methods
            """A custom operation."""

            resource_keys = set()

            @property
            def resource_params(self):
                return {}

        @qp.register_resources(
            {
                qp.Z: 1,
                qp.decomposition.controlled_resource_rep(
                    CustomOp,
                    {},
                    num_control_wires=1,
                    num_zero_control_values=0,
                    num_work_wires=0,
                ): 1,
            }
        )
        def custom_controlled_decomp(wires):
            qp.Z(wires=wires[0])
            qp.ctrl(CustomOp(wires=wires[1]), control=wires[0])

        op1 = qp.ctrl(CustomOp(wires=[0]), control=[1])
        op2 = qp.ctrl(CustomOp(wires=[0]), control=[1, 2], control_values=[True, False])
        op3 = qp.ctrl(CustomControlledOp(wires=[0, 1]), control=[2], control_values=[False])
        graph = DecompositionGraph(
            operations=[op1, op2, op3],
            gate_set={"CNOT", "Toffoli", "CCZ", "RZ", "RX", "GlobalPhase"},
            alt_decomps={
                CustomOp: [custom_decomp, second_decomp],
                CustomControlledOp: [custom_controlled_decomp],
            },
        )
        # 18 op nodes and 16 decomposition nodes, and the dummy starting node
        assert len(graph._graph.nodes()) == 35
        # 16 edges from decompositions to ops and 36 edges from ops to decompositions
        # and 6 edge from the dummy starting node to the target gate set.
        assert len(graph._graph.edges()) == 58

        solution = graph.solve()

        # Check that decomposition rules are found for the necessary controlled operators.
        assert solution.decomposition(op1)
        assert solution.decomposition(op2)
        assert solution.decomposition(op3)
        assert solution.decomposition(qp.ctrl(qp.GlobalPhase(0.5), control=[1]))
        assert solution.decomposition(qp.ctrl(qp.GlobalPhase(0.5), control=[1, 2]))
        assert solution.decomposition(qp.ctrl(CustomOp(wires=[1]), control=[0, 2]))

    def test_flip_controlled_adjoint(self, _):
        """Tests that the controlled form of an adjoint operator is decomposed properly."""

        op = qp.ctrl(qp.adjoint(qp.U1(0.5, wires=0)), control=[1])
        graph = DecompositionGraph(operations=[op], gate_set={"ControlledPhaseShift"})
        solution = graph.solve()
        with qp.queuing.AnnotatedQueue() as q:
            solution.decomposition(op)(*op.parameters, wires=op.wires, **op.hyperparameters)
        assert q.queue == [qp.adjoint(qp.ops.Controlled(qp.U1(0.5, wires=0), control_wires=[1]))]

    def test_decompose_with_single_work_wire(self, _):
        """Tests that the Lemma 7.11 decomposition from https://arxiv.org/pdf/quant-ph/9503016 is applied correctly."""

        op = qp.ctrl(qp.Rot(0.123, 0.234, 0.345, wires=0), control=[1, 2, 3])

        graph = DecompositionGraph(operations=[op], gate_set={"MultiControlledX", "CRot"})
        solution = graph.solve(num_work_wires=1)
        with qp.queuing.AnnotatedQueue() as q:
            rule = solution.decomposition(op, num_work_wires=1)
            rule(*op.parameters, wires=op.wires, **op.hyperparameters)
        tape = qp.tape.QuantumScript.from_queue(q)
        [tape], _ = qp.transforms.resolve_dynamic_wires([tape], min_int=4)
        assert tape.operations == [
            qp.MultiControlledX(wires=[1, 2, 3, 4]),
            qp.CRot(0.123, 0.234, 0.345, wires=[4, 0]),
            qp.MultiControlledX(wires=[1, 2, 3, 4]),
        ]
        assert solution.resource_estimate(op, num_work_wires=1) == to_resources(
            {controlled_resource_rep(qp.X, {}, num_control_wires=3): 2, qp.CRot: 1}
        )


@patch(
    "pennylane.decomposition.decomposition_graph.list_decomps",
    side_effect=lambda x: decompositions[x],
)
class TestSymbolicDecompositions:
    """Tests decompositions of symbolic ops."""

    def test_cancel_adjoint(self, _):
        """Tests that a nested adjoint operator is flattened properly."""

        op = qp.adjoint(qp.adjoint(qp.RX(0.5, wires=[0])))

        graph = DecompositionGraph(operations=[op], gate_set={"RX"})
        # 2 operator nodes (Adjoint(Adjoint(RX)) and RX), and 1 decomposition node.
        # and the dummy starting node
        assert len(graph._graph.nodes()) == 4
        assert len(graph._graph.edges()) == 3

        solution = graph.solve()
        with qp.queuing.AnnotatedQueue() as q:
            solution.decomposition(op)(*op.parameters, wires=op.wires, **op.hyperparameters)

        assert q.queue == [qp.RX(0.5, wires=[0])]
        assert solution.resource_estimate(op) == to_resources({qp.RX: 1})

    def test_adjoint_custom(self, _):
        """Tests adjoint of an operator that defines its own adjoint."""

        op = qp.adjoint(qp.RX(0.5, wires=[0]))

        graph = DecompositionGraph(operations=[op], gate_set={"RX"})
        # 2 operator nodes (Adjoint(RX) and RX), and 1 decomposition node, and 1 dummy starting node
        assert len(graph._graph.nodes()) == 4
        assert len(graph._graph.edges()) == 3

        solution = graph.solve()
        with qp.queuing.AnnotatedQueue() as q:
            solution.decomposition(op)(*op.parameters, wires=op.wires, **op.hyperparameters)

        assert q.queue == [qp.RX(-0.5, wires=[0])]
        assert solution.resource_estimate(op) == to_resources({qp.RX: 1})

    def test_adjoint_general(self, _):
        """Tests decomposition of a generalized adjoint operation."""

        class CustomOp(Operation):  # pylint: disable=too-few-public-methods
            """A custom operation."""

            resource_keys = set()

            @property
            def resource_params(self):
                return {}

        @qp.register_resources({qp.H: 1, qp.CNOT: 2, qp.RX: 1, qp.T: 1})
        def custom_decomp(phi, wires):
            qp.H(wires[0])
            qp.CNOT(wires=wires[:2])
            qp.RX(phi, wires=wires[1])
            qp.CNOT(wires=wires[1:3])
            qp.T(wires[2])

        op = qp.adjoint(CustomOp(0.5, wires=[0, 1, 2]))
        graph = DecompositionGraph(
            operations=[op],
            gate_set={"H", "CNOT", "RX", "PhaseShift"},
            alt_decomps={CustomOp: [custom_decomp]},
        )
        # 10 operator nodes: A(CustomOp), A(H), A(CNOT), A(RX), A(T), H, CNOT, RX, A(PhaseShift), PhaseShift
        # 6 decomposition nodes for: A(CustomOp), A(CNOT), A(RX), A(T), A(PhaseShift), A(H)
        # 1 dummy starting node
        assert len(graph._graph.nodes()) == 17
        # 9 edges from ops to decompositions and 6 edges from decompositions to ops.
        # and 4 edges from the dummy starting node to the target gate set.
        assert len(graph._graph.edges()) == 19

        solution = graph.solve()
        with qp.queuing.AnnotatedQueue() as q:
            solution.decomposition(op)(*op.parameters, wires=op.wires, **op.hyperparameters)

        assert q.queue == [
            qp.adjoint(qp.T(2)),
            qp.adjoint(qp.CNOT(wires=[1, 2])),
            qp.adjoint(qp.RX(0.5, wires=1)),
            qp.adjoint(qp.CNOT(wires=[0, 1])),
            qp.adjoint(qp.H(wires=0)),
        ]
        assert solution.resource_estimate(op) == to_resources(
            {qp.H: 1, qp.CNOT: 2, qp.RX: 1, qp.PhaseShift: 1},
        )

    def test_nested_powers(self, _):
        """Tests nested power decompositions."""

        op = qp.pow(qp.pow(qp.H(0), 3), 2)
        graph = DecompositionGraph(operations=[op], gate_set={"H"})
        # 3 operator nodes: Pow(Pow(H)), Pow(H), and H
        # 1 decomposition nodes for Pow(Pow(H)) and 1 nodes for Pow(H)
        # and the dummy starting node
        assert len(graph._graph.nodes()) == 5
        # 2 edges from decompositions to ops and 1 edge from ops to decompositions
        # and 1 edge from the dummy starting node to the target gate set. Note that
        # H**6 decomposes to nothing, so H isn't counted.
        assert len(graph._graph.edges()) == 4

        solution = graph.solve()
        with qp.queuing.AnnotatedQueue() as q:
            solution.decomposition(op)(*op.parameters, wires=op.wires, **op.hyperparameters)

        assert q.queue == [qp.pow(qp.H(0), 6)]
        assert solution.resource_estimate(op) == to_resources({})

        op2 = qp.pow(qp.H(0), 6)
        with qp.queuing.AnnotatedQueue() as q:
            solution.decomposition(op2)(*op2.parameters, wires=op2.wires, **op2.hyperparameters)

        assert q.queue == []
        assert solution.resource_estimate(op2) == to_resources({})

    def test_custom_symbolic_decompositions(self, _):
        """Tests that custom symbolic decompositions are used."""

        @qp.register_resources({qp.RX: 1})
        def my_adjoint_rx(theta, wires, **__):
            qp.RX(-theta, wires=wires)

        graph = DecompositionGraph(
            operations=[
                qp.adjoint(qp.H(0)),
                qp.pow(qp.H(1), 3),
                qp.ops.Controlled(qp.H(0), control_wires=1),
                qp.adjoint(qp.RX(0.5, wires=0)),
            ],
            fixed_decomps={"Adjoint(RX)": my_adjoint_rx},
            gate_set={"H", "CH", "RX"},
        )

        op1 = qp.adjoint(qp.H(0))
        op2 = qp.pow(qp.H(1), 3)
        op3 = qp.ops.Controlled(qp.H(0), control_wires=1)
        op4 = qp.adjoint(qp.RX(0.5, wires=0))

        solution = graph.solve()
        with qp.queuing.AnnotatedQueue() as q:
            solution.decomposition(op1)(*op1.parameters, wires=op1.wires, **op1.hyperparameters)
            solution.decomposition(op2)(*op2.parameters, wires=op2.wires, **op2.hyperparameters)
            solution.decomposition(op3)(*op3.parameters, wires=op3.wires, **op3.hyperparameters)
            solution.decomposition(op4)(*op4.parameters, wires=op4.wires, **op4.hyperparameters)

        assert q.queue == [qp.H(0), qp.H(1), qp.CH(wires=[1, 0]), qp.RX(-0.5, wires=0)]
        assert solution.resource_estimate(op1) == to_resources({qp.H: 1})
        assert solution.resource_estimate(op2) == to_resources({qp.H: 1})
        assert solution.resource_estimate(op3) == to_resources({qp.CH: 1})
        assert solution.resource_estimate(op4) == to_resources({qp.RX: 1})

    def test_special_pow_decomps(self, _):
        """Tests special cases for decomposing a power."""

        class CustomOp(Operation):  # pylint: disable=too-few-public-methods
            """A custom operation."""

            resource_keys = set()

            @property
            def resource_params(self):
                return {}

        graph = DecompositionGraph(
            operations=[qp.pow(CustomOp(0), 0), qp.pow(CustomOp(1), 1)], gate_set={"CustomOp"}
        )
        # 3 operator nodes: Pow(CustomOp, 0), Pow(CustomOp, 1), and CustomOp
        # 1 decomposition node for Pow(CustomOp, 0) and 1 node for Pow(CustomOp, 1)
        # and the dummy starting node
        assert len(graph._graph.nodes()) == 6
        # 2 edges from decompositions to ops and 1 edge from ops to decompositions
        # and 1 edge from the dummy starting node to the target gate set.
        # and 1 edge from the dummy starting node to the empty decomposition.
        assert len(graph._graph.edges()) == 5

        op1 = qp.pow(CustomOp(0), 0)
        op2 = qp.pow(CustomOp(1), 1)

        solution = graph.solve()
        with qp.queuing.AnnotatedQueue() as q:
            solution.decomposition(op1)(*op1.parameters, wires=op1.wires, **op1.hyperparameters)
            solution.decomposition(op2)(*op2.parameters, wires=op2.wires, **op2.hyperparameters)

        assert q.queue == [CustomOp(1)]
        assert solution.resource_estimate(op1) == to_resources({})
        assert solution.resource_estimate(op2) == to_resources({CustomOp: 1})

    def test_general_pow_decomps(self, _):
        """Tests the more general power decomposition rules."""

        class CustomOp(Operation):  # pylint: disable=too-few-public-methods
            """A custom operation."""

            resource_keys = set()

            @property
            def resource_params(self):
                return {}

        graph = DecompositionGraph(
            operations=[qp.pow(CustomOp(0), 2), qp.pow(qp.adjoint(CustomOp(1)), 2)],
            gate_set={"CustomOp", "Adjoint(CustomOp)"},
        )
        # 5 operator nodes: Pow(CustomOp), Pow(Adjoint(CustomOp)), Adjoint(Pow(CustomOp)),
        # Adjoint(CustomOp), CustomOp, and the dummy starting node
        # 3 decomposition nodes for each of Pow(CustomOp), Pow(Adjoint(CustomOp)), Adjoint(Pow(CustomOp))
        assert len(graph._graph.nodes()) == 9
        # 3 edges from decompositions to ops and 3 edges from ops to decompositions
        # and 2 edges from the dummy starting node to the target gate set.
        assert len(graph._graph.edges()) == 8

        op1 = qp.pow(CustomOp(0), 2)
        op2 = qp.pow(qp.adjoint(CustomOp(1)), 2)

        solution = graph.solve()
        with qp.queuing.AnnotatedQueue() as q:
            solution.decomposition(op1)(*op1.parameters, wires=op1.wires, **op1.hyperparameters)
            solution.decomposition(op2)(*op2.parameters, wires=op2.wires, **op2.hyperparameters)

        assert q.queue == [
            CustomOp(0),
            CustomOp(0),
            qp.adjoint(qp.pow(CustomOp(1), 2)),
        ]
