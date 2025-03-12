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

from unittest.mock import patch

import numpy as np
import pytest
from conftest import decompositions

import pennylane as qml
from pennylane.decomposition import DecompositionGraph, Resources
from pennylane.decomposition.decomposition_graph import DecompositionError


@patch(
    "pennylane.decomposition.decomposition_graph.list_decomps",
    side_effect=lambda x: decompositions[x],
)
class TestDecompositionGraph:

    @pytest.mark.unit
    def test_get_decomp_rule(self, _):
        """Tests the internal method that gets the decomposition rules for an operator."""

        @qml.register_resources({qml.PhaseShift: 2, qml.RX: 1})
        def custom_hadamard(wires):
            qml.PhaseShift(np.pi / 2, wires=wires)
            qml.RX(np.pi / 2, wires=wires)
            qml.PhaseShift(np.pi / 2, wires=wires)

        @qml.register_resources({qml.PhaseShift: 1, qml.RY: 1})
        def custom_hadamard_2(wires):
            qml.PhaseShift(np.pi / 2, wires=wires)
            qml.RY(np.pi / 2, wires=wires)

        graph = DecompositionGraph(operations=[qml.Hadamard(0)], target_gate_set={"RX", "RY", "RZ"})
        assert graph._get_decompositions(qml.Hadamard) == decompositions[qml.Hadamard]

        graph = DecompositionGraph(
            operations=[qml.Hadamard(0)],
            target_gate_set={"RX", "RY", "RZ"},
            fixed_decomps={qml.Hadamard: custom_hadamard},
        )
        assert graph._get_decompositions(qml.Hadamard) == [custom_hadamard]

        graph = DecompositionGraph(
            operations=[qml.Hadamard(0)],
            target_gate_set={"RX", "RY", "RZ"},
            alt_decomps={qml.Hadamard: [custom_hadamard, custom_hadamard_2]},
        )
        assert (
            graph._get_decompositions(qml.Hadamard)
            == [
                custom_hadamard,
                custom_hadamard_2,
            ]
            + decompositions[qml.Hadamard]
        )

    @pytest.mark.unit
    def test_graph_construction(self, _):
        """Tests constructing a graph from a single Hadamard."""

        op = qml.Hadamard(wires=[0])
        graph = DecompositionGraph(operations=[op], target_gate_set={"RX", "RZ", "GlobalPhase"})
        # 5 ops and 3 decompositions (2 for Hadamard and 1 for RY)
        assert len(graph._graph.nodes()) == 8
        # 8 edges from ops to decompositions and 3 from decompositions to ops
        assert len(graph._graph.edges()) == 11

        # Check that graph construction stops at gates in the target gate set.
        graph2 = DecompositionGraph(operations=[op], target_gate_set={"RY", "RZ", "GlobalPhase"})
        # 5 ops and 2 decompositions (RY is in the target gate set now)
        assert len(graph2._graph.nodes()) == 7
        # 6 edges from ops to decompositions and 2 from decompositions to ops
        assert len(graph2._graph.edges()) == 8

    @pytest.mark.unit
    def test_graph_solve(self, _):
        """Tests solving a simple graph for the optimal decompositions."""

        op = qml.Hadamard(wires=[0])
        graph = DecompositionGraph(
            operations=[op],
            target_gate_set={"RX", "RY", "RZ", "GlobalPhase"},
        )
        graph.solve()

        # verify that the better decomposition rule is chosen when both are valid.
        expected_resource = Resources(
            num_gates=3,
            gate_counts={
                qml.resource_rep(qml.RZ): 1,
                qml.resource_rep(qml.RY): 1,
                qml.resource_rep(qml.GlobalPhase): 1,
            },
        )
        assert graph.resource_estimates(op) == expected_resource
        assert graph.decomposition(op).compute_resources() == expected_resource

    @pytest.mark.unit
    def test_decomposition_not_found(self, _):
        """Tests that the correct error is raised if a decomposition isn't found."""

        op = qml.Hadamard(wires=[0])
        graph = DecompositionGraph(operations=[op], target_gate_set={"RX", "RY", "GlobalPhase"})
        with pytest.raises(DecompositionError, match="Decomposition not found for {'Hadamard'}"):
            graph.solve()

    @pytest.mark.unit
    def test_decomposition_with_resource_params(self, _):
        """Tests operators with non-empty resource params."""

        class CustomOp(qml.operation.Operation):  # pylint: disable=too-few-public-methods
            """A custom operation."""

            resource_param_keys = ("num_wires",)

            @property
            def resource_params(self):
                return {"num_wires": len(self.wires)}

        def _custom_resource(num_wires):
            return {
                qml.resource_rep(qml.MultiRZ, num_wires=num_wires): 1,
                qml.resource_rep(qml.MultiRZ, num_wires=num_wires - 1): 2,
            }

        @qml.register_resources(_custom_resource)
        def _custom_decomp(*_, **__):
            raise NotImplementedError

        decompositions[CustomOp] = [_custom_decomp]

        op = CustomOp(wires=[0, 1, 2, 3])
        graph = DecompositionGraph(
            operations=[op],
            target_gate_set={"RX", "RZ", "CZ", "GlobalPhase"},
        )
        # 10 ops and 7 decompositions (1 for the custom op, 1 for each of the two MultiRZs,
        # 1 for CNOT, 2 for Hadamard, and 1 for RY)
        assert len(graph._graph.nodes()) == 17
        # 16 edges from ops to decompositions and 7 from decompositions to ops
        assert len(graph._graph.edges()) == 23

        graph.solve()
        assert graph.resource_estimates(op) == Resources(
            num_gates=129,
            gate_counts={
                qml.resource_rep(qml.CZ): 14,
                qml.resource_rep(qml.RZ): 59,
                qml.resource_rep(qml.RX): 28,
                qml.resource_rep(qml.GlobalPhase): 28,
            },
        )
        assert graph.decomposition(op).compute_resources(**op.resource_params) == Resources(
            num_gates=3,
            gate_counts={
                qml.resource_rep(qml.MultiRZ, num_wires=4): 1,
                qml.resource_rep(qml.MultiRZ, num_wires=3): 2,
            },
        )
        assert graph.decomposition(qml.Hadamard(wires=[0])).compute_resources() == Resources(
            num_gates=4,
            gate_counts={
                qml.resource_rep(qml.RZ): 2,
                qml.resource_rep(qml.RX): 1,
                qml.resource_rep(qml.GlobalPhase): 1,
            },
        )


@patch(
    "pennylane.decomposition.decomposition_graph.list_decomps",
    side_effect=lambda x: decompositions[x],
)
class TestControlledDecompositions:  # pylint: disable=too-few-public-methods
    """Tests that the decomposition graph can handle controlled decompositions."""

    def test_general_controlled_op(self, _):
        """Tests that a general controlled op can be decomposed."""

        class CustomOp(qml.operation.Operation):  # pylint: disable=too-few-public-methods
            """A custom operation."""

            resource_param_keys = ()

            @property
            def resource_params(self):
                return {}

        @qml.register_resources({qml.X: 1, qml.GlobalPhase: 1})
        def custom_decomp(wires):
            qml.X(wires[0])
            qml.GlobalPhase(np.pi, wires=wires)

        @qml.register_resources({qml.Z: 1, qml.GlobalPhase: 1})
        def second_decomp(wires):
            qml.Z(wires=wires[0])
            qml.GlobalPhase(np.pi / 2, wires=wires)

        class CustomControlledOp(qml.operation.Operation):  # pylint: disable=too-few-public-methods
            """A custom operation."""

            resource_param_keys = ()

            @property
            def resource_params(self):
                return {}

        @qml.register_resources(
            {
                qml.Z: 1,
                qml.controlled_resource_rep(
                    CustomOp,
                    {},
                    num_control_wires=1,
                    num_zero_control_values=0,
                    num_work_wires=0,
                ): 1,
            }
        )
        def custom_controlled_decomp(wires):
            qml.Z(wires=wires[0])
            qml.ctrl(CustomOp(wires=wires[1]), control=wires[0])

        op1 = qml.ctrl(CustomOp(wires=[0]), control=[1])
        op2 = qml.ctrl(CustomOp(wires=[0]), control=[1, 2], control_values=[True, False])
        op3 = qml.ctrl(CustomControlledOp(wires=[0, 1]), control=[2], control_values=[False])
        graph = DecompositionGraph(
            operations=[op1, op2, op3],
            target_gate_set={"CNOT", "Toffoli", "CCZ", "RZ", "RX", "GlobalPhase"},
            alt_decomps={
                CustomOp: [custom_decomp, second_decomp],
                CustomControlledOp: [custom_controlled_decomp],
            },
        )
        # 18 op nodes and 16 decomposition nodes.
        assert len(graph._graph.nodes()) == 34
        # 16 edges from decompositions to ops and 36 edges from ops to decompositions
        assert len(graph._graph.edges()) == 52

        graph.solve()

        # Check that decomposition rules are found for the necessary controlled operators.
        assert graph.decomposition(op1)
        assert graph.decomposition(op2)
        assert graph.decomposition(op3)
        assert graph.decomposition(qml.ctrl(qml.GlobalPhase(0.5), control=[1]))
        assert graph.decomposition(qml.ctrl(qml.GlobalPhase(0.5), control=[1, 2]))
        assert graph.decomposition(qml.ctrl(CustomOp(wires=[1]), control=[0, 2]))
