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
from conftest import decompositions, to_resources

import pennylane as qml
from pennylane.decomposition import DecompositionError, DecompositionGraph


@pytest.mark.unit
@patch(
    "pennylane.decomposition.decomposition_graph.list_decomps",
    side_effect=lambda x: decompositions[x],
)
class TestDecompositionGraph:

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

        alt_dec = [custom_hadamard, custom_hadamard_2]
        graph = DecompositionGraph(
            operations=[qml.Hadamard(0)],
            target_gate_set={"RX", "RY", "RZ"},
            alt_decomps={qml.Hadamard: alt_dec},
        )
        exp_dec = alt_dec + decompositions[qml.Hadamard]
        assert graph._get_decompositions(qml.Hadamard) == exp_dec

        graph = DecompositionGraph(
            operations=[qml.Hadamard(0)],
            target_gate_set={"RX", "RY", "RZ"},
            alt_decomps={qml.Hadamard: alt_dec},
            fixed_decomps={qml.Hadamard: custom_hadamard},
        )
        assert graph._get_decompositions(qml.Hadamard) == [custom_hadamard]

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

    def test_graph_solve(self, _):
        """Tests solving a simple graph for the optimal decompositions."""

        op = qml.Hadamard(wires=[0])
        graph = DecompositionGraph(
            operations=[op],
            target_gate_set={"RX", "RY", "RZ", "GlobalPhase"},
        )
        graph.solve()

        # verify that the better decomposition rule is chosen when both are valid.
        expected_resource = to_resources({qml.RZ: 1, qml.RY: 1, qml.GlobalPhase: 1})
        assert graph.resource_estimate(op) == expected_resource
        assert graph.decomposition(op).compute_resources() == expected_resource

    def test_decomposition_not_found(self, _):
        """Tests that the correct error is raised if a decomposition isn't found."""

        op = qml.Hadamard(wires=[0])
        graph = DecompositionGraph(operations=[op], target_gate_set={"RX", "RY", "GlobalPhase"})
        with pytest.raises(DecompositionError, match="Decomposition not found for {'Hadamard'}"):
            graph.solve()

    def test_lazy_solve(self, _):
        """Tests the lazy keyword argument."""

        class CustomOp(qml.operation.Operation):  # pylint: disable=too-few-public-methods
            """A custom operation."""

            resource_keys = set()

            @property
            def resource_params(self):
                return {}

        class AnotherOp(qml.operation.Operation):  # pylint: disable=too-few-public-methods
            """Another custom operation."""

            resource_keys = set()

            @property
            def resource_params(self):
                return {}

        @qml.register_resources({qml.RZ: 1, qml.CNOT: 1})
        def _custom_decomp(*_, **__):
            raise NotImplementedError

        @qml.register_resources({AnotherOp: 1, qml.CNOT: 1})
        def _custom_decomp_2(*_, **__):
            raise NotImplementedError

        @qml.register_resources({qml.RZ: 2, qml.CNOT: 2})
        def _another_decomp(*_, **__):
            raise NotImplementedError

        graph = DecompositionGraph(
            operations=[CustomOp(wires=[0])],
            target_gate_set={"CNOT", "RZ"},
            alt_decomps={
                CustomOp: [_custom_decomp, _custom_decomp_2],
                AnotherOp: [_another_decomp],
            },
        )
        graph.solve(lazy=True)
        assert not graph.is_solved_for(AnotherOp(wires=[0, 1]))

        with pytest.raises(DecompositionError, match="is unsolved in this decomposition graph."):
            graph.resource_estimate(AnotherOp(wires=[0, 1]))

        with pytest.raises(DecompositionError, match="is unsolved in this decomposition graph."):
            graph.decomposition(AnotherOp(wires=[0, 1]))

        graph.solve(lazy=False)
        assert graph.is_solved_for(AnotherOp(wires=[0, 1]))

    def test_decomposition_with_resource_params(self, _):
        """Tests operators with non-empty resource params."""

        class CustomOp(qml.operation.Operation):  # pylint: disable=too-few-public-methods
            """A custom operation."""

            resource_keys = {"num_wires"}

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

        op = CustomOp(wires=[0, 1, 2, 3])
        graph = DecompositionGraph(
            operations=[op],
            target_gate_set={"RX", "RZ", "CZ", "GlobalPhase"},
            alt_decomps={CustomOp: [_custom_decomp]},
        )
        # 10 ops (CustomOp, MultiRZ(4), MultiRZ(3), CNOT, CZ, RX, RY, RZ, Hadamard, GlobalPhase)
        # 7 decompositions (1 for CustomOp, 1 for each of the two MultiRZs, 1 for CNOT, 2 for Hadamard, and 1 for RY)
        assert len(graph._graph.nodes()) == 17
        # 16 edges from ops to decompositions and 7 from decompositions to ops
        assert len(graph._graph.edges()) == 23

        graph.solve()
        assert graph.resource_estimate(op) == to_resources(
            {qml.CZ: 14, qml.RZ: 59, qml.RX: 28, qml.GlobalPhase: 28}
        )
        assert graph.decomposition(op).compute_resources(**op.resource_params) == to_resources(
            {
                qml.resource_rep(qml.MultiRZ, num_wires=4): 1,
                qml.resource_rep(qml.MultiRZ, num_wires=3): 2,
            }
        )
        assert graph.decomposition(qml.Hadamard(wires=[0])).compute_resources() == to_resources(
            {qml.RZ: 2, qml.RX: 1, qml.GlobalPhase: 1}
        )


@pytest.mark.unit
@patch(
    "pennylane.decomposition.decomposition_graph.list_decomps",
    side_effect=lambda x: decompositions[x],
)
class TestControlledDecompositions:
    """Tests that the decomposition graph can handle controlled decompositions."""

    def test_controlled_global_phase(self, _):
        """Tests that a controlled global phase can be decomposed."""

        op1 = qml.ctrl(qml.GlobalPhase(0.5), control=[1])
        op2 = qml.ctrl(qml.GlobalPhase(0.5), control=[1, 2])
        graph = DecompositionGraph(
            [op1, op2], target_gate_set={"ControlledPhaseShift", "PhaseShift"}
        )
        # 4 op nodes and 2 decomposition nodes.
        assert len(graph._graph.nodes()) == 6
        # 2 edges from decompositions to ops and 2 edges from ops to decompositions
        assert len(graph._graph.edges()) == 4

        # Verify the decompositions
        graph.solve()
        with qml.queuing.AnnotatedQueue() as q:
            graph.decomposition(op1)(*op1.parameters, wires=op1.wires, **op1.hyperparameters)
            graph.decomposition(op2)(*op2.parameters, wires=op2.wires, **op2.hyperparameters)

        assert q.queue == [
            qml.PhaseShift(-0.5, wires=[1]),
            qml.ControlledPhaseShift(-0.5, wires=[1, 2]),
        ]

    def test_custom_controlled_op(self, _):
        """Tests that a general controlled op can be decomposed into a custom op if applicable."""

        op1 = qml.ops.Controlled(qml.X(0), control_wires=[1])
        op2 = qml.ops.Controlled(qml.H(0), control_wires=[1])
        graph = DecompositionGraph(
            operations=[op1, op2],
            target_gate_set={"CNOT", "CH"},
        )
        # 4 op nodes and 2 decomposition nodes.
        assert len(graph._graph.nodes()) == 6
        # 2 edges from decompositions to ops and 2 edges from ops to decompositions
        assert len(graph._graph.edges()) == 4

        # Verify the decompositions
        graph.solve()
        with qml.queuing.AnnotatedQueue() as q:
            graph.decomposition(op1)(*op1.parameters, wires=op1.wires, **op1.hyperparameters)
            graph.decomposition(op2)(*op2.parameters, wires=op2.wires, **op2.hyperparameters)

        assert q.queue == [qml.CNOT(wires=[1, 0]), qml.CH(wires=[1, 0])]

    def test_controlled_base_decomposition(self, _):
        """Tests applying control on the decomposition of the target operator."""

        class CustomOp(qml.operation.Operation):  # pylint: disable=too-few-public-methods
            """A custom operation."""

            resource_keys = set()

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

            resource_keys = set()

            @property
            def resource_params(self):
                return {}

        @qml.register_resources(
            {
                qml.Z: 1,
                qml.decomposition.controlled_resource_rep(
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


@patch(
    "pennylane.decomposition.decomposition_graph.list_decomps",
    side_effect=lambda x: decompositions[x],
)
class TestSymbolicDecompositions:
    """Tests decompositions of symbolic ops."""

    def test_adjoint_adjoint(self, _):
        """Tests that a nested adjoint operator is flattened properly."""

        op = qml.adjoint(qml.adjoint(qml.RX(0.5, wires=[0])))

        graph = DecompositionGraph(operations=[op], target_gate_set={"RX"})
        # 2 operator nodes (Adjoint(Adjoint(RX)) and RX), and 1 decomposition node.
        assert len(graph._graph.nodes()) == 3
        assert len(graph._graph.edges()) == 2

        graph.solve()
        with qml.queuing.AnnotatedQueue() as q:
            graph.decomposition(op)(*op.parameters, wires=op.wires, **op.hyperparameters)

        assert q.queue == [qml.RX(0.5, wires=[0])]
        assert graph.resource_estimate(op) == to_resources({qml.RX: 1})

    def test_adjoint_pow(self, _):
        """tests that an adjoint of a power of am operator that has adjoint is decomposed."""

        op = qml.adjoint(qml.pow(qml.H(0), z=3))

        graph = DecompositionGraph(operations=[op], target_gate_set={"Hadamard"})
        # 3 operator nodes: Adjoint(Pow(H)), Pow(H), and H
        # 2 decomposition nodes for Adjoint(Pow(H)) and Pow(H)
        assert len(graph._graph.nodes()) == 5
        # 2 edges from decompositions to ops and 2 edges from ops to decompositions
        assert len(graph._graph.edges()) == 4

        graph.solve()
        with qml.queuing.AnnotatedQueue() as q:
            graph.decomposition(op)(*op.parameters, wires=op.wires, **op.hyperparameters)

        assert q.queue == [qml.pow(qml.H(0), z=3)]
        # TODO: There should just be a single `H` after we have full support of Pow decompositions.
        assert graph.resource_estimate(op) == to_resources({qml.H: 3})

    def test_adjoint_custom(self, _):
        """Tests adjoint of an operator that defines its own adjoint."""

        op = qml.adjoint(qml.RX(0.5, wires=[0]))

        graph = DecompositionGraph(operations=[op], target_gate_set={"RX"})
        # 2 operator nodes (Adjoint(RX) and RX), and 1 decomposition node.
        assert len(graph._graph.nodes()) == 3
        assert len(graph._graph.edges()) == 2

        graph.solve()
        with qml.queuing.AnnotatedQueue() as q:
            graph.decomposition(op)(*op.parameters, wires=op.wires, **op.hyperparameters)

        assert q.queue == [qml.RX(-0.5, wires=[0])]
        assert graph.resource_estimate(op) == to_resources({qml.RX: 1})

    def test_adjoint_controlled(self, _):
        """Tests that the adjoint of a controlled operator is decomposed properly."""

        op = qml.adjoint(qml.ops.Controlled(qml.RX(0.5, wires=[0]), control_wires=1))
        op2 = qml.adjoint(qml.ctrl(qml.U1(0.5, wires=0), control=[1]))

        graph = DecompositionGraph(
            operations=[op, op2], target_gate_set={"ControlledPhaseShift", "CRX"}
        )
        # 5 operator nodes: Adjoint(C(RX)), Adjoint(C(U1)), CRX, C(U1), ControlledPhaseShift
        # 3 decomposition nodes leading into Adjoint(C(RX)), Adjoint(C(U1)), and C(U1)
        assert len(graph._graph.nodes()) == 8
        # 3 edges from decompositions to ops and 3 edges from ops to decompositions
        assert len(graph._graph.edges()) == 6

        graph.solve()
        with qml.queuing.AnnotatedQueue() as q:
            graph.decomposition(op)(*op.parameters, wires=op.wires, **op.hyperparameters)
            graph.decomposition(op2)(*op2.parameters, wires=op2.wires, **op2.hyperparameters)

        assert q.queue == [qml.CRX(-0.5, wires=[1, 0]), qml.ctrl(qml.U1(-0.5, wires=0), control=1)]

        op3 = qml.ctrl(qml.U1(-0.5, wires=0), control=1)
        with qml.queuing.AnnotatedQueue() as q:
            graph.decomposition(op3)(*op3.parameters, wires=op3.wires, **op3.hyperparameters)

        assert q.queue == [qml.ControlledPhaseShift(-0.5, wires=[1, 0])]
        assert graph.resource_estimate(op2) == to_resources({qml.ControlledPhaseShift: 1})

    def test_adjoint_general(self, _):
        """Tests decomposition of a generalized adjoint operation."""

        class CustomOp(qml.operation.Operation):  # pylint: disable=too-few-public-methods
            """A custom operation."""

            resource_keys = set()

            @property
            def resource_params(self):
                return {}

        @qml.register_resources({qml.H: 1, qml.CNOT: 2, qml.RX: 1, qml.T: 1})
        def custom_decomp(phi, wires):
            qml.H(wires[0])
            qml.CNOT(wires=wires[:2])
            qml.RX(phi, wires=wires[1])
            qml.CNOT(wires=wires[1:3])
            qml.T(wires[2])

        op = qml.adjoint(CustomOp(0.5, wires=[0, 1, 2]))
        graph = DecompositionGraph(
            operations=[op],
            target_gate_set={"Hadamard", "CNOT", "RX", "PhaseShift"},
            alt_decomps={CustomOp: [custom_decomp]},
        )
        # 10 operator nodes: A(CustomOp), A(H), A(CNOT), A(RX), A(T), H, CNOT, RX, A(PhaseShift), PhaseShift
        # 6 decomposition nodes for: A(CustomOp), A(H), A(CNOT), A(RX), A(T), A(PhaseShift)
        assert len(graph._graph.nodes()) == 16
        # 9 edges from ops to decompositions and 6 edges from decompositions to ops.
        assert len(graph._graph.edges()) == 15

        graph.solve()
        with qml.queuing.AnnotatedQueue() as q:
            graph.decomposition(op)(*op.parameters, wires=op.wires, **op.hyperparameters)

        assert q.queue == [
            qml.adjoint(qml.T(2)),
            qml.adjoint(qml.CNOT(wires=[1, 2])),
            qml.adjoint(qml.RX(0.5, wires=1)),
            qml.adjoint(qml.CNOT(wires=[0, 1])),
            qml.adjoint(qml.H(wires=0)),
        ]
        assert graph.resource_estimate(op) == to_resources(
            {qml.H: 1, qml.CNOT: 2, qml.RX: 1, qml.PhaseShift: 1}
        )

    def test_pow_pow(self, _):
        """Tests nested power decompositions."""

        op = qml.pow(qml.pow(qml.H(0), 3), 2)
        graph = DecompositionGraph(operations=[op], target_gate_set={"Hadamard"})
        # 3 operator nodes: Pow(Pow(H)), Pow(H), and H
        # 2 decomposition nodes for Pow(Pow(H)) and Pow(H)
        assert len(graph._graph.nodes()) == 5
        # 2 edges from decompositions to ops and 2 edges from ops to decompositions
        assert len(graph._graph.edges()) == 4

        graph.solve()
        with qml.queuing.AnnotatedQueue() as q:
            graph.decomposition(op)(*op.parameters, wires=op.wires, **op.hyperparameters)

        assert q.queue == [qml.pow(qml.H(0), 6)]
        assert graph.resource_estimate(op) == to_resources({qml.H: 6})

        op2 = qml.pow(qml.H(0), 6)
        with qml.queuing.AnnotatedQueue() as q:
            graph.decomposition(op2)(*op2.parameters, wires=op2.wires, **op2.hyperparameters)

        assert q.queue == [qml.H(0), qml.H(0), qml.H(0), qml.H(0), qml.H(0), qml.H(0)]
        assert graph.resource_estimate(op2) == to_resources({qml.H: 6})
