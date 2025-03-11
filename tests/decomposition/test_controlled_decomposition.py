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

"""Tests the decomposition rules defined for controlled operations."""

import pytest

import pennylane as qml
from pennylane.decomposition.controlled_decomposition import (
    CustomControlledDecomposition,
    controlled_global_phase_decomp,
    controlled_x_decomp,
)
from pennylane.decomposition.resources import CompressedResourceOp, Resources


class TestControlledDecompositionRules:
    """Tests the decomposition rule defined for different controlled operations."""

    def test_single_controlled_global_phase(self):
        """Tests GlobalPhase controlled on a single wire."""

        op = qml.ctrl(
            qml.GlobalPhase(0.5, wires=[0, 1]),
            control=[2],
        )

        with qml.queuing.AnnotatedQueue() as q:
            controlled_global_phase_decomp(*op.parameters, wires=op.wires, **op.hyperparameters)

        assert q.queue == [qml.PhaseShift(-0.5, wires=[2])]
        assert controlled_global_phase_decomp.compute_resources(**op.resource_params) == Resources(
            num_gates=1, gate_counts={CompressedResourceOp(qml.PhaseShift): 1}
        )

    def test_double_controlled_global_phase(self):
        """Tests global phase controlled on two wires."""

        op = qml.ctrl(
            qml.GlobalPhase(0.5, wires=[0, 1]), control=[2, 3], control_values=[False, True]
        )

        with qml.queuing.AnnotatedQueue() as q:
            controlled_global_phase_decomp(*op.parameters, wires=op.wires, **op.hyperparameters)

        assert q.queue == [qml.X(2), qml.ControlledPhaseShift(-0.5, wires=[2, 3]), qml.X(2)]
        assert controlled_global_phase_decomp.compute_resources(**op.resource_params) == Resources(
            num_gates=3,
            gate_counts={
                CompressedResourceOp(qml.X): 2,
                CompressedResourceOp(qml.ControlledPhaseShift): 1,
            },
        )

    def test_multi_controlled_global_phase(self):
        """Tests GlobalPhase controlled on multiple wires."""

        op = qml.ctrl(
            qml.GlobalPhase(0.5, wires=[0, 1]),
            control=[2, 3, 4],
            control_values=[False, True, False],
        )

        with qml.queuing.AnnotatedQueue() as q:
            controlled_global_phase_decomp(*op.parameters, wires=op.wires, **op.hyperparameters)

        assert q.queue == [
            qml.X(2),
            qml.X(4),
            qml.ops.Controlled(qml.PhaseShift(-0.5, wires=[4]), control_wires=[2, 3]),
            qml.X(2),
            qml.X(4),
        ]
        assert controlled_global_phase_decomp.compute_resources(**op.resource_params) == Resources(
            num_gates=5,
            gate_counts={
                CompressedResourceOp(qml.X): 4,
                CompressedResourceOp(
                    qml.ops.Controlled,
                    {
                        "base_class": qml.PhaseShift,
                        "base_params": {},
                        "num_control_wires": 2,
                        "num_zero_control_values": 0,
                        "num_work_wires": 0,
                    },
                ): 1,
            },
        )


class TestControlledX:
    """Tests decompositions of different versions of controlled X gates."""

    def test_single_controlled_x(self):
        """Tests that a single-controlled X decomposes to a CNOT."""

        op = qml.ops.Controlled(qml.X(0), control_wires=[1])
        with qml.queuing.AnnotatedQueue() as q:
            controlled_x_decomp(*op.parameters, wires=op.wires, **op.hyperparameters)

        assert q.queue == [qml.CNOT(wires=[1, 0])]
        assert controlled_x_decomp.compute_resources(**op.resource_params) == Resources(
            num_gates=1, gate_counts={CompressedResourceOp(qml.CNOT): 1}
        )

        op = qml.ops.Controlled(qml.X(0), control_wires=[1], control_values=[0])
        with qml.queuing.AnnotatedQueue() as q:
            controlled_x_decomp(*op.parameters, wires=op.wires, **op.hyperparameters)

        assert q.queue == [qml.PauliX(1), qml.CNOT(wires=[1, 0]), qml.PauliX(1)]
        assert controlled_x_decomp.compute_resources(**op.resource_params) == Resources(
            num_gates=3,
            gate_counts={CompressedResourceOp(qml.CNOT): 1, CompressedResourceOp(qml.PauliX): 2},
        )

    def test_double_controlled_x(self):
        """Tests that a double-controlled X decomposes to a Toffoli gate."""

        op = qml.ops.Controlled(qml.X(0), control_wires=[1, 2])
        with qml.queuing.AnnotatedQueue() as q:
            controlled_x_decomp(*op.parameters, wires=op.wires, **op.hyperparameters)

        assert q.queue == [qml.Toffoli(wires=[1, 2, 0])]
        assert controlled_x_decomp.compute_resources(**op.resource_params) == Resources(
            num_gates=1, gate_counts={CompressedResourceOp(qml.Toffoli): 1}
        )

        op = qml.ops.Controlled(qml.X(0), control_wires=[1, 2], control_values=[0, 1])
        with qml.queuing.AnnotatedQueue() as q:
            controlled_x_decomp(*op.parameters, wires=op.wires, **op.hyperparameters)

        assert q.queue == [qml.PauliX(1), qml.Toffoli(wires=[1, 2, 0]), qml.PauliX(1)]
        assert controlled_x_decomp.compute_resources(**op.resource_params) == Resources(
            num_gates=3,
            gate_counts={CompressedResourceOp(qml.Toffoli): 1, CompressedResourceOp(qml.PauliX): 2},
        )

    def test_multi_controlled_x(self):
        """Tests that a multi-controlled X decomposes to a MultiControlledX gate."""

        op = qml.ops.Controlled(qml.X(0), control_wires=[1, 2, 3], work_wires=[4])
        with qml.queuing.AnnotatedQueue() as q:
            controlled_x_decomp(*op.parameters, wires=op.wires, **op.hyperparameters)

        assert q.queue == [qml.MultiControlledX(wires=[1, 2, 3, 0], work_wires=[4])]
        assert controlled_x_decomp.compute_resources(**op.resource_params) == Resources(
            num_gates=1,
            gate_counts={
                CompressedResourceOp(
                    qml.MultiControlledX,
                    {
                        "num_control_wires": 3,
                        "num_zero_control_values": 0,
                        "num_work_wires": 1,
                    },
                ): 1
            },
        )

        op = qml.ops.Controlled(
            qml.X(0), control_wires=[1, 2, 3], control_values=[0, 1, 0], work_wires=[4]
        )
        with qml.queuing.AnnotatedQueue() as q:
            controlled_x_decomp(*op.parameters, wires=op.wires, **op.hyperparameters)

        assert q.queue == [
            qml.MultiControlledX(wires=[1, 2, 3, 0], control_values=[0, 1, 0], work_wires=[4])
        ]
        assert controlled_x_decomp.compute_resources(**op.resource_params) == Resources(
            num_gates=1,
            gate_counts={
                CompressedResourceOp(
                    qml.MultiControlledX,
                    {
                        "num_control_wires": 3,
                        "num_zero_control_values": 2,
                        "num_work_wires": 1,
                    },
                ): 1,
            },
        )


class TestCustomControlledOperators:
    """Tests decompositions involving custom controlled ops."""

    @pytest.mark.parametrize(
        "op, decomp, resources, custom_op_type",
        [
            # Single-qubit controlled on 0
            (
                qml.ops.Controlled(qml.Z(1), control_wires=[0], control_values=[0]),
                [qml.X(0), qml.CZ(wires=[0, 1]), qml.X(0)],
                Resources(
                    num_gates=3,
                    gate_counts={CompressedResourceOp(qml.CZ): 1, CompressedResourceOp(qml.X): 2},
                ),
                qml.ops.CZ,
            ),
            # Single-qubit controlled on 1
            (
                qml.ops.Controlled(qml.Z(1), control_wires=[0], control_values=[1]),
                [qml.CZ(wires=[0, 1])],
                Resources(
                    num_gates=1,
                    gate_counts={CompressedResourceOp(qml.CZ): 1},
                ),
                qml.ops.CZ,
            ),
            # Two-qubit controlled
            (
                qml.ops.Controlled(qml.Z(2), control_wires=[0, 1], control_values=[1, 0]),
                [qml.X(1), qml.CCZ(wires=[0, 1, 2]), qml.X(1)],
                Resources(
                    num_gates=3,
                    gate_counts={CompressedResourceOp(qml.CCZ): 1, CompressedResourceOp(qml.X): 2},
                ),
                qml.ops.CCZ,
            ),
            # Parametrized controlled
            (
                qml.ops.Controlled(qml.RX(0.5, wires=1), control_wires=[0], control_values=[0]),
                [qml.X(0), qml.CRX(0.5, wires=[0, 1]), qml.X(0)],
                Resources(
                    num_gates=3,
                    gate_counts={CompressedResourceOp(qml.CRX): 1, CompressedResourceOp(qml.X): 2},
                ),
                qml.ops.CRX,
            ),
            # Controlled on two qubits
            (
                qml.ops.Controlled(qml.SWAP(wires=[1, 2]), control_wires=[0], control_values=[0]),
                [qml.X(0), qml.CSWAP(wires=[0, 1, 2]), qml.X(0)],
                Resources(
                    num_gates=3,
                    gate_counts={
                        CompressedResourceOp(qml.CSWAP): 1,
                        CompressedResourceOp(qml.X): 2,
                    },
                ),
                qml.CSWAP,
            ),
        ],
    )
    def test_controlled_to_custom_controlled(self, op, decomp, resources, custom_op_type):
        """Tests that a controlled op decomposes to its corresponding custom op if applicable."""

        decomp_rule = CustomControlledDecomposition(custom_op_type)
        with qml.queuing.AnnotatedQueue() as q:
            decomp_rule(*op.parameters, wires=op.wires, **op.hyperparameters)

        assert q.queue == decomp
        assert decomp_rule.compute_resources(**op.resource_params) == resources
