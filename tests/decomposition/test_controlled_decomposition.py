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

import pennylane as qml
from pennylane.decomposition.controlled_decomposition import controlled_global_phase_decomp
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
