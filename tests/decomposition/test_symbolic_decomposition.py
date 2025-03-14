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

"""Tests the decomposition rules defined for symbolic operations other than controlled."""

import pytest

import pennylane as qml
from pennylane.decomposition.resources import Resources
from pennylane.decomposition.symbolic_decomposition import (
    AdjointDecomp,
    adjoint_adjoint_decomp,
    adjoint_controlled_decomp,
    has_adjoint_decomp,
)


class TestAdjointDecompositionRules:
    """Tests the decomposition rules defined for the adjoint of operations."""

    def test_adjoint_adjoint(self):
        """Tests that the adjoint of an adjoint cancels out."""

        op = qml.adjoint(qml.adjoint(qml.RX(0.5, wires=0)))

        with qml.queuing.AnnotatedQueue() as q:
            adjoint_adjoint_decomp(*op.parameters, wires=op.wires, **op.hyperparameters)

        assert q.queue == [qml.RX(0.5, wires=0)]
        assert adjoint_adjoint_decomp.compute_resources(**op.resource_params) == Resources(
            {qml.resource_rep(qml.RX): 1}
        )

    @pytest.mark.jax
    def test_adjoint_adjoint_capture(self):
        """Tests that the adjoint of an adjoint works with capture."""

        from pennylane.tape.plxpr_conversion import CollectOpsandMeas

        op = qml.adjoint(qml.adjoint(qml.RX(0.5, wires=0)))

        capture_enabled = qml.capture.enabled()
        qml.capture.enable()

        def circuit():
            adjoint_adjoint_decomp(*op.parameters, wires=op.wires, **op.hyperparameters)

        plxpr = qml.capture.make_plxpr(circuit)()
        collector = CollectOpsandMeas()
        collector.eval(plxpr.jaxpr, plxpr.consts)
        assert collector.state["ops"] == [qml.RX(0.5, wires=0)]

        if not capture_enabled:
            qml.capture.disable()

    def test_adjoint_controlled(self):
        """Tests that the adjoint of a controlled operation is correctly decomposed."""

        op = qml.adjoint(qml.ctrl(qml.U1(0.5, wires=0), control=1))

        with qml.queuing.AnnotatedQueue() as q:
            adjoint_controlled_decomp(*op.parameters, wires=op.wires, **op.hyperparameters)

        assert q.queue == [qml.ctrl(qml.U1(-0.5, wires=0), control=1)]
        assert adjoint_controlled_decomp.compute_resources(**op.resource_params) == Resources(
            {
                qml.controlled_resource_rep(
                    qml.U1, {}, num_control_wires=1, num_zero_control_values=0, num_work_wires=0
                ): 1
            }
        )

    def test_adjoint_controlled_x(self):
        """Tests the adjoint of controlled X operations are correctly decomposed."""

        op1 = qml.adjoint(qml.ops.Controlled(qml.X(1), control_wires=[0]))
        op2 = qml.adjoint(qml.ops.Controlled(qml.X(2), control_wires=[0, 1]))
        op3 = qml.adjoint(qml.ops.Controlled(qml.X(3), control_wires=[0, 1, 2]))

        with qml.queuing.AnnotatedQueue() as q:
            adjoint_controlled_decomp(*op1.parameters, wires=op1.wires, **op1.hyperparameters)
            adjoint_controlled_decomp(*op2.parameters, wires=op2.wires, **op2.hyperparameters)
            adjoint_controlled_decomp(*op3.parameters, wires=op3.wires, **op3.hyperparameters)

        assert q.queue == [
            qml.CNOT(wires=[0, 1]),
            qml.Toffoli(wires=[0, 1, 2]),
            qml.MultiControlledX(wires=[0, 1, 2, 3]),
        ]

        assert adjoint_controlled_decomp.compute_resources(**op1.resource_params) == Resources(
            {qml.resource_rep(qml.CNOT): 1}
        )
        assert adjoint_controlled_decomp.compute_resources(**op2.resource_params) == Resources(
            {qml.resource_rep(qml.Toffoli): 1}
        )
        assert adjoint_controlled_decomp.compute_resources(**op3.resource_params) == Resources(
            {
                qml.resource_rep(
                    qml.MultiControlledX,
                    num_control_wires=3,
                    num_zero_control_values=0,
                    num_work_wires=0,
                ): 1
            }
        )

    def test_adjoint_custom(self):
        """Tests that the adjoint of an operation that has adjoint is correctly decomposed."""

        op1 = qml.adjoint(qml.H(0))
        op2 = qml.adjoint(qml.RX(0.5, wires=0))

        with qml.queuing.AnnotatedQueue() as q:
            has_adjoint_decomp(*op1.parameters, wires=op1.wires, **op1.hyperparameters)
            has_adjoint_decomp(*op2.parameters, wires=op2.wires, **op2.hyperparameters)

        assert q.queue == [qml.H(0), qml.RX(-0.5, wires=0)]
        assert has_adjoint_decomp.compute_resources(**op1.resource_params) == Resources(
            {qml.resource_rep(qml.H): 1}
        )
        assert has_adjoint_decomp.compute_resources(**op2.resource_params) == Resources(
            {qml.resource_rep(qml.RX): 1}
        )

    def test_adjoint_general(self):
        """Tests the adjoint of a general operator can be correctly decomposed."""

        class CustomOp(qml.operation.Operator):  # pylint: disable=too-few-public-methods

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
        rule = AdjointDecomp(custom_decomp)

        with qml.queuing.AnnotatedQueue() as q:
            rule(*op.parameters, wires=op.wires, **op.hyperparameters)

        assert q.queue == [
            qml.adjoint(qml.T(2)),
            qml.adjoint(qml.CNOT(wires=[1, 2])),
            qml.adjoint(qml.RX(0.5, wires=1)),
            qml.adjoint(qml.CNOT(wires=[0, 1])),
            qml.adjoint(qml.H(wires=0)),
        ]

        assert rule.compute_resources(**op.resource_params) == Resources(
            {
                qml.adjoint_resource_rep(qml.T): 1,
                qml.adjoint_resource_rep(qml.CNOT): 2,
                qml.adjoint_resource_rep(qml.RX): 1,
                qml.adjoint_resource_rep(qml.H): 1,
            }
        )
