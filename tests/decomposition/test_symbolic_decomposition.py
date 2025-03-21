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
from pennylane.decomposition.resources import Resources, pow_resource_rep
from pennylane.decomposition.symbolic_decomposition import (
    AdjointDecomp,
    adjoint_adjoint_decomp,
    adjoint_controlled_decomp,
    adjoint_pow_decomp,
    pow_decomp,
    pow_pow_decomp,
    same_type_adjoint_decomp,
)
from tests.decomposition.conftest import to_resources


class TestAdjointDecompositionRules:
    """Tests the decomposition rules defined for the adjoint of operations."""

    def test_adjoint_adjoint(self):
        """Tests that the adjoint of an adjoint cancels out."""

        op = qml.adjoint(qml.adjoint(qml.RX(0.5, wires=0)))

        with qml.queuing.AnnotatedQueue() as q:
            adjoint_adjoint_decomp(*op.parameters, wires=op.wires, **op.hyperparameters)

        assert q.queue == [qml.RX(0.5, wires=0)]
        assert adjoint_adjoint_decomp.compute_resources(**op.resource_params) == to_resources(
            {qml.RX: 1}
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

        op1 = qml.adjoint(qml.ctrl(qml.U1(0.5, wires=0), control=1))
        op2 = qml.adjoint(qml.ops.Controlled(qml.RX(0.5, wires=0), control_wires=[1]))

        with qml.queuing.AnnotatedQueue() as q:
            adjoint_controlled_decomp(*op1.parameters, wires=op1.wires, **op1.hyperparameters)
            adjoint_controlled_decomp(*op2.parameters, wires=op2.wires, **op2.hyperparameters)

        assert q.queue == [qml.ctrl(qml.U1(-0.5, wires=0), control=1), qml.CRX(-0.5, wires=[1, 0])]
        assert adjoint_controlled_decomp.compute_resources(**op1.resource_params) == Resources(
            {
                qml.decomposition.controlled_resource_rep(
                    qml.U1, {}, num_control_wires=1, num_zero_control_values=0, num_work_wires=0
                ): 1,
            }
        )
        assert adjoint_controlled_decomp.compute_resources(**op2.resource_params) == Resources(
            {qml.resource_rep(qml.CRX): 1}
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

    def test_same_type_adjoint(self):
        """Tests that the adjoint of an operation that has adjoint of same
        type is correctly decomposed."""

        op1 = qml.adjoint(qml.H(0))
        op2 = qml.adjoint(qml.RX(0.5, wires=0))

        with qml.queuing.AnnotatedQueue() as q:
            same_type_adjoint_decomp(*op1.parameters, wires=op1.wires, **op1.hyperparameters)
            same_type_adjoint_decomp(*op2.parameters, wires=op2.wires, **op2.hyperparameters)

        assert q.queue == [qml.H(0), qml.RX(-0.5, wires=0)]
        assert same_type_adjoint_decomp.compute_resources(**op1.resource_params) == to_resources(
            {qml.H: 1}
        )
        assert same_type_adjoint_decomp.compute_resources(**op2.resource_params) == to_resources(
            {qml.RX: 1}
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
                qml.decomposition.adjoint_resource_rep(qml.T): 1,
                qml.decomposition.adjoint_resource_rep(qml.CNOT): 2,
                qml.decomposition.adjoint_resource_rep(qml.RX): 1,
                qml.decomposition.adjoint_resource_rep(qml.H): 1,
            }
        )

    def test_adjoint_pow(self):
        """Tests decomposing the adjoint of a Pow of an operator that has adjoint."""

        op = qml.adjoint(qml.pow(qml.H(0), 3))
        with qml.queuing.AnnotatedQueue() as q:
            adjoint_pow_decomp(*op.parameters, wires=op.wires, **op.hyperparameters)

        assert q.queue == [qml.pow(qml.H(0), 3)]
        assert adjoint_pow_decomp.compute_resources(**op.resource_params) == to_resources(
            {pow_resource_rep(qml.H, {}, 3): 1}
        )


class TestPowDecomposition:
    """Tests the decomposition rule defined for Pow."""

    def test_pow_pow(self):
        """Test the decomposition rule for nested powers."""

        op = qml.pow(qml.pow(qml.H(0), 3), 2)
        with qml.queuing.AnnotatedQueue() as q:
            pow_pow_decomp(*op.parameters, wires=op.wires, **op.hyperparameters)

        assert q.queue == [qml.pow(qml.H(0), 6)]
        assert pow_pow_decomp.compute_resources(**op.resource_params) == to_resources(
            {pow_resource_rep(qml.H, {}, 6): 1}
        )

    def test_pow_general(self):
        """Tests repeating the same op z number of times."""

        op = qml.pow(qml.H(0), 3)
        with qml.queuing.AnnotatedQueue() as q:
            pow_decomp(*op.parameters, wires=op.wires, **op.hyperparameters)

        assert q.queue == [qml.H(0), qml.H(0), qml.H(0)]
        assert pow_decomp.compute_resources(**op.resource_params) == to_resources({qml.H: 3})

    @pytest.mark.jax
    def test_pow_general_capture(self):
        """Tests that the general pow decomposition works with capture."""

        from pennylane.tape.plxpr_conversion import CollectOpsandMeas

        op = qml.pow(qml.H(0), 3)

        capture_enabled = qml.capture.enabled()
        qml.capture.enable()

        def circuit():
            pow_decomp(*op.parameters, wires=op.wires, **op.hyperparameters)

        plxpr = qml.capture.make_plxpr(circuit)()
        collector = CollectOpsandMeas()
        collector.eval(plxpr.jaxpr, plxpr.consts)
        assert collector.state["ops"] == [qml.H(0), qml.H(0), qml.H(0)]

        if not capture_enabled:
            qml.capture.disable()

    def test_non_integer_pow_not_implemented(self):
        """Tests that NotImplementedError is raised when z isn't a positive integer."""

        op = qml.pow(qml.H(0), 0.5)
        with pytest.raises(NotImplementedError, match="Non-integer or negative powers"):
            pow_decomp.compute_resources(**op.resource_params)
        op = qml.pow(qml.H(0), -1)
        with pytest.raises(NotImplementedError, match="Non-integer or negative powers"):
            pow_decomp.compute_resources(**op.resource_params)
