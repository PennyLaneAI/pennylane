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
