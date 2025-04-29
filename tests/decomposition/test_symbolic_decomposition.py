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
from pennylane import queuing
from pennylane.decomposition import DecompositionNotApplicable
from pennylane.decomposition.resources import (
    Resources,
    adjoint_resource_rep,
    pow_resource_rep,
    resource_rep,
)
from pennylane.decomposition.symbolic_decomposition import (
    adjoint_rotation,
    cancel_adjoint,
    flip_pow_adjoint,
    make_adjoint_decomp,
    merge_powers,
    pow_of_self_adjoint,
    pow_rotation,
    repeat_pow_base,
    self_adjoint,
)
from tests.decomposition.conftest import to_resources


@pytest.mark.unit
class TestAdjointDecompositionRules:
    """Tests the decomposition rules defined for the adjoint of operations."""

    def test_cancel_adjoint(self):
        """Tests that the adjoint of an adjoint cancels out."""

        op = qml.adjoint(qml.adjoint(qml.RX(0.5, wires=0)))

        with qml.queuing.AnnotatedQueue() as q:
            cancel_adjoint(*op.parameters, wires=op.wires, **op.hyperparameters)

        assert q.queue == [qml.RX(0.5, wires=0)]
        assert cancel_adjoint.compute_resources(**op.resource_params) == to_resources({qml.RX: 1})

    @pytest.mark.jax
    def test_cancel_adjoint_capture(self):
        """Tests that the adjoint of an adjoint works with capture."""

        from pennylane.tape.plxpr_conversion import CollectOpsandMeas

        op = qml.adjoint(qml.adjoint(qml.RX(0.5, wires=0)))

        capture_enabled = qml.capture.enabled()
        qml.capture.enable()

        def circuit():
            cancel_adjoint(*op.parameters, wires=op.wires, **op.hyperparameters)

        plxpr = qml.capture.make_plxpr(circuit)()
        collector = CollectOpsandMeas()
        collector.eval(plxpr.jaxpr, plxpr.consts)
        assert collector.state["ops"] == [qml.RX(0.5, wires=0)]

        if not capture_enabled:
            qml.capture.disable()

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
        rule = make_adjoint_decomp(custom_decomp)

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
                adjoint_resource_rep(qml.T): 1,
                adjoint_resource_rep(qml.CNOT): 2,
                adjoint_resource_rep(qml.RX): 1,
                adjoint_resource_rep(qml.H): 1,
            }
        )

    def test_adjoint_rotation(self):
        """Tests the adjoint_rotation decomposition."""

        class CustomOp(qml.operation.Operator):  # pylint: disable=too-few-public-methods

            resource_keys = set()

            @property
            def resource_params(self):
                return {}

        op = qml.adjoint(CustomOp(0.5, wires=[0, 1, 2]))
        with queuing.AnnotatedQueue() as q:
            adjoint_rotation(*op.parameters, wires=op.wires, **op.hyperparameters)

        assert q.queue == [CustomOp(-0.5, wires=[0, 1, 2])]
        assert adjoint_rotation.compute_resources(**op.resource_params) == Resources(
            {resource_rep(CustomOp): 1}
        )

    def test_self_adjoint(self):
        """Tests the self_adjoint decomposition."""

        class CustomOp(qml.operation.Operator):  # pylint: disable=too-few-public-methods

            resource_keys = set()

            @property
            def resource_params(self):
                return {}

        op = qml.adjoint(CustomOp(0.5, wires=[0, 1, 2]))
        with queuing.AnnotatedQueue() as q:
            self_adjoint(*op.parameters, wires=op.wires, **op.hyperparameters)

        assert q.queue == [CustomOp(0.5, wires=[0, 1, 2])]
        assert self_adjoint.compute_resources(**op.resource_params) == Resources(
            {resource_rep(CustomOp): 1}
        )


@pytest.mark.unit
class TestPowDecomposition:
    """Tests the decomposition rule defined for Pow."""

    def test_merge_powers(self):
        """Test the decomposition rule for nested powers."""

        op = qml.pow(qml.pow(qml.H(0), 3), 2)
        with qml.queuing.AnnotatedQueue() as q:
            merge_powers(*op.parameters, wires=op.wires, **op.hyperparameters)

        assert q.queue == [qml.pow(qml.H(0), 6)]
        assert merge_powers.compute_resources(**op.resource_params) == to_resources(
            {pow_resource_rep(qml.H, {}, 6): 1}
        )

    def test_repeat_pow_base(self):
        """Tests repeating the same op z number of times."""

        op = qml.pow(qml.H(0), 3)
        with qml.queuing.AnnotatedQueue() as q:
            repeat_pow_base(*op.parameters, wires=op.wires, **op.hyperparameters)

        assert q.queue == [qml.H(0), qml.H(0), qml.H(0)]
        assert repeat_pow_base.compute_resources(**op.resource_params) == to_resources({qml.H: 3})

    @pytest.mark.jax
    def test_repeat_pow_base_capture(self):
        """Tests that the general pow decomposition works with capture."""

        from pennylane.tape.plxpr_conversion import CollectOpsandMeas

        op = qml.pow(qml.H(0), 3)

        capture_enabled = qml.capture.enabled()
        qml.capture.enable()

        def circuit():
            repeat_pow_base(*op.parameters, wires=op.wires, **op.hyperparameters)

        plxpr = qml.capture.make_plxpr(circuit)()
        collector = CollectOpsandMeas()
        collector.eval(plxpr.jaxpr, plxpr.consts)
        assert collector.state["ops"] == [qml.H(0), qml.H(0), qml.H(0)]

        if not capture_enabled:
            qml.capture.disable()

    def test_non_integer_pow_not_applicable(self):
        """Tests that DecompositionNotApplicable is raised when z isn't a positive integer."""

        op = qml.pow(qml.H(0), 0.5)
        with pytest.raises(DecompositionNotApplicable):
            repeat_pow_base.compute_resources(**op.resource_params)
        op = qml.pow(qml.H(0), -1)
        with pytest.raises(DecompositionNotApplicable):
            repeat_pow_base.compute_resources(**op.resource_params)

    def test_flip_pow_adjoint(self):
        """Tests the flip_pow_adjoint decomposition."""

        class CustomOp(qml.operation.Operator):  # pylint: disable=too-few-public-methods

            resource_keys = set()

            @property
            def resource_params(self):
                return {}

        op = qml.pow(qml.adjoint(CustomOp(0.5, wires=[0, 1, 2])), 2)
        with queuing.AnnotatedQueue() as q:
            flip_pow_adjoint(*op.parameters, wires=op.wires, **op.hyperparameters)

        assert q.queue == [qml.adjoint(qml.pow(CustomOp(0.5, wires=[0, 1, 2]), 2))]
        assert flip_pow_adjoint.compute_resources(**op.resource_params) == Resources(
            {
                adjoint_resource_rep(
                    qml.ops.Pow, {"base_class": CustomOp, "base_params": {}, "z": 2}
                ): 1
            }
        )

    def test_pow_of_self_adjoint(self):
        """Tests the pow_of_self_adjoint decomposition."""

        class CustomOp(qml.operation.Operator):  # pylint: disable=too-few-public-methods

            resource_keys = set()

            @property
            def resource_params(self):
                return {}

        op1 = qml.pow(CustomOp(wires=[0, 1, 2]), 1)
        op2 = qml.pow(CustomOp(wires=[0, 1, 2]), 2)
        op3 = qml.pow(CustomOp(wires=[0, 1, 2]), 3)
        op4 = qml.pow(CustomOp(wires=[0, 1, 2]), 4)

        with qml.queuing.AnnotatedQueue() as q:
            pow_of_self_adjoint(*op1.parameters, wires=op1.wires, **op1.hyperparameters)
            pow_of_self_adjoint(*op2.parameters, wires=op2.wires, **op2.hyperparameters)
            pow_of_self_adjoint(*op3.parameters, wires=op3.wires, **op3.hyperparameters)
            pow_of_self_adjoint(*op4.parameters, wires=op4.wires, **op4.hyperparameters)

        assert q.queue == [CustomOp(wires=[0, 1, 2]), CustomOp(wires=[0, 1, 2])]
        assert pow_of_self_adjoint.compute_resources(**op1.resource_params) == Resources(
            {resource_rep(CustomOp): 1}
        )
        assert pow_of_self_adjoint.compute_resources(**op3.resource_params) == Resources(
            {resource_rep(CustomOp): 1}
        )
        assert pow_of_self_adjoint.compute_resources(**op2.resource_params) == Resources()
        assert pow_of_self_adjoint.compute_resources(**op4.resource_params) == Resources()

        with pytest.raises(DecompositionNotApplicable):
            pow_of_self_adjoint.compute_resources(CustomOp, {}, z=0.5)

    def test_pow_rotations(self):
        """Tests the pow_rotations decomposition."""

        class CustomOp(qml.operation.Operator):  # pylint: disable=too-few-public-methods

            resource_keys = set()

            @property
            def resource_params(self):
                return {}

        op = qml.pow(CustomOp(0.3, wires=[0, 1, 2]), 2.5)
        with queuing.AnnotatedQueue() as q:
            pow_rotation(*op.parameters, wires=op.wires, **op.hyperparameters)

        assert q.queue == [CustomOp(0.3 * 2.5, wires=[0, 1, 2])]
        assert pow_rotation.compute_resources(**op.resource_params) == Resources(
            {resource_rep(CustomOp): 1}
        )
