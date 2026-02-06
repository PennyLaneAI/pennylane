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

import pennylane as qp
from pennylane import queuing
from pennylane.decomposition.resources import (
    Resources,
    adjoint_resource_rep,
    pow_resource_rep,
    resource_rep,
)
from pennylane.decomposition.symbolic_decomposition import (
    adjoint_rotation,
    cancel_adjoint,
    controlled_resource_rep,
    ctrl_single_work_wire,
    flip_control_adjoint,
    flip_pow_adjoint,
    make_adjoint_decomp,
    make_controlled_decomp,
    merge_powers,
    pow_involutory,
    pow_rotation,
    repeat_pow_base,
    self_adjoint,
    to_controlled_qubit_unitary,
)

# pylint: disable=no-name-in-module
from tests.decomposition.conftest import to_resources


@pytest.mark.unit
class TestAdjointDecompositionRules:
    """Tests the decomposition rules defined for the adjoint of operations."""

    def test_cancel_adjoint(self):
        """Tests that the adjoint of an adjoint cancels out."""

        op = qp.adjoint(qp.adjoint(qp.RX(0.5, wires=0)))

        with qp.queuing.AnnotatedQueue() as q:
            cancel_adjoint(*op.parameters, wires=op.wires, **op.hyperparameters)

        assert q.queue == [qp.RX(0.5, wires=0)]
        assert cancel_adjoint.compute_resources(**op.resource_params) == to_resources({qp.RX: 1})

    @pytest.mark.capture
    def test_cancel_adjoint_capture(self):
        """Tests that the adjoint of an adjoint works with capture."""

        from pennylane.tape.plxpr_conversion import CollectOpsandMeas

        op = qp.adjoint(qp.adjoint(qp.RX(0.5, wires=0)))

        def circuit():
            cancel_adjoint(*op.parameters, wires=op.wires, **op.hyperparameters)

        plxpr = qp.capture.make_plxpr(circuit)()
        collector = CollectOpsandMeas()
        collector.eval(plxpr.jaxpr, plxpr.consts)
        assert collector.state["ops"] == [qp.RX(0.5, wires=0)]

    def test_adjoint_general(self):
        """Tests the adjoint of a general operator can be correctly decomposed."""

        class CustomOp(qp.operation.Operator):  # pylint: disable=too-few-public-methods

            resource_keys = set()

            @property
            def resource_params(self):
                return {}

        @qp.register_resources({qp.H: 1, qp.CNOT: 2, qp.RX: 1, qp.T: 1})
        def _custom_decomp(phi, wires):
            qp.H(wires[0])
            qp.CNOT(wires=wires[:2])
            qp.RX(phi, wires=wires[1])
            qp.CNOT(wires=wires[1:3])
            qp.T(wires[2])

        op = qp.adjoint(CustomOp(0.5, wires=[0, 1, 2]))
        rule = make_adjoint_decomp(_custom_decomp)

        with qp.queuing.AnnotatedQueue() as q:
            rule(*op.parameters, wires=op.wires, **op.hyperparameters)

        assert q.queue == [
            qp.adjoint(qp.T(2)),
            qp.adjoint(qp.CNOT(wires=[1, 2])),
            qp.adjoint(qp.RX(0.5, wires=1)),
            qp.adjoint(qp.CNOT(wires=[0, 1])),
            qp.adjoint(qp.H(wires=0)),
        ]

        assert rule.compute_resources(**op.resource_params) == Resources(
            {
                adjoint_resource_rep(qp.T): 1,
                adjoint_resource_rep(qp.CNOT): 2,
                adjoint_resource_rep(qp.RX): 1,
                adjoint_resource_rep(qp.H): 1,
            }
        )

    def test_adjoint_rotation(self):
        """Tests the adjoint_rotation decomposition."""

        class CustomOp(qp.operation.Operator):  # pylint: disable=too-few-public-methods

            resource_keys = set()

            @property
            def resource_params(self):
                return {}

        op = qp.adjoint(CustomOp(0.5, wires=[0, 1, 2]))
        with queuing.AnnotatedQueue() as q:
            adjoint_rotation(*op.parameters, wires=op.wires, **op.hyperparameters)

        assert q.queue == [CustomOp(-0.5, wires=[0, 1, 2])]
        assert adjoint_rotation.compute_resources(**op.resource_params) == Resources(
            {resource_rep(CustomOp): 1}
        )

    def test_self_adjoint(self):
        """Tests the self_adjoint decomposition."""

        class CustomOp(qp.operation.Operator):  # pylint: disable=too-few-public-methods

            resource_keys = set()

            @property
            def resource_params(self):
                return {}

        op = qp.adjoint(CustomOp(0.5, wires=[0, 1, 2]))
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

        op = qp.pow(qp.pow(qp.H(0), 3), 2)
        with qp.queuing.AnnotatedQueue() as q:
            merge_powers(*op.parameters, wires=op.wires, **op.hyperparameters)

        assert q.queue == [qp.pow(qp.H(0), 6)]
        assert merge_powers.compute_resources(**op.resource_params) == to_resources(
            {pow_resource_rep(qp.H, {}, 6): 1}
        )

    def test_repeat_pow_base(self):
        """Tests repeating the same op z number of times."""

        op = qp.pow(qp.H(0), 3)
        with qp.queuing.AnnotatedQueue() as q:
            repeat_pow_base(*op.parameters, wires=op.wires, **op.hyperparameters)

        assert q.queue == [qp.H(0), qp.H(0), qp.H(0)]
        assert repeat_pow_base.compute_resources(**op.resource_params) == to_resources({qp.H: 3})

    @pytest.mark.capture
    def test_repeat_pow_base_capture(self):
        """Tests that the general pow decomposition works with capture."""

        from pennylane.tape.plxpr_conversion import CollectOpsandMeas

        op = qp.pow(qp.H(0), 3)

        def circuit():
            repeat_pow_base(*op.parameters, wires=op.wires, **op.hyperparameters)

        plxpr = qp.capture.make_plxpr(circuit)()
        collector = CollectOpsandMeas()
        collector.eval(plxpr.jaxpr, plxpr.consts)
        assert collector.state["ops"] == [qp.H(0), qp.H(0), qp.H(0)]

    def test_non_integer_pow_not_applicable(self):
        """Tests that is_applicable returns False when z isn't a positive integer."""

        op = qp.pow(qp.H(0), 0.5)
        assert not repeat_pow_base.is_applicable(**op.resource_params)
        op = qp.pow(qp.H(0), -1)
        assert not repeat_pow_base.is_applicable(**op.resource_params)

    def test_flip_pow_adjoint(self):
        """Tests the flip_pow_adjoint decomposition."""

        class CustomOp(qp.operation.Operator):  # pylint: disable=too-few-public-methods

            resource_keys = set()

            @property
            def resource_params(self):
                return {}

        op = qp.pow(qp.adjoint(CustomOp(0.5, wires=[0, 1, 2])), 2)
        with queuing.AnnotatedQueue() as q:
            flip_pow_adjoint(*op.parameters, wires=op.wires, **op.hyperparameters)

        assert q.queue == [qp.adjoint(qp.pow(CustomOp(0.5, wires=[0, 1, 2]), 2))]
        assert flip_pow_adjoint.compute_resources(**op.resource_params) == Resources(
            {
                adjoint_resource_rep(
                    qp.ops.Pow, {"base_class": CustomOp, "base_params": {}, "z": 2}
                ): 1
            }
        )

    def test_pow_involutory(self):
        """Tests the pow_involutory decomposition."""

        class CustomOp(qp.operation.Operator):  # pylint: disable=too-few-public-methods

            resource_keys = set()

            @property
            def resource_params(self):
                return {}

        op1 = qp.pow(CustomOp(wires=[0, 1, 2]), 1)
        op2 = qp.pow(CustomOp(wires=[0, 1, 2]), 2)
        op3 = qp.pow(CustomOp(wires=[0, 1, 2]), 3)
        op4 = qp.pow(CustomOp(wires=[0, 1, 2]), 4)
        op5 = qp.pow(CustomOp(wires=[0, 1, 2]), 4.5)

        with qp.queuing.AnnotatedQueue() as q:
            pow_involutory(*op1.parameters, wires=op1.wires, **op1.hyperparameters)
            pow_involutory(*op2.parameters, wires=op2.wires, **op2.hyperparameters)
            pow_involutory(*op3.parameters, wires=op3.wires, **op3.hyperparameters)
            pow_involutory(*op4.parameters, wires=op4.wires, **op4.hyperparameters)
            pow_involutory(*op5.parameters, wires=op5.wires, **op5.hyperparameters)

        assert q.queue == [
            CustomOp(wires=[0, 1, 2]),
            CustomOp(wires=[0, 1, 2]),
            qp.pow(CustomOp(wires=[0, 1, 2]), 0.5),
        ]
        assert pow_involutory.compute_resources(**op1.resource_params) == Resources(
            {resource_rep(CustomOp): 1}
        )
        assert pow_involutory.compute_resources(**op3.resource_params) == Resources(
            {resource_rep(CustomOp): 1}
        )
        assert pow_involutory.compute_resources(**op2.resource_params) == Resources()
        assert pow_involutory.compute_resources(**op4.resource_params) == Resources()
        assert pow_involutory.compute_resources(**op5.resource_params) == Resources(
            {pow_resource_rep(CustomOp, {}, 0.5): 1}
        )

        assert not pow_involutory.is_applicable(CustomOp, {}, z=0.5)

    def test_pow_rotations(self):
        """Tests the pow_rotations decomposition."""

        class CustomOp(qp.operation.Operator):  # pylint: disable=too-few-public-methods

            resource_keys = set()

            @property
            def resource_params(self):
                return {}

        op = qp.pow(CustomOp(0.3, wires=[0, 1, 2]), 2.5)
        with queuing.AnnotatedQueue() as q:
            pow_rotation(*op.parameters, wires=op.wires, **op.hyperparameters)

        assert q.queue == [CustomOp(0.3 * 2.5, wires=[0, 1, 2])]
        assert pow_rotation.compute_resources(**op.resource_params) == Resources(
            {resource_rep(CustomOp): 1}
        )


class CustomMultiQubitOp(qp.operation.Operation):  # pylint: disable=too-few-public-methods
    """A custom op."""

    resource_keys = {"num_wires"}

    @property
    def resource_params(self):
        return {"num_wires": len(self.wires)}


def _custom_resource(num_wires):
    return {
        qp.X: 1,
        qp.CNOT: 1,
        qp.Toffoli: 1,
        qp.resource_rep(
            qp.MultiControlledX,
            num_control_wires=3,
            num_zero_control_values=1,
            num_work_wires=1,
            work_wire_type="zeroed",
        ): 1,
        qp.RX: 1,
        qp.Rot: 1,
        qp.CRZ: 1,
        resource_rep(qp.MultiRZ, num_wires=num_wires): 1,
        controlled_resource_rep(
            qp.MultiRZ,
            {"num_wires": num_wires - 1},
            num_control_wires=1,
        ): 1,
        resource_rep(qp.PauliRot, pauli_word="XYX"): 1,
        qp.Z: 1,
        qp.CZ: 1,
    }


@qp.register_resources(_custom_resource)
def custom_decomp(*params, wires, **_):
    qp.X(wires[0])
    qp.CNOT(wires=wires[:2])
    qp.Toffoli(wires=wires[:3])
    qp.MultiControlledX(wires=wires[:4], control_values=[1, 0, 1], work_wires=[4])
    qp.RX(params[0], wires=wires[0])
    qp.Rot(params[0], params[1], params[2], wires=wires[0])
    qp.CRZ(params[0], wires=wires[:2])
    qp.MultiRZ(params[0], wires=wires)
    qp.ctrl(qp.MultiRZ(params[0], wires=wires[1:]), control=wires[0])
    qp.PauliRot(params[0], "XYX", wires=wires[:3])
    qp.Z(wires[0])
    qp.CZ(wires=wires[:2])


@pytest.mark.unit
class TestControlledDecomposition:
    """Tests applying control on the decomposition of the base operator."""

    def test_single_control_wire(self):
        """Tests a single control wire."""

        rule = make_controlled_decomp(custom_decomp)

        # Single control wire controlled on 1
        op = qp.ctrl(
            CustomMultiQubitOp(0.5, 0.6, 0.7, wires=[0, 1, 2, 3, 4, 5]), control=[6], work_wires=[7]
        )
        with qp.queuing.AnnotatedQueue() as q:
            rule(*op.parameters, wires=op.wires, **op.hyperparameters)

        expected_ops = [
            qp.CNOT(wires=[6, 0]),
            qp.Toffoli(wires=[6, 0, 1]),
            qp.MultiControlledX(wires=[6, 0, 1, 2], work_wires=[7]),
            qp.MultiControlledX(
                wires=[6, 0, 1, 2, 3], control_values=[1, 1, 0, 1], work_wires=[7, 4]
            ),
            qp.CRX(0.5, wires=[6, 0]),
            qp.CRot(0.5, 0.6, 0.7, wires=[6, 0]),
            qp.ops.Controlled(qp.RZ(0.5, wires=[1]), control_wires=[6, 0], work_wires=[7]),
            qp.ops.Controlled(
                qp.MultiRZ(0.5, wires=[0, 1, 2, 3, 4, 5]),
                control_wires=[6],
                work_wires=[7],
            ),
            qp.ops.Controlled(
                qp.MultiRZ(0.5, wires=[1, 2, 3, 4, 5]),
                control_wires=[6, 0],
                work_wires=[7],
            ),
            qp.ops.Controlled(
                qp.PauliRot(0.5, "XYX", wires=[0, 1, 2]),
                control_wires=[6],
                work_wires=[7],
            ),
            qp.CZ(wires=[6, 0]),
            qp.CCZ(wires=[6, 0, 1]),
        ]
        for actual, expected in zip(q.queue, expected_ops, strict=True):
            qp.assert_equal(actual, expected)

        actual_resources = rule.compute_resources(**op.resource_params)
        assert actual_resources == Resources(
            {
                qp.resource_rep(qp.CNOT): 1,
                qp.resource_rep(qp.Toffoli): 1,
                qp.resource_rep(
                    qp.MultiControlledX,
                    num_control_wires=3,
                    num_zero_control_values=0,
                    num_work_wires=1,
                    work_wire_type="borrowed",
                ): 1,
                qp.resource_rep(
                    qp.MultiControlledX,
                    num_control_wires=4,
                    num_zero_control_values=1,
                    num_work_wires=2,
                    work_wire_type="borrowed",
                ): 1,
                qp.resource_rep(qp.CRX): 1,
                qp.resource_rep(qp.CRot): 1,
                qp.decomposition.controlled_resource_rep(
                    qp.RZ, {}, num_control_wires=2, num_work_wires=1
                ): 1,
                qp.decomposition.controlled_resource_rep(
                    qp.MultiRZ,
                    {"num_wires": 6},
                    num_control_wires=1,
                    num_work_wires=1,
                ): 1,
                qp.decomposition.controlled_resource_rep(
                    qp.MultiRZ,
                    {"num_wires": 5},
                    num_control_wires=2,
                    num_work_wires=1,
                ): 1,
                qp.decomposition.controlled_resource_rep(
                    qp.PauliRot,
                    {"pauli_word": "XYX"},
                    num_control_wires=1,
                    num_work_wires=1,
                ): 1,
                qp.resource_rep(qp.CZ): 1,
                qp.resource_rep(qp.CCZ): 1,
            }
        )

    def test_double_control_wire(self):
        """Tests two control wires."""

        rule = make_controlled_decomp(custom_decomp)

        # Single control wire controlled on 1
        op = qp.ctrl(
            CustomMultiQubitOp(0.5, 0.6, 0.7, wires=[0, 1, 2, 3, 4, 5]),
            control=[6, 7],
            control_values=[False, True],
            work_wires=[8],
        )
        with qp.queuing.AnnotatedQueue() as q:
            rule(*op.parameters, wires=op.wires, **op.hyperparameters)

        expected_ops = [
            qp.X(6),
            qp.Toffoli(wires=[6, 7, 0]),
            qp.MultiControlledX(wires=[6, 7, 0, 1], work_wires=[8]),
            qp.MultiControlledX(wires=[6, 7, 0, 1, 2], work_wires=[8]),
            qp.MultiControlledX(
                wires=[6, 7, 0, 1, 2, 3],
                control_values=[1, 1, 1, 0, 1],
                work_wires=[8, 4],
            ),
            qp.ops.Controlled(qp.RX(0.5, wires=0), control_wires=[6, 7], work_wires=[8]),
            qp.ops.Controlled(
                qp.Rot(0.5, 0.6, 0.7, wires=0),
                control_wires=[6, 7],
                work_wires=[8],
            ),
            qp.ops.Controlled(qp.RZ(0.5, wires=[1]), control_wires=[6, 7, 0], work_wires=[8]),
            qp.ops.Controlled(
                qp.MultiRZ(0.5, wires=[0, 1, 2, 3, 4, 5]),
                control_wires=[6, 7],
                work_wires=[8],
            ),
            qp.ops.Controlled(
                qp.MultiRZ(0.5, wires=[1, 2, 3, 4, 5]),
                control_wires=[6, 7, 0],
                work_wires=[8],
            ),
            qp.ops.Controlled(
                qp.PauliRot(0.5, "XYX", wires=[0, 1, 2]),
                control_wires=[6, 7],
                work_wires=[8],
            ),
            qp.CCZ(wires=[6, 7, 0]),
            qp.ops.Controlled(qp.Z(1), control_wires=[6, 7, 0], work_wires=[8]),
            qp.X(6),
        ]

        for actual, expected in zip(q.queue, expected_ops, strict=True):
            qp.assert_equal(actual, expected)

        actual_resources = rule.compute_resources(**op.resource_params)
        assert actual_resources == Resources(
            {
                qp.resource_rep(qp.X): 2,
                qp.resource_rep(qp.Toffoli): 1,
                qp.resource_rep(
                    qp.MultiControlledX,
                    num_control_wires=3,
                    num_zero_control_values=0,
                    num_work_wires=1,
                    work_wire_type="borrowed",
                ): 1,
                qp.resource_rep(
                    qp.MultiControlledX,
                    num_control_wires=4,
                    num_zero_control_values=0,
                    num_work_wires=1,
                    work_wire_type="borrowed",
                ): 1,
                qp.resource_rep(
                    qp.MultiControlledX,
                    num_control_wires=5,
                    num_zero_control_values=1,
                    num_work_wires=2,
                    work_wire_type="borrowed",
                ): 1,
                qp.decomposition.controlled_resource_rep(
                    qp.RX, {}, num_control_wires=2, num_work_wires=1
                ): 1,
                qp.decomposition.controlled_resource_rep(
                    qp.Rot, {}, num_control_wires=2, num_work_wires=1
                ): 1,
                qp.decomposition.controlled_resource_rep(
                    qp.RZ, {}, num_control_wires=3, num_work_wires=1
                ): 1,
                qp.decomposition.controlled_resource_rep(
                    qp.MultiRZ,
                    {"num_wires": 6},
                    num_control_wires=2,
                    num_work_wires=1,
                ): 1,
                qp.decomposition.controlled_resource_rep(
                    qp.MultiRZ,
                    {"num_wires": 5},
                    num_control_wires=3,
                    num_work_wires=1,
                ): 1,
                qp.decomposition.controlled_resource_rep(
                    qp.PauliRot,
                    {"pauli_word": "XYX"},
                    num_control_wires=2,
                    num_work_wires=1,
                ): 1,
                qp.resource_rep(qp.CCZ): 1,
                qp.decomposition.controlled_resource_rep(
                    qp.Z, {}, num_control_wires=3, num_work_wires=1
                ): 1,
            }
        )

    def test_multi_control_wires(self):
        """Tests with multiple (more than 2) control wires."""

        rule = make_controlled_decomp(custom_decomp)

        # Single control wire controlled on 1
        op = qp.ctrl(
            CustomMultiQubitOp(0.5, 0.6, 0.7, wires=[0, 1, 2, 3, 4, 5]),
            control=[6, 7, 9],
            control_values=[False, True, False],
            work_wires=[8],
        )
        with qp.queuing.AnnotatedQueue() as q:
            rule(*op.parameters, wires=op.wires, **op.hyperparameters)

        expected_ops = [
            qp.X(6),
            qp.X(9),
            qp.MultiControlledX(wires=[6, 7, 9, 0], work_wires=[8]),
            qp.MultiControlledX(wires=[6, 7, 9, 0, 1], work_wires=[8]),
            qp.MultiControlledX(wires=[6, 7, 9, 0, 1, 2], work_wires=[8]),
            qp.MultiControlledX(
                wires=[6, 7, 9, 0, 1, 2, 3],
                control_values=[1, 1, 1, 1, 0, 1],
                work_wires=[8, 4],
            ),
            qp.ops.Controlled(qp.RX(0.5, wires=0), control_wires=[6, 7, 9], work_wires=[8]),
            qp.ops.Controlled(
                qp.Rot(0.5, 0.6, 0.7, wires=0),
                control_wires=[6, 7, 9],
                work_wires=[8],
            ),
            qp.ops.Controlled(qp.RZ(0.5, wires=[1]), control_wires=[6, 7, 9, 0], work_wires=[8]),
            qp.ops.Controlled(
                qp.MultiRZ(0.5, wires=[0, 1, 2, 3, 4, 5]),
                control_wires=[6, 7, 9],
                work_wires=[8],
            ),
            qp.ops.Controlled(
                qp.MultiRZ(0.5, wires=[1, 2, 3, 4, 5]),
                control_wires=[6, 7, 9, 0],
                work_wires=[8],
            ),
            qp.ops.Controlled(
                qp.PauliRot(0.5, "XYX", wires=[0, 1, 2]),
                control_wires=[6, 7, 9],
                work_wires=[8],
            ),
            qp.ops.Controlled(qp.Z(0), control_wires=[6, 7, 9], work_wires=[8]),
            qp.ops.Controlled(qp.Z(1), control_wires=[6, 7, 9, 0], work_wires=[8]),
            qp.X(6),
            qp.X(9),
        ]

        for actual, expected in zip(q.queue, expected_ops, strict=True):
            qp.assert_equal(actual, expected)

        actual_resources = rule.compute_resources(**op.resource_params)
        assert actual_resources == Resources(
            {
                qp.resource_rep(qp.X): 4,
                qp.resource_rep(
                    qp.MultiControlledX,
                    num_control_wires=3,
                    num_zero_control_values=0,
                    num_work_wires=1,
                    work_wire_type="borrowed",
                ): 1,
                qp.resource_rep(
                    qp.MultiControlledX,
                    num_control_wires=4,
                    num_zero_control_values=0,
                    num_work_wires=1,
                    work_wire_type="borrowed",
                ): 1,
                qp.resource_rep(
                    qp.MultiControlledX,
                    num_control_wires=5,
                    num_zero_control_values=0,
                    num_work_wires=1,
                    work_wire_type="borrowed",
                ): 1,
                qp.resource_rep(
                    qp.MultiControlledX,
                    num_control_wires=6,
                    num_zero_control_values=1,
                    num_work_wires=2,
                    work_wire_type="borrowed",
                ): 1,
                qp.decomposition.controlled_resource_rep(
                    qp.RX, {}, num_control_wires=3, num_work_wires=1
                ): 1,
                qp.decomposition.controlled_resource_rep(
                    qp.Rot, {}, num_control_wires=3, num_work_wires=1
                ): 1,
                qp.decomposition.controlled_resource_rep(
                    qp.RZ, {}, num_control_wires=4, num_work_wires=1
                ): 1,
                qp.decomposition.controlled_resource_rep(
                    qp.MultiRZ,
                    {"num_wires": 6},
                    num_control_wires=3,
                    num_work_wires=1,
                ): 1,
                qp.decomposition.controlled_resource_rep(
                    qp.MultiRZ,
                    {"num_wires": 5},
                    num_control_wires=4,
                    num_work_wires=1,
                ): 1,
                qp.decomposition.controlled_resource_rep(
                    qp.PauliRot,
                    {"pauli_word": "XYX"},
                    num_control_wires=3,
                    num_work_wires=1,
                ): 1,
                qp.decomposition.controlled_resource_rep(
                    qp.Z, {}, num_control_wires=3, num_work_wires=1
                ): 1,
                qp.decomposition.controlled_resource_rep(
                    qp.Z, {}, num_control_wires=4, num_work_wires=1
                ): 1,
            }
        )

    def test_flip_control_adjoint(self):
        """Tests the flip_control_adjoint decomposition."""

        op = qp.ctrl(qp.adjoint(CustomMultiQubitOp(0.5, wires=[0, 1])), control=2)
        with queuing.AnnotatedQueue() as q:
            flip_control_adjoint(*op.parameters, wires=op.wires, **op.hyperparameters)

        assert q.queue == [qp.adjoint(qp.ctrl(CustomMultiQubitOp(0.5, wires=[0, 1]), 2))]
        assert flip_control_adjoint.compute_resources(**op.resource_params) == Resources(
            {
                adjoint_resource_rep(
                    qp.ops.Controlled,
                    {
                        "base_class": CustomMultiQubitOp,
                        "base_params": {"num_wires": 2},
                        "num_control_wires": 1,
                        "num_zero_control_values": 0,
                        "num_work_wires": 0,
                        "work_wire_type": "borrowed",
                    },
                ): 1
            }
        )

    @pytest.mark.unit
    def test_controlled_decomp_with_work_wire(self):
        """Tests the controlled decomposition with a single work wire (Lemma 7.11 from https://arxiv.org/pdf/quant-ph/9503016)."""

        U = qp.Rot.compute_matrix(0.123, 0.234, 0.345)
        op = qp.ctrl(qp.QubitUnitary(U, wires=0), control=[1, 2])

        with queuing.AnnotatedQueue() as q:
            qp.Projector([0], wires=3)
            ctrl_single_work_wire(*op.parameters, wires=op.wires, **op.hyperparameters)

        tape = qp.tape.QuantumScript.from_queue(q)
        [tape], _ = qp.transforms.resolve_dynamic_wires([tape], min_int=3)
        mat = qp.matrix(tape, wire_order=[0, 1, 2, 3])
        expected_mat = qp.matrix(op @ qp.Projector([0], wires=3), wire_order=[0, 1, 2, 3])
        assert qp.math.allclose(mat, expected_mat)

    @pytest.mark.unit
    def test_controlled_decomp_with_work_wire_not_applicable(self):
        """Tests that the controlled_decomp_with_work_wire is not applicable sometimes."""

        op = qp.ctrl(qp.RX(0.5, wires=0), control=[1], control_values=[0], work_wires=[3])
        assert not ctrl_single_work_wire.is_applicable(**op.resource_params)

        op = qp.ctrl(qp.RX(0.5, wires=0), control=[1, 2])
        assert not ctrl_single_work_wire.is_applicable(**op.resource_params)

    def test_decompose_to_controlled_unitary(self):
        """Tests the decomposition to controlled qubit unitary"""

        op = qp.ctrl(qp.Rot(0.1, 0.2, 0.3, wires=0), control=[1, 2, 3], work_wires=[4, 5])
        with queuing.AnnotatedQueue() as q:
            to_controlled_qubit_unitary(*op.parameters, wires=op.wires, **op.hyperparameters)

        assert q.queue == [
            qp.ControlledQubitUnitary(
                qp.Rot.compute_matrix(0.1, 0.2, 0.3), wires=[1, 2, 3, 0], work_wires=[4, 5]
            )
        ]
        assert to_controlled_qubit_unitary.compute_resources(**op.resource_params) == Resources(
            {
                resource_rep(
                    qp.ControlledQubitUnitary,
                    num_target_wires=1,
                    num_control_wires=3,
                    num_zero_control_values=0,
                    num_work_wires=2,
                    work_wire_type="borrowed",
                ): 1
            }
        )
