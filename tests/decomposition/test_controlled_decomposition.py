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
    ControlledBaseDecomposition,
    CustomControlledDecomposition,
    controlled_global_phase_decomp,
    controlled_x_decomp,
)
from pennylane.decomposition.resources import (
    CompressedResourceOp,
    Resources,
    controlled_resource_rep,
    resource_rep,
)
from tests.decomposition.conftest import to_resources


@pytest.mark.unit
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
        assert controlled_global_phase_decomp.compute_resources(
            **op.resource_params
        ) == to_resources({qml.PhaseShift: 1})

    def test_single_controlled_global_phase_on_0(self):
        """Tests GlobalPhase controlled on a single wire with control value being 0."""

        op = qml.ctrl(qml.GlobalPhase(0.5, wires=[0, 1]), control=[2], control_values=[0])

        with qml.queuing.AnnotatedQueue() as q:
            controlled_global_phase_decomp(*op.parameters, wires=op.wires, **op.hyperparameters)

        assert q.queue == [qml.PhaseShift(0.5, wires=[2]), qml.GlobalPhase(0.5)]
        assert controlled_global_phase_decomp.compute_resources(
            **op.resource_params
        ) == to_resources({qml.GlobalPhase: 1, qml.PhaseShift: 1})

    def test_double_controlled_global_phase(self):
        """Tests global phase controlled on two wires."""

        op = qml.ctrl(
            qml.GlobalPhase(0.5, wires=[0, 1]), control=[2, 3], control_values=[False, True]
        )

        with qml.queuing.AnnotatedQueue() as q:
            controlled_global_phase_decomp(*op.parameters, wires=op.wires, **op.hyperparameters)

        assert q.queue == [qml.X(2), qml.ControlledPhaseShift(-0.5, wires=[2, 3]), qml.X(2)]
        assert controlled_global_phase_decomp.compute_resources(
            **op.resource_params
        ) == to_resources({qml.X: 2, qml.ControlledPhaseShift: 1})

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
            {
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
            }
        )


@pytest.mark.unit
class TestControlledX:
    """Tests decompositions of different versions of controlled X gates."""

    def test_single_controlled_x(self):
        """Tests that a single-controlled X decomposes to a CNOT."""

        op = qml.ops.Controlled(qml.X(0), control_wires=[1])
        with qml.queuing.AnnotatedQueue() as q:
            controlled_x_decomp(*op.parameters, wires=op.wires, **op.hyperparameters)

        assert q.queue == [qml.CNOT(wires=[1, 0])]
        assert controlled_x_decomp.compute_resources(**op.resource_params) == to_resources(
            {qml.CNOT: 1}
        )

        op = qml.ops.Controlled(qml.X(0), control_wires=[1], control_values=[0])
        with qml.queuing.AnnotatedQueue() as q:
            controlled_x_decomp(*op.parameters, wires=op.wires, **op.hyperparameters)

        assert q.queue == [qml.CNOT(wires=[1, 0]), qml.PauliX(0)]
        assert controlled_x_decomp.compute_resources(**op.resource_params) == to_resources(
            {qml.CNOT: 1, qml.PauliX: 1},
        )

    def test_double_controlled_x(self):
        """Tests that a double-controlled X decomposes to a Toffoli gate."""

        op = qml.ops.Controlled(qml.X(0), control_wires=[1, 2])
        with qml.queuing.AnnotatedQueue() as q:
            controlled_x_decomp(*op.parameters, wires=op.wires, **op.hyperparameters)

        assert q.queue == [qml.Toffoli(wires=[1, 2, 0])]
        assert controlled_x_decomp.compute_resources(**op.resource_params) == to_resources(
            {qml.Toffoli: 1}
        )

        op = qml.ops.Controlled(qml.X(0), control_wires=[1, 2], control_values=[0, 1])
        with qml.queuing.AnnotatedQueue() as q:
            controlled_x_decomp(*op.parameters, wires=op.wires, **op.hyperparameters)

        assert q.queue == [qml.PauliX(1), qml.Toffoli(wires=[1, 2, 0]), qml.PauliX(1)]
        assert controlled_x_decomp.compute_resources(**op.resource_params) == to_resources(
            {qml.Toffoli: 1, qml.PauliX: 2},
        )

    def test_multi_controlled_x(self):
        """Tests that a multi-controlled X decomposes to a MultiControlledX gate."""

        op = qml.ops.Controlled(qml.X(0), control_wires=[1, 2, 3], work_wires=[4])
        with qml.queuing.AnnotatedQueue() as q:
            controlled_x_decomp(*op.parameters, wires=op.wires, **op.hyperparameters)

        assert q.queue == [qml.MultiControlledX(wires=[1, 2, 3, 0], work_wires=[4])]
        assert controlled_x_decomp.compute_resources(**op.resource_params) == Resources(
            {
                CompressedResourceOp(
                    qml.MultiControlledX,
                    {
                        "num_control_wires": 3,
                        "num_zero_control_values": 0,
                        "num_work_wires": 1,
                    },
                ): 1
            }
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
            {
                CompressedResourceOp(
                    qml.MultiControlledX,
                    {
                        "num_control_wires": 3,
                        "num_zero_control_values": 2,
                        "num_work_wires": 1,
                    },
                ): 1,
            }
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    "op, decomp, resources, custom_op_type",
    [
        # Single-qubit controlled on 0
        (
            qml.ops.Controlled(qml.Z(1), control_wires=[0], control_values=[0]),
            [qml.X(0), qml.CZ(wires=[0, 1]), qml.X(0)],
            to_resources({qml.CZ: 1, qml.X: 2}),
            qml.ops.CZ,
        ),
        # Single-qubit controlled on 1
        (
            qml.ops.Controlled(qml.Z(1), control_wires=[0], control_values=[1]),
            [qml.CZ(wires=[0, 1])],
            to_resources({qml.CZ: 1}),
            qml.ops.CZ,
        ),
        # Controlled on two qubits
        (
            qml.ops.Controlled(qml.Z(2), control_wires=[0, 1], control_values=[1, 0]),
            [qml.X(1), qml.CCZ(wires=[0, 1, 2]), qml.X(1)],
            to_resources({qml.CCZ: 1, qml.X: 2}),
            qml.ops.CCZ,
        ),
        # Parametrized controlled
        (
            qml.ops.Controlled(qml.RX(0.5, wires=1), control_wires=[0], control_values=[0]),
            [qml.X(0), qml.CRX(0.5, wires=[0, 1]), qml.X(0)],
            to_resources({qml.CRX: 1, qml.X: 2}),
            qml.ops.CRX,
        ),
        # Two-qubit controlled
        (
            qml.ops.Controlled(qml.SWAP(wires=[1, 2]), control_wires=[0], control_values=[0]),
            [qml.X(0), qml.CSWAP(wires=[0, 1, 2]), qml.X(0)],
            to_resources({qml.CSWAP: 1, qml.X: 2}),
            qml.CSWAP,
        ),
    ],
)
def test_controlled_to_custom_controlled(op, decomp, resources, custom_op_type):
    """Tests that a controlled op decomposes to its corresponding custom op if applicable."""

    decomp_rule = CustomControlledDecomposition(custom_op_type)
    with qml.queuing.AnnotatedQueue() as q:
        decomp_rule(*op.parameters, wires=op.wires, **op.hyperparameters)

    assert q.queue == decomp
    assert decomp_rule.compute_resources(**op.resource_params) == resources


class CustomOp(qml.operation.Operation):  # pylint: disable=too-few-public-methods
    """A custom op."""

    resource_param_keys = ("num_wires",)

    @property
    def resource_params(self):
        return {"num_wires": len(self.wires)}


def _custom_resource(num_wires):
    return {
        qml.X: 1,
        qml.CNOT: 1,
        qml.Toffoli: 1,
        qml.resource_rep(
            qml.MultiControlledX,
            num_control_wires=3,
            num_zero_control_values=1,
            num_work_wires=1,
        ): 1,
        qml.RX: 1,
        qml.Rot: 1,
        qml.CRZ: 1,
        resource_rep(qml.MultiRZ, num_wires=num_wires): 1,
        controlled_resource_rep(
            qml.MultiRZ,
            {"num_wires": num_wires - 1},
            num_control_wires=1,
        ): 1,
        resource_rep(qml.PauliRot, pauli_word="XYX"): 1,
        qml.Z: 1,
        qml.CZ: 1,
    }


@qml.register_resources(_custom_resource)
def custom_decomp(*params, wires, **_):
    qml.X(wires[0])
    qml.CNOT(wires=wires[:2])
    qml.Toffoli(wires=wires[:3])
    qml.MultiControlledX(wires=wires[:4], control_values=[1, 0, 1], work_wires=[4])
    qml.RX(params[0], wires=wires[0])
    qml.Rot(params[0], params[1], params[2], wires=wires[0])
    qml.CRZ(params[0], wires=wires[:2])
    qml.MultiRZ(params[0], wires=wires)
    qml.ctrl(qml.MultiRZ(params[0], wires=wires[1:]), control=wires[0])
    qml.PauliRot(params[0], "XYX", wires=wires[:3])
    qml.Z(wires[0])
    qml.CZ(wires=wires[:2])


@pytest.mark.unit
class TestControlledBaseDecomposition:
    """Tests applying control on the decomposition of the base operator."""

    def test_single_control_wire(self):
        """Tests a single control wire."""

        rule = ControlledBaseDecomposition(custom_decomp)

        # Single control wire controlled on 1
        op = qml.ctrl(
            CustomOp(0.5, 0.6, 0.7, wires=[0, 1, 2, 3, 4, 5]), control=[6], work_wires=[7]
        )
        with qml.queuing.AnnotatedQueue() as q:
            rule(*op.parameters, wires=op.wires, **op.hyperparameters)

        expected_ops = [
            qml.CNOT(wires=[6, 0]),
            qml.Toffoli(wires=[6, 0, 1]),
            qml.MultiControlledX(wires=[6, 0, 1, 2], work_wires=[7]),
            qml.MultiControlledX(
                wires=[6, 0, 1, 2, 3], control_values=[1, 1, 0, 1], work_wires=[7, 4]
            ),
            qml.CRX(0.5, wires=[6, 0]),
            qml.CRot(0.5, 0.6, 0.7, wires=[6, 0]),
            qml.ops.Controlled(qml.RZ(0.5, wires=[1]), control_wires=[6, 0], work_wires=[7]),
            qml.ops.Controlled(
                qml.MultiRZ(0.5, wires=[0, 1, 2, 3, 4, 5]),
                control_wires=[6],
                work_wires=[7],
            ),
            qml.ops.Controlled(
                qml.MultiRZ(0.5, wires=[1, 2, 3, 4, 5]),
                control_wires=[6, 0],
                work_wires=[7],
            ),
            qml.ops.Controlled(
                qml.PauliRot(0.5, "XYX", wires=[0, 1, 2]),
                control_wires=[6],
                work_wires=[7],
            ),
            qml.CZ(wires=[6, 0]),
            qml.CCZ(wires=[6, 0, 1]),
        ]
        for actual, expected in zip(q.queue, expected_ops, strict=True):
            qml.assert_equal(actual, expected)

        actual_resources = rule.compute_resources(**op.resource_params)
        assert actual_resources == Resources(
            {
                qml.resource_rep(qml.CNOT): 1,
                qml.resource_rep(qml.Toffoli): 1,
                qml.resource_rep(
                    qml.MultiControlledX,
                    num_control_wires=3,
                    num_zero_control_values=0,
                    num_work_wires=1,
                ): 1,
                qml.resource_rep(
                    qml.MultiControlledX,
                    num_control_wires=4,
                    num_zero_control_values=1,
                    num_work_wires=2,
                ): 1,
                qml.resource_rep(qml.CRX): 1,
                qml.resource_rep(qml.CRot): 1,
                qml.decomposition.controlled_resource_rep(
                    qml.RZ, {}, num_control_wires=2, num_work_wires=1
                ): 1,
                qml.decomposition.controlled_resource_rep(
                    qml.MultiRZ,
                    {"num_wires": 6},
                    num_control_wires=1,
                    num_work_wires=1,
                ): 1,
                qml.decomposition.controlled_resource_rep(
                    qml.MultiRZ,
                    {"num_wires": 5},
                    num_control_wires=2,
                    num_work_wires=1,
                ): 1,
                qml.decomposition.controlled_resource_rep(
                    qml.PauliRot,
                    {"pauli_word": "XYX"},
                    num_control_wires=1,
                    num_work_wires=1,
                ): 1,
                qml.resource_rep(qml.CZ): 1,
                qml.resource_rep(qml.CCZ): 1,
            }
        )

    def test_double_control_wire(self):
        """Tests two control wires."""

        rule = ControlledBaseDecomposition(custom_decomp)

        # Single control wire controlled on 1
        op = qml.ctrl(
            CustomOp(0.5, 0.6, 0.7, wires=[0, 1, 2, 3, 4, 5]),
            control=[6, 7],
            control_values=[False, True],
            work_wires=[8],
        )
        with qml.queuing.AnnotatedQueue() as q:
            rule(*op.parameters, wires=op.wires, **op.hyperparameters)

        expected_ops = [
            qml.X(6),
            qml.Toffoli(wires=[6, 7, 0]),
            qml.MultiControlledX(wires=[6, 7, 0, 1], work_wires=[8]),
            qml.MultiControlledX(wires=[6, 7, 0, 1, 2], work_wires=[8]),
            qml.MultiControlledX(
                wires=[6, 7, 0, 1, 2, 3],
                control_values=[1, 1, 1, 0, 1],
                work_wires=[8, 4],
            ),
            qml.ops.Controlled(qml.RX(0.5, wires=0), control_wires=[6, 7], work_wires=[8]),
            qml.ops.Controlled(
                qml.Rot(0.5, 0.6, 0.7, wires=0),
                control_wires=[6, 7],
                work_wires=[8],
            ),
            qml.ops.Controlled(qml.RZ(0.5, wires=[1]), control_wires=[6, 7, 0], work_wires=[8]),
            qml.ops.Controlled(
                qml.MultiRZ(0.5, wires=[0, 1, 2, 3, 4, 5]),
                control_wires=[6, 7],
                work_wires=[8],
            ),
            qml.ops.Controlled(
                qml.MultiRZ(0.5, wires=[1, 2, 3, 4, 5]),
                control_wires=[6, 7, 0],
                work_wires=[8],
            ),
            qml.ops.Controlled(
                qml.PauliRot(0.5, "XYX", wires=[0, 1, 2]),
                control_wires=[6, 7],
                work_wires=[8],
            ),
            qml.CCZ(wires=[6, 7, 0]),
            qml.ops.Controlled(qml.Z(1), control_wires=[6, 7, 0], work_wires=[8]),
            qml.X(6),
        ]

        for actual, expected in zip(q.queue, expected_ops, strict=True):
            qml.assert_equal(actual, expected)

        actual_resources = rule.compute_resources(**op.resource_params)
        assert actual_resources == Resources(
            {
                qml.resource_rep(qml.X): 2,
                qml.resource_rep(qml.Toffoli): 1,
                qml.resource_rep(
                    qml.MultiControlledX,
                    num_control_wires=3,
                    num_zero_control_values=0,
                    num_work_wires=1,
                ): 1,
                qml.resource_rep(
                    qml.MultiControlledX,
                    num_control_wires=4,
                    num_zero_control_values=0,
                    num_work_wires=1,
                ): 1,
                qml.resource_rep(
                    qml.MultiControlledX,
                    num_control_wires=5,
                    num_zero_control_values=1,
                    num_work_wires=2,
                ): 1,
                qml.decomposition.controlled_resource_rep(
                    qml.RX, {}, num_control_wires=2, num_work_wires=1
                ): 1,
                qml.decomposition.controlled_resource_rep(
                    qml.Rot, {}, num_control_wires=2, num_work_wires=1
                ): 1,
                qml.decomposition.controlled_resource_rep(
                    qml.RZ, {}, num_control_wires=3, num_work_wires=1
                ): 1,
                qml.decomposition.controlled_resource_rep(
                    qml.MultiRZ,
                    {"num_wires": 6},
                    num_control_wires=2,
                    num_work_wires=1,
                ): 1,
                qml.decomposition.controlled_resource_rep(
                    qml.MultiRZ,
                    {"num_wires": 5},
                    num_control_wires=3,
                    num_work_wires=1,
                ): 1,
                qml.decomposition.controlled_resource_rep(
                    qml.PauliRot,
                    {"pauli_word": "XYX"},
                    num_control_wires=2,
                    num_work_wires=1,
                ): 1,
                qml.resource_rep(qml.CCZ): 1,
                qml.decomposition.controlled_resource_rep(
                    qml.Z, {}, num_control_wires=3, num_work_wires=1
                ): 1,
            }
        )

    def test_multi_control_wires(self):
        """Tests with multiple (more than 2) control wires."""

        rule = ControlledBaseDecomposition(custom_decomp)

        # Single control wire controlled on 1
        op = qml.ctrl(
            CustomOp(0.5, 0.6, 0.7, wires=[0, 1, 2, 3, 4, 5]),
            control=[6, 7, 9],
            control_values=[False, True, False],
            work_wires=[8],
        )
        with qml.queuing.AnnotatedQueue() as q:
            rule(*op.parameters, wires=op.wires, **op.hyperparameters)

        expected_ops = [
            qml.X(6),
            qml.X(9),
            qml.MultiControlledX(wires=[6, 7, 9, 0], work_wires=[8]),
            qml.MultiControlledX(wires=[6, 7, 9, 0, 1], work_wires=[8]),
            qml.MultiControlledX(wires=[6, 7, 9, 0, 1, 2], work_wires=[8]),
            qml.MultiControlledX(
                wires=[6, 7, 9, 0, 1, 2, 3],
                control_values=[1, 1, 1, 1, 0, 1],
                work_wires=[8, 4],
            ),
            qml.ops.Controlled(qml.RX(0.5, wires=0), control_wires=[6, 7, 9], work_wires=[8]),
            qml.ops.Controlled(
                qml.Rot(0.5, 0.6, 0.7, wires=0),
                control_wires=[6, 7, 9],
                work_wires=[8],
            ),
            qml.ops.Controlled(qml.RZ(0.5, wires=[1]), control_wires=[6, 7, 9, 0], work_wires=[8]),
            qml.ops.Controlled(
                qml.MultiRZ(0.5, wires=[0, 1, 2, 3, 4, 5]),
                control_wires=[6, 7, 9],
                work_wires=[8],
            ),
            qml.ops.Controlled(
                qml.MultiRZ(0.5, wires=[1, 2, 3, 4, 5]),
                control_wires=[6, 7, 9, 0],
                work_wires=[8],
            ),
            qml.ops.Controlled(
                qml.PauliRot(0.5, "XYX", wires=[0, 1, 2]),
                control_wires=[6, 7, 9],
                work_wires=[8],
            ),
            qml.ops.Controlled(qml.Z(0), control_wires=[6, 7, 9], work_wires=[8]),
            qml.ops.Controlled(qml.Z(1), control_wires=[6, 7, 9, 0], work_wires=[8]),
            qml.X(6),
            qml.X(9),
        ]

        for actual, expected in zip(q.queue, expected_ops, strict=True):
            qml.assert_equal(actual, expected)

        actual_resources = rule.compute_resources(**op.resource_params)
        assert actual_resources == Resources(
            {
                qml.resource_rep(qml.X): 4,
                qml.resource_rep(
                    qml.MultiControlledX,
                    num_control_wires=3,
                    num_zero_control_values=0,
                    num_work_wires=1,
                ): 1,
                qml.resource_rep(
                    qml.MultiControlledX,
                    num_control_wires=4,
                    num_zero_control_values=0,
                    num_work_wires=1,
                ): 1,
                qml.resource_rep(
                    qml.MultiControlledX,
                    num_control_wires=5,
                    num_zero_control_values=0,
                    num_work_wires=1,
                ): 1,
                qml.resource_rep(
                    qml.MultiControlledX,
                    num_control_wires=6,
                    num_zero_control_values=1,
                    num_work_wires=2,
                ): 1,
                qml.decomposition.controlled_resource_rep(
                    qml.RX, {}, num_control_wires=3, num_work_wires=1
                ): 1,
                qml.decomposition.controlled_resource_rep(
                    qml.Rot, {}, num_control_wires=3, num_work_wires=1
                ): 1,
                qml.decomposition.controlled_resource_rep(
                    qml.RZ, {}, num_control_wires=4, num_work_wires=1
                ): 1,
                qml.decomposition.controlled_resource_rep(
                    qml.MultiRZ,
                    {"num_wires": 6},
                    num_control_wires=3,
                    num_work_wires=1,
                ): 1,
                qml.decomposition.controlled_resource_rep(
                    qml.MultiRZ,
                    {"num_wires": 5},
                    num_control_wires=4,
                    num_work_wires=1,
                ): 1,
                qml.decomposition.controlled_resource_rep(
                    qml.PauliRot,
                    {"pauli_word": "XYX"},
                    num_control_wires=3,
                    num_work_wires=1,
                ): 1,
                qml.decomposition.controlled_resource_rep(
                    qml.Z, {}, num_control_wires=3, num_work_wires=1
                ): 1,
                qml.decomposition.controlled_resource_rep(
                    qml.Z, {}, num_control_wires=4, num_work_wires=1
                ): 1,
            }
        )
