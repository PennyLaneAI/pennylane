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

from textwrap import dedent

import pytest

import pennylane as qml
from pennylane import queuing
from pennylane.decomposition.reconstruct import get_decomp_kwargs, register_reconstructor
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
    flip_zero_control,
    make_adjoint_decomp,
    make_controlled_decomp,
    merge_powers,
    pow_involutory,
    pow_involutory_no_reconstructor,
    pow_rotation,
    qjit_compatible_adjoint_rotation,
    qjit_compatible_cancel_adjoint,
    qjit_compatible_flip_pow_adjoint,
    qjit_compatible_self_adjoint,
    repeat_pow_base,
    self_adjoint,
    to_controlled_qubit_unitary,
)

# pylint: disable=no-name-in-module
from tests.decomposition.conftest import to_resources


class CustomOpWithoutReconstructor(
    qml.operation.Operator
):  # pylint: disable=too-few-public-methods

    resource_keys = {"key"}

    @property
    def resource_params(self):
        return {"key": 0}


class CustomOpWithReconstructor(qml.operation.Operator):  # pylint: disable=too-few-public-methods

    resource_keys = {"key"}

    @property
    def resource_params(self):
        return {"key": 0}


@register_reconstructor(CustomOpWithReconstructor)
def _reconstruct_custom_op(*_, wires, **__):
    return CustomOpWithReconstructor(wires)


@pytest.mark.unit
class TestAdjointDecompositionRules:
    """Tests the decomposition rules defined for the adjoint of operations."""

    @pytest.mark.parametrize(
        ("op_type", "rule"),
        [
            (CustomOpWithoutReconstructor, cancel_adjoint),
            (CustomOpWithReconstructor, qjit_compatible_cancel_adjoint),
        ],
    )
    def test_cancel_adjoint(self, op_type, rule):
        """Tests that the adjoint of an adjoint cancels out."""

        op = qml.adjoint(qml.adjoint(op_type(0.5, wires=0)))

        kwargs = get_decomp_kwargs(op)
        with qml.queuing.AnnotatedQueue() as q:
            rule(*op.parameters, wires=op.wires, **kwargs)

        assert q.queue == [op_type(0.5, wires=0)]
        assert cancel_adjoint.compute_resources(**op.resource_params) == to_resources(
            {resource_rep(op_type, key=0): 1}
        )

    @pytest.mark.capture
    @pytest.mark.parametrize(
        ("op_type", "rule"),
        [
            (CustomOpWithoutReconstructor, cancel_adjoint),
            (CustomOpWithReconstructor, qjit_compatible_cancel_adjoint),
        ],
    )
    def test_cancel_adjoint_capture(self, op_type, rule):
        """Tests that the adjoint of an adjoint works with capture."""

        from pennylane.tape.plxpr_conversion import CollectOpsandMeas

        op = qml.adjoint(qml.adjoint(op_type(0.5, wires=0)))

        kwargs = get_decomp_kwargs(op)

        def circuit():
            rule(*op.parameters, wires=op.wires, **kwargs)

        plxpr = qml.capture.make_plxpr(circuit)()
        collector = CollectOpsandMeas()
        collector.eval(plxpr.jaxpr, plxpr.consts)
        assert collector.state["ops"] == [op_type(0.5, wires=0)]

    @pytest.mark.parametrize(
        "op_type, use_reconstructor",
        [(CustomOpWithReconstructor, True), (CustomOpWithoutReconstructor, False)],
    )
    def test_adjoint_general(self, op_type, use_reconstructor):
        """Tests the adjoint of a general operator can be correctly decomposed."""

        @qml.register_resources({qml.H: 1, qml.CNOT: 2, qml.RX: 1, qml.T: 1})
        def _custom_decomp(phi, wires, **_):
            qml.H(wires[0])
            qml.CNOT(wires=wires[:2])
            qml.RX(phi, wires=wires[1])
            qml.CNOT(wires=wires[1:3])
            qml.T(wires[2])

        op = qml.adjoint(op_type(0.5, wires=[0, 1, 2]))
        rule = make_adjoint_decomp(_custom_decomp, use_reconstructor)

        if not use_reconstructor:
            assert str(rule) == dedent("""
                With base_decomposition defined as:

                @qml.register_resources({qml.H: 1, qml.CNOT: 2, qml.RX: 1, qml.T: 1})
                def _custom_decomp(phi, wires, **_):
                    qml.H(wires[0])
                    qml.CNOT(wires=wires[:2])
                    qml.RX(phi, wires=wires[1])
                    qml.CNOT(wires=wires[1:3])
                    qml.T(wires[2])

                The full decomposition is defined as:

                @register_condition(_condition_fn)
                @register_resources(
                    _resource_fn,
                    work_wires=base_decomposition._work_wire_spec,
                    exact=base_decomposition.exact_resources,
                    name=f"adjoint({base_decomposition.name})",
                )
                def _impl(*params, wires, base):
                    # pylint: disable=protected-access
                    qml.adjoint(base_decomposition._impl)(*params, wires=wires, **base.hyperparameters)
                """).strip()
        else:
            assert str(rule) == dedent("""
                With base_decomposition defined as:

                @qml.register_resources({qml.H: 1, qml.CNOT: 2, qml.RX: 1, qml.T: 1})
                def _custom_decomp(phi, wires, **_):
                    qml.H(wires[0])
                    qml.CNOT(wires=wires[:2])
                    qml.RX(phi, wires=wires[1])
                    qml.CNOT(wires=wires[1:3])
                    qml.T(wires[2])

                The full decomposition is defined as:

                @register_condition(_condition_fn)
                @register_resources(
                    _resource_fn,
                    work_wires=base_decomposition._work_wire_spec,
                    exact=base_decomposition.exact_resources,
                    name=f"adjoint({base_decomposition.name})",
                )
                def _impl_using_reconstructor(*params, wires, base_params, **_):
                    # pylint: disable=protected-access
                    qml.adjoint(base_decomposition._impl)(*params, wires=wires, **base_params)
                """).strip()

        assert rule.name == "adjoint(_custom_decomp)"

        kwargs = get_decomp_kwargs(op)
        with qml.queuing.AnnotatedQueue() as q:
            rule(*op.parameters, wires=op.wires, **kwargs)

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

    @pytest.mark.parametrize(
        ("op_type", "rule"),
        [
            (CustomOpWithoutReconstructor, adjoint_rotation),
            (CustomOpWithReconstructor, qjit_compatible_adjoint_rotation),
        ],
    )
    def test_adjoint_rotation(self, op_type, rule):
        """Tests the adjoint_rotation decomposition."""

        op = qml.adjoint(op_type(0.5, wires=[0, 1, 2]))
        kwargs = get_decomp_kwargs(op)
        with queuing.AnnotatedQueue() as q:
            rule(*op.parameters, wires=op.wires, **kwargs)

        assert q.queue == [op_type(-0.5, wires=[0, 1, 2])]
        assert rule.compute_resources(**op.resource_params) == Resources(
            {resource_rep(op_type, key=0): 1}
        )

    @pytest.mark.parametrize(
        ("op_type", "rule"),
        [
            (CustomOpWithoutReconstructor, self_adjoint),
            (CustomOpWithReconstructor, qjit_compatible_self_adjoint),
        ],
    )
    def test_self_adjoint(self, op_type, rule):
        """Tests the self_adjoint decomposition."""

        op = qml.adjoint(op_type(0.5, wires=[0, 1, 2]))
        kwargs = get_decomp_kwargs(op)
        with queuing.AnnotatedQueue() as q:
            rule(*op.parameters, wires=op.wires, **kwargs)

        assert q.queue == [op_type(0.5, wires=[0, 1, 2])]
        assert rule.compute_resources(**op.resource_params) == Resources(
            {resource_rep(op_type, key=0): 1}
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

    @pytest.mark.capture
    def test_repeat_pow_base_capture(self):
        """Tests that the general pow decomposition works with capture."""

        from pennylane.tape.plxpr_conversion import CollectOpsandMeas

        op = qml.pow(qml.H(0), 3)

        def circuit():
            repeat_pow_base(*op.parameters, wires=op.wires, **op.hyperparameters)

        plxpr = qml.capture.make_plxpr(circuit)()
        collector = CollectOpsandMeas()
        collector.eval(plxpr.jaxpr, plxpr.consts)
        assert collector.state["ops"] == [qml.H(0), qml.H(0), qml.H(0)]

    def test_non_integer_pow_not_applicable(self):
        """Tests that is_applicable returns False when z isn't a positive integer."""

        op = qml.pow(qml.H(0), 0.5)
        assert not repeat_pow_base.is_applicable(**op.resource_params)
        op = qml.pow(qml.H(0), -1)
        assert not repeat_pow_base.is_applicable(**op.resource_params)

    @pytest.mark.parametrize(
        ("base_op", "rule"),
        [
            (CustomOpWithoutReconstructor, flip_pow_adjoint),
            (CustomOpWithReconstructor, qjit_compatible_flip_pow_adjoint),
        ],
    )
    def test_flip_pow_adjoint(self, base_op, rule):
        """Tests the flip_pow_adjoint and qjit_compatible_flip_pow_adjoint decompositions."""

        op = qml.pow(qml.adjoint(base_op(0.5, wires=[0, 1, 2])), 2)

        rule_params = get_decomp_kwargs(op)

        with queuing.AnnotatedQueue() as q:
            rule(*op.parameters, wires=op.wires, **rule_params)

        assert q.queue == [qml.adjoint(qml.pow(base_op(0.5, wires=[0, 1, 2]), 2))]
        assert rule.compute_resources(**op.resource_params) == Resources(
            {
                adjoint_resource_rep(
                    qml.ops.Pow,
                    {"base_class": base_op, "base_params": {"key": 0}, "z": 2},
                ): 1
            }
        )

    @pytest.mark.parametrize(
        ("op_type", "rule"),
        [
            (CustomOpWithoutReconstructor, pow_involutory_no_reconstructor),
            (CustomOpWithReconstructor, pow_involutory),
        ],
    )
    def test_pow_involutory(self, op_type, rule):
        """Tests the pow_involutory decomposition."""

        op1 = qml.pow(op_type(wires=[0, 1, 2]), 1)
        op2 = qml.pow(op_type(wires=[0, 1, 2]), 2)
        op3 = qml.pow(op_type(wires=[0, 1, 2]), 3)
        op4 = qml.pow(op_type(wires=[0, 1, 2]), 4)
        op5 = qml.pow(op_type(wires=[0, 1, 2]), 4.5)

        rule1_params = get_decomp_kwargs(op1)
        rule2_params = get_decomp_kwargs(op2)
        rule3_params = get_decomp_kwargs(op3)
        rule4_params = get_decomp_kwargs(op4)
        rule5_params = get_decomp_kwargs(op5)

        with qml.queuing.AnnotatedQueue() as q:
            rule(*op1.parameters, wires=op1.wires, **rule1_params)
            rule(*op2.parameters, wires=op2.wires, **rule2_params)
            rule(*op3.parameters, wires=op3.wires, **rule3_params)
            rule(*op4.parameters, wires=op4.wires, **rule4_params)
            rule(*op5.parameters, wires=op5.wires, **rule5_params)

        assert q.queue == [
            op_type(wires=[0, 1, 2]),
            op_type(wires=[0, 1, 2]),
            qml.pow(op_type(wires=[0, 1, 2]), 0.5),
        ]
        assert rule.compute_resources(**op1.resource_params) == Resources(
            {resource_rep(op_type, key=0): 1}
        )
        assert rule.compute_resources(**op3.resource_params) == Resources(
            {resource_rep(op_type, key=0): 1}
        )
        assert rule.compute_resources(**op2.resource_params) == Resources()
        assert rule.compute_resources(**op4.resource_params) == Resources()
        assert rule.compute_resources(**op5.resource_params) == Resources(
            {pow_resource_rep(op_type, {"key": 0}, 0.5): 1}
        )

        assert not rule.is_applicable(op_type, {}, z=0.5)

    def test_pow_rotations(self):
        """Tests the pow_rotations decomposition."""

        op = qml.pow(CustomOpWithoutReconstructor(0.3, wires=[0, 1, 2]), 2.5)
        with queuing.AnnotatedQueue() as q:
            pow_rotation(*op.parameters, wires=op.wires, **op.hyperparameters)

        assert q.queue == [CustomOpWithoutReconstructor(0.3 * 2.5, wires=[0, 1, 2])]
        assert pow_rotation.compute_resources(**op.resource_params) == Resources(
            {resource_rep(CustomOpWithoutReconstructor, key=0): 1}
        )


class CustomMultiQubitOp(qml.operation.Operation):  # pylint: disable=too-few-public-methods
    """A custom op."""

    resource_keys = {"num_wires"}

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
            work_wire_type="zeroed",
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
class TestControlledDecomposition:
    """Tests applying control on the decomposition of the base operator."""

    def test_single_control_wire(self):
        """Tests a single control wire."""

        rule = make_controlled_decomp(custom_decomp)

        # Single control wire controlled on 1
        op = qml.ctrl(
            CustomMultiQubitOp(0.5, 0.6, 0.7, wires=[0, 1, 2, 3, 4, 5]), control=[6], work_wires=[7]
        )
        with qml.queuing.AnnotatedQueue() as q:
            rule(*op.parameters, wires=op.wires, **op.hyperparameters)

        expected_ops = [
            qml.CNOT(wires=[6, 0]),
            qml.MultiControlledX(wires=[6, 0, 1], work_wires=[7]),
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
                qml.resource_rep(
                    qml.MultiControlledX,
                    num_control_wires=2,
                    num_zero_control_values=0,
                    num_work_wires=1,
                    work_wire_type="borrowed",
                ): 1,
                qml.resource_rep(
                    qml.MultiControlledX,
                    num_control_wires=3,
                    num_zero_control_values=0,
                    num_work_wires=1,
                    work_wire_type="borrowed",
                ): 1,
                qml.resource_rep(
                    qml.MultiControlledX,
                    num_control_wires=4,
                    num_zero_control_values=1,
                    num_work_wires=2,
                    work_wire_type="borrowed",
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

        rule = make_controlled_decomp(custom_decomp)

        assert str(rule) == dedent("""
            With base_decomposition defined as:

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

            The full decomposition is defined as:

            @register_condition(_condition_fn)
            @register_resources(
                _resource_fn,
                work_wires=base_decomposition._work_wire_spec,
                exact=base_decomposition.exact_resources,
                name=f"controlled({base_decomposition.name})",
            )
            def _impl(*params, wires, control_wires, control_values, work_wires, work_wire_type, base, **_):
                zero_control_wires = [w for w, val in zip(control_wires, control_values) if not val]
                for w in zero_control_wires:
                    qml.PauliX(w)
                # We're extracting control wires and base wires from the wires argument instead
                # of directly using control_wires and base.wires, `wires` is properly traced, but
                # `control_wires` and `base.wires` are not.
                qml.ctrl(
                    base_decomposition._impl,  # pylint: disable=protected-access
                    control=wires[: len(control_wires)],
                    work_wires=work_wires,
                    work_wire_type=work_wire_type,
                )(*params, wires=wires[-len(base.wires) :], **base.hyperparameters)
                for w in zero_control_wires:
                    qml.PauliX(w)
            """).strip()

        assert rule.name == "controlled(custom_decomp)"

        # Single control wire controlled on 1
        op = qml.ctrl(
            CustomMultiQubitOp(0.5, 0.6, 0.7, wires=[0, 1, 2, 3, 4, 5]),
            control=[6, 7],
            control_values=[False, True],
            work_wires=[8],
        )
        with qml.queuing.AnnotatedQueue() as q:
            rule(*op.parameters, wires=op.wires, **op.hyperparameters)

        expected_ops = [
            qml.X(6),
            qml.MultiControlledX(wires=[6, 7, 0], work_wires=[8]),
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
                qml.resource_rep(
                    qml.MultiControlledX,
                    num_control_wires=2,
                    num_zero_control_values=0,
                    num_work_wires=1,
                    work_wire_type="borrowed",
                ): 1,
                qml.resource_rep(
                    qml.MultiControlledX,
                    num_control_wires=3,
                    num_zero_control_values=0,
                    num_work_wires=1,
                    work_wire_type="borrowed",
                ): 1,
                qml.resource_rep(
                    qml.MultiControlledX,
                    num_control_wires=4,
                    num_zero_control_values=0,
                    num_work_wires=1,
                    work_wire_type="borrowed",
                ): 1,
                qml.resource_rep(
                    qml.MultiControlledX,
                    num_control_wires=5,
                    num_zero_control_values=1,
                    num_work_wires=2,
                    work_wire_type="borrowed",
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

        rule = make_controlled_decomp(custom_decomp)

        # Single control wire controlled on 1
        op = qml.ctrl(
            CustomMultiQubitOp(0.5, 0.6, 0.7, wires=[0, 1, 2, 3, 4, 5]),
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
                    work_wire_type="borrowed",
                ): 1,
                qml.resource_rep(
                    qml.MultiControlledX,
                    num_control_wires=4,
                    num_zero_control_values=0,
                    num_work_wires=1,
                    work_wire_type="borrowed",
                ): 1,
                qml.resource_rep(
                    qml.MultiControlledX,
                    num_control_wires=5,
                    num_zero_control_values=0,
                    num_work_wires=1,
                    work_wire_type="borrowed",
                ): 1,
                qml.resource_rep(
                    qml.MultiControlledX,
                    num_control_wires=6,
                    num_zero_control_values=1,
                    num_work_wires=2,
                    work_wire_type="borrowed",
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

    def test_flip_control_adjoint(self):
        """Tests the flip_control_adjoint decomposition."""

        op = qml.ctrl(qml.adjoint(CustomMultiQubitOp(0.5, wires=[0, 1])), control=2)
        with queuing.AnnotatedQueue() as q:
            flip_control_adjoint(*op.parameters, wires=op.wires, **op.hyperparameters)

        assert q.queue == [qml.adjoint(qml.ctrl(CustomMultiQubitOp(0.5, wires=[0, 1]), 2))]
        assert flip_control_adjoint.compute_resources(**op.resource_params) == Resources(
            {
                adjoint_resource_rep(
                    qml.ops.Controlled,
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

    def test_flip_zero_control(self):
        """Tests the `flip_zero_control` modifier."""

        @qml.register_resources({qml.CNOT: 3, qml.H: 2})
        def _custom_controlled_rule(wires, **_):
            qml.CNOT(wires[0:2])
            qml.H(wires[1])
            qml.CNOT(wires[1:3])
            qml.H(wires[1])
            qml.CNOT(wires[0:2])

        rule = flip_zero_control(_custom_controlled_rule)
        assert rule.name == "flip_zero_ctrl_values(_custom_controlled_rule)"

        assert str(rule) == dedent("""
            With inner_decomp defined as:

            @qml.register_resources({qml.CNOT: 3, qml.H: 2})
            def _custom_controlled_rule(wires, **_):
                qml.CNOT(wires[0:2])
                qml.H(wires[1])
                qml.CNOT(wires[1:3])
                qml.H(wires[1])
                qml.CNOT(wires[0:2])

            The full decomposition is defined as:

            @register_condition(_condition_fn)
            @register_resources(
                _resource_fn,
                work_wires=inner_decomp._work_wire_spec,
                exact=inner_decomp.exact_resources,
                name=f"flip_zero_ctrl_values({inner_decomp.name})",
            )
            def _impl(*params, wires, control_wires, control_values, **kwargs):
                zero_control_wires = [w for w, val in zip(control_wires, control_values) if not val]
                for w in zero_control_wires:
                    qml.PauliX(w)
                inner_decomp(
                    *params,
                    wires=wires,
                    control_wires=control_wires,
                    control_values=[1] * len(control_wires),  # all control values are 1 now
                    **kwargs,
                )
                for w in zero_control_wires:
                    qml.PauliX(w)
            """).strip()

        with qml.queuing.AnnotatedQueue() as q:
            rule(wires=[2, 3, 4], control_wires=[2, 3], control_values=[True, False])

        assert q.queue == [
            qml.X(3),
            qml.CNOT([2, 3]),
            qml.H(3),
            qml.CNOT([3, 4]),
            qml.H(3),
            qml.CNOT([2, 3]),
            qml.X(3),
        ]

    @pytest.mark.unit
    def test_controlled_decomp_with_work_wire(self):
        """Tests the controlled decomposition with a single work wire (Lemma 7.11 from https://arxiv.org/pdf/quant-ph/9503016)."""

        U = qml.Rot.compute_matrix(0.123, 0.234, 0.345)
        op = qml.ctrl(qml.QubitUnitary(U, wires=0), control=[1, 2])

        with queuing.AnnotatedQueue() as q:
            qml.Projector([0], wires=3)
            ctrl_single_work_wire(*op.parameters, wires=op.wires, **op.hyperparameters)

        tape = qml.tape.QuantumScript.from_queue(q)
        [tape], _ = qml.transforms.resolve_dynamic_wires([tape], min_int=3)
        mat = qml.matrix(tape, wire_order=[0, 1, 2, 3])
        expected_mat = qml.matrix(op @ qml.Projector([0], wires=3), wire_order=[0, 1, 2, 3])
        assert qml.math.allclose(mat, expected_mat)

    @pytest.mark.unit
    def test_controlled_decomp_with_work_wire_not_applicable(self):
        """Tests that the controlled_decomp_with_work_wire is not applicable sometimes."""

        op = qml.ctrl(qml.RX(0.5, wires=0), control=[1], control_values=[0], work_wires=[3])
        assert not ctrl_single_work_wire.is_applicable(**op.resource_params)

        op = qml.ctrl(qml.RX(0.5, wires=0), control=[1, 2])
        assert not ctrl_single_work_wire.is_applicable(**op.resource_params)

    def test_decompose_to_controlled_unitary(self):
        """Tests the decomposition to controlled qubit unitary"""

        op = qml.ctrl(qml.Rot(0.1, 0.2, 0.3, wires=0), control=[1, 2, 3], work_wires=[4, 5])
        with queuing.AnnotatedQueue() as q:
            to_controlled_qubit_unitary(*op.parameters, wires=op.wires, **op.hyperparameters)

        assert q.queue == [
            qml.ControlledQubitUnitary(
                qml.Rot.compute_matrix(0.1, 0.2, 0.3), wires=[1, 2, 3, 0], work_wires=[4, 5]
            )
        ]
        assert to_controlled_qubit_unitary.compute_resources(**op.resource_params) == Resources(
            {
                resource_rep(
                    qml.ControlledQubitUnitary,
                    num_target_wires=1,
                    num_control_wires=3,
                    num_zero_control_values=0,
                    num_work_wires=2,
                    work_wire_type="borrowed",
                ): 1
            }
        )
