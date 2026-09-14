# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This module contains tests for ``qp.subcircuit``.
"""

import inspect

import pytest

import pennylane as qp
from pennylane.core import Operator2
from pennylane.ops.functions import assert_valid
from pennylane.typing import Bool, Float, Wire


@qp.subcircuit(dynamic_argnames=("phi",), arg_specs={"phi": float, "wires": Wire[1]})
@qp.register_resources({qp.H: 1, qp.RZ: 1})
def FixedOp(phi, wires):
    """Custom operator created using qp.subcircuit."""
    qp.H(wires)
    qp.RZ(phi, wires)


def _loop_op_resources(phi, w, n_iters):  # pylint: disable=unused-argument
    """Resource function for the decomposition rule of LoopOp."""
    return {qp.H: n_iters, qp.RZ: n_iters}


@qp.subcircuit(dynamic_argnames=("phi",), wire_argnames=("w",), compilable_argnames=("n_iters",))
@qp.register_resources(_loop_op_resources)
def LoopOp(phi, w, n_iters):
    """Custom operator with control flow created using qp.subcircuit."""

    @qp.for_loop(n_iters)
    def loop(_):
        qp.H(w)
        qp.RZ(phi, w)

    loop()


class TestClassCreation:
    """Tests that ``subcircuit`` builds a valid ``Operator2`` subclass."""

    @pytest.mark.usefixtures("enable_and_disable_capture")
    def test_validity(self):
        """Test that the decomposition rule of operators returned by ``subcircuit`` is valid."""
        op = FixedOp(1.5, wires=1)
        assert_valid(op)

    # Differentiation test failing without program capture, but we don't care about PL classic
    @pytest.mark.usefixtures("enable_capture")
    def test_control_flow_validity(self):
        """Test that the decomposition rule of operators that contain control flow returned
        by ``subcircuit`` is valid."""
        op = LoopOp(1.5, w=1, n_iters=100)
        assert_valid(op, skip_differentiation=not qp.capture.enabled())

    def test_not_decomposition_rule_error(self):
        """Test that an error is raised if the input function is not a ``DecompositionRule``."""

        def invalid_fn(wires):
            qp.H(wires)

        with pytest.raises(TypeError, match="The provided quantum function must register"):
            _ = qp.subcircuit(invalid_fn)

    def test_returns_operator2_subclass(self):
        """Test that ``subcircuit`` returns a subclass of ``Operator2``."""
        assert isinstance(FixedOp, type) and issubclass(FixedOp, Operator2)

    def test_class_variables(self):
        """Test that various class variables of the operator created by ``subcircuit``
        match the original function."""
        assert FixedOp.__name__ == "FixedOp"
        assert FixedOp.__qualname__ == "FixedOp"
        assert FixedOp.__doc__ == "Custom operator created using qp.subcircuit."
        assert FixedOp.__module__ == __name__

    def test_dynamic_argnames_are_set(self):
        """Test that the ``dynamic_argnames`` passed to ``subcircuit`` are set on the class."""
        assert LoopOp.dynamic_argnames == ("phi",)

    def test_wire_argnames_are_set(self):
        """Test that the ``wire_argnames`` passed to ``subcircuit`` are set on the class."""
        assert LoopOp.wire_argnames == ("w",)

    def test_compilable_argnames_are_set(self):
        """Test that the ``compilable_argnames`` passed to ``subcircuit`` are set on the class."""
        assert LoopOp.compilable_argnames == ("n_iters",)

    def test_wire_argnames_default(self):
        """Test that ``wire_argnames`` defaults to ``("wires",)`` when not provided."""
        assert FixedOp.wire_argnames == ("wires",)

    def test_static_argnames(self):
        """Test that the ``static_argnames`` are set on the class."""

        @qp.subcircuit(dynamic_argnames=("phi",), static_argnames=("n",))
        @qp.register_resources({qp.RZ: 1})
        def StaticOp(phi, wires, n):  # pylint: disable=unused-argument
            qp.RZ(phi, wires)

        assert StaticOp.static_argnames == ("n",)

    def test_hybrid_argnames(self):
        """Test that the ``hybrid_argnames`` are set on the class."""

        @qp.subcircuit(dynamic_argnames=("phi",), hybrid_argnames=("n",))
        @qp.register_resources({qp.RZ: 1})
        def HybridOp(phi, wires, n):  # pylint: disable=unused-argument
            qp.RZ(phi, wires)

        assert HybridOp.hybrid_argnames == ("n",)

    def test_arg_specs(self):
        """Test that the ``arg_specs`` passed to ``subcircuit`` are set (and canonicalized) on
        the class."""
        # arg_specs were specified for ``FixedOp``. The Python ``float`` class was used,
        # but it gets canonicalized to ``qp.typing.Float``
        assert FixedOp.arg_specs == {"phi": Float, "wires": Wire[1]}
        # has_fixed_sig gets set automatically
        assert FixedOp.has_fixed_sig

    def test_extra_class_attributes(self):
        """Extra keyword arguments are passed through as class attributes."""

        @qp.subcircuit(custom_attr=42)
        @qp.register_resources({qp.RZ: 1})
        def AttrOp(wires):
            qp.H(wires)

        assert AttrOp.custom_attr == 42

    def test_signature(self):
        """Test that the signature of the returned operator matches the original function."""
        assert str(inspect.signature(FixedOp)) == "(phi, wires)"
        # 'self' should be prepended to __init__'s signature
        assert str(inspect.signature(FixedOp.__init__)) == "(self, phi, wires)"

        assert str(inspect.signature(LoopOp)) == "(phi, w, n_iters)"
        # 'self' should be prepended to __init__'s signature
        assert str(inspect.signature(LoopOp.__init__)) == "(self, phi, w, n_iters)"


class TestDecorator:
    """Tests that ``subcircuit`` can be used as a decorator correctly."""

    def test_call_with_arguments(self):
        """Test that ``@subcircuit(...)`` configures and returns a class."""

        @qp.subcircuit(dynamic_argnames=("phi",))
        @qp.register_resources({qp.RZ: 1})
        def ConfiguredOp(phi, wires):
            qp.RZ(phi, wires)

        assert issubclass(ConfiguredOp, Operator2)
        assert ConfiguredOp.__name__ == "ConfiguredOp"
        assert ConfiguredOp.dynamic_argnames == ("phi",)

    def test_bare_decorator(self):
        """Test that ``@subcircuit`` (no call) can be used as a decorator."""

        @qp.subcircuit
        @qp.register_resources({qp.X: 2})
        def DirectOp(wires):
            qp.X(wires)
            qp.X(wires)

        assert issubclass(DirectOp, Operator2)
        assert DirectOp.__name__ == "DirectOp"
        assert DirectOp.wire_argnames == ("wires",)
        assert DirectOp.dynamic_argnames == ()


class TestDecomposition:
    """Tests that the quantum function is registered as a decomposition rule."""

    def test_decomposition_rule_registered(self):
        """Test that a decomposition rule is registered automatically for the operator
        returned by ``subcircuit``."""
        rules = qp.list_decomps(FixedOp)
        assert len(rules) == 1
        assert rules[0].name == "FixedOp_decomp"

    def test_decomposition_matches_qfunc(self):
        """Test that the decomposition of an instance matches the quantum function body."""
        op = FixedOp(0.5, wires=0)
        decomp = op.decomposition()

        expected = [qp.H(0), qp.RZ(0.5, wires=0)]
        assert len(decomp) == len(expected)
        for actual, exp in zip(decomp, expected):
            qp.assert_equal(actual, exp)

    def test_decomposition_with_control_flow_matches_qfunc(self):
        """Test that a decomposition containing of an instance that contains control flow
        matches the quantum function body."""
        op = LoopOp(0.5, w=0, n_iters=3)
        decomp = op.decomposition()

        expected = [qp.H(0), qp.RZ(0.5, wires=0)] * 3
        assert len(decomp) == len(expected)
        for actual, exp in zip(decomp, expected):
            qp.assert_equal(actual, exp)

    def test_adjoint_decomposition(self):
        """Test that the adjoint of an operator created by ``subcircuit`` decomposes correctly."""
        op = qp.adjoint(FixedOp(0.5, wires=0))
        decomp = op.decomposition()

        # The adjoint reverses the body and adjoints each operator.
        expected = [qp.adjoint(qp.RZ(0.5, wires=0)), qp.adjoint(qp.H(0))]
        assert len(decomp) == len(expected)
        for actual, exp in zip(decomp, expected):
            qp.assert_equal(actual, exp)

    def test_controlled_decomposition(self):
        """Test that the controlled version of an operator created by ``subcircuit`` decomposes
        correctly."""
        op = qp.ctrl(FixedOp(0.5, wires=1), control=0)
        decomp = op.decomposition()

        # Each operator in the body becomes controlled on the same control wire.
        expected = [qp.ctrl(qp.H(1), control=0), qp.ctrl(qp.RZ(0.5, wires=1), control=0)]
        assert len(decomp) == len(expected)
        for actual, exp in zip(decomp, expected):
            qp.assert_equal(actual, exp)


class TestAdditionalDecompositionRules:
    """Tests that additional decomposition rules can be registered for operators created by
    ``subcircuit``, including rules for their controlled and adjoint versions.

    Registrations are scoped with ``qp.decomposition.local_decomps`` so they do not leak into
    the global registry (and other tests).
    """

    @pytest.mark.usefixtures("enable_and_disable_capture")
    def test_register_additional_rule(self):
        """Test that an additional (and valid) decomposition rule can be registered for the
        operator itself."""
        with qp.decomposition.local_decomps():

            # An alternative but equivalent decomposition of ``H`` followed by ``RZ(phi)``.
            @qp.register_resources({qp.H: 1, qp.PhaseShift: 1, qp.GlobalPhase: 1})
            def alt_fixed(phi, wires):
                qp.H(wires)
                qp.PhaseShift(phi, wires)
                qp.GlobalPhase(phi / 2)

            qp.add_decomps(FixedOp, alt_fixed)

            assert {rule.name for rule in qp.list_decomps(FixedOp)} == {
                "FixedOp_decomp",
                "alt_fixed",
            }
            assert_valid(FixedOp(1.5, wires=0))

    @pytest.mark.usefixtures("enable_and_disable_capture")
    def test_register_adjoint_rule(self):
        """Test that a valid adjoint decomposition rule can be registered for the operator."""
        with qp.decomposition.local_decomps():

            @qp.register_resources({qp.RZ: 1, qp.H: 1})
            def adjoint_fixed(base):
                qp.RZ(-base.phi, base.wires)
                qp.H(base.wires)

            qp.add_decomps("Adjoint(FixedOp)", adjoint_fixed)

            assert [rule.name for rule in qp.list_decomps("Adjoint(FixedOp)")] == ["adjoint_fixed"]
            assert_valid(qp.adjoint(FixedOp(1.5, wires=0)))

    # We test only with capture because when capture is disabled, testing the decompositions of the
    # operator fails because assert_valid parametrizes the validation over the number of control wires,
    # leading to custom control dispatches being used if there is only one control wire. Skipping the test
    # in this case is fine because we don't care about PL classic
    @pytest.mark.usefixtures("enable_capture")
    def test_register_controlled_rule(self):
        """Test that a valid controlled decomposition rule can be registered for the operator."""
        with qp.decomposition.local_decomps():

            def _ctrl_resources(
                base, control_wires, control_values, work_wires, work_wire_type
            ):  # pylint: disable=unused-argument
                resources = {}
                resources[
                    qp.ctrl(
                        qp.H(Wire[1]),
                        Wire[len(control_wires)],
                        Bool[len(control_values)],
                        Wire[(len(work_wires))],
                        work_wire_type,
                    )
                ] = 1
                resources[
                    qp.ctrl(
                        qp.RZ(Float, Wire[1]),
                        Wire[len(control_wires)],
                        Bool[len(control_values)],
                        Wire[(len(work_wires))],
                        work_wire_type,
                    )
                ] = 1
                return resources

            @qp.register_resources(_ctrl_resources, exact=False)
            def controlled_fixed(base, control_wires, control_values, work_wires, work_wire_type):
                qp.ctrl(
                    qp.H(base.wires),
                    control_wires,
                    control_values=control_values,
                    work_wires=work_wires,
                    work_wire_type=work_wire_type,
                )
                qp.ctrl(
                    qp.RZ(base.phi, base.wires),
                    control_wires,
                    control_values=control_values,
                    work_wires=work_wires,
                    work_wire_type=work_wire_type,
                )

            qp.add_decomps("C(FixedOp)", controlled_fixed)

            assert [rule.name for rule in qp.list_decomps("C(FixedOp)")] == ["controlled_fixed"]
            assert_valid(qp.ctrl(FixedOp(1.5, wires=0), control=[1, 2, 3]))
