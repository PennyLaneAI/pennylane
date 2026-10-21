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
Tests for ``qp.testing.assert_valid_decomp_rule``, ``qp.testing.assert_decomposable`` and
``qp.testing.decomp_rule_to_tape``.
"""

# pylint: disable=too-few-public-methods

import numpy as np
import pytest

import pennylane as qp
from pennylane.core.operator import Operator
from pennylane.testing import assert_decomposable, assert_valid_decomp_rule, decomp_rule_to_tape
from pennylane.typing import Wire
from tests.core.operator.operator2_utils import OneWireDynOp


@qp.register_resources({qp.H: 2, qp.CZ: 1})
def _cnot_to_cz(wires):
    qp.H(wires[1])
    qp.CZ(wires)
    qp.H(wires[1])


def _assert_ops_equal(ops, expected):
    for op, exp in zip(ops, expected, strict=True):
        qp.assert_equal(op, exp, check_interface=False)


class TestDecompRuleToTape:
    """Tests for decomp_rule_to_tape."""

    def test_eager(self):
        """Test that the operations queued by the rule are recorded."""
        tape = decomp_rule_to_tape(qp.CNOT([0, 1]), _cnot_to_cz)
        _assert_ops_equal(tape.operations, [qp.H(1), qp.CZ([0, 1]), qp.H(1)])

    @pytest.mark.capture
    def test_capture(self):
        """Test that a captured rule gives the same operations as an eager one."""
        tape = decomp_rule_to_tape(qp.CNOT([0, 1]), _cnot_to_cz)
        _assert_ops_equal(tape.operations, [qp.H(1), qp.CZ([0, 1]), qp.H(1)])

    @pytest.mark.parametrize("capture", [False, pytest.param(True, marks=pytest.mark.capture)])
    def test_operator2_dynamic_args(self, capture):
        """Test that the dynamic arguments of an Operator2 reach the rule."""

        @qp.register_resources({OneWireDynOp: 2})
        def rule(phi, wires):
            OneWireDynOp(phi, wires=wires)
            OneWireDynOp(2 * phi, wires=wires)

        assert qp.capture.enabled() is capture
        tape = decomp_rule_to_tape(OneWireDynOp(0.5, wires=0), rule)
        _assert_ops_equal(tape.operations, [OneWireDynOp(0.5, 0), OneWireDynOp(1.0, 0)])

    @pytest.mark.capture
    def test_operator1_hyperparameters_captured(self):
        """Test that the hyperparameters of a legacy operator are passed to a captured rule."""

        class MyOp(Operator):
            num_wires = 1

            def __init__(self, phi, wires, n):
                super().__init__(phi, wires=wires)
                self.hyperparameters["n"] = n

        @qp.register_resources({qp.RX: 2})
        def rule(phi, wires, n):
            for _ in range(n):
                qp.RX(phi, wires)

        tape = decomp_rule_to_tape(MyOp(0.5, 0, n=2), rule)
        _assert_ops_equal(tape.operations, [qp.RX(0.5, 0), qp.RX(0.5, 0)])


class TestAssertValidDecompositionRule:
    """Tests for assert_valid_decomp_rule."""

    @pytest.mark.usefixtures("enable_and_disable_capture")
    def test_valid_rule(self):
        """Test that a correct rule passes in both capture modes."""
        assert_valid_decomp_rule(qp.CNOT([0, 1]), _cnot_to_cz)

    def test_rule_with_non_int_counts(self):
        """Test that a rule with non-int counts raises an error."""

        class MyOp(Operator):
            num_wires = 2

        op = MyOp([0, 1])

        def rule(wires):
            qp.X(wires[0])
            qp.X(wires[1])
            qp.Y(wires[0])
            qp.Y(wires[1])

        rule_float_counts = qp.register_resources({qp.X: 2.0, qp.Y: 3.0})(rule)
        with pytest.raises(
            AssertionError,
            match="Resource count for 'PauliX' in 'MyOp' decomp rule 'rule' must be an integer",
        ):
            assert_valid_decomp_rule(op, rule_float_counts)

        rule_float_counts = qp.register_resources({qp.X: 2, qp.Y: 3.0})(rule)
        with pytest.raises(
            AssertionError,
            match="Resource count for 'PauliY' in 'MyOp' decomp rule 'rule' must be an integer",
        ):
            assert_valid_decomp_rule(op, rule_float_counts)

    @pytest.mark.parametrize("numpy_int", (np.int64, np.int32, np.uint8))
    def test_numpy_ints_are_not_allowed(self, numpy_int):
        """Test that numpy integer types are not allowed."""

        class MyOp(Operator):
            num_wires = 2

        op = MyOp([0, 1])

        def rule(wires):
            qp.X(wires[0])
            qp.X(wires[1])

        rule = qp.register_resources({qp.X: numpy_int(2)})(rule)

        with pytest.raises(
            AssertionError,
            match="Resource count for 'PauliX' in 'MyOp' decomp rule 'rule' must be an integer",
        ):
            assert_valid_decomp_rule(op, rule)

    def test_bad_new_decomposition_rule_exact(self):
        """Test that an informative error is raised if the
        claimed-to-be-exact resources of a decomposition rule are not correct."""

        class MyOp(Operator):
            num_wires = 2

        op = MyOp([0, 1])

        def rule(wires):
            qp.X(wires[0])
            qp.X(wires[1])
            qp.Y(wires[0])
            qp.Y(wires[1])

        rule_wrong_numbers = qp.register_resources({qp.X: 2, qp.Y: 3})(rule)
        with pytest.raises(AssertionError, match=r"PauliY :      2  !=  3$"):
            assert_valid_decomp_rule(op, rule_wrong_numbers)

        rule_wrong_ops = qp.register_resources({qp.X: 2, qp.Z: 2})(rule)
        with pytest.raises(AssertionError, match=r"PauliY :\s+2  !=  0\n  PauliZ :\s+0  !=  2$"):
            assert_valid_decomp_rule(op, rule_wrong_ops)

        rule_zero_count = qp.register_resources({qp.X: 2, qp.Y: 2, qp.Z: 0})(rule)
        assert_valid_decomp_rule(op, rule_zero_count)

    def test_bad_new_decomposition_rule_inexact(self):
        """Test that an informative error is raised if the
        decomposition contains an operator type that is not accounted for in
        the resource function even when exact=False."""

        class MyOp(Operator):
            num_wires = 2

        def rule(wires):
            qp.X(wires[0])
            qp.X(wires[1])
            qp.Y(wires[0])
            qp.Y(wires[1])

        rule_wrong_ops = qp.register_resources({qp.X: 2, qp.Z: 2}, exact=False)(rule)
        op = MyOp([0, 1])
        with pytest.raises(
            AssertionError,
            match=r"The resources is missing gates in the actual decomposition: \['PauliY'\]",
        ):
            assert_valid_decomp_rule(op, rule_wrong_ops)

    def test_new_decomposition_rule_with_mcm_skips_matrix_check(self, mocker):
        """Test that matrix check is skipped for decompositions containing mid-circuit measurements."""

        class MyOp(Operator):
            num_wires = 1

            @staticmethod
            def compute_matrix():
                return qp.Hadamard.compute_matrix()

        op = MyOp([0])

        def mcm_rule(wires):
            qp.ops.measure(wires[0])

        rule = qp.register_resources({qp.ops.MidMeasure(wires=Wire[1]): 1})(mcm_rule)

        spy = mocker.spy(qp.ops.functions, "matrix")
        assert_valid_decomp_rule(op, rule)
        spy.assert_not_called()

    @pytest.mark.capture
    def test_new_decomposition_rule_capture(self):
        """A captured decomposition is converted to a tape before validating its resources."""

        class MyOp(Operator):
            num_wires = 3

        @qp.register_resources({qp.S: 3})
        def rule(wires):  # pylint: disable=unused-argument
            @qp.for_loop(3)
            def loop(i):
                qp.S(i)

            loop()  # pylint: disable=no-value-for-parameter

        assert_valid_decomp_rule(MyOp([0, 1, 2]), rule)

    @pytest.mark.capture
    def test_new_decomposition_rule_capture_operator2(self):
        """Operator2 dynamic and wire arguments are forwarded as capture inputs."""

        @qp.register_resources({OneWireDynOp: 3})
        def rule(phi, wires):  # pylint: disable=unused-argument
            @qp.for_loop(3)
            def loop(i):
                OneWireDynOp(phi, wires=i)

            loop()  # pylint: disable=no-value-for-parameter

        assert_valid_decomp_rule(OneWireDynOp(0.5, wires=0), rule)


class MyOneWireOp(Operator):
    """A one-wire operator without any registered decomposition rules."""

    num_wires = 1


@qp.register_condition(lambda: False)
@qp.register_resources({qp.Y: 5})
def _never_applicable(wires):
    raise NotImplementedError


@qp.register_resources({qp.X: 1})
def _to_x(wires):
    qp.X(wires)


@qp.register_resources({qp.X: 2})
def _to_x_bad_resources(wires):
    qp.X(wires)


class TestAssertDecomposable:
    """Tests for ``qp.testing.assert_decomposable``."""

    def test_valid_rules(self):
        """Test that no error is raised if the registered rules are valid."""
        assert_decomposable(qp.CNOT([0, 1]))

    def test_no_rules(self):
        """Test that an error is raised if no rules are registered for the operator."""
        with pytest.raises(AssertionError, match="No decomposition rules are registered"):
            assert_decomposable(MyOneWireOp(0))

    def test_no_applicable_rules(self):
        """Test that an error is raised if none of the registered rules are applicable."""
        with qp.decomposition.local_decomps():
            qp.add_decomps(MyOneWireOp, _never_applicable)
            with pytest.raises(AssertionError, match="None of the decomposition rules"):
                assert_decomposable(MyOneWireOp(0))

    def test_inapplicable_rules_are_skipped(self):
        """Test that inapplicable rules are not checked if another rule is applicable."""
        with qp.decomposition.local_decomps():
            qp.add_decomps(MyOneWireOp, _never_applicable, _to_x)
            assert_decomposable(MyOneWireOp(0))

    def test_invalid_applicable_rule(self):
        """Test that an error is raised if an applicable rule is invalid."""
        with qp.decomposition.local_decomps():
            qp.add_decomps(MyOneWireOp, _to_x, _to_x_bad_resources)
            expected_message = "_to_x_bad_resources for MyOneWireOp fails"
            with pytest.raises(AssertionError, match=expected_message) as exc_info:
                assert_decomposable(MyOneWireOp(0))
            assert "do not match the resource estimate" in str(exc_info.value.__cause__)
