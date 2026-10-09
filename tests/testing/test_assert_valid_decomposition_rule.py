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
Tests for ``qp.testing.assert_valid_decomposition_rule`` and ``qp.testing.decomp_rule_to_tape``.
"""

# pylint: disable=too-few-public-methods

import numpy as np
import pytest

import pennylane as qp
from pennylane.core.operator import Operator
from pennylane.testing import assert_valid_decomposition_rule, decomp_rule_to_tape
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
    """Tests for assert_valid_decomposition_rule."""

    @pytest.mark.parametrize("capture", [False, pytest.param(True, marks=pytest.mark.capture)])
    def test_valid_rule(self, capture):
        """Test that a correct rule passes in both capture modes."""
        assert qp.capture.enabled() is capture
        assert_valid_decomposition_rule(qp.CNOT([0, 1]), _cnot_to_cz)

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
            assert_valid_decomposition_rule(op, rule_float_counts)

        rule_float_counts = qp.register_resources({qp.X: 2, qp.Y: 3.0})(rule)
        with pytest.raises(
            AssertionError,
            match="Resource count for 'PauliY' in 'MyOp' decomp rule 'rule' must be an integer",
        ):
            assert_valid_decomposition_rule(op, rule_float_counts)

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
            assert_valid_decomposition_rule(op, rule)

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
        with pytest.raises(AssertionError, match="The numbers are off"):
            assert_valid_decomposition_rule(op, rule_wrong_numbers)

        rule_wrong_ops = qp.register_resources({qp.X: 2, qp.Z: 2})(rule)
        with pytest.raises(AssertionError, match="Missing entirely in gate counts"):
            assert_valid_decomposition_rule(op, rule_wrong_ops)

    def test_bad_new_decomposition_rule_inexact(self):
        """Test that an informative error is raised if the
        inexact resources of a decomposition rule are not correct."""

        class MyOp(Operator):
            num_wires = 2

        def rule(wires):
            qp.X(wires[0])
            qp.X(wires[1])
            qp.Y(wires[0])
            qp.Y(wires[1])

        rule_wrong_ops = qp.register_resources({qp.X: 2, qp.Z: 2}, exact=False)(rule)
        op = MyOp([0, 1])
        with pytest.raises(AssertionError, match="Gate counts expected from"):
            assert_valid_decomposition_rule(op, rule_wrong_ops)

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

        spy = mocker.spy(qp, "matrix")
        assert_valid_decomposition_rule(op, rule)
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

        assert_valid_decomposition_rule(MyOp([0, 1, 2]), rule)

    @pytest.mark.capture
    def test_new_decomposition_rule_capture_operator2(self):
        """Operator2 dynamic and wire arguments are forwarded as capture inputs."""

        @qp.register_resources({OneWireDynOp: 3})
        def rule(phi, wires):  # pylint: disable=unused-argument
            @qp.for_loop(3)
            def loop(i):
                OneWireDynOp(phi, wires=i)

            loop()  # pylint: disable=no-value-for-parameter

        assert_valid_decomposition_rule(OneWireDynOp(0.5, wires=0), rule)
