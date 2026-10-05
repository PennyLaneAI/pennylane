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
Tests for the internal adder block operators and the arithmetic templates built from them.
"""

from collections import Counter
from itertools import product

import pytest

import pennylane as qp
from pennylane.ops.functions.assert_valid import _test_decomposition_rule
from pennylane.templates.subroutines.arithmetic.adder_blocks import (
    CtrlRightFullAdder,
    CtrlRightHalfAdder,
    LeftFullAdder,
    RightFullAdder,
    RightHalfAdder,
)
from pennylane.templates.subroutines.arithmetic.out_multiplier import (
    OutMultiplier,
    _out_multiplier_with_caddsub,
)
from pennylane.templates.subroutines.arithmetic.out_square import (
    OutSquare,
    _out_square_with_adder_zeroed,
    _out_square_with_caddsub,
)
from pennylane.templates.subroutines.arithmetic.signed_out_square import SignedOutSquare


def _run_on_basis_state(make_op, bits):
    """Apply the operator created by ``make_op`` to the basis state ``bits`` on the wires
    ``range(len(bits))`` and return the resulting basis state."""

    @qp.set_shots(1)
    @qp.qnode(qp.device("default.qubit", wires=len(bits)))
    def circuit():
        qp.BasisState(bits, wires=range(len(bits)))
        make_op()
        return qp.sample(wires=range(len(bits)))

    return [int(b) for b in circuit()[0]]


def _maj(c, x, y):
    return int(c + x + y >= 2)


class TestAdderBlocks:
    """Test the action and decompositions of the adder block operators."""

    @pytest.mark.parametrize(("c", "x", "y"), list(product([0, 1], repeat=3)))
    def test_left_full_adder_block(self, c, x, y):
        """Test that LeftFullAdder computes the output carry."""
        out = _run_on_basis_state(lambda: LeftFullAdder([0, 1, 2, 3]), [c, x, y, 0])
        assert out == [c, x ^ c, y ^ c, _maj(c, x, y)]

    @pytest.mark.parametrize(("c", "x", "y"), list(product([0, 1], repeat=3)))
    def test_right_full_adder_block(self, c, x, y):
        """Test that RightFullAdder uncomputes the carry and writes the sum bit."""
        bits = [c, x ^ c, y ^ c, _maj(c, x, y)]
        out = _run_on_basis_state(lambda: RightFullAdder([0, 1, 2, 3]), bits)
        assert out == [c, x, x ^ y ^ c, 0]

    @pytest.mark.parametrize(("c", "y"), list(product([0, 1], repeat=2)))
    def test_right_half_adder_block(self, c, y):
        """Test that RightHalfAdder uncomputes the carry and writes the sum bit."""
        out = _run_on_basis_state(lambda: RightHalfAdder([0, 1, 2]), [c, y, c & y])
        assert out == [c, y ^ c, 0]

    @pytest.mark.parametrize(("ctrl_0", "ctrl_1", "c", "x", "y"), list(product([0, 1], repeat=5)))
    def test_ctrl_right_full_adder_block(
        self, ctrl_0, ctrl_1, c, x, y
    ):  # pylint: disable=too-many-arguments
        """Test that CtrlRightFullAdder acts like RightFullAdder if all controls are
        set, and like the inverse of LeftFullAdder otherwise."""
        bits = [ctrl_0, ctrl_1, c, x ^ c, y ^ c, _maj(c, x, y), 0]
        out = _run_on_basis_state(
            lambda: CtrlRightFullAdder([0, 1], [2, 3, 4, 5], [6], "zeroed"), bits
        )
        new_y = x ^ y ^ c if ctrl_0 and ctrl_1 else y
        assert out == [ctrl_0, ctrl_1, c, x, new_y, 0, 0]

    @pytest.mark.parametrize(("ctrl_0", "ctrl_1", "c", "y"), list(product([0, 1], repeat=4)))
    def test_ctrl_right_half_adder_block(self, ctrl_0, ctrl_1, c, y):
        """Test that CtrlRightHalfAdder acts like RightHalfAdder if all controls are
        set, and like the inverse of TemporaryAND otherwise."""
        bits = [ctrl_0, ctrl_1, c, y, c & y]
        out = _run_on_basis_state(lambda: CtrlRightHalfAdder([0, 1], [2, 3, 4]), bits)
        new_y = y ^ c if ctrl_0 and ctrl_1 else y
        assert out == [ctrl_0, ctrl_1, c, new_y, 0]

    @pytest.mark.usefixtures("enable_and_disable_capture")
    @pytest.mark.parametrize(
        "op",
        [
            LeftFullAdder([0, 1, 2, 3]),
            RightFullAdder([0, 1, 2, 3]),
            RightHalfAdder([0, 1, 2]),
            CtrlRightFullAdder([4, 5], [0, 1, 2, 3], [6], "zeroed"),
            CtrlRightHalfAdder([4], [0, 1, 2]),
        ],
    )
    def test_decomposition_rule(self, op):
        """Test the decomposition rules of the adder blocks."""
        for rule in qp.list_decomps(type(op)):
            _test_decomposition_rule(op, rule)


_ELEMENTARY_GATE_SET = {
    "CNOT",
    "TemporaryAND",
    "Adjoint(TemporaryAND)",
    "PauliX",
    "Toffoli",
    "MultiControlledX",
    "MultiX",
    "GlobalPhase",
}


def _r(start, stop):
    return list(range(start, stop))


_GATE_COUNT_CASES = [
    (
        lambda: qp.SemiAdder(_r(0, 3), _r(3, 7), _r(7, 10)),
        None,
        {"Adjoint(TemporaryAND)": 3, "CNOT": 14, "TemporaryAND": 3},
    ),
    (
        lambda: qp.SemiAdder(_r(0, 2), _r(2, 7), _r(7, 11)),
        None,
        {"Adjoint(TemporaryAND)": 4, "CNOT": 10, "TemporaryAND": 4},
    ),
    (
        lambda: qp.SemiAdder(_r(0, 5), _r(5, 8), _r(8, 10)),
        None,
        {"Adjoint(TemporaryAND)": 2, "CNOT": 9, "TemporaryAND": 2},
    ),
    (lambda: qp.SemiAdder(_r(0, 2), [2], []), None, {"CNOT": 1}),
    (
        lambda: qp.ctrl(qp.SemiAdder(_r(0, 3), _r(3, 7), _r(7, 10)), control=[10]),
        None,
        {"Adjoint(TemporaryAND)": 3, "CNOT": 12, "TemporaryAND": 3, "Toffoli": 4},
    ),
    (
        lambda: qp.ctrl(
            qp.SemiAdder(_r(0, 2), _r(2, 7), _r(7, 11)), control=[11, 12], control_values=[1, 0]
        ),
        None,
        {
            "Adjoint(TemporaryAND)": 4,
            "CNOT": 6,
            "MultiControlledX": 5,
            "PauliX": 2,
            "TemporaryAND": 4,
        },
    ),
    (
        lambda: qp.ctrl(qp.SemiAdder(_r(0, 5), _r(5, 8), _r(8, 10)), control=[10]),
        None,
        {"Adjoint(TemporaryAND)": 2, "CNOT": 6, "TemporaryAND": 2, "Toffoli": 4},
    ),
    (
        lambda: OutSquare(_r(0, 3), _r(3, 9), _r(9, 21), True),
        {OutSquare: _out_square_with_adder_zeroed},
        {"Adjoint(TemporaryAND)": 3, "CNOT": 11, "MultiControlledX": 4, "TemporaryAND": 5},
    ),
    (
        lambda: OutSquare(_r(0, 4), _r(4, 12), _r(12, 24), True),
        {OutSquare: _out_square_with_adder_zeroed},
        {"Adjoint(TemporaryAND)": 7, "CNOT": 24, "MultiControlledX": 8, "TemporaryAND": 10},
    ),
    (
        lambda: OutSquare(_r(0, 3), _r(3, 9), _r(9, 21), True),
        {OutSquare: _out_square_with_caddsub},
        {
            "Adjoint(TemporaryAND)": 14,
            "CNOT": 42,
            "MultiControlledX": 8,
            "MultiX": 4,
            "PauliX": 4,
            "TemporaryAND": 14,
        },
    ),
    (
        lambda: OutSquare(_r(0, 4), _r(4, 11), _r(11, 23), False),
        {OutSquare: _out_square_with_caddsub},
        {
            "Adjoint(TemporaryAND)": 22,
            "CNOT": 75,
            "MultiControlledX": 12,
            "MultiX": 4,
            "PauliX": 8,
            "TemporaryAND": 22,
        },
    ),
    (
        lambda: OutMultiplier(_r(0, 3), _r(3, 5), _r(5, 10), work_wires=_r(10, 20)),
        {OutMultiplier: _out_multiplier_with_caddsub},
        {
            "Adjoint(TemporaryAND)": 22,
            "CNOT": 69,
            "MultiControlledX": 12,
            "PauliX": 31,
            "TemporaryAND": 22,
        },
    ),
    (
        lambda: OutMultiplier(
            _r(0, 2), _r(2, 5), _r(5, 11), work_wires=_r(11, 21), output_wires_zeroed=True
        ),
        {OutMultiplier: _out_multiplier_with_caddsub},
        {
            "Adjoint(TemporaryAND)": 19,
            "CNOT": 78,
            "MultiControlledX": 8,
            "PauliX": 29,
            "TemporaryAND": 19,
        },
    ),
    (
        lambda: SignedOutSquare(_r(0, 3), _r(3, 9), _r(9, 21), True),
        None,
        {
            "Adjoint(TemporaryAND)": 4,
            "CNOT": 12,
            "MultiControlledX": 5,
            "MultiX": 2,
            "PauliX": 8,
            "TemporaryAND": 5,
        },
    ),
    (
        lambda: SignedOutSquare(_r(0, 4), _r(4, 13), _r(13, 25), False),
        None,
        {
            "Adjoint(TemporaryAND)": 38,
            "CNOT": 79,
            "MultiControlledX": 13,
            "MultiX": 6,
            "PauliX": 14,
            "TemporaryAND": 38,
        },
    ),
]


@pytest.mark.usefixtures("enable_graph_decomposition")
@pytest.mark.parametrize(("make_op", "fixed_decomps", "expected"), _GATE_COUNT_CASES)
def test_elementary_gate_counts(make_op, fixed_decomps, expected):
    """Test the gate counts of arithmetic templates built from adder blocks, fully decomposed
    into elementary gates, against fixed reference values."""
    tape = qp.tape.make_qscript(make_op)()
    [decomposed], _ = qp.transforms.decompose(
        tape, gate_set=_ELEMENTARY_GATE_SET, fixed_decomps=fixed_decomps
    )
    assert dict(Counter(op.name for op in decomposed.operations)) == expected
