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
Tests for the SignedeutSquare template.
"""

from itertools import product

import pytest

import pennylane as qp
from pennylane import numpy as np

# from pennylane.ops import Controlled
from pennylane.ops.functions.assert_valid import _test_decomposition_rule
from pennylane.templates.subroutines.arithmetic.signed_out_square import SignedOutSquare


@pytest.mark.parametrize("output_wires_zeroed", [False, True])
@pytest.mark.jax
def test_standard_validity_signed_out_square(output_wires_zeroed):
    """Check the operation using the assert_valid function."""
    x_wires = [0, 1, 2, 3]
    output_wires = [4, 5, 6, 7, 8, 9, 10]
    work_wires = [11, 12, 13, 14, 15, 16, 17, 18, 19]
    op = SignedOutSquare(x_wires, output_wires, work_wires, output_wires_zeroed)
    qp.ops.functions.assert_valid(op)


def _test_square_correctness(all_wires, rule, seed, output_wires_zeroed, use_jit):
    """Test the correctness of a decomposition rule for ``SignedOutSquare``."""
    if use_jit:
        import jax

        jax.config.update("jax_enable_x64", True)

    x_wires, output_wires, work_wires = all_wires

    dev = qp.device("lightning.qubit")

    @qp.set_shots(10)
    @qp.qnode(dev)
    def circuit(x_sign, x_u, b):
        qp.BasisEmbedding(x_sign, wires=x_wires[:1])
        qp.BasisEmbedding(x_u, wires=x_wires[1:])
        qp.BasisEmbedding(b, wires=output_wires)
        rule(x_wires, output_wires, work_wires, output_wires_zeroed)
        return (
            qp.sample(wires=x_wires),
            qp.sample(wires=output_wires),
            qp.sample(wires=(work_wires or None)),
        )

    if use_jit:
        circuit = jax.jit(circuit)

    mod = 2 ** len(output_wires)
    rng = np.random.default_rng(seed)
    num_x = 2 ** (len(x_wires) - 1)
    x_us = rng.choice(num_x, size=min(num_x, 5))
    x_signs = [0, 1]
    if output_wires_zeroed:
        bs = [0]
    else:
        bs = [0, mod // 2 + 1, mod - 1]

    for x_sign, x_u, b in product(x_signs, x_us, bs):
        output = circuit(x_sign, x_u, b)
        assert len(output) == 3
        out_ints = [int("".join(map(str, out[0])), 2) for out in output]
        expected = [int(x_sign * num_x + x_u), int((b + (x_u - num_x * x_sign) ** 2) % mod), 0]

        # n = len(x_wires)
        # tmp_exp_out = ((2 * x**2 + 2 ** (2 * n) - (2**n - x) - 2 * x * 2**n) // 2) % mod
        # tmp_exp_out = ((2 * x**2 - 2 * x * 2**n) // 2) % mod
        if len(work_wires) > 0:
            assert out_ints == expected, f"\n{out_ints}\n{expected} ({b=})"
            # {(out_ints[1], tmp_exp_out)}
        else:
            # Skip work wire check
            assert out_ints[:-1] == expected[:-1], f"\n{out_ints[:-1]}\n{expected[:-1]} ({b=})"


class TestSignedOutSquare:
    """Test the SignedOutSquare template."""

    @pytest.mark.parametrize("output_wires_zeroed", [False, True])
    @pytest.mark.parametrize(
        ("x_wires", "output_wires", "work_wires"),
        [
            (
                [0, 1],
                [3, 4, 5],
                [6, 7, 8],
            ),
            (
                [0, 1, 2],
                [3, 4, 5, 6, 7],
                [8, 9, 10, 11, 12],
            ),
            (
                [0, 1, 2],
                [3, 4, 5, 6, 7, 8, 9],
                [10, 11, 12, 13, 14, 15, 16],
            ),
            (
                [0, 1, 2, 3],
                [4, 5, 6, 7, 8],
                [9, 10, 11, 12, 13],
            ),
            (
                [0, 1, 2, 3],
                [4, 5, 6, 7],
                [9, 10, 11, 12, 13],
            ),
        ],
    )
    @pytest.mark.parametrize("use_jit", [pytest.param(True, marks=pytest.mark.jax), False])
    def test_operation_result(
        self,
        x_wires,
        output_wires,
        work_wires,
        output_wires_zeroed,
        use_jit,
        seed,
    ):  # pylint: disable=too-many-arguments
        """Test the correctness of the SignedOutSquare template output."""
        all_wires = (x_wires, output_wires, work_wires)
        _test_square_correctness(
            all_wires, SignedOutSquare.compute_decomposition, seed, output_wires_zeroed, use_jit
        )

    @pytest.mark.catalyst
    @pytest.mark.external
    @pytest.mark.usefixtures("enable_graph_decomposition")
    @pytest.mark.parametrize("output_wires_zeroed", [False, True])
    def test_qjit_dynamic_wires(self, output_wires_zeroed):
        """Test the SignedOutSquare template with dynamic wires."""
        x_wires = np.array([0, 1, 2, 3])
        output_wires = np.array([4, 5, 6, 7, 8, 9])
        work_wires = np.array([10, 11, 12, 13, 14, 15])

        dev = qp.device("lightning.qubit")

        x = 13  # = (1101)_2 => -3
        mod = 2 ** len(output_wires)
        if output_wires_zeroed:
            z = 0
        else:
            z = mod - 2  # Some number close to causing overflows

        # gate_set = {"HybridAdjoint", "TemporaryAND", "CNOT", "X", "Adjoint(TemporaryAND)", "BasisEmbedding", "MidMeasureMP", "Cond"}

        @qp.qjit
        @qp.set_shots(1)
        # @qp.decompose(
        # gate_set=gate_set, max_expansion=2, fixed_decomps={"C(SemiAdder)": qp.list_decomps("C(SemiAdder)")[0]}
        # )
        @qp.qnode(dev)
        def circuit(x, z, x_wires, work_wires):
            qp.BasisEmbedding(x, wires=x_wires)
            qp.BasisEmbedding(z, wires=output_wires)
            SignedOutSquare(x_wires, output_wires, work_wires, output_wires_zeroed)
            return (
                qp.sample(wires=x_wires),
                qp.sample(wires=output_wires),
                qp.sample(wires=work_wires),
            )

        output = circuit(x, z, x_wires, work_wires)
        out_ints = [int("".join(map(str, out[0])), 2) for out in output]
        assert np.allclose(out_ints, [x, (z + (x - 16) ** 2) % mod, 0])

    @pytest.mark.parametrize(
        ("x_wires", "output_wires", "work_wires", "msg_match"),
        [
            (
                [0, 1, 2],
                [3, 4, 5],
                [3, 10, 11],
                "None of the wires in work_wires should be included in output_wires.",
            ),
            (
                [0, 1, 2],
                [3, 4, 5],
                [1, 10, 9],
                "None of the wires in work_wires should be included in x_wires.",
            ),
            (
                [0, 1, 2],
                [2, 4, 5],
                [9, 10, 6],
                "None of the wires in output_wires should be included in x_wires.",
            ),
        ],
    )
    def test_wires_overlap_error(
        self, x_wires, output_wires, work_wires, msg_match
    ):  # pylint: disable=too-many-arguments
        """Test an error is raised when some registers overlap."""
        with pytest.raises(ValueError, match=msg_match):
            SignedOutSquare(x_wires, output_wires, work_wires)

    @pytest.mark.parametrize(
        "x_wires, output_wires, work_wires, zeroed, should_raise",
        [
            ([0, 1, 2], [3, 4, 5], [6, 7], True, True),
            ([0, 1, 2], [3, 4, 5], [6, 7, 8], True, False),
            ([0, 1, 2], [3, 4, 5], [6, 7], True, True),
            ([0, 1], [3, 4, 5], [6, 7, 8], True, False),
            ([0, 1], [3, 4, 5], [6, 7], True, False),
            ([0, 1], [3, 4, 5], [6], True, True),
            ([0, 1, 2], [3, 4, 5], [6, 7, 8], False, False),
            ([0, 1, 2], [3, 4, 5], [6, 7], False, True),
            ([0, 1], [3, 4, 5], [6, 7, 8], False, False),
            ([0, 1], [3, 4, 5], [6, 7], False, True),
            ([0], [3, 4, 5], [6, 7, 8], False, False),
            ([0], [3, 4, 5], [6, 7], False, True),
            ([0], [3, 4, 5], [6], False, True),
        ],
    )
    def test_work_wire_number(
        self, x_wires, output_wires, work_wires, zeroed, should_raise
    ):  # pylint: disable=too-many-arguments
        """Test an error is raised (only) when too few work wires are supplied."""
        if should_raise:
            msg_match = "SignedOutSquare requires at least"
            with pytest.raises(ValueError, match=msg_match):
                SignedOutSquare(x_wires, output_wires, work_wires, output_wires_zeroed=zeroed)
        else:
            SignedOutSquare(x_wires, output_wires, work_wires, output_wires_zeroed=zeroed)

    @pytest.mark.parametrize("output_wires_zeroed", [False, True])
    def test_decomposition(self, output_wires_zeroed):
        """Test that compute_decomposition and decomposition work as expected."""
        x_wires, output_wires, work_wires = (
            [0, 1, 2],
            [3, 4, 5, 6, 7],
            [8, 9, 10, 11, 12],
        )
        decomp = SignedOutSquare(
            x_wires, output_wires, work_wires, output_wires_zeroed
        ).decomposition()
        expected = [
            qp.OutSquare([1, 2], [3, 4, 5, 6, 7], [8, 9, 10, 11, 12], output_wires_zeroed),
            qp.BasisState([1], [1]),
            qp.X(4),
            qp.X(8),
            qp.ctrl(
                qp.SemiAdder([1, 2], [3, 4], [8]),
                control=[0],
                work_wires=[9, 10, 11, 12],
                work_wire_type="zeroed",
            ),
            qp.X(4),
            qp.ops.MidMeasure(8, reset=True),
            qp.BasisState([1], [1]),
            qp.X(3),
            # qp.X(4),
            qp.SemiAdder([0], [3], [8, 9, 10, 11, 12]),
            qp.X(3),
            # qp.X(4),
        ]
        for op1, op2 in zip(decomp, expected):
            if isinstance(op1, qp.ops.MidMeasure):
                assert isinstance(op2, qp.ops.MidMeasure)
                assert op1.wires == op2.wires
                assert op1.reset == op2.reset
                assert op1.postselect == op2.postselect
            else:
                qp.assert_equal(op1, op2)

    @pytest.mark.parametrize(
        ("x_wires", "output_wires", "work_wires", "output_wires_zeroed"),
        [
            ([0, 1, 2, 3], [4, 5, 6], [9, 10, 11, 12, 13], True),
            ([0, 1, 2, 3], [4, 5, 6], [9, 10, 11, 12, 13], False),
            ([0, 1], [3, 5], [9, 10], True),
            ([0, 1], [3, 5], [9, 10], False),
            ([0, 1], [3, 4, 5, 6], [9, 10, 11], True),
            ([0, 1], [3, 4, 5, 6], [9, 10, 11, 12, 13], False),
            ([0, 1], [3, 4, 5, 6, 7], [9, 10, 11, 12, 13], True),
            ([0, 1], [3, 4, 5, 6, 7], [9, 10, 11, 12, 13], False),
        ],
    )
    @pytest.mark.parametrize("use_jit", [pytest.param(True, marks=pytest.mark.jax), False])
    def test_decomposition_new(
        self,
        x_wires,
        output_wires,
        work_wires,
        output_wires_zeroed,
        use_jit,
        seed,
    ):  # pylint: disable=too-many-arguments
        """Tests the decomposition rule implemented with the new system."""
        op = SignedOutSquare(x_wires, output_wires, work_wires, output_wires_zeroed)
        for rule in qp.list_decomps(SignedOutSquare):
            _test_decomposition_rule(op, rule)
            assert rule.is_applicable(**op.resource_params)
            all_wires = (x_wires, output_wires, work_wires)
            _test_square_correctness(all_wires, rule, seed, output_wires_zeroed, use_jit)
