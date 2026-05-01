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
Tests for the OutSquare template.
"""

import pytest

import pennylane as qp
from pennylane import numpy as np
from pennylane.ops import Controlled
from pennylane.ops.functions.assert_valid import _test_decomposition_rule
from pennylane.templates.subroutines.arithmetic.out_square import OutSquare


@pytest.mark.parametrize("output_wires_zeroed", [False, True])
@pytest.mark.jax
def test_standard_validity_out_square(output_wires_zeroed):
    """Check the operation using the assert_valid function."""
    x_wires = [0, 1, 2, 3]
    output_wires = [4, 5, 6, 7, 8, 9, 10]
    work_wires = [11, 12, 13, 14, 15, 16, 17, 18, 19]
    op = OutSquare(x_wires, output_wires, work_wires, output_wires_zeroed)
    qp.ops.functions.assert_valid(op)


def _test_square_correctness(all_wires, rule, seed, output_wires_zeroed, use_jit):
    """Test the correctness of a decomposition rule for an ``OutMultiplier`` op."""
    x_wires, output_wires, work_wires = all_wires
    total_wires = x_wires + output_wires
    if work_wires:
        total_wires += work_wires

    dev = qp.device("lightning.qubit")

    @qp.qnode(dev)
    def circuit(x_state, y_state, output_state):
        qp.StatePrep(x_state, x_wires)
        qp.StatePrep(y_state, output_wires)
        rule(x_wires, output_wires, work_wires, output_wires_zeroed)
        qp.adjoint(qp.StatePrep)(output_state, x_wires + output_wires)
        return qp.probs(wires=total_wires)

    if use_jit:
        fixed_decomps = {"C(SemiAdder)": qp.list_decomps("C(SemiAdder)")[0]}
        qp.qjit(qp.decompose(circuit, max_expansion=2, fixed_decomps=fixed_decomps))

    rng = np.random.default_rng(seed)

    num_x = 2 ** len(x_wires)
    x_state = rng.random(num_x) + 1j * rng.random(num_x)
    x_state /= np.linalg.norm(x_state)

    num_y = 2 ** len(output_wires)

    if output_wires_zeroed:
        y_state = np.zeros(num_y, dtype=complex)
        y_state[0] = 1.0
    else:
        y_state = rng.random(num_y) + 1j * rng.random(num_y)
        y_state /= np.linalg.norm(y_state)

    output_state = np.zeros((num_x, num_y), dtype=complex)
    for x in range(num_x):
        output_state[x] = x_state[x] * np.roll(y_state, x**2)
    output_state = output_state.reshape(-1)

    probs = circuit(x_state, y_state, output_state)
    assert np.isclose(probs[0], 1.0)
    assert np.allclose(probs[1:], 0.0)


class TestOutSquare:
    """Test the OutSquare template."""

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
        """Test the correctness of the OutSquare template output."""
        all_wires = (x_wires, output_wires, work_wires)
        _test_square_correctness(
            all_wires, OutSquare.compute_decomposition, seed, output_wires_zeroed, use_jit
        )

    @pytest.mark.catalyst
    @pytest.mark.external
    @pytest.mark.usefixtures("enable_graph_decomposition")
    @pytest.mark.parametrize("output_wires_zeroed", [False, True])
    def test_qjit_dynamic_wires(self, output_wires_zeroed):
        """Test the OutSquare template with dynamic wires."""
        x_wires = np.array([0, 1, 2, 3])
        output_wires = np.array([4, 5, 6, 7, 8])
        work_wires = np.array([9, 10, 11, 12, 13, 14, 15])

        dev = qp.device("lightning.qubit")

        x = 13
        mod = 2 ** len(output_wires)
        if output_wires_zeroed:
            y = 0
        else:
            y = mod - 2  # Some number close to causing overflows

        fixed_decomps = {"C(SemiAdder)": qp.list_decomps("C(SemiAdder)")[0]}

        @qp.qjit
        @qp.set_shots(1)
        @qp.decompose(max_expansion=2, fixed_decomps=fixed_decomps)
        @qp.qnode(dev)
        def circuit(x, y, x_wires, work_wires):
            qp.BasisEmbedding(x, wires=x_wires)
            qp.BasisEmbedding(y, wires=output_wires)
            OutSquare(x_wires, output_wires, work_wires, output_wires_zeroed)
            return (
                qp.sample(wires=x_wires),
                qp.sample(wires=output_wires),
                qp.sample(wires=work_wires),
            )

        output = circuit(x, y, x_wires, work_wires)
        out_ints = [int("".join(map(str, out[0])), 2) for out in output]
        assert np.allclose(out_ints, [x, (y + x**2) % mod, 0])

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
            OutSquare(x_wires, output_wires, work_wires)

    @pytest.mark.parametrize(
        "x_wires, output_wires, work_wires, zeroed, should_raise",
        [
            ([0, 1, 2], [3, 4, 5], [6, 7, 8], True, False),
            ([0, 1, 2], [3, 4, 5], [6, 7], True, True),
            ([0, 1], [3, 4, 5], [6, 7, 8], True, False),
            ([0, 1], [3, 4, 5], [6, 7], True, True),
            ([0], [3, 4, 5], [6, 7, 8], True, False),
            ([0], [3, 4, 5], [6, 7], True, False),
            ([0], [3, 4, 5], [6], True, True),
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
            msg_match = "OutSquare requires at least"
            with pytest.raises(ValueError, match=msg_match):
                OutSquare(x_wires, output_wires, work_wires, output_wires_zeroed=zeroed)
        else:
            OutSquare(x_wires, output_wires, work_wires, output_wires_zeroed=zeroed)

    @pytest.mark.parametrize("output_wires_zeroed", [False, True])
    def test_decomposition(self, output_wires_zeroed):
        """Test that compute_decomposition and decomposition work as expected."""
        x_wires, output_wires, work_wires = (
            [0, 1, 2],
            [3, 4, 5, 6],
            [7, 8, 9, 10],
        )
        decomp = OutSquare(x_wires, output_wires, work_wires, output_wires_zeroed).decomposition()

        if output_wires_zeroed:
            expected = [
                # controlled copy
                qp.CNOT([2, 6]),
                qp.TemporaryAND([2, 1, 5]),
                qp.TemporaryAND([2, 0, 4]),
                # First CNOT-wrapped controlled adder, shifted by 1
                qp.CNOT([1, 7]),
                Controlled(
                    qp.SemiAdder([0, 1, 2], [3, 4, 5], [8, 9]),
                    control_wires=[7],
                    work_wires=[10],
                    work_wire_type="zeroed",
                ),
                qp.CNOT([1, 7]),
                # Second CNOT-wrapped controlled adder, shifted by 2
                qp.CNOT([0, 7]),
                Controlled(
                    qp.SemiAdder([0, 1, 2], [3, 4], [8]),
                    control_wires=[7],
                    work_wires=[9, 10],
                    work_wire_type="zeroed",
                ),
                qp.CNOT([0, 7]),
            ]
        else:
            expected = [
                # controlled copy (="zeroth" CNOT-wrapped controlled adder)
                qp.CNOT([2, 7]),
                Controlled(
                    qp.SemiAdder([0, 1, 2], [3, 4, 5, 6], [8, 9, 10]),
                    control_wires=[7],
                    work_wire_type="zeroed",
                ),
                qp.CNOT([2, 7]),
                # First CNOT-wrapped controlled adder, shifted by 1
                qp.CNOT([1, 7]),
                Controlled(
                    qp.SemiAdder([0, 1, 2], [3, 4, 5], [8, 9]),
                    control_wires=[7],
                    work_wires=[10],
                    work_wire_type="zeroed",
                ),
                qp.CNOT([1, 7]),
                # Second CNOT-wrapped controlled adder, shifted by 2
                qp.CNOT([0, 7]),
                Controlled(
                    qp.SemiAdder([0, 1, 2], [3, 4], [8]),
                    control_wires=[7],
                    work_wires=[9, 10],
                    work_wire_type="zeroed",
                ),
                qp.CNOT([0, 7]),
            ]

        assert decomp == expected

    @pytest.mark.parametrize(
        ("x_wires", "output_wires", "work_wires", "output_wires_zeroed", "applicable_rules"),
        [
            ([0, 1, 2, 3], [4, 5, 6], [9, 10, 11], True, [0]),
            ([0, 1, 2, 3], [4, 5, 6], [9, 10, 11], False, [0]),
            ([0, 1, 2, 3], [4, 5, 6], [9, 10, 11, 12, 13], True, [0, 1]),
            ([0, 1, 2, 3], [4, 5, 6], [9, 10, 11, 12, 13], False, [0, 1]),
            ([0, 1], [3, 5], [9, 10], True, [0]),
            ([0, 1], [3, 5], [9, 10], False, [0]),
            ([0, 1], [3, 5], [9, 10, 11, 12], True, [0, 1]),
            ([0, 1], [3, 5], [9, 10, 11, 12], False, [0, 1]),
            ([0, 1], [3, 4, 5, 6], [9, 10, 11], True, [0]),
            ([0, 1], [3, 4, 5, 6], [9, 10, 11, 12, 13], True, [0, 1]),
            ([0, 1], [3, 4, 5, 6], [9, 10, 11, 12, 13], False, [0]),
            ([0, 1], [3, 4, 5, 6], [9, 10, 11, 12, 13, 14], False, [0, 1]),
            ([0, 1], [3, 4, 5, 6, 7], [9, 10, 11], True, [0]),
            ([0, 1], [3, 4, 5, 6, 7], [9, 10, 11, 12, 13, 14], True, [0, 1]),
            ([0, 1], [3, 4, 5, 6, 7], [9, 10, 11, 12, 13, 14], False, [0]),
            ([0, 1], [3, 4, 5, 6, 7], [9, 10, 11, 12, 13, 14, 15], False, [0, 1]),
        ],
    )
    @pytest.mark.parametrize("use_jit", [pytest.param(True, marks=pytest.mark.jax), False])
    def test_decomposition_new(
        self,
        x_wires,
        output_wires,
        work_wires,
        output_wires_zeroed,
        applicable_rules,
        use_jit,
        seed,
    ):  # pylint: disable=too-many-arguments
        """Tests the decomposition rule implemented with the new system."""
        op = OutSquare(x_wires, output_wires, work_wires, output_wires_zeroed)
        for j, rule in enumerate(qp.list_decomps(OutSquare)):
            applicable = rule.is_applicable(**op.resource_params)
            assert applicable is (j in applicable_rules)
            _test_decomposition_rule(op, rule)
            if applicable:
                all_wires = (x_wires, output_wires, work_wires)
                _test_square_correctness(all_wires, rule, seed, output_wires_zeroed, use_jit)
