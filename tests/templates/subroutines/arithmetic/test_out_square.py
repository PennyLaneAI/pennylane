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

import pennylane as qml
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
    work_wires = [11, 12, 13, 14, 15, 16, 17]
    op = OutSquare(x_wires, output_wires, work_wires, output_wires_zeroed)
    qml.ops.functions.assert_valid(op)


class TestOutSquare:
    """Test the OutSquare template."""

    @pytest.mark.parametrize("output_wires_zeroed", [False, True])
    @pytest.mark.parametrize(
        ("x_wires", "output_wires", "work_wires", "x_values"),
        [
            (
                [0, 1],
                [3, 4, 5],
                [6, 7, 8],
                [0, 1, 2, 3],
            ),
            (
                [0, 1, 2],
                [3, 4, 5, 6, 7],
                [8, 9, 10, 11, 12],
                [0, 1, 3, 4, 5],
            ),
            (
                [0, 1, 2],
                [3, 4, 5, 6, 7, 8, 9],
                [10, 11, 12, 13, 14, 15, 16],
                [0, 1, 3, 6],
            ),
            (
                [0, 1, 2, 3],
                [4, 5, 6, 7, 8],
                [9, 10, 11, 12, 13],
                [0, 2, 5, 9, 13],
            ),
            (
                [0, 1, 2, 3],
                [4, 5, 6, 7],
                [9, 10, 11, 12, 13],
                [0, 2, 5, 9, 13],
            ),
        ],
    )
    @pytest.mark.parametrize("use_jit", [pytest.param(True, marks=pytest.mark.jax), False])
    def test_operation_result(
        self,
        x_wires,
        output_wires,
        work_wires,
        x_values,
        output_wires_zeroed,
        use_jit,
    ):  # pylint: disable=too-many-arguments
        """Test the correctness of the OutSquare template output."""
        if use_jit:
            import jax

            jax.config.update("jax_enable_x64", True)

        dev = qml.device("lightning.qubit")

        mod = 2 ** len(output_wires)
        if output_wires_zeroed:
            z = 0
        else:
            z = mod - 2  # Some number close to causing overflows

        @qml.set_shots(1)
        @qml.qnode(dev)
        def circuit(x, z):
            qml.BasisEmbedding(x, wires=x_wires)
            qml.BasisEmbedding(z, wires=output_wires)
            OutSquare(x_wires, output_wires, work_wires, output_wires_zeroed)
            return (
                qml.sample(wires=x_wires),
                qml.sample(wires=output_wires),
                qml.sample(wires=work_wires),
            )

        if use_jit:
            circuit = jax.jit(circuit)

        for x in x_values:
            output = circuit(x, z)
            out_ints = [int("".join(map(str, out[0])), 2) for out in output]
            assert np.allclose(out_ints, [x, (z + x**2) % mod, 0])

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
                qml.CNOT([2, 6]),
                qml.TemporaryAND([2, 1, 5]),
                qml.TemporaryAND([2, 0, 4]),
                # First CNOT-wrapped controlled adder, shifted by 1
                qml.CNOT([1, 7]),
                Controlled(qml.SemiAdder([0, 1, 2], [3, 4, 5], [8, 9, 10]), control_wires=[7]),
                qml.CNOT([1, 7]),
                # Second CNOT-wrapped controlled adder, shifted by 2
                qml.CNOT([0, 7]),
                Controlled(qml.SemiAdder([0, 1, 2], [3, 4], [8, 9, 10]), control_wires=[7]),
                qml.CNOT([0, 7]),
            ]
        else:
            expected = [
                # controlled copy (="zeroth" CNOT-wrapped controlled adder)
                qml.CNOT([2, 7]),
                Controlled(qml.SemiAdder([0, 1, 2], [3, 4, 5, 6], [8, 9, 10]), control_wires=[7]),
                qml.CNOT([2, 7]),
                # First CNOT-wrapped controlled adder, shifted by 1
                qml.CNOT([1, 7]),
                Controlled(qml.SemiAdder([0, 1, 2], [3, 4, 5], [8, 9, 10]), control_wires=[7]),
                qml.CNOT([1, 7]),
                # Second CNOT-wrapped controlled adder, shifted by 2
                qml.CNOT([0, 7]),
                Controlled(qml.SemiAdder([0, 1, 2], [3, 4], [8, 9, 10]), control_wires=[7]),
                qml.CNOT([0, 7]),
            ]

        assert decomp == expected

    @pytest.mark.parametrize(
        ("x_wires", "output_wires", "work_wires", "output_wires_zeroed"),
        [
            ([0, 1, 2], [3, 5], [9, 10], False),
            ([0, 1, 2], [3, 5], [9, 10], True),
            ([0, 1], [3, 4, 5, 6], [9, 10, 11, 12], False),
            ([0, 1], [3, 4, 5, 6], [9, 10, 11], True),
            ([0, 1, 2], [3, 4, 5, 6], [9, 10, 11, 12], False),
            ([0, 1, 2], [3, 4, 5, 6], [9, 10, 11, 12], True),
        ],
    )
    def test_decomposition_new(
        self, x_wires, output_wires, work_wires, output_wires_zeroed
    ):  # pylint: disable=too-many-arguments
        """Tests the decomposition rule implemented with the new system."""
        op = OutSquare(x_wires, output_wires, work_wires, output_wires_zeroed)
        for rule in qml.list_decomps(OutSquare):
            _test_decomposition_rule(op, rule)
