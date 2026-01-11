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
Tests for the SignedOutMultiplier template.
"""

import pytest

import pennylane as qml
from pennylane import numpy as np
from pennylane.ops.functions.assert_valid import _test_decomposition_rule
from pennylane.templates.subroutines.arithmetic.signed_out_multiplier import SignedOutMultiplier


@pytest.mark.jax
def test_standard_validity_signed_out_multiplier():
    """Check the operation using the assert_valid function."""
    x_wires = [0, 1]
    y_wires = [2, 3, 4]
    output_wires = [6, 7, 8, 9, 10]
    work_wires = [5, 11, 12]
    op = SignedOutMultiplier(x_wires, y_wires, output_wires, work_wires)
    qml.ops.functions.assert_valid(op)


class TestSignedOutMultiplier:
    """Test the qml.SignedOutMultiplier template."""

    @pytest.mark.parametrize(
        ("x_wires", "y_wires", "output_wires", "work_wires", "x", "y"),
        [
            (
                [0, 1, 2],
                [3, 4, 5],
                [6, 7, 8],
                [9, 10],
                2,
                -3,
            ),
            (
                [0, 1],
                [3, 4, 5],
                [6, 7, 8, 2],
                [9, 10],
                -2,
                -4,
            ),
            (
                [0, 1, 2],
                [3, 4],
                [5, 6, 7, 8],
                [9, 10],
                -3,
                1,
            ),
            (
                [0, 1, 2, 3],
                [4, 5],
                [6, 7, 8, 9, 10],
                [11, 12, 13, 14],
                0,
                0,
            ),
            (
                [0, 1, 2],
                [3, 4, 5],
                [6, 7, 8],
                [9, 10, 11],
                0,
                3,
            ),
            (
                [0, 1, 2],
                [3, 4, 5],
                [6, 7, 8],
                [9, 10, 11],
                -4,
                0,
            ),
        ],
    )
    @pytest.mark.parametrize("z", [0, 1, 3, 6, -1, -5, -8])
    def test_operation_result(self, x_wires, y_wires, output_wires, work_wires, x, y, z):
        # pylint: disable=too-many-arguments
        """Test the correctness of the SignedOutMultiplier template output."""
        dev = qml.device("default.qubit")
        mod = 2 ** len(output_wires)

        @qml.set_shots(1)
        @qml.qnode(dev)
        def circuit(x, y, z):
            x = x % 2 ** (len(x_wires) - 1) + 2 ** (len(x_wires) - 1) * int(x < 0)
            y = y % 2 ** (len(y_wires) - 1) + 2 ** (len(y_wires) - 1) * int(y < 0)
            qml.BasisEmbedding(x, wires=x_wires)
            qml.BasisEmbedding(y, wires=y_wires)
            qml.BasisEmbedding(z, wires=output_wires)
            SignedOutMultiplier(x_wires, y_wires, output_wires, work_wires=work_wires)
            return (
                qml.sample(wires=x_wires),
                qml.sample(wires=y_wires),
                qml.sample(wires=output_wires),
            )

        z = ((z + mod // 2) % mod) - mod // 2
        output = circuit(x, y, z)
        out_int_x, out_int_y, out_int_z = (
            int("".join(str(bit) for bit in out[0][1:]), 2) - 2 ** (len(wires) - 1) * out[0][0]
            for out, wires in zip(output, (x_wires, y_wires, output_wires))
        )
        expected = ((z + x * y + mod // 2) % mod) - mod // 2
        assert np.allclose(out_int_x, x)
        assert np.allclose(out_int_y, y)
        assert np.allclose(out_int_z, expected)

    @pytest.mark.parametrize(
        ("x_wires", "y_wires", "output_wires", "work_wires", "msg_match"),
        [
            (
                [0, 1, 2],
                [3, 4, 5],
                [6, 7, 8],
                [1, 10],
                "None of the wires in x_wires should be included in work_wires.",
            ),
            (
                [0, 1, 2],
                [3, 4, 5],
                [6, 7, 8],
                [3, 10],
                "None of the wires in y_wires should be included in work_wires.",
            ),
            (
                [0, 1, 2],
                [2, 4, 5],
                [6, 7, 8],
                [9, 10],
                "None of the wires in x_wires should be included in y_wires.",
            ),
            (
                [0, 1, 2],
                [3, 7, 5],
                [6, 7, 8],
                [9, 10],
                "None of the wires in y_wires should be included in output_wires.",
            ),
            (
                [0, 1, 7],
                [3, 4, 5],
                [6, 7, 8],
                [9, 10],
                "None of the wires in x_wires should be included in output_wires.",
            ),
        ],
    )
    def test_wires_error(
        self, x_wires, y_wires, output_wires, work_wires, msg_match
    ):  # pylint: disable=too-many-arguments
        """Test an error is raised when some work_wires don't meet the requirements"""
        with pytest.raises(ValueError, match=msg_match):
            SignedOutMultiplier(x_wires, y_wires, output_wires, work_wires)

    @pytest.mark.parametrize("len_x, len_y, len_work", [(2, 2, 0), (2, 3, 1), (3, 2, 1)])
    def test_validation_of_num_work_wires(self, len_x, len_y, len_work):
        """Test that the right number of work wires is being required."""

        with pytest.raises(ValueError, match=f"SignedOutMultiplier requires at least {len_work+1}"):
            SignedOutMultiplier(
                x_wires=range(len_x),
                y_wires=range(len_x, len_x + len_y),
                output_wires=range(len_x + len_y, 2 * (len_x + len_y)),
                work_wires=range(2 * (len_x + len_y), 2 * (len_x + len_y) + len_work),
            )

    def test_decomposition(self):
        """Test that compute_decomposition and decomposition work as expected."""
        x_wires, y_wires, output_wires, work_wires = (
            [0, 1, 2],
            [3, 5],
            [6, 8],
            [9, 10],
        )
        multiplier_decomposition = SignedOutMultiplier(
            x_wires, y_wires, output_wires, work_wires
        ).compute_decomposition(x_wires, y_wires, output_wires, work_wires)

        op_list = [
            qml.OutMultiplier([1, 2], [5], [6, 8], work_wires=[9, 10]),
            qml.Toffoli([0, 3, 6]),
        ]

        for op1, op2 in zip(multiplier_decomposition, op_list):
            qml.assert_equal(op1, op2)

    @pytest.mark.parametrize(
        ("x_wires", "y_wires", "output_wires", "work_wires"),
        [
            ([0, 1, 2], [3, 5], [6, 8], [9, 10]),
            ([0, 1, 2], [3, 6], [5, 8], [9, 10]),
        ],
    )
    def test_decomposition_new(
        self, x_wires, y_wires, output_wires, work_wires
    ):  # pylint: disable=too-many-arguments
        """Tests the decomposition rule implemented with the new system."""
        op = qml.SignedOutMultiplier(x_wires, y_wires, output_wires, work_wires)
        for rule in qml.list_decomps(qml.SignedOutMultiplier):
            _test_decomposition_rule(op, rule)

    def test_work_wires_added_correctly(self):
        """Test that no work wires are added if work_wire = None"""
        wires = qml.SignedOutMultiplier(
            x_wires=[1, 2], y_wires=[3, 4], output_wires=[5, 6], work_wires=[7, 8]
        ).wires
        assert wires == qml.wires.Wires([1, 2, 3, 4, 5, 6, 7, 8])

    @pytest.mark.jax
    def test_jit_compatible(self):
        """Test that the template is compatible with the JIT compiler."""

        import jax

        jax.config.update("jax_enable_x64", True)

        x_list = [0, 1, 1]
        y_list = [1, 0, 1]
        x_wires = [0, 1, 2]
        y_wires = [3, 4, 5]
        output_wires = [6, 7, 8, 9, 10]
        work_wires = [11, 12, 13]
        dev = qml.device("default.qubit")

        @jax.jit
        @qml.set_shots(1)
        @qml.qnode(dev)
        def circuit():
            qml.BasisEmbedding(x_list, wires=x_wires)
            qml.BasisEmbedding(y_list, wires=y_wires)
            SignedOutMultiplier(x_wires, y_wires, output_wires, work_wires)
            return qml.sample(wires=output_wires)

        # pylint: disable=bad-reversed-sequence
        out = circuit()[0, :]
        out = int("".join(str(bit) for bit in out[1:]), 2) - 2 ** (len(output_wires) - 1) * out[0]
        assert jax.numpy.allclose(out, -9)
