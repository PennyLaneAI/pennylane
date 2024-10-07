# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

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
Tests for the OutPoly template.
"""

import pytest

import pennylane as qml
from pennylane import numpy as np
from pennylane.templates.subroutines.out_poly import (
    _binary_to_decimal,
    _decimal_to_binary_list,
    _get_coefficients_and_controls,
)


@pytest.mark.parametrize(
    ("input_list",),
    [
        ([1, 0, 0, 1],),
        ([0, 1, 1, 1],),
        ([1, 1, 0, 0, 0, 1],),
        ([1, 1, 0],),
    ],
)
def test_binary_decimal_conversion(input_list):
    """Tests that the conversion between decimal and binary works correctly."""
    assert _decimal_to_binary_list(_binary_to_decimal(input_list), len(input_list)) == input_list


def test_get_coeffs_function():

    dic = _get_coefficients_and_controls(lambda x, y: x**2 * y, 16, 2, 2)
    # `dic` should contain the coefficient of (2x0 + x1)^2 * (2y0 + y1)

    # key format (x0, x1, y0, y1)
    expected_dic = {
        (0, 1, 0, 1): 1,  # x1.y1
        (0, 1, 1, 0): 2,  # + 2 x1.y0
        (1, 0, 0, 1): 4,  # + 4 x0.y1
        (1, 0, 1, 0): 8,  # + 8 x0.y0
        (1, 1, 0, 1): 4,  # + 4 x0.x1.y1
        (1, 1, 1, 0): 8,  # + 8 x0.x1.y0
    }

    for key in dic.keys():
        assert key in expected_dic
        assert dic[key] == expected_dic[key]


def f_test(x, y, z):
    return x**2 + y * x * z**5 - z**3 + 3


def test_standard_validity_OutPoly():
    """Check the operation using the assert_valid function."""
    wires = qml.registers({"x": 3, "y": 3, "z": 3, "output": 3, "aux": 2})

    op = qml.OutPoly(
        f_test,
        [wires["x"], wires["y"], wires["z"], wires["output"]],
        mod=5,
        work_wires=wires["aux"],
    )

    qml.ops.functions.assert_valid(op)


class TestOutPoly:
    """Test the qml.OutPoly template."""

    @pytest.mark.parametrize(
        ("f", "x_wires", "y_wires", "output_wires", "mod", "work_wires"),
        [
            (lambda x, y: 3 * x**3 - 3 * y, [0, 1, 2], [3, 4, 5], [6, 7, 8, 9], 6, [10, 11]),
            (lambda x, y: 2 * x * y - 3 * x, [0, 1, 2], [3, 4, 5], [6, 7, 8, 9], 7, [10, 11]),
            (lambda x, y: -2 * x - 3 * y + 1, [0, 1, 2], [3, 4, 5], [6, 7, 8, 9], None, None),
            (lambda x, y: 2 * x * y - 3 * x + 5, [0, 1, 2], [3, 4, 5], [6, 7, 8, 9], None, None),
        ],
    )
    def test_operation_result(
        self, f, x_wires, y_wires, output_wires, mod, work_wires
    ):  # pylint: disable=too-many-arguments
        """Test the correctness of the OutPoly template output."""
        dev = qml.device("default.qubit")

        @qml.qnode(dev)
        def circuit():

            qml.BasisEmbedding(2, wires=x_wires)
            qml.BasisEmbedding(1, wires=y_wires)
            qml.OutPoly(f, [x_wires, y_wires, output_wires], mod=mod, work_wires=work_wires)
            return qml.probs(wires=output_wires)

        if mod is None:
            mod = int(2 ** len(output_wires))
        assert np.isclose(np.argmax(circuit()), f(2, 1) % mod)

    @pytest.mark.parametrize(
        ("x_wires", "y_wires", "output_wires", "mod", "work_wires", "msg_match"),
        [
            ([0, 1, 2], [3, 4, 5], [6, 7, 8, 9], 6.1, [10, 11], "mod must be integer."),
            ([0, 1, 2], [3, 4, 5], [6, 7, 8, 9], 6, [0, 11], "None of the wires in"),
            ([0, 1, 2], [3, 4, 5], [6, 7, 8, 9], 6, [10], "If mod is not"),
            ([0, 1, 2], None, [6, 7, 8, 9], 6, [10, 11], "The function takes"),
        ],
    )
    def test_operation_and_test_wires_error(
        self, x_wires, y_wires, output_wires, mod, work_wires, msg_match
    ):  # pylint: disable=too-many-arguments
        """Test that proper errors are raised"""

        with pytest.raises(ValueError, match=msg_match):
            if y_wires:
                qml.OutPoly(
                    lambda x, y: 3 * x**3 - 3 * y,
                    [x_wires, y_wires, output_wires],
                    mod=mod,
                    work_wires=work_wires,
                )
            else:
                qml.OutPoly(
                    lambda x, y: 3 * x**3 - 3 * y,
                    [x_wires, output_wires],
                    mod=mod,
                    work_wires=work_wires,
                )

    def test_error_not_input(self):
        """Test that an error appears if f or register wires are not provided"""

        with pytest.raises(ValueError, match="The register wires and the function f"):

            qml.OutPoly(
                registers_wires=[[0, 1], [2, 3]],
            )

        with pytest.raises(ValueError, match="The register wires and the function f"):
            qml.OutPoly(
                f=lambda x: x,
            )

    def test_decomposition(self):
        """Test that compute_decomposition and decomposition work as expected."""

        def f(x, y):
            return x + y

        expected_decomposition = [
            qml.QFT(wires=[3]),
            qml.ctrl(qml.PhaseAdder(1, x_wires=[3]), control=[2]),
            qml.ctrl(qml.PhaseAdder(1, x_wires=[3]), control=[1]),
            qml.adjoint(qml.QFT(wires=[3])),
        ]

        ops = qml.OutPoly(f, [[0, 1], [2], [3]]).decomposition()

        for op1, op2 in zip(expected_decomposition, ops):
            assert qml.equal(op1, op2)

    @pytest.mark.jax
    def test_jit_compatible(self):
        """Test that the template is compatible with the JIT compiler."""

        import jax

        wires = qml.registers({"x": 3, "y": 3, "z": 3, "output": 3, "aux": 2})

        def f(x, y, z):
            return x**2 + y * x * z**5 - z**3 + 3

        dev = qml.device("default.qubit", wires=14, shots=1)

        @jax.jit
        @qml.qnode(dev)
        def circuit():
            # loading values for x, y and z
            qml.BasisEmbedding(1, wires=wires["x"])
            qml.BasisEmbedding(2, wires=wires["y"])
            qml.BasisEmbedding(3, wires=wires["z"])

            # applying the polynomial
            qml.OutPoly(
                f,
                [wires["x"], wires["y"], wires["z"], wires["output"]],
                mod=6,
                work_wires=wires["aux"],
            )
            return qml.sample(wires=wires["output"])

        assert np.allclose(circuit(), [0, 0, 1])
