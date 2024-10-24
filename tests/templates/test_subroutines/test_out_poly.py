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
    _get_polynomial,
    _mobius_inversion_of_zeta_transform,
)


def test_get_polynomial():
    """Test the private function _get_polynomial by checking its output for a specific function.

    The function under test takes as input a callable `f`, a modulus `mod`, and the bit sizes of the variables.
    It returns a dictionary where the keys represent the binary form of the variables involved in each term of
    the polynomial, and the values represent the corresponding coefficients of those terms, reduced by `mod`.

    In this test, It is used the function `lambda x, y: x**2 * y`, with two variables `x` and `y`, each represented by
    2 bits (hence `variable_sizes=(2, 2)`). It is expected the resulting dictionary to reflect the expansion of
    `(2x_0 + x_1)^2 * (2y_0 + y_1)`.

    The expected keys represent which bits (variables) are involved in each term, and the expected values
    are the coefficients of those terms.

    Key format: (x0, x1, y0, y1), where x0, x1 are bits for `x` and y0, y1 are bits for `y`.
    """
    dic = _get_polynomial(lambda x, y: x**2 * y, 16, 2, 2)

    expected_dic = {
        (0, 1, 0, 1): 1,  # x1.y1: 1
        (0, 1, 1, 0): 2,  # x1.y0: 2
        (1, 0, 0, 1): 4,  # x0.y1: 4
        (1, 0, 1, 0): 8,  # x0.y0: 8
        (1, 1, 0, 1): 4,  # x0.x1.y1: 4
        (1, 1, 1, 0): 8,  # x0.x1.y0: 8
    }

    for key in dic.keys():
        assert key in expected_dic
        assert dic[key] == expected_dic[key]


def test_mobius_inversion_of_zeta_transform():
    """Test that the Möbius inversion works correctly"""

    f_values = [1, 3, 4, 10]
    mod = 20

    expected_values = [1, 2, 3, 4]
    result = _mobius_inversion_of_zeta_transform(f_values.copy(), mod)
    assert result == expected_values


def f_test(x, y, z):
    return x**2 + y * x * z**5 - z**3 + 3


def test_standard_validity_OutPoly():
    """Check the operation using the assert_valid function."""
    wires = qml.registers({"x": 3, "y": 3, "z": 3, "output": 3, "aux": 2})

    op = qml.OutPoly(
        f_test,
        input_registers=[wires["x"], wires["y"], wires["z"]],
        output_wires=wires["output"],
    )

    qml.ops.functions.assert_valid(op)


class TestOutPoly:
    """Test the qml.OutPoly template."""

    @pytest.mark.parametrize(
        ("polynomial_function", "input_registers", "output_wires", "mod", "work_wires"),
        [
            (lambda x, y: 3 * x**3 - 3 * y, [[0, 1, 2], [3, 4, 5]], [6, 7, 8, 9], 6, [10, 11]),
            (lambda x, y: 2 * x * y - 3 * x, [[0, 1, 2], [3, 4, 5]], [6, 7, 8, 9], 7, [10, 11]),
            (lambda x, y: -2 * x - 3 * y + 1, [[0, 1, 2], [3, 4, 5]], [6, 7, 8, 9], None, None),
            (lambda x, y: 2 * x * y - 3 * x + 5, [[0, 1, 2], [3, 4, 5]], [6, 7, 8, 9], None, None),
        ],
    )
    def test_operation_result(
        self, polynomial_function, input_registers, output_wires, mod, work_wires
    ):  # pylint: disable=too-many-arguments
        """Test the correctness of the OutPoly template output."""

        x_wires, y_wires = input_registers
        dev = qml.device("default.qubit")

        @qml.qnode(dev)
        def circuit():

            qml.BasisEmbedding(2, wires=x_wires)
            qml.BasisEmbedding(1, wires=y_wires)
            qml.OutPoly(
                polynomial_function,
                input_registers=input_registers,
                output_wires=output_wires,
                mod=mod,
                work_wires=work_wires,
            )
            return qml.probs(wires=output_wires)

        if mod is None:
            mod = int(2 ** len(output_wires))
        assert np.isclose(np.argmax(circuit()), polynomial_function(2, 1) % mod)

    @pytest.mark.parametrize(
        ("input_registers", "output_wires", "mod", "work_wires", "msg_match"),
        [
            ([[0, 1, 2], [3, 4, 5]], [6, 7, 8, 9], 6.1, [10, 11], "mod must be an integer."),
            (
                [[0, 1, 2], [3, 4, 5]],
                [6, 7, 8, 9],
                6,
                [0, 11],
                "None of the wires in",
            ),
            ([[0, 1, 2], [3, 4, 5]], [6, 7, 8, 9], 6, [10], "If mod is not"),
        ],
    )
    def test_operation_and_test_wires_error(
        self, input_registers, output_wires, mod, work_wires, msg_match
    ):  # pylint: disable=too-many-arguments
        """Test that proper errors are raised"""
        x_wires, y_wires = input_registers

        with pytest.raises(ValueError, match=msg_match):
            if y_wires:
                qml.OutPoly(
                    lambda x, y: 3 * x**3 - 3 * y,
                    input_registers=input_registers,
                    output_wires=output_wires,
                    mod=mod,
                    work_wires=work_wires,
                )
            else:
                qml.OutPoly(
                    lambda x, y: 3 * x**3 - 3 * y,
                    input_registers=[x_wires],
                    output_wires=output_wires,
                    mod=mod,
                    work_wires=work_wires,
                )

    def test_non_integer_coeffs(self):
        """Test that an error is raised if the coefficient of the polynomial are not integer"""
        reg = qml.registers({"x_wires": 3, "y_wires": 3, "output_wires": 4})

        def f(x, y):
            return 2.5 * x + 2 * y

        @qml.qnode(qml.device("default.qubit", shots=1))
        def circuit():

            qml.OutPoly(
                f,
                input_registers=(reg["x_wires"], reg["y_wires"]),
                output_wires=reg["output_wires"],
            )

            return qml.sample(wires=reg["output_wires"])

        with pytest.raises(AssertionError, match="The polynomial function must"):
            circuit()

    def test_decomposition(self):
        """Test that compute_decomposition and decomposition work as expected."""

        def polynomial_function(x, y):
            return x + y

        expected_decomposition = [
            qml.QFT(wires=[3]),
            qml.ctrl(qml.PhaseAdder(1, x_wires=[3]), control=[2]),
            qml.ctrl(qml.PhaseAdder(1, x_wires=[3]), control=[1]),
            qml.adjoint(qml.QFT(wires=[3])),
        ]

        ops = qml.OutPoly(
            polynomial_function, input_registers=[[0, 1], [2]], output_wires=[3]
        ).decomposition()

        for op1, op2 in zip(expected_decomposition, ops):
            qml.assert_equal(op1, op2)

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
                input_registers=[wires["x"], wires["y"], wires["z"]],
                output_wires=wires["output"],
                mod=6,
                work_wires=wires["aux"],
            )
            return qml.sample(wires=wires["output"])

        assert np.allclose(circuit(), [0, 0, 1])
