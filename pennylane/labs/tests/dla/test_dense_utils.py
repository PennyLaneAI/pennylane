# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for pennylane/labs/dla/dense_util.py functionality"""
# pylint: disable=no-self-use, unnecessary-lambda-assignment
import numpy as np
import pytest

import pennylane as qml
from pennylane import I, X, Y, Z
from pennylane.labs.dla import lie_closure_dense, op_to_adjvec, pauli_coefficients, pauli_decompose

# Make an operator matrix on given wire and total wire count
I_ = lambda w, n: I(w).matrix(wire_order=range(n))
X_ = lambda w, n: X(w).matrix(wire_order=range(n))
Y_ = lambda w, n: Y(w).matrix(wire_order=range(n))
Z_ = lambda w, n: Z(w).matrix(wire_order=range(n))

# Make the i-th 4**n-dimensional unit vector
uv = lambda i, n: np.eye(4**n)[i]


class TestPauliCoefficients:
    """Tests for ``pauli_coefficients`` utility function."""

    # I  X  Y  Z
    # 0  1  2  3

    @pytest.mark.parametrize(
        "H, expected",
        [
            (0.3 * X_(0, 1), [0, 0.3, 0, 0]),
            (-2.3 * Y_(0, 1), [0, 0, -2.3, 0]),
            (Z_(0, 1), [0, 0, 0, 1.0]),
            (0.3 * X_(0, 1) - 0.6 * Z_(0, 1), [0, 0.3, 0, -0.6]),
            (X_(0, 1) - Z_(0, 1) + Y_(0, 1), [0, 1, 1, -1]),
            (I_(0, 1) * 0.7, [0.7, 0, 0, 0]),
            (Y_(0, 1) * 0, [0, 0, 0, 0]),
        ],
    )
    def test_one_qubit(self, H, expected):
        """Test that Pauli coefficients for a one-qubit matrix are correct."""
        coeffs = pauli_coefficients(H)
        assert coeffs.shape == (4,)
        assert coeffs.dtype == np.float64
        assert np.allclose(coeffs, expected)

    def test_one_qubit_batched(self):
        """Test that batched Pauli coefficients for a one-qubit matrix are correct."""
        H = np.stack(
            [
                0.3 * X_(0, 1),
                -2.3 * Y_(0, 1),
                Z_(0, 1),
                0.3 * X_(0, 1) - 0.6 * Z_(0, 1),
                X_(0, 1) - Z_(0, 1) + Y_(0, 1),
                I_(0, 1) * 0.7,
                Y_(0, 1) * 0,
            ]
        )
        expected = [
            [0, 0.3, 0, 0],
            [0, 0, -2.3, 0],
            [0, 0, 0, 1.0],
            [0, 0.3, 0, -0.6],
            [0, 1, 1, -1],
            [0.7, 0, 0, 0],
            [0, 0, 0, 0],
        ]
        coeffs = pauli_coefficients(H)
        assert coeffs.shape == (7, 4)
        assert coeffs.dtype == np.float64
        assert np.allclose(coeffs, expected)

    #  qubit
    #  0 \ 1    I    X    Y    Z
    #         -------------------
    #   I    |  0    1    2    3
    #   X    |  4    5    6    7
    #   Y    |  8    9   10   11
    #   Z    | 12   13   14   15

    @pytest.mark.parametrize(
        "H, expected",
        [
            (I_(0, 2) + I_(1, 2), 2 * uv(0, 2)),
            (0.3 * X_(0, 2), 0.3 * uv(4, 2)),
            (-2.3 * Y_(0, 2), -2.3 * uv(8, 2)),
            (Z_(0, 2) - Z_(1, 2), uv(12, 2) - uv(3, 2)),
            (0.3 * X_(0, 2) - 0.6 * Z_(0, 2), 0.3 * uv(4, 2) - 0.6 * uv(12, 2)),
            (X_(1, 2) - Z_(1, 2) + Y_(0, 2) @ X_(1, 2), uv(1, 2) - uv(3, 2) + uv(9, 2)),
            (X_(1, 2) @ Z_(0, 2) - Y_(0, 2) @ Z_(1, 2), uv(13, 2) - uv(11, 2)),
        ],
    )
    def test_two_qubits(self, H, expected):
        """Test that Pauli coefficients for a two-qubit matrix are correct."""
        coeffs = pauli_coefficients(H)
        assert coeffs.shape == (16,)
        assert coeffs.dtype == np.float64
        assert np.allclose(coeffs, expected)

    def test_two_qubits_batched(self):
        """Test that batched Pauli coefficients for a two-qubit matrix are correct."""
        H = np.stack(
            [
                I_(0, 2) + I_(1, 2),
                0.3 * X_(0, 2),
                -2.3 * Y_(0, 2),
                Z_(0, 2) - Z_(1, 2),
                0.3 * X_(0, 2) - 0.6 * Z_(0, 2),
                X_(1, 2) - Z_(1, 2) + Y_(0, 2) @ X_(1, 2),
                X_(1, 2) @ Z_(0, 2) - Y_(0, 2) @ Z_(1, 2),
            ]
        )
        expected = np.array(
            [
                2 * uv(0, 2),
                0.3 * uv(4, 2),
                -2.3 * uv(8, 2),
                uv(12, 2) - uv(3, 2),
                0.3 * uv(4, 2) - 0.6 * uv(12, 2),
                uv(1, 2) - uv(3, 2) + uv(9, 2),
                uv(13, 2) - uv(11, 2),
            ]
        )
        coeffs = pauli_coefficients(H)
        assert coeffs.shape == (7, 16)
        assert coeffs.dtype == np.float64
        assert np.allclose(coeffs, expected)


class TestPauliDecompose:
    """Tests for ``pauli_decompose`` utility function."""

    # I  X  Y  Z
    # 0  1  2  3

    @pytest.mark.parametrize(
        "H, expected",
        [
            (0.3 * X_(0, 1), 0.3 * X(0)),
            (-2.3 * Y_(0, 1), -2.3 * Y(0)),
            (Z_(0, 1), Z(0)),
            (0.3 * X_(0, 1) - 0.6 * Z_(0, 1), 0.3 * X(0) - 0.6 * Z(0)),
            (X_(0, 1) - Z_(0, 1) + Y_(0, 1), X(0) - Z(0) + Y(0)),
            (I_(0, 1) * 0.7, I(0) * 0.7),
            (Y_(0, 1) * 0, 0 * I(0)),  # Zero operator is mapped to zero times identity, not Y
        ],
    )
    @pytest.mark.parametrize("pauli", [False, True])
    def test_one_qubit(self, H, expected, pauli):
        """Test that Pauli decomposition for a one-qubit matrix is correct."""
        op = pauli_decompose(H, pauli=pauli)
        if pauli:
            expected = expected.pauli_rep
            expected.simplify()
            assert isinstance(op, qml.pauli.PauliSentence)
            assert all(c.dtype == np.float64 for c in op.values())
            assert set(op.keys()) == set(expected.keys())
            assert all(np.isclose(op[k], expected[k]) for k in op.keys())
        else:
            assert isinstance(op, qml.operation.Operator)
            assert qml.equal(op, expected)

    @pytest.mark.parametrize("pauli", [False, True])
    def test_one_qubit_batched(self, pauli):
        """Test that batched Pauli decomposition for a one-qubit matrix is correct."""
        H = np.stack(
            [
                0.3 * X_(0, 1),
                -2.3 * Y_(0, 1),
                Z_(0, 1),
                0.3 * X_(0, 1) - 0.6 * Z_(0, 1),
                X_(0, 1) - Z_(0, 1) + Y_(0, 1),
                I_(0, 1) * 0.7,
                Y_(0, 1) * 0,
            ]
        )
        expected = [
            0.3 * X(0),
            -2.3 * Y(0),
            Z(0),
            0.3 * X(0) - 0.6 * Z(0),
            X(0) - Z(0) + Y(0),
            I(0) * 0.7,
            I(0) * 0,
        ]
        op = pauli_decompose(H, pauli=pauli)
        assert isinstance(op, list)
        assert len(op) == len(expected)
        if pauli:
            for _op, e in zip(op, expected):
                e = e.pauli_rep
                e.simplify()
                assert isinstance(_op, qml.pauli.PauliSentence)
                assert all(c.dtype == np.float64 for c in _op.values())
                assert set(_op.keys()) == set(e.keys())
                assert all(np.isclose(_op[k], e[k]) for k in _op.keys())
        else:
            for _op, e in zip(op, expected):
                assert isinstance(_op, qml.operation.Operator)
                assert qml.equal(_op, e)

    @pytest.mark.parametrize(
        "H, expected",
        [
            (I_(0, 2) + I_(1, 2), 2 * I()),
            (0.3 * X_(0, 2), 0.3 * X(0)),
            (-2.3 * Y_(0, 2), -2.3 * Y(0)),
            (Z_(0, 2) - Z_(1, 2), Z(0) - Z(1)),
            (0.3 * X_(0, 2) - 0.6 * Z_(0, 2), 0.3 * X(0) - 0.6 * Z(0)),
            (X_(1, 2) - Z_(1, 2) + Y_(0, 2) @ X_(1, 2), X(1) - Z(1) + Y(0) @ X(1)),
            (X_(1, 2) @ Z_(0, 2) - Y_(0, 2) @ Z_(1, 2), X(1) @ Z(0) - Y(0) @ Z(1)),
        ],
    )
    @pytest.mark.parametrize("pauli", [False, True])
    def test_two_qubits(self, H, expected, pauli):
        """Test that Pauli decomposition for a two-qubit matrix is correct."""
        op = pauli_decompose(H, pauli=pauli)
        if pauli:
            expected = expected.pauli_rep
            expected.simplify()
            assert isinstance(op, qml.pauli.PauliSentence)
            assert all(c.dtype == np.float64 for c in op.values())
            assert set(op.keys()) == set(expected.keys())
            assert all(np.isclose(op[k], expected[k]) for k in op.keys())
        else:
            assert isinstance(op, qml.operation.Operator)
            assert qml.equal(op, expected)

    @pytest.mark.parametrize("pauli", [False, True])
    def test_two_qubits_batched(self, pauli):
        """Test that batched Pauli decomposition for a two-qubit matrix is correct."""
        H = np.stack(
            [
                I_(0, 2) + I_(1, 2),
                0.3 * X_(0, 2),
                -2.3 * Y_(0, 2),
                Z_(0, 2) - Z_(1, 2),
                0 * X_(0, 2),
                0.3 * X_(0, 2) - 0.6 * Z_(0, 2),
                X_(1, 2) - Z_(1, 2) + Y_(0, 2) @ X_(1, 2),
                X_(1, 2) @ Z_(0, 2) - Y_(0, 2) @ Z_(1, 2),
            ]
        )
        expected = [
            2 * I(),
            0.3 * X(0),
            -2.3 * Y(0),
            Z(0) - Z(1),
            0 * I(0),
            0.3 * X(0) - 0.6 * Z(0),
            X(1) - Z(1) + Y(0) @ X(1),
            X(1) @ Z(0) - Y(0) @ Z(1),
        ]
        op = pauli_decompose(H, pauli=pauli)
        assert isinstance(op, list)
        assert len(op) == len(expected)
        if pauli:
            for _op, e in zip(op, expected):
                e = e.pauli_rep
                e.simplify()
                assert isinstance(_op, qml.pauli.PauliSentence)
                assert all(c.dtype == np.float64 for c in _op.values())
                assert set(_op.keys()) == set(e.keys())
                assert all(np.isclose(_op[k], e[k]) for k in _op.keys())
        else:
            for _op, e in zip(op, expected):
                assert isinstance(_op, qml.operation.Operator)
                assert qml.equal(_op, e)


def test_op_to_adjvec_consistent_with_input_types():
    """Test that op_to_adjvec yields the same results independently of the input type"""

    g = list(qml.pauli.pauli_group(3))  # su(8)
    g = lie_closure_dense(g)

    m = g[:32]

    res1 = op_to_adjvec(m, g)

    g = list(qml.pauli.pauli_group(3))  # su(8)
    g = qml.lie_closure(g)
    g = [_.pauli_rep for _ in g]

    m = g[:32]

    res2 = np.array(op_to_adjvec(m, g))
    assert res1.shape == res2.shape
    assert np.allclose(res1, res2)
