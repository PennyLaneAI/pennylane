# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for the algebraic prerequisites such as rings and matrices for Clifford+T decomposition."""

import numpy as np
import pytest

from pennylane.ops.op_math.decompositions.rings import DyadicMatrix, SO3Matrix, ZOmega, ZSqrtTwo

# pylint: disable=too-few-public-methods


class TestZSqrtTwo:
    """Tests for the ZSqrtTwo class."""

    @pytest.mark.parametrize(
        "a, b",
        [(0, 0), (2, 1), (3, -2), (-4, 0), (0, 5)],
    )
    def test_init_and_repr(self, a, b):
        """Test that ZSqrtTwo initializes correctly and has correct representations."""
        z_sqrt_two = ZSqrtTwo(a, b)
        assert z_sqrt_two.a == a
        assert z_sqrt_two.b == b

        assert repr(z_sqrt_two) == f"ZSqrtTwo(a={a}, b={b})"
        # pylint:disable=condition-evals-to-constant
        assert (
            str(z_sqrt_two)
            == (f"{a}" if a else "" + " + " if a and b else "" + f"{b}√2" if b else "")
            or "0"
        )

    def test_arithmetic_operations(self):
        """Test arithmetic operations on ZSqrtTwo."""
        z1 = ZSqrtTwo(1, 2)
        z2 = ZSqrtTwo(3, 4)

        assert z1 + z2 == ZSqrtTwo(4, 6) == float(z1) + float(z2)
        assert z2 + 10 == ZSqrtTwo(13, 4) == float(z2) + 10
        assert z2 + 2.0 == ZSqrtTwo(5, 4) == float(z2) + 2.0
        assert z1 - z2 == ZSqrtTwo(-2, -2) == float(z1) - float(z2)
        assert z1 * z2 == ZSqrtTwo(19, 10) == float(z1) * float(z2)
        assert z2 * 10 == ZSqrtTwo(30, 40) == float(z2) * 10
        assert z2 * 2.0 == ZSqrtTwo(6, 8) == float(z2) * 2.0
        assert ZSqrtTwo(14, 7) / z1 == ZSqrtTwo(2, 3)  # Computed manually
        assert z1**0 == ZSqrtTwo(1, 0) == float(z1) ** 0
        assert z1**2 == ZSqrtTwo(9, 4) == float(z1) ** 2
        assert ZSqrtTwo(5, 0) == 5
        assert z1.sqrt() is None  # Not all ZSqrtTwo instances have a square root
        assert z2.sqrt() is None  # Not all ZSqrtTwo instances have a square root
        assert ZSqrtTwo(30, 6).sqrt() is None
        assert (z1**2).sqrt() == z1
        assert z1.conj() == ZSqrtTwo(1, 2)
        assert z2.adj2() == ZSqrtTwo(3, -4) and z1.adj2() == ZSqrtTwo(1, -2)
        assert z2 // 2 == ZSqrtTwo(1, 2)
        assert z2 % 2 == ZSqrtTwo(1, 0)
        assert z1.to_omega() == ZOmega(a=-2, b=0, c=2, d=1)
        assert z2.to_omega() == ZOmega(a=-4, b=0, c=4, d=3)
        assert 1 - ZSqrtTwo(1, 2) == ZSqrtTwo(0, -2)

    def test_arithmetic_errors(self):
        """Test that arithmetic operations raise errors for invalid types."""
        z1 = ZSqrtTwo(1, 2)
        z2 = complex(3, 4)

        with pytest.raises(TypeError, match="Unsupported type"):
            _ = z1 + z2
        with pytest.raises(TypeError, match="Unsupported type"):
            _ = z1 * z2
        with pytest.raises(TypeError, match="Unsupported type"):
            _ = z1 / z2
        with pytest.raises(TypeError, match="Unsupported type"):
            _ = z1 // z2
        with pytest.raises(ValueError, match="Negative powers"):
            _ = z1**-2
        with pytest.raises(ValueError, match="Non-integer powers"):
            _ = z1**1.5


class TestZOmega:
    """Tests for the ZOmega class."""

    @pytest.mark.parametrize(
        "a, b, c, d",
        [
            (0, 0, 0, 0),
            (1, 2, 3, 4),
            (-1, -2, -3, -4),
            (5, -6, 7, -8),
        ],
    )
    def test_init_and_repr(self, a, b, c, d):
        """Test that ZOmega initializes correctly and has correct representations."""
        z_omega = ZOmega(a, b, c, d)
        assert z_omega.a == a
        assert z_omega.b == b
        assert z_omega.c == c
        assert z_omega.d == d
        string_repr = [
            f"{coeff} {var}".strip()
            for coeff, var in zip([a, b, c, d], ["ω^3", "ω^2", "ω", ""])
            if coeff
        ]
        assert str(z_omega) == " + ".join(string_repr) if string_repr else "0"
        assert repr(z_omega) == f"ZOmega(a={a}, b={b}, c={c}, d={d})"

    def test_arithmetic_operations(self):
        """Test arithmetic operations on ZOmega."""
        z1 = ZOmega(1, 2, 3, 4)
        z2 = ZOmega(5, 6, 7, 8)

        assert z1 + z2 == ZOmega(6, 8, 10, 12) == complex(z1) + complex(z2)
        assert z2 + 10 == ZOmega(5, 6, 7, 18) == complex(z2) + 10
        assert z2 + 2.0 == ZOmega(5, 6, 7, 10) == complex(z2) + 2.0
        assert z1 - z2 == ZOmega(-4, -4, -4, -4) == complex(z1) - complex(z2)
        assert z1 * z2 == ZOmega(60, 56, 36, -2) == complex(z1) * complex(z2)
        assert z2 * 10 == ZOmega(50, 60, 70, 80) == complex(z2) * 10
        assert z2 * 2.0 == ZOmega(10, 12, 14, 16) == complex(z2) * 2.0
        assert z1**2 == ZOmega(20, 24, 20, 6) == complex(z1) ** 2
        assert z2**0 == ZOmega(0, 0, 0, 1) == complex(z2) ** 0
        assert z1.conj() == ZOmega(-3, -2, -1, 4) == complex(z1).conjugate()
        assert z2.adj2() == ZOmega(-5, 6, -7, 8) and z1.adj2() == ZOmega(-1, 2, -3, 4)
        assert (z2 * 4) / 2 == ZOmega(5 * 2, 6 * 2, 7 * 2, 8 * 2)
        assert z2 // 2 == ZOmega(5 // 2, 6 // 2, 7 // 2, 8 // 2)
        assert abs(z2) == 14788 and abs(z1) == 388  # Computed manually
        assert ZOmega(0, 0, 0, 5) == 5
        assert z1.parity() == 0 and z2.parity() == 0  # Both z1 and z2 have even parity
        assert z2.norm() == abs(complex(z2)) * abs(complex(z2.conj()))
        assert (z1 - ZOmega(a=2, b=2, c=2)).to_sqrt_two() == ZSqrtTwo(a=4, b=1)
        assert 1 - ZOmega() == ZOmega(d=1)

    def test_arithmetic_errors(self):
        """Test that arithmetic operations raise errors for invalid types."""
        z1 = ZOmega(1, 2, 3, 4)
        z2 = "Dyadic(1, 2)"

        with pytest.raises(TypeError, match="Unsupported type"):
            _ = z1 + z2
        with pytest.raises(TypeError, match="Unsupported type"):
            _ = z1 * z2
        with pytest.raises(TypeError, match="Unsupported type"):
            _ = z1 / z2
        with pytest.raises(TypeError, match="Unsupported type"):
            _ = z1 // z2
        with pytest.raises(ValueError, match="Negative powers"):
            _ = z1**-2
        with pytest.raises(ValueError, match="Cannot convert ZOmega to ZSqrtTwo"):
            _ = z1.to_sqrt_two()
        with pytest.raises(ValueError, match="Non-integer powers"):
            _ = z1**1.5


class TestDyadicMatrix:
    """Tests for the DyadicMatrix class."""

    def test_init_and_repr(self):
        """Test that DyadicMatrix initializes correctly and has correct representations."""
        z1 = ZOmega(1, 2, 3, 4)
        z2 = ZOmega(5, 6, 7, 8)
        dyadic_matrix = DyadicMatrix(z1, z2, z1, z2)

        assert repr(dyadic_matrix) == f"DyadicMatrix(a={z1}, b={z2}, c={z1}, d={z2}, k=0)"
        assert str(dyadic_matrix) == f"[[{z1}, {z2}], [{z1}, {z2}]]"
        assert dyadic_matrix * 2 == DyadicMatrix(2 * z1, 2 * z2, 2 * z1, 2 * z2)
        assert np.allclose(
            dyadic_matrix.ndarray,
            np.array([[complex(z1), complex(z2)], [complex(z1), complex(z2)]]),
        )
        assert dyadic_matrix.conj() == DyadicMatrix(z1.conj(), z2.conj(), z1.conj(), z2.conj())
        assert dyadic_matrix.adj2() == DyadicMatrix(z1.adj2(), z2.adj2(), z1.adj2(), z2.adj2())
        assert dyadic_matrix * 2 == dyadic_matrix.mult2k(k=1)
        assert dyadic_matrix * 1.0 == dyadic_matrix.mult2k(k=0)
        assert dyadic_matrix + 2.0 == DyadicMatrix(z1, z2, z1, z2) + (2 + 0j) == dyadic_matrix + 2

        z3 = DyadicMatrix(z1, z2, z1, z2, k=2)
        z4, z5 = ZOmega(a=-3, b=6, c=9, d=3), ZOmega(a=-3, b=18, c=21, d=3)
        assert dyadic_matrix + z3 == DyadicMatrix(z4, z5, z4, z5, k=1)

    def test_arithmetic_errors(self):
        """Test that arithmetic operations raise errors for invalid types."""
        z1 = ZOmega(1, 2, 3, 4)
        z2 = "Dyadic(1, 2)"
        dyadic_matrix = DyadicMatrix(z1, z2, z1, z2)

        with pytest.raises(TypeError, match="Unsupported type"):
            _ = dyadic_matrix + z2
        with pytest.raises(TypeError, match="Unsupported type"):
            _ = dyadic_matrix * z2
        with pytest.raises(TypeError, match="Unsupported type"):
            _ = dyadic_matrix @ z2


class TestSO3Matrix:
    """Tests for the SO3Matrix class."""

    def test_init_and_repr(self):
        """Test that SO3Matrix initializes correctly and has correct representations."""
        z1 = ZOmega(1, 2, 3, 4)
        z2 = ZOmega(5, 6, 7, 8)
        dyadic_matrix = DyadicMatrix(z1, z2, z1, z2)
        so3_matrix = SO3Matrix(dyadic_matrix)

        assert repr(so3_matrix) == f"SO3Matrix(matrix={dyadic_matrix}, k={so3_matrix.k})"

        so3mat, so3k = so3_matrix.so3mat, so3_matrix.k
        str_repr = "["
        for i in range(3):
            str_repr += f"[{so3mat[i][0]}, {so3mat[i][1]}, {so3mat[i][2]}], \n"
        str_repr = str_repr.rstrip(", \n") + "]" + (f" * 1 / √2^{so3k}" if so3k else "")
        assert str(so3_matrix) == str_repr

        assert np.allclose(
            so3_matrix.parity_mat, np.array([[1, 0, 0], [0, 0, 0], [1, 0, 0]])
        )  # Computed manually
        assert np.allclose(so3_matrix.parity_vec, [1, 0, 1])  # Computed manually

        z3 = ZOmega()
        assert np.allclose(SO3Matrix(DyadicMatrix(z3, z3, z3, z3)).ndarray, 0.0)
