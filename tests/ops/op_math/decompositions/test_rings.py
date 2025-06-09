# Copyright 2018-2025 Xanadu Quantum Technologies Inc.

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

import pytest

from pennylane.ops.op_math.decompositions.rings import ZOmega, ZSqrtTwo

# pylint: disable=too-few-public-methods


class TestZSqrtTwo:
    """Tests for the ZSqrtTwo class."""

    @pytest.mark.parametrize(
        "a, b",
        [
            (0, 0),
            (2, 1),
            (3, -2),
            (-4, 0),
        ],
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
        assert z1 - z2 == ZSqrtTwo(-2, -2) == float(z1) - float(z2)
        assert z1 * z2 == ZSqrtTwo(19, 10) == float(z1) * float(z2)
        assert z2 / z1 == ZSqrtTwo(0, 1)  # Computed manually
        assert z1**2 == ZSqrtTwo(9, 4) == float(z1) ** 2
        assert (z1**2).sqrt() == z1
        assert z1.conj() == ZSqrtTwo(1, 2)
        assert z2.adj2() == ZSqrtTwo(3, -4) and z1.adj2() == ZSqrtTwo(1, -2)
        assert z2 // 2 == ZSqrtTwo(1, 2)
        assert z1.to_omega() == ZOmega(a=-2, b=0, c=2, d=1)
        assert z2.to_omega() == ZOmega(a=-4, b=0, c=4, d=3)


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
        assert repr(z_omega) == f"ZOmega(a={a}, b={b}, c={c}, d={d})"
        assert (
            str(z_omega)
            == (
                f"{a} ω^3"
                if a
                else (
                    "" + " + "
                    if a and b
                    else (
                        "" f"{b} ω^2"
                        if b
                        else (
                            "" + " + "
                            if b and c
                            else (
                                "" f"{c} ω"
                                if c
                                else "" + " + " if c and d else "" f"{d}" if d else ""
                            )
                        )
                    )
                )
            )
            or f"{d}"
        )

    def test_arithmetic_operations(self):
        """Test arithmetic operations on ZOmega."""
        z1 = ZOmega(1, 2, 3, 4)
        z2 = ZOmega(5, 6, 7, 8)

        assert z1 + z2 == ZOmega(6, 8, 10, 12) == complex(z1) + complex(z2)
        assert z1 - z2 == ZOmega(-4, -4, -4, -4) == complex(z1) - complex(z2)
        assert z1 * z2 == ZOmega(60, 56, 36, -2) == complex(z1) * complex(z2)
        assert z1**2 == ZOmega(20, 24, 20, 6) == complex(z1) ** 2
        assert z1.conj() == ZOmega(-3, -2, -1, 4) == complex(z1).conjugate()
        assert z2.adj2() == ZOmega(-5, 6, -7, 8) and z1.adj2() == ZOmega(-1, 2, -3, 4)
        assert z2 // 2 == ZOmega(5 // 2, 6 // 2, 7 // 2, 8 // 2)
        assert abs(z2) == 14788 and abs(z1) == 388  # Computed manually


class TestDyadicMatrix:
    """Tests for the DyadicMatrix class."""


class TestSO3Matrix:
    """Tests for the SO3Matrix class."""
