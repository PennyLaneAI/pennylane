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
"""Tests for the norm solver functions for Clifford+T decomposition."""

from functools import reduce

import pytest
from flaky import flaky

from pennylane.ops.op_math.decompositions.norm_solver import (
    _factorize_prime_zomega,
    _factorize_prime_zsqrt_two,
    _gcd,
    _integer_factorize,
    _primality_test,
    _prime_factorize,
    _solve_diophantine,
    _sqrt_modulo_p,
)
from pennylane.ops.op_math.decompositions.rings import ZOmega, ZSqrtTwo


class TestGCD:
    """Tests for the GCD function."""

    @pytest.mark.parametrize(
        "a, b, expected",
        [
            (ZSqrtTwo(28), ZSqrtTwo(12), ZSqrtTwo(4)),
            (ZSqrtTwo(15), ZSqrtTwo(25), ZSqrtTwo(5)),
            (ZSqrtTwo(81), ZSqrtTwo(63), ZSqrtTwo(9)),
            (ZSqrtTwo(144), ZSqrtTwo(108), ZSqrtTwo(36)),
            (ZSqrtTwo(23, 72), ZSqrtTwo(23, 72), ZSqrtTwo(23, 72)),
            (ZSqrtTwo(28, 15), ZSqrtTwo(12, 25), ZSqrtTwo(58, 41)),
            (ZSqrtTwo(35, 42), ZSqrtTwo(22, 16), ZSqrtTwo(11, 8)),
            (ZSqrtTwo(-1, 7), ZSqrtTwo(7, 0), ZSqrtTwo(1, 0)),
        ],
    )
    def test_gcd_zsqrt_two(self, a, b, expected):
        """Test the GCD function."""
        assert _gcd(a, b) == expected
        res1, res2 = a / expected, b / expected
        assert res1 * expected == a
        assert res2 * expected == b

    @pytest.mark.parametrize(
        "a, b, expected",
        [
            (ZOmega(d=28), ZOmega(d=12), ZOmega(d=4)),
            (ZOmega(d=15), ZOmega(d=25), ZOmega(d=5)),
            (ZOmega(d=81), ZOmega(d=63), ZOmega(d=9)),
            (ZOmega(d=144), ZOmega(d=108), ZOmega(d=36)),
            (ZOmega(d=28, b=12), ZOmega(d=32, b=44), ZOmega(d=4)),
            (ZOmega(a=81, c=63), ZOmega(a=36, c=42), ZOmega(a=3, c=3)),
        ],
    )
    def test_gcd_zomega(self, a, b, expected):
        """Test the GCD function."""
        assert _gcd(a, b) == expected


class TestFactorization:
    """Tests for the factorization functions."""

    @pytest.mark.parametrize(
        "num, expected",
        [
            (28, None),  # 28 = 2^2 * 7 (7 is not included)
            (1, []),
            (0, []),
            (100, [2, 2, 5, 5]),
            (60, [2, 2, 3, 5]),
            (53, [53]),  # 53 is prime
        ],
    )
    def test_prime_factorize(self, num, expected):
        """Test prime factorization."""
        assert _prime_factorize(num) == expected

    @flaky(max_runs=5)
    @pytest.mark.parametrize(
        "num, expected",
        [
            (28, [2, 4, 7]),
            (1, [None]),
            (0, [None]),
            (100, [2, 4, 5, 10, 20, 50]),
            (60, [2, 3, 4, 5, 15, 20, 30]),
            (47, [None]),  # 47 is prime
        ],
    )
    def test_integer_factorize(self, num, expected):
        """Test integer factorization."""
        assert _integer_factorize(num) in expected

    @pytest.mark.parametrize(
        "num, expected",
        [
            (2, [ZSqrtTwo(0, 1), ZSqrtTwo(0, 1)]),
            (3, [ZSqrtTwo(3, 0)]),
            (7, [ZSqrtTwo(3, 1), ZSqrtTwo(3, -1)]),
            (29, [ZSqrtTwo(29, 0)]),
            (47, [ZSqrtTwo(7, 1), ZSqrtTwo(7, -1)]),
            (15, None),
        ],
    )
    def test_factorize_prime_zsqrt_two(self, num, expected):
        """Test factorization of prime Z-sqrt(2)."""
        assert _factorize_prime_zsqrt_two(num) == expected
        if expected is not None:
            assert reduce(lambda x, y: x * y, expected, ZSqrtTwo(1, 0)) == num

    @pytest.mark.parametrize(
        "num, expected",
        [
            (3, ZOmega(a=1, c=1, d=1)),
            (27, None),
            (5, ZOmega(b=-1, d=2)),
            (7, None),
            (11, ZOmega(a=1, c=1, d=3)),
            (13, ZOmega(b=2, d=3)),
        ],
    )
    def test_factorize_prime_zomega(self, num, expected):
        """Test factorization of prime Z-omega."""
        zsqrt_two = _factorize_prime_zsqrt_two(num)[-1]
        assert _factorize_prime_zomega(zsqrt_two, num) == expected
        if expected is not None:
            assert num in (expected, expected * expected.conj())

    @pytest.mark.parametrize(
        "num, expected",
        [
            (2, True),
            (4, False),
            (5, True),
            (29, True),
            (561, False),  # 561 is not prime (3*11*17)
            (1729, False),  # 1729 is not prime (1*7*13*19)
            (7901, True),  # 7901 is prime
            (41041, False),  # 41041 is not prime (7*11*13*41)
            (101 * 431, False),  # 101 and 431 are prime, but product is not
        ],
    )
    def test_primality_test(self, num, expected):
        """Test primality test."""
        assert _primality_test(num) == expected

    @pytest.mark.parametrize(
        "nums, expected",
        [
            ((3, 2), 1),
            ((0, 1), 0),
            ((4, 5), 3),
            ((9, 11), 3),
            ((16, 17), 4),
            ((25, 29), 24),
            ((-2, 7), None),
            ((-1, 6), None),  # 6 is not prime
            ((56, 101), 37),
            ((3, 4), None),  # 4 is not prime
        ],
    )
    def test_sqrt_modulo_p(self, nums, expected):
        """Test square root modulo p."""
        assert _sqrt_modulo_p(*nums) == expected
        if expected is not None and nums[1] != 2:
            assert expected**2 % nums[1] == nums[0]

    @pytest.mark.parametrize(
        "num, expected",
        [
            (ZSqrtTwo(0, 0), ZOmega()),
            (ZSqrtTwo(0, 1), None),
            (ZSqrtTwo(2, 1), ZOmega(a=0, b=0, c=1, d=1)),
            (ZSqrtTwo(2, -1), ZOmega(a=1, b=-1, c=0, d=0)),
            (ZSqrtTwo(7, 0), None),
            (ZSqrtTwo(23, 0), None),
            (ZSqrtTwo(7, 2), -ZOmega(a=1, b=1, c=2, d=-1)),
            (ZSqrtTwo(17, 0), None),
            (ZSqrtTwo(5, 2), ZOmega(a=-2, b=-1, c=0, d=0)),
            (ZSqrtTwo(13, 6), ZOmega(a=3, b=0, c=0, d=-2)),
        ],
    )
    def test_solve_diophantine(self, num, expected):
        """Test `solve_diophantine` solves diophantine equation."""
        assert _solve_diophantine(num) == expected
        if expected is not None:
            assert (expected.conj() * expected).to_sqrt_two() == num

    @pytest.mark.parametrize(
        "num, expected, factor",
        [
            (
                ZOmega(-26687414, 10541729, 10614512, 40727366),
                ZOmega(-30805761, 23432014, -2332111, -20133911),
                52,
            ),
            (
                ZOmega(-22067493351, 22078644868, 52098814989, 16270802723),
                ZOmega(-4737137864, -21764478939, 70433513740, -5852668010),
                73,
            ),
        ],
    )
    def test_solve_diophantine_large_number(self, num, expected, factor):
        """Test `solve_diophantine` solves diophantine equation."""
        xi = ZSqrtTwo(2**factor) - num.norm().to_sqrt_two()
        assert _solve_diophantine(xi) == expected
