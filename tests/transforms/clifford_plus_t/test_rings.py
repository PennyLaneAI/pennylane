# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Test rings.py, needed for the Clifford+T transform."""

import pytest

from pennylane.transforms.decompositions.clifford_plus_t.rings import (
    root_two,
    omega,
    ZRootTwo,
    DRootTwo,
    Z2,
    Dyadic,
    Fraction,
    ZOmega,
    DOmega,
    hibit,
)


class TestDyadic:
    """Test the Dyadic class."""

    def test_invalid_init(self):
        """Test Dyadic fails to initialize with invalid inputs."""
        with pytest.raises(TypeError, match="x and k must be ints"):
            _ = Dyadic(1.0, 2.0)

        with pytest.raises(ValueError, match="exponent k must be greater than or equal to zero"):
            _ = Dyadic(1, -1)

    def test_non_Dyadic_fraction(self):
        """Test that non-Dyadic fractions fail to be constructed."""
        with pytest.raises(TypeError, match="only Dyadic fractions are expected, got 1/3"):
            _ = Fraction(1, 3)

    def test_conjugate(self):
        """Test that Dyadic has a conjugate defined."""
        d = Dyadic(1, 3)
        assert d.conjugate() is d

    def test_subtract(self):
        """Test substraction dunder."""
        d = Dyadic(3, 4)
        assert d - 4 == Dyadic(-61, 4)
        assert d - Dyadic(1, 5) == Dyadic(5, 5)
        with pytest.raises(TypeError, match="cannot subtract 1 of type ZRootTwo from Dyadic"):
            _ = d - ZRootTwo(1, 0)


class TestZRootTwo:
    """Test the ZRootTwo class."""

    def test_negative_sqrt_is_None(self):
        """The square root of negative values doesn't exist, so it should return None."""
        assert ZRootTwo(-1, 2).sqrt() is None

    def test_constructor(self):
        """Test construction of ZRootTwo instances."""
        z = ZRootTwo(1, 2)
        assert isinstance(z.a, int)
        assert isinstance(z.b, int)

        z_float = ZRootTwo(1.0, 2)
        assert isinstance(z_float.a, int)
        assert isinstance(z_float.b, int)

        with pytest.raises(TypeError, match="Cannot cast 1.1 of unknown type"):
            _ = ZRootTwo(1.1, 2)

    def test_truediv(self):
        """Test the division of ZRootTwo rings with other types."""
        assert root_two(3, 4) / root_two(0, 2) == DRootTwo(2, Dyadic(3, 2))
        with pytest.raises(TypeError, match="cannot divide RootTwo ring by 1 of type int"):
            _ = root_two(1, 2) / 1

    def test_invalid_pow(self):
        """Test that the pow dunder does not work with non-int values."""
        z = root_two(0, 2)
        with pytest.raises(ValueError, match="Cannot raise RootTwo to non-int power 2.0"):
            _ = z**2.0

        assert z**-3 == DRootTwo(0, Dyadic(1, 5))
        assert root_two(1, 2) ** 3 == ZRootTwo(25, 22)


class TestDRootTwo:
    """Test the DRootTwo class."""

    # pylint:disable=too-few-public-methods

    def test_constructor(self):
        """Test construction of DRootTwo instances."""
        d = DRootTwo(1, Dyadic(2, 0))
        assert isinstance(d.a, Dyadic)
        assert isinstance(d.b, Dyadic)

        d_float = DRootTwo(1.0, 2)
        assert isinstance(d_float.a, Dyadic)
        assert d_float.a == Dyadic(1, 0)
        assert isinstance(d_float.b, Dyadic)

        with pytest.raises(TypeError, match="Cannot cast 1.1 of unknown type"):
            _ = DRootTwo(1.1, 2)


class TestOmega:
    """Test the Omega rings."""

    def test_invalid_pow(self):
        """Test that the pow dunder does not work with certain values."""
        w = omega(1, 2, 3, 4)
        with pytest.raises(ValueError, match="Cannot raise Omage to non-int power 2.0"):
            _ = w**2.0
        with pytest.raises(ValueError, match="cannot raise Omega to negative power"):
            _ = w**-3

    def test_invalid_arithmetic(self):
        """Omega rings cannot perform arithmetic with other types.
        It is largely unimplemented because it is never needed."""
        with pytest.raises(TypeError, match="cannot multiply Omega value"):
            _ = omega(1, 2, 3, 4) * 2

        with pytest.raises(TypeError, match="cannot add Omega value"):
            _ = omega(1, 2, 3, 4) + 2

    @pytest.mark.parametrize(
        "abcd, expected",
        [
            ((1, 0, 0, 0), 3),
            ((0, 0, 1, 0), 1),
            ((-1, 0, 0, 0), 7),
            ((0, 0, -1, 0), 5),
            ((1, 0, -1, 0), None),
            ((2, 0, 0, 0), None),
        ],
    )
    def test_log(self, abcd, expected):
        """Test the log function."""
        assert ZOmega(*abcd).log() == expected

    def test_to_root_two(self):
        """Test the RootTwo converter."""
        assert ZOmega(4, 0, -4, 3).to_root_two() == ZRootTwo(3, -4)
        with pytest.raises(ValueError, match="non-real value"):
            _ = ZOmega(1, 2, 3, 4).to_root_two()

    def test_omega_constructor(self):
        zomega = omega(1, 2, 3, 4)
        assert isinstance(zomega, ZOmega)
        assert zomega == ZOmega(1, 2, 3, 4)

        domega = omega(1, 2, 3, Dyadic(3, 2))
        assert isinstance(domega, DOmega)
        assert domega == DOmega(1, 2, 3, Dyadic(3, 2))

        with pytest.raises(TypeError, match=r"Unknown types {<class 'float'>} found"):
            _ = omega(1.1, 2.2, 3.3, 0)


class TestRepr:
    """Test the repr of various rings."""

    def test_repr_Z2(self):
        """Test Z2 repr."""
        assert repr(Z2(0, 1, 2, 3)) == "Z2[0101]"

    @pytest.mark.parametrize(
        "abcd,expected",
        [
            ((1, 2, 3, 4), "(1ω**3 + 2ω**2 + 3ω + 4)/1"),
            ((Dyadic(5, 3), 2, 3, 4), "(5ω**3 + 16ω**2 + 24ω + 32)/8"),
            ((Dyadic(5, 3), Dyadic(7, 4), 0, 4), "(10ω**3 + 7ω**2 + 64)/16"),
            ((0, 0, 0, 0), "0"),
            ((2, 0, Dyadic(-3, 2), 0), "(8ω**3 + -3ω)/4"),
            ((0, 0, Dyadic(2, 4), 0), "2ω/16"),
        ],
    )
    def test_repr_DOmega(self, abcd, expected):
        """Test DOmega repr."""
        assert repr(DOmega(*abcd)) == expected

    @pytest.mark.parametrize(
        "abcd,expected",
        [
            ((1, 2, 3, 4), "1ω**3 + 2ω**2 + 3ω + 4"),
            ((0, 0, 0, 0), "0"),
            ((2, 0, -3, 0), "2ω**3 + -3ω"),
            ((0, 0, 1, 0), "1ω"),
        ],
    )
    def test_repr_ZOmega(self, abcd, expected):
        """Test DOmega repr."""
        assert repr(ZOmega(*abcd)) == expected

    @pytest.mark.parametrize(
        "a,b,expected",
        [
            (2, 0, "2"),
            (2, 1, "2+1√2"),
            (0, 1, "1√2"),
            (Dyadic(1, 1), Dyadic(3, 2), "1/2+3/4√2"),
            (Dyadic(1, 1), Dyadic(-3, 2), "1/2-3/4√2"),
        ],
    )
    def test_repr_RootTwo(self, a, b, expected):
        """Test RootTwo reprs."""
        assert repr(root_two(a, b)) == expected


class TestEqualDunder:
    """Test the __eq__ dunder for various rings."""

    def test_equal_Z2(self):
        """Test Z2 equality."""
        assert Z2(4, 5, 6, 7) == Z2(2, 3, 4, 5)  # they get reduced at construction
        assert Z2(1, 0, 1, 0) != ZOmega(1, 0, 1, 0)  # type matters
        assert Z2(1, 0, 1, 0) != Z2(1, 1, 1, 1)

    def test_equal_omega(self):
        """Test various equality rules for omega rings."""
        assert DOmega(1, 2, 3, 4) == ZOmega(1, 2, 3, 4)  # which Omega ring does not matter
        assert ZOmega(0, 0, 0, 5) == 5
        assert DOmega(0, 0, 0, Dyadic(3, 4)) == Dyadic(3, 4)

        with pytest.raises(TypeError, match="cannot compare Omega ring with"):
            # they are equivalent but it is not implemented
            assert ZOmega(-1, 0, 1, 1) != ZRootTwo(1, 1)


@pytest.mark.parametrize("n,expected", [(0, 0), (3, 2), (4, 3)])
def test_hibit(n, expected):
    """Test the hibit helper function for intsqrt."""
    assert hibit(n) == expected
