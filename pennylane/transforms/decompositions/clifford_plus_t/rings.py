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
"""Various ring definitions for the gridsynth implementation."""
# pylint:disable=too-few-public-methods

from abc import ABC, abstractclassmethod
from fractions import Fraction as _Fraction
import numpy as np

SQRT2 = float(np.sqrt(2))
OMEGA = (1 + 1j) / SQRT2


class Dyadic:
    """Dyadic numbers, of the form x/(2**k) | x,k E Z"""

    def __init__(self, x, k=0):
        if not isinstance(x, int) or not isinstance(k, int):
            raise TypeError(f"x and k must be ints for Dyadic; got: {x}, {k}")
        if k < 0:
            raise ValueError(f"exponent k must be greater than or equal to zero; got {k}")
        self.x = x
        self.k = k

    def __float__(self):
        return self.x / (2**self.k)

    def __mul__(self, other):
        if isinstance(other, Dyadic):
            return Fraction(self.x * other.x, 2 ** (self.k + other.k))
        if isinstance(other, int):
            return Fraction(self.x * other, 2**self.k)
        return float(self) * other

    def __add__(self, other):
        if isinstance(other, Dyadic):
            if self.k == other.k:
                return Fraction(self.x + other.x, 2**self.k)
            a, b = (self, other) if self.k < other.k else (other, self)
            k_delta = b.k - a.k
            return Fraction(a.x * int(2**k_delta) + b.x, 2**b.k)
        if isinstance(other, int):
            return Fraction(self.x + other * 2**self.k, 2**self.k)
        return float(self) + other

    def __sub__(self, other):
        if isinstance(other, Dyadic):
            return self + (-other)
        if isinstance(other, int):
            return Fraction(self.x - other * 2**self.k, 2**self.k)
        raise TypeError(f"cannot subtract {other} of type {type(other).__name__} from Dyadic")

    def __rsub__(self, other):
        return -self + other

    def __eq__(self, other):
        if isinstance(other, Dyadic):
            return self.x == other.x and self.k == other.k
        return np.isclose(float(self), float(other))

    def __gt__(self, other):
        return float(self) > float(other)

    def __repr__(self):
        return str(self.x) if self.k == 0 else f"{self.x}/{2**self.k}"

    def __neg__(self):
        return Dyadic(-self.x, self.k)

    __radd__ = __add__
    __rmul__ = __mul__

    def conjugate(self):
        """The complex conjugate. Defaults to itself, needed for Matrix.adjoint()."""
        return self


class Fraction(_Fraction):
    """An override of fractions.Fraction to returns ints or Dyadics when appropriate."""

    def __new__(cls, numerator=0, denominator=None, *, _normalize=True):
        self = super().__new__(
            Fraction, numerator=numerator, denominator=denominator, _normalize=_normalize
        )
        if self.denominator == 1:
            return self.numerator
        if (log_d := np.log2(self.denominator)).is_integer():
            return Dyadic(self.numerator, int(log_d))
        raise TypeError(f"only Dyadic fractions are expected, got {self}")


def root_two(a, b):
    """Determine the RootTwo ring (if any) that should be used, and return that type."""
    types = {type(a), type(b)} - {int}

    if types == {Dyadic}:
        return DRootTwo(a, b)
    if types:
        return a + b * SQRT2
    return ZRootTwo(a, b)


class RootTwo(ABC):
    """Any object that can be represented as a + b√2"""

    def __init__(self, a, b):
        """Cast inputs to the correct type before saving to the ring."""
        self.a = self.cast(a)
        self.b = self.cast(b)

    @abstractclassmethod
    def cast(cls, value):
        """Cast a value to the type for this ring."""

    def conjugate(self):
        """The complex conjugate. Defaults to itself, needed for Matrix.adjoint()."""
        return self

    def adj2(self):
        """Return the root-2 adjoint."""
        return type(self)(self.a, -self.b)

    def norm(self):
        """Return the norm."""
        return self.a**2 - 2 * self.b**2

    def __repr__(self):
        if self.b == 0:
            return repr(self.a)
        if self.a == 0:
            return f"{self.b}√2"
        if self.b > 0:
            return f"{self.a}+{self.b}√2"
        return f"{self.a}{self.b}√2"

    def __float__(self):
        """Convert from ring to float."""
        return float(self.a) + float(self.b) * SQRT2

    def __add__(self, other):
        if isinstance(other, RootTwo):
            return root_two(self.a + other.a, self.b + other.b)
        return root_two(self.a + other, self.b)

    def __sub__(self, other):
        if isinstance(other, RootTwo):
            return root_two(self.a - other.a, self.b - other.b)
        return root_two(self.a - other, self.b)

    def __rsub__(self, other):
        return -self + other

    def __mul__(self, other):
        if isinstance(other, RootTwo):
            return root_two(
                self.a * other.a + 2 * self.b * other.b,
                self.a * other.b + self.b * other.a,
            )
        return root_two(self.a * other, self.b * other)

    def __eq__(self, other):
        if isinstance(other, RootTwo):
            return self.a == other.a and self.b == other.b
        return np.isclose(float(self), float(other))

    def __gt__(self, other):
        return float(self) > float(other)

    def __lt__(self, other):
        return float(self) < float(other)

    def __ge__(self, other):
        return float(self) >= float(other)

    def __le__(self, other):
        return float(self) <= float(other)

    def __neg__(self):
        return type(self)(-self.a, -self.b)

    def __pow__(self, power):
        if not isinstance(power, int):
            raise ValueError(f"Cannot raise RootTwo to non-int power {power}")
        if power == 0:
            return type(self)(1, 0)
        if power < 0:
            return 1 / (self**-power)
        result = self
        power -= 1
        while power > 0:
            result *= self
            power -= 1
        return result

    def __truediv__(self, other):
        if isinstance(other, RootTwo):
            return self * (1 / other)
        raise TypeError(f"cannot divide RootTwo ring by {other} of type {type(other).__name__}")

    def __rtruediv__(self, other):
        return other / float(self)

    __radd__ = __add__
    __rmul__ = __mul__


class ZRootTwo(RootTwo):
    """Z[√2]"""

    @classmethod
    def cast(cls, value):
        t = type(value)
        if t is int:
            return value
        if t is float and value.is_integer():
            return int(value)
        raise TypeError(f"Cannot cast {value} of unknown type {type(value)} to int")

    def sqrt(self):
        """Return the square root."""
        a = self.a
        d = self.norm()
        r = intsqrt(d)
        x1 = intsqrt((a + r) // 2)
        x2 = intsqrt((a - r) // 2)
        y1 = intsqrt((a - r) // 4)
        y2 = intsqrt((a + r) // 4)
        for o in (ZRootTwo(x1, y1), ZRootTwo(x2, y2), ZRootTwo(x1, -y1), ZRootTwo(x2, -y2)):
            if o * o == self:
                return o
        return None

    def denomexp(self):  # pylint:disable=no-self-use
        """Return the smallest denominator exponent."""
        return 0

    def __rtruediv__(self, other):
        if isinstance(other, int):
            k = self.a**2 - 2 * self.b**2
            return root_two(Fraction(other * self.a, k), Fraction(-other * self.b, k))
        return super().__rtruediv__(other)


class DRootTwo(RootTwo):
    """D[√2]"""

    @classmethod
    def cast(cls, value):
        t = type(value)
        if t is Dyadic:
            return value
        if t is int:
            return Dyadic(value)
        if t is float and value.is_integer():
            return Dyadic(int(value))
        raise TypeError(f"Cannot cast {value} of unknown type {type(value)} to Dyadic")

    def denomexp(self):
        """Return the smallest denominator exponent."""
        return max(2 * self.a.k, 2 * self.b.k - 1)


def omega(a, b, c, d):
    """Determine the Omega ring (if any) that should be used, and return that type."""
    types = {type(i) for i in (a, b, c, d)} - {int}
    if not types:
        return ZOmega(a, b, c, d)
    if types == {Dyadic}:
        return DOmega(a, b, c, d)
    raise TypeError(
        f"Unknown types {types - {Dyadic}} found. Cannot make Omega-ring instance from values {(a, b, c, d)}"
    )


class Omega(ABC):
    """Any Omega-ring."""

    def __init__(self, a, b, c, d):
        self.a = self.cast(a)
        self.b = self.cast(b)
        self.c = self.cast(c)
        self.d = self.cast(d)

    @abstractclassmethod
    def cast(cls, value):
        """Cast a value to the type for this ring."""

    @staticmethod
    def from_root_two(v: RootTwo):
        """Convert a real RootTwo ring value to a Omega value."""
        return omega(-v.b, 0, v.b, v.a)

    def to_root_two(self):
        """Convert a real-valued Omega value to a RootTwo value, or fail."""
        if self.imag != 0:
            raise ValueError("non-real value:", self)
        return self.real

    def log(self):
        """Return the omega-log of this instance."""
        vals = [self.a, self.b, self.c, self.d]
        if vals.count(0) != 3:
            return None
        if 1 in vals:
            return 3 - vals.index(1)
        if -1 in vals:
            return 7 - vals.index(-1)
        return None

    def __repr__(self):
        vals = [self.a, self.b, self.c, self.d]
        coeffs = ["ω**3", "ω**2", "ω", ""]
        res = [f"{val}{coeff}" for val, coeff in zip(vals, coeffs) if val != 0]
        if len(res) == 0:
            return "0"
        if len(res) == 1:
            return res[0]
        return " + ".join(res)

    def __complex__(self):
        return self.a * OMEGA**3 + self.b * 1j + self.c * OMEGA + self.d

    @property
    def real(self):
        """The real component of a Omega value."""
        return root_two(self.d, (self.c - self.a) * Dyadic(1, 1))

    @property
    def imag(self):
        """The imaginary component of a Omega value."""
        return root_two(self.b, (self.a + self.c) * Dyadic(1, 1))

    def conjugate(self):
        """The complex conjugate."""
        return type(self)(-self.c, -self.b, -self.a, self.d)

    def adj2(self):
        """Return the root-2 adjoint. Assumes elements are not RootTwo types."""
        return type(self)(-self.a, self.b, -self.c, self.d)

    def norm(self):
        """Return the norm. Assumes elements are not RootTwo types."""
        a, b, c, d = (self.a, self.b, self.c, self.d)
        return (a**2 + b**2 + c**2 + d**2) ** 2 - 2 * (a * b + b * c + c * d - d * a) ** 2

    def __mul__(self, other):
        # if isinstance(other, (Dyadic, int)):
        #     return omega(self.a * other, self.b * other, self.c * other, self.d * other)
        # if isinstance(other, RootTwo):
        #     other = self.from_root_two(other)
        if isinstance(other, Omega):
            a, b, c, d = (self.a, self.b, self.c, self.d)
            _a, _b, _c, _d = (other.a, other.b, other.c, other.d)
            new_a = a * _d + b * _c + c * _b + d * _a
            new_b = b * _d + c * _c + d * _b - a * _a
            new_c = c * _d + d * _c - a * _b - b * _a
            new_d = d * _d - a * _c - b * _b - c * _a
            return omega(new_a, new_b, new_c, new_d)
        raise TypeError(
            f"cannot multiply Omega value by value `{other}` of unknown type {type(other).__name__}"
        )

    def __add__(self, other):
        # if isinstance(other, (Dyadic, int)):
        #     return omega(self.a + other, self.b + other, self.c + other, self.d + other)
        if isinstance(other, Omega):
            return omega(self.a + other.a, self.b + other.b, self.c + other.c, self.d + other.d)
        raise TypeError(
            f"cannot add Omega value to value `{other}` of unknown type {type(other).__name__}"
        )

    def __sub__(self, other):
        return self + (-other)

    def __neg__(self):
        return type(self)(-self.a, -self.b, -self.c, -self.d)

    def __eq__(self, other):
        if isinstance(other, Omega):
            return (
                self.a == other.a and self.b == other.b and self.c == other.c and self.d == other.d
            )
        if isinstance(other, (int, Dyadic)):
            return self.a == 0 and self.b == 0 and self.c == 0 and self.d == other
        raise TypeError(f"cannot compare Omega ring with {other} of type {type(other).__name__}")

    def __pow__(self, power):
        if not isinstance(power, int):
            raise ValueError(f"Cannot raise Omage to non-int power {power}")
        if power == 0:
            return ZOmega(0, 0, 0, 1)
        if power < 0:
            raise ValueError("cannot raise Omega to negative power")
        result = self
        power -= 1
        while power > 0:
            result *= self
            power -= 1
        return result


class ZOmega(Omega):
    """Z[ω]"""

    @classmethod
    def cast(cls, value):
        return ZRootTwo.cast(value)

    def denomexp(self):  # pylint:disable=no-self-use
        """Return the smallest denominator exponent."""
        return 0


class DOmega(Omega):
    """D[ω]"""

    @classmethod
    def cast(cls, value):
        return DRootTwo.cast(value)

    def __repr__(self):
        vals = [self.a, self.b, self.c, self.d]
        max_k = max(v.k for v in vals)
        vals = [v.x * 2 ** (max_k - v.k) for v in vals]
        coeffs = ["ω**3", "ω**2", "ω", ""]
        res = [f"{val}{coeff}" for val, coeff in zip(vals, coeffs) if val != 0]
        if len(res) == 0:
            return "0"
        if len(res) == 1:
            return f"{res[0]}/{2**max_k}"
        return f"({' + '.join(res)})/{2**max_k}"

    def denomexp(self):
        """Return the smallest denominator exponent."""
        vals = [self.a, self.b, self.c, self.d]
        k = max(val.k for val in vals)
        a, b, c, d = [(val.x if val.k == k else 0) for val in vals]
        if k > 0 and (a - c) % 2 == 0 and (b - d) % 2 == 0:
            return 2 * k - 1
        return 2 * k


class Z2:
    """Z2[ω]. Like Z[ω], but elements are in {0, 1}"""

    def __init__(self, a, b, c, d):
        self.a = a % 2
        self.b = b % 2
        self.c = c % 2
        self.d = d % 2

    def reducible(self):
        """Return whether or not this instance is reducible."""
        return self.a == self.c and self.b == self.d

    def __hash__(self):
        return hash((self.a, self.b, self.c, self.d))

    def __eq__(self, other):
        if not isinstance(other, Z2):
            return False
        return self.a == other.a and self.b == other.b and self.c == other.c and self.d == other.d

    def __repr__(self):
        return f"Z2[{self.a}{self.b}{self.c}{self.d}]"


def omega_power(k):
    """ω ** k. Use k%8 because ω is cyclic."""
    return ZOmega(0, 0, 1, 0) ** (k % 8)


def hibit(n: int) -> int:
    """Get the position of the left-most bit, plus one. O(nlogn) steps."""
    if n == 0:
        return 0

    def aux2(upper, lower):
        if (upper - lower) < 2:
            return upper
        middle = (upper + lower) // 2
        if n >= 2**middle:
            return aux2(upper, middle)
        return aux2(middle, lower)

    def aux(k):
        if n >= 2**k:
            return aux(2 * k)
        return aux2(k, k // 2)

    return aux(1)


def intsqrt(n: int) -> int:
    """Return the floor of the square root of n, using integer arithmetic to avoid errors."""
    if n <= 0:
        return 0

    def iterate(m):
        m_sq = m * m
        if m_sq <= n < (m_sq + 2 * m + 1):
            return m
        return iterate((m + n // m) // 2)

    b = hibit(n)
    a = 2 ** (b // 2)
    return iterate(a)
