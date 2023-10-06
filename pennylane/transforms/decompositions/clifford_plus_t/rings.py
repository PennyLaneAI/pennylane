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
OMEGA = np.exp(np.pi * 0.25j)


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
        if isinstance(other, RootTwo):
            return other * self
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
        if isinstance(other, RootTwo):
            return other + self
        return float(self) + other

    def __sub__(self, other):
        if isinstance(other, Dyadic):
            return self + (-other)
        if isinstance(other, int):
            return Fraction(self.x - other * 2**self.k, 2**self.k)
        if isinstance(other, RootTwo):
            return -other + self
        return float(self) - other

    def __rsub__(self, other):
        return -self + other

    def __truediv__(self, other):
        if isinstance(other, int):
            return Fraction(self.x, other * 2**self.k)
        return float(self) / other

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
        return self


class Matrix(np.ndarray):
    """Assumes input is a 2x2 matrix."""

    @classmethod
    def array(cls, value) -> "Matrix":
        """
        Constructor to build a 2x2 matrix.

        The ndarray constructor is messy, it's easier to provide this (like np.array). Also, we
        set the type to "object" explicitly to ensure numpy does not convert ints to np.int64,
        as that would break the type-checking in the ``root_two`` function.
        """
        return np.array(value).astype("object").view(cls)

    def inverse(self):
        """Return the inverse of the matrix, assuming |det M| = 1."""
        (a, b), (c, d) = self
        det = a * d - b * c
        if det == 1:
            return self.array([[d, -b], [-c, a]])
        if det == -1:
            return self.array([[-d, b], [c, -a]])
        raise ValueError("can only get special inverse.")

    def adjoint(self):
        """Returns the adjoint of the matrix.
        TODO: not implemented."""
        return np.conj(self).T


def root_two(a, b):
    """Determine the RootTwo ring (if any) that should be used, and return that type."""
    types = {type(a), type(b)} - {int}

    if not types.issubset({Fraction, Dyadic}):
        return a + b * SQRT2
    if types == {Dyadic}:
        return DRootTwo(a, b)
    if types:  # at least one is a Fraction
        # return QRootTwo(a, b)
        raise ValueError("No code should produce non-Dyadic fraction")
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
        return float(self) / other

    def __rtruediv__(self, other):
        if isinstance(other, RootTwo):
            return other * (1 / self)
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

    def __truediv__(self, other):
        if isinstance(other, int):
            return root_two(Fraction(self.a, other), Fraction(self.b, other))
        return super().__truediv__(other)

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

    def __truediv__(self, other):
        if isinstance(other, int):
            a = Fraction(self.a.x, other * 2**self.a.k)
            b = Fraction(self.b.x, other * 2**self.b.k)
            return root_two(a, b)
        return super().__truediv__(other)


class QRootTwo(RootTwo):
    """Q[√2]"""

    @classmethod
    def cast(cls, value):
        t = type(value)
        if t in {Dyadic, Fraction}:
            return value
        if t is int:
            return Dyadic(value)
        if t is float and value.is_integer():
            return Dyadic(int(value))
        raise TypeError(f"Cannot cast {value} of unknown type {type(value)} to Fraction")


class DOmega:
    """D[ω]"""

    def __init__(self, a: Dyadic, b: Dyadic, c: Dyadic, d: Dyadic):
        self.a = DRootTwo.cast(a)
        self.b = DRootTwo.cast(b)
        self.c = DRootTwo.cast(c)
        self.d = DRootTwo.cast(d)

    @classmethod
    def from_root_two(cls, v: RootTwo):
        """Convert a real RootTwo ring value to a DOmega value."""
        return cls(-v.b, 0, v.b, v.a)

    def __repr__(self):
        vals = [self.a, self.b, self.c, self.d]
        max_k = max(v.k for v in vals)
        vals = [v.x * 2 ** (max_k - v.k) for v in vals]
        coeffs = ["ω**3", "ω**2", "ω", ""]
        res = [f"{val}{coeff}" for val, coeff in zip(vals, coeffs) if val != 0]
        if len(res) == 0:
            return 0
        if len(res) == 1:
            return f"{res[0]}/{2**max_k}"
        return f"({' + '.join(res)})/{2**max_k}"
        # res = []
        # if self.a != 0:
        #     res.append(f"{self.a}ω**3")
        # if self.b != 0:
        #     res.append(f"{self.b}ω**2")
        # if self.c != 0:
        #     res.append(f"{self.c}ω")
        # if self.d != 0:
        #     res.append(repr(self.d))
        # return " + ".join(res) or "0"

    def __complex__(self):
        return self.a * OMEGA**3 + self.b * 1j + self.c * OMEGA + self.d

    def __float__(self):
        if self.b == 0 and self.a == -self.c:
            return self.real
        raise ValueError("Requested float for complex-valued DOmega instance:", self)

    @property
    def real(self):
        """The real component of a DOmega value."""
        return root_two(self.d, (self.c - self.a) / 2)

    @property
    def imag(self):
        """The imaginary component of a DOmega value."""
        return root_two(self.b, (self.a + self.c) / 2)

    def __mul__(self, other):
        # if isinstance(other, (Dyadic, int)):
        #     return DOmega(self.a * other, self.b * other, self.c * other, self.d * other)
        # if isinstance(other, RootTwo):
        #     other = self.from_root_two(other)
        if isinstance(other, DOmega):
            a, b, c, d = (self.a, self.b, self.c, self.d)
            _a, _b, _c, _d = (other.a, other.b, other.c, other.d)
            new_a = a * _d + b * _c + c * _b + d * _a
            new_b = b * _d + c * _c + d * _b - a * _a
            new_c = c * _d + d * _c - a * _b - b * _a
            new_d = d * _d - a * _c - b * _b - c * _a
            return DOmega(new_a, new_b, new_c, new_d)
        raise TypeError(
            f"cannot multiply DOmega value by value `{other}` of unknown type {type(other).__name__}"
        )

    def __add__(self, other):
        # if isinstance(other, (Dyadic, int)):
        #     return DOmega(self.a + other, self.b + other, self.c + other, self.d + other)
        if isinstance(other, DOmega):
            return DOmega(self.a + other.a, self.b + other.b, self.c + other.c, self.d + other.d)
        raise TypeError(
            f"cannot add DOmega value to value `{other}` of unknown type {type(other).__name__}"
        )

    def __sub__(self, other):
        return self + (-other)

    def __neg__(self):
        return DOmega(-self.a, -self.b, -self.c, -self.d)
