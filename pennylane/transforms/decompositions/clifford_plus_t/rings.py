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

from fractions import Fraction
import numpy as np

SQRT2 = float(np.sqrt(2))


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
            return Dyadic(self.x * other.x, self.k + other.k)
        if isinstance(other, int):
            return Dyadic(self.x * other, self.k)
        if isinstance(other, RootTwo):
            return other * self
        return float(self) * other

    def __add__(self, other):
        if isinstance(other, Dyadic):
            if self.k == other.k:
                return Dyadic(self.x + other.x, self.k)
            a, b = (self, other) if self.k < other.k else (other, self)
            k_delta = b.k - a.k
            return Dyadic(a.x * int(2**k_delta) + b.x, b.k)
        if isinstance(other, int):
            return Dyadic(self.x + other * 2**self.k, self.k)
        if isinstance(other, RootTwo):
            return other + self
        return float(self) + other

    def __sub__(self, other):
        if isinstance(other, Dyadic):
            return self + (-other)
        if isinstance(other, int):
            return Dyadic(self.x - other * 2**self.k, self.k)
        if isinstance(other, RootTwo):
            return -other + self
        return float(self) - other

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


class Matrix(np.ndarray):
    """Assumes input is a 2x2 matrix."""

    @classmethod
    def array(cls, value) -> "Matrix":
        """The ndarray constructor is messy, it's easier to provide this (like np.array)."""
        return np.array(value).view(cls)

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


class RootTwo:
    """Any object that can be represented as a + b√2"""

    def __new__(cls, a, b):
        types = {type(a), type(b)} - {int}
        if not types:
            return object.__new__(ZRootTwo)
        if types == {Dyadic}:
            return object.__new__(DRootTwo)
        if types == {Fraction}:
            return object.__new__(QRootTwo)
        return float.__new__(float, a + b * SQRT2)

    def __init__(self, a, b):
        """Default constructor."""
        dtype = self.dtype  # pylint:disable=no-member
        self.a = a if isinstance(a, dtype) else dtype(a)
        self.b = b if isinstance(b, dtype) else dtype(b)

    def conjugate(self):
        """The complex conjugate. Defaults to itself, needed for Matrix.adjoint()."""
        return self

    def adj2(self):
        """Return the root-2 adjoint."""
        return type(self)(self.a, -self.b)

    def __repr__(self):
        if self.a == 0:
            return f"{self.b}√2"
        if self.b == 0:
            return repr(self.a)
        if self.b > 0:
            return f"{self.a}+{self.b}√2"
        return f"{self.a}{self.b}√2"

    def __float__(self):
        """Convert from ring to float."""
        return float(self.a) + float(self.b) * SQRT2

    def __add__(self, other):
        if isinstance(other, RootTwo):
            return RootTwo(self.a + other.a, self.b + other.b)
        return RootTwo(self.a + other, self.b)

    def __sub__(self, other):
        if isinstance(other, RootTwo):
            return RootTwo(self.a - other.a, self.b - other.b)
        return RootTwo(self.a - other, self.b)

    def __mul__(self, other):
        if isinstance(other, RootTwo):
            return RootTwo(
                self.a * other.a + 2 * self.b * other.b,
                self.a * other.b + self.b * other.a,
            )
        return RootTwo(self.a * other, self.b * other)

    def __eq__(self, other):
        if isinstance(other, RootTwo):
            return self.a == other.a and self.b == other.b
        return np.isclose(float(self), float(other))

    def __gt__(self, other):
        return float(self) > float(other)

    def __lt__(self, other):
        return float(self) < float(other)

    def __neg__(self):
        return type(self)(-self.a, -self.b)

    def __pow__(self, power):
        if not isinstance(power, int):
            raise ValueError(f"Cannot raise RootTwo to non-int power {power}")
        if power == 0:
            return 1
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
    """Z√2"""

    dtype = int

    def __truediv__(self, other):
        if isinstance(other, int):
            return QRootTwo(Fraction(self.a, other), Fraction(self.b, other))
        return super().__truediv__(other)

    def __rtruediv__(self, other):
        if isinstance(other, int):
            k = self.a**2 - 2 * self.b**2
            return QRootTwo(Fraction(other * self.a, k), Fraction(-other * self.b, k))
        return super().__rtruediv__(other)


class DRootTwo(RootTwo):
    """D√2"""

    dtype = Dyadic


class QRootTwo(RootTwo):
    """Q√2"""

    dtype = Fraction
