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

import numpy as np

SQRT2 = float(np.sqrt(2))


class RootTwo:
    """Any object that can be represented as a + b√2"""

    def __init__(self, a, b):
        """Default constructor."""
        self.a = a
        self.b = b

    def __float__(self):
        """Convert from ring to float."""
        return float(self.a) + float(self.b) * SQRT2

    def adj2(self):
        """Return the root-2 adjoint."""
        return type(self)(self.a, -self.b)

    def __repr__(self):
        if self.b == 0:
            return repr(self.a)
        if self.b > 0:
            return f"{self.a}+{self.b}√2"
        return f"{self.a}{self.b}√2"

    def __add__(self, other):
        if isinstance(other, RootTwo):
            a = self.a + other.a
            b = self.b + other.b
            return a if b == 0 else RootTwo(a, b)
        return RootTwo(self.a + other, self.b)

    def __mul__(self, other):
        if isinstance(other, RootTwo):
            return RootTwo(
                self.a * other.a + 2 * self.b * other.b,
                self.a * other.b + self.b * other.a,
            )
        return RootTwo(self.a * other, self.b * other)

    __rmul__ = __mul__

    def __eq__(self, other):
        if isinstance(other, RootTwo):
            return self.a == other.a and self.b == other.b
        return np.isclose(float(self), float(other))

    def __neg__(self):
        return RootTwo(-self.a, -self.b)


class Dyadic:
    """Dyadic numbers, of the form x/(2**k) | x,k E Z"""

    def __init__(self, x, k):
        self.x = x
        self.k = k

    def __float__(self):
        return self.x / (2**self.k)

    def __mul__(self, other):
        if isinstance(other, Dyadic):
            return Dyadic(self.x * other.x, self.k * other.k)
        if isinstance(other, int):
            return Dyadic(self.x * other, self.k)
        return float(self) * other

    __rmul__ = __mul__

    def __add__(self, other):
        if isinstance(other, Dyadic):
            if self.k == other.k:
                return Dyadic(self.x + other.x, self.k)
            a, b = (self, other) if self.k < other.k else (other, self)
            k_delta = b.k - a.k
            return Dyadic(a.x * int(2**k_delta) + b.x, b.k)
        if isinstance(other, int):
            return Dyadic(self.x + other * 2**self.k, self.k)
        return other + float(self)

    def __eq__(self, other):
        if isinstance(other, Dyadic):
            return self.x == other.x and self.k == other.k
        return np.isclose(float(self), float(other))

    def __gt__(self, other):
        return float(self) >= float(other)

    def __repr__(self):
        denom = 2 if self.k == 1 else f"2^{self.k}"
        return f"({self.x}/{denom})"


class Matrix(list):
    """Assumes input is a 2x2 matrix."""

    def __matmul__(self, other):
        (a1, b1), (c1, d1) = self
        (a2, b2), (c2, d2) = other
        return Matrix(
            [
                [a1 * a2 + b1 * c2, a1 * b2 + b1 * d2],
                [c1 * a2 + d1 * c2, c1 * b2 + d1 * d2],
            ],
        )

    def inverse(self):
        """Return the inverse of the matrix, assuming |det M| = 1."""
        (a, b), (c, d) = self
        det = a * d - b * c
        if det == 1:
            return Matrix([[d, -b], [-c, a]])
        if det == -1:
            return Matrix([[-d, b], [c, -a]])
        raise ValueError("can only get special inverse.")

    def adjoint(self):
        """Returns the adjoint of the matrix.
        TODO: not implemented."""
        (a, b), (c, d) = self
        return Matrix([[a, c], [b, d]])
