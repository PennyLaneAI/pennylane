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
"""This module provides classes for algebraic prerequisites for Clifford+T decompositions"""


from __future__ import annotations

import math

import numpy as np

_SQRT2 = math.sqrt(2)
_OMEGA = (1 + 1j) / _SQRT2


class ZSqrtTwo:
    r"""
    A class representing the ring of integers adjoined with the square root of 2.

    ..math::
        \mathbb{Z}[\sqrt{2}] = \{ a + b\sqrt{2} \mid a, b \in \mathbb{Z} \}

    """

    def __init__(self, a: int, b: int):
        """
        Initialize the ZSqrtTwo object with integers a and b.

        Args:
            a (int): The integer part of the element.
            b (int): The coefficient of sqrt(2) in the element.
        """
        self.a = int(a)
        self.b = int(b)

    def __str__(self: ZSqrtTwo) -> str:
        return f"{self.a} + {self.b}√2"

    def __repr__(self: ZSqrtTwo) -> str:
        return f"ZSqrtTwo({self.a}, {self.b})"

    def __float__(self: ZSqrtTwo) -> float:
        return float(self.a) + float(self.b) * _SQRT2

    def __add__(self, other: ZSqrtTwo | int | float) -> ZSqrtTwo:
        if isinstance(other, ZSqrtTwo):
            return ZSqrtTwo(self.a + other.a, self.b + other.b)
        if isinstance(other, (int, float)):
            return ZSqrtTwo(self.a + other, self.b)
        raise TypeError(f"Unsupported type {type(other)} for addition with ZSqrtTwo")

    def __sub__(self, other: ZSqrtTwo | int | float) -> ZSqrtTwo:
        return self + (-other)

    def __mul__(self, other: ZSqrtTwo | int | float) -> ZSqrtTwo:
        if isinstance(other, ZSqrtTwo):
            return ZSqrtTwo(
                self.a * other.a + 2 * self.b * other.b,
                self.a * other.b + self.b * other.a,
            )
        if isinstance(other, (int, float)):
            return ZSqrtTwo(self.a * other, self.b * other)
        raise TypeError(f"Unsupported type {type(other)} for multiplication with ZSqrtTwo")

    __radd__ = __add__
    __rsub__ = __sub__
    __rmul__ = __mul__

    def __pow__(self, power: int) -> ZSqrtTwo:
        if power == 0:
            return ZSqrtTwo(1, 0)
        if power < 0:
            return 1 / (self**-power)
        result = self
        while power > 1:
            result *= self
            power -= 1
        return result

    def __truediv__(self, other: ZSqrtTwo) -> ZSqrtTwo:
        if isinstance(other, ZSqrtTwo):
            return self * (1 / other)
        raise TypeError(f"cannot divide RootTwo ring by {other} of type {type(other).__name__}")

    def __rtruediv__(self, other: ZSqrtTwo) -> ZSqrtTwo:
        return other / float(self)

    def __eq__(self, other) -> bool:
        if isinstance(other, ZSqrtTwo):
            return self.a == other.a and self.b == other.b
        if isinstance(other, (int, float)):
            return self.a == other and self.b == 0
        return np.isclose(float(self), float(other))

    def __abs__(self) -> float:
        return self.a**2 - 2 * self.b**2

    def __neg__(self) -> ZSqrtTwo:
        return ZSqrtTwo(-self.a, -self.b)

    def conj(self) -> ZSqrtTwo:
        """Return the standard conjugate."""
        return ZSqrtTwo(self.a, self.b)

    def adj2(self) -> ZSqrtTwo:
        """Return the root-2 adjoint."""
        return ZSqrtTwo(self.a, -self.b)

    def sqrt(self) -> ZSqrtTwo | None:
        """Return the square root."""
        r = math.isqrt(max((d := abs(self)), 0))
        if r * r != d:
            return None

        xs = (math.isqrt((self.a + r) // 2), math.isqrt((self.a - r) // 2))
        ys = (math.isqrt((self.a - r) // 4), math.isqrt((self.a + r) // 4))
        for x, y in zip(xs, ys):
            zrt = ZSqrtTwo(x, y)
            if zrt * zrt == self:
                return zrt
            art = zrt.adj2()
            if art * art == self:
                return art
        return None

    def to_omega(self) -> ZOmega:
        """Convert to the an ring of integers adjoined with omega."""
        return ZOmega(-self.b, 0, self.b, self.a)


class ZOmega:
    r"""
    A class representing the ring of integers adjoined with the eight root of unity.

    ..math::
        \mathbb{Z}[\omega] = \{ a\omega^3 + b\omega^2 + c\omega + d \mid a, b, c, d \in \mathbb{Z} \}

    where :math:`\omega = (1 + i) / \sqrt{2}` is the eighth root of unity.

    Args:
        a (int): Coefficient of :math:`\omega^3`.
        b (int): Coefficient of :math:`\omega^2`.
        c (int): Coefficient of :math:`\omega`.
        d (int): Constant term.
    """

    def __init__(self, a: int, b: int, c: int, d: int):
        self.a = int(a)
        self.b = int(b)
        self.c = int(c)
        self.d = int(d)

    def __str__(self: ZOmega) -> str:
        return f"{self.a}ω^3 + {self.b}ω^2 + {self.c}ω + {self.d}"

    def __repr__(self: ZOmega) -> str:
        return f"ZOmega({self.a}, {self.b}, {self.c}, {self.d})"

    def __complex__(self) -> complex:
        return complex(self.a * _OMEGA**3 + self.b * _OMEGA**2 + self.c * _OMEGA + self.d)

    def __abs__(self: ZOmega) -> float:
        a, b, c, d = (self.a, self.b, self.c, self.d)
        return (a**2 + b**2 + c**2 + d**2) ** 2 - 2 * (a * b + b * c + c * d - d * a) ** 2

    def __mul__(self, other: ZOmega | int | float) -> ZOmega:
        if isinstance(other, ZOmega):
            a, b, c, d = (self.a, self.b, self.c, self.d)
            _a, _b, _c, _d = (other.a, other.b, other.c, other.d)
            new_a = a * _d + b * _c + c * _b + d * _a
            new_b = b * _d + c * _c + d * _b - a * _a
            new_c = c * _d + d * _c - a * _b - b * _a
            new_d = d * _d - a * _c - b * _b - c * _a
            return ZOmega(new_a, new_b, new_c, new_d)
        if isinstance(other, (int, float)):
            return ZOmega(self.a * other, self.b * other, self.c * other, self.d * other)
        raise TypeError(
            f"cannot multiply Omega value by value `{other}` of unknown type {type(other).__name__}"
        )

    def __add__(self, other: ZOmega | int | float) -> ZOmega:
        if isinstance(other, ZOmega):
            return ZOmega(self.a + other.a, self.b + other.b, self.c + other.c, self.d + other.d)
        if isinstance(other, (int, float)):
            return ZOmega(self.a, self.b, self.c, self.d + other)
        raise TypeError(
            f"cannot add Omega value to value `{other}` of unknown type {type(other).__name__}"
        )

    def __sub__(self, other):
        return self + (-other)

    __radd__ = __add__
    __rsub__ = __sub__
    __rmul__ = __mul__

    def __neg__(self: ZOmega) -> ZOmega:
        return ZOmega(-self.a, -self.b, -self.c, -self.d)

    def __eq__(self, other: ZOmega) -> bool:
        if isinstance(other, ZOmega):
            return (
                self.a == other.a and self.b == other.b and self.c == other.c and self.d == other.d
            )
        raise TypeError(f"cannot compare Omega ring with {other} of type {type(other).__name__}")

    def __pow__(self, power: int) -> ZOmega:
        if power < 0:
            raise ValueError("cannot raise Omega to negative power")
        if power == 0:
            return ZOmega(0, 0, 0, 1)
        result = self
        while power > 1:
            result *= self
            power -= 1
        return result

    def conj(self: ZOmega) -> ZOmega:
        """The complex conjugate."""
        return ZOmega(-self.c, -self.b, -self.a, self.d)

    def adj2(self: ZOmega) -> ZOmega:
        """Return the root-2 adjoint."""
        return ZOmega(-self.a, self.b, -self.c, self.d)

    def norm(self: ZOmega) -> float:
        """Return the norm squared."""
        return self * self.conj()

    def to_root_two(self: ZOmega) -> ZSqrtTwo:
        """
        Convert to the ring of integers adjoined with the square root of 2.

        Returns:
            ZSqrtTwo: The corresponding element in the ZSqrtTwo ring.
        """
        return ZSqrtTwo(self.a + self.b, self.c + self.d)

    def __truediv__(self, other: int | float) -> ZOmega:
        if isinstance(other, (int, float)):
            return ZOmega(self.a / other, self.b / other, self.c / other, self.d / other)
        if isinstance(other, ZOmega):
            pass
        raise TypeError(f"cannot divide Omega value by {other} of type {type(other).__name__}")

    def __mod__(self, other: ZOmega) -> ZOmega:
        pass
