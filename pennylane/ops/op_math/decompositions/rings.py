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
from functools import lru_cache
from typing import List

import numpy as np

_SQRT2 = math.sqrt(2)
_OMEGA = (1 + 1j) / _SQRT2


class ZSqrtTwo:
    r"""Represents the elements of the ring of integers adjoined with the square root of 2.

    ..math::
        \mathbb{Z}[\sqrt{2}] = \{ a + b\sqrt{2} \mid a, b \in \mathbb{Z} \}

    Args:
        a (int): The integer part of the element.
        b (int): The coefficient of sqrt(2) in the element.
    """

    def __init__(self, a: int = 0, b: int = 0) -> None:
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
    r"""Represents the elements of the ring of integers adjoined with :math:`\omega`, the eight root of unity.

    ..math::
        \mathbb{Z}[\omega] = \{ a\omega^3 + b\omega^2 + c\omega + d \mid a, b, c, d \in \mathbb{Z} \}

    where :math:`\omega = (1 + i) / \sqrt{2}` is the eighth root of unity.

    Args:
        a (int): Coefficient of :math:`\omega^3`.
        b (int): Coefficient of :math:`\omega^2`.
        c (int): Coefficient of :math:`\omega`.
        d (int): Constant term.
    """

    def __init__(self, a: int = 0, b: int = 0, c: int = 0, d: int = 0) -> None:
        self.a = int(a)
        self.b = int(b)
        self.c = int(c)
        self.d = int(d)

    def __str__(self: ZOmega) -> str:
        return f"{self.a} ω^3 + {self.b} ω^2 + {self.c} ω + {self.d}"

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

    def __truediv__(self, other: int | float) -> ZOmega:
        if isinstance(other, (int, float)):
            return ZOmega(self.a / other, self.b / other, self.c / other, self.d / other)
        if isinstance(other, ZOmega):
            pass
        raise TypeError(f"cannot divide Omega value by {other} of type {type(other).__name__}")

    def __mod__(self, other: ZOmega) -> ZOmega:
        pass

    def conj(self: ZOmega) -> ZOmega:
        """The complex conjugate."""
        return ZOmega(-self.c, -self.b, -self.a, self.d)

    def adj2(self: ZOmega) -> ZOmega:
        """Return the root-2 adjoint."""
        return ZOmega(-self.a, self.b, -self.c, self.d)

    def norm(self: ZOmega) -> float:
        """Return the norm squared."""
        return self * self.conj()

    def parity(self: ZOmega) -> int:
        """Return the parity."""
        return (self.a + self.c) % 2

    def to_sqrt_two(self: ZOmega) -> ZSqrtTwo:
        """
        Convert to the ring of integers adjoined with the square root of 2.

        Returns:
            ZSqrtTwo: The corresponding element in the ZSqrtTwo ring.
        """
        return ZSqrtTwo(self.a + self.b, self.c + self.d)

    def sqrt2scale(self) -> ZOmega:
        r"""Multiply the element by :math:`\sqrt{2}`"""


class SU2Matrix:
    r"""Represents the SU(2) matrices over the ring :math:`\mathbb{Z}[\omega]` (`~pennylane.ZOmega`).

    .. math::        
        \frac{1}{\sqrt{2}^k} 
        \begin{pmatrix}
        a & b \\
        c & d
        \end{pmatrix}

    where :math:`a, b, c, d \in \mathbb{Z}[\omega]`, :math:`k \in \mathbb{Z}`, and the determinant is `1`.

    Args:
        a (ZOmega): Element at position (0, 0) of the matrix.
        b (ZOmega): Element at position (0, 1) of the matrix.
        c (ZOmega): Element at position (1, 0) of the matrix.
        d (ZOmega): Element at position (1, 1) of the matrix.
        k (int): Optional integer to scale the matrix by a factor of :math:`1 / \sqrt{2}^k`.

    """

    # pylint:disable = too-many-positional-arguments, too-many-arguments
    def __init__(self, a: ZOmega, b: ZOmega, c: ZOmega, d: ZOmega, k: int = 0) -> None:
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.k = k

    def __str__(self: SU2Matrix) -> str:
        return f"[[{self.a}, {self.b}], [{self.c}, {self.d}]]"

    def __repr__(self: SU2Matrix) -> str:
        return f"SU2Matrix({self.a}, {self.b}, {self.c}, {self.d})"

    def __neg__(self: SU2Matrix) -> SU2Matrix:
        return SU2Matrix(-self.a, -self.b, -self.c, -self.d, self.k)

    def __eq__(self: SU2Matrix, other: SU2Matrix) -> bool:
        """Check if two SU(2) matrices are equal."""
        return (
            self.a == other.a
            and self.b == other.b
            and self.c == other.c
            and self.d == other.d
            and self.k == other.k
        )

    def __mul__(self: SU2Matrix, other: int) -> SU2Matrix:
        """Multiply the matrix by an integer."""
        if isinstance(other, float) and other.is_integer():
            other = int(other)

        return SU2Matrix(
            self.a * other,
            self.b * other,
            self.c * other,
            self.d * other,
            self.k,
        )

    def __matmul__(self: SU2Matrix, other: SU2Matrix) -> SU2Matrix:
        """Multiply two SU(2) matrices."""
        if not isinstance(other, SU2Matrix):
            raise TypeError(f"Cannot multiply SU2Matrix with {type(other).__name__}")

        return SU2Matrix(
            a=self.a * other.a + self.b * other.c,
            b=self.a * other.b + self.b * other.d,
            c=self.c * other.a + self.d * other.c,
            d=self.c * other.b + self.d * other.d,
            k=self.k + other.k,
        )

    @property
    def toarray(self: SU2Matrix) -> np.ndarray:
        """Convert the matrix to a NumPy array."""
        return _SQRT2**self.k * np.array(
            [
                [complex(self.a), complex(self.b)],
                [complex(self.c), complex(self.d)],
            ],
            dtype=np.complex128,
        )

    @property
    def flatten(self: SU2Matrix) -> List[ZOmega]:
        """Flatten the matrix to a 1D NumPy array."""
        return [self.a, self.b, self.c, self.d]

    def conj(self: SU2Matrix) -> SU2Matrix:
        """Return the conjugate of the matrix."""
        return SU2Matrix(
            self.a.conj(),
            self.b.conj(),
            self.c.conj(),
            self.d.conj(),
            self.k,
        )

    def adj2(self: SU2Matrix) -> SU2Matrix:
        """Return the root-2 adjoint of the matrix."""
        return SU2Matrix(
            self.a.adj2(),
            self.b.adj2(),
            self.c.adj2(),
            self.d.adj2(),
            self.k,
        )

    def mult2k(self: SU2Matrix, k: int) -> SU2Matrix:
        """Multiply the matrix by :math:`2^k`, i.e., an integer power of 2."""
        if k == 0:
            return self
        (
            e_scale,
            k_scale,
        ) = int(
            math.pow(2, k)
        ), int(2 * k)
        return SU2Matrix(
            self.a * e_scale,
            self.b * e_scale,
            self.c * e_scale,
            self.d * e_scale,
            self.k + k_scale,
        )


class SO3Matrix:
    r"""Represents the SO(3) matrices over the ring :math:`\mathbb{Z}[\sqrt{2}]` (`~pennylane.ZSqrtTwo`)."""

    def __init__(self, su2mat: SU2Matrix, k: int = 0) -> None:
        """Initialize the SO(3) matrix with a SU(2) matrix and an integer k."""
        self.su2mat = su2mat
        self.k = k
        self.so3mat = self.from_su2(su2mat)

    def from_su2(self, su2mat) -> List[List[ZSqrtTwo]]:
        """Return the SO(3) matrix as a list of lists."""
        su2_elems = su2mat.flatten()

        k = 2 * su2mat.k
        if any(s.parity for s in su2_elems):
            z_sqrt2 = [(ZSqrtTwo((s.c - s.a), s.d), ZSqrtTwo((s.c + s.a), s.b)) for s in su2_elems]
            k += 2
        else:
            z_sqrt2 = [
                (ZSqrtTwo(s.d, (s.c - s.a) // 2), ZSqrtTwo(s.b, (s.c + s.a) // 2))
                for s in su2_elems
            ]

        a_, b_, c_, d_ = zip(*z_sqrt2)
        so3_mat = [
            [
                a_[0] * d_[0] + a_[1] * d_[1] + b_[0] * c_[0] + b_[1] * c_[1],
                a_[1] * d_[0] + b_[0] * c_[1] - b_[1] * c_[0] - a_[0] * d_[1],
                a_[0] * c_[0] + a_[1] * c_[1] - b_[0] * d_[0] - b_[1] * d_[1],
                a_[0] * d_[1] - a_[1] * d_[0] + b_[0] * c_[1] - b_[1] * c_[0],
                a_[0] * d_[0] + a_[1] * d_[1] - b_[0] * c_[0] - b_[1] * c_[1],
                a_[0] * d_[1] - a_[1] * d_[0] + b_[0] * c_[1] - b_[1] * c_[0],
                2 * (a_[0] * b_[0] + a_[1] * b_[1]),
                2 * (a_[1] * b_[0] - a_[0] * b_[1]),
                a_[0] ** 2 + a_[1] ** 2 - b_[0] ** 2 - b_[1] ** 2,
            ]
        ]
        return so3_mat

    def reducek(self: SO3Matrix) -> SO3Matrix:
        """Reduce the k value of the SO(3) matrix."""


@lru_cache
def _clifford_gates_to_SU2():
    """Return a dictionary mapping Clifford gates to their corresponding SU(2) matrices."""
    return {
        "I": SU2Matrix(ZOmega(d=1), ZOmega(), ZOmega(), ZOmega(d=1)),
        "X": SU2Matrix(ZOmega(), ZOmega(d=1), ZOmega(d=1), ZOmega()),
        "Y": SU2Matrix(ZOmega(), ZOmega(b=-1), ZOmega(b=1), ZOmega()),
        "Z": SU2Matrix(ZOmega(d=1), ZOmega(), ZOmega(), ZOmega(d=-1)),
        "H": SU2Matrix(ZOmega(d=1), ZOmega(d=1), ZOmega(d=1), ZOmega(d=-1), k=1),
        "S": SU2Matrix(ZOmega(d=1), ZOmega(), ZOmega(), ZOmega(b=1)),
        "T": SU2Matrix(ZOmega(d=1), ZOmega(), ZOmega(), ZOmega(c=1)),
        "S*": SU2Matrix(ZOmega(d=1), ZOmega(), ZOmega(), ZOmega(b=-1)),
        "T*": SU2Matrix(ZOmega(d=1), ZOmega(), ZOmega(), ZOmega(a=-1)),
    }


@lru_cache
def _clifford_gates_to_SO3():
    """Return a dictionary mapping Clifford gates to their corresponding SO(3) matrices."""
    su2_matrices = _clifford_gates_to_SU2()
    return {gate: SO3Matrix(su2, k=su2.k) for gate, su2 in su2_matrices.items()}
