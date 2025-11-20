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
"""This module provides classes for algebraic prerequisites for Clifford+T decompositions"""


from __future__ import annotations

import math
from copy import deepcopy

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
        if self.b == 0:
            return str(self.a)
        if self.a == 0:
            return f"{self.b}√2"
        return f"{self.a} + {self.b}√2"

    def __repr__(self: ZSqrtTwo) -> str:
        return f"ZSqrtTwo(a={self.a}, b={self.b})"

    def __float__(self: ZSqrtTwo) -> float:
        return float(self.a) + float(self.b) * _SQRT2

    def __add__(self, other: ZSqrtTwo | int | float) -> ZSqrtTwo:
        if isinstance(other, ZSqrtTwo):
            return ZSqrtTwo(self.a + other.a, self.b + other.b)
        if isinstance(other, int) or (isinstance(other, float) and other.is_integer()):
            return ZSqrtTwo(self.a + int(other), self.b)
        raise TypeError(f"Unsupported type {type(other)} for addition with ZSqrtTwo")

    def __sub__(self, other: ZSqrtTwo | int | float) -> ZSqrtTwo:
        return self + (-other)

    def __mul__(self, other: ZSqrtTwo | int | float) -> ZSqrtTwo:
        if isinstance(other, ZSqrtTwo):
            return ZSqrtTwo(
                self.a * other.a + 2 * self.b * other.b,
                self.a * other.b + self.b * other.a,
            )
        if isinstance(other, int) or (isinstance(other, float) and other.is_integer()):
            return ZSqrtTwo(self.a * int(other), self.b * int(other))
        raise TypeError(f"Unsupported type {type(other)} for multiplication with ZSqrtTwo")

    def __rsub__(self, other: ZSqrtTwo | int | float) -> ZSqrtTwo:
        return other + (-self)

    __radd__ = __add__
    __rmul__ = __mul__

    def __pow__(self, power: int) -> ZSqrtTwo:
        if power == 0:
            return ZSqrtTwo(1, 0)
        if power < 0:
            raise ValueError(f"Negative powers {power} are unsupported for ZSqrtTwo.")
        if isinstance(power, float) and not power.is_integer():
            raise ValueError(f"Non-integer powers {power} are unsupported for ZSqrtTwo.")
        result = self
        while power > 1:
            result *= self
            power -= 1
        return result

    def __truediv__(self, other: ZSqrtTwo) -> ZSqrtTwo:
        if isinstance(other, ZSqrtTwo):
            return (self * other.adj2()) / abs(other)
        if isinstance(other, int) or (isinstance(other, float) and other.is_integer()):
            other = int(other)
            if self.a % other == 0 and self.b % other == 0:
                return ZSqrtTwo(self.a // other, self.b // other)
        raise TypeError(f"Unsupported type {type(other)} for dividing ZSqrtTwo")

    def __floordiv__(self, other: int) -> ZSqrtTwo:
        if isinstance(other, int):
            return ZSqrtTwo(self.a // other, self.b // other)
        raise TypeError(f"Unsupported type {type(other)} for floor dividing ZSqrtTwo")

    def __eq__(self, other) -> bool:
        if isinstance(other, ZSqrtTwo):
            return self.a == other.a and self.b == other.b
        if isinstance(other, int) or (isinstance(other, float) and other.is_integer()):
            return self.a == int(other) and self.b == 0
        return np.isclose(float(self), float(other))

    def __abs__(self) -> float:
        return self.a**2 - 2 * self.b**2

    def __neg__(self) -> ZSqrtTwo:
        return ZSqrtTwo(-self.a, -self.b)

    def __mod__(self, other: ZSqrtTwo | int | float) -> ZSqrtTwo:
        if isinstance(other, int) or (isinstance(other, float) and other.is_integer()):
            return ZSqrtTwo(self.a % int(other), self.b % int(other))

        d = abs(other)
        n1, n2 = (self.a * other.a - 2 * self.b * other.b), (self.b * other.a - self.a * other.b)
        return self - ZSqrtTwo(round(n1 / d), round(n2 / d)) * other

    @property
    def flatten(self: ZSqrtTwo) -> list[int]:
        """Flatten to a list."""
        return [self.a, self.b]

    def conj(self) -> ZSqrtTwo:
        r"""Return the complex conjugate.

        .. math::
            (a + b\sqrt{2})^{\dagger} = a + b\sqrt{2}
        """
        return ZSqrtTwo(self.a, self.b)

    def adj2(self) -> ZSqrtTwo:
        r"""Return the adjoint, i.e., the root-2 conjugate.

        .. math::
            (a + b\sqrt{2})^{\bullet} = a - b\sqrt{2}
        """
        return ZSqrtTwo(self.a, -self.b)

    def sqrt(self) -> ZSqrtTwo | None:
        """Return the square root."""
        r = math.isqrt(max((d := abs(self)), 0))
        if r * r != d:
            return None

        xs = (math.isqrt((self.a + r) // 2), math.isqrt((self.a - r) // 2))
        ys = (math.isqrt((self.a - r) // 4), math.isqrt((self.a + r) // 4))

        res = None
        for x, y in zip(xs, ys):
            zrt = ZSqrtTwo(x, y)
            if zrt * zrt == self:
                res = zrt
                break
            art = zrt.adj2()
            if art * art == self:  # pragma: no cover
                res = art
                break

        return res

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
        terms = []
        if self.a:
            terms.append(f"{self.a} ω^3")
        if self.b:
            terms.append(f"{self.b} ω^2")
        if self.c:
            terms.append(f"{self.c} ω")
        if self.d:
            terms.append(f"{self.d}")
        return " + ".join(terms) if terms else "0"

    def __repr__(self: ZOmega) -> str:
        return f"ZOmega(a={self.a}, b={self.b}, c={self.c}, d={self.d})"

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
        if isinstance(other, int) or (isinstance(other, float) and other.is_integer()):
            other = int(other)
            return ZOmega(self.a * other, self.b * other, self.c * other, self.d * other)
        raise TypeError(f"Unsupported type {type(other)} for multiplication with ZOmega")

    def __add__(self, other: ZOmega | int | float) -> ZOmega:
        if isinstance(other, ZOmega):
            return ZOmega(self.a + other.a, self.b + other.b, self.c + other.c, self.d + other.d)
        if isinstance(other, int) or (isinstance(other, float) and other.is_integer()):
            return ZOmega(self.a, self.b, self.c, self.d + int(other))
        raise TypeError(f"Unsupported type {type(other)} for addition with ZOmega")

    def __sub__(self, other: ZOmega | int | float) -> ZOmega:
        return self + (-other)

    def __rsub__(self, other: ZOmega | int | float) -> ZOmega:
        return other + (-self)

    __radd__ = __add__
    __rmul__ = __mul__

    def __neg__(self: ZOmega) -> ZOmega:
        return ZOmega(-self.a, -self.b, -self.c, -self.d)

    def __eq__(self, other: ZOmega) -> bool:
        if isinstance(other, ZOmega):
            return (
                self.a == other.a and self.b == other.b and self.c == other.c and self.d == other.d
            )
        if isinstance(other, int):
            return self.a == 0 and self.b == 0 and self.c == 0 and self.d == other
        return np.isclose(complex(self), complex(other))

    def __pow__(self, power: int) -> ZOmega:
        if power < 0:
            raise ValueError(f"Negative powers {power} are unsupported for ZOmega.")
        if isinstance(power, float) and not power.is_integer():
            raise ValueError(f"Non-integer powers {power} are unsupported for ZOmega.")
        if power == 0:
            return ZOmega(0, 0, 0, 1)
        result = self
        while power > 1:
            result *= self
            power -= 1
        return result

    def __truediv__(self, other: int | float) -> ZOmega:
        if isinstance(other, int) or (isinstance(other, float) and other.is_integer()):
            other = int(other)
            if all(a % other == 0 for a in self.flatten):
                return ZOmega(self.a / other, self.b / other, self.c / other, self.d / other)
        raise TypeError(f"Unsupported type {type(other)} for division with ZOmega")

    def __floordiv__(self, other: int) -> ZOmega:
        if isinstance(other, int) or (isinstance(other, float) and other.is_integer()):
            other = int(other)
            return ZOmega(self.a // other, self.b // other, self.c // other, self.d // other)
        raise TypeError(f"Unsupported type {type(other)} for floor division with ZOmega")

    def __mod__(self, other: ZOmega) -> ZOmega:
        d = abs(other)
        n = self * other.conj() * ((other * other.conj()).adj2())
        return ZOmega(*[(s + d // 2) // d for s in n.flatten]) * other - self

    @classmethod
    def from_sqrt_pair(cls, alpha: ZSqrtTwo, beta: ZSqrtTwo, shift: ZOmega) -> ZOmega:
        """Return ``ZOmega`` element as :math:`A + 1j * B + shift`, where :math:`A` and
        :math:`B` are ``ZSqrtTwo`` elements and ``shift`` is ``ZOmega`` element."""
        return cls(beta.b - alpha.b, beta.a, beta.b + alpha.b, alpha.a) + shift

    @property
    def flatten(self: ZOmega) -> list[int]:
        """Flatten to a list."""
        return [self.a, self.b, self.c, self.d]

    def conj(self: ZOmega) -> ZOmega:
        r"""Return the complex conjugate.

        .. math::
            (a\omega^3 + b\omega^2 + c\omega + d)^{\dagger} = -c\omega^3 + b\omega^2 - a\omega + d
        """
        return ZOmega(-self.c, -self.b, -self.a, self.d)

    def adj2(self: ZOmega) -> ZOmega:
        r"""Return the adjoint, i.e., the root-2 conjugate.

        .. math::
            (a\omega^3 + b\omega^2 + c\omega + d)^{\bullet} = -a\omega^3 + b\omega^2 - c\omega + d
        """
        return ZOmega(-self.a, self.b, -self.c, self.d)

    def norm(self: ZOmega) -> float:
        """Return the norm squared."""
        return self * self.conj()

    def parity(self: ZOmega) -> int:
        """Return the parity indicating structure of real and imaginary parts as a DyadicMatrix element."""
        return (self.a + self.c) % 2

    def to_sqrt_two(self: ZOmega) -> ZSqrtTwo:
        """Convert to the ring of integers adjoined with the square root of 2.

        Returns:
            ZSqrtTwo: The corresponding element in the ZSqrtTwo ring.
        """
        if (self.c + self.a) == 0 and self.b == 0:
            return ZSqrtTwo(self.d, (self.c - self.a) // 2)
        raise ValueError("Cannot convert ZOmega to ZSqrtTwo.")

    def normalize(self: ZOmega) -> tuple[ZOmega, int]:
        """Normalize the ZOmega element and return the number of times 2 was factored out."""
        res, ix = self, 0
        while (res.a + res.c) % 2 == 0 and (res.b + res.d) % 2 == 0:
            a = (res.b - res.d) // 2
            b = (res.a + res.c) // 2
            c = (res.b + res.d) // 2
            d = (res.c - res.a) // 2
            res = ZOmega(a, b, c, d)
            ix += 1
        return res, ix


class DyadicMatrix:
    r"""Represents the matrices over the ring :math:`\mathbb{D}[\omega]`,
    the ring of dyadic fractions adjoined with :math:`\omega`.

    The dyadic fractions :math:`\mathbb{D} = \mathbb{Z}[\frac{1}{2}]` are defined as
    :math:`\mathbb{D} = \{ a / 2^k \mid a \in \mathbb{Z}, k \in  \{0\} \cup \mathbb{N}\}`. This gives:

    .. math::
        \mathbb{D}[omega] = \mathbb{Z}[\frac{1}{\sqrt{2}}, i] = \{ a\omega^3 + b\omega^2 + c\omega + d \mid a, b, c, d \in \mathbb{Z}[\frac{1}{\sqrt{2}}] \}

    The `~pennylane.ZOmega` (or :math:`\mathbb{Z}[\omega]) represents a subset of :math:`\mathbb{D}[\omega]`,
    and therefore can be used to construct the elements of a ``DyadicMatrix``, which is represented as:

    .. math::
        \frac{1}{\sqrt{2}^k}
        \begin{pmatrix}
        a_{00} & a_{01} \\
        a_{10} & a_{11}
        \end{pmatrix},

    where :math:`a_{ij} \in \mathbb{D}[\omega]` and :math:`k \in \mathbb{Z}`.

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
        self.normalize()

    def __str__(self: DyadicMatrix) -> str:
        return f"[[{self.a}, {self.b}], [{self.c}, {self.d}]]"

    def __repr__(self: DyadicMatrix) -> str:
        return f"DyadicMatrix(a={self.a}, b={self.b}, c={self.c}, d={self.d}, k={self.k})"

    def __neg__(self: DyadicMatrix) -> DyadicMatrix:
        return DyadicMatrix(-self.a, -self.b, -self.c, -self.d, self.k)

    def __eq__(self: DyadicMatrix, other: DyadicMatrix) -> bool:
        """Check if two dyadic matrices are equal."""
        return (
            self.a == other.a
            and self.b == other.b
            and self.c == other.c
            and self.d == other.d
            and self.k == other.k
        )

    def __mul__(self: DyadicMatrix, other: int | ZOmega) -> DyadicMatrix:
        """Multiply the matrix by an integer."""
        if isinstance(other, float) and other.is_integer():
            other = int(other)

        return DyadicMatrix(
            self.a * other,
            self.b * other,
            self.c * other,
            self.d * other,
            self.k,
        )

    def __add__(self: DyadicMatrix, other: int | float | complex | DyadicMatrix) -> DyadicMatrix:
        """Add two dyadic matrices."""
        if isinstance(other, int) or (isinstance(other, float) and other.is_integer()):
            other = DyadicMatrix(ZOmega(d=int(other)), ZOmega(), ZOmega(), ZOmega(d=int(other)))
        if isinstance(other, complex) and other.real.is_integer() and other.imag.is_integer():
            z_omega = ZOmega(b=other.imag, d=other.real)
            other = DyadicMatrix(z_omega, ZOmega(), ZOmega(), z_omega)
        if not isinstance(other, DyadicMatrix):
            raise TypeError(f"Unsupported type {type(other)} for addition with DyadicMatrix")
        # Ensure k is the maximum of both matrices
        A, B = (self, other) if self.k >= other.k else (other, self)
        k_scale, k_parity = int(math.pow(2, (A.k - B.k) // 2)), (A.k - B.k) % 2
        b_elems = []
        for b_elem in B.flatten:
            a, b, c, d = (s * k_scale for s in b_elem.flatten)
            if k_parity != 0:  # sqrt(2) factor
                a, b, c, d = [(b - d), (c + a), (b + d), (c - a)]
            b_elems.append(ZOmega(a, b, c, d))

        return DyadicMatrix(
            a=A.a + b_elems[0], b=A.b + b_elems[1], c=A.c + b_elems[2], d=A.d + b_elems[3], k=A.k
        )

    def __matmul__(self: DyadicMatrix, other: DyadicMatrix) -> DyadicMatrix:
        """Multiply two dyadic matrices."""
        if not isinstance(other, DyadicMatrix):
            raise TypeError(
                f"Unsupported type {type(other)} for matrix multiplication with DyadicMatrix"
            )
        return DyadicMatrix(
            a=self.a * other.a + self.b * other.c,
            b=self.a * other.b + self.b * other.d,
            c=self.c * other.a + self.d * other.c,
            d=self.c * other.b + self.d * other.d,
            k=self.k + other.k,
        )

    @property
    def ndarray(self: DyadicMatrix) -> np.ndarray:
        """Convert the matrix to a NumPy array."""
        return (_SQRT2**-self.k) * np.array(
            [
                [complex(self.a), complex(self.b)],
                [complex(self.c), complex(self.d)],
            ],
            dtype=np.complex128,
        )

    @property
    def flatten(self: DyadicMatrix) -> list[ZOmega]:
        """Flatten the matrix elements to a list."""
        return [self.a, self.b, self.c, self.d]

    def conj(self: DyadicMatrix) -> DyadicMatrix:
        """Return the conjugate of the matrix."""
        return DyadicMatrix(
            self.a.conj(),
            self.b.conj(),
            self.c.conj(),
            self.d.conj(),
            self.k,
        )

    def adj2(self: DyadicMatrix) -> DyadicMatrix:
        """Return the root-2 adjoint of the matrix."""
        return DyadicMatrix(
            self.a.adj2(),
            self.b.adj2(),
            self.c.adj2(),
            self.d.adj2(),
            self.k,
        )

    def mult2k(self: DyadicMatrix, k: int) -> DyadicMatrix:
        """Multiply the matrix by :math:`2^k`, i.e., an integer power of 2."""
        if k == 0:
            return self
        k_val = min(0, self.k - 2 * k)
        k_scale = abs(k_val % 2)
        e_scale = int(math.pow(2, (k_scale - k_val) // 2))
        return DyadicMatrix(
            self.a * e_scale,
            self.b * e_scale,
            self.c * e_scale,
            self.d * e_scale,
            self.k + k_scale,
        )

    def normalize(self: DyadicMatrix) -> None:
        """Reduce the k value of the dyadic matrix.

        Example:
            >>> A = DyadicMatrix(ZOmega(d=2), ZOmega(d=2), ZOmega(d=2), ZOmega(d=2), k = 4)
            >>> A.normalize()
            >>> A
            DyadicMatrix(a=1, b=1, c=1, d=1, k=2)
        """
        # a, b, c, d = self.flatten
        # Factoring 2: Derived using (1 - w^4) [a', b', c', d'] => [a, b, c, d]
        if all(ZOmega() == s for s in self.flatten):
            self.k = 0
            return

        while all(np.allclose([_s % 2 for _s in s.flatten], 0) for s in self.flatten):
            self.k -= 2
            self.a, self.b, self.c, self.d = (s // 2 for s in self.flatten)

        # Factoring sqrt(2): Derived using (w - w^3) [a', b', c', d'] => [a, b, c, d]
        sqrt2_flag = True
        while sqrt2_flag and self.k > 0:
            for s in self.flatten:
                if (s.a + s.c) % 2 or (s.b + s.d) % 2:
                    sqrt2_flag = False
                    break
            if sqrt2_flag:
                self.k -= 1
                elements = self.flatten
                for i in range(4):
                    a, b, c, d = elements[i].flatten
                    elements[i] = ZOmega((b - d) // 2, (c + a) // 2, (b + d) // 2, (c - a) // 2)
                self.a, self.b, self.c, self.d = elements


class SO3Matrix:
    r"""Represents the :math:`SO(3)` matrices over the ring :math:`\mathbb{D}[\sqrt{2}]`,
    the ring of dyadic integers adjoined with :math:`\sqrt{2}`.

    The `~pennylane.ZSqrtTwo` (or :math:`\mathbb{Z}[\sqrt{2}]) represents a subset of this ring,
    and can be used to construct its elements. The matrix form is usually represented as:

    .. math::

        \frac{1}{\sqrt{2}^k}
        \begin{pmatrix}
        a_{00} & a_{01} & a_{11} \\
        a_{10} & a_{11} & a_{12} \\
        a_{20} & a_{21} & a_{22}
        \end{pmatrix},

    where :math:`a_{ij} \in \mathbb{Z}[\sqrt{2}]` and :math:`k \in \mathbb{Z}`.

    Args:
        matrix (DyadicMatrix): The :class:`~pennylane.DyadicMatrix` matrix from which the :math:`SO(3)` matrix is derived.
    """

    def __init__(self, matrix: DyadicMatrix) -> None:
        """Initialize the SO(3) matrix with a dyadic matrix and an integer k."""
        self.matrix = matrix
        self.k = 0
        self.so3mat = self.from_matrix(matrix)
        self.normalize()

    def __str__(self: SO3Matrix) -> str:
        """Return a string representation of the SO(3) matrix."""
        elements = self.flatten
        str_repr = "["
        for i in range(3):
            str_repr += f"[{elements[i * 3]}, {elements[i * 3 + 1]}, {elements[i * 3 + 2]}], \n"
        str_repr = str_repr.rstrip(", \n") + "]" + (f" * 1 / √2^{self.k}" if self.k else "")
        return str_repr

    def __repr__(self: SO3Matrix) -> str:
        """Return a string representation of the SO(3) matrix."""
        return f"SO3Matrix(matrix={self.matrix}, k={self.k})"

    def __matmul__(self, other: SO3Matrix) -> SO3Matrix:
        res = deepcopy(self)
        res.matrix = self.matrix @ other.matrix
        res.k = self.k + other.k

        us_self, us_other = self.flatten, other.flatten
        res.so3mat = [
            [
                us_self[0] * us_other[0] + us_self[1] * us_other[3] + us_self[2] * us_other[6],
                us_self[0] * us_other[1] + us_self[1] * us_other[4] + us_self[2] * us_other[7],
                us_self[0] * us_other[2] + us_self[1] * us_other[5] + us_self[2] * us_other[8],
            ],
            [
                us_self[3] * us_other[0] + us_self[4] * us_other[3] + us_self[5] * us_other[6],
                us_self[3] * us_other[1] + us_self[4] * us_other[4] + us_self[5] * us_other[7],
                us_self[3] * us_other[2] + us_self[4] * us_other[5] + us_self[5] * us_other[8],
            ],
            [
                us_self[6] * us_other[0] + us_self[7] * us_other[3] + us_self[8] * us_other[6],
                us_self[6] * us_other[1] + us_self[7] * us_other[4] + us_self[8] * us_other[7],
                us_self[6] * us_other[2] + us_self[7] * us_other[5] + us_self[8] * us_other[8],
            ],
        ]
        res.normalize()
        return res

    def __eq__(self: SO3Matrix, other: SO3Matrix) -> bool:
        return self.k == other.k and all(x == y for (x, y) in zip(self.flatten, other.flatten))

    @property
    def flatten(self: SO3Matrix) -> list[ZOmega]:
        """Flatten the matrix to a 1D NumPy array."""
        return [l for row in self.so3mat for l in row]

    @property
    def ndarray(self: SO3Matrix) -> np.ndarray:
        """Convert the matrix to a NumPy array."""
        matrix = np.array(list(map(float, self.flatten)))
        return (_SQRT2**-self.k) * matrix.reshape(3, 3)

    @property
    def parity_mat(self: SO3Matrix) -> np.ndarray:
        """Return the parity of the SO(3) matrix."""
        return np.array([[x.a % 2 for x in row] for row in self.so3mat], dtype=np.int8)

    @property
    def parity_vec(self: SO3Matrix) -> np.ndarray:
        """Return the permutation vector of the SO(3) matrix."""
        return np.sum(self.parity_mat, axis=1)

    def from_matrix(self, matrix: DyadicMatrix) -> list[list[ZSqrtTwo]]:
        """Return the SO(3) matrix as a list of lists."""
        su2_elems, k = matrix.flatten, 2 * matrix.k
        if any(s.parity for s in su2_elems):
            z_sqrt2 = [(ZSqrtTwo((s.c - s.a), s.d), ZSqrtTwo((s.c + s.a), s.b)) for s in su2_elems]
            k += 2
        else:  # pragma: no cover
            z_sqrt2 = [
                (ZSqrtTwo(s.d, (s.c - s.a) // 2), ZSqrtTwo(s.b, (s.c + s.a) // 2))
                for s in su2_elems
            ]

        a_, b_, c_, d_ = z_sqrt2
        so3_mat = [
            [
                a_[0] * d_[0] + a_[1] * d_[1] + b_[0] * c_[0] + b_[1] * c_[1],
                a_[1] * d_[0] + b_[0] * c_[1] - b_[1] * c_[0] - a_[0] * d_[1],
                a_[0] * c_[0] + a_[1] * c_[1] - b_[0] * d_[0] - b_[1] * d_[1],
            ],
            [
                a_[0] * d_[1] - a_[1] * d_[0] + b_[0] * c_[1] - b_[1] * c_[0],
                a_[0] * d_[0] + a_[1] * d_[1] - b_[0] * c_[0] - b_[1] * c_[1],
                a_[0] * c_[1] - a_[1] * c_[0] - b_[0] * d_[1] + b_[1] * d_[0],
            ],
            [
                2 * (a_[0] * b_[0] + a_[1] * b_[1]),
                2 * (a_[1] * b_[0] - a_[0] * b_[1]),
                a_[0] ** 2 + a_[1] ** 2 - b_[0] ** 2 - b_[1] ** 2,
            ],
        ]
        self.k = k
        return so3_mat

    def normalize(self: SO3Matrix) -> None:
        """Reduce the k value of the SO(3) matrix.

        Example:
            >>> A = DyadicMatrix(ZOmega(d=2), ZOmega(d=2), ZOmega(d=2), ZOmega(d=2), k = 4) * 2
            >>> B = SO3Matrix(A @ A)
            >>> B.normalize()
            >>> B
            SO3Matrix(matrix=[[1, 1], [1, 1]], k=-6)
        """
        elements = self.flatten
        if all(s.a == 0 and s.b == 0 for s in elements):
            self.k = 0
            return
        # Factoring 2: Derived using (a + b . √2) => 2 . (a//2 + b//2 . √2)
        while all(s.a % 2 == 0 and s.b % 2 == 0 for s in elements):
            self.k -= 2
            elements = [s // 2 for s in elements]
        # Factoring sqrt(2): Derived using (a + b . √2) => √2 (b + a // 2 . √2)
        while all(s.a % 2 == 0 for s in elements) and self.k > 0:
            self.k -= 1
            elements = [ZSqrtTwo(s.b, s.a // 2) for s in elements]
        self.so3mat = [elements[i : i + 3] for i in range(0, len(elements), 3)]
