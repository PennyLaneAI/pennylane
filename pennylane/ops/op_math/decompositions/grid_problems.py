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
"""Contains 1-D and 2-D grid problem solving utilities."""

from __future__ import annotations

import math
from collections.abc import Iterable
from copy import copy
from functools import cached_property, lru_cache
from itertools import chain

from pennylane.ops.op_math.decompositions.rings import _SQRT2, ZOmega, ZSqrtTwo

_LAMBDA = 1 + math.sqrt(2)


class Ellipse:
    r"""A class representing an ellipse as a positive definite matrix.

    The ellipse is defined by the positive definite matrix :math:`D`:

    .. math::
        D = \begin{bmatrix}
            a & b \\
            b & d
        \end{bmatrix}

    The matrix is related to the equation of the ellipse :math:`ax^2 + 2bxy + dy^2 = 1` by:

    .. math::
        a = \frac{\sin^2(\theta)}{m^2} + \frac{\cos^2(\theta)}{n^2}
        b = \cos(\theta) \sin(\theta) \left(\frac{1}{m^2} - \frac{1}{n^2}\right)
        d = \frac{\sin^2(\theta)}{m^2} + \frac{\cos^2(\theta)}{n^2}

    where :math:`m` and :math:`n` are the lengths of the semi-major and semi-minor axes,
    and :math:`\theta` is the angle of the ellipse.

    Args:
        D (list[float, float, float]): The elements of the positive definite matrix.
        p (tuple[float, float]): The center of the ellipse.

    .. note::
      We obtain corresponding matrix :math:`D^{\prime}` with determinant :math:`1` from :math:`D`.
      These are useful for computing key properties of ``EllipseState``, i.e., a pair of ellipses:
        - a = eλ^{-z}, d = eλ^z, b^2 = e^2 - 1; (Eq. 31, arXiv:1403.2975)
        - 2z * log(λ) = log(d / a) => z = 0.5 * log(d / a) / log(λ)
    """

    def __init__(
        self,
        D: tuple[float, float, float],
        p: tuple[float, float] = (0, 0),
    ):
        self.a, self.b, self.d = D
        self.p = p
        self.z = 0.5 * math.log2(self.d / self.a) / math.log2(_LAMBDA)
        self.e = math.sqrt(self.a * self.d)

    @classmethod
    def from_region(cls, theta: float, epsilon: float, k: int = 0):
        r"""Create an ellipse that bounds the region :math:`u | u \bullet z \geq 1 - \epsilon^2 / 2`,
        with :math:`u \in \frac{1}{\sqrt{2}^k} \mathbb{Z}[\omega]` and :math`z = \exp{-i\theta / 2}`.
        """
        t = epsilon**2 / 2
        scale = (2 ** (k // 2)) * (_SQRT2 ** (k % 2))

        a, b = scale * t, scale * epsilon

        a2, b2 = 1 / a**2, 1 / b**2
        d2 = a2 - b2

        zx, zy = math.cos(theta), math.sin(theta)
        a = d2 * zx * zx + b2
        d = d2 * zy * zy + b2
        b = d2 * zx * zy

        const = 1 - t
        p = (const * scale * zx, const * scale * zy)

        return cls([a, b, d], p)

    def __repr__(self) -> str:
        """Return a string representation of the ellipse."""
        return f"Ellipse(a={self.a}, b={self.b}, d={self.d}, p={self.p})"

    def __eq__(self, other: Ellipse) -> bool:
        """Check if the ellipses are equal."""
        return self.a == other.a and self.b == other.b and self.d == other.d and self.p == other.p

    @property
    def discriminant(self) -> float:
        """Calculate the discriminant of the characteristic polynomial associated with the ellipse."""
        return (self.a + self.d) ** 2 - 4 * (self.a * self.d - self.b**2)

    @property
    def determinant(self) -> float:
        """Calculate the determinant of the ellipse."""
        return self.a * self.d - self.b**2

    @property
    def positive_semi_definite(self) -> bool:
        """Check if the ellipse is positive semi-definite."""
        return (self.a + self.d) + math.sqrt(self.discriminant) >= 0

    @property
    def uprightness(self) -> float:
        """Calculate the uprightness of the ellipse (Eq. 32, arXiv:1403.2975)."""
        return math.pi / (2 * self.e) ** 2

    @staticmethod
    def b_from_uprightness(uprightness: float) -> float:
        """Calculate the b value of the ellipse from its uprightness (Eq. 33, arXiv:1403.2975)."""
        return math.sqrt((math.pi / (4 * uprightness)) ** 2 - 1)

    def contains(self, x: float, y: float) -> bool:
        """Check if the point (x, y) is inside the ellipse."""
        x_, y_ = x - self.p[0], y - self.p[1]
        return (self.a * x_**2 + 2 * self.b * x_ * y_ + self.d * y_**2) <= 1

    def normalize(self) -> tuple[Ellipse, float]:
        """Normalize the ellipse to have a determinant of 1."""
        s_val = 1 / math.sqrt(self.determinant)
        return self.scale(scale=s_val), s_val

    def scale(self, scale: float) -> Ellipse:
        """Scale the ellipse by a factor of scale."""
        D = (self.a * scale, self.b * scale, self.d * scale)
        return Ellipse(D, self.p)

    def x_points(self, y: float) -> tuple[float, float]:
        """Compute the x-points of the ellipse for a given y-value."""
        y = float(y) - self.p[1]  # shift y to the origin
        discriminant = y**2 * (self.b**2 - self.a * self.d) + self.a
        if discriminant < 0:
            raise ValueError(f"Point y={y} is outside the ellipse")

        x0, d0 = self.p[0], math.sqrt(discriminant)
        x1 = (-self.b * y - d0) / self.a
        x2 = (-self.b * y + d0) / self.a
        return x0 + x1, x0 + x2

    def y_points(self, x: float) -> tuple[float, float] | None:
        """Compute the y-points of the ellipse for a given x-value."""
        x = float(x) - self.p[0]  # shift x to the origin
        discriminant = (self.b * x) ** 2 - self.d * (self.a * x**2 - 1)
        if discriminant < 0:
            raise ValueError(f"Point x={x} is outside the ellipse")

        y0, d0 = self.p[1], math.sqrt(discriminant)
        y1 = (-self.b * x - d0) / self.d
        y2 = (-self.b * x + d0) / self.d
        return y0 + y1, y0 + y2

    def bounding_box(self) -> tuple[float, float, float, float]:
        """Return the bounding box of the ellipse in the form [[x0, x1], [y0, y1]]."""
        denom = self.determinant
        x, y = math.sqrt(self.d / denom), math.sqrt(self.a / denom)
        return (-x, x, -y, y)

    def offset(self, offset: float) -> Ellipse:
        """Return the ellipse shifted by the offset."""
        p_offset = (self.p[0] + offset, self.p[1] + offset)
        return Ellipse((self.a, self.b, self.d), p_offset)

    def apply_grid_op(self, grid_op: GridOp) -> Ellipse:
        """Apply a grid operation :math:`G` to the ellipse :math:`E` as :math:`G^T E G`."""
        ga, gb, gc, gd = grid_op.flatten

        D = (
            ga**2 * self.a + 2 * ga * gc * self.b + self.d * gc**2,
            ga * gb * self.a + (ga * gd + gb * gc) * self.b + gc * gd * self.d,
            gb**2 * self.a + 2 * gb * gd * self.b + self.d * gd**2,
        )

        p1, p2 = self.p
        gda, gdb, gdc, gdd = grid_op.inverse().flatten

        return Ellipse(D, (gda * p1 + gdb * p2, gdc * p1 + gdd * p2))


class EllipseState:
    """A class representing a state as a pair of normalized ellipses.

    This is based on the Definition A.1 of arXiv:1403.2975,
    where the pair of ellipses are represented by real symmetric
    positive semi-definite matrices of determinant :math:`1`.

    Args:
        e1 (Ellipse): The first ellipse.
        e2 (Ellipse): The second ellipse.
    """

    def __init__(self, e1: Ellipse, e2: Ellipse):
        self.e1 = e1
        self.e2 = e2

    def __repr__(self) -> str:
        """Return a string representation of the state."""
        return f"EllipseState(e1={self.e1}, e2={self.e2})"

    @cached_property
    def skew(self) -> float:
        """Calculate the skew of the state."""
        # Uses Definition A.1 of arXiv:1403.2975
        return self.e1.b**2 + self.e2.b**2

    @cached_property
    def bias(self) -> float:
        """Calculate the bias of the state."""
        # Uses Definition A.1 of arXiv:1403.2975
        return self.e2.z - self.e1.z

    def skew_grid_op(self) -> GridOp:
        """Calculate the special grid operation for the state for reducing the skew."""
        # Uses Lemma A.5 (Step Lemma) of arXiv:1403.2975 for obtaining special grid op.
        grid_op = GridOp.from_string("I")
        state = EllipseState(self.e1, self.e2)
        while (skew := state.skew) >= 15:
            new_grid_op, state = state.reduce_skew()
            grid_op = grid_op * new_grid_op
            if state.skew > 0.9 * skew:  # pragma: no cover
                raise ValueError(f"Skew was not decreased for state {state}")

        return grid_op

    def apply_grid_op(self, grid_op: GridOp) -> EllipseState:
        """Apply a grid operation :math:`G` to the state."""
        # Uses Definition A.3 of arXiv:1403.2975
        return EllipseState(self.e1.apply_grid_op(grid_op), self.e2.apply_grid_op(grid_op.adj2()))

    def apply_shift_op(self) -> tuple[EllipseState, int]:
        """Apply a shift operator to the state."""
        # Uses Definition A.6 and Lemma A.8 of arXiv:1403.2975
        k = int(math.floor((1 - self.bias) / 2))
        pk_pow, nk_pow = _LAMBDA ** (k), _LAMBDA ** (-k)
        e1, e2 = copy(self.e1), copy(self.e2)
        e1.a, e1.d, e1.z = e1.a * pk_pow, e1.d * nk_pow, e1.z - k
        e2.a, e2.d, e2.z = e2.a * nk_pow, e2.d * pk_pow, e2.z + k
        e2.b *= (-1) ** k
        return EllipseState(e1, e2), k

    # pylint: disable=too-many-branches
    def reduce_skew(self) -> tuple[GridOp, EllipseState]:
        """Reduce the skew of the state.

        This uses Step Lemma described in Appendix A.6 of arXiv:1403.2975.

        Returns:
            tuple[GridOp, EllipseState]: A tuple containing the grid operation
                and the state with reduced skew.
        """
        if any(not e.positive_semi_definite for e in (self.e1, self.e2)):  # pragma: no cover
            raise ValueError("Ellipse is not positive semi-definite")

        sign, k = 1, 0
        grid_op = GridOp.from_string("I")

        if self.e2.b < 0:
            grid_op = grid_op * GridOp.from_string("Z")

        if (self.e1.z + self.e2.z) < 0:
            sign *= -1  # Bookkeeping for the sign flip
            grid_op = grid_op * GridOp.from_string("X")

        if abs(self.bias) > 2:
            n = int(round((1 - sign * self.bias) / 4))
            grid_op = grid_op * (GridOp.from_string("U") ** n)

        n_grid_op = GridOp.from_string("I")
        new_state = self.apply_grid_op(grid_op)

        if abs(new_state.bias) > 1:
            new_state, k = new_state.apply_shift_op()

            if new_state.e2.b < 0:
                grid_op_z = GridOp.from_string("Z")
                new_state = new_state.apply_grid_op(grid_op_z)
                n_grid_op = n_grid_op * grid_op_z

            if (new_state.e1.z + new_state.e2.z) < 0:  # pragma: no cover
                grid_op_x = GridOp.from_string("X")
                new_state = new_state.apply_grid_op(grid_op_x)
                n_grid_op = n_grid_op * grid_op_x

        e1, e2 = new_state.e1, new_state.e2

        if -0.8 <= e1.z <= 0.8 and -0.8 <= e2.z <= 0.8:
            n_grid_op = n_grid_op * GridOp.from_string("R")
        else:
            if e1.b >= 0:
                if e1.z <= 0.3 and e2.z >= 0.8:
                    n_grid_op = n_grid_op * GridOp.from_string("K")
                elif e1.z >= 0.8 and e2.z <= 0.3:
                    n_grid_op = n_grid_op * GridOp.from_string("K").adj2()
                elif e1.z >= 0.3 and e2.z >= 0.3:
                    n = int(max(1, math.floor((_LAMBDA ** min(e1.z, e2.z)) / 2)))
                    n_grid_op = n_grid_op * (GridOp.from_string("A") ** n)
                else:  # pragma: no cover
                    raise ValueError(f"Skew couldn't be reduced for the state {new_state}")
            else:
                if e1.z >= -0.2 and e2.z >= -0.2:
                    n = int(max(1, math.floor((_LAMBDA ** min(e1.z, e2.z)) / _SQRT2)))
                    n_grid_op = n_grid_op * (GridOp.from_string("B") ** n)
                else:  # pragma: no cover
                    raise ValueError(f"Skew couldn't be reduced for the state {new_state}")

        if k != 0:
            n_grid_op = n_grid_op.apply_shift_op(k)

        grid_op = grid_op * n_grid_op
        return grid_op, self.apply_grid_op(grid_op)


class GridOp:
    r"""A class representing a grid operation on a 2D grid.

    This follows Definition 5.10 and Lemma 5.11 of arXiv:1403.2975,
    where a grid operator :math:`G` is described as a linear map,
    which can be identifed with a :math:`2 \times 2` matrix as follows:

    .. math::
        G = \begin{pmatrix} a & b \\ c & d \end{pmatrix}.

    Each entry :math:`m` of the matrix is of the form :math:`m = m_0 + m_1 / \sqrt{2}`,
    where :math:`m_0, m_1 \in \mathbb{Z}`. They satisfy :math:`a_0+b_0+c_0+d_0 \equiv 0 (\mod 2)`
    and :math:`a_1 \equiv b_1 \equiv c_1 \equiv d_1 (\mod 2)`.

    Args:
        a (tuple[int, int]): The a-coefficient of the grid operation.
        b (tuple[int, int]): The b-coefficient of the grid operation.
        c (tuple[int, int]): The c-coefficient of the grid operation.
        d (tuple[int, int]): The d-coefficient of the grid operation.
        check_valid (bool): If ``True``, the grid operation will be checked to be a valid
            grid operation. Default is ``True``.

    .. note::
        The coefficients are given as tuples of the form (a_0, a_1), which corresponds
        to an element being :math:`a = a_0 + a_1 / \sqrt{2}`.
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        a: tuple[int, int],
        b: tuple[int, int],
        c: tuple[int, int],
        d: tuple[int, int],
        check_valid: bool = True,
    ) -> None:
        """Initialize the grid operation."""
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        if check_valid:
            assert (a[0] + b[0] + c[0] + d[0]) % 2 == 0, "sum of a_0, b_0, c_0, d_0 must be even"
            assert (
                a[1] % 2 == b[1] % 2 == c[1] % 2 == d[1] % 2
            ), "a_1, b_1, c_1, d_1 must have same parity"

    def __repr__(self) -> str:
        """Return a string representation of the grid operation."""
        return f"GridOp(a={self.a}, b={self.b}, c={self.c}, d={self.d})"

    def __eq__(self, other: GridOp) -> bool:
        """Check if the grid operations are equal."""
        return self.a == other.a and self.b == other.b and self.c == other.c and self.d == other.d

    def __mul__(self, other: GridOp | ZOmega) -> GridOp | ZOmega:
        if isinstance(other, GridOp):
            return self._mul_grid_op(other)
        if isinstance(other, ZOmega):
            return self._mul_z_omega(other)
        raise TypeError(f"Cannot multiply GridOp with {type(other)}")

    def __pow__(self, n: int) -> GridOp:
        """Raise the grid operator to a power."""
        if self == self.from_string("I") or n == 0:  # Identity:
            return GridOp((1, 0), (0, 0), (0, 0), (1, 0))
        if self == self.from_string("A"):  # A
            return GridOp((1, 0), (-2 * n, 0), (0, 0), (1, 0))
        if self == self.from_string("B"):  # B
            return GridOp((1, 0), (0, 2 * n), (0, 0), (1, 0))
        if self == self.from_string("U"):  # U
            (c1, c2), c3 = (ZSqrtTwo(1, 1) ** abs(n)).flatten, (-1) ** (n < 0)
            return GridOp((c3 * c1, 2 * c2), (0, 0), (0, 0), (-c3 * c1, 2 * c2))

        x, res = self, GridOp((1, 0), (0, 0), (0, 0), (1, 0))
        while n > 0:
            if n % 2 == 1:
                res = res * x
            x = x * x
            n //= 2
        return res

    def _mul_grid_op(self, other: GridOp) -> GridOp:
        """Multiply the grid operator with another grid operator."""
        a = (
            self.a[0] * other.a[0]
            + self.b[0] * other.c[0]
            + (self.a[1] * other.a[1] + self.b[1] * other.c[1]) // 2,
            self.a[0] * other.a[1]
            + self.a[1] * other.a[0]
            + self.b[0] * other.c[1]
            + self.b[1] * other.c[0],
        )
        b = (
            self.a[0] * other.b[0]
            + self.b[0] * other.d[0]
            + (self.a[1] * other.b[1] + self.b[1] * other.d[1]) // 2,
            self.a[0] * other.b[1]
            + self.a[1] * other.b[0]
            + self.b[0] * other.d[1]
            + self.b[1] * other.d[0],
        )
        c = (
            self.c[0] * other.a[0]
            + self.d[0] * other.c[0]
            + (self.c[1] * other.a[1] + self.d[1] * other.c[1]) // 2,
            self.c[0] * other.a[1]
            + self.c[1] * other.a[0]
            + self.d[0] * other.c[1]
            + self.d[1] * other.c[0],
        )
        d = (
            self.c[0] * other.b[0]
            + self.d[0] * other.d[0]
            + (self.c[1] * other.b[1] + self.d[1] * other.d[1]) // 2,
            self.c[0] * other.b[1]
            + self.c[1] * other.b[0]
            + self.d[0] * other.d[1]
            + self.d[1] * other.d[0],
        )
        return GridOp(a, b, c, d)

    def _mul_z_omega(self, other: ZOmega) -> ZOmega:
        """Multiply the grid operator with a ZOmega element."""
        x1, y1 = other.d, other.b
        x2, y2 = other.c - other.a, other.c + other.a
        a_ = self.a[0] * x2 + self.a[1] * x1 + self.b[0] * y2 + self.b[1] * y1
        c_ = self.c[0] * x2 + self.c[1] * x1 + self.d[0] * y2 + self.d[1] * y1
        d_ = self.a[0] * x1 + self.b[0] * y1 + (self.a[1] * x2 + self.b[1] * y2) // 2
        b_ = self.c[0] * x1 + self.d[0] * y1 + (self.c[1] * x2 + self.d[1] * y2) // 2
        return ZOmega((c_ - a_) // 2, b_, (c_ + a_) // 2, d_)

    @classmethod
    def from_string(cls, string: str) -> GridOp:
        """Return the grid operation from a string.

        Args:
            string: Supported string are: "I", "R", "A", "B", "K", "X", "Z".
                (Fig 6, arXiv:1403.2975).

        Returns:
            GridOp: The grid operation corresponding to the string.
        """
        return _useful_grid_ops().get(string, "I")

    @property
    def determinant(self) -> tuple[float, float]:
        r"""Calculate the determinant of the grid operation.

        The determinant will be of form :math:`a + b / \sqrt{2}` and is given as tuple(a, b).
        """
        return (
            self.a[0] * self.d[0]
            - self.b[0] * self.c[0]
            + (self.a[1] * self.d[1] - self.b[1] * self.c[1]) // 2,
            self.a[0] * self.d[1]
            - self.b[0] * self.c[1]
            + self.a[1] * self.d[0]
            - self.b[1] * self.c[0],
        )

    @property
    def is_special(self) -> bool:
        """Check if the grid operation is special based on Proposition 5.13 of arXiv:1403.2975."""
        det1, det2 = self.determinant
        return det2 == 0 and det1 in {1, -1}

    @property
    def flatten(self) -> list[float]:
        """Flatten the grid operation based on description in Lemma 5.11 of arXiv:1403.2975."""
        return [
            self.a[0] + self.a[1] / _SQRT2,
            self.b[0] + self.b[1] / _SQRT2,
            self.c[0] + self.c[1] / _SQRT2,
            self.d[0] + self.d[1] / _SQRT2,
        ]

    def inverse(self) -> GridOp:
        """Compute the inverse of the grid operation."""
        det1, det2 = self.determinant
        if det2 == 0 and det1 in {1, -1}:
            return GridOp(
                tuple(d // det1 for d in self.d),
                tuple(-b // det1 for b in self.b),
                tuple(-c // det1 for c in self.c),
                tuple(a // det1 for a in self.a),
            )
        raise ValueError("Grid operator needs to be special to have an inverse.")

    def transpose(self) -> GridOp:
        """Transpose the grid operation."""
        return GridOp(
            self.d,
            self.b,
            self.c,
            self.a,
        )

    def adj2(self) -> GridOp:
        """Compute the sqrt(2)-conjugate of the grid operation."""
        return GridOp(
            (self.a[0], -self.a[1]),
            (self.b[0], -self.b[1]),
            (self.c[0], -self.c[1]),
            (self.d[0], -self.d[1]),
        )

    def apply_to_ellipse(self, ellipse: Ellipse) -> Ellipse:
        """Apply the grid operator to an ellipse based on Lemma A.4's proof in arXiv:1403.2975."""
        return ellipse.apply_grid_op(self)

    def apply_to_state(self, state: EllipseState) -> tuple[Ellipse, Ellipse]:
        """Apply the grid operator to a state based on Definition A.3 of arXiv:1403.2975."""
        return state.apply_grid_op(self)

    def apply_shift_op(self, k: int) -> GridOp:
        """Apply a shift operator to the grid operation based on Lemma A.9 of arXiv:1403.2975."""
        k, sign = abs(k), (-1) ** (k < 0)  # +1 if k > 0, -1 if k < 0
        s1, s2 = (ZSqrtTwo(1, 1) ** k).flatten
        return GridOp(
            (self.a[1] * s2 + sign * self.a[0] * s1, 2 * self.a[0] * s2 + sign * self.a[1] * s1),
            self.b,
            self.c,
            (self.d[1] * s2 - sign * self.d[0] * s1, 2 * self.d[0] * s2 - sign * self.d[1] * s1),
        )


@lru_cache(maxsize=1)
def _useful_grid_ops() -> dict[str, GridOp]:
    """Generate a list of useful grid operations based on Fig 6 of arXiv:1403.2975."""
    return {
        "I": GridOp((1, 0), (0, 0), (0, 0), (1, 0)),
        "A": GridOp((1, 0), (-2, 0), (0, 0), (1, 0)),
        "B": GridOp((1, 0), (0, 2), (0, 0), (1, 0)),
        "K": GridOp((-1, 1), (0, -1), (1, 1), (0, 1)),
        "R": GridOp((0, 1), (0, -1), (0, 1), (0, 1)),
        "U": GridOp((1, 2), (0, 0), (0, 0), (-1, 2)),
        "X": GridOp((0, 0), (1, 0), (1, 0), (0, 0)),
        "Z": GridOp((1, 0), (0, 0), (0, 0), (-1, 0)),
    }


class GridIterator:
    r"""Iterate over the solutions to the scaled grid problem.

    This is based on the Section 5 of arXiv:1403.2975 and implements Proposition 5.22,
    to enumerate all solutions to the scaled grid problem over the :math:`\epsilon`-region
    and a unit disk.

    It implements an ``__iter__`` method to iterate over the solutions efficiently as a generator.

    Args:
        theta (float): The angle of the grid problem.
        epsilon (float): The epsilon of the grid problem.
        max_trials (int): The maximum number of iterations. Default is ``20``.
    """

    def __init__(self, theta: float = 0.0, epsilon: float = 1e-3, max_trials: int = 20):
        self.theta = theta
        self.epsilon = epsilon
        self.zval = math.cos(theta), math.sin(theta)
        self.kmin = int(3 * math.log2(1 / epsilon) // 2)
        self.max_trials = max_trials
        self.target = 1 - self.epsilon**2 / 2

    def __repr__(self) -> str:
        """Return a string representation of the grid iterator."""
        return f"GridIterator(theta={self.theta}, epsilon={self.epsilon}, max_trials={self.max_trials})"

    def __iter__(self) -> Iterable[tuple[ZOmega, int]]:
        """Iterate over the solutions to the scaled grid problem."""
        # Warm start for an initial guess, where 14 is kmin for 1e-3.
        k, i_ = min(self.kmin, 14), 6  # Give 6 trials for warm start.
        e_, t_ = max(self.epsilon, 1e-3), min(self.target, 0.9999995)
        e1 = Ellipse.from_region(self.theta, e_, k)  # Ellipse for the epsilon-region.
        e2 = Ellipse((1, 0, 1), (0, 0))  # Ellipse for the unit disk.
        en, _ = e1.normalize()  # Normalize the epsilon-region.
        grid_op = EllipseState(en, e2).skew_grid_op()  # Skew grid operation for the epsilon-region.

        guess_solutions = (  # Solutions for the trivial cases.
            ZOmega(a=-1),
            ZOmega(b=-1),
            ZOmega(b=1),
            ZOmega(c=1),
            ZOmega(d=1),
            ZOmega(d=-1),
        )

        for sol in guess_solutions:
            complx_sol = complex(sol)
            dot_prod = self.zval[0] * complx_sol.real + self.zval[1] * complx_sol.imag
            norm_zsqrt_two = abs(float(sol.norm().to_sqrt_two()))
            if norm_zsqrt_two <= 1:
                if dot_prod >= self.target:
                    yield sol, 0

        int_s, init_k = [], []  # Fallback solution.
        for ix in range(self.max_trials):
            # Update the radius of the unit disk.
            radius = 2**-k
            e2_ = Ellipse(
                (radius, 0, radius), (0, 0)
            )  # Ellipse.from_axes(p=(0, 0), theta=0, axes=(radius, radius))
            # Apply the grid operation to the state and solve the two-dimensional grid problem.
            state = EllipseState(e1, e2_).apply_grid_op(grid_op)
            potential_solutions = self.solve_two_dim_problem(state)
            try:
                for solution in potential_solutions:
                    # Normalize the solution and obtain the scaling exponent of sqrt(2).
                    scaled_sol, kf = (grid_op * solution).normalize()

                    complx_sol = complex(scaled_sol)
                    sol_real, sol_imag = complx_sol.real, complx_sol.imag

                    k_ = k - kf  # Update the scaling exponent of sqrt(2).
                    dot_prod = (self.zval[0] * sol_real + self.zval[1] * sol_imag) / (
                        2 ** (k_ // 2) * (math.sqrt(2) ** (k_ % 2))
                    )

                    # Check if the solution is follows the constraints of the target-region.
                    norm_zsqrt_two = abs(float(scaled_sol.norm().to_sqrt_two()))
                    if norm_zsqrt_two <= 2**k_:
                        if dot_prod >= self.target:
                            yield scaled_sol, k_
                        elif dot_prod >= t_:
                            int_s.append(scaled_sol)
                            init_k.append(k_)

                if ix == i_:
                    k, e_, t_ = max(self.kmin, k + 1), self.epsilon, t_ / 10
                    en_, _ = Ellipse.from_region(self.theta, e_, self.kmin).normalize()
                    grid_op = EllipseState(en_, e2).skew_grid_op()
                else:
                    k = k + 1

                e1 = Ellipse.from_region(self.theta, e_, k)

            except (ValueError, ZeroDivisionError):  # pragma: no cover
                break

        for s, k in zip(int_s + [ZOmega(d=1)], init_k + [0]):
            yield s, k

    def solve_two_dim_problem(
        self, state: EllipseState, num_points: int = 1000
    ) -> Iterable[ZOmega]:
        r"""Solve the grid problem for the state(E1, E2).

        The solutions :math:`u \in Z[\omega]` are such that :math:`u \in E1` and
        :math:`u.adj2() \in E2`, where ``adj2`` is :math:`\sqrt(2)` conjugation.

        This is based on Proposition 5.21 and Theorem 5.18 of arXiv:1403.2975.

        Args:
            state: The state corresponding to the grid problem.
            num_points: The number of points to use to determine if the rectangle is wider
                than the other. Default is ``1000``.

        Returns:
            Iterable[ZOmega]: The list of solutions to the two dimensional grid problem.
        """
        e1, e2 = state.e1, state.e2
        state2 = EllipseState(e1.offset(-1 / _SQRT2), e2.offset(1 / _SQRT2))

        bbox1 = e1.bounding_box()
        bbox11 = tuple(bb_ + e1.p[ix_ // 2] for ix_, bb_ in enumerate(bbox1))
        bbox12 = tuple(bb_ - 1 / _SQRT2 for bb_ in bbox11)

        bbox2 = e2.bounding_box()
        bbox21 = tuple(bb_ + e2.p[ix_ // 2] for ix_, bb_ in enumerate(bbox2))
        bbox22 = tuple(bb_ + 1 / _SQRT2 for bb_ in bbox21)

        # Check if it is easier to solve problem for either of x-or-y-interval.
        # Based on this, we can try to first solve for alpha-or-beta and then refine for other.
        # If both of them are balanced, rely on doing naive search over both.
        bbox_zip = list(zip((bbox11, bbox12), (bbox21, bbox22)))
        num_x1, num_x2 = (self.bbox_grid_points(bb1[:2] + bb2[:2]) for bb1, bb2 in bbox_zip)
        num_y1, num_y2 = (self.bbox_grid_points(bb1[2:] + bb2[2:]) for bb1, bb2 in bbox_zip)
        num_b1 = [num_x1 > num_points * num_y1, num_y1 > num_points * num_x1]
        num_b2 = [num_x2 > num_points * num_y2, num_y2 > num_points * num_x2]

        # Solve the problem for the two cosets of ZOmega ring and add non-zero offset to odd one.
        potential_sols1 = self.solve_upright_problem(state, bbox11, bbox21, num_b1, ZOmega())
        potential_sols2 = self.solve_upright_problem(state2, bbox12, bbox22, num_b2, ZOmega(c=1))

        for solution in chain(potential_sols1, potential_sols2):
            sol1, sol2 = complex(solution), complex(solution.adj2())
            x1, y1 = sol1.real, sol1.imag
            x2, y2 = sol2.real, sol2.imag
            if e1.contains(x1, y1) and e2.contains(x2, y2):
                yield solution

    # pylint:disable = too-many-arguments, too-many-branches
    def solve_upright_problem(
        self,
        state: EllipseState,
        bbox1: tuple[float],
        bbox2: tuple[float],
        num_b: list[bool],
        shift: ZOmega,
    ) -> Iterable[ZOmega]:
        r"""Iterates over the solutions to the grid problem for two upright rectangles.

        The solutions :math:`u \in Z[\omega]` are such that :math:`u \in A` and
        :math:`u.adj2() \in B`, where ``adj2`` is :math:`\sqrt(2)` conjugation
        and two rectangles :math:`A` and :math:`B`, form the subregions of
        :math:`\mathbb{R}^2` of the form :math:`[x0, x1] \times [y0, y1]`.

        Args:
            state (State): The state of the grid problem.
            bbox1 (tuple[float]): The bounding box of the first rectangle.
            bbox2 (tuple[float]): The bounding box of the second rectangle.
            num_b (list[bool]): Whether the second rectangle is wider than the first.
            shift (ZOmega): The shift operator.

        Returns:
            Iterable[ZOmega]: The list of solutions to the upright grid problem for two rectangles.
        """
        e1, e2 = state.e1, state.e2
        Ax0, Ax1, Ay0, Ay1 = bbox1
        Bx0, Bx1, By0, By1 = bbox2

        if num_b[0]:  # pragma: no cover # If it is easier to solve for beta first.
            beta_solutions1 = self.solve_one_dim_problem(Ay0, Ay1, By0, By1)
            for beta in beta_solutions1:
                Ax0_tmp, Ax1_tmp = e1.x_points(beta)
                Bx0_tmp, Bx1_tmp = e2.x_points(beta.adj2())
                if Ax1_tmp - Ax0_tmp > 0 and Bx1_tmp - Bx0_tmp > 0:
                    new_alpha_solutions = self.solve_one_dim_problem(
                        Ax0_tmp, Ax1_tmp, Bx0_tmp, Bx1_tmp
                    )
                    for alpha in new_alpha_solutions:
                        yield ZOmega.from_sqrt_pair(alpha, beta, shift)
        elif num_b[1]:  # If it is easier to solve for alpha first.
            alpha_solutions1 = self.solve_one_dim_problem(Ax0, Ax1, Bx0, Bx1)
            for alpha in alpha_solutions1:
                Ay0_tmp, Ay1_tmp = e1.y_points(alpha)
                By0_tmp, By1_tmp = e2.y_points(alpha.adj2())
                if Ay1_tmp - Ay0_tmp > 0 and By1_tmp - By0_tmp > 0:
                    new_beta_solutions = self.solve_one_dim_problem(
                        Ay0_tmp, Ay1_tmp, By0_tmp, By1_tmp
                    )
                    for beta in new_beta_solutions:
                        yield ZOmega.from_sqrt_pair(alpha, beta, shift)
        else:  # If both of them are balanced, solve for both and refine.
            alpha_solutions1 = self.solve_one_dim_problem(Ax0, Ax1, Bx0, Bx1)
            beta_solutions1 = self.solve_one_dim_problem(Ay0, Ay1, By0, By1)
            found_beta1_solutions = []
            for alpha in alpha_solutions1:
                if len(found_beta1_solutions) == 0:
                    for beta in beta_solutions1:
                        found_beta1_solutions.append(beta)
                        yield ZOmega.from_sqrt_pair(alpha, beta, shift)
                else:
                    for beta in found_beta1_solutions:
                        yield ZOmega.from_sqrt_pair(alpha, beta, shift)

    @staticmethod
    def bbox_grid_points(bbox: tuple[float, float, float, float]) -> int:
        """Count the number of grid points in a bounding box.

        This gives an estimation on the expected number of solution within
        the bounding box. This is based on the Lemma 16 of arXiv:1212.6253.

        Args:
            bbox (tuple[float, float, float, float]): The bounding box.

        Returns:
            int: The number of grid points in the bounding box.
        """
        d_ = math.log2(ZSqrtTwo(1, 1))
        l1, l2 = ZSqrtTwo(1, 1), ZSqrtTwo(-1, 1)
        d1, d2 = (bbox[1] - bbox[0], bbox[3] - bbox[2])

        # Find the integer scaling factor for the x and y intervals, such that
        # \delta_{1/2} \cdot (\lambda - 1)^{k_{1/2}} < 1, where \lambda= 1/√2.
        k1, k2 = (int(math.floor(math.log2(d) / d_ + 1)) for d in (d1, d2))
        if abs(k1) > abs(k2):  # If y-interval is wider than x-interval, swap.
            bbox, k1, k2 = (bbox[2], bbox[3], bbox[0], bbox[1]), k2, k1

        # Scale the x and y intervals to enter the intended interval.
        # Look at `solve_one_dim_problem` for more details.
        x_scale = float((l1 if k1 < 0 else l2) ** abs(k1))
        y_scale = float((-1) ** k1 * (l2 if k1 < 0 else l1) ** abs(k1))

        x0_scaled, x1_scaled = x_scale * bbox[0], x_scale * bbox[1]
        y0_scaled, y1_scaled = sorted((y_scale * bbox[2], y_scale * bbox[3]))

        # Check if we are indeed within the intended interval.
        if x1_scaled - x0_scaled < 1 - _SQRT2:  # pragma: no cover
            raise ValueError(f"Value should be larger than 1 - sqrt(2) for bbox {bbox}")

        # Use the constraints x0 <= a + b * sqrt(2) <= x1 and y0 <= a - b * sqrt(2) <= y1
        # to obtain the bounds on b and eliminate the variable a to obtain the bounds on b.
        lower_bound_b = (x0_scaled - y1_scaled) / (2 * _SQRT2)
        upper_bound_b = (x1_scaled - y0_scaled) / (2 * _SQRT2)
        return 1 + int(upper_bound_b - lower_bound_b)

    @staticmethod
    def solve_one_dim_problem(
        x0: float,
        x1: float,
        y0: float,
        y1: float,
    ) -> Iterable[ZSqrtTwo]:
        r"""Iterates the solutions to the one dimensional grid problem given intervals :math:`[x0, x1]` and :math:`[y0, y1]`.

        Given two real intervals :math:`[x0, x1]` and :math:`[y0, y1]`
        such that :math:`\sqrt{(x1 - x0)*(y1 - y0)} >= (1 + \sqrt(2))`,
        iterates over all solutions of the form :math:`a + b\sqrt(2)` such that
        :math:`a + b\sqrt(2) \in [x0, x1]` and :math:`a - b\sqrt(2) \in [y0, y1]`.

        This is based on the Lemmas 16 and 17 of arXiv:1212.6253.

        Args:
            x0 (float): The lower bound of the x-interval.
            x1 (float): The upper bound of the x-interval.
            y0 (float): The lower bound of the y-interval.
            y1 (float): The upper bound of the y-interval.

        Returns:
            Iterable[ZSqrtTwo]: The list of solutions to the one dimensional grid problem.
        """
        d_ = math.log2(ZSqrtTwo(1, 1))
        l1, l2 = ZSqrtTwo(1, 1), ZSqrtTwo(-1, 1)
        d1, d2 = (x1 - x0, y1 - y0)

        f_adj2 = False  # Check if we need to apply the sqrt(2) conjugation.
        # Find the integer scaling factor for the x and y intervals, such that
        # \delta_{1/2} \cdot (\lambda - 1)^{k_{1/2}} < 1, where \lambda= 1/√2.
        k1, k2 = (int(math.floor(math.log2(d) / d_ + 1)) for d in (d1, d2))
        if abs(k1) > abs(k2):  # If y-interval is wider than x-interval, swap.
            f_adj2, k1, k2 = True, k2, k1
            x0, x1, y0, y1 = y0, y1, x0, x1

        # Turn the problem into a scaled grid problem,
        # such that we get to solve the specific case of:
        # -1 + √2 <= x1 - x0 < 1 and (x1 - x0)(y1 - y0) >= (1 + √2)^2.
        s_scale = ZSqrtTwo(1, 1) ** abs(k1)
        x_scale = float((l1 if k1 < 0 else l2) ** abs(k1))
        y_scale = float((-1) ** k1 * (l2 if k1 < 0 else l1) ** abs(k1))

        x0_scaled, x1_scaled = x_scale * x0, x_scale * x1
        y0_scaled, y1_scaled = sorted((y_scale * y0, y_scale * y1))

        # Check if we are solving the problem for the intended interval.
        if x1_scaled - x0_scaled < 1 - _SQRT2:  # pragma: no cover
            bbox = (x0_scaled, x1_scaled, y0_scaled, y1_scaled)
            raise ValueError(f"Value should be larger than 1 - sqrt(2) for bbox {bbox}")

        # Use the constraints y0 <= a - b * sqrt(2) <= y1 and x0 <= a + b * sqrt(2) <= x1
        # to obtain the bounds on b and eliminate the variable a to obtain the bounds on b.
        lower_bound_b = (x0_scaled - y1_scaled) / (2 * _SQRT2)
        upper_bound_b = (x1_scaled - y0_scaled) / (2 * _SQRT2)

        for b in range(int(math.floor(upper_bound_b)), int(math.ceil(lower_bound_b)) - 1, -1):
            # Use the constraints x0 <= a + b * sqrt(2) <= x1 to obtain the bounds on a.
            lower_bound_a = x0_scaled - b * _SQRT2
            upper_bound_a = x1_scaled - b * _SQRT2
            if upper_bound_a - lower_bound_a >= 1:  # pragma: no cover
                raise ValueError(f"Value should be less than one for {(x0, x1, y0, y1)}")

            # Check if the bounds on the interval contains an integer.
            if math.ceil(lower_bound_a) == math.floor(upper_bound_a):
                a = int(math.ceil(lower_bound_a))
                # Check if the solution satisfies both bounds on x and y.
                if (x0_scaled + y0_scaled <= 2 * a) and (2 * a <= x1_scaled + y1_scaled):
                    alpha, beta = a + b * _SQRT2, a - b * _SQRT2
                    # Check if the consecutive solutions are within the desired bounds.
                    if x0_scaled <= alpha <= x1_scaled and y0_scaled <= beta <= y1_scaled:
                        # Undo the scaling to obtain the solution.
                        sol = ZSqrtTwo(a, b) / s_scale if k1 < 0 else ZSqrtTwo(a, b) * s_scale
                        yield sol if not f_adj2 else sol.adj2()
