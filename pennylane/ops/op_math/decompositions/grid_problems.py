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
"""Contains 1-D and 2-D grid problem solving utilities."""

from __future__ import annotations

import math
from copy import copy
from functools import lru_cache

from pennylane.ops.op_math.decompositions.rings import _SQRT2, ZOmega

_LAMBDA = 1 + math.sqrt(2)


class Ellipse:
    r"""A class representing an ellipse as a positive definite matrix.

    The ellipse is defined by the positive definite matrix :math:`D`:

    .. math::
        D = \begin{bmatrix}
            a & b \\
            b & d
        \end{bmatrix}

    The matrix is related to the equation of the ellipse :math:`ax^2 + 2bxy + cy^2 = 1` by:

    .. math::
        a = \frac{\sin^2(\theta)}{m^2} + \frac{\cos^2(\theta)}{n^2}
        b = \cos(\theta) \sin(\theta) \left(\frac{1}{m^2} - \frac{1}{n^2}\right)
        d = \frac{\sin^2(\theta)}{m^2} + \frac{\cos^2(\theta)}{n^2}

    where :math:`m` and :math:`n` are the lengths of the semi-major and semi-minor axes,
    and :math:`\theta` is the angle of the ellipse.

    Args:
        D (list[float, float, float]): The elements of the positive definite matrix.
        p (tuple[float, float]): The center of the ellipse.
        axes (tuple[float, float]): The lengths of the semi-major and semi-minor axes.
    """

    def __init__(
        self,
        D: tuple[float, float, float],
        p: tuple[float, float] = (0, 0),
        axes: tuple[float, float] = (0, 0),
    ):
        self.a, self.b, self.d = D
        self.p = p
        self.axes = axes
        # a = eλ^{-z}, d = eλ^z, b^2 = e^2 - 1; (Eq. 31, arXiv:1403.2975)
        self.z = 0.5 * math.log2(self.d / self.a) / math.log2(_LAMBDA)
        self.e = math.sqrt(self.a * self.d)

    @classmethod
    def from_axes(cls, p: tuple[float, float], theta: float, axes: tuple[float, float]):
        """Create an ellipse from its axes and center point."""
        a = (math.sin(theta) / axes[1]) ** 2 + (math.cos(theta) / axes[0]) ** 2
        b = math.cos(theta) * math.sin(theta) * (1 / (axes[0]) ** 2 - 1 / (axes[1]) ** 2)
        c = (math.sin(theta) / axes[0]) ** 2 + (math.cos(theta) / axes[1]) ** 2
        return cls([a, b, c], p, axes)

    @classmethod
    def from_region(cls, epsilon: float, theta: float, k: int = 0):
        r"""Create an ellipse that bounds the region :math:`u | u \bullet z \geq 1 - \epsilon^2 / 2`,
        with :math:`u \in \frac{1}{\sqrt{2}^k} \mathbb{Z}[\omega]` and :math`z = \exp{-i\theta / 2}`.
        """
        const = 1 - epsilon**2 / 2
        shift = (1 + const) / 3
        scale = (2 ** (k // 2)) * (_SQRT2 ** (k % 2))
        semi_major = 2 * scale * shift
        semi_minor = 2 * scale * math.sqrt((const + 0.5 * epsilon**2) * epsilon**2 / 3)
        x, y = scale * (const + shift) * math.cos(theta), scale * (const + shift) * math.sin(theta)
        return cls.from_axes((x, y), theta, (semi_major, semi_minor))

    def __repr__(self) -> str:
        """Return a string representation of the ellipse."""
        return f"Ellipse(a={self.a}, b={self.b}, d={self.d}, p={self.p}, axes={self.axes}, z={self.z}, e={self.e})"

    @property
    def descriminant(self) -> float:
        """Calculate the descriminant of the ellipse."""
        return 1 * ((self.a + self.d) ** 2 - 4 * (self.a * self.d - self.b**2))

    @property
    def determinant(self) -> float:
        """Calculate the determinant of the ellipse."""
        return self.a * self.d - self.b**2

    @property
    def positive_semi_definite(self) -> bool:
        """Check if the ellipse is positive semi-definite."""
        return self.descriminant >= 0

    @property
    def uprightness(self) -> float:
        """Calculate the uprightness of the ellipse (Eq. 32, arXiv:1403.2975)."""
        return math.pi / (4 * self.e)

    def b_from_uprightness(self, uprightness: float) -> float:
        """Calculate the b value of the ellipse from its uprightness (Eq. 33, arXiv:1403.2975)."""
        return math.sqrt((math.pi / (4 * uprightness)) ** 2 - 1)

    def contains(self, x: float, y: float) -> bool:
        """Check if the point (x, y) is inside the ellipse."""
        return (self.a * x**2 + 2 * self.b * x * y + self.d * y**2) <= 1

    def normalize(self) -> tuple["Ellipse", float]:
        """Normalize the ellipse to have a determinant of 1."""
        s_val = math.sqrt(self.determinant)
        return self.scale(scale=s_val), s_val

    def scale(self, scale: float) -> "Ellipse":
        """Scale the ellipse by a factor of scale."""
        D = (self.a * scale, self.b * scale, self.d * scale)
        axes = (self.axes[0] * math.sqrt(scale), self.axes[1] * math.sqrt(scale))
        return Ellipse(D, self.p, axes)

    def y_points(self, x: float) -> tuple[float, float] | None:
        """Compute the y-points of the ellipse for a given x-value."""
        x -= self.p[0]  # shift x to the origin
        descriminant = (self.b * x) ** 2 - self.d * (self.a * x**2 - 1)
        if descriminant < 0:
            return None

        y0, d0 = self.p[1], math.sqrt(descriminant)
        y1 = (-self.b * x - d0) / self.d
        y2 = (-self.b * x + d0) / self.d
        return y0 + y1, y0 + y2

    def x_points(self, y: float) -> tuple[float, float]:
        """Compute the x-points of the ellipse for a given y-value."""
        y -= self.p[1]  # shift y to the origin
        descriminant = y**2 * (self.b**2 - self.a * self.d) + self.a
        if descriminant < 0:
            return None

        x0, d0 = self.p[0], math.sqrt(descriminant)
        x1 = (-self.b * y - d0) / self.a
        x2 = (-self.b * y + d0) / self.a
        return x0 + x1, x0 + x2

    def bounding_box(self) -> tuple[float, float, float, float]:
        """Return the bounding box of the ellipse in the form [[x0, x1], [y0, y1]]."""
        denom = self.determinant
        x, y = math.sqrt((self.d / denom)), math.sqrt((self.a / denom))
        return (-x, x, -y, y)

    def apply_grid_op(self, grid_op: GridOp) -> "Ellipse":
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


class State:
    """A class representing pair of normalized ellipses."""

    def __init__(self, e1: Ellipse, e2: Ellipse):
        self.e1 = e1
        self.e2 = e2

    def __repr__(self) -> str:
        """Return a string representation of the state."""
        return f"State(e1={self.e1}, e2={self.e2})"

    @property
    def skew(self) -> float:
        """Calculate the skew of the state."""
        # Uses Definition A.1 of arXiv:1403.2975
        return self.e1.b**2 + self.e2.b**2

    @property
    def bias(self) -> float:
        """Calculate the bias of the state."""
        # Uses Definition A.1 of arXiv:1403.2975
        return self.e1.z - self.e2.z

    def grid_op(self) -> GridOp:
        """Calculate the grid operation of the state."""
        # Uses Lemma A.5 (Step Lemma) of arXiv:1403.2975 for obtaining special grid op.
        grid_op = GridOp.from_string("I")
        state = State(self.e1, self.e2)
        while (skew := state.skew) >= 15:
            new_grid_op, state = state.reduce_skew()
            grid_op = grid_op * new_grid_op
            if state.skew > 0.9 * skew:
                raise ValueError(f"Skew was not decreased for state {state}")

        return grid_op

    def apply_grid_op(self, grid_op: GridOp) -> State:
        """Apply a grid operation :math:`G` to the state."""
        # Uses Definition A.3 of arXiv:1403.2975
        return State(self.e1.apply_grid_op(grid_op), self.e2.apply_grid_op(grid_op.adj2()))

    def apply_shift_op(self, k: int) -> tuple[State, int]:
        """Apply a shift operator to the state."""
        # Uses Definition A.6 of arXiv:1403.2975
        k = int(math.floor((1 - self.bias()) / 2))
        pk_pow, nk_pow = _LAMBDA**k, _LAMBDA**-k
        e1, e2 = copy(self.e1), copy(self.e2)
        e1.a, e1.d, e1.z = e1.a * pk_pow, e1.d * nk_pow, e1.z - k
        e2.a, e2.d, e2.z = e2.a * nk_pow, e2.d * pk_pow, e2.z + k
        e2.b *= (-1) ** k
        # TODO: Update the e values
        return State(e1, e2), k

    # pylint: disable=too-many-branches
    def reduce_skew(self) -> tuple[GridOp, State]:
        """Reduce the skew of the state.

        This uses Step Lemma described in Appendix A.6 of arXiv:1403.2975.

        Returns:
            tuple[GridOp, State]: A tuple containing the grid operation and the state with reduced skew.
        """
        if any(not e.positive_semi_definite for e in (self.e1, self.e2)):
            raise ValueError("Ellipse is not positive semi-definite")

        k = 0
        state = copy(self)
        grid_op = GridOp.from_string("I")
        grid_op_z, grid_op_x = GridOp.from_string("Z"), GridOp.from_string("X")

        if abs(state.bias) > 1:
            state, k = state.apply_shift_op(k)

        if state.e2.b < 0:
            grid_op = grid_op * grid_op_z

        if (state.e1.z + state.e2.z) < 0:
            grid_op = grid_op * grid_op_x

        state = state.apply_grid_op(grid_op)
        e1, e2 = state.e1, state.e2

        if e1.b >= 0:
            if e1.z >= -0.8 and e1.z <= 0.8 and e2.z >= -0.8 and e2.z <= 0.8:
                grid_op = grid_op * GridOp.from_string("R")
            elif e1.z <= 0.3 and e2.z >= 0.8:
                grid_op = grid_op * GridOp.from_string("K")
            elif e1.z >= 0.3 and e2.z >= 0.3:
                c = min(e1.z, e2.z)
                n = int(max(1, math.floor((_LAMBDA**c) / 2)))
                grid_op = grid_op * (GridOp.from_string("A") ** n)
            elif e1.z >= 0.8 and e2.z <= 0.3:
                grid_op = grid_op * GridOp.from_string("K").adj2()
            else:
                raise ValueError(f"Skew couldn't be reduced for the state {state}")
        else:
            if e1.z >= -0.8 and e1.z <= 0.8 and e2.z >= -0.8 and e2.z <= 0.8:
                grid_op = grid_op * GridOp.from_string("R")
            elif e1.z >= -0.2 and e2.z >= -0.2:
                c = min(e1.z, e2.z)
                n = int(max(1, math.floor((_LAMBDA**c) / 2)))
                grid_op = grid_op * (GridOp.from_string("B") ** n)
            else:
                raise ValueError(f"Skew couldn't be reduced for the state {state}")

        if k != 0:
            grid_op = grid_op.apply_shift_op(k)

        return grid_op, state.apply_grid_op(grid_op)


class GridOp:
    """A class representing a grid operation on a 2D grid."""

    def __init__(
        self,
        a: tuple[float, float],
        b: tuple[float, float],
        c: tuple[float, float],
        d: tuple[float, float],
    ) -> None:
        """Initialize the grid operation."""
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def __repr__(self) -> str:
        """Return a string representation of the grid operation."""
        return f"GridOp(a={self.a}, b={self.b}, c={self.c}, d={self.d})"

    def __eq__(self, other: "GridOp") -> bool:
        """Check if the grid operations are equal."""
        return self.a == other.a and self.b == other.b and self.c == other.c and self.d == other.d

    def __mul__(self, other: "GridOp" | "ZOmega") -> "GridOp" | "ZOmega":
        if isinstance(other, GridOp):
            return self._mul_grid_op(other)
        if isinstance(other, ZOmega):
            return self._mul_z_omega(other)
        raise TypeError(f"Cannot multiply GridOp with {type(other)}")

    def __pow__(self, n: int) -> "GridOp":
        """Raise the grid operator to a power."""
        if self == self.from_string("I"):  # Identity:
            return self
        if self == self.from_string("A"):  # A
            return GridOp((1, 0), (-2 * n, 0), (0, 0), (1, 0))
        if self == self.from_string("B"):  # B
            return GridOp((1, 0), (0, 2 * n), (0, 0), (1, 0))

        x, res = self, GridOp((1, 0), (0, 0), (0, 0), (1, 0))
        while n > 0:
            if n % 2 == 1:
                res = res * x
            x = x * x
            n //= 2
        return res

    def _mul_grid_op(self, other: "GridOp") -> "GridOp":
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

    def _mul_z_omega(self, other: "ZOmega") -> "ZOmega":
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
            string: Supported string are: "I", "R", "A", "B", "K", "X", "Z" (Fig 6, arXiv:1403.2975).

        Returns:
            GridOp: The grid operation corresponding to the string.
        """
        return _useful_grid_ops().get(string, "I")

    @property
    def determinant(self) -> tuple[float, float]:
        """Calculate the determinant of the grid operation."""
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
        """Check if the grid operation is special."""
        det1, det2 = self.determinant
        return det2 == 0 and det1 in {1, -1}

    @property
    def flatten(self) -> list[float]:
        """Flatten the grid operation."""
        return [
            self.a[0] + self.a[1] / _SQRT2,
            self.b[0] + self.b[1] / _SQRT2,
            self.c[0] + self.c[1] / _SQRT2,
            self.d[0] + self.d[1] / _SQRT2,
        ]

    def inverse(self) -> "GridOp":
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

    def transpose(self) -> "GridOp":
        """Transpose the grid operation."""
        return GridOp(
            self.d,
            self.b,
            self.c,
            self.a,
        )

    def adj2(self) -> "GridOp":
        """Compute the sqrt(2)-conjugate of the grid operation."""
        return GridOp(
            (self.a[0], -self.a[1]),
            (self.b[0], -self.b[1]),
            (self.c[0], -self.c[1]),
            (self.d[0], -self.d[1]),
        )

    def apply_to_ellipse(self, ellipse: Ellipse) -> Ellipse:
        """Apply the grid operator to an ellipse."""
        return ellipse.apply_grid_op(self)

    def apply_to_state(self, state: State) -> tuple[Ellipse, Ellipse]:
        """Apply the grid operator to a state."""
        return state.apply_grid_op(self)

    def apply_shift_op(self, k: int) -> "GridOp":
        """Apply a shift operator to the grid operation."""
        # Uses Lemma A.9 of arXiv:1403.2975
        k, sign = abs(k), (-1) ** (k < 0)  # +1 if k > 0, -1 if k < 0
        grid_op = self
        for _ in range(k):
            grid_op = GridOp(
                (grid_op.a[0] + sign * grid_op.a[1], 2 * grid_op.a[0] + sign * grid_op.a[1]),
                grid_op.b,
                grid_op.c,
                (grid_op.d[1] - sign * grid_op.d[0], 2 * grid_op.d[0] - sign * grid_op.d[1]),
            )
        return grid_op


@lru_cache(maxsize=1)
def _useful_grid_ops() -> dict["str", "GridOp"]:
    """Generate a list of useful grid operations (Fig 6, arXiv:1403.2975)."""
    return {
        "I": GridOp((1, 0), (0, 0), (0, 0), (1, 0)),
        "R": GridOp((0, 1), (0, -1), (0, 1), (0, 1)),
        "A": GridOp((1, 0), (-2, 0), (0, 0), (1, 0)),
        "B": GridOp((1, 0), (0, 2), (0, 0), (1, 0)),
        "K": GridOp((-1, 1), (0, -1), (1, 1), (0, 1)),
        "X": GridOp((0, 0), (1, 0), (1, 0), (0, 0)),
        "Z": GridOp((1, 0), (0, 0), (0, 0), (-1, 0)),
    }
