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

    def skew(self, ellipse: "Ellipse") -> float:
        """Calculate the skew of the ellipse."""
        return self.b**2 + ellipse.b**2

    def bias(self, ellipse: "Ellipse") -> float:
        """Calculate the bias of the ellipse."""
        return self.z - ellipse.z

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

    def apply_shift_op(self, ellipse: "Ellipse") -> int:
        """Apply shift operators for the pair, returning the shift exponent."""
        k = int(math.floor((1 - self.bias(ellipse)) / 2))
        pk_pow, nk_pow = _LAMBDA**k, _LAMBDA**-k
        self.a, self.d, self.z = self.a * pk_pow, self.d * nk_pow, self.z - k
        ellipse.a, ellipse.d, ellipse.z = ellipse.a * nk_pow, ellipse.d * pk_pow, ellipse.z + k
        ellipse.b *= (-1) ** k
        return k


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
        if self == GridOp((1, 0), (0, 0), (0, 0), (1, 0)):  # Identity:
            return self
        if self == GridOp((1, 0), (-2, 0), (0, 0), (1, 0)):  # _useful_grid_ops()["A"]
            return GridOp((1, 0), (-2 * n, 0), (0, 0), (1, 0))
        if self == GridOp((1, 0), (0, 2), (0, 0), (1, 0)):  # _useful_grid_ops()["B"]
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

    def apply_to_ellipses(self, ellipse1: Ellipse, ellipse2: Ellipse) -> tuple[Ellipse, Ellipse]:
        """Apply the grid operator to pairt of ellipses."""
        return self.apply_to_ellipse(ellipse1), self.adj2().apply_to_ellipse(ellipse2)


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
