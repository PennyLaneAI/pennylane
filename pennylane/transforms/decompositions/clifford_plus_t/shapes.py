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
"""Various shapes for the gridsynth implementation."""

import numpy as np

from .conversion import op_from_d_root_two


class Point:
    """A pair representing a coordinate in the Cartesian plane."""

    def __init__(self, x, y):
        """Turn x and y into a typed Point."""
        self.x = x
        self.y = y

    def __iter__(self):
        """So that we can unpack the x and y values."""
        return iter((self.x, self.y))

    def transform(self, opG):
        """Apply an operator to a point."""
        (a, b), (c, d) = opG
        x1, y1 = self
        x2 = a * x1 + b * y1
        y2 = c * x1 + d * y1
        return Point(x2, y2)


class Ellipse:
    """An ellipse object, defined by an operator and a centre."""

    def __init__(self, operator, point: Point):
        """Define an ellipse by its operator and centre."""
        self.operator = operator
        self.point = point

    def transform(self, opG):
        """Apply an operator to an ellipse."""
        opG_inv = opG.inverse()
        mat = np.conj(opG_inv).T @ self.operator @ opG_inv
        return Ellipse(mat, self.point.transform(opG))

    def bounding_box(self):
        """Construct the bounding box corners around the ellipse."""
        x, y = self.point
        a, d = self.operator.diagonal()
        sqrt_det = np.sqrt(np.linalg.det(self.operator))
        w = np.sqrt(d) / sqrt_det
        h = np.sqrt(a) / sqrt_det
        return (x - w, x + w), (y - h, y + h)


class ConvexSet:
    """A convex set."""

    def __init__(self, ellipse: Ellipse, characteristic_fn, line_intersector):
        """Construct a convex set with its components."""
        self.ellipse = ellipse
        self.characteristic_fn = characteristic_fn
        self.line_intersector = line_intersector

    def characteristic_transform(self, opG):
        """Returns a new characteristic function that first applies an operator to the point."""
        opG_inv = opG.inverse()

        def char_fn(point: Point):
            """The transformed characteristic function."""
            return self.characteristic_fn(point.transform(opG_inv))

        return char_fn

    def intersector_transform(self, opG):
        """Returns a new line intersector that first applies an operator to the point."""
        opG_inv = opG.inverse()

        def intersector(v: Point, w: Point):
            """The transformed line intersector."""
            return self.line_intersector(v.transform(opG_inv), w.transform(opG_inv))

        return intersector

    def transform(self, opG) -> "ConvexSet":
        """Returns a new convex set with transforms applied to all components."""
        opG_from_d_root_two = op_from_d_root_two(opG)
        ell = self.ellipse.transform(opG_from_d_root_two)
        char_fn = self.characteristic_transform(opG)
        intersector = self.intersector_transform(opG_from_d_root_two)
        return ConvexSet(ell, char_fn, intersector)
