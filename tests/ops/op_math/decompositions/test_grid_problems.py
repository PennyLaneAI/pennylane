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
"""Tests for the grid problems for Clifford+T decomposition."""

# pylint: disable = too-few-public-methods

import inspect
import math

import pytest

from pennylane import math as pmath
from pennylane.ops.op_math.decompositions.grid_problems import (
    Ellipse,
    EllipseState,
    GridIterator,
    GridOp,
)
from pennylane.ops.op_math.decompositions.rings import ZOmega


class TestEllipse:
    """Tests for the Ellipse class."""

    def test_init_and_repr(self):
        """Test that Ellipse initializes correctly and has correct representations."""
        ellipse = Ellipse((1, 0, 1), (4, 5))
        assert ellipse.a == 1 and ellipse.b == 0 and ellipse.d == 1
        assert ellipse.p == (4, 5)
        assert repr(ellipse) == "Ellipse(a=1, b=0, d=1, p=(4, 5))"
        assert str(ellipse) == "Ellipse(a=1, b=0, d=1, p=(4, 5))"
        assert ellipse.determinant == 1 and ellipse.discriminant == 0
        assert ellipse.bounding_box() == (-1, 1, -1, 1)
        assert ellipse.offset(1) == Ellipse((1, 0, 1), p=(5, 6))
        assert ellipse.b_from_uprightness(0.2) == math.sqrt((math.pi / 0.8) ** 2 - 1)
        assert ellipse.positive_semi_definite
        assert ellipse.x_points(4) == (4.0, 4.0)
        assert ellipse.y_points(5) == (5.0, 5.0)
        assert pmath.allclose(ellipse.uprightness, 0.7853981633974483)

    def test_raise_error(self):
        """Test that Ellipse raises an error when the ellipse is not valid."""
        ellipse = Ellipse((1, 0, 1), (4, 5))
        with pytest.raises(ValueError, match="is outside the ellipse"):
            ellipse.x_points(24)
        with pytest.raises(ValueError, match="is outside the ellipse"):
            ellipse.y_points(24)


class TestEllipseState:
    """Tests for the EllipseState class."""

    def test_init_and_repr(self):
        """Test that EllipseState initializes correctly and has correct representations."""
        e1 = Ellipse((1, 0, 1), (4, 5))
        e2 = Ellipse((2, 1, 2), (2, 3))
        state = EllipseState(e1, e2)
        assert state.e1 == e1
        assert state.e2 == e2
        assert repr(state) == f"EllipseState(e1={e1}, e2={e2})"
        assert state.skew == 1.0 and state.bias == 0.0


class TestGridOp:
    """Tests for the GridOp class."""

    def test_init_and_repr(self):
        """Test that GridOp initializes correctly and has correct representations."""
        grid_op = GridOp.from_string("I")
        assert repr(grid_op) == "GridOp(a=(1, 0), b=(0, 0), c=(0, 0), d=(1, 0))"
        assert grid_op**3 == grid_op
        assert GridOp.from_string("R") ** 3 == GridOp(a=(0, -1), b=(0, -1), c=(0, 1), d=(0, -1))
        assert GridOp.from_string("K").is_special
        assert GridOp(a=(0, 1), b=(0, -1), c=(1, 1), d=(-1, 1)).transpose() == GridOp(
            a=(-1, 1), b=(0, -1), c=(1, 1), d=(0, 1)
        )

        grid_op = GridOp.from_string("B")
        e1 = Ellipse((1, 0, 1), (4, 5))
        e2 = Ellipse((2, 1, 2), (2, 3))
        state = EllipseState(e1, e2)
        state1 = grid_op.apply_to_state(state)
        state2 = grid_op.inverse().apply_to_state(state1)
        assert state.e1 == state2.e1 and state.e2 == state2.e2
        assert state1.e1 == grid_op.apply_to_ellipse(e1)

    def test_raise_error(self):
        """Test that GridOp raises an error when an operation is not permitted."""
        with pytest.raises(TypeError, match="Cannot multiply GridOp with"):
            _ = GridOp.from_string("B") * 3

        with pytest.raises(ValueError, match="Grid operator needs to be special"):
            _ = GridOp(a=(1, 1), b=(-2, -1), c=(1, 1), d=(-1, 4), check_valid=False).inverse()


class TestGridIterator:
    """Tests for the GridIterator class."""

    # pylint: disable = too-many-arguments
    @pytest.mark.parametrize(
        "x0, x1, y0, y1, num",
        [
            (8.9, 9.5, -21, -18, 2),
            (246.023423, 248.5823575862261, 778, 779.0106829464769, 3),
            (13734300, 13734500, -13874089.232, -13874089.181, 6),
        ],
    )
    def test_one_dim_problem(self, x0, x1, y0, y1, num):
        """Test that the one dimensional grid problem is solved correctly."""
        gitr = GridIterator()
        sols = gitr.solve_one_dim_problem(x0, x1, y0, y1)
        assert inspect.isgenerator(sols)
        assert gitr.bbox_grid_points((x0, x1, y0, y1)) == num

        ix = 0
        for sol in sols:
            ix, s1, s2 = ix + 1, float(sol), float(sol.adj2())
            assert x0 <= s1 <= x1
            assert y0 <= s2 <= y1
        assert 0 < ix <= num

    @pytest.mark.parametrize(
        "bbox1, bbox2, res",
        [
            ((5, 6, 4, 5), (2, 3, -1, 0), (1, 2, 3, 4)),
            ((-4, -3.8, 2.2, 2.4), (-9, -8, 2.5, 4.2), (-2, 3, 1, -6)),
        ],
    )
    def test_upright_problem(self, bbox1, bbox2, res):
        """Test that the upright grid problem is solved correctly."""
        D, num_b, shifts = (1, 0, 1), [0, 0], [ZOmega(), ZOmega(c=1)]
        bbox3 = tuple(bb_ - 1 / math.sqrt(2) for bb_ in bbox1)
        bbox4 = tuple(bb_ + 1 / math.sqrt(2) for bb_ in bbox2)
        state = EllipseState(Ellipse(D), Ellipse(D))

        gitr = GridIterator()
        sols1 = gitr.solve_upright_problem(state, bbox1, bbox2, num_b, shifts[0])
        sols2 = gitr.solve_upright_problem(state, bbox3, bbox4, num_b, shifts[1])
        assert inspect.isgenerator(sols1) and inspect.isgenerator(sols2)
        assert ZOmega(*res) in list(sols1) + list(sols2)

    @pytest.mark.parametrize(
        "e1, e2, res",
        [
            (
                Ellipse((3, 0.5, 1.0), (-1.7, 13.95)),
                Ellipse((3, 0.3, 0.3), (-12.3, -7.9)),
                (4, 3, 12, -7),
            ),
            (
                Ellipse((2, 0.6, 0.9), (-1.8, 14.93)),
                Ellipse((2, 0.4, 0.2), (-11.3, -6.9)),
                (4, 4, 11, -6),
            ),
        ],
    )
    def test_two_dim_problem(self, e1, e2, res):
        """Test that the two dimensional grid problem is solved correctly."""
        state = EllipseState(e1, e2)
        gitr = GridIterator()
        sols = gitr.solve_two_dim_problem(state)
        assert inspect.isgenerator(sols)
        assert ZOmega(*res) in list(sols)

    @pytest.mark.parametrize(
        "theta, epsilon",
        [
            (0.0, 1e-3),
            (math.pi / 4, 1e-4),
            (math.pi / 8, 1e-4),
            (math.pi / 3, 1e-3),
            (math.pi / 5, 1e-5),
            (0.392125483789636, 1e-6),
            (0.6789684841313233, 1e-3),
            (0.056202026824044335, 1e-5),
            (0.21375826964510297, 1e-4),
            (0.5549739238125396, 1e-6),
            (-0.454645364564563, 1e-3),
            (-0.5549739238125396, 1e-2),
        ],
    )
    def test_grid_iterator(self, theta, epsilon):
        """Test that the two dimensional grid problem is solved correctly."""
        grid_sols = GridIterator(theta=theta, epsilon=epsilon)
        assert repr(grid_sols) == f"GridIterator(theta={theta}, epsilon={epsilon}, max_trials=20)"
        assert hasattr(grid_sols, "__iter__")
        u_sol, k_sol = next(iter(grid_sols))
        assert u_sol is not None
        u_sol_complex = complex(u_sol) / math.sqrt(2) ** k_sol
        assert pmath.allclose(u_sol_complex.real, pmath.cos(theta), atol=epsilon)
        assert pmath.allclose(u_sol_complex.imag, pmath.sin(theta), atol=epsilon)
