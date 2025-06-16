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
"""Tests for the grid problems for Clifford+T decomposition."""

# pylint: disable = too-few-public-methods, unused-variable, unused-import

import pytest

from pennylane.ops.op_math.decompositions.grid_problems import Ellipse, GridIterator, GridOp, State
from pennylane.ops.op_math.decompositions.rings import ZOmega, ZSqrtTwo


class TestEllipse:
    """Tests for the Ellipse class."""

    def test_init_and_repr(self):
        """Test that Ellipse initializes correctly and has correct representations."""
        ellipse = Ellipse((1, 2, 3), (4, 5), (6, 7))
        assert ellipse.a == 1
        assert ellipse.b == 2
        assert ellipse.d == 3
        assert ellipse.p == (4, 5)
        assert ellipse.axes == (6, 7)
        assert repr(ellipse) == "Ellipse(a=1, b=2, d=3, p=(4, 5), axes=(6, 7))"
        assert str(ellipse) == "Ellipse(a=1, b=2, d=3, p=(4, 5), axes=(6, 7))"
        assert ellipse.determinant == 1 * 2 * 3 - 2 * 2 * 2
        assert ellipse.descriminant == 1 * 2 * 3 - 2 * 2 * 2
        assert ellipse.bounding_box() == (-1, 1, -1, 1)
        assert ellipse.offset(1) == Ellipse((1, 2, 3), (5, 6), (6, 7))
        assert ellipse.apply_grid_op(GridOp((1, 2, 3, 4), (5, 6, 7, 8))) == Ellipse(
            (1, 2, 3), (5, 6), (6, 7)
        )
        assert ellipse.apply_grid_op(GridOp((1, 2, 3, 4), (5, 6, 7, 8))) == Ellipse(
            (1, 2, 3), (5, 6), (6, 7)
        )


class TestState:
    """Tests for the State class."""

    def test_init_and_repr(self):
        """Test that State initializes correctly and has correct representations."""
        e1 = Ellipse((1, 2, 3), (4, 5), (6, 7))
        e2 = Ellipse((8, 9, 10), (11, 12), (13, 14))
        state = State(e1, e2)
        assert state.e1 == e1
        assert state.e2 == e2
        assert (
            repr(state)
            == "State(e1=Ellipse(a=1, b=2, d=3, p=(4, 5), axes=(6, 7)), e2=Ellipse(a=8, b=9, d=10, p=(11, 12), axes=(13, 14)))"
        )
        assert (
            str(state)
            == "State(e1=Ellipse(a=1, b=2, d=3, p=(4, 5), axes=(6, 7)), e2=Ellipse(a=8, b=9, d=10, p=(11, 12), axes=(13, 14)))"
        )


class TestGridOp:
    """Tests for the GridOp class."""

    def test_init_and_repr(self):
        """Test that GridOp initializes correctly and has correct representations."""
        grid_op = GridOp((1, 2, 3, 4), (5, 6, 7, 8))
        assert grid_op.a == 1
        assert grid_op.b == 2
        assert grid_op.c == 3
        assert grid_op.d == 4

    def test_grid_op_apply(self):
        """Test that GridOp applies correctly."""
        grid_op = GridOp((1, 2, 3, 4), (5, 6, 7, 8))
        state = State(Ellipse((1, 2, 3), (4, 5), (6, 7)), Ellipse((8, 9, 10), (11, 12), (13, 14)))
        assert grid_op.apply(state) == State(
            Ellipse((1, 2, 3), (5, 6), (6, 7)), Ellipse((8, 9, 10), (12, 13), (13, 14))
        )


class TestGridIterator:
    """Tests for the GridIterator class."""

    def test_one_dim_problem(self):
        """Test that the one dimensional grid problem is solved correctly."""
        grid_iterator = GridIterator(0.1, 0.2)
        state = State(Ellipse((1, 2, 3), (4, 5), (6, 7)), Ellipse((8, 9, 10), (11, 12), (13, 14)))
        solutions = grid_iterator.solve_one_dim_problem(0, 1, 0, 1)
        assert len(solutions) == 1
        assert solutions[0] == ZSqrtTwo(1, 0)

    def test_upright_problem(self):
        """Test that the upright grid problem is solved correctly."""
        grid_iterator = GridIterator(0.1, 0.2)
        state = State(Ellipse((1, 2, 3), (4, 5), (6, 7)), Ellipse((8, 9, 10), (11, 12), (13, 14)))
        solutions = grid_iterator.solve_upright_problem(
            state, (0, 1, 0, 1), (0, 1, 0, 1), True, ZOmega()
        )
        assert len(solutions) == 1
        assert solutions[0] == ZOmega(1, 0, 0, 0)

    def test_two_dim_problem(self):
        """Test that the two dimensional grid problem is solved correctly."""
        grid_iterator = GridIterator(0.1, 0.2)
        state = State(Ellipse((1, 2, 3), (4, 5), (6, 7)), Ellipse((8, 9, 10), (11, 12), (13, 14)))
        solutions = grid_iterator.solve_two_dim_problem(state)
        assert len(solutions) == 1
        assert solutions[0] == ZOmega(1, 0, 0, 0)
