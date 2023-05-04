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
"""
Unit tests for :mod:`pennylane.shots`.
"""

import copy
import pytest

from pennylane.measurements import Shots, ShotCopies

ERROR_MSG = "Shots must be a single positive integer, a tuple"


class TestShotsConstruction:
    """Tests the Shots class."""

    def test_copy(self):
        """Tests that creating a Shots from another Shots instance returns the same instance."""
        x = Shots(123)
        y = Shots(x)
        assert y is x
        assert y._frozen  # pylint:disable=protected-access

        z = copy.copy(x)
        assert z is x
        assert z._frozen  # pylint:disable=protected-access

    def test_deepcopy(self):
        x = Shots([1, 1, 2, 3])
        y = copy.deepcopy(x)
        assert y is x
        assert y._frozen  # pylint:disable=protected-access

    def test_None(self):
        """Tests the constructor when shots is None."""
        shots = Shots(None)
        assert shots.shot_vector == ()
        assert shots.total_shots is None

    def test_int(self):
        """Tests the constructor when shots is an int."""
        shots = Shots(100)
        assert shots.shot_vector == (ShotCopies(100, 1),)
        assert shots.total_shots == 100

    def test_tuple(self):
        """Tests the constructor when shots is a tuple."""
        shots = Shots((5, 6))
        assert shots.shot_vector == (ShotCopies(5, 1), ShotCopies(6, 1))
        assert shots.total_shots == 11
        assert isinstance(shots.shot_vector, tuple)

    def test_sequence_all_tuple(self):
        """Tests that a sequence of tuples is allowed."""
        shots = Shots([(1, 2), (1, 5), (3, 4)])
        assert shots.shot_vector == (ShotCopies(1, 7), ShotCopies(3, 4))
        assert shots.total_shots == 19
        assert isinstance(shots.shot_vector, tuple)

    @pytest.mark.parametrize(
        "shot_list,expected,total",
        [
            (
                [1, 3, 3, 4, 4, 4, 3],
                (ShotCopies(1, 1), ShotCopies(3, 2), ShotCopies(4, 3), ShotCopies(3, 1)),
                22,
            ),
            ([5, 5, 5], (ShotCopies(5, 3),), 15),
            ([1, (4, 2)], (ShotCopies(1, 1), ShotCopies(4, 2)), 9),
            ((5,), (ShotCopies(5, 1),), 5),
            ((5, 6, 7), (ShotCopies(5, 1), ShotCopies(6, 1), ShotCopies(7, 1)), 18),
            (((5, 6)), (ShotCopies(5, 1), ShotCopies(6, 1)), 11),
            (((5, 6),), (ShotCopies(5, 6),), 30),
            (((5, 6), 7), (ShotCopies(5, 6), ShotCopies(7, 1)), 37),
            ((5, (6, 7)), (ShotCopies(5, 1), ShotCopies(6, 7)), 47),
            (((5, 6), (7, 8)), (ShotCopies(5, 6), ShotCopies(7, 8)), 86),
        ],
    )
    def test_sequence(self, shot_list, expected, total):
        """Tests the constructor when shots is a Sequence[int]."""
        shots = Shots(shot_list)
        assert shots.shot_vector == expected
        assert shots.total_shots == total

    @pytest.mark.parametrize("shot_arg", ["123", [1.1, 2], [-1, 2], 1.5, (1.1, 2)])
    def test_other_fails(self, shot_arg):
        """Tests that all other values for shots is not allowed."""
        with pytest.raises(ValueError, match=ERROR_MSG):
            _ = Shots(shot_arg)

    def test_zero_shots_fails(self):
        with pytest.raises(ValueError, match=ERROR_MSG):
            _ = Shots(0)

    def test_Shots_frozen_after_init(self):
        """Tests that Shots instances are frozen after creation."""
        shots = Shots(10)
        with pytest.raises(AttributeError, match="Shots is an immutable class"):
            shots.total_shots = 20

    @pytest.mark.parametrize(
        "shots,expected", [(None, False), (100, False), ([1, 2], True), [[100], False]]
    )
    def test_has_partitioned_shots(self, shots, expected):
        """Tests the has_partitioned_shots method."""
        assert Shots(shots).has_partitioned_shots is expected
