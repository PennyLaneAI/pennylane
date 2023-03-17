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

import pytest

from pennylane.measurements import Shots, ShotCopies


class TestShots:
    """Tests the Shots class."""

    def test_None_construction(self):
        """Tests the constructor when shots is None."""
        shots = Shots(None)
        assert shots.shot_list == []
        assert shots.shot_vector == []
        assert shots.total_shots == 0

    def test_int_construction(self):
        """Tests the constructor when shots is an int."""
        shots = Shots(100)
        assert shots.shot_list == [100]
        assert shots.shot_vector == [ShotCopies(100, 1)]
        assert shots.total_shots == 100

    @pytest.mark.parametrize(
        "shot_list,expected",
        [
            (
                [1, 3, 3, 4, 4, 4, 3],
                [ShotCopies(1, 1), ShotCopies(3, 2), ShotCopies(4, 3), ShotCopies(3, 1)],
            ),
            ([5, 5, 5], [ShotCopies(5, 3)]),
        ],
    )
    def test_sequence_construction(self, shot_list, expected):
        """Tests the constructor when shots is a Sequence[int]."""
        shots = Shots(shot_list)
        assert shots.shot_list == shot_list
        assert shots.shot_vector == expected
        assert shots.total_shots == sum(shot_list)

    @pytest.mark.parametrize("shot_arg", ["123", [1.1, 2], [-1, 2], [1, (4, 2)], 1.5])
    def test_other_construction_fails(self, shot_arg):
        """Tests that all other values for shots is not allowed."""
        with pytest.raises(
            ValueError,
            match="Shots must be a single non-negative integer or a sequence of non-negative integers.",
        ):
            _ = Shots(shot_arg)

    def test_zero_shots_fails(self):
        with pytest.raises(
            ValueError, match="The specified number of shots needs to be at least 1. Got 0."
        ):
            _ = Shots(0)

    def test_Shots_frozen_after_init(self):
        """Tests that Shots instances are frozen after creation."""
        shots = Shots(10)
        with pytest.raises(AttributeError, match="Shots is an immutable class"):
            shots.total_shots = 20

    @pytest.mark.parametrize("shots,expected", [(100, False), ([1, 2], True), [[100], False]])
    def test_has_partitioned_shots(self, shots, expected):
        """Tests the has_partitioned_shots method."""
        assert Shots(shots).has_partitioned_shots is expected
