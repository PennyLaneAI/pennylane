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

from pennylane.shots import ShotAPI, ShotTuple


class TestShotAPI:
    """Tests the ShotAPI class."""

    def test_None_construction(self):
        """Tests the constructor when shots is None."""
        shots = ShotAPI(None)
        assert shots.shot_list == []
        assert shots.shot_vector == []
        assert shots.total_shots == 0

    def test_int_construction(self):
        """Tests the constructor when shots is an int."""
        shots = ShotAPI(100)
        assert shots.shot_list == [100]
        assert shots.shot_vector == [ShotTuple(100, 1)]
        assert shots.total_shots == 100

    @pytest.mark.parametrize(
        "shot_list,expected",
        [
            (
                [1, 3, 3, 4, 4, 4, 3],
                [ShotTuple(1, 1), ShotTuple(3, 2), ShotTuple(4, 3), ShotTuple(3, 1)],
            ),
            ([5, 5, 5], [ShotTuple(5, 3)]),
        ],
    )
    def test_sequence_construction(self, shot_list, expected):
        """Tests the constructor when shots is a Sequence[int]."""
        shots = ShotAPI(shot_list)
        assert shots.shot_list == shot_list
        assert shots.shot_vector == expected
        assert shots.total_shots == sum(shot_list)

    @pytest.mark.parametrize("shot_arg", ["123", [1.1, 2], [1, (4, 2)]])
    def test_other_construction_fails(self, shot_arg):
        """Tests that all other values for shots is not allowed."""
        with pytest.raises(
            ValueError,
            match="Shots must be a single non-negative integer or a sequence of non-negative integers.",
        ):
            _ = ShotAPI(shot_arg)

    def test_zero_shots_fails(self):
        with pytest.raises(
            ValueError, match="The specified number of shots needs to be at least 1. Got 0."
        ):
            _ = ShotAPI(0)
