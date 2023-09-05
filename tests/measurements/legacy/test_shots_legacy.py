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


class TestShotCopies:
    """Test that the ShotCopies class displays well."""

    sc_data = (ShotCopies(1, 1), ShotCopies(100, 1), ShotCopies(100, 2), ShotCopies(10, 100))

    str_data = (
        "1 shots",
        "100 shots",
        "100 shots x 2",
        "10 shots x 100",
    )

    @pytest.mark.parametrize("expected_str, sc", zip(str_data, sc_data))
    def test_str(self, expected_str, sc):
        """Test the str method works well"""
        assert expected_str == str(sc)

    repr_data = (
        "ShotCopies(1 shots x 1)",
        "ShotCopies(100 shots x 1)",
        "ShotCopies(100 shots x 2)",
        "ShotCopies(10 shots x 100)",
    )

    @pytest.mark.parametrize("expected_str, sc", zip(repr_data, sc_data))
    def test_repr(self, expected_str, sc):
        """Test the repr method works well"""
        assert expected_str == repr(sc)


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
        shots1 = Shots(None)
        shots2 = Shots()  # this also defaults to None
        assert shots1.shot_vector == ()
        assert shots2.shot_vector == ()
        assert shots1.total_shots is None
        assert shots2.total_shots is None

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

    shot_data = (
        Shots(None),
        Shots(10),
        Shots((1, 10, 100)),
        Shots((1, 10, 10, 100, 100, 100)),
    )

    str_data = (
        "Shots(total=None)",
        "Shots(total=10)",
        "Shots(total=111, vector=[1 shots, 10 shots, 100 shots])",
        "Shots(total=321, vector=[1 shots, 10 shots x 2, 100 shots x 3])",
    )

    @pytest.mark.parametrize("expected_str, shots_obj", zip(str_data, shot_data))
    def test_str(self, expected_str, shots_obj):
        """Test that the string representation is correct."""
        assert expected_str == str(shots_obj)

    repr_data = (
        "Shots(total_shots=None, shot_vector=())",
        "Shots(total_shots=10, shot_vector=(ShotCopies(10 shots x 1),))",
        "Shots(total_shots=111, shot_vector=(ShotCopies(1 shots x 1), "
        "ShotCopies(10 shots x 1), ShotCopies(100 shots x 1)))",
        "Shots(total_shots=321, shot_vector=(ShotCopies(1 shots x 1), "
        "ShotCopies(10 shots x 2), ShotCopies(100 shots x 3)))",
    )

    @pytest.mark.parametrize("expected_str, shots_obj", zip(repr_data, shot_data))
    def test_repr(self, expected_str, shots_obj):
        """Test that the repr is correct"""
        assert expected_str == repr(shots_obj)

    def test_eq(self):
        """Test that the equality function behaves correctly"""
        for s in self.shot_data:
            assert s == copy.copy(s)
            assert s == Shots(s.shot_vector if s.shot_vector else None)

    def test_eq_edge_case(self):
        """Test edge cases for equality function are correct"""
        assert Shots((1, 2)) != Shots((2, 1))
        assert Shots((1, 10, 1)) != Shots((1, 1, 10))
        assert Shots((5, 5)) != Shots(10)
        assert Shots((1, 2, (10, 2))) == Shots((1, 2, 10, 10))

    def test_hash(self):
        """Test that the hash function behaves correctly"""
        for s in self.shot_data:
            hash_s = hash(s)
            assert hash_s == hash(copy.copy(s))
            assert hash_s == hash(Shots(s.shot_vector if s.shot_vector else None))

    @pytest.mark.parametrize(
        "shots, expected",
        [
            (100, [100]),
            ([(100, 1)], [100]),
            ([(100, 2)], [100, 100]),
            ([100, 200], [100, 200]),
            ([(100, 2), 200], [100, 100, 200]),
            ([(100, 3), 200, (300, 2)], [100, 100, 100, 200, 300, 300]),
        ],
    )
    def test_iter(self, shots, expected):
        """Test that iteration over Shots works correctly"""
        actual = list(Shots(shots))
        assert actual == expected

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


class TestProperties:
    """Tests various properties of the Shots class."""

    @pytest.mark.parametrize(
        "shots,expected",
        [
            (None, False),
            (1, True),
            ([1, 2], True),
            ([1, (2, 3)], True),
        ],
    )
    def test_bool_dunder(self, shots, expected):
        """Tests the Truthy/Falsy values of various Shots objects."""
        assert bool(Shots(shots)) is expected

    def test_Shots_frozen_after_init(self):
        """Tests that Shots instances are frozen after creation."""
        shots = Shots(10)
        with pytest.raises(AttributeError, match="Shots is an immutable class"):
            shots.total_shots = 20

    @pytest.mark.parametrize(
        "shots,expected", [(None, False), (100, False), ([1, 2], True), [[100], False]]
    )
    def test_has_partitioned_shots(self, shots, expected):
        """Tests the has_partitioned_shots property."""
        assert Shots(shots).has_partitioned_shots is expected

    @pytest.mark.parametrize(
        "shots, expected",
        [
            (None, 0),
            (10, 1),
            ([10, 10], 2),
            ([10, 10, 20], 3),
            ([100, (10, 3)], 4),
            ([(10, 3), (20, 2)], 5),
        ],
    )
    def test_num_copies(self, shots, expected):
        """Tests the num_copies property."""
        assert Shots(shots).num_copies == expected
