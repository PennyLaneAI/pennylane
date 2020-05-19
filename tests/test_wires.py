# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

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
Unit tests for :mod:`pennylane.wires`.
"""
import pytest
import numpy as np

from pennylane.wires import Wires, WireError


class TestWires:
    """Wires class tests."""

    @pytest.mark.parametrize("iterable", [np.array([0, 2, 1, 3]),
                                          [0, 1, 2],
                                          (4, 1, 3),
                                          range(3)])
    def test_common_iterables_as_inputs(self, iterable):
        """Tests that a Wires object can be created from standard iterable inputs."""

        wires = Wires(iterable)
        assert wires.wire_list == list(iterable)

    @pytest.mark.parametrize("index", [1, 0, 4])
    def test_integer_as_inputs(self, index):
        """Tests that a Wires object can be created from integer representing a single wire index."""

        wires = Wires(index)
        assert wires.wire_list == [index]

    @pytest.mark.parametrize("wrong_input", [1.2,
                                             -1,
                                             None])
    def test_error_for_repeated_indices(self, wrong_input):
        """Tests that a Wires object cannot be created from illegal inputs."""

        with pytest.raises(WireError, match="received unexpected wires input"):
            Wires(wrong_input)

    @pytest.mark.parametrize("iterable", [np.array([4, 1, 1, 3]),
                                          [4, 1, 1, 3],
                                          (4, 1, 1, 3)])
    def test_error_for_repeated_indices(self, iterable):
        """Tests that a Wires object cannot have repeated indices."""

        with pytest.raises(WireError, match="Each wire must be represented by a unique index"):
            Wires(iterable)

    @pytest.mark.parametrize("iterable", [np.array([4., 1., 0., 3.]),  # entries are np.int64
                                          [4., 1., 0., 3.]  # entries are floats
                                          ])
    def test_integerlike_indices_converted_to_integers(self, iterable):
        """Tests that a Wires object converts integer-like floats to integer elements."""

        wires = Wires(iterable)
        for w in wires:
            assert isinstance(w, int)

    @pytest.mark.parametrize("iterable", [np.array([4., 1.2, 0., 3.]),  # non-integer-like np.int64
                                          [4., 1., 0., 3.0001],  # non-integer-like floats
                                          ['a', 'b', 'c', 'd']])  # non-integer-like characters
    def test_error_for_non_integerlike_indices(self, iterable):
        """Tests that a Wires object throws error when indices are not integer-like."""

        with pytest.raises(WireError, match="Wire indices must be integers"):
            Wires(iterable)

    @pytest.mark.parametrize("iterable", [np.array([4, 1, 0, 3]),
                                          [4, 1, 0, 3],
                                          (4, 1, 0, 3),
                                          range(4)])
    def test_indexing(self, iterable):
        """Tests that a Wires object can be indexed."""

        wires = Wires(iterable)
        for i in range(len(iterable)):
            assert wires[i] == iterable[i]

    def test_is_ordered(self):
        """Tests that a Wires object is not equal to another Wires object with a different ordering of the indices."""

        wires1 = Wires([1, 2, 3])
        wires2 = Wires([3, 2, 1])
        assert wires1 != wires2

    def test_slicing(self):
        """Tests that a Wires object can be sliced."""

        wires = Wires([1, 2, 3])
        assert wires[:2] == [1, 2]

    def test_length(self):
        """Tests that a Wires object returns the correct length."""

        wires = Wires([1, 2, 3, 4, 5])
        assert len(wires) == 5

    def test_retrieving_index(self):
        """Tests that the correct index of a Wires object is retrieved."""

        wires = Wires([1, 2, 3, 4, 5])
        assert wires.index(4) == 3

    def test_equality(self):
        """Tests that we can compare Wires objects with the '==' and '!=' operators."""

        wires1 = Wires([1, 2, 3])
        wires2 = Wires([4, 5, 6])
        wires3 = Wires([1, 2, 3])

        assert wires1 != wires2
        assert wires1 == wires3

    def test_representation(self):
        """Tests the string representation."""

        wires_str = str(Wires([1, 2, 3]))

        assert wires_str == "<Wires = {}>".format([1, 2, 3])

    def test_convert_to_numpy_array(self):
        """Tests that Wires object can be converted to a standard numpy array."""

        wires = Wires([4, 0, 1])
        array = np.array(wires)
        assert isinstance(array, np.ndarray)
        assert array.shape == (3, )
        for w1, w2 in zip(array, np.array([4, 0, 1])):
            assert w1 == w2
        
    def test_min_max(self):
        """Tests that the min() and max() functions of a Wires object return correct index."""

        wires = Wires([1, 2, 13, 4, 5])
        assert max(wires) == 13
        assert min(wires) == 1

    def test_combine_method(self):
        """Tests the ``combine()`` method."""

        wires = Wires([1, 2, 3])
        wires2 = Wires([1, 4, 5, 2])

        new_wires = wires.combine(wires2)
        assert wires.wire_list == [1, 2, 3]  # check original object remains the same
        assert new_wires.wire_list == [1, 2, 3, 4, 5]

        with pytest.raises(WireError, match="expected a `pennylane.wires.Wires` object"):
            wires.combine([8, 5])

    def test_intersection_method(self):
        """Tests the ``intersect()`` method."""

        wires1 = Wires([4, 0, 1])
        wires2 = Wires([0, 4, 3])
        res = wires1.intersect(wires2)
        assert res == Wires([4, 0])

        with pytest.raises(WireError, match="expected a `pennylane.wires.Wires` object"):
            wires1.intersect([8, 5])

    def test_difference_method(self):
        """Tests the ``difference()`` method."""

        wires1 = Wires([4, 0, 1])
        wires2 = Wires([0, 2, 3])
        res = wires1.difference(wires2)
        assert res == Wires([4, 1])

        with pytest.raises(WireError, match="expected a `pennylane.wires.Wires` object"):
            wires1.difference([8, 5])

    @pytest.mark.parametrize("wires2, target", [(Wires([1, 0, 3]), True),  # correct number of wires
                                                (Wires([2, 1]), False)])  # incorrect number of wires
    def test_injective_map_exists_method(self, wires2, target):
        """Tests the ``injective_map_exists()`` method."""

        wires = Wires([0, 1, 2])
        res = wires.injective_map_exists(wires2)
        assert res == target

        with pytest.raises(WireError, match="expected a `pennylane.wires.Wires` object"):
            wires.injective_map_exists([8, 5])

    def test_get_index_method(self):
        """Tests the ``get_indices()`` method."""

        wires = Wires([4, 0, 1])
        wires2 = Wires([1, 4])
        res = wires.get_indices(wires2)
        assert res == [2, 0]

        with pytest.raises(WireError, match="expected a `pennylane.wires.Wires` object"):
            wires.get_indices([8, 5])

    def test_select_method(self):
        """Tests the ``select()`` method."""

        wires = Wires([4, 0, 1, 5, 6])

        assert wires.select([2, 3, 0]) == Wires([1, 5, 4])
        assert wires.select(1) == Wires([0])
        assert wires.select([4, 5, 7], periodic_boundary=True) == Wires([6, 4, 1])

    def test_select_random_method(self):
        """Tests the ``select_random()`` method."""

        wires = Wires([4, 0, 1, 5, 6])

        assert len(wires.select_random(2)) == 2
        # check that seed makes call deterministic
        assert wires.select_random(4, seed=1) == wires.select_random(4, seed=1)

        with pytest.raises(WireError, match="cannot sample"):
            wires.select_random(6)
