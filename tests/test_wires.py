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
    """Tests for the ``Wires`` class."""

    @pytest.mark.parametrize("iterable", [Wires([0, 1, 2]),
                                          np.array([0, 1, 2]),
                                          [0, 1, 2],
                                          (0, 1, 2),
                                          range(3)
                                          ])
    def test_creation_from_common_iterables(self, iterable):
        """Tests that a Wires object can be created from standard iterable inputs."""

        wires = Wires(iterable)
        assert wires.wire_list == [0, 1, 2]

    @pytest.mark.parametrize("iterable", [[1, 0, 4],
                                          ['a', 'b', 'c'],
                                          ['a', 1, "ancilla"],
                                          [Wires(['a']), Wires([1, 2])]])
    def test_creation_from_different_wire_types(self, iterable):
        """Tests that a Wires object can be created from iterables of different
        objects representing a single wire index."""

        wires = Wires(iterable)
        assert wires.wire_list == list(iterable)

    @pytest.mark.parametrize("wire", [1, 'a', -1.4])
    def test_creation_from_single_object(self, wire):
        """Tests that a Wires object can be created from a non-iterable object
        representing a single wire index."""

        wires = Wires(wire)
        assert wires.wire_list == [wire]

    @pytest.mark.parametrize("iterable", [np.array([4, 1, 1, 3]),
                                          [4, 1, 1, 3],
                                          (4, 1, 1, 3),
                                          ['a', 'a', 'b'],
                                          [Wires([1, 0]), Wires([1, 0]), Wires([3])]])
    def test_error_for_repeated_wires(self, iterable):
        """Tests that a Wires object cannot be created from iterables with repeated indices."""

        with pytest.raises(WireError, match="Each wire must be represented by a unique index"):
            Wires(iterable)

    @pytest.mark.parametrize("iterable", [[4, 1, 0, 3],
                                          ['a', 'b', 'c']])
    def test_indexing_and_slicing(self, iterable):
        """Tests the indexing and slicing of Wires objects."""

        wires = Wires(iterable)

        # check single index
        for i in range(len(iterable)):
            assert wires[i] == Wires(iterable[i])
        # check slicing
        assert wires[:2] == Wires(iterable[:2])

    def test_equality(self):
        """Tests that we can compare Wires objects with the '==' and '!=' operators."""

        wires1 = Wires([1, 2, 3])
        wires2 = Wires([3, 2, 1])
        wires3 = Wires([1, 2, 3])
        assert wires1 != wires2
        assert wires1 == wires3

    def test_is_ordered(self):
        """Tests that a Wires object is not equal to another Wires object with a different ordering of the indices."""

        wires1 = Wires([1, 2, 3])
        wires2 = Wires([3, 2, 1])
        assert wires1 != wires2

    def test_length(self):
        """Tests that a Wires object returns the correct length."""

        wires = Wires([1, 2, 3, 4, 5])
        assert len(wires) == 5

    def test_representation(self):
        """Tests the string representation."""

        wires_str = str(Wires([1, 2, 3]))

        assert wires_str == "<Wires = {}>".format([1, 2, 3])

    def test_set(self):
        """Tests that the implementation of __hash__ allows for the set() function to work."""

        wires = Wires([0, 1, 2])
        list_of_wires = [Wires([1]), Wires([1]), Wires([1, 2, 3]), Wires([4])]

        assert set(wires) == {Wires([0]), Wires([1]), Wires([2])}
        assert set(list_of_wires) == {Wires([1]), Wires([1, 2, 3]), Wires([4])}

    def test_convert_to_numpy_array(self):
        """Tests that Wires object can be converted to a numpy array."""

        wires = Wires([4, 0, 1])
        array = wires.as_ndarray()
        assert isinstance(array, np.ndarray)
        assert array.shape == (3, )
        for w1, w2 in zip(array, np.array([4, 0, 1])):
            assert w1 == w2

    def test_convert_to_list(self):
        """Tests that Wires object can be converted to a list."""

        wires = Wires([4, 0, 1])
        lst = wires.as_list()
        assert isinstance(lst, list)
        assert wires.wire_list == lst

    @pytest.mark.parametrize("iterable", [[4, 1, 0, 3],
                                          ['a', 'b', 'c']])
    def test_index_method(self, iterable):
        """Tests the ``index()`` method."""

        wires = Wires(iterable)
        element = iterable[1]
        # check for non-Wires inputs
        assert wires.index(element) == 1
        # check for Wires inputs
        assert wires.index(Wires([element])) == 1
        # check that Wires of length >1 produce an error
        with pytest.raises(WireError, match="Can only retrieve index"):
            wires.index(Wires([1, 2]))

    def test_indices_method(self):
        """Tests the ``indices()`` method."""

        wires = Wires([4, 0, 1])
        # for non-Wires inputs
        assert wires.indices(Wires([1, 4])) == [2, 0]
        # for Wires inputs
        assert wires.indices([1, 4]) == [2, 0]

    def test_select_random_method(self):
        """Tests the ``select_random()`` method."""

        wires = Wires([4, 0, 1, 5, 6])

        assert len(wires.select_random(2)) == 2
        # check that seed makes call deterministic
        assert wires.select_random(4, seed=1) == wires.select_random(4, seed=1)

        with pytest.raises(WireError, match="cannot sample"):
            wires.select_random(6)

    def test_subset_method(self):
        """Tests the ``subset()`` method."""

        wires = Wires([4, 0, 1, 5, 6])

        assert wires.subset([2, 3, 0]) == Wires([1, 5, 4])
        assert wires.subset(1) == Wires([0])
        assert wires.subset([4, 5, 7], periodic_boundary=True) == Wires([6, 4, 1])

    def test_combined_method(self):
        """Tests the ``combined()`` method."""

        wires1 = Wires([1, 2, 3])
        wires2 = Wires([1, 4, 5, 2])

        new_wires = Wires.combined(wires1, wires2)
        assert new_wires.wire_list == [1, 2, 3, 4, 5]

        new_wires = Wires.combined(wires1, wires2, order_by_first=False)
        assert new_wires.wire_list == [1, 4, 5, 2, 3]

        with pytest.raises(WireError, match="expected a `pennylane.wires.Wires` object"):
            Wires.combined([3, 4], [8, 5])

    def test_shared_method(self):
        """Tests the ``shared()`` method."""

        wires1 = Wires([4, 0, 1])
        wires2 = Wires([0, 4, 3])

        res = Wires.shared(wires1, wires2)
        assert res == Wires([4, 0])

        res = Wires.shared(wires1, wires2, order_by_first=False)
        assert res == Wires([0, 4])

        with pytest.raises(WireError, match="expected a `pennylane.wires.Wires` object"):
            Wires.shared([2, 1], [8, 5])

    def test_unique_method(self):
        """Tests the ``unique()`` method."""

        wires1 = Wires([4, 0, 1])
        wires2 = Wires([0, 2, 3])
        res = Wires.unique(wires1, wires2)
        assert res == Wires([4, 1, 2, 3])

        res = Wires.unique(wires1, wires2, order_by_first=False)
        assert res == Wires([2, 3, 4, 1])

        with pytest.raises(WireError, match="expected a `pennylane.wires.Wires` object"):
            Wires.unique([2, 1], [8, 5])

    def test_merge_method(self):
        """Tests the ``merge()`` method."""

        list_of_wires = [Wires([0, 1]), Wires([2]), Wires([3, 4])]
        merged = Wires.merge(list_of_wires)

        assert merged == Wires([0, 1, 2, 3, 4])

        # check error for merging the same wires
        with pytest.raises(WireError, match="Cannot merge Wires objects that contain"):
            Wires.merge([Wires(0), Wires(0)])

        # check error for wrong inputs
        with pytest.raises(WireError, match="Expected list of Wires objects"):
            Wires.merge([[0, 1], [2]])
