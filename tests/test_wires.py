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
        assert len(wires) == len(iterable)

    @pytest.mark.parametrize("length", [1, 0, 4])
    def test_integer_as_inputs(self, length):
        """Tests that a Wires object can be created from integer representing the number of wires."""

        wires = Wires(length)
        assert length == len(wires)

    @pytest.mark.parametrize("wrong_length", [-1, -2])
    def test_exception_for_negative_integer_input(self, wrong_length):
        """Tests that a Wires object cannot be created from negative integer."""

        with pytest.raises(WireError, match="Number of wires cannot be negative"):
            Wires(wrong_length)

    @pytest.mark.parametrize("wrong_input", [1.2,
                                             None])
    def test_error_for_repeated_indices(self, wrong_input):
        """Tests that a Wires object cannot be created from inputs that are neither integers nor iterables."""

        with pytest.raises(WireError, match="Unexpected wires input"):
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

    def test_error_for_negative_indices(self):
        """Tests that a Wires object throws error when indices are negative."""

        with pytest.raises(WireError, match="Wire indices must be non-negative"):
            Wires([8, -1, 0, 5])

    @pytest.mark.parametrize("iterable", [np.array([4, 1, 0, 3]),
                                          [4, 1, 0, 3],
                                          (4, 1, 0, 3),
                                          range(4)])
    def test_indexing(self, iterable):
        """Tests that a Wires object can be indexed."""

        wires = Wires(iterable)

        for i in range(len(iterable)):
            assert wires[i] == iterable[i]

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

    def test_min_max(self):
        """Tests that the min() and max() functions of a Wires object return correct index."""

        wires = Wires([1, 2, 13, 4, 5])
        assert max(wires) == 13
        assert min(wires) == 1

    def test_extend_method(self):
        """Tests that a Wires object can be extended by another."""

        wires = Wires([1, 2, 3])
        wires2 = Wires([4, 5])

        wires.extend(wires2)
        assert wires.wire_list == [1, 2, 3, 4, 5]

        with pytest.raises(WireError, match="Expected wires object to extend this Wires object"):
            wires.extend([8, 5])

    @pytest.mark.parametrize("wires2, target", [(Wires([1, 0, 3]), True),  # correct number of wires
                                                (Wires([2, 1]), False)])  # incorrect number of wires
    def test_injective_map_exists_method(self, wires2, target):
        """Tests that the ``injective_map_exists()`` method produces the right output."""

        wires1 = Wires([0, 1, 2])
        res = wires1.injective_map_exists(wires2)
        assert res == target
