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

    @pytest.mark.parametrize("iterable", [np.array([4, 1, 1, 3]),
                                          [4, 1, 1, 3],
                                          (4, 1, 1, 3)])
    def test_error_for_repeated_indices(self, iterable):
        """Tests that a Wires object cannot have repeated indices."""

        with pytest.raises(WireError, match="XXX"):
            wires = Wires(iterable)

    @pytest.mark.parametrize("iterable", [np.array([4., 1., 0., 3.]),
                                          [4., 1., 0., 3.],
                                          (4., 1., 0., 3.)])
    def test_error_for_noninteger_floats(self, iterable):
        """Tests that a Wires object converts floats to integer elements."""

        with pytest.raises(WireError, match="XXX"):
            wires = Wires(iterable)

    @pytest.mark.parametrize("iterable", [np.array([4., 1.2, 0., 3.]),
                                          [4., 1., 0., 3.0001],
                                          (4.00000002, 1., 0., 3.)])
    def test_error_for_noninteger_floats(self, iterable):
        """Tests that a Wires object throws an error if the rounding error of float indices
        is not larger than the tolerance of 1e-8."""

        with pytest.raises(WireError, match="XXX"):
            wires = Wires(iterable)

    @pytest.mark.parametrize("iterable", [np.array([4, 1, 0, 3]),
                                          [4, 1, 0, 3],
                                          (4, 1, 0, 3),
                                          range(4)])
    def test_indexing(self, iterable):
        """Tests that a Wires object can be indexed."""

        wires = Wires(iterable)

    @pytest.mark.parametrize("iterable", [np.array([4, 1, 0, 3]),
                                          [4, 1, 0, 3],
                                          (4, 1, 0, 3),
                                          range(4)])
    def test_iteration(self, iterable):
        """Tests that a Wires object can be iterated over."""

        wires = Wires(iterable)

    def test_slicing(self, iterable):
        """Tests that a Wires object can be sliced."""

        wires = Wires(iterable)

    def test_length(self, iterable):
        """Tests that a Wires object returns the correct length."""

        wires = Wires(iterable)

    def test_comparison_operation(self, iterable):
        """Tests that a Wires can be compared with another."""

        wires = Wires(iterable)

    def test_retrieving_index(self, iterable):
        """Tests that the correct index of a Wires object is retrieved."""

        wires = Wires(iterable)

    def test_min_max(self, iterable):
        """Tests that the min() and max() functions of a Wires object work."""

        wires = Wires(iterable)
