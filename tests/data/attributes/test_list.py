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
Tests for the ``DatasetList`` attribute type.
"""


from itertools import combinations

import numpy as np
import pytest

from pennylane.data import DatasetList

pytestmark = pytest.mark.data


def _generate_slices(len_: int):
    """Generates slices ``[{start}:{stop}:{step}]`` with all valid
    combinations of start,stop,step for a list of length ``len_``.
    """
    for x, y in combinations(range(-len_, len_), 2):
        for step in range(1, len_):
            yield slice(x, y, step)
            yield slice(y, x, step)


class TestList:
    """Test bind and value initialization for ``DatasetList``, and test
    that indexing, slicing and delete behaviour matches the built in ``list``."""

    def test_default_init(self):
        """Test that a DatasetList can be initialized without arguments."""
        dset_list = DatasetList()

        assert dset_list == []
        assert dset_list.info.type_id == "list"
        assert dset_list.info.py_type == "list"
        assert len(dset_list) == 0

    @pytest.mark.parametrize("input_type", (list, tuple))
    @pytest.mark.parametrize("value", [[], [1], [1, 2, 3], ["a", "b", "c"], [{"a": 1}]])
    def test_value_init(self, input_type, value):
        """Test that a DatasetList can be initialized from
        a list."""

        lst = DatasetList(input_type(value))
        assert lst == value
        assert len(lst) == len(value)
        with np.printoptions(legacy="1.21"):
            assert repr(lst) == repr(value)

    @pytest.mark.parametrize("input_type", (list, tuple))
    @pytest.mark.parametrize("value", [[], [1], [1, 2, 3], ["a", "b", "c"], [{"a": 1}]])
    def test_bind_init(self, input_type, value):
        """Test that a DatasetList can be initialized from
        a previously initialized HDF5 group."""

        bind = DatasetList(input_type(value)).bind
        assert DatasetList(bind=bind) == value

    @pytest.mark.parametrize("slc", _generate_slices(3))
    def test_slice(self, slc):
        """Test that slicing a DatasetList works exactly like slicing
        a built-in list."""
        builtin = [0, 1, 2]
        ds = DatasetList(builtin)
        assert builtin[slc] == ds[slc]

    @pytest.mark.parametrize("index", range(-3, 3))
    def test_indexing(self, index):
        """Test that indexing a DatasetList works exactly like slicing
        a built-in list."""
        builtin = [0, 1, 2]
        ds = DatasetList(builtin)
        assert builtin[index] == ds[index]

    @pytest.mark.parametrize("index", (-4, 3))
    def test_indexing_out_of_range(self, index):
        """Test that DatasetList raises an IndexError if given an
        index less than its negative length, or greater than or equal
        to its length.
        """
        ds = DatasetList([1, 2, 3])
        with pytest.raises(IndexError):
            _ = ds[index]

        with pytest.raises(IndexError):
            del ds[index]

    @pytest.mark.parametrize("index", range(-5, 5))
    def test_insert(self, index):
        """Test that insert on a DatasetList works exactly like inserting
        on a built-in list, including accepting out-of range indexes."""
        builtin = [0, 1, 2, {"a": 1}]
        ds = DatasetList(builtin)

        builtin.insert(index, 10)
        ds.insert(index, 10)

        assert ds == builtin

    @pytest.mark.parametrize("index", range(-4, 3))
    def test_delitem(self, index):
        """Test that __delitem__ can remove an object from any index while
        preserving the list structure."""
        builtin = [0, 1, 2, {"a": 1}]
        ds = DatasetList(builtin)

        del ds[index]
        del builtin[index]

        assert ds == builtin
        assert len(ds) == len(builtin)

    @pytest.mark.parametrize("index", range(-4, 3))
    def test_setitem(self, index):
        """Test that __setitem__ can replace an object at any index while preserving
        the list structure."""
        builtin = [0, 1, 2, {"a": 1}]
        ds = DatasetList(builtin)

        ds[index] = "test"
        builtin[index] = "test"

        assert ds == builtin
        assert len(ds) == len(builtin)

    @pytest.mark.parametrize("index", (-2, 1))
    def test_setitem_out_of_range(self, index):
        """Test that __setitem__ raises an IndexError when given an index
        that is out of range of the list."""
        ds = DatasetList([0])

        with pytest.raises(IndexError):
            ds[index] = 1

    @pytest.mark.parametrize("input_type", (list, tuple))
    @pytest.mark.parametrize("value", [[], [1], [1, 2, 3], ["a", "b", "c"], [{"a": 1}]])
    def test_copy(self, input_type, value):
        """Test that a `DatasetList` can be copied."""

        ds = DatasetList(input_type(value))
        ds_copy = ds.copy()

        assert ds_copy == value
        assert len(ds_copy) == len(value)
        with np.printoptions(legacy="1.21"):
            assert repr(ds_copy) == repr(value)

    @pytest.mark.parametrize("input_type", (list, tuple))
    @pytest.mark.parametrize("value", [[], [1], [1, 2, 3], ["a", "b", "c"], [{"a": 1}]])
    def test_equality(self, input_type, value):
        """Test that a `DatasetList`  can be compared to other objects."""
        ds = DatasetList(input_type(value))

        assert ds == input_type(value)
        assert ds != value.append("additional")
        for variable in ["string", 1, 1.0, {"0": 1}, True]:
            assert ds != variable

    @pytest.mark.parametrize("value", [[], [1], [1, 2, 3], ["a", "b", "c"], [{"a": 1}]])
    def test_string_conversion(self, value):
        """Test that a `DatasetList` is converted to a string correctly."""

        dset_dict = DatasetList(value)
        with np.printoptions(legacy="1.21"):
            assert str(dset_dict) == str(value)

    @pytest.mark.parametrize("value", [[1], [1, 2, 3], ["a", "b", "c"], [{"a": 1}]])
    def test_deleting_elements(self, value):
        """Test that elements can be removed from a `DatasetList`."""
        ds = DatasetList(value)
        del value[0]
        del ds[0]
        assert ds == value
