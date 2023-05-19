from itertools import combinations

import pytest

from pennylane.data import DatasetList


def _generate_slices(list_len: int):
    for x, y in combinations(range(-list_len, list_len), 2):
        for step in range(1, list_len):
            yield slice(x, y, step)
            yield slice(y, x, step)


class TestList:
    def test_default_init(self):
        """Test that a DatasetList can be initialized without arguments."""
        dset_list = DatasetList()

        assert dset_list == []
        assert dset_list.info.type_id == "list"
        assert dset_list.info.py_type == "list"
        assert len(dset_list) == 0

    @pytest.mark.parametrize("value", [[], [1], [1, 2, 3], ["a", "b", "c"], [{"a": 1}]])
    def test_value_init(self, value):
        """Test that a DatasetList can be initialized from
        a list."""

        lst = DatasetList(value)
        assert lst == value
        assert repr(lst) == repr(value)

    @pytest.mark.parametrize("value", [[], [1], [1, 2, 3], ["a", "b", "c"], [{"a": 1}]])
    def test_bind_init(self, value):
        """Test that a DatasetList can be initialized from
        a previously initialized HDF5 group."""

        bind = DatasetList(value).bind
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
            ds[index]

    @pytest.mark.parametrize("index", range(-5, 5))
    def test_insert(self, index):
        """Test that insert on a DatasetList works exactly like inserting
        on a built-in list, including accepting out-of range indexes."""
        builtin = [0, 1, 2, {"a": 1}]
        ds = DatasetList(builtin)

        builtin.insert(index, 10)
        ds.insert(index, 10)

        assert ds[index] == builtin[index]
