from itertools import combinations

import pytest
import zarr

from pennylane.data import AttributeType, DatasetList


def _generate_slices(list_len: int):
    for x, y in combinations(range(-list_len, list_len), 2):
        for step in range(1, list_len):
            yield slice(x, y, step)
            yield slice(y, x, step)


class TestList:
    @pytest.mark.parametrize("value", [[1], [1, 2, 3], ["a", "b", "c"], [{"a": 1}]])
    def test_value_init(self, value):
        """Test that a DatasetList can be initialized from
        a list."""

        lst = DatasetList(value)
        assert lst == value

    @pytest.mark.parametrize("value", [[], [1, 2, 3], ["a", "b", "c"], [{"a": 1}]])
    def test_bind_init(self, value):
        """Test that a DatasetList can be initialized from
        a previously initialized Zarr group."""

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

    @pytest.mark.parametrize("index", range(-5, 5))
    def test_insert(self, index):
        """Test that insert on a DatasetList works exactly like inserting
        on a built-in list, including accepting out-of range indexes."""
        builtin = [0, 1, 2, {"a": 1}]
        ds = DatasetList(builtin)

        builtin.insert(index, 10)
        ds.insert(index, 10)

        assert ds[index] == builtin[index]
