import numpy as np
import pytest

from pennylane.data.attributes import DatasetArray, DatasetScalar
from pennylane.data.base.attribute import AttributeInfo


class TestDatasetArray:
    @pytest.mark.parametrize("value", [[1, 2, 3], [[1], [2]]])
    def test_from_value(self, value):
        """Test that a DatasetArray can be initialized from a numpy
        array."""
        value = np.array(value)
        dset_arr = DatasetArray(value)

        assert (dset_arr == value).all()

    def test_from_bind(self):
        """Test that DatasetArray can be initialized from a Zarr array
        that was created by a DatasetArray."""

        dset_attr = DatasetArray(np.array([1]))

        bind = dset_attr.bind

        del dset_attr

        assert DatasetArray(bind=bind).get_value() == np.array([1])


class TestDatasetScalar:
    @pytest.mark.parametrize("value", [1, 1.0, complex(1, 2)])
    def test_from_value(self, value):
        """Test that a DatasetScalar can be initialized from a number."""
        scalar = DatasetScalar(value)
        assert scalar == value
