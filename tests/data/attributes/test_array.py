import numpy as np
import pytest

from pennylane.data.attributes import DatasetArray, DatasetScalar


class TestDatasetArray:
    @pytest.mark.parametrize("value", [[1, 2, 3], [[1], [2]]])
    def test_value_init(self, value):
        """Test that a DatasetArray is correctly value-initialized."""
        value = np.array(value)

        arr = DatasetArray(value)

        assert (arr == value).all()
        assert (np.array(arr.bind) == value).all()
        assert arr.bind.dtype == value.dtype
        assert arr.info.py_type == "numpy.ndarray"
        assert arr.info.type_id == "array"

    @pytest.mark.parametrize("value", [[1, 2, 3], [[1], [2]]])
    def test_bind_init(self, value):
        """Test that DatasetArray can be initialized from a HDF5 array
        that was created by a DatasetArray."""
        value = np.array(value)
        bind = DatasetArray(value).bind
        arr = DatasetArray(bind=bind)

        assert (arr == value).all()
        assert (np.array(arr.bind) == value).all()
        assert arr.bind.dtype == value.dtype
        assert arr.info.py_type == "numpy.ndarray"
        assert arr.info.type_id == "array"
