import numpy as np
import pytest

from pennylane.data.attributes import DatasetArray, DatasetScalar
from pennylane.data.base.attribute import AttributeInfo


class TestDatasetArray:
    @pytest.mark.parametrize("value", [[1, 2, 3], [[1], [2]]])
    def test_from_value(self, value):
        value = np.array(value)
        dset_arr = DatasetArray(value)

        assert (dset_arr == value).all()

    def test_set_value(self):
        dset_arr = DatasetArray([1, 2])

        dset_arr.set_value([2, 3], AttributeInfo())

        assert (dset_arr.get_value() == np.array([2, 3])).all()


class TestDatasetScalar:
    @pytest.mark.parametrize("value", [1, 1.0, complex(1, 2)])
    def test_from_value(self, value):
        scalar = DatasetScalar(value)
        assert scalar == value

    def test_set_value(self):
        scalar = DatasetScalar(1)

        scalar.set_value(2, AttributeInfo())

        assert scalar.get_value() == 2
