import numpy as np
import pytest

from pennylane.data.attributes import DatasetArray


class TestDatasetArray:
    @pytest.mark.parametrize("value", [[1, 2, 3], [[1], [2]]])
    def test_from_value(self, value):
        value = np.array(value)
        dset_arr = DatasetArray(value)

        assert (dset_arr == value).all()
