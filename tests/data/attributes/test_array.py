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
Tests for the ``DatasetArray`` attribute type.
"""

import numpy as np
import pytest

from pennylane.data.attributes import DatasetArray


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
