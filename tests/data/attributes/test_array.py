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

import pennylane.data.attributes.array
from pennylane import numpy as qml_numpy
from pennylane.data.attributes import DatasetArray

pytestmark = pytest.mark.data


class TestDatasetArray:
    @pytest.mark.parametrize(
        "value, py_type, interface",
        [
            (np.array([1, 2, 3]), "numpy.ndarray", "numpy"),
            (qml_numpy.tensor([1, 2, 3]), "pennylane.numpy.tensor.tensor", "autograd"),
        ],
    )
    def test_value_init(self, value, interface, py_type):
        """Test that a DatasetArray is correctly value-initialized."""
        arr = DatasetArray(value)

        assert isinstance(arr.get_value(), type(value))
        assert (arr.get_value() == value).all()
        assert arr.bind.dtype == value.dtype
        assert arr.info.py_type == py_type
        assert arr.info.type_id == "array"
        assert arr.info["array_interface"] == interface

    @pytest.mark.parametrize("array_class", [qml_numpy.tensor])
    @pytest.mark.parametrize("requires_grad", [True, False])
    def test_value_init_requires_grad(self, array_class, requires_grad):
        """Test that a DatasetArray preserves the ``requires_grad`` attribute
        of array interfaces."""
        value = array_class([1, 2, 3, 4], requires_grad=requires_grad)

        arr = DatasetArray(value)

        assert arr.get_value().requires_grad == requires_grad

    @pytest.mark.parametrize("interface", ["jax", "torch"])
    def test_value_init_invalid_interface(self, monkeypatch, interface):
        """Test that a TypeError is raised if value initialized with an
        incompatible array type."""
        monkeypatch.setattr(pennylane.data.attributes.array, "get_interface", lambda _: interface)

        with pytest.raises(TypeError):
            DatasetArray([1, 2, 3, 4])

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
