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
Tests for :mod:`pennylane.data.base.dataset`.
"""


from numbers import Number
from pathlib import Path

import numpy as np
import pytest

from pennylane.data import AttributeInfo, Dataset, DatasetScalar, attribute
from pennylane.data.base._hdf5 import h5py


class MyDataset(Dataset):  # pylint: disable=too-few-public-methods
    """A dataset subclass for testing."""

    category_id = "my_dataset"

    description: str = attribute(doc="description")


class TestDataset:
    """Tests initialization and instance methods of the ``Dataset`` class."""

    def test_init(self):
        """Test that initializing a Dataset with keyword arguments
        creates the expected attributes.
        """
        ds = Dataset(
            description="test",
            x=DatasetScalar(1, AttributeInfo(doc="A variable")),
            y=np.array([1, 2, 3]),
            z="abc",
        )

        assert ds.description == "test"
        assert ds.x == 1
        assert (ds.y == np.array([1, 2, 3])).all()
        assert ds.z == "abc"

    def test_setattr(self):
        """Test that __setattrr__ successfully sets new attributes."""
        ds = Dataset()

        ds.x = 2.0
        ds.q = "attribute"

        assert ds.q == "attribute"
        assert ds.x == 2.0

    def test_setattr_with_attribute_type(self):
        """Test that __setattr__ with a DatasetType passes through the AttributeInfo."""
        ds = Dataset()
        ds.x = DatasetScalar(2, info=AttributeInfo(doc="docstring", py_type=Number))

        assert ds.attrs["x"].info.py_type == "numbers.Number"
        assert ds.attrs["x"].info.doc == "docstring"

    def test_subclass_category_id(self):
        """Test that a subclass of Dataset preserves the defined
        category_id."""

        assert MyDataset.category_id == "my_dataset"
        ds = MyDataset(description="test")
        assert ds.category_id == "my_dataset"

    def test_setattr_preserves_field_info(self):
        """Test that __setattr__ preserves AttributeInfo for fields."""

        ds = MyDataset(description="test")

        assert ds.attrs["description"].info.doc == MyDataset.fields["description"].info.doc
        assert ds.attrs["description"].info["type_id"] == "string"

    @pytest.mark.parametrize("mode", ["w-", "w", "a"])
    def test_write(self, tmp_path, mode):
        """Test that the ``write`` method creates a Zarr file that contains
        the all the data in the dataset."""
        ds = Dataset(x=DatasetScalar(1.0, AttributeInfo(py_type=int, doc="an int")))

        path: Path = tmp_path / "test"
        ds.write(path, mode=mode)

        zgrp = h5py.open_group(path, mode="r")

        ds_2 = Dataset(zgrp)

        assert ds_2.bind is not ds.bind
        assert ds.attrs == ds_2.attrs

    @pytest.mark.parametrize("params", [{}, {"x": "y", "z": "a"}])
    def test_params(self, params):
        """Test that dataset params can be set."""
        ds = Dataset(params=params)

        assert ds.params == params
