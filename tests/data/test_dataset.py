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

from pennylane.data import AttributeInfo, Dataset, DatasetScalar, field
from pennylane.data.base.hdf5 import open_group


class MyDataset(
    Dataset, data_name="my_dataset", params=("x", "y")
):  # pylint: disable=too-few-public-methods
    """A dataset subclass for testing."""

    x: str = field()
    y: str = field()
    description: str = field(doc="description")


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

    @pytest.mark.parametrize("data_name, expect", [(None, "generic"), ("other_name", "other_name")])
    def test_init_dataname(self, data_name, expect):
        """Test that a base dataset's data_name can be set on init, and that
        it defaults to the class data name if none is provided."""
        ds = Dataset(data_name=data_name)

        assert ds.data_name == expect

    @pytest.mark.parametrize(
        "data_name, expect", [(None, "my_dataset"), ("other_name", "other_name")]
    )
    def test_subclass_init_dataname(self, data_name, expect):
        """Test that a subclassed datasets' data_name can be set on init, and that
        it defaults to the class data name if none is provided."""
        ds = MyDataset(x="1", y="2", description="abc", data_name=data_name)

        assert ds.data_name == expect

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

        ds = MyDataset(x="1", y="2", description="test")
        assert ds.data_name == "my_dataset"

    def test_dataset_bind_init_from_subclass(self):
        """Test that Dataset can be bind-initialized from a HDF5 group that
        was initialized by a subclass of Dataset."""

        bind = MyDataset(x="1", y="2", description="test").bind

        ds = Dataset(bind)

        assert ds.data_name == "my_dataset"
        assert ds.params == {"x": "1", "y": "2"}

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

        zgrp = open_group(path, mode="r")

        ds_2 = Dataset(zgrp)

        assert ds_2.bind is not ds.bind
        assert ds.attrs == ds_2.attrs

    @pytest.mark.parametrize(
        "params, expect", [(None, {}), (tuple(), {}), (("x", "y"), {"x": "1", "y": "2"})]
    )
    def test_params_base(self, params, expect):
        """Test that dataset params can be set."""
        ds = Dataset(x="1", y="2", params=params)

        assert ds.params == expect

    def test_subclass_params(self):
        """Test that dataset subclasses' params can be set."""
        ds = MyDataset(x="1", y="2", description="abc")

        assert ds.params == {"x": "1", "y": "2"}
