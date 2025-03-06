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

# pylint: disable=too-few-public-methods, too-many-public-methods

from numbers import Number
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

from pennylane.data import (
    AttributeInfo,
    Dataset,
    DatasetJSON,
    DatasetNotWriteableError,
    DatasetScalar,
    attribute,
    field,
)
from pennylane.data.base.dataset import UNSET, _init_arg
from pennylane.data.base.hdf5 import open_group


class MyDataset(
    Dataset, data_name="my_dataset", identifiers=("x", "y")
):  # pylint: disable=too-many-public-methods
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
        assert ds.list_attributes() == ["description", "x", "y", "z"]

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

    def test_subclass_bind_init(self):
        """Test that Dataset can be bind-initialized from a HDF5 group that
        was initialized by a subclass of Dataset."""

        bind = MyDataset(x="1", y="2", description="test").bind

        ds = Dataset(bind)

        assert ds.data_name == "my_dataset"
        assert ds.identifiers == {"x": "1", "y": "2"}

    def test_subclass_category_id(self):
        """Test that a subclass of Dataset preserves the defined
        category_id."""

        ds = MyDataset(x="1", y="2", description="test")
        assert ds.data_name == "my_dataset"

    def test_getattr_unset_fields(self):
        """Test that __getattr__ returns UNSET if a dataset field
        is not set."""

        my_dataset = MyDataset(x="a", y="b")
        assert my_dataset.description is UNSET

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

    def test_setattr_preserves_field_info(self):
        """Test that __setattr__ preserves AttributeInfo for fields."""

        ds = MyDataset(description="test")

        assert ds.attrs["description"].info.doc == MyDataset.fields["description"].info.doc
        assert ds.attrs["description"].info["type_id"] == "string"

    def test_setattr_field_conflicting_type(self):
        """Test that __setattr__ raises a TypeError when setting an attribute
        with a `DatasetAttribute` different from the field type."""

        dset = MyDataset(x="1", y="2", description="test")

        with pytest.raises(
            TypeError, match="Expected 'x' to be of type 'DatasetString', but got 'DatasetScalar'."
        ):
            dset.x = DatasetScalar(1)

    def test_delattr(self):
        """Test that __delattr__ removes the attribute from the dataset
        and the underlying file."""

        ds = Dataset(x=1)

        del ds.x

        with pytest.raises(AttributeError, match="'Dataset' object has no attribute 'x'"):
            _ = ds.x

        assert "x" not in ds.bind

    def test_delattr_no_such_attribute(self):
        """Test the __delattr__ raises an AttributeError if the attribute
        does not exist."""

        ds = Dataset(y=1)

        with pytest.raises(AttributeError, match="'Dataset' object has no attribute 'x'"):
            del ds.x

    def test_repr_shortened(self):
        """Test the __repr__ has the expected format when there is more than 2 attributes."""

        ds = Dataset(x=1, y="A string", width=3, z=None, length=4, identifiers=("length", "width"))

        assert repr(ds) == "<Dataset = length: 4, width: 3, attributes: ['x', 'y', ...]>"

    def test_repr(self):
        """Test the __repr__ has the expected format when there is less than 2 attributes."""

        ds = Dataset(x=1, y="A string")

        assert repr(ds) == "<Dataset = attributes: ['x', 'y']>"

    def test__dir__(self):
        """Test that __dir__ returns all attributes."""

        ds = Dataset(x=1, y="2")
        assert dir(ds) == ["x", "y"]

    @pytest.mark.parametrize(
        "identifiers, expect", [(None, {}), (tuple(), {}), (("x", "y"), {"x": "1", "y": "2"})]
    )
    def test_identifiers_base(self, identifiers, expect):
        """Test that dataset identifiers can be set."""
        ds = Dataset(x="1", y="2", identifiers=identifiers)

        assert ds.identifiers == expect

    def test_identifiers_base_missing(self):
        """Test that identifiers whose attribute is missing on the
        dataset will not be in the returned dict."""
        ds = Dataset(x="1", identifiers=("x", "y"))

        assert ds.identifiers == {"x": "1"}

    def test_subclass_identifiers(self):
        """Test that dataset subclasses' identifiers can be set."""
        ds = MyDataset(x="1", y="2", description="abc")

        assert ds.identifiers == {"x": "1", "y": "2"}

    def test_subclass_identifiers_missing(self):
        """Test that dataset subclasses' identifiers can be set."""
        ds = MyDataset(x="1", description="abc")

        assert ds.identifiers == {"x": "1"}

    def test_attribute_info(self):
        """Test that attribute info can be set and accessed
        on a dataset attribute."""

        dset = Dataset(x=1, y=attribute(3, doc="y Documentation"))

        dset.attr_info["x"].doc = "x Documentation"

        assert dset.attr_info["x"].doc == "x Documentation"
        assert dset.attr_info["y"].doc == "y Documentation"

    @pytest.mark.parametrize("mode", ["w-", "w", "a"])
    def test_open_create(self, tmp_path, mode):
        """Test that open() can create a new dataset on disk."""

        path = Path(tmp_path, "dset.h5")
        dset = Dataset.open(Path(tmp_path, "dset.h5"), mode=mode)

        assert isinstance(dset, Dataset)
        assert Path(tmp_path, "dset.h5").exists()
        assert Path(dset.bind.filename) == path.absolute()

    def test_open_existing_read_only(self, tmp_path):
        """Test that open() can load an existing dataset
        on disk in read-only mode."""

        path = Path(tmp_path, "dset.h5")
        existing = Dataset.open(path, mode="w")
        existing.data = "some data"

        existing.close()

        dset = Dataset.open(path, mode="r")

        assert isinstance(dset, Dataset)
        assert Path(dset.bind.filename) == path.absolute()
        assert dset.data == "some data"

        with pytest.raises(DatasetNotWriteableError):
            dset.other_data = "some other data"

        dset.close()

    def test_open_existing_read_write(self, tmp_path):
        """Test that open() can load an existing dataset
        on disk for modification."""

        path = Path(tmp_path, "dset.h5")
        existing = Dataset.open(path, mode="w")
        existing.data = "some data"

        existing.close()

        dset = Dataset.open(path, mode="a")

        assert isinstance(dset, Dataset)
        assert Path(dset.bind.filename) == path.absolute()
        assert dset.data == "some data"

        dset.other_data = "some other data"

        dset.close()

        assert Dataset.open(path, "r").other_data == "some other data"

    def test_open_existing_copy(self, tmp_path):
        """Test that open() can load an existing dataset
        on disk and copy it into memory."""

        path = Path(tmp_path, "dset.h5")
        existing = Dataset.open(path, "w")

        existing.data = "some data"
        existing.close()

        dset = Dataset.open(path, "copy")

        assert dset.bind.filename != str(path)
        assert dset.data == "some data"

        dset.other_data = "other data"
        dset.close()

        with pytest.raises(AttributeError):
            _ = Dataset.open(path).other_data

    @pytest.mark.parametrize(
        "attributes_arg, attributes_expect", [(None, ["x", "y", "z"]), (["x", "y"], ["x", "y"])]
    )
    def test_read_from_path(self, tmp_path, attributes_arg, attributes_expect):
        """Test that read() can load attributes from a Dataset file."""
        path = Path(tmp_path, "dset.h5")
        existing = Dataset.open(path, "w")

        existing.x = 1
        existing.y = attribute("abc", doc="abc")
        existing.z = [1, 2, 3]

        existing.close()

        dset = Dataset()
        dset.read(path, attributes=attributes_arg)

        assert dset.list_attributes() == attributes_expect
        assert dset.y == "abc"
        assert dset.attr_info["y"].doc == "abc"

    @pytest.mark.parametrize(
        "attributes_arg, attributes_expect", [(None, ["x", "y", "z"]), (["x", "y"], ["x", "y"])]
    )
    def test_read_from_dataset(self, attributes_arg, attributes_expect):
        """Test that read() can load attributes from a Dataset file."""

        existing = Dataset()

        existing.x = 1
        existing.y = attribute("abc", doc="abc")
        existing.z = [1, 2, 3]

        dset = Dataset(x=2)
        dset.read(existing, attributes=attributes_arg)

        assert dset.list_attributes() == attributes_expect
        assert dset.y == "abc"
        assert dset.attr_info["y"].doc == "abc"

    @pytest.mark.parametrize("overwrite, expect_x", [(False, 2), (True, 1)])
    def test_read_overwrite(self, overwrite, expect_x):
        """Test that overwrite determnines whether an existing attribute
        is overwritten by read()."""

        existing = Dataset(x=1)
        new = Dataset(x=2)

        new.read(existing, overwrite=overwrite)
        assert new.x == expect_x

    @pytest.mark.parametrize("mode", ["w-", "w", "a"])
    def test_write(self, tmp_path, mode):
        """Test that the ``write`` method creates a `hdf5` file that contains
        the all the data in the dataset."""
        ds = Dataset(x=DatasetScalar(1.0, AttributeInfo(py_type=int, doc="an int")))

        path: Path = tmp_path / "test"
        ds.write(path, mode=mode)

        zgrp = open_group(path, mode="r")

        ds_2 = Dataset(zgrp)

        assert ds_2.bind is not ds.bind
        assert ds.attrs == ds_2.attrs

    @pytest.mark.parametrize(
        "attributes_arg,attributes_expect",
        [
            (["x"], ["x", "y"]),
            (["x", "y", "data"], ["x", "y", "data"]),
            (["data"], ["x", "y", "data"]),
        ],
    )
    def test_write_partial_always_copies_identifiers(self, attributes_arg, attributes_expect):
        """Test that ``write`` will always copy attributes that are identifiers."""
        ds = Dataset(x="a", y="b", data="Some data", identifiers=("x", "y"))
        ds_2 = Dataset()

        ds.write(ds_2, attributes=attributes_arg)
        assert set(ds_2.list_attributes()) == set(attributes_expect)
        assert ds_2.identifiers == ds.identifiers

    def test_init_subclass(self):
        """Test that __init_subclass__() does the following:

        - Does not create fields for _InitArg and ClassVar
        - collects fields from type annotations, including py_type
        - gets data name and identifiers from class arguments
        """

        class NewDataset(
            Dataset, data_name="new_dataset", identifiers=("x", "y")
        ):  # pylint: disable= too-few-public-methods
            """Dataset"""

            class_info: ClassVar[str] = "Class variable"

            init_arg: int = _init_arg(1)

            x: int
            y: str = field(doc="y documentation")
            jsonable: dict[str, str] = field(DatasetJSON, doc="json data")

        ds = NewDataset(init_arg=3, x=1, y="abc", jsonable={"a": "b"})

        assert ds.data_name == "new_dataset"
        assert ds.identifiers == {"x": 1, "y": "abc"}

        assert ds.attr_info["x"].py_type == "int"
        assert ds.attr_info["y"].py_type == "str"
        assert ds.attr_info["y"].doc == "y documentation"
        assert ds.attr_info["jsonable"].py_type == "dict[str, str]"
        assert ds.attr_info["jsonable"].doc == "json data"

    def test_dataset_as_attribute(self):
        """Test that a Dataset can be an attribute of
        another dataset."""

        child = Dataset(x=1, y=2)
        parent = Dataset(child=child)

        assert isinstance(parent.child, Dataset)
        assert parent.child.list_attributes() == child.list_attributes()
