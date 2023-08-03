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
Tests for the :class:`pennylane.data.base.mapper.AttributeTypeMapper` class.
"""

from unittest.mock import MagicMock

import pytest

import pennylane.data.base.mapper
from pennylane.data import AttributeInfo, Dataset, DatasetScalar
from pennylane.data.base.hdf5 import create_group
from pennylane.data.base.mapper import AttributeTypeMapper

pytestmark = pytest.mark.data


class TestMapper:  # pylint: disable=too-few-public-methods
    """Tests for :class:`pennylane.data.mapper.AttributeTypeMapper`."""

    def test_info(self):
        """Test that info() returns the attribute info from bind."""
        bind = create_group()

        info = AttributeInfo(bind.attrs)
        info.doc = "documentation"

        mapper = AttributeTypeMapper(bind)

        assert mapper.info == info

    def test_set_item_attribute_with_info(self):
        """Test that set_item() copies info from the
        info argument when passed a DatasetAttribute."""
        dset = Dataset()
        mapper = AttributeTypeMapper(dset.bind)

        mapper.set_item(
            "x", DatasetScalar(1, info=AttributeInfo(doc="abc")), info=AttributeInfo(extra="xyz")
        )

        assert dset.attr_info["x"].doc == "abc"
        assert dset.attr_info["x"]["extra"] == "xyz"

    def test_set_item_other_value_error(self, monkeypatch):
        """Test that set_item() only captures a ValueError if it is
        caused by a HDF5 file not being writeable."""
        monkeypatch.setattr(
            pennylane.data.base.mapper,
            "match_obj_type",
            MagicMock(side_effect=ValueError("Something")),
        )

        with pytest.raises(ValueError, match="Something"):
            AttributeTypeMapper(create_group()).set_item("x", 1, None)

    def test_repr(self):
        """Test that __repr__ is equivalent to dict.__repr__."""
        mapper = AttributeTypeMapper(create_group())

        mapper["x"] = 1
        mapper["y"] = {"a": "b"}

        assert repr(mapper) == repr({"x": DatasetScalar(1), "y": {"a": "b"}})

    def test_str(self):
        """Test that __str__ is equivalent to dict.__str__."""
        mapper = AttributeTypeMapper(create_group())

        mapper["x"] = 1
        mapper["y"] = {"a": "b"}

        assert str(mapper) == str({"x": DatasetScalar(1), "y": {"a": "b"}})
