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
Tests for the :class:`pennylane.data.mapper.AttributeTypeMapper` class.
"""

from pennylane.data import AttributeInfo, Dataset, DatasetScalar
from pennylane.data.base.hdf5 import create_group
from pennylane.data.base.mapper import AttributeTypeMapper


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
