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
Tests for the ``DatasetString`` attribute type.
"""

import pytest

from pennylane.data.attributes.string import DatasetString

pytestmark = pytest.mark.data


@pytest.mark.parametrize("value", ["", "abc"])
class TestDatasetString:
    """Tests for ``DatasetString`` value and bind initialization."""

    def test_value_init(self, value):
        """Tests that a ``Datastring`` can be successfully value-initialized."""
        dstr = DatasetString(value)

        assert dstr == value

    def test_bind_init(self, value):
        """Tests that a ``Datastring`` is correctly bind-initialzed."""
        bind = DatasetString(value).bind

        dset_str = DatasetString(bind=bind)

        assert dset_str.get_value() == value
