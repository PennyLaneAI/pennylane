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
Tests for the ``DatasetJSON`` attribute type.
"""

import pytest

from pennylane.data.attributes.json import DatasetJSON

pytestmark = pytest.mark.data


@pytest.mark.parametrize(
    "value",
    ["string", 1, {"one": "two"}, None, [1, "two"]],
)
class TestDatasetJSON:
    def test_value_init(self, value):
        """Test that DatasetJSON is correctly value-initialized."""
        dset_json = DatasetJSON(value)

        assert dset_json.get_value() == value

    def test_bind_init(self, value):
        """Test that DatasetJSON is correctly bind-initialized."""
        bind = DatasetJSON(value).bind

        assert DatasetJSON(bind=bind).get_value() == value
