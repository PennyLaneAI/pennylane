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
Tests for the ``DatasetNone`` attribute type.
"""

import pytest

from pennylane.data.attributes.none import DatasetNone

pytestmark = pytest.mark.data


class TestDatasetsNone:
    """Test that ``DatasetNone`` is correctly bind and value
    initialized."""

    def test_value_init(self):
        """Test that DatasetsNone can be value-initialized."""

        dsets_none = DatasetNone(None)

        assert dsets_none.get_value() is None
        assert bool(dsets_none) is False
        assert dsets_none.info["type_id"] == "none"
        assert dsets_none.bind.shape is None

    def test_bind_init(self):
        """Test that DatasetsNone can be bind-initialized."""
        bind = DatasetNone(None).bind

        dsets_none = DatasetNone(bind=bind)

        assert dsets_none.get_value() is None
        assert bool(dsets_none) is False
        assert dsets_none.info["type_id"] == "none"
        assert dsets_none.bind.shape is None

    def test_default_value(self):
        dsets_none = DatasetNone(None)
        dsets_default = DatasetNone()

        assert dsets_none == dsets_default
