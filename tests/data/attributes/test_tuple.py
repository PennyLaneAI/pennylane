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
Tests for the ``DatasetTuple`` attribute type.
"""
import pytest

from pennylane.data.attributes.tuple import DatasetTuple

pytestmark = pytest.mark.data


class TestDatasetTuple:
    """Tests for DatasetTuple."""

    @pytest.mark.parametrize("value", [tuple(), (1, 2), (["a", "b"], 3, [])])
    def test_value_init(self, value):
        """Tests that a ``DatasetTuple`` can be successfully value-initialized."""
        dset_tuple = DatasetTuple(value)

        assert dset_tuple.get_value() == value

    @pytest.mark.parametrize("value", [tuple(), (1, 2), (["a", "b"], 3, [])])
    def test_bind_init(self, value):
        """Tests that a ``DatasetTuple`` is correctly bind-initialzed."""
        bind = DatasetTuple(value).bind

        dset_tuple = DatasetTuple(bind=bind)

        assert dset_tuple.get_value() == value

    def test_default_init(self):
        """Tests that a ``DatasetTuple`` can be default-initialzed to
        an empty tuple."""
        dset_tuple = DatasetTuple()
        assert dset_tuple.get_value() == tuple()
