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
Tests for the ``DatasetScalar`` attribute type.
"""

import pytest

from pennylane.data.attributes.scalar import DatasetScalar

pytestmark = pytest.mark.data


@pytest.mark.parametrize("value, py_type", [(1, "int"), (1.0, "float"), (complex(1, 2), "complex")])
class TestDatasetScalar:
    """Test that a ``DatasetScalar`` is correctly bind and value initialized."""

    def test_value_init(self, value, py_type):
        """Test that DatasetScalar can be initialized from a value."""

        scalar = DatasetScalar(value)

        assert scalar == value
        assert scalar.bind[()] == value
        assert scalar.bind.shape == ()
        assert scalar.info.py_type == py_type
        assert scalar.info.type_id == "scalar"

    def test_bind_init(self, value, py_type):
        """Test that DatasetScalar can be initialized from a HDF5 array
        that was created by a DatasetScalar."""

        bind = DatasetScalar(value).bind
        scalar = DatasetScalar(bind=bind)

        assert scalar == value
        assert scalar.bind[()] == value
        assert scalar.bind.shape == ()
        assert scalar.info.py_type == py_type
        assert scalar.info.type_id == "scalar"
