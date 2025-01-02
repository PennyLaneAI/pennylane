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
Tests for the ``DatasetDict`` attribute type.
"""

import numpy as np
import pytest

from pennylane.data.attributes import DatasetDict

pytestmark = pytest.mark.data


class TestDatasetDict:
    def test_default_init(self):
        """Test that a DatasetDict can be initialized without arguments."""
        dset_dict = DatasetDict()

        assert dset_dict == {}
        assert dset_dict.info.type_id == "dict"
        assert dset_dict.info.py_type == "dict"
        assert dset_dict.bind.keys() == set()
        assert len(dset_dict) == 0

    @pytest.mark.parametrize(
        "value", [{"a": 1, "b": 2}, {}, {"a": 1, "b": {"x": "y", "z": [1, 2]}}]
    )
    def test_value_init(self, value):
        """Test that a DatasetDict is correctly value-initialized."""
        dset_dict = DatasetDict(value)

        assert dset_dict == value
        assert dset_dict.info.type_id == "dict"
        assert dset_dict.info.py_type == "dict"
        assert dset_dict.bind.keys() == value.keys()
        assert len(dset_dict) == len(value)
        with np.printoptions(legacy="1.21"):
            assert repr(value) == repr(dset_dict)

    @pytest.mark.parametrize(
        "value", [{"a": 1, "b": 2}, {}, {"a": 1, "b": {"x": "y", "z": [1, 2]}}]
    )
    def test_bind_init(self, value):
        """Test that a DatasetDict is correctly bind-initialized."""
        bind = DatasetDict(value).bind

        dset_dict = DatasetDict(bind=bind)

        assert dset_dict == value
        assert dset_dict.info.type_id == "dict"
        assert dset_dict.info.py_type == "dict"
        assert dset_dict.bind.keys() == value.keys()
        assert len(dset_dict) == len(value)

    @pytest.mark.parametrize("value", [{}, {"a": 1, "b": 2}])
    def test_setitem(self, value):
        """Test that __setitem__ can replace or insert a new element
        into a DatasetDict."""

        ds = DatasetDict(value)
        ds["a"] = 2
        value["a"] = 2

        assert ds == value

    @pytest.mark.parametrize("data", [{1: 2}, {None: 3}, {str: 3}])
    def test_init_key_type_not_str_exc(self, data):
        """Tests that a ``TypeError` is raised when trying to create a DatasetDict with non-string
        keys."""
        with pytest.raises(TypeError, match="'DatasetDict' keys must be strings"):
            DatasetDict(data)

    @pytest.mark.parametrize(
        "value", [{"a": 1, "b": 2}, {}, {"a": 1, "b": {"x": "y", "z": [1, 2]}}]
    )
    def test_copy(self, value):
        """Test that `DatasetDict.copy` can copy contents to a built-in dictionary."""
        dset_dict = DatasetDict(value)
        builtin_dict = dset_dict.copy()

        assert isinstance(builtin_dict, dict)
        assert builtin_dict == value

        assert builtin_dict.keys() == value.keys()
        assert len(builtin_dict) == len(value)
        with np.printoptions(legacy="1.21"):
            assert repr(builtin_dict) == repr(value)

    @pytest.mark.parametrize(
        "value", [{"a": 1, "b": 2}, {}, {"a": 1, "b": {"x": "y", "z": [1, 2]}}]
    )
    def test_equality(self, value):
        """Test that `DatasetDict` can be compared to other objects."""
        dset_dict = DatasetDict(value)
        dset_dict2 = DatasetDict(value)

        assert dset_dict == dset_dict2
        assert dset_dict == value

        assert dset_dict != []
        dset_dict2["additional"] = "additional"
        assert dset_dict != dset_dict2

    def test_equality_same_length(self):
        """Test that `DatasetDict` objects are different when they have same length
        but different keys."""
        dset_dict = DatasetDict({"a": 1, "b": 2})
        assert dset_dict != {"a": 1, "c": 2}

    @pytest.mark.parametrize(
        "value", [{"a": 1, "b": 2}, {}, {"a": 1, "b": {"x": "y", "z": [1, 2]}}]
    )
    def test_string_conversion(self, value):
        dset_dict = DatasetDict(value)
        with np.printoptions(legacy="1.21"):
            assert str(dset_dict) == str(value)
