import pytest

from pennylane.data.attributes import DatasetDict
from pennylane.data.base.typing_util import UNSET


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
