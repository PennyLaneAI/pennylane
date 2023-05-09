import pytest

from pennylane.data.attributes import DatasetDict


class TestDatasetDict:
    @pytest.mark.parametrize("value", [{"a": 1, "b": 2}, {}, {"a": 1, "b": {"x": "y", "z": [1, 2]}}])
    def test_value_init(self, value):
        """Test that a DatasetDict is correctly value-initialized."""
        dset_dict = DatasetDict(value)

        assert dset_dict == value
        assert dset_dict.info.type_id == "dict"
        assert dset_dict.info.py_type == "dict"
        assert dset_dict.bind.keys() == value.keys()
        assert len(dset_dict) == len(value)

    @pytest.mark.parametrize("value", [{"a": 1, "b": 2}, {}, {"a": 1, "b": {"x": "y", "z": [1, 2]}}])
    def test_bind_init(self, value):
        """Test that a DatasetDict is correctly bind-initialized."""
        bind = DatasetDict(value).bind

        dset_dict = DatasetDict(bind=bind)

        assert dset_dict == value
        assert dset_dict.info.type_id == "dict"
        assert dset_dict.info.py_type == "dict"
        assert dset_dict.bind.keys() == value.keys()
        assert len(dset_dict) == len(value)
        