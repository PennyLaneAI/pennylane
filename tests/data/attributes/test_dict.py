import pytest

from pennylane.data.attributes import DatasetDict


class TestDatasetString:
    @pytest.mark.parametrize("value", [{"a": 1, "b": 2}, {}, {"a": 1, "b": {"x": "y"}}])
    def test_from_value(self, value):
        dset_dict = DatasetDict(value)

        assert dset_dict == value
