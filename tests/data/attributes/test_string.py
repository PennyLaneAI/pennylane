import pytest

from pennylane.data.attributes.string import DatasetString


class TestDatasetString:
    @pytest.mark.parametrize("value", ["", "abc"])
    def test_from_value(self, value):
        dstr = DatasetString(value)

        assert dstr == value
