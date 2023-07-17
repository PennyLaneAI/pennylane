import numpy as np

from pennylane.data import AttributeInfo, Dataset, DatasetString, attribute


class TestDataset:
    """Tests for basic dataset functionality."""

    def test_init(self):
        """Test that a dataset can be initialized with attributes, and
        that those attributes can be accessed."""
        ds = Dataset(a=[1, "two", {"x": 1}], x=1, y="abc", z=np.array([1, 2, 3, 4]))

        assert ds.a == [1, "two", {"x": 1}]
        assert ds.x == 1
        assert ds.y == "abc"
        assert all(np.array([1, 2, 3, 4]) == ds.z)

    def test_init_attribute_info(self):
        """Test that a Dataset attribute can have attribute info assigned."""

        ds = Dataset(x=attribute(1, doc="A number"))

        assert ds.attr_info["x"].doc == "A number"

    def test_init_with_attribute_type(self):
        """Test that a Dataset attribute can be created by passing a `DatasetAttribute`
        object to __init__."""

        ds = Dataset(x=DatasetString("abc", AttributeInfo(doc="A string")))

        assert ds.x == "abc"
        assert ds.attr_info["x"].doc == "A string"
