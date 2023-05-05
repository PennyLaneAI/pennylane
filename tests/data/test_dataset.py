import numpy as np

from pennylane.data import Dataset


class TestDataset:
    def test_init(self):
        """Test that initializing a Dataset with keyword arguments
        creates the expected attributes.
        """
        ds = Dataset(description="test", x=1, y=np.array([1, 2, 3]), z="abc")

        assert ds.description == "test"
        assert ds.x == 1
        assert (ds.y == np.array([1, 2, 3])).all()
        assert ds.z == "abc"

    def test_setattr(self):
        """Test that __setattrr__ succesfully sets new and existing attributes."""
        ds = Dataset(description="test", x=1, y=np.array([1, 2, 3]), z="abc")

        ds.x = 2.0
        ds.q = "attribute"

        assert ds.q == "attribute"
        assert ds.x == 2.0
