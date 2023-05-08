from numbers import Number

import numpy as np

from pennylane.data import AttributeInfo, Dataset, DatasetScalar


class TestDataset:
    def test_init(self):
        """Test that initializing a Dataset with keyword arguments
        creates the expected attributes.
        """
        ds = Dataset(
            description="test",
            x=DatasetScalar(1, AttributeInfo(doc="A variable")),
            y=np.array([1, 2, 3]),
            z="abc",
        )

        assert ds.description == "test"
        assert ds.x == 1
        assert (ds.y == np.array([1, 2, 3])).all()
        assert ds.z == "abc"

    def test_setattr(self):
        """Test that __setattrr__ succesfully sets new and existing attributes."""
        ds = Dataset(description="test", x=1)

        ds.x = 2.0
        ds.q = "attribute"

        assert ds.q == "attribute"
        assert ds.x == 2.0

    def test_setattr_with_attribute_type(self):
        """Test that __setattr__ with a DatasetType passes through the AttributeInfo."""
        ds = Dataset(description="test")
        ds.x = DatasetScalar(2, info=AttributeInfo(doc="docstring", py_type=Number))

        assert ds.attrs["x"].info.py_type == "numbers.Number"
        assert ds.attrs["x"].info.doc == "docstring"

    def test_setattr_preserves_field_info(self):
        """Test that __setattr__ preserves AttributeInfo for fields."""
        ds = Dataset(description="test")

        ds.description = "a dataset"
        assert ds.attrs["description"].info.doc == Dataset.fields["description"].info.doc

    def test_setattr_with_attribute_type_updates_info(self):
        ds = Dataset(
            description="test", x=DatasetScalar(1.0, AttributeInfo(py_type=int, doc="an int"))
        )

        ds.x = DatasetScalar(2, AttributeInfo(doc="a float"))
