import pytest

from pennylane.data.attributes.scalar import DatasetScalar


@pytest.mark.parametrize(
    "value, py_type", [(1, "int"), (1.0, "float"), (complex(1, 2), "complex")]
)
class TestDatasetScalar:
    def test_value_init(self, value, py_type):
        """Test that DatasetScalar can be initialized from a value."""

        scalar = DatasetScalar(value)

        assert scalar == value
        assert scalar.bind[()] == value
        assert scalar.bind.shape == ()
        assert scalar.info.py_type == py_type
        assert scalar.info.type_id == "scalar"

    def test_bind_init(self, value, py_type):
        """Test that DatasetScalar can be initialized from a Zarr array
        that was created by a DatasetScalar."""

        bind = DatasetScalar(value).bind
        scalar = DatasetScalar(bind=bind)

        assert scalar == value
        assert scalar.bind[()] == value
        assert scalar.bind.shape == ()
        assert scalar.info.py_type == py_type
        assert scalar.info.type_id == "scalar"
