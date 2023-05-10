from pennylane.data.attributes.wires import DatasetWires
import pytest
import numpy as np
from pennylane.wires import Wires


@pytest.mark.parametrize(
    "value",
    [
        np.array([0, 1, 2]),
        [0, 1, 2],
        (0, 1, 2),
        range(3),
        [1, 0, 4],
        ["a", "b", "c"],
        [0, 1, None],
        ["a", 1, "ancilla"],
    ],
)
class TestDatasetWires:
    def test_value_init(self, value):
        """Test that a DatasetOperator can be value-initialized
        from a Wires object."""
        value = Wires(value)
        dset_wires = DatasetWires(value)

        assert dset_wires.get_value() == value
        assert dset_wires.info["type_id"] == "wires"

    def test_bind_init(self, value):
        value = Wires(value)
        bind = DatasetWires(value).bind

        dset_wires = DatasetWires(bind=bind)

        assert dset_wires.get_value() == value
