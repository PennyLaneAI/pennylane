import numpy as np
import pytest

from pennylane.data.attributes.operator._wires import wires_to_json
from pennylane.wires import Wires
import json


@pytest.mark.parametrize(
    "in_, out",
    [
        (np.array([0, 1, 2]), "[0, 1, 2]"),
        ([0, 1, 2], "[0, 1, 2]"),
        ((0, 1, 2), "[0, 1, 2]"),
        (range(3), "[0, 1, 2]"),
        (["a", "b"], '["a", "b"]'),
        ([0, 1, None], "[0, 1, null]"),
        (["a", 1, None], '["a", 1, null]'),
        (None, "[null]"),
        (1, "[1]"),
        ("a", '["a"]'),
        (np.int64(1), '[1]')
    ],
)
class TestWiresToJson:
    def test_wires_output_is_expected(self, in_, out):
        """Test that ``wires_to_json`` returns the expected output for json-serializable
        wire labels, as well as numpy integers, to to a json list."""

        assert wires_to_json(Wires(in_)) == out

    def test_wires_to_json_hash_equal(self, in_, out):
        """Test that the hash of the wires object is the same when
        loading from JSON."""

        in_ = Wires(in_)
        out = Wires(json.loads(out))

        assert hash(in_) == hash(out)
        for in_w, out_w in zip(in_, out):
            assert hash(in_w) == hash(out_w)
