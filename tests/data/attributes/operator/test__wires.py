# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Tests for the wire serialization functions in :mod:`pennylane.data.attributes.operator._wires`
"""

import json

import numpy as np
import pytest

import pennylane as qml
from pennylane.data.attributes.operator._wires import UnserializableWireError, wires_to_json
from pennylane.wires import Wires

pytestmark = pytest.mark.data


class TestWiresToJson:

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
            (1, "[1]"),
            ("a", '["a"]'),
            (np.int64(1), "[1]"),
        ],
    )
    def test_wires_output_is_expected(self, in_, out):
        """Test that ``wires_to_json`` returns the expected output for json-serializable
        wire labels, as well as numpy integers, to to a json list."""

        in_ = Wires(in_)
        assert wires_to_json(Wires(in_)) == out

    @pytest.mark.parametrize(
        "in_",
        [
            np.array([0, 1, 2]),
            [0, 1, 2],
            (0, 1, 2),
            range(3),
            ["a", "b"],
            ["a", 0, 1, None],
            1,
            "a",
            np.int64(3),
        ],
    )
    def test_wires_hash_equal(self, in_):
        """Test that the hash of the wires object is the same when
        loading from JSON."""

        in_ = Wires(in_)
        out = Wires(json.loads(wires_to_json(in_)))

        assert hash(in_) == hash(out)
        for in_w, out_w in zip(in_, out):
            assert hash(in_w) == hash(out_w)

    @pytest.mark.parametrize("in_", [[np.float64(1)], [qml.PauliX(1)]])
    def test_unserializable(self, in_):
        """Test that wires_to_json raises an ``UnserializableWiresError`` when
        the wires are not json types or integers."""
        in_ = Wires(in_)

        with pytest.raises(UnserializableWireError):
            wires_to_json(in_)

    def test_bad_integral_unserializable(self):
        """Test that wires_to_json raises an ``UnserializableWiresError`` if any
        of the wires are integer-like, but have a different hash if converted
        to int."""

        class BadInt(int):
            def __hash__(self) -> int:
                return 0

            def __int__(self) -> int:
                return 1

        with pytest.raises(UnserializableWireError):
            wires_to_json(Wires([BadInt()]))
