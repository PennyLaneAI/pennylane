# Copyright 2026 Xanadu Quantum Technologies Inc.

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
Tests for the 'mark' functionality.
"""


import pennylane as qml
from pennylane.fourier.mark import MarkedOp, mark


class TestMarkedOp:
    """Tests for the 'MarkedOp' operator."""

    # pylint:disable=protected-access
    def test_flatten_unflatten(self):
        """Tests the unflatten and flatten methods."""

        op = MarkedOp(qml.X(0), marker="my-x")
        data, metadata = op._flatten()
        assert data[0] == qml.X(0)
        assert metadata[0] == ("marker", "my-x")

        unflattened_op = MarkedOp._unflatten(data, metadata)
        assert unflattened_op == op

    def test_repr(self):
        """Tests the 'repr'."""

        op = MarkedOp(qml.X(0), marker="my-x")
        assert repr(op) == 'MarkedOp(X(0), marker="my-x")'

    def test_label(self):
        """Tests the 'label' method."""

        op = MarkedOp(qml.X(0), marker="my-x")
        assert op.label() == 'X("my-x")'

        op = MarkedOp(qml.RX(1.2345, wires=0), marker="my-x")
        assert op.label() == 'RX("my-x")'
        assert op.label(decimals=2) == 'RX\n(1.23, "my-x")'

    def test_marker_property(self):
        """Tests the 'custom_label' property."""

        op = MarkedOp(qml.X(0), marker="my-x")
        assert hasattr(op, "marker")
        assert op.marker == "my-x"


def test_mark():
    """Tests for the 'mark' function."""

    op = qml.X(0)
    marked_op = mark(op, marker="my-x")
    assert isinstance(marked_op, MarkedOp)
    assert hasattr(marked_op, "marker")
    assert marked_op.marker == "my-x"
