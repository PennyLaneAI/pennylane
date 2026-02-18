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
Tests for the 'label' functionality.
"""


import pennylane as qml
from pennylane.drawer.label import LabelledOp, label


class TestLabelledOp:
    """Tests for the 'LabelledOp' operator."""

    # pylint:disable=protected-access
    def test_flatten_unflatten(self):
        """Tests the unflatten and flatten methods."""

        op = LabelledOp(qml.X(0), custom_label="my-x")
        data, metadata = op._flatten()
        assert data[0] == qml.X(0)
        assert metadata[0] == ("custom_label", "my-x")

        unflattened_op = LabelledOp._unflatten(data, metadata)
        assert unflattened_op == op

    def test_repr(self):
        """Tests the 'repr'."""

        op = LabelledOp(qml.X(0), custom_label="my-x")
        assert repr(op) == 'LabelledOp(X(0), custom_label="my-x")'

    def test_label(self):
        """Tests the 'label' method."""

        op = LabelledOp(qml.X(0), custom_label="my-x")
        assert op.label() == 'X("my-x")'

        op = LabelledOp(qml.RX(1.2345, wires=0), custom_label="my-x")
        assert op.label() == 'RX("my-x")'
        assert op.label(decimals=2) == 'RX\n(1.23, "my-x")'

    def test_custom_label_property(self):
        """Tests the 'custom_label' property."""

        op = LabelledOp(qml.X(0), custom_label="my-x")
        assert hasattr(op, "custom_label")
        assert op.custom_label == "my-x"


def test_label():
    """Tests the label function."""

    op = qml.X(0)
    labelled_op = label(op, new_label="my-x")
    assert isinstance(labelled_op, LabelledOp)
    assert hasattr(labelled_op, "custom_label")
    assert labelled_op.custom_label == "my-x"
