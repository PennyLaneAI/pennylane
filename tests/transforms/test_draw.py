# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

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
Integration tests for the draw transform
"""
import pytest

import pennylane as qml
from pennylane import numpy as np

from pennylane.transforms import draw

@qml.qnode(qml.device('default.qubit', wires=(0, "a", 1.234)))
def circuit(x, y, z):
    qml.RX(x, wires=0)
    qml.RY(y, wires="a")
    qml.RZ(z, wires=1.234)
    return qml.expval(qml.PauliZ(0))

class TestLabelling:
    """Test the wire labels."""

    def test_any_wire_labels(self):
        """Test wire labels with different kinds of objects."""

        split_str = draw(circuit)(1.2,2.3,3.4).split("\n")
        assert split_str[0][0:6] == "    0:"
        assert split_str[1][0:6] == "    a:"
        assert split_str[2][0:6] == "1.234:"

    def test_wire_order(self):
        """Test wire_order keyword changes order of the wires."""

        split_str = draw(circuit, wire_order=[1.234, "a", 0, "b"])(1.2, 2.3, 3.4).split("\n")
        assert split_str[0][0:6] == "1.234:"
        assert split_str[1][0:6] == "    a:"
        assert split_str[2][0:6] == "    0:"

    def test_show_all_wires(self):
        """Test show_all_wires=True forces empty wires to display."""

        @qml.qnode(qml.device('default.qubit', wires=(0, 1)))
        def circuit():
            return qml.expval(qml.PauliZ(0))

        split_str = draw(circuit, show_all_wires=True)().split("\n")
        assert split_str[0][0:2] == "0:"
        assert split_str[1][0:2] == "1:"

class TestDecimals:
    """Test the decimals keyword argument."""

    def test_decimals(self):
        """Test decimals keyword makes the operation parameters included to given precision"""

        expected = '    0: ──RX(1.23)─┤  <Z>\n    a: ──RY(2.35)─┤     \n1.234: ──RZ(3.46)─┤     '
        assert draw(circuit, decimals=2)(1.234,2.345, 3.456) == expected

    def test_decimals_multiparameters(self):
        """Test decimals also displays parameters when the operation has multiple parameters."""

        @qml.qnode(qml.device('default.qubit', wires=(0)))
        def circuit(x):
            qml.Rot(*x, wires=0)
            return qml.expval(qml.PauliZ(0))

        