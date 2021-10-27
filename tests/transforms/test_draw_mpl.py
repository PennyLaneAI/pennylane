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
Unit tests for the draw_mpl transform.
"""

import pytest

import pennylane as qml

mpl = pytest.importorskip("mpl")

def test_default():
    """Tests default usage"""

    dev = qml.device("default.qubit", wires=(0, "a", 1.23))

    @qml.qnode(dev)
    def circuit(x, y):
        qml.RX(x, wires=0)
        qml.CNOT(wires=(0,"a"))
        qml.RY(y, wires=1.23)
        return qml.expval(qml.PauliZ(0))

    # not constructed before calling
    fig, ax = qml.transforms.draw_mpl(circuit)(1.23, 2.34)

    assert isinstance(fig, mpl.figure.Figure)
    assert isinstance(ax, mpl.axes._axes.Axes)

    # proxy for whether correct things were drawn
    assert len(ax.patches) == 7
    assert len(ax.lines) == 7
    assert len(ax.texts) == 5

    assert ax.texts[0].get_text() == "0"
    assert ax.texts[1].get_text() == "a"
    assert ax.texts[2].get_text() == "1.23"
    assert ax.texts[3].get_text() == "RX"
    assert ax.texts[4].get_text() == "RY"

def test_decimals():
    """Test decimals changes operation labelling"""

    dev = qml.device("default.qubit", wires=(0, "a", 1.23))

    @qml.qnode(dev)
    def circuit(x, y):
        qml.RX(x, wires=0)
        qml.CNOT(wires=(0,"a"))
        qml.RY(y, wires=1.23)
        return qml.expval(qml.PauliZ(0))

    fig, ax = qml.transforms.draw_mpl(circuit, decimals=2)(1.23, 2.34)

    assert ax.texts[3].get_text() == "RX\n(1.23)"
    assert ax.texts[4].get_text() == "RY\n(2.34)"

def test_wire_order():
    """Test wire_order changes order of wires"""

    dev = qml.device("default.qubit", wires=(0, "a", 1.23))

    @qml.qnode(dev)
    def circuit(x, y):
        qml.RX(x, wires=0)
        qml.CNOT(wires=(0,"a"))
        qml.RY(y, wires=1.23)
        return qml.expval(qml.PauliZ(0))

    fig, ax = qml.transforms.draw_mpl(circuit, wire_order=(1.23, "a"))(1.23, 2.34)

    assert len(ax.texts) == 5

    assert ax.texts[0].get_text() == "1.23"
    assert ax.texts[1].get_text() == "a"
    assert ax.texts[2].get_text() == "0"

def test_empty_wires():
    """Test empty wires do not appear by default"""

    dev = qml.device('default.qubit', wires=(0,"a", 1.23))

    @qml.qnode(dev)
    def circuit():
        qml.RX(1.23, wires=0)
        return qml.expval(qml.PauliZ(0))

    fig, ax = qml.transforms.draw_mpl(circuit)()

    assert len(ax.lines) == 2
    assert ax.texts[0].get_text() == "0"
    assert ax.texts[1].get_text() == "RX"

def test_show_all_wires():
    """Test show_all_wires=True displays empty wires."""

    dev = qml.device('default.qubit', wires=(0,"a", 1.23))

    @qml.qnode(dev)
    def circuit():
        qml.RX(1.23, wires=0)
        return qml.expval(qml.PauliZ(0))

    fig, ax = qml.transforms.draw_mpl(circuit, show_all_wires=True)()

    assert len(ax.lines) == 4
    assert ax.texts[0].get_text() == "0"
    assert ax.texts[1].get_text() == "a"
    assert ax.texts[2].get_text() == "1.23"

