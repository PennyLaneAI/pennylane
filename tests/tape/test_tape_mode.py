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
"""Unit tests enabling and disabling tape mode"""
import pytest

import pennylane as qml


def test_enable_tape_mode_decorator():
    """Test that the enable_tape function properly
    enables tape mode when creating QNodes using the decorator."""
    dev = qml.device("default.qubit", wires=1)

    qml.disable_tape()

    @qml.qnode(dev)
    def circuit(x, y):
        qml.RX(x, wires=0)
        return qml.expval(qml.PauliZ(0))

    circuit(0.5, 0.1)

    assert not isinstance(circuit, qml.tape.QNode)
    assert isinstance(circuit, qml.qnodes.BaseQNode)
    assert not hasattr(circuit, "qtape")

    qml.enable_tape()

    assert "tape" in qml.expval.__module__

    @qml.qnode(dev)
    def circuit(x, y):
        qml.RX(x, wires=0)
        return qml.expval(qml.PauliZ(0))

    circuit(0.5, 0.1)

    assert isinstance(circuit, qml.tape.QNode)
    assert not isinstance(circuit, qml.qnodes.BaseQNode)
    assert hasattr(circuit, "qtape")


def test_enable_tape_mode_class():
    """Test that the enable_tape function properly
    enables tape mode when creating QNodes using the class."""
    dev = qml.device("default.qubit", wires=1)

    qml.disable_tape()

    def circuit(x, y):
        qml.RX(x, wires=0)
        return qml.expval(qml.PauliZ(0))

    qnode = qml.QNode(circuit, dev)
    qnode(0.5, 0.1)

    assert not isinstance(qnode, qml.tape.QNode)
    assert isinstance(qnode, qml.qnodes.BaseQNode)
    assert not hasattr(qnode, "qtape")

    qml.enable_tape()

    def circuit(x, y):
        qml.RX(x, wires=0)
        return qml.expval(qml.PauliZ(0))

    qnode = qml.QNode(circuit, dev)
    qnode(0.5, 0.1)

    assert isinstance(qnode, qml.tape.QNode)
    assert not isinstance(qnode, qml.qnodes.BaseQNode)
    assert hasattr(qnode, "qtape")



def test_disable_tape():
    """Test that the disable_tape function reverts QNode creation
    to standard behaviour"""
    dev = qml.device("default.qubit", wires=1)

    def circuit(x, y):
        qml.RX(x, wires=0)
        return qml.expval(qml.PauliZ(0))

    # doesn't matter how many times we call it
    qml.enable_tape()
    qml.enable_tape()

    qnode = qml.QNode(circuit, dev)
    qnode(0.5, 0.1)

    assert isinstance(qnode, qml.tape.QNode)
    assert not isinstance(qnode, qml.qnodes.BaseQNode)
    assert hasattr(qnode, "qtape")

    qml.disable_tape()

    assert "tape" not in qml.expval.__module__

    qnode = qml.QNode(circuit, dev)
    qnode(0.5, 0.1)

    assert not isinstance(qnode, qml.tape.QNode)
    assert isinstance(qnode, qml.qnodes.BaseQNode)
    assert not hasattr(qnode, "qtape")

    qml.enable_tape()


def test_disable_tape_exception():
    """Test that disabling tape mode raises a warning
    if not currently in tape mode"""
    qml.disable_tape()
    with pytest.warns(UserWarning, match="Tape mode is not currently enabled"):
        qml.disable_tape()
    qml.enable_tape()


def test_tape_mode_detection():
    """Test that the function `tape_mode_active` returns True
    only if tape mode is activated."""
    qml.disable_tape()
    assert not qml.tape_mode_active()
    qml.enable_tape()
    assert qml.tape_mode_active()
