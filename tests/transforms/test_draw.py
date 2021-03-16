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
Unit tests for the draw transform.
"""
import pytest

import pennylane as qml
from pennylane import numpy as np


def test_drawing():
    """Test circuit drawing"""

    x = np.array(0.1, requires_grad=True)
    y = np.array([0.2, 0.3], requires_grad=True)
    z = np.array(0.4, requires_grad=True)

    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev, interface="autograd")
    def circuit(p1, p2=y, **kwargs):
        qml.RX(p1, wires=0)
        qml.RY(p2[0] * p2[1], wires=1)
        qml.RX(kwargs["p3"], wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

    result = qml.draw(circuit)(p1=x, p3=z)
    expected = """\
 0: ──RX(0.1)───RX(0.4)──╭C──╭┤ ⟨Z ⊗ X⟩ 
 1: ──RY(0.06)───────────╰X──╰┤ ⟨Z ⊗ X⟩ 
"""

    assert result == expected


def test_drawing_ascii():
    """Test circuit drawing when using ASCII characters"""
    from pennylane import numpy as np

    x = np.array(0.1, requires_grad=True)
    y = np.array([0.2, 0.3], requires_grad=True)
    z = np.array(0.4, requires_grad=True)

    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev, interface="autograd")
    def circuit(p1, p2=y, **kwargs):
        qml.RX(p1, wires=0)
        qml.RY(p2[0] * p2[1], wires=1)
        qml.RX(kwargs["p3"], wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

    result = qml.draw(circuit, charset="ascii")(p1=x, p3=z)
    expected = """\
 0: --RX(0.1)---RX(0.4)--+C--+| <Z @ X> 
 1: --RY(0.06)-----------+X--+| <Z @ X> 
"""

    assert result == expected


def test_show_all_wires_error():
    """Test that show_all_wires will raise an error if the provided wire
    order does not contain all wires on the device"""

    dev = qml.device('default.qubit', wires=[-1, "a", "q2", 0])

    @qml.qnode(dev)
    def circuit():
        qml.Hadamard(wires=-1)
        qml.CNOT(wires=[-1, "q2"])
        return qml.expval(qml.PauliX(wires="q2"))

    with pytest.raises(ValueError, match="must contain all wires"):
        qml.draw(circuit, show_all_wires=True, wire_order=[-1, "a"])()


def test_missing_wire():
    """Test that wires not specifically mentioned in the wire
    reordering are appended at the bottom of the circuit drawing"""

    dev = qml.device('default.qubit', wires=["a", -1, "q2"])

    @qml.qnode(dev)
    def circuit():
        qml.Hadamard(wires=-1)
        qml.CNOT(wires=["a", "q2"])
        qml.RX(0.2, wires="a")
        return qml.expval(qml.PauliX(wires="q2"))

    # test one missing wire
    res = qml.draw(circuit, wire_order=["q2", "a"])()
    expected = [
        " q2: ──╭X───────────┤ ⟨X⟩ ",
        "  a: ──╰C──RX(0.2)──┤     ",
        " -1: ───H───────────┤     \n"
    ]

    assert res == "\n".join(expected)

    # test one missing wire
    res = qml.draw(circuit, wire_order=["q2", -1])()
    expected = [
        " q2: ─────╭X───────────┤ ⟨X⟩ ",
        " -1: ──H──│────────────┤     ",
        "  a: ─────╰C──RX(0.2)──┤     \n"
    ]

    assert res == "\n".join(expected)

    # test multiple missing wires
    res = qml.draw(circuit, wire_order=["q2"])()
    expected = [
        " q2: ─────╭X───────────┤ ⟨X⟩ ",
        " -1: ──H──│────────────┤     ",
        "  a: ─────╰C──RX(0.2)──┤     \n"
    ]

    assert res == "\n".join(expected)


def test_invalid_wires():
    """Test that an exception is raised if a wire in the wire
    ordering does not exist on the device"""
    dev = qml.device('default.qubit', wires=["a", -1, "q2"])

    @qml.qnode(dev)
    def circuit():
        qml.Hadamard(wires=-1)
        qml.CNOT(wires=["a", "q2"])
        qml.RX(0.2, wires="a")
        return qml.expval(qml.PauliX(wires="q2"))

    with pytest.raises(ValueError, match="contains wires not contained on the device"):
        qml.draw(circuit, wire_order=["q2", 5])()
