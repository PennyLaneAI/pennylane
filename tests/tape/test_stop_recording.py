# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

import pennylane as qml
from pennylane.tape import stop_recording


def test_stop_recording_on_function_inside_QNode():
    """Test that the stop_recording transform when applied to a function
    is not recorded by a QNode"""
    dev = qml.device("default.qubit", wires=1)

    @stop_recording()
    def my_op():
        return [qml.RX(0.123, wires=0), qml.RY(2.32, wires=0), qml.RZ(1.95, wires=0)]

    res = []

    @qml.qnode(dev)
    def my_circuit():
        res.extend(my_op())
        return qml.expval(qml.PauliZ(0))

    my_circuit.construct([], {})
    tape = my_circuit.qtape

    assert len(tape.operations) == 0
    assert len(res) == 3


def test_stop_recording_directly_on_op():
    """Test that stop_recording transform works when directly applied to an op"""
    dev = qml.device("default.qubit", wires=1)
    res = []

    @qml.qnode(dev)
    def my_circuit():
        op1 = stop_recording()(qml.RX)(np.pi / 4.0, wires=0)
        op2 = qml.RY(np.pi / 4.0, wires=0)
        res.extend([op1, op2])
        return qml.expval(qml.PauliZ(0))

    my_circuit.construct([], {})
    tape = my_circuit.qtape

    assert len(tape.operations) == 1
    assert tape.operations[0] == res[1]
    assert len(res) == 2


def test_nested_stop_recording_on_function():
    """Test that stop_recording works when nested with other stop_recordings"""

    @stop_recording()
    @stop_recording()
    def my_op():
        return [
            qml.RX(0.123, wires=0),
            qml.RY(2.32, wires=0),
            qml.RZ(1.95, wires=0),
        ]

    # the stop_recording function will still work outside of any queuing contexts
    res = my_op()
    assert len(res) == 3

    dev = qml.device("default.qubit", wires=1)

    @qml.qnode(dev)
    def my_circuit():
        my_op()

        with stop_recording():
            qml.PauliX(wires=0)
            my_op()

        qml.Hadamard(wires=0)
        my_op()
        return qml.state()

    my_circuit.construct([], {})
    tape = my_circuit.qtape

    assert len(tape.operations) == 1
    assert tape.operations[0].name == "Hadamard"


def test_stop_recording_qnode_qfunc():
    """A QNode with a stop_recording qfunc will result in no quantum measurements."""
    dev = qml.device("default.qubit", wires=1)

    @qml.qnode(dev)
    @stop_recording()
    def my_circuit():
        qml.PauliX(wires=0)
        return qml.expval(qml.PauliZ(0))

    result = my_circuit()
    assert len(result) == 0

    tape = my_circuit.qtape
    assert len(tape.operations) == 0
    assert len(tape.measurements) == 0


def test_stop_recording_qnode():
    """A stop_recording QNode is unaffected"""
    dev = qml.device("default.qubit", wires=1)

    @stop_recording()
    @qml.qnode(dev)
    def my_circuit():
        qml.RX(np.pi, wires=0)
        return qml.expval(qml.PauliZ(0))

    result = my_circuit()
    assert result == -1.0
