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
"""Tests for the gradients.hamiltonian module."""
import pennylane as qml
from pennylane.gradients.hamiltonian_grad import hamiltonian_grad


def test_behaviour():
    """Test that the function behaves as expected."""

    dev = qml.device("default.qubit", wires=2)

    with qml.tape.QuantumTape() as tape:
        qml.RY(0.3, wires=0)
        qml.RX(0.5, wires=1)
        qml.CNOT(wires=[0, 1])
        qml.expval(qml.Hamiltonian([-1.5, 2.0], [qml.PauliZ(0), qml.PauliZ(1)]))

    tape.trainable_params = {2, 3}
    tapes, processing_fn = hamiltonian_grad(tape, idx=0)
    res1 = processing_fn(dev.batch_execute(tapes))

    tapes, processing_fn = hamiltonian_grad(tape, idx=1)
    res2 = processing_fn(dev.batch_execute(tapes))

    with qml.tape.QuantumTape() as tape1:
        qml.RY(0.3, wires=0)
        qml.RX(0.5, wires=1)
        qml.CNOT(wires=[0, 1])
        qml.expval(qml.PauliZ(0))

    with qml.tape.QuantumTape() as tape2:
        qml.RY(0.3, wires=0)
        qml.RX(0.5, wires=1)
        qml.CNOT(wires=[0, 1])
        qml.expval(qml.PauliZ(1))

    dev.reset()
    res_expected1 = qml.math.squeeze(dev.execute(tape1))
    dev.reset()
    res_expected2 = qml.math.squeeze(dev.execute(tape2))

    assert res_expected1 == res1
    assert res_expected2 == res2
