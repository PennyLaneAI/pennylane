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

import pytest
import numpy as np
import pennylane as qml
import pennylane.tape
from pennylane.interfaces.autograd import AutogradInterface
from pennylane.interfaces.tf import TFInterface

"""Defines the device used for all tests"""

dev = qml.device("default.qubit", wires=4)

"""Defines circuits to be used in queueing/output tests"""

with pennylane.tape.QuantumTape() as tape1:
    qml.PauliX(0)
    H1 = 1.5 * qml.PauliZ(0) @ qml.PauliZ(1)
    qml.expval(H1)

with pennylane.tape.QuantumTape() as tape2:
    qml.Hadamard(0)
    qml.Hadamard(1)
    qml.PauliZ(1)
    qml.PauliX(2)

    H2 = qml.PauliX(0) @ qml.PauliZ(2) + 3 * qml.PauliZ(2) - 2 * qml.PauliX(0) + qml.PauliX(2) + qml.PauliZ(0) @ qml.PauliX(1)
    qml.expval(H2)

with pennylane.tape.QuantumTape() as tape3:
    qml.Hadamard(0)
    qml.Hadamard(1)
    qml.PauliZ(1)
    qml.PauliX(2)

    H3 = qml.Hamiltonian([1, 3, -2, 1, 1], [qml.PauliX(0) @ qml.PauliZ(2), qml.PauliZ(2), qml.PauliX(0), qml.PauliX(2), qml.PauliZ(0) @ qml.PauliX(1)])
    qml.expval(H3)

H4 = 1.5 * qml.PauliZ(0) @ qml.PauliZ(1) + 0.3 * qml.PauliX(1)

with qml.tape.QuantumTape() as tape4:
    qml.PauliX(0)
    qml.expval(H4)

TAPES = [tape1, tape2, tape3]
OUTPUTS = [-1.5, -6, -6, -1.5]

"""Defines the data to be used for differentiation tests"""

H = [
    qml.Hamiltonian([-0.2, 0.5, 1], [qml.PauliX(1), qml.PauliZ(1) @ qml.PauliY(2), qml.PauliZ(0)])
]

GRAD_VAR = [
    np.array([0.1, 0.67, 0.3, 0.4, -0.5, 0.7])
]
TF_VAR = []
TORCH_VAR = []

GRAD_OUT = [0.42294409781940356]
TF_OUT = []
TORCH_OUT = []

class TestHamiltonianExpval:
    """Tests for the hamiltonian_expand transform"""

    @pytest.mark.parametrize(("tape", "output"), zip(TAPES, OUTPUTS))
    def test_hamiltonians(self, tape, output):
        """Tests that the hamiltonian_expand transform returns the correct value"""

        tapes, fn = qml.transforms.hamiltonian_expand(tape)
        results = dev.batch_execute(tapes)
        expval = fn(results)

        assert np.isclose(output, expval)

    def test_hamiltonian_error(self):

        with pennylane.tape.QuantumTape() as tape:
            qml.expval(qml.PauliZ(0))

        with pytest.raises(ValueError, match=r"Passed tape must end in"):
            tapes, fn = qml.transforms.hamiltonian_expand(tape)

    @pytest.mark.parametrize(("tape", "hamiltonian", "output"), zip(TAPES, [H1, H2, H3, H4], OUTPUTS))
    def test_hamiltonian_in_qnode(self, tape, hamiltonian, output):
        @qml.qnode(dev)
        def circuit():
            [operation.queue() for operation in tape.operations]
            return qml.expval(hamiltonian)

        assert np.isclose(float(circuit()), output)

    @pytest.mark.parametrize(("H", "var", "output"), zip(H, GRAD_VAR, GRAD_OUT))
    def test_hamiltonian_dif_autograd(self, H, var, output):
        """Tests that the hamiltonian_expand tape transform is differentiable with the Autograd interface"""

        with qml.tape.JacobianTape() as tape:
            for i in range(2):
                qml.RX(np.array(0), wires=0)
                qml.RX(np.array(0), wires=1)
                qml.RX(np.array(0), wires=2)
                qml.CNOT(wires=[0, 1])
                qml.CNOT(wires=[1, 2])
                qml.CNOT(wires=[2, 0])

            qml.expval(H)

        AutogradInterface.apply(tape)

        def cost(x):
            tape.set_parameters(x)
            tapes, fn = qml.transforms.hamiltonian_expand(tape)
            res = [t.execute(dev) for t in tapes]
            return fn(res)

        assert np.isclose(cost(var) == output)

    @pytest.mark.parametrize(("H", "var", "output"), zip(H, TF_VAR, TF_OUT))
    def test_hamiltonian_dif_tensor(self, H, var, output):
        """Tests that the hamiltonian_expand tape transform is differentiable with the Tensorflow interface"""
        pass

    @pytest.mark.parametrize(("H", "var", "output"), zip(H, TORCH_VAR, TORCH_OUT))
    def test_hamiltonian_def_torch(self, H, var, output):
        """Tests that the hamiltonian_expand tape transform is differentiable with the PyTorch interface"""
        pass
