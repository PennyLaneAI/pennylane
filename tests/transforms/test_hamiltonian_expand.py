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
from pennylane import numpy as pnp

"""Defines the device used for all tests"""

dev = qml.device("default.qubit", wires=4)

"""Defines circuits to be used in queueing/output tests"""

with pennylane.tape.QuantumTape() as tape1:
    qml.PauliX(0)
    H1 = qml.Hamiltonian([1.5], [qml.PauliZ(0) @ qml.PauliZ(1)])
    qml.expval(H1)

with pennylane.tape.QuantumTape() as tape2:
    qml.Hadamard(0)
    qml.Hadamard(1)
    qml.PauliZ(1)
    qml.PauliX(2)
    H2 = qml.Hamiltonian(
        [1, 3, -2, 1, 1],
        [
            qml.PauliX(0) @ qml.PauliZ(2),
            qml.PauliZ(2),
            qml.PauliX(0),
            qml.PauliX(2),
            qml.PauliZ(0) @ qml.PauliX(1),
        ],
    )
    qml.expval(H2)

H3 = 1.5 * qml.PauliZ(0) @ qml.PauliZ(1) + 0.3 * qml.PauliX(1)

with qml.tape.QuantumTape() as tape3:
    qml.PauliX(0)
    qml.expval(H3)


H4 = (
    qml.PauliX(0) @ qml.PauliZ(2)
    + 3 * qml.PauliZ(2)
    - 2 * qml.PauliX(0)
    + qml.PauliZ(2)
    + qml.PauliZ(2)
)
H4 += qml.PauliZ(0) @ qml.PauliX(1) @ qml.PauliY(2)

with qml.tape.QuantumTape() as tape4:
    qml.Hadamard(0)
    qml.Hadamard(1)
    qml.PauliZ(1)
    qml.PauliX(2)

    qml.expval(H4)

TAPES = [tape1, tape2, tape3, tape4]
OUTPUTS = [-1.5, -6, -1.5, -8]


class TestHamiltonianExpand:
    """Tests for the hamiltonian_expand transform"""

    @pytest.mark.parametrize(("tape", "output"), zip(TAPES, OUTPUTS))
    def test_hamiltonians(self, tape, output):
        """Tests that the hamiltonian_expand transform returns the correct value"""

        tapes, fn = qml.transforms.hamiltonian_expand(tape)
        results = dev.batch_execute(tapes)
        expval = fn(results)

        assert np.isclose(output, expval)

        qs = qml.tape.QuantumScript(tape.operations, tape.measurements)
        tapes, fn = qml.transforms.hamiltonian_expand(qs)
        results = dev.batch_execute(tapes)
        expval = fn(results)

        assert np.isclose(output, expval)

    @pytest.mark.parametrize(("tape", "output"), zip(TAPES, OUTPUTS))
    def test_hamiltonians_no_grouping(self, tape, output):
        """Tests that the hamiltonian_expand transform returns the correct value
        if we switch grouping off"""

        tapes, fn = qml.transforms.hamiltonian_expand(tape, group=False)
        results = dev.batch_execute(tapes)
        expval = fn(results)

        assert np.isclose(output, expval)

        qs = qml.tape.QuantumScript(tape.operations, tape.measurements)
        tapes, fn = qml.transforms.hamiltonian_expand(qs, group=False)
        results = dev.batch_execute(tapes)
        expval = fn(results)

        assert np.isclose(output, expval)

    def test_grouping_is_used(self):
        """Test that the grouping in a Hamiltonian is used"""
        H = qml.Hamiltonian(
            [1.0, 2.0, 3.0], [qml.PauliZ(0), qml.PauliX(1), qml.PauliX(0)], grouping_type="qwc"
        )
        assert H.grouping_indices is not None

        with qml.tape.QuantumTape() as tape:
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.PauliX(wires=2)
            qml.expval(H)

        tapes, fn = qml.transforms.hamiltonian_expand(tape, group=False)
        assert len(tapes) == 2

        qs = qml.tape.QuantumScript(tape.operations, tape.measurements)
        tapes, fn = qml.transforms.hamiltonian_expand(qs, group=False)
        assert len(tapes) == 2

    def test_number_of_tapes(self):
        """Tests that the the correct number of tapes is produced"""

        H = qml.Hamiltonian([1.0, 2.0, 3.0], [qml.PauliZ(0), qml.PauliX(1), qml.PauliX(0)])

        with qml.tape.QuantumTape() as tape:
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.PauliX(wires=2)
            qml.expval(H)

        tapes, fn = qml.transforms.hamiltonian_expand(tape, group=False)
        assert len(tapes) == 3

        tapes, fn = qml.transforms.hamiltonian_expand(tape, group=True)
        assert len(tapes) == 2

    def test_number_of_qscripts(self):
        """Tests the correct number of quantum scripts are produced."""

        H = qml.Hamiltonian([1.0, 2.0, 3.0], [qml.PauliZ(0), qml.PauliX(1), qml.PauliX(0)])
        qs = qml.tape.QuantumScript(measurements=[qml.expval(H)])

        tapes, fn = qml.transforms.hamiltonian_expand(qs, group=False)
        assert len(tapes) == 3

        tapes, fn = qml.transforms.hamiltonian_expand(qs, group=True)
        assert len(tapes) == 2

    def test_hamiltonian_error(self):

        with pennylane.tape.QuantumTape() as tape:
            qml.expval(qml.PauliZ(0))

        with pytest.raises(ValueError, match=r"Passed tape must end in"):
            tapes, fn = qml.transforms.hamiltonian_expand(tape)

    @pytest.mark.autograd
    def test_hamiltonian_dif_autograd(self, tol):
        """Tests that the hamiltonian_expand tape transform is differentiable with the Autograd interface"""

        H = qml.Hamiltonian(
            [-0.2, 0.5, 1], [qml.PauliX(1), qml.PauliZ(1) @ qml.PauliY(2), qml.PauliZ(0)]
        )

        var = pnp.array([0.1, 0.67, 0.3, 0.4, -0.5, 0.7, -0.2, 0.5, 1.0], requires_grad=True)
        output = 0.42294409781940356
        output2 = [
            9.68883500e-02,
            -2.90832724e-01,
            -1.04448033e-01,
            -1.94289029e-09,
            3.50307411e-01,
            -3.41123470e-01,
            0.0,
            -0.43657,
            0.64123,
        ]

        with qml.tape.QuantumTape() as tape:
            for i in range(2):
                qml.RX(np.array(0), wires=0)
                qml.RX(np.array(0), wires=1)
                qml.RX(np.array(0), wires=2)
                qml.CNOT(wires=[0, 1])
                qml.CNOT(wires=[1, 2])
                qml.CNOT(wires=[2, 0])

            qml.expval(H)

        def cost(x):
            tape.set_parameters(x, trainable_only=False)
            tapes, fn = qml.transforms.hamiltonian_expand(tape)
            res = qml.execute(tapes, dev, qml.gradients.param_shift)
            return fn(res)

        assert np.isclose(cost(var), output)

        grad = qml.grad(cost)(var)
        assert len(grad) == len(output2)
        for g, o in zip(grad, output2):
            assert np.allclose(g, o, atol=tol)

    @pytest.mark.tf
    def test_hamiltonian_dif_tensorflow(self):
        """Tests that the hamiltonian_expand tape transform is differentiable with the Tensorflow interface"""

        import tensorflow as tf

        H = qml.Hamiltonian(
            [-0.2, 0.5, 1], [qml.PauliX(1), qml.PauliZ(1) @ qml.PauliY(2), qml.PauliZ(0)]
        )
        var = tf.Variable([[0.1, 0.67, 0.3], [0.4, -0.5, 0.7]], dtype=tf.float64)
        output = 0.42294409781940356
        output2 = [
            9.68883500e-02,
            -2.90832724e-01,
            -1.04448033e-01,
            -1.94289029e-09,
            3.50307411e-01,
            -3.41123470e-01,
        ]

        with tf.GradientTape() as gtape:
            with qml.tape.QuantumTape() as tape:
                for i in range(2):
                    qml.RX(var[i, 0], wires=0)
                    qml.RX(var[i, 1], wires=1)
                    qml.RX(var[i, 2], wires=2)
                    qml.CNOT(wires=[0, 1])
                    qml.CNOT(wires=[1, 2])
                    qml.CNOT(wires=[2, 0])
                qml.expval(H)

            tapes, fn = qml.transforms.hamiltonian_expand(tape)
            res = fn(qml.execute(tapes, dev, qml.gradients.param_shift, interface="tf"))

            assert np.isclose(res, output)

            g = gtape.gradient(res, var)
            assert np.allclose(list(g[0]) + list(g[1]), output2)
