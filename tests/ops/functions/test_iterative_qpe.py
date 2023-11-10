# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

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
Unit tests for the iterative_qpe function
"""
import pytest

import numpy as np
import pennylane as qml


class TestIQPE:
    """Test to check that the iterative quantum phase estimation function works as expected."""

    @pytest.mark.parametrize("phi", (1.0, 2.0, 3.0))
    def test_compare_qpe(self, phi):
        """Test to check that the results obtained are equivalent to those of QuantumPhaseEstimation"""

        # TODO: When we have general statistics on measurements we can calculate it exactly with qml.probs
        dev = qml.device("default.qubit", shots=10000000)

        @qml.qnode(dev)
        def circuit_iterative():
            # Initial state
            qml.PauliX(wires=[0])

            # Iterative QPE
            measurements = qml.iterative_qpe(qml.RZ(phi, wires=[0]), estimation_wire=[1], iters=3)

            return [qml.sample(op=meas) for meas in measurements]

        sample_list = np.array(circuit_iterative())
        sample_list = sample_list.T
        output = qml.probs().process_samples(np.array(sample_list), wire_order=[0, 1, 2])

        @qml.qnode(dev)
        def circuit_qpe():
            # Initial state
            qml.PauliX(wires=[0])

            # Iterative QPE
            qml.QuantumPhaseEstimation(qml.RZ(phi, wires=[0]), estimation_wires=[1, 2, 3])

            return qml.probs(wires=[1, 2, 3])

        assert np.allclose(np.round(output, 2), np.round(circuit_qpe(), 2))

    @pytest.mark.jax
    def test_check_gradients_jax(self):
        """Test to check that the gradients are correct comparing with the expanded circuit using JAX"""

        import jax

        dev = qml.device("default.qubit")

        @qml.qnode(dev)
        def circuit(theta):
            _ = qml.iterative_qpe(qml.RZ(theta, wires=[0]), [1], iters=2)
            return qml.expval(qml.PauliZ(0))

        @qml.qnode(dev)
        def manual_circuit(phi):
            qml.Hadamard(wires=[1])
            qml.ctrl(qml.RZ(phi, wires=[0]) ** 2, control=[1])
            qml.Hadamard(wires=[1])
            qml.CNOT(wires=[1, 2])
            qml.CNOT(wires=[2, 1])
            qml.Hadamard(wires=[1])
            qml.ctrl(qml.RZ(phi, wires=[0]), control=[1])
            qml.ctrl(qml.PhaseShift(-np.pi / 2, wires=[1]), control=[2])
            qml.Hadamard(wires=[1])
            qml.CNOT(wires=[1, 3])
            qml.CNOT(wires=[3, 1])

            return qml.expval(qml.PauliZ(0))

        phi = jax.numpy.array(1.0)
        assert jax.numpy.isclose(jax.grad(circuit)(phi), jax.grad(manual_circuit)(phi))

    @pytest.mark.torch
    def test_check_gradients_torch(self):
        """Test to check that the gradients are correct comparing with the expanded circuit using PyTorch"""

        import torch

        dev = qml.device("default.qubit")

        @qml.qnode(dev)
        def circuit(theta):
            _ = qml.iterative_qpe(qml.RZ(theta, wires=[0]), [1], iters=2)
            return qml.expval(qml.PauliZ(0))

        @qml.qnode(dev)
        def manual_circuit(phi):
            qml.Hadamard(wires=[1])
            qml.ctrl(qml.RZ(phi, wires=[0]) ** 2, control=[1])
            qml.Hadamard(wires=[1])
            qml.CNOT(wires=[1, 2])
            qml.CNOT(wires=[2, 1])
            qml.Hadamard(wires=[1])
            qml.ctrl(qml.RZ(phi, wires=[0]), control=[1])
            qml.ctrl(qml.PhaseShift(-np.pi / 2, wires=[1]), control=[2])
            qml.Hadamard(wires=[1])
            qml.CNOT(wires=[1, 3])
            qml.CNOT(wires=[3, 1])

            return qml.expval(qml.PauliZ(0))

        phi = torch.tensor(1.0, requires_grad=True)
        assert torch.isclose(torch.func.grad(circuit)(phi), torch.func.grad(manual_circuit)(phi))

    @pytest.mark.tensorflow
    def test_check_gradients_tf(self):
        """Test to check that the gradients are correct comparing with the expanded circuit using TensorFlow"""

        import tensorflow as tf

        def grad(f):
            def wrapper(x):
                with tf.GradientTape() as tape:
                    y = f(x)

                return tape.gradient(y, x)

            return wrapper

        dev = qml.device("default.qubit")

        @qml.qnode(dev)
        def circuit(theta):
            _ = qml.iterative_qpe(qml.RZ(theta, wires=[0]), [1], iters=2)
            return qml.expval(qml.PauliZ(0))

        @qml.qnode(dev)
        def manual_circuit(phi):
            qml.Hadamard(wires=[1])
            qml.ctrl(qml.RZ(phi, wires=[0]) ** 2, control=[1])
            qml.Hadamard(wires=[1])
            qml.CNOT(wires=[1, 2])
            qml.CNOT(wires=[2, 1])
            qml.Hadamard(wires=[1])
            qml.ctrl(qml.RZ(phi, wires=[0]), control=[1])
            qml.ctrl(qml.PhaseShift(-np.pi / 2, wires=[1]), control=[2])
            qml.Hadamard(wires=[1])
            qml.CNOT(wires=[1, 3])
            qml.CNOT(wires=[3, 1])

            return qml.expval(qml.PauliZ(0))

        phi = tf.Variable(1.0)
        assert np.isclose(grad(circuit)(phi), grad(manual_circuit)(phi))

    @pytest.mark.parametrize("iters", (1, 2, 3, 4))
    def test_size_return(self, iters):
        """Test to check that the size of the returned list is correct"""

        dev = qml.device("default.qubit", shots=1)

        @qml.qnode(dev)
        def circuit():
            m = qml.iterative_qpe(qml.RZ(1.0, wires=[0]), [1], iters=iters)
            return [qml.sample(op=meas) for meas in m]

        assert len(circuit()) == iters

    @pytest.mark.parametrize("wire", (1, "a", "abc", 6))
    def test_wires_args(self, wire):
        """Test to check that all types of wires are working"""

        with qml.tape.QuantumTape() as tape:
            qml.iterative_qpe(qml.RZ(1.0, wires=[0]), wire, iters=3)

        assert wire in tape.wires

    @pytest.mark.parametrize("phi", (1.2, 2.3, 3.4))
    def test_measurement_processes_probs(self, phi):
        """Test to check that the measurement process prob works correctly"""

        dev = qml.device("default.qubit")

        @qml.qnode(dev)
        def circuit_qpe():
            # Initial state
            qml.PauliX(wires=[0])

            # Iterative QPE
            qml.QuantumPhaseEstimation(qml.RZ(phi, wires=[0]), estimation_wires=[1, 2, 3])

            return [qml.probs(wires=i) for i in [1, 2, 3]]

        @qml.qnode(dev)
        def circuit_iterative():
            # Initial state
            qml.PauliX(wires=[0])

            # Iterative QPE
            measurements = qml.iterative_qpe(qml.RZ(phi, wires=[0]), estimation_wire=[1], iters=3)

            return [qml.probs(op=i) for i in measurements]

        assert np.allclose(circuit_qpe(), circuit_iterative())

    @pytest.mark.parametrize("phi", (1.2, 2.3, 3.4))
    def test_measurement_processes_expval(self, phi):
        """Test to check that the measurement process expval works correctly"""

        dev = qml.device("default.qubit")

        @qml.qnode(dev)
        def circuit_qpe():
            # Initial state
            qml.PauliX(wires=[0])

            # Iterative QPE
            qml.QuantumPhaseEstimation(qml.RZ(phi, wires=[0]), estimation_wires=[1, 2, 3])

            # We will use the projector as an observable
            return [qml.expval(qml.Hermitian([[0, 0], [0, 1]], wires=i)) for i in [1, 2, 3]]

        @qml.qnode(dev)
        def circuit_iterative():
            # Initial state
            qml.PauliX(wires=[0])

            # Iterative QPE
            measurements = qml.iterative_qpe(qml.RZ(phi, wires=[0]), estimation_wire=[1], iters=3)

            return [qml.expval(op=i) for i in measurements]

        assert np.allclose(circuit_qpe(), circuit_iterative())
