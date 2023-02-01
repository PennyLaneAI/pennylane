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
"""Unit tests for create_initial_state in devices/qubit."""

import pytest

import numpy as np

import pennylane as qml
from pennylane.devices.qubit import simulate

ml_frameworks_list = [
    "numpy",
    pytest.param("autograd", marks=pytest.mark.autograd),
    pytest.param("jax", marks=pytest.mark.jax),
    pytest.param("torch", marks=pytest.mark.torch),
    pytest.param("tensorflow", marks=pytest.mark.tf),
]


class TestCurrentlyUnsupportedCases:
    def test_hamiltonian_observable(self):
        """Test that measuring hamiltonians gives a NotImplementedError."""

        H = qml.Hamiltonian([2.0], [qml.PauliX(0)])
        qs = qml.tape.QuantumScript(measurements=[qml.expval(H)])
        with pytest.raises(NotImplementedError):
            simulate(qs)

    def test_sample_based_observable(self):
        """Test sample-only measurements raise a notimplementedError."""

        qs = qml.tape.QuantumScript(measurements=[qml.sample(wires=0)])
        with pytest.raises(NotImplementedError):
            simulate(qs)


class TestBasicCircuit:
    @pytest.mark.autograd
    def test_autograd_results_and_backprop(self):
        """Tests execution and gradients with autograd"""
        phi = qml.numpy.array(-0.52)

        def f(x):
            qs = qml.tape.QuantumScript([qml.RX(x, wires=0)], [qml.expval(qml.PauliZ(0))])
            return simulate(qs)[0]

        result = f(phi)
        assert qml.math.allclose(result, np.cos(phi))

        g = qml.grad(f)(phi)
        assert qml.math.allclose(g, -np.sin(phi))

        h = qml.grad(qml.grad(f))(phi)
        assert qml.math.allclose(h, -np.cos(phi))

    @pytest.mark.jax
    @pytest.mark.parametrize("use_jit", (True, False))
    def test_jax_results_and_backprop(self, use_jit):
        """Tests exeuction and gradients with jax."""
        import jax

        phi = jax.numpy.array(0.678)

        def f(x):
            qs = qml.tape.QuantumScript([qml.RX(x, wires=0)], [qml.expval(qml.PauliZ(0))])
            return simulate(qs, like="jax")[0]

        if use_jit:
            f = jax.jit(f)

        result = f(phi)
        assert qml.math.allclose(result, np.cos(phi))

        g = jax.grad(f)(phi)
        assert qml.math.allclose(g, -np.sin(phi))

        h = jax.grad(jax.grad(f))(phi)
        assert qml.math.allclose(h, -np.cos(phi))

    @pytest.mark.torch
    def test_torch_results_and_backprop(self):
        """Tests execution and gradients of a simple circuit with torch."""

        import torch

        phi = torch.tensor(-0.526, requires_grad=True)

        def f(x):
            qs = qml.tape.QuantumScript([qml.RX(x, wires=0)], [qml.expval(qml.PauliZ(0))])
            return simulate(qs, like="torch")[0]

        result = f(phi)
        result.backward()

        assert qml.math.allclose(result, torch.cos(phi))
        assert qml.math.allclose(phi.grad, -torch.sin(phi))

    @pytest.mark.tf
    def test_tf_results_and_backprop(self):
        """Tests execution and gradients of a simple circuit with tensorflow."""
        import tensorflow as tf

        phi = tf.Variable(4.873)

        with tf.GradientTape() as grad_tape:
            qs = qml.tape.QuantumScript([qml.RX(phi, wires=0)], [qml.expval(qml.PauliZ(0))])
            result = simulate(qs, like="tensorflow")[0]

        grads = grad_tape.jacobian(result, [phi])

        assert qml.math.allclose(result, np.cos(phi))
        assert qml.math.allclose(grads[0], -np.sin(phi))
