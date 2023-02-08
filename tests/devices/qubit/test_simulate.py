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


def test_custom_operation():
    """Test execution works with a manually defined operator if it has a matrix."""

    # pylint: disable=too-few-public-methods
    class MyOperator(qml.operation.Operator):
        num_wires = 1

        @staticmethod
        def compute_matrix():
            return qml.PauliX.compute_matrix()

    qs = qml.tape.QuantumScript([MyOperator(0)], [qml.expval(qml.PauliZ(0))])

    result = simulate(qs)
    assert qml.math.allclose(result[0], -1.0)


class TestBasicCircuit:
    """Tests a basic circuit with one rx gate and two simple expectation values."""

    def test_basic_circuit_numpy(self):
        """Test execution with a basic circuit."""
        phi = np.array(0.397)
        qs = qml.tape.QuantumScript(
            [qml.RX(phi, wires=0)], [qml.expval(qml.PauliY(0)), qml.expval(qml.PauliZ(0))]
        )
        result = simulate(qs)

        assert isinstance(result, tuple)
        assert len(result) == 2

        assert np.allclose(result[0], -np.sin(phi))
        assert np.allclose(result[1], np.cos(phi))

    @pytest.mark.autograd
    def test_autograd_results_and_backprop(self):
        """Tests execution and gradients with autograd"""
        phi = qml.numpy.array(-0.52)

        def f(x):
            qs = qml.tape.QuantumScript(
                [qml.RX(x, wires=0)], [qml.expval(qml.PauliY(0)), qml.expval(qml.PauliZ(0))]
            )
            return qml.numpy.array(simulate(qs))

        result = f(phi)
        expected = np.array([-np.sin(phi), np.cos(phi)])
        assert qml.math.allclose(result, expected)

        g = qml.jacobian(f)(phi)
        expected = np.array([-np.cos(phi), -np.sin(phi)])
        assert qml.math.allclose(g, expected)

    @pytest.mark.jax
    @pytest.mark.parametrize("use_jit", (True, False))
    def test_jax_results_and_backprop(self, use_jit):
        """Tests exeuction and gradients with jax."""
        import jax

        phi = jax.numpy.array(0.678)

        def f(x):
            qs = qml.tape.QuantumScript(
                [qml.RX(x, wires=0)], [qml.expval(qml.PauliY(0)), qml.expval(qml.PauliZ(0))]
            )
            return simulate(qs)

        if use_jit:
            f = jax.jit(f)

        result = f(phi)
        assert qml.math.allclose(result[0], -np.sin(phi))
        assert qml.math.allclose(result[1], np.cos(phi))

        g = jax.jacobian(f)(phi)
        assert qml.math.allclose(g[0], -np.cos(phi))
        assert qml.math.allclose(g[1], -np.sin(phi))

    @pytest.mark.torch
    def test_torch_results_and_backprop(self):
        """Tests execution and gradients of a simple circuit with torch."""

        import torch

        phi = torch.tensor(-0.526, requires_grad=True)

        def f(x):
            qs = qml.tape.QuantumScript(
                [qml.RX(x, wires=0)], [qml.expval(qml.PauliY(0)), qml.expval(qml.PauliZ(0))]
            )
            return simulate(qs)

        result = f(phi)
        assert qml.math.allclose(result[0], -torch.sin(phi))
        assert qml.math.allclose(result[1], torch.cos(phi))

        g = torch.autograd.functional.jacobian(f, phi + 0j)
        assert qml.math.allclose(g[0], -torch.cos(phi))
        assert qml.math.allclose(g[1], -torch.sin(phi))

    # pylint: disable=invalid-unary-operand-type
    @pytest.mark.tf
    def test_tf_results_and_backprop(self):
        """Tests execution and gradients of a simple circuit with tensorflow."""
        import tensorflow as tf

        phi = tf.Variable(4.873)

        with tf.GradientTape(persistent=True) as grad_tape:
            qs = qml.tape.QuantumScript(
                [qml.RX(phi, wires=0)], [qml.expval(qml.PauliY(0)), qml.expval(qml.PauliZ(0))]
            )
            result = simulate(qs)

        assert qml.math.allclose(result[0], -tf.sin(phi))
        assert qml.math.allclose(result[1], tf.cos(phi))

        grad0 = grad_tape.jacobian(result[0], [phi])
        grad1 = grad_tape.jacobian(result[1], [phi])

        assert qml.math.allclose(grad0[0], -tf.cos(phi))
        assert qml.math.allclose(grad1[0], -tf.sin(phi))


# pylint: disable=too-few-public-methods
class TestStatePrep:
    """Tests integration with various state prep methods."""

    def test_basis_state(self):
        """Test that the basis state method operator the quantum script into the right state."""
        qs = qml.tape.QuantumScript(
            measurements=[qml.probs(wires=(0, 1, 2))], prep=[qml.BasisState([0, 1], wires=(0, 1))]
        )
        probs = simulate(qs)[0]
        expected = np.zeros(8)
        expected[2] = 1.0
        assert qml.math.allclose(probs, expected)


class TestQInfoMeasurements:

    measurements = [
        qml.density_matrix(0),
        qml.density_matrix(1),
        qml.density_matrix((0, 1)),
        qml.vn_entropy(0),
        qml.vn_entropy(1),
        qml.mutual_info(0, 1),
    ]

    def expected_results(self, phi):

        density_i = np.array([[np.cos(phi / 2) ** 2, 0], [0, np.sin(phi / 2) ** 2]])
        density_both = np.array(
            [
                [np.cos(phi / 2) ** 2, 0, 0, 0.0 + np.sin(phi) * 0.5j],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0.0 - np.sin(phi) * 0.5j, 0, 0, np.sin(phi / 2) ** 2],
            ]
        )
        eig_1 = (1 + np.sqrt(1 - 4 * np.cos(phi / 2) ** 2 * np.sin(phi / 2) ** 2)) / 2
        eig_2 = (1 - np.sqrt(1 - 4 * np.cos(phi / 2) ** 2 * np.sin(phi / 2) ** 2)) / 2
        eigs = [eig_1, eig_2]
        eigs = [eig for eig in eigs if eig > 0]
        rho_log_rho = eigs * np.log(eigs)
        expected_entropy = -np.sum(rho_log_rho)
        mutual_info = 2 * expected_entropy

        return (density_i, density_i, density_both, expected_entropy, expected_entropy, mutual_info)

    def calculate_entropy_grad(self, phi):
        eig_1 = (1 + np.sqrt(1 - 4 * np.cos(phi / 2) ** 2 * np.sin(phi / 2) ** 2)) / 2
        eig_2 = (1 - np.sqrt(1 - 4 * np.cos(phi / 2) ** 2 * np.sin(phi / 2) ** 2)) / 2
        eigs = [eig_1, eig_2]
        eigs = np.maximum(eigs, 1e-08)

        return -(
            (np.log(eigs[0]) + 1)
            * (np.sin(phi / 2) ** 3 * np.cos(phi / 2) - np.sin(phi / 2) * np.cos(phi / 2) ** 3)
            / np.sqrt(1 - 4 * np.cos(phi / 2) ** 2 * np.sin(phi / 2) ** 2)
        ) - (
            (np.log(eigs[1]) + 1)
            * (np.sin(phi / 2) * np.cos(phi / 2) * (np.cos(phi / 2) ** 2 - np.sin(phi / 2) ** 2))
            / np.sqrt(1 - 4 * np.cos(phi / 2) ** 2 * np.sin(phi / 2) ** 2)
        )

    def expected_grad(self, phi):

        p_2 = phi / 2
        g_density_i = np.array([[-np.cos(p_2) * np.sin(p_2), 0], [0, np.sin(p_2) * np.cos(p_2)]])
        g_density_both = np.array(
            [
                [-np.cos(p_2) * np.sin(p_2), 0, 0, 0.0 + 0.5j * np.cos(phi)],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0.0 - 0.5j * np.cos(phi), 0, 0, np.sin(p_2) * np.cos(p_2)],
            ]
        )
        g_entropy = self.calculate_entropy_grad(phi)
        g_mutual_info = 2 * g_entropy
        return (g_density_i, g_density_i, g_density_both, g_entropy, g_entropy, g_mutual_info)

    def test_qinfo_numpy(self):
        """Test quantum info measurements with numpy"""
        phi = -0.623
        qs = qml.tape.QuantumScript([qml.IsingXX(phi, wires=(0, 1))], self.measurements)

        results = simulate(qs)
        for val1, val2 in zip(results, self.expected_results(phi)):
            assert qml.math.allclose(val1, val2)

    @pytest.mark.jax
    @pytest.mark.parametrize("use_jit", (True, False))
    def test_qinfo_jax(self, use_jit):
        """Test qinfo meausrements work with jitting."""

        import jax

        def f(x):
            qs = qml.tape.QuantumScript([qml.IsingXX(x, wires=(0, 1))], self.measurements)
            return simulate(qs)

        if use_jit:
            f = jax.jit(f)

        phi = jax.numpy.array(-0.792 + 0j)

        results = f(phi)
        for val1, val2 in zip(results, self.expected_results(phi)):
            assert qml.math.allclose(val1, val2)

        def complex_out(phi):
            return tuple(res + 0j for res in f(phi))

        grads = jax.jacobian(complex_out, holomorphic=True)(phi)
        expected_grads = self.expected_grad(phi)

        # Writing this way makes it easier to figure out which is failing
        # density 0
        assert qml.math.allclose(grads[0], expected_grads[0])
        # density 1
        assert qml.math.allclose(grads[1], expected_grads[1])
        # density both
        # this is currently broken for some strange reason
        assert qml.math.allclose(grads[2], expected_grads[2])
        # entropy 0
        assert qml.math.allclose(grads[3], expected_grads[3])
        # entropy 1
        assert qml.math.allclose(grads[4], expected_grads[4])
        # mutual info
        assert qml.math.allclose(grads[5], expected_grads[5])
