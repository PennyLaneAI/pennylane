# Copyright 2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the classical shadows transforms"""

# pylint: disable=too-few-public-methods
import pytest

import pennylane as qml
from pennylane import numpy as np
from pennylane.shadows.transforms import _replace_obs


def hadamard_circuit(wires, shots=10000, interface="autograd"):
    """Hadamard circuit to put all qubits in equal superposition (locally)"""
    dev = qml.device("default.qubit", wires=wires)

    @qml.set_shots(shots)
    @qml.qnode(dev, interface=interface)
    def circuit():
        for i in range(wires):
            qml.Hadamard(wires=i)
        return qml.classical_shadow(wires=range(wires))

    return circuit


def max_entangled_circuit(wires, shots=10000, interface="autograd"):
    """maximally entangled state preparation circuit"""
    dev = qml.device("default.qubit", wires=wires)

    @qml.set_shots(shots)
    @qml.qnode(dev, interface=interface)
    def circuit():
        qml.Hadamard(wires=0)
        for i in range(1, wires):
            qml.CNOT(wires=[0, i])
        return qml.classical_shadow(wires=range(wires))

    return circuit


def qft_circuit(wires, shots=10000, interface="autograd"):
    """Quantum Fourier Transform circuit"""
    dev = qml.device("default.qubit", wires=wires)

    one_state = np.zeros(wires)
    one_state[-1] = 1

    @qml.set_shots(shots)
    @qml.qnode(dev, interface=interface)
    def circuit():
        qml.BasisState(one_state, wires=range(wires))
        qml.QFT(wires=range(wires))
        return qml.classical_shadow(wires=range(wires))

    return circuit


def basic_entangler_circuit(n_wires, shots=10000, interface="autograd"):
    dev = qml.device("default.qubit", wires=n_wires)

    @qml.set_shots(shots)
    @qml.qnode(dev, interface=interface)
    def circuit(x):
        qml.BasicEntanglerLayers(weights=x, wires=range(n_wires))
        return qml.classical_shadow(wires=range(n_wires))

    return circuit


def basic_entangler_circuit_exact_state(n_wires, sub_wires, interface="autograd"):
    dev = qml.device("default.qubit", wires=n_wires)

    @qml.qnode(dev, interface=interface)
    def circuit(x):
        qml.BasicEntanglerLayers(weights=x, wires=range(n_wires))
        return qml.density_matrix(sub_wires)

    return circuit


def basic_entangler_circuit_exact_expval(n_wires, interface="autograd"):
    dev = qml.device("default.qubit", wires=n_wires)

    @qml.qnode(dev, interface=interface)
    def circuit(x, obs):
        qml.BasicEntanglerLayers(weights=x, wires=range(n_wires))
        return [qml.expval(ob) for ob in obs]

    return circuit


class TestReplaceObs:
    """Test that the _replace_obs transform works as expected"""

    def test_replace_tape(self):
        """Test that the transform works for tapes"""
        tape = qml.tape.QuantumScript([], [qml.classical_shadow(wires=0)])
        new_tapes, _ = _replace_obs(tape, qml.probs, wires=0)

        assert len(new_tapes) == 1
        assert len(new_tapes[0].operations) == 0
        assert len(new_tapes[0].observables) == 1
        assert isinstance(new_tapes[0].observables[0], qml.measurements.ProbabilityMP)

    def test_replace_qnode(self):
        """Test that the transform works for QNodes"""
        circuit = hadamard_circuit(2, shots=1000)
        circuit = _replace_obs(circuit, qml.probs, wires=[0, 1])
        res = circuit()

        assert isinstance(res, np.ndarray)
        assert res.shape == (4,)


@pytest.mark.autograd
class TestStateForward:
    """Test that the state reconstruction is correct for a variety of states"""

    @pytest.mark.parametrize("wires", [1, 3])
    @pytest.mark.parametrize("diffable", [True, False])
    def test_hadamard_state(self, wires, diffable):
        """Test that the state reconstruction is correct for a uniform
        superposition of qubits"""
        circuit = hadamard_circuit(wires)
        circuit = qml.shadows.shadow_state(circuit, wires=range(wires), diffable=diffable)

        actual = circuit()
        expected = np.ones((2**wires, 2**wires)) / (2**wires)

        assert qml.math.allclose(actual, expected, atol=1e-1)

    @pytest.mark.parametrize("wires", [1, 3])
    @pytest.mark.parametrize("diffable", [True, False])
    def test_max_entangled_state(self, wires, diffable):
        """Test that the state reconstruction is correct for a maximally entangled state"""
        circuit = max_entangled_circuit(wires)
        circuit = qml.shadows.shadow_state(circuit, wires=range(wires), diffable=diffable)

        actual = circuit()
        expected = np.zeros((2**wires, 2**wires))
        expected[np.array([0, 0, -1, -1]), np.array([0, -1, 0, -1])] = 0.5

        assert qml.math.allclose(actual, expected, atol=1e-1)

    @pytest.mark.parametrize("diffable", [True, False])
    def test_partial_state(self, diffable):
        """Test that the state reconstruction is correct for a subset
        of the qubits"""
        wires_list = [[0], [0, 1]]

        circuit = max_entangled_circuit(3)
        circuit = qml.shadows.shadow_state(circuit, wires=wires_list, diffable=diffable)

        actual = circuit()

        expected = [
            np.array([[0.5, 0], [0, 0.5]]),
            np.array([[0.5, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0.5]]),
        ]

        assert qml.math.allclose(actual[0], expected[0], atol=1e-1)
        assert qml.math.allclose(actual[1], expected[1], atol=1e-1)

    def test_large_state_warning(self):
        """Test that a warning is raised when the system to get the state
        of is large"""
        circuit = hadamard_circuit(8, shots=1)
        tape = qml.workflow.construct_tape(circuit)()

        msg = "Differentiable state reconstruction for more than 8 qubits is not recommended"
        with pytest.warns(UserWarning, match=msg):
            qml.shadows.shadow_state(tape, wires=[0, 1, 2, 3, 4, 5, 6, 7], diffable=True)

    def test_multi_measurement_error(self):
        """Test that an error is raised when classical shadows is returned
        with other measurement processes"""
        dev = qml.device("default.qubit", wires=2)

        @qml.set_shots(100)
        @qml.qnode(dev)
        def circuit_shadow():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.classical_shadow(wires=[0, 1]), qml.expval(qml.PauliZ(0))

        res = circuit_shadow()
        assert isinstance(res, tuple) and len(res) == 2

        @qml.set_shots(100)
        @qml.qnode(dev)
        def circuit_expval():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.shadow_expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(0))

        res = circuit_expval()
        assert isinstance(res, tuple) and len(res) == 2


@pytest.mark.all_interfaces
class TestStateForwardInterfaces:
    """Test that state reconstruction works for all interfaces"""

    @pytest.mark.parametrize("interface", ["autograd", "jax", "torch"])
    @pytest.mark.parametrize("diffable", [True, False])
    def test_qft_state(self, interface, diffable):
        """Test that the state reconstruction is correct for a QFT state"""
        circuit = qft_circuit(3, interface=interface)
        circuit = qml.shadows.shadow_state(circuit, wires=[0, 1, 2], diffable=diffable)

        actual = circuit()
        expected = np.exp(np.arange(8) * 2j * np.pi / 8) / np.sqrt(8)
        expected = np.outer(expected, np.conj(expected))

        assert qml.math.allclose(actual, expected, atol=1e-1)


class TestStateBackward:
    """Test that the gradient of the state reconstruction is correct"""

    # make rotations close to pi / 2 to ensure gradients are not too small
    x = np.random.uniform(
        0.8, 2, size=qml.BasicEntanglerLayers.shape(n_layers=1, n_wires=3)
    ).tolist()

    @pytest.mark.autograd
    def test_backward_autograd(self):
        """Test the gradient of the state for the autograd interface"""
        # pylint: disable=cell-var-from-loop
        shadow_circuit = basic_entangler_circuit(3, shots=20000, interface="autograd")

        sub_wires = [[0, 1], [1, 2]]
        shadow_circuit = qml.shadows.shadow_state(shadow_circuit, wires=sub_wires, diffable=True)

        x = np.array(self.x, requires_grad=True)

        # for autograd in particular, take only the real part since it doesn't
        # support complex differentiation
        actual = qml.jacobian(lambda x: qml.math.real(qml.math.stack(shadow_circuit(x))))(x)

        for act, w in zip(qml.math.unstack(actual), sub_wires):
            exact_circuit = basic_entangler_circuit_exact_state(3, w, "autograd")
            expected = qml.jacobian(lambda x: qml.math.real(exact_circuit(x)))(x)

            assert qml.math.allclose(act, expected, atol=1e-1)

    @pytest.mark.jax
    def test_backward_jax(self):
        """Test the gradient of the state for the JAX interface"""
        # pylint: disable=cell-var-from-loop
        import jax
        from jax import numpy as jnp

        shadow_circuit = basic_entangler_circuit(3, shots=20000, interface="jax")

        sub_wires = [[0, 1], [1, 2]]
        shadow_circuit = qml.shadows.shadow_state(shadow_circuit, wires=sub_wires, diffable=True)

        x = jnp.array(self.x)
        actual = jax.jacobian(lambda x: qml.math.real(qml.math.stack(shadow_circuit(x))))(x)

        for act, w in zip(qml.math.unstack(actual), sub_wires):
            exact_circuit = basic_entangler_circuit_exact_state(3, w, "jax")
            expected = jax.jacobian(lambda x: qml.math.real(exact_circuit(x)))(x)

            assert qml.math.allclose(act, expected, atol=1e-1)

    @pytest.mark.tf
    def test_backward_tf(self):
        """Test the gradient of the state for the tensorflow interface"""
        import tensorflow as tf

        shadow_circuit = basic_entangler_circuit(3, shots=20000, interface="tf")

        sub_wires = [[0, 1], [1, 2]]
        shadow_circuit = qml.shadows.shadow_state(shadow_circuit, wires=sub_wires, diffable=True)

        x = tf.Variable(self.x, dtype="float64")

        with tf.GradientTape() as tape:
            out = qml.math.stack(shadow_circuit(x))

        actual = tape.jacobian(out, x)

        for act, w in zip(qml.math.unstack(actual), sub_wires):
            exact_circuit = basic_entangler_circuit_exact_state(3, w, "tf")

            with tf.GradientTape() as tape2:
                out2 = exact_circuit(x)

            expected = tape2.jacobian(out2, x)

            assert qml.math.allclose(act, expected, atol=1e-1)

    @pytest.mark.torch
    @pytest.mark.xfail(reason="see pytorch/pytorch/issues/94397")
    def test_backward_torch(self):
        """Test the gradient of the state for the torch interface"""
        import torch

        shadow_circuit = basic_entangler_circuit(3, shots=20000, interface="torch")

        sub_wires = [[0, 1], [1, 2]]
        shadow_circuit = qml.shadows.shadow_state(shadow_circuit, wires=sub_wires, diffable=True)

        x = torch.tensor(self.x, requires_grad=True)

        actual = torch.autograd.functional.jacobian(lambda x: qml.math.stack(shadow_circuit(x)), x)

        for act, w in zip(qml.math.unstack(actual), sub_wires):
            exact_circuit = basic_entangler_circuit_exact_state(3, w, "torch")
            expected = torch.autograd.functional.jacobian(exact_circuit, x)

            assert qml.math.allclose(act, expected, atol=1e-1)
