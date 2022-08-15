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

"""Unit tests for the classical shadows class"""
# pylint:disable=no-self-use, import-outside-toplevel, redefined-outer-name, unpacking-non-sequence, too-few-public-methods, not-an-iterable, inconsistent-return-statements

import pytest

import pennylane as qml
import pennylane.numpy as np
from pennylane.shadows import ClassicalShadow
from pennylane.shadows import median_of_means, pauli_expval

np.random.seed(777)

wires = range(3)
shots = 10000
dev = qml.device("default.qubit", wires=wires, shots=shots)


@qml.qnode(dev)
def qnode(n_wires):
    """Hadamard gate on all wires"""
    for i in range(n_wires):
        qml.Hadamard(i)
    return qml.classical_shadow(wires=range(n_wires))


shadows = [ClassicalShadow(*qnode(n_wires)) for n_wires in range(2, 3)]


class TestUnitTestClassicalShadows:
    """Unit Tests for ClassicalShadow class"""

    @pytest.mark.parametrize("shadow", shadows)
    def test_unittest_snapshots(self, shadow):
        """Test the output shape of snapshots method"""
        T, n = shadow.bitstrings.shape
        assert (T, n) == shadow.recipes.shape
        assert shadow.local_snapshots().shape == (T, n, 2, 2)
        assert shadow.global_snapshots().shape == (T, 2**n, 2**n)


class TestIntegrationShadows:
    """Integration tests for classical shadows class"""

    @pytest.mark.parametrize("shadow", shadows)
    def test_pauli_string_expval(self, shadow):
        """Testing the output of expectation values match those of exact evaluation"""

        o1 = qml.PauliX(0)
        res1 = shadow.expval(o1, k=2)

        o2 = qml.PauliX(0) @ qml.PauliX(1)
        res2 = shadow.expval(o2, k=2)

        res_exact = 1.0
        assert qml.math.allclose(res1, res_exact, atol=1e-1)
        assert qml.math.allclose(res2, res_exact, atol=1e-1)

    Hs = [
        qml.PauliX(0),
        qml.PauliX(0) @ qml.PauliX(1),
        1.0 * qml.PauliX(0),
        0.5 * qml.PauliX(1) + 0.5 * qml.PauliX(1),
        qml.Hamiltonian([1.0], [qml.PauliX(0) @ qml.PauliX(1)]),
    ]

    @pytest.mark.parametrize("H", Hs)
    @pytest.mark.parametrize("shadow", shadows)
    def test_expval_input_types(self, shadow, H):
        """Test ClassicalShadow.expval can handle different inputs"""
        assert qml.math.allclose(shadow.expval(H, k=2), 1.0, atol=1e-1)

    def test_reconstruct_bell_state(self):
        """Test that a bell state can be faithfully reconstructed"""
        wires = range(2)

        dev = qml.device("default.qubit", wires=wires, shots=10000)

        @qml.qnode(dev)
        def qnode(n_wires):
            qml.Hadamard(0)
            qml.CNOT(wires=[0, 1])
            return qml.classical_shadow(wires=range(n_wires))

        # should prepare the bell state
        bitstrings, recipes = qnode(2)
        shadow = ClassicalShadow(bitstrings, recipes)
        global_snapshots = shadow.global_snapshots()

        state = np.sum(global_snapshots, axis=0) / shadow.snapshots
        bell_state = np.array([[0.5, 0, 0, 0.5], [0, 0, 0, 0], [0, 0, 0, 0], [0.5, 0, 0, 0.5]])
        assert qml.math.allclose(state, bell_state, atol=1e-1)

        # reduced state should yield maximally mixed state
        local_snapshots = shadow.local_snapshots(wires=[0])
        assert qml.math.allclose(np.mean(local_snapshots, axis=0)[0], 0.5 * np.eye(2), atol=1e-1)

        # alternative computation
        bitstrings, recipes = qnode(1)
        shadow = ClassicalShadow(bitstrings, recipes)
        global_snapshots = shadow.global_snapshots()
        local_snapshots = shadow.local_snapshots(wires=[0])

        state = np.sum(global_snapshots, axis=0) / shadow.snapshots
        assert qml.math.allclose(state, 0.5 * np.eye(2), atol=1e-1)


def hadamard_circuit(wires, shots=10000, interface="autograd"):
    """Hadamard circuit to put all qubits in equal superposition (locally)"""
    dev = qml.device("default.qubit", wires=wires, shots=shots)

    @qml.qnode(dev, interface=interface)
    def circuit():
        for i in range(wires):
            qml.Hadamard(wires=i)
        return qml.classical_shadow(wires=range(wires))

    return circuit


def max_entangled_circuit(wires, shots=10000, interface="autograd"):
    """maximally entangled state preparation circuit"""
    dev = qml.device("default.qubit", wires=wires, shots=shots)

    @qml.qnode(dev, interface=interface)
    def circuit():
        qml.Hadamard(wires=0)
        for i in range(1, wires):
            qml.CNOT(wires=[0, i])
        return qml.classical_shadow(wires=range(wires))

    return circuit


def qft_circuit(wires, shots=10000, interface="autograd"):
    """Quantum Fourier Transform circuit"""
    dev = qml.device("default.qubit", wires=wires, shots=shots)

    one_state = np.zeros(wires)
    one_state[-1] = 1

    @qml.qnode(dev, interface=interface)
    def circuit():
        qml.BasisState(one_state, wires=range(wires))
        qml.QFT(wires=range(wires))
        return qml.classical_shadow(wires=range(wires))

    return circuit


# marked slow because state reconstruction for high number of qubits is slow
@pytest.mark.slow
@pytest.mark.all_interfaces
class TestStateReconstruction:
    """Test that the state reconstruction is correct for a variety of states"""

    @pytest.mark.parametrize("wires", [1, 2, 3, 4])
    @pytest.mark.parametrize("interface", ["autograd", "jax", "tf", "torch"])
    def test_hadamard_reconstruction(self, wires, interface):
        """Test that the state reconstruction is correct for a uniform
        superposition of qubits"""
        circuit = hadamard_circuit(wires, interface=interface)
        bits, recipes = circuit()
        shadow = ClassicalShadow(bits, recipes)

        state = shadow.global_snapshots()
        assert state.shape == (10000, 2**wires, 2**wires)

        state = np.mean(state, axis=0)
        expected = np.ones((2**wires, 2**wires)) / (2**wires)

        assert qml.math.allclose(state, expected, atol=1e-1)

    @pytest.mark.parametrize("wires", [1, 2, 3, 4])
    @pytest.mark.parametrize("interface", ["autograd", "jax", "tf", "torch"])
    def test_max_entangled_reconstruction(self, wires, interface):
        """Test that the state reconstruction is correct for a maximally
        entangled state"""
        circuit = max_entangled_circuit(wires, interface=interface)
        bits, recipes = circuit()
        shadow = ClassicalShadow(bits, recipes)

        state = shadow.global_snapshots()
        assert state.shape == (10000, 2**wires, 2**wires)

        state = np.mean(state, axis=0)
        expected = np.zeros((2**wires, 2**wires))
        expected[np.array([0, 0, -1, -1]), np.array([0, -1, 0, -1])] = 0.5

        assert qml.math.allclose(state, expected, atol=1e-1)

    @pytest.mark.parametrize("wires", [1, 2, 3, 4])
    @pytest.mark.parametrize("interface", ["autograd", "jax", "tf", "torch"])
    def test_qft_reconstruction(self, wires, interface):
        """Test that the state reconstruction is correct for a QFT state"""
        circuit = qft_circuit(wires, interface=interface)
        bits, recipes = circuit()
        shadow = ClassicalShadow(bits, recipes)

        state = shadow.global_snapshots()
        assert state.shape == (10000, 2**wires, 2**wires)

        state = np.mean(state, axis=0)
        expected = np.exp(np.arange(2**wires) * 2j * np.pi / (2**wires)) / (2 ** (wires / 2))
        expected = np.outer(expected, np.conj(expected))

        assert qml.math.allclose(state, expected, atol=1e-1)

    @pytest.mark.parametrize("wires", [1, 2, 3, 4])
    @pytest.mark.parametrize("interface", ["autograd", "jax", "tf", "torch"])
    @pytest.mark.parametrize("snapshots", [10, 100, 1000])
    def test_subset_reconstruction_integer(self, wires, interface, snapshots):
        """Test that the state reconstruction is correct for different numbers
        of used snapshots"""
        circuit = hadamard_circuit(wires, interface=interface)
        bits, recipes = circuit()
        shadow = ClassicalShadow(bits, recipes)

        state = shadow.global_snapshots(snapshots=snapshots)
        assert state.shape == (snapshots, 2**wires, 2**wires)

    @pytest.mark.parametrize("wires", [1, 2, 3, 4])
    @pytest.mark.parametrize("interface", ["autograd", "jax", "tf", "torch"])
    def test_subset_reconstruction_iterable(self, wires, interface):
        """Test that the state reconstruction is correct for different indices
        of considered snapshots"""
        circuit = hadamard_circuit(wires, interface=interface)
        bits, recipes = circuit()
        shadow = ClassicalShadow(bits, recipes)

        # choose 1000 random indices
        snapshots = np.random.choice(np.arange(10000, dtype=np.int64), size=1000, replace=False)
        state = shadow.global_snapshots(snapshots=snapshots)
        assert state.shape == (len(snapshots), 2**wires, 2**wires)

        # check the results against obtaining the full global snapshots
        expected = shadow.global_snapshots()
        for i, t in enumerate(snapshots):
            assert np.allclose(expected[t], state[i])

    def test_large_state_warning(self, monkeypatch):
        """Test that a warning is raised when a very large state is reconstructed"""
        circuit = hadamard_circuit(17, shots=2)
        bits, recipes = circuit()
        shadow = ClassicalShadow(bits, recipes)

        msg = "Querying density matrices for n_wires > 16 is not recommended, operation will take a long time"

        with monkeypatch.context() as m:
            # don't run the actual state computation since we only want the warning
            m.setattr(ClassicalShadow, "_obtain_global_snapshots", lambda *args, **kwargs: None)

            with pytest.warns(UserWarning, match=msg):
                shadow.global_snapshots()


@pytest.mark.all_interfaces
class TestExpvalEstimation:
    """Test that the expval estimation is correct for a variety of observables"""

    @pytest.mark.parametrize("interface", ["autograd", "jax", "tf", "torch"])
    @pytest.mark.parametrize(
        "obs, expected",
        [
            (qml.PauliX(1), 1),
            (qml.PauliX(0) @ qml.PauliX(2), 1),
            (qml.PauliX(0) @ qml.Identity(1) @ qml.PauliX(2), 1),
            (qml.PauliY(2), 0),
            (qml.PauliY(1) @ qml.PauliZ(2), 0),
            (qml.PauliX(0) @ qml.PauliY(1), 0),
            (qml.PauliX(0) @ qml.PauliY(1) @ qml.Identity(2), 0),
        ],
    )
    def test_hadamard_expval(self, interface, obs, expected):
        """Test that the expval estimation is correct for a uniform
        superposition of qubits"""
        circuit = hadamard_circuit(3, shots=100000, interface=interface)
        bits, recipes = circuit()
        shadow = ClassicalShadow(bits, recipes)

        actual = shadow.expval(obs, k=10)
        assert actual.shape == ()
        assert actual.dtype == np.float64
        assert qml.math.allclose(actual, expected, atol=1e-1)

    @pytest.mark.parametrize("interface", ["autograd", "jax", "tf", "torch"])
    @pytest.mark.parametrize(
        "obs, expected",
        [
            (qml.PauliX(1), 0),
            (qml.PauliX(0) @ qml.PauliX(2), 0),
            (qml.PauliZ(2), 0),
            (qml.Identity(1) @ qml.PauliZ(2), 0),
            (qml.PauliZ(1) @ qml.PauliZ(2), 1),
            (qml.PauliX(0) @ qml.PauliY(1), 0),
            (qml.PauliX(0) @ qml.PauliY(1) @ qml.Identity(2), 0),
            (qml.PauliY(0) @ qml.PauliX(1) @ qml.PauliY(2), -1),
        ],
    )
    def test_max_entangled_expval(self, interface, obs, expected):
        """Test that the expval estimation is correct for a maximally
        entangled state"""
        circuit = max_entangled_circuit(3, shots=100000, interface=interface)
        bits, recipes = circuit()
        shadow = ClassicalShadow(bits, recipes)

        actual = shadow.expval(obs, k=10)
        assert actual.shape == ()
        assert actual.dtype == np.float64
        assert qml.math.allclose(actual, expected, atol=1e-1)

    @pytest.mark.parametrize("interface", ["autograd", "jax", "tf", "torch"])
    @pytest.mark.parametrize(
        "obs, expected",
        [
            (qml.PauliX(0), -1),
            (qml.PauliX(0) @ qml.PauliX(1), 0),
            (qml.PauliX(0) @ qml.PauliX(2), -1 / np.sqrt(2)),
            (qml.PauliX(0) @ qml.Identity(1) @ qml.PauliX(2), -1 / np.sqrt(2)),
            (qml.PauliZ(2), 0),
            (qml.PauliX(1) @ qml.PauliY(2), 0),
            (qml.PauliY(1) @ qml.PauliX(2), 1 / np.sqrt(2)),
            (qml.Identity(0) @ qml.PauliY(1) @ qml.PauliX(2), 1 / np.sqrt(2)),
            (qml.PauliX(0) @ qml.PauliY(1) @ qml.PauliY(2), -1 / np.sqrt(2)),
            (qml.PauliY(0) @ qml.PauliX(1) @ qml.PauliX(2), 0),
        ],
    )
    def test_qft_expval(self, interface, obs, expected):
        """Test that the expval estimation is correct for a QFT state"""
        circuit = qft_circuit(3, shots=100000, interface=interface)
        bits, recipes = circuit()
        shadow = ClassicalShadow(bits, recipes)

        actual = shadow.expval(obs, k=10)
        assert actual.shape == ()
        assert actual.dtype == np.float64
        assert qml.math.allclose(actual, expected, atol=1e-1)

    def test_non_pauli_error(self):
        """Test that an error is raised when a non-Pauli observable is passed"""
        circuit = hadamard_circuit(3)
        bits, recipes = circuit()
        shadow = ClassicalShadow(bits, recipes)

        H = qml.Hadamard(0) @ qml.Hadamard(2)

        msg = "Observable must be a linear combination of Pauli observables"
        with pytest.raises(ValueError, match=msg):
            shadow.expval(H, k=10)


def convert_to_interface(arr, interface):
    """Dispatch arrays for different interfaces"""
    import jax.numpy as jnp
    import tensorflow as tf
    import torch

    if interface == "autograd":
        return arr

    if interface == "jax":
        return jnp.array(arr)

    if interface == "tf":
        return tf.constant(arr)

    if interface == "torch":
        return torch.tensor(arr)


@pytest.mark.all_interfaces
class TestMedianOfMeans:
    """Test the median of means function"""

    # TODO: add tests for the gradient once we implement the post-processing reconstruction

    @pytest.mark.parametrize(
        "arr, num_batches, expected",
        [
            (np.array([0.1]), 1, 0.1),
            (np.array([0.1, 0.2]), 1, 0.15),
            (np.array([0.1, 0.2]), 2, 0.15),
            (np.array([0.2, 0.1, 0.4]), 1, 0.7 / 3),
            (np.array([0.2, 0.1, 0.4]), 2, 0.275),
            (np.array([0.2, 0.1, 0.4]), 3, 0.2),
        ],
    )
    @pytest.mark.parametrize("interface", ["autograd", "jax", "tf", "torch"])
    def test_output(self, arr, num_batches, expected, interface):
        """Test that the output is correct"""
        arr = convert_to_interface(arr, interface)

        actual = median_of_means(arr, num_batches)
        assert actual.shape == ()
        assert np.allclose(actual, expected)


@pytest.mark.all_interfaces
class TestPauliExpval:
    """Test the Pauli expectation value function"""

    @pytest.mark.parametrize("word", [[0, 0, 1], [0, 2, -1], [-1, -1, 1]])
    @pytest.mark.parametrize("interface", ["autograd", "jax", "tf", "torch"])
    def test_word_not_present(self, word, interface):
        """Test that the output is 0 if the Pauli word is not present in the recipes"""
        bits = convert_to_interface(np.array([[0, 0, 0]]), interface)
        recipes = convert_to_interface(np.array([[0, 0, 0]]), interface)

        actual = pauli_expval(bits, recipes, np.array(word))
        assert actual.shape == (1,)
        assert actual[0] == 0

    single_bits = np.array([[1, 0, 1]])
    single_recipes = np.array([[0, 1, 2]])

    @pytest.mark.parametrize(
        "word, expected", [([0, 1, 2], 27), ([0, 1, -1], -9), ([-1, -1, 2], -3), ([-1, -1, -1], 1)]
    )
    @pytest.mark.parametrize("interface", ["autograd", "jax", "tf", "torch"])
    def test_single_word_present(self, word, expected, interface):
        """Test that the output is correct if the Pauli word appears once in the recipes"""
        bits = convert_to_interface(self.single_bits, interface)
        recipes = convert_to_interface(self.single_recipes, interface)

        actual = pauli_expval(bits, recipes, np.array(word))
        assert actual.shape == (1,)
        assert actual[0] == expected

    multi_bits = np.array([[1, 0, 1], [0, 0, 1], [1, 1, 1]])
    multi_recipes = np.array([[0, 1, 2], [0, 1, 2], [0, 1, 0]])

    @pytest.mark.parametrize(
        "word, expected",
        [
            ([0, 1, 2], [27, -27, 0]),
            ([0, 1, -1], [-9, 9, 9]),
            ([-1, -1, 2], [-3, -3, 0]),
            ([-1, -1, -1], [1, 1, 1]),
        ],
    )
    @pytest.mark.parametrize("interface", ["autograd", "jax", "tf", "torch"])
    def test_multi_word_present(self, word, expected, interface):
        """Test that the output is correct if the Pauli word appears multiple
        times in the recipes"""
        bits = convert_to_interface(self.multi_bits, interface)
        recipes = convert_to_interface(self.multi_recipes, interface)

        actual = pauli_expval(bits, recipes, np.array(word))
        assert actual.shape == (self.multi_bits.shape[0],)
        assert np.all(actual == expected)

