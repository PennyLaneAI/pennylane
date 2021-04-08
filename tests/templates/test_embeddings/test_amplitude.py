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
"""
Tests for the AmplitudeEmbedding template.
"""
import pytest
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

FEATURES = [
    np.array([0, 1, 0, 0]),
    1 / np.sqrt(4) * np.array([1, 1, 1, 1]),
    np.array([complex(-np.sqrt(0.1), 0.0), np.sqrt(0.3), complex(0, -np.sqrt(0.1)), np.sqrt(0.5)]),
]

NOT_ENOUGH_FEATURES = [
    np.array([0, 1, 0]),
    1 / np.sqrt(3) * np.array([1, 1, 1]),
    np.array([complex(-np.sqrt(0.1), 0.0), np.sqrt(0.3), complex(0, -np.sqrt(0.6))]),
]

TOO_MANY_FEATURES = [
    [0, 0, 0, 1, 0],
    1 / np.sqrt(8) * np.array([1] * 8),
    [complex(-np.sqrt(0.1), 0.0), np.sqrt(0.3), complex(0, -np.sqrt(0.6)), 0.0, 0.0],
]


class TestDecomposition:
    """Tests that the template defines the correct decomposition."""

    def test_expansion(self):
        """Checks the queue for the default settings."""

        op = qml.templates.AmplitudeEmbedding(features=FEATURES[0], wires=range(2))
        tape = op.expand()

        assert len(tape.operations) == 1
        assert tape.operations[0].name == "QubitStateVector"

    @pytest.mark.parametrize("inpt", FEATURES)
    def test_prepares_correct_state(self, inpt):
        """Checks the state for real and complex inputs."""

        n_qubits = 2
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(x=None):
            qml.templates.AmplitudeEmbedding(features=x, wires=range(n_qubits), normalize=False)
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        circuit(x=inpt)
        state = circuit.device.state.ravel()
        assert np.allclose(state, inpt)

    @pytest.mark.parametrize("inpt", NOT_ENOUGH_FEATURES)
    @pytest.mark.parametrize("pad", [complex(0.1, 0.1), 0.0, 1.0])
    def test_prepares_padded_state(self, inpt, pad):
        """Checks the state for real and complex padding constants."""

        n_qubits = 2
        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev)
        def circuit(x=None):
            qml.templates.AmplitudeEmbedding(
                features=x, wires=range(n_qubits), pad_with=pad, normalize=False
            )
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        circuit(x=inpt)
        state = circuit.device.state.ravel()
        assert len(set(state[len(inpt) :])) == 1

    def test_custom_wire_labels(self, tol):
        """Test that template can deal with non-numeric, nonconsecutive wire labels."""
        features = np.array([0, 1 / 2, 0, 1 / 2, 0, 0, 1 / 2, 1 / 2])

        dev = qml.device("default.qubit", wires=3)
        dev2 = qml.device("default.qubit", wires=["z", "a", "k"])

        @qml.qnode(dev)
        def circuit():
            qml.templates.AmplitudeEmbedding(features, wires=range(3))
            return qml.expval(qml.Identity(0))

        @qml.qnode(dev2)
        def circuit2():
            qml.templates.AmplitudeEmbedding(features, wires=["z", "a", "k"])
            return qml.expval(qml.Identity("z"))

        circuit()
        circuit2()

        assert np.allclose(dev.state, dev2.state, atol=tol, rtol=0)


class TestInputs:
    """Test inputs and pre-processing."""

    @pytest.mark.parametrize("inpt", FEATURES)
    def test_throws_exception_if_not_normalized(self, inpt):
        """Checks exception when state is not normalized and `normalize=False`."""
        not_nrmlzd = 2 * inpt
        n_qubits = 2
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(x=None):
            qml.templates.AmplitudeEmbedding(
                features=x, wires=range(n_qubits), pad_with=None, normalize=False
            )
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        with pytest.raises(ValueError, match="Features must be a vector of length"):
            circuit(x=not_nrmlzd)

    def test_throws_exception_if_features_wrong_shape(self):
        """Checks exception if features has more than one dimension."""

        n_qubits = 2
        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev)
        def circuit(x=None):
            qml.templates.AmplitudeEmbedding(features=x, wires=range(n_qubits))
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(ValueError, match="Features must be a one-dimensional (tensor|vector)"):
            circuit(x=[[1.0, 0.0], [0.0, 0.0]])

    @pytest.mark.parametrize("inpt", NOT_ENOUGH_FEATURES + TOO_MANY_FEATURES)
    def test_throws_exception_if_fewer_features_than_amplitudes(self, inpt):
        """Checks exception if the number of features is wrong and
        no automatic padding is chosen."""

        n_qubits = 2
        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev)
        def circuit(x=None):
            qml.templates.AmplitudeEmbedding(
                features=x, wires=range(n_qubits), pad_with=None, normalize=False
            )
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(ValueError, match="Features must be of length"):
            circuit(x=inpt)

    @pytest.mark.parametrize("inpt", TOO_MANY_FEATURES)
    def test_throws_exception_if_more_features_than_amplitudes_padding(self, inpt):
        """Checks exception if the number of features is larger than the number of amplitudes, and
        automatic padding is chosen."""

        n_qubits = 2
        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev)
        def circuit(x=None):
            qml.templates.AmplitudeEmbedding(
                features=x, wires=range(n_qubits), pad_with=0.0, normalize=False
            )
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(ValueError, match="Features must be of length"):
            circuit(x=inpt)

    def test_amplitude_embedding_tolerance_value(self):
        """Tests that a small enough tolerance value is used for Amplitude
        Embedding."""
        inputs = np.array(
            [
                0.25895178024895,
                0.115997030111517,
                0.175840500169049,
                0.16545033015906,
                0.016337370015706,
                0.006616800006361,
                0.22326375021464,
                0.161815530155566,
                0.234776190225708,
                0.082623190079432,
                0.291982110280705,
                0.295344560283937,
                0.05998731005767,
                0.056911140054713,
                0.274260680263668,
                0.163596590157278,
                0.048460970046589,
                0.292306260281016,
                0.292451040281155,
                0.007849840007547,
                0.218302930209871,
                0.326763300314142,
                0.163634550157314,
                0.275472160264832,
                0.105510810101436,
            ]
        )

        tolerance = 10e-10
        num_qubits = 5
        dev = qml.device("default.qubit", wires=num_qubits)
        assert np.isclose(np.sum(np.abs(inputs) ** 2), 1, tolerance)

        @qml.qnode(dev)
        def circuit(x=None):
            qml.templates.AmplitudeEmbedding(
                x, list(range(num_qubits)), pad_with=0.0, normalize=True
            )
            return qml.expval(qml.PauliZ(0))

        # No normalization error is raised
        circuit(x=inputs)

    def test_deprecated_pad_arg(self):
        """Test that the pad argument raises a deprecation warning"""

        num_qubits = 2
        dev = qml.device("default.qubit", wires=num_qubits)
        inputs = np.array([1.0, 0.0, 0.0, 0.0])

        @qml.qnode(dev)
        def circuit(x=None):
            qml.templates.AmplitudeEmbedding(x, list(range(num_qubits)), pad=0.0, normalize=True)
            return qml.expval(qml.PauliZ(0))

        with pytest.warns(
            PendingDeprecationWarning,
            match="will be replaced by the pad_with option in future versions",
        ):
            circuit(x=inputs)


def circuit_template(features):
    qml.templates.AmplitudeEmbedding(features, wires=range(3))
    return qml.state()


def circuit_decomposed(features):
    # need to cast to complex tensor, which is implicitly done in the template
    qml.QubitStateVector(qml.math.cast(features, np.complex128), wires=range(3))
    return qml.state()


class TestInterfaces:
    """Tests that the template is compatible with all interfaces."""

    def test_list_and_tuples(self, tol):
        """Tests common iterables as inputs."""

        features = [1 / 2, 0, 1 / 2, 0, 1 / 2, 1 / 2, 0, 0]

        dev = qml.device("default.qubit", wires=3)

        circuit = qml.QNode(circuit_template, dev)
        circuit2 = qml.QNode(circuit_decomposed, dev)

        res = circuit(features)
        res2 = circuit2(features)
        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

        res = circuit(tuple(features))
        res2 = circuit2(tuple(features))
        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

    def test_autograd(self, tol):
        """Tests autograd tensors."""

        features = pnp.array([1 / 2, 0, 1 / 2, 0, 1 / 2, 1 / 2, 0, 0], requires_grad=True)

        dev = qml.device("default.qubit", wires=3)

        circuit = qml.QNode(circuit_template, dev)
        circuit2 = qml.QNode(circuit_decomposed, dev)

        res = circuit(features)
        res2 = circuit2(features)

        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

    def test_jax(self, tol):
        """Tests jax tensors."""

        jax = pytest.importorskip("jax")
        import jax.numpy as jnp

        features = jnp.array([1 / 2, 0, 1 / 2, 0, 1 / 2, 1 / 2, 0, 0])

        dev = qml.device("default.qubit", wires=3)

        circuit = qml.QNode(circuit_template, dev, interface="jax")
        circuit2 = qml.QNode(circuit_decomposed, dev, interface="jax")

        res = circuit(features)
        res2 = circuit2(features)

        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

    def test_tf(self, tol):
        """Tests tf tensors."""

        tf = pytest.importorskip("tensorflow")

        features = tf.Variable([1 / 2, 0, 1 / 2, 0, 1 / 2, 1 / 2, 0, 0])

        dev = qml.device("default.qubit", wires=3)

        circuit = qml.QNode(circuit_template, dev, interface="tf")
        circuit2 = qml.QNode(circuit_decomposed, dev, interface="tf")

        res = circuit(features)
        res2 = circuit2(features)

        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

    def test_torch(self, tol):
        """Tests torch tensors."""

        torch = pytest.importorskip("torch")

        features = torch.tensor([1 / 2, 0, 1 / 2, 0, 1 / 2, 1 / 2, 0, 0], requires_grad=True)

        dev = qml.device("default.qubit", wires=3)

        circuit = qml.QNode(circuit_template, dev, interface="torch")
        circuit2 = qml.QNode(circuit_decomposed, dev, interface="torch")

        res = circuit(features)
        res2 = circuit2(features)

        assert qml.math.allclose(res, res2, atol=tol, rtol=0)
