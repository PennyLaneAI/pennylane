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
import numpy as np
import pytest

import pennylane as qml
from pennylane import numpy as pnp

FEATURES = [
    np.array([0, 1, 0, 0]),
    1 / np.sqrt(4) * np.array([1, 1, 1, 1]),
    np.array([complex(-np.sqrt(0.1), 0.0), np.sqrt(0.3), complex(0, -np.sqrt(0.1)), np.sqrt(0.5)]),
]

BROADCASTED_FEATURES = [np.eye(4)[:3], np.ones((5, 4)) / 2]

NOT_ENOUGH_FEATURES = [
    np.array([0, 1, 0]),
    1 / np.sqrt(3) * np.array([1, 1, 1]),
    np.array([complex(-np.sqrt(0.1), 0.0), np.sqrt(0.3), complex(0, -np.sqrt(0.6))]),
]

NOT_ENOUGH_BROADCASTED_FEATURES = [np.eye(4)[:3, :3], np.ones((5, 2)) / np.sqrt(2)]

TOO_MANY_FEATURES = [
    [0, 0, 0, 1, 0],
    1 / np.sqrt(8) * np.array([1] * 8),
    [complex(-np.sqrt(0.1), 0.0), np.sqrt(0.3), complex(0, -np.sqrt(0.6)), 0.0, 0.0],
]

TOO_MANY_BROADCASTED_FEATURES = [np.eye(6)[:3, :5], np.ones((3, 8)) / np.sqrt(8)]


@pytest.mark.jax
def test_standard_validity():
    """Check the operation using the assert_valid function."""

    op = qml.AmplitudeEmbedding(features=FEATURES[0], wires=range(2))

    qml.ops.functions.assert_valid(op, skip_differentiation=True)


class TestDecomposition:
    """Tests that the template defines the correct decomposition."""

    def test_expansion(self):
        """Checks the queue for the default settings."""

        op = qml.AmplitudeEmbedding(features=FEATURES[0], wires=range(2))
        tape = qml.tape.QuantumScript(op.decomposition())

        assert len(tape.operations) == 1
        assert tape.operations[0].name == "MottonenStatePreparation"
        assert tape.batch_size is None

    def test_expansion_broadcasted(self):
        """Checks the queue for the default settings."""

        op = qml.AmplitudeEmbedding(features=BROADCASTED_FEATURES[0], wires=range(2))
        assert op.batch_size == 3
        tape = qml.tape.QuantumScript(op.decomposition())

        assert len(tape.operations) == 1
        assert tape.operations[0].name == "MottonenStatePreparation"
        assert tape.batch_size == 3

    @pytest.mark.parametrize("normalize", (True, False))
    @pytest.mark.parametrize("inpt", FEATURES)
    def test_prepares_correct_state(self, inpt, normalize):
        """Checks the state for real and complex inputs."""

        n_qubits = 2
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(x=None):
            qml.AmplitudeEmbedding(features=x, wires=range(n_qubits), normalize=normalize)
            return qml.state()

        state = circuit(x=inpt).ravel()
        assert np.allclose(state, inpt)

    @pytest.mark.parametrize("normalize", (True, False))
    @pytest.mark.parametrize("inpt", BROADCASTED_FEATURES)
    def test_prepares_correct_broadcasted_state(self, inpt, normalize):
        """Checks the state for real and complex inputs."""

        n_qubits = 2
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(x=None):
            qml.AmplitudeEmbedding(features=x, wires=range(n_qubits), normalize=normalize)
            return qml.state()

        state = circuit(x=inpt)
        assert np.allclose(state, inpt)

    @pytest.mark.parametrize("inpt", NOT_ENOUGH_FEATURES)
    @pytest.mark.parametrize("pad", [complex(0.1, 0.1), 0.0, 1.0])
    def test_prepares_padded_state(self, inpt, pad):
        """Checks the state for real and complex padding constants."""

        n_qubits = 2
        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev)
        def circuit(x=None):
            qml.AmplitudeEmbedding(features=x, wires=range(n_qubits), pad_with=pad, normalize=False)
            return qml.state()

        state = circuit(x=inpt).ravel()
        # Make sure all padded values are the same constant
        # by checking how many different values there are
        assert len(set(state[len(inpt) :])) == 1

    @pytest.mark.parametrize("inpt", NOT_ENOUGH_BROADCASTED_FEATURES)
    @pytest.mark.parametrize("pad", [complex(0.1, 0.1), 0.0, 1.0])
    def test_prepares_padded_state_broadcasted(self, inpt, pad):
        """Checks the state for real and complex padding constants."""

        n_qubits = 2
        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev)
        def circuit(x=None):
            qml.AmplitudeEmbedding(features=x, wires=range(n_qubits), pad_with=pad, normalize=False)
            return qml.state()

        state = circuit(x=inpt)
        # Make sure all padded values are the same constant
        # by checking how many different values there are
        assert len(np.unique(state[:, inpt.shape[1] :])) == 1

    def test_custom_wire_labels(self, tol):
        """Test that template can deal with non-numeric, nonconsecutive wire labels."""
        features = np.array([0, 1 / 2, 0, 1 / 2, 0, 0, 1 / 2, 1 / 2])

        dev = qml.device("default.qubit", wires=3)
        dev2 = qml.device("default.qubit", wires=["z", "a", "k"])

        @qml.qnode(dev)
        def circuit():
            qml.AmplitudeEmbedding(features, wires=range(3))
            return qml.expval(qml.Identity(0)), qml.state()

        @qml.qnode(dev2)
        def circuit2():
            qml.AmplitudeEmbedding(features, wires=["z", "a", "k"])
            return qml.expval(qml.Identity("z")), qml.state()

        res1, state1 = circuit()
        res2, state2 = circuit2()

        assert np.allclose(res1, res2, atol=tol, rtol=0)
        assert np.allclose(state1, state2, atol=tol, rtol=0)


class TestInputs:
    """Test inputs and pre-processing."""

    @pytest.mark.parametrize("inpt", FEATURES + BROADCASTED_FEATURES)
    def test_throws_exception_if_not_normalized(self, inpt):
        """Checks exception when state is not normalized and `normalize=False`."""
        not_nrmlzd = 2 * inpt
        n_qubits = 2
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(x=None):
            qml.AmplitudeEmbedding(
                features=x, wires=range(n_qubits), pad_with=None, normalize=False
            )
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        with pytest.raises(ValueError, match="The state must be a vector of norm 1.0"):
            circuit(x=not_nrmlzd)

    def test_throws_exception_if_features_wrong_shape(self):
        """Checks exception if features has more than two dimensions."""

        n_qubits = 2
        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev)
        def circuit(x=None):
            qml.AmplitudeEmbedding(features=x, wires=range(n_qubits))
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(
            ValueError,
            match="State must be a one-dimensional tensor, or two-dimensional with batching;",
        ):
            circuit(x=[[[1.0, 0.0], [0.0, 0.0]], [[1.0, 0.0], [0.0, 0.0]]])

    @pytest.mark.parametrize(
        "inpt",
        NOT_ENOUGH_FEATURES
        + TOO_MANY_FEATURES
        + NOT_ENOUGH_BROADCASTED_FEATURES
        + TOO_MANY_BROADCASTED_FEATURES,
    )
    def test_throws_exception_if_fewer_features_than_amplitudes(self, inpt):
        """Checks exception if the number of features is wrong and
        no automatic padding is chosen."""

        n_qubits = 2
        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev)
        def circuit(x=None):
            qml.AmplitudeEmbedding(
                features=x, wires=range(n_qubits), pad_with=None, normalize=False
            )
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(ValueError, match="State must be of length"):
            circuit(x=inpt)

    @pytest.mark.parametrize("inpt", TOO_MANY_FEATURES + TOO_MANY_BROADCASTED_FEATURES)
    def test_throws_exception_if_more_features_than_amplitudes_padding(self, inpt):
        """Checks exception if the number of features is larger than the number of amplitudes, and
        automatic padding is chosen."""

        n_qubits = 2
        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev)
        def circuit(x=None):
            qml.AmplitudeEmbedding(features=x, wires=range(n_qubits), pad_with=0.0, normalize=False)
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(ValueError, match="Input state must be of length"):
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
            qml.AmplitudeEmbedding(x, list(range(num_qubits)), pad_with=0.0, normalize=True)
            return qml.expval(qml.PauliZ(0))

        # No normalization error is raised
        circuit(x=inputs)

    def test_id(self):
        """Tests that the id attribute can be set."""
        template = qml.AmplitudeEmbedding(np.array([1, 0]), wires=[0], id="a")
        assert template.id == "a"


def circuit_template(features, pad_with=None, normalize=False):
    """AmplitudeEmbedding circuit. For three wires, all test features match
    the expected length. For four wires, we get a test case for `pad_with`."""
    num_wires = 3 if pad_with is None else 4
    qml.AmplitudeEmbedding(features, wires=range(num_wires), pad_with=pad_with, normalize=normalize)
    return qml.state()


def circuit_decomposed(features, pad_with=None, normalize=False):
    """AmplitudeEmbedding circuit reexpressed as manual state preparation.
    This function expects the length of the features to match."""
    num_wires = 3 if pad_with is None else 4
    if not isinstance(features, (list, tuple)):
        features = qml.math.cast_like(features, 1.0) if "int" in str(features.dtype) else features
    if pad_with is not None:
        # and need to pad manually in order to double the size of the vector
        # from 8 (3 qubits) to 16 (4 qubits). Also, normalize
        pad_with = qml.math.cast_like(pad_with, features)
        features = qml.math.hstack([features, qml.math.ones_like(features) * pad_with])
    if pad_with is not None or normalize:
        shape = qml.math.shape(features)
        norm = qml.math.reshape(qml.math.linalg.norm(features, axis=-1), (*shape[:-1], 1))
        features = features / norm
    # need to cast to complex tensor, which is implicitly done in the template
    qml.StatePrep(qml.math.cast(features, np.complex128), wires=range(num_wires))
    return qml.state()


all_features = [
    [0.5, 0, 0.5, 0, 0.5, 0.5, 0, 0],
    [
        [0.5, 0, 0.5, 0, 0.5, 0.5, 0, 0],
        [0.5, 0, 0.5, 0, 0.5, 0.5, 0, 0],
        [0.5, 0, 0.5, 0, 0.5, 0.5, 0, 0],
    ],
]


@pytest.mark.parametrize("features", all_features)
@pytest.mark.parametrize("pad_with", [None, 0.0, 0.1])
@pytest.mark.parametrize("normalize", [False, True])
class TestInterfaces:
    """Tests that the template is compatible with all interfaces."""

    def test_list_and_tuples(self, tol, features, pad_with, normalize):
        """Tests common iterables as inputs."""

        dev = qml.device("default.qubit")

        circuit = qml.QNode(circuit_template, dev)
        circuit2 = qml.QNode(circuit_decomposed, dev)

        res = circuit(features, pad_with, normalize)
        res2 = circuit2(features, pad_with)
        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

        res = circuit(tuple(features), pad_with, normalize)
        res2 = circuit2(tuple(features), pad_with)
        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

    @pytest.mark.autograd
    def test_autograd(self, tol, features, pad_with, normalize):
        """Tests autograd tensors."""

        features = pnp.array(features, requires_grad=True)

        dev = qml.device("default.qubit")

        circuit = qml.QNode(circuit_template, dev, interface="autograd")
        circuit2 = qml.QNode(circuit_decomposed, dev)

        res = circuit(features, pad_with, normalize)
        res2 = circuit2(features, pad_with)

        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

    @pytest.mark.jax
    def test_jax(self, tol, features, pad_with, normalize):
        """Tests jax tensors."""
        import jax.numpy as jnp

        features = jnp.array(features)

        dev = qml.device("default.qubit")

        circuit = qml.QNode(circuit_template, dev, interface="jax")
        circuit2 = qml.QNode(circuit_decomposed, dev)

        res = circuit(features, pad_with, normalize)
        res2 = circuit2(features, pad_with)

        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

    @pytest.mark.jax
    def test_jax_jit(self, tol, features, pad_with, normalize):
        """Tests jax tensors with JIT compilation."""
        import jax
        import jax.numpy as jnp

        features = jnp.array(features)

        dev = qml.device("default.qubit")

        circuit = jax.jit(qml.QNode(circuit_template, dev, interface="jax"), static_argnums=[1, 2])
        circuit2 = qml.QNode(circuit_template, dev)

        res = circuit(features, pad_with, normalize)
        res2 = circuit2(features, pad_with, normalize)

        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

    @pytest.mark.tf
    def test_tf(self, tol, features, pad_with, normalize):
        """Tests tensorflow tensors."""
        import tensorflow as tf

        features = tf.Variable(features)

        dev = qml.device("default.qubit")

        circuit = qml.QNode(circuit_template, dev, interface="tensorflow")
        circuit2 = qml.QNode(circuit_decomposed, dev)

        res = circuit(features, pad_with, normalize)
        res2 = circuit2(features, pad_with)

        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

    @pytest.mark.tf
    def test_tf_jit(self, tol, features, pad_with, normalize):
        """Tests tensorflow tensors with JIT compilation."""
        import tensorflow as tf

        features = tf.Variable(features)

        dev = qml.device("default.qubit")

        circuit = tf.function(jit_compile=True)(
            qml.QNode(circuit_template, dev, interface="tensorflow")
        )
        circuit2 = tf.function(jit_compile=True)(qml.QNode(circuit_decomposed, dev))

        res = circuit(features, pad_with, normalize)
        res2 = circuit2(features, pad_with)

        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

    @pytest.mark.torch
    def test_torch(self, tol, features, pad_with, normalize):
        """Tests Torch tensors."""
        import torch

        features = torch.tensor(features, requires_grad=True)

        dev = qml.device("default.qubit")

        circuit = qml.QNode(circuit_template, dev, interface="torch")
        circuit2 = qml.QNode(circuit_decomposed, dev)

        res = circuit(features, pad_with, normalize)
        res2 = circuit2(features, pad_with)

        assert qml.math.allclose(res, res2, atol=tol, rtol=0)


int_features = [
    [10, 0, 2, 35, 0, 41, 56, 0],
    [
        [10, 0, 2, 35, 0, 41, 56, 0],
        [10, 0, 2, 35, 0, 41, 56, 0],
        [10, 0, 2, 35, 0, 41, 56, 0],
    ],
]


@pytest.mark.parametrize("features", int_features)
@pytest.mark.parametrize("pad_with", [None, 0.0, 0.1])
@pytest.mark.parametrize("dtype", ["int8", "uint8", "int32", "float32", "int64", "float64"])
class TestInterfaceDtypes:
    """Unit tests to verify that AmplitudeEmbedding works correctly for features with all dtypes"""

    @pytest.mark.autograd
    def test_autograd(self, tol, features, pad_with, dtype):
        """Tests autograd tensors."""

        dtype = getattr(pnp, dtype)
        features = pnp.array(features, dtype=dtype)

        dev = qml.device("default.qubit")

        circuit = qml.QNode(circuit_template, dev, interface="autograd")
        circuit2 = qml.QNode(circuit_decomposed, dev)

        res = circuit(features, pad_with, normalize=True)
        res2 = circuit2(features, pad_with, normalize=True)

        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

    @pytest.mark.jax
    def test_jax(self, tol, features, pad_with, dtype):
        """Tests jax tensors."""
        import jax.numpy as jnp

        dtype = getattr(jnp, dtype)
        features = jnp.array(features, dtype=dtype)

        dev = qml.device("default.qubit")

        circuit = qml.QNode(circuit_template, dev, interface="jax")
        circuit2 = qml.QNode(circuit_decomposed, dev)

        res = circuit(features, pad_with, normalize=True)
        res2 = circuit2(features, pad_with, normalize=True)

        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

    @pytest.mark.jax
    def test_jax_jit(self, tol, features, pad_with, dtype):
        """Tests jax tensors with JIT compilation."""
        import jax
        import jax.numpy as jnp

        dtype = getattr(jnp, dtype)
        features = jnp.array(features, dtype=dtype)

        dev = qml.device("default.qubit")

        circuit = jax.jit(qml.QNode(circuit_template, dev, interface="jax"), static_argnums=[1, 2])
        circuit2 = jax.jit(qml.QNode(circuit_decomposed, dev), static_argnums=[1, 2])

        res = circuit(features, pad_with, normalize=True)
        res2 = circuit2(features, pad_with, normalize=True)

        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

    @pytest.mark.tf
    def test_tf(self, tol, features, pad_with, dtype):
        """Tests tensorflow tensors."""
        import tensorflow as tf

        dtype = getattr(tf, dtype)
        features = tf.Variable(features, dtype=dtype)

        dev = qml.device("default.qubit")

        circuit = qml.QNode(circuit_template, dev, interface="tensorflow")
        circuit2 = qml.QNode(circuit_decomposed, dev)

        res = circuit(features, pad_with, normalize=True)
        res2 = circuit2(features, pad_with, normalize=True)

        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

    @pytest.mark.tf
    def test_tf_jit(self, tol, features, pad_with, dtype):
        """Tests tensorflow tensors with JIT compilation."""
        import tensorflow as tf

        dtype = getattr(tf, dtype)
        features = tf.Variable(features, dtype=dtype)

        dev = qml.device("default.qubit")

        circuit = tf.function(jit_compile=True)(
            qml.QNode(circuit_template, dev, interface="tensorflow")
        )
        circuit2 = tf.function(jit_compile=True)(qml.QNode(circuit_decomposed, dev))

        res = circuit(features, pad_with, normalize=True)
        res2 = circuit2(features, pad_with, normalize=True)

        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

    @pytest.mark.torch
    def test_torch(self, tol, features, pad_with, dtype):
        """Tests Torch tensors."""
        import torch

        dtype = getattr(torch, dtype)
        features = torch.tensor(features, dtype=dtype)

        dev = qml.device("default.qubit")

        circuit = qml.QNode(circuit_template, dev, interface="torch")
        circuit2 = qml.QNode(circuit_decomposed, dev)

        res = circuit(features, pad_with, normalize=True)
        res2 = circuit2(features, pad_with, normalize=True)

        assert qml.math.allclose(res, res2, atol=tol, rtol=0)


@pytest.mark.jax
@pytest.mark.parametrize("shots, atol", [(10000, 0.05), (None, 1e-8)])
def test_jacobian_with_and_without_jit_has_same_output(shots, atol, seed):
    """Test that the jacobian of AmplitudeEmbedding is the same with and without jit."""

    import jax

    dev = qml.device("default.qubit", seed=seed)

    @qml.set_shots(shots)
    @qml.qnode(dev, diff_method="parameter-shift")
    def circuit(coeffs):
        qml.AmplitudeEmbedding(coeffs, normalize=True, wires=[0, 1])
        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

    params = jax.numpy.array([0.4, 0.5, 0.1, 0.3])
    jac_fn = jax.jacobian(circuit)
    jac_jit_fn = jax.jit(jac_fn)

    jac = jac_fn(params)

    jac_jit = jac_jit_fn(params)

    assert qml.math.allclose(jac, jac_jit, atol=atol)
