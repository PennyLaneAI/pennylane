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
Tests for the ArbitraryUnitary template.
"""
import pytest
import numpy as np
from pennylane import numpy as pnp
import pennylane as qml
from pennylane.templates.subroutines.arbitrary_unitary import (
    _all_pauli_words_but_identity,
    _tuple_to_word,
    _n_k_gray_code,
)

# fmt: off
PAULI_WORD_TEST_DATA = [
    (1, ["X", "Y", "Z"]),
    (
        2,
        ["XI", "YI", "ZI", "ZX", "IX", "XX", "YX", "YY", "ZY", "IY", "XY", "XZ", "YZ", "ZZ", "IZ"],
    ),
    (
        3,
        [
            "XII", "YII", "ZII", "ZXI", "IXI", "XXI", "YXI", "YYI", "ZYI", "IYI", "XYI", "XZI", "YZI",
            "ZZI", "IZI", "IZX", "XZX", "YZX", "ZZX", "ZIX", "IIX", "XIX", "YIX", "YXX", "ZXX", "IXX",
            "XXX", "XYX", "YYX", "ZYX", "IYX", "IYY", "XYY", "YYY", "ZYY", "ZZY", "IZY", "XZY", "YZY",
            "YIY", "ZIY", "IIY", "XIY", "XXY", "YXY", "ZXY", "IXY", "IXZ", "XXZ", "YXZ", "ZXZ", "ZYZ",
            "IYZ", "XYZ", "YYZ", "YZZ", "ZZZ", "IZZ", "XZZ", "XIZ", "YIZ", "ZIZ", "IIZ",
        ]
    ),
]

GRAY_CODE_TEST_DATA = [
    (2, 2, [[0, 0], [1, 0], [1, 1], [0, 1]]),
    (2, 3, [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 0, 1], [0, 0, 1]]),
    (4, 2, [
        [0, 0], [1, 0], [2, 0], [3, 0], [3, 1], [0, 1], [1, 1], [2, 1],
        [2, 2], [3, 2], [0, 2], [1, 2], [1, 3], [2, 3], [3, 3], [0, 3]
    ]),
    (3, 3, [
        [0, 0, 0], [1, 0, 0], [2, 0, 0], [2, 1, 0], [0, 1, 0], [1, 1, 0], [1, 2, 0], [2, 2, 0], [0, 2, 0],
        [0, 2, 1], [1, 2, 1], [2, 2, 1], [2, 0, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1], [2, 1, 1], [0, 1, 1],
        [0, 1, 2], [1, 1, 2], [2, 1, 2], [2, 2, 2], [0, 2, 2], [1, 2, 2], [1, 0, 2], [2, 0, 2], [0, 0, 2]
    ]),
]


# fmt: on


class TestHelpers:
    """Tests the helper functions."""

    @pytest.mark.parametrize("n,k,expected_code", GRAY_CODE_TEST_DATA)
    def test_n_k_gray_code(self, n, k, expected_code):
        """Test that _n_k_gray_code produces the Gray code correctly."""
        for expected_word, word in zip(expected_code, _n_k_gray_code(n, k)):
            assert expected_word == word

    @pytest.mark.parametrize("num_wires,expected_pauli_words", PAULI_WORD_TEST_DATA)
    def test_all_pauli_words_but_identity(self, num_wires, expected_pauli_words):
        """Test that the correct Pauli words are returned."""
        for expected_pauli_word, pauli_word in zip(
            expected_pauli_words, _all_pauli_words_but_identity(num_wires)
        ):
            assert expected_pauli_word == pauli_word

    @pytest.mark.parametrize(
        "tuple,expected_word",
        [
            ((0,), "I"),
            ((1,), "X"),
            ((2,), "Y"),
            ((3,), "Z"),
            ((0, 0, 0), "III"),
            ((1, 2, 3), "XYZ"),
            ((1, 2, 3, 0, 0, 3, 2, 1), "XYZIIZYX"),
        ],
    )
    def test_tuple_to_word(self, tuple, expected_word):
        assert _tuple_to_word(tuple) == expected_word


class TestDecomposition:
    """Tests that the template defines the correct decomposition."""

    def test_correct_gates_single_wire(self):
        """Test that the correct gates are applied on a single wire."""
        weights = np.arange(3, dtype=float)

        op = qml.ArbitraryUnitary(weights, wires=[0])
        queue = op.expand().operations

        for gate in queue:
            assert gate.name == "PauliRot"
            assert gate.wires.tolist() == [0]

        pauli_words = ["X", "Y", "Z"]

        for i, op in enumerate(queue):
            assert op.data[0] == weights[i]
            assert op.hyperparameters["pauli_word"] == pauli_words[i]

    def test_correct_gates_two_wires(self):
        """Test that the correct gates are applied on two wires."""
        weights = np.arange(15, dtype=float)

        op = qml.ArbitraryUnitary(weights, wires=[0, 1])
        queue = op.expand().operations

        for gate in queue:
            assert gate.name == "PauliRot"
            assert gate.wires.tolist() == [0, 1]

        pauli_words = [
            "XI",
            "YI",
            "ZI",
            "ZX",
            "IX",
            "XX",
            "YX",
            "YY",
            "ZY",
            "IY",
            "XY",
            "XZ",
            "YZ",
            "ZZ",
            "IZ",
        ]

        for i, op in enumerate(queue):
            assert op.data[0] == weights[i]
            assert op.hyperparameters["pauli_word"] == pauli_words[i]

    def test_custom_wire_labels(self, tol):
        """Test that template can deal with non-numeric, nonconsecutive wire labels."""

        weights = np.random.random(size=(63,))
        dev = qml.device("default.qubit", wires=3)
        dev2 = qml.device("default.qubit", wires=["z", "a", "k"])

        @qml.qnode(dev)
        def circuit():
            qml.ArbitraryUnitary(weights, wires=range(3))
            return qml.expval(qml.Identity(0))

        @qml.qnode(dev2)
        def circuit2():
            qml.ArbitraryUnitary(weights, wires=["z", "a", "k"])
            return qml.expval(qml.Identity("z"))

        circuit()
        circuit2()

        assert np.allclose(dev.state, dev2.state, atol=tol, rtol=0)


class TestInputs:
    """Test inputs and pre-processing."""

    def test_exception_wrong_dim(self):
        """Verifies that exception is raised if the
        number of dimensions of features is incorrect."""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(weights):
            qml.ArbitraryUnitary(weights, wires=range(2))
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(ValueError, match="Weights tensor must be of shape"):
            weights = np.array([0, 1])
            circuit(weights)

    def test_id(self):
        """Tests that the id attribute can be set."""
        template = qml.ArbitraryUnitary(np.random.random(size=(63,)), wires=range(3), id="a")
        assert template.id == "a"


class TestAttributes:
    """Tests class attributes and methods."""

    @pytest.mark.parametrize(
        "n_wires, expected_shape",
        [
            (1, (3,)),
            (2, (15,)),
            (3, (63,)),
        ],
    )
    def test_shape(self, n_wires, expected_shape):
        """Test that the shape method returns the correct shape of the weights tensor"""

        shape = qml.ArbitraryUnitary.shape(n_wires)
        assert shape == expected_shape


# test data for gradient tests

paulis_two_qubits = PAULI_WORD_TEST_DATA[1][1]


def circuit_template(weights):
    qml.ArbitraryUnitary(weights, wires=range(2))
    return qml.expval(qml.PauliZ(0))


def circuit_decomposed(weights):
    for i in range(qml.math.shape(weights)[0]):
        qml.PauliRot(weights[i], paulis_two_qubits[i], wires=[0, 1])
    return qml.expval(qml.PauliZ(0))


class TestInterfaces:
    """Tests that the template is compatible with all interfaces, including the computation
    of gradients."""

    def test_list_and_tuples(self, tol):
        """Tests common iterables as inputs."""

        weights = list(range(15))

        dev = qml.device("default.qubit", wires=2)

        circuit = qml.QNode(circuit_template, dev)
        circuit2 = qml.QNode(circuit_decomposed, dev)

        res = circuit(weights)
        res2 = circuit2(weights)
        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

    @pytest.mark.autograd
    def test_autograd(self, tol):
        """Tests the autograd interface."""

        weights = pnp.array(np.random.random(size=(15,)), requires_grad=True)

        dev = qml.device("default.qubit", wires=2)

        circuit = qml.QNode(circuit_template, dev)
        circuit2 = qml.QNode(circuit_decomposed, dev)

        res = circuit(weights)
        res2 = circuit2(weights)
        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

        grad_fn = qml.grad(circuit)
        grads = grad_fn(weights)

        grad_fn2 = qml.grad(circuit2)
        grads2 = grad_fn2(weights)

        assert np.allclose(grads, grads2, atol=tol, rtol=0)

    @pytest.mark.jax
    def test_jax(self, tol):
        """Tests the jax interface."""

        import jax
        import jax.numpy as jnp

        weights = jnp.array(np.random.random(size=(15,)))
        dev = qml.device("default.qubit", wires=2)

        circuit = qml.QNode(circuit_template, dev, interface="jax")
        circuit2 = qml.QNode(circuit_decomposed, dev, interface="jax")

        res = circuit(weights)
        res2 = circuit2(weights)
        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

        grad_fn = jax.grad(circuit)
        grads = grad_fn(weights)

        grad_fn2 = jax.grad(circuit2)
        grads2 = grad_fn2(weights)

        assert np.allclose(grads[0], grads2[0], atol=tol, rtol=0)

    @pytest.mark.tf
    def test_tf(self, tol):
        """Tests the tf interface."""

        import tensorflow as tf

        weights = tf.Variable(np.random.random(size=(15,)))
        dev = qml.device("default.qubit", wires=2)

        circuit = qml.QNode(circuit_template, dev, interface="tf")
        circuit2 = qml.QNode(circuit_decomposed, dev, interface="tf")

        res = circuit(weights)
        res2 = circuit2(weights)
        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

        with tf.GradientTape() as tape:
            res = circuit(weights)
        grads = tape.gradient(res, [weights])

        with tf.GradientTape() as tape2:
            res2 = circuit2(weights)
        grads2 = tape2.gradient(res2, [weights])

        assert np.allclose(grads[0], grads2[0], atol=tol, rtol=0)

    @pytest.mark.torch
    def test_torch(self, tol):
        """Tests the torch interface."""

        import torch

        weights = torch.tensor(np.random.random(size=(15,)), requires_grad=True)

        dev = qml.device("default.qubit", wires=3)

        circuit = qml.QNode(circuit_template, dev, interface="torch")
        circuit2 = qml.QNode(circuit_decomposed, dev, interface="torch")

        res = circuit(weights)
        res2 = circuit2(weights)
        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

        res = circuit(weights)
        res.backward()
        grads = [weights.grad]

        res2 = circuit2(weights)
        res2.backward()
        grads2 = [weights.grad]

        assert np.allclose(grads[0], grads2[0], atol=tol, rtol=0)
