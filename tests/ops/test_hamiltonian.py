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
Tests for the Hamiltonian class.
"""
import numpy as np
import pytest

import pennylane as qml
from pennylane import numpy as pnp

# Make test data in different interfaces, if installed
COEFFS_PARAM_INTERFACE = [
    ([-0.05, 0.17], 1.7, "autograd"),
    (np.array([-0.05, 0.17]), np.array(1.7), "autograd"),
    (pnp.array([-0.05, 0.17], requires_grad=True), pnp.array(1.7, requires_grad=True), "autograd"),
]

try:
    from jax import numpy as jnp

    COEFFS_PARAM_INTERFACE.append((jnp.array([-0.05, 0.17]), jnp.array(1.7), "jax"))
except ImportError:
    pass

try:
    import tf

    COEFFS_PARAM_INTERFACE.append(
        (tf.Variable([-0.05, 0.17], dtype=tf.double), tf.Variable(1.7, dtype=tf.double), "tf")
    )
except ImportError:
    pass

try:
    import torch

    COEFFS_PARAM_INTERFACE.append((torch.tensor([-0.05, 0.17]), torch.tensor([1.7]), "torch"))
except ImportError:
    pass


def circuit1(param):
    """First Pauli subcircuit"""
    qml.RX(param, wires=0)
    qml.RY(param, wires=0)
    return qml.expval(qml.PauliX(0))


def circuit2(param):
    """Second Pauli subcircuit"""
    qml.RX(param, wires=0)
    qml.RY(param, wires=0)
    return qml.expval(qml.PauliZ(0))


dev = qml.device("default.qubit", wires=2)


class TestHamiltonianCoefficients:
    """Test the creation of a Hamiltonian"""

    @pytest.mark.parametrize("coeffs", [el[0] for el in COEFFS_PARAM_INTERFACE])
    def test_creation_different_coeff_types(self, coeffs):
        """Check that Hamiltonian's coefficients and data attributes are set correctly."""
        H = qml.Hamiltonian(coeffs, [qml.PauliX(0), qml.PauliZ(0)])
        assert np.allclose(coeffs, H.coeffs)
        assert np.allclose([coeffs[i] for i in range(qml.math.shape(coeffs)[0])], H.data)

    @pytest.mark.parametrize("coeffs", [el[0] for el in COEFFS_PARAM_INTERFACE])
    def test_simplify(self, coeffs):
        """Test that simplify works with different coefficient types."""
        H1 = qml.Hamiltonian(coeffs, [qml.PauliX(0), qml.PauliZ(1)])
        H2 = qml.Hamiltonian(coeffs, [qml.PauliX(0), qml.Identity(0) @ qml.PauliZ(1)])
        H2.simplify()
        assert H1.compare(H2)
        assert H1.data == H2.data


class TestHamiltonianArithmeticTF:
    """Tests creation of Hamiltonians using arithmetic
    operations with TensorFlow tensor coefficients."""

    def test_hamiltonian_equal(self):
        """Tests equality"""
        tf = pytest.importorskip("tensorflow")

        coeffs = tf.Variable([0.5, -1.6])
        obs = [qml.PauliX(0), qml.PauliY(1)]
        H1 = qml.Hamiltonian(coeffs, obs)

        coeffs2 = tf.Variable([-1.6, 0.5])
        obs2 = [qml.PauliY(1), qml.PauliX(0)]
        H2 = qml.Hamiltonian(coeffs2, obs2)

        assert H1.compare(H2)

    def test_hamiltonian_add(self):
        """Tests that Hamiltonians are added correctly"""
        tf = pytest.importorskip("tensorflow")

        coeffs = tf.Variable([0.5, -1.6])
        obs = [qml.PauliX(0), qml.PauliY(1)]
        H1 = qml.Hamiltonian(coeffs, obs)

        coeffs2 = tf.Variable([0.5, -0.4])
        H2 = qml.Hamiltonian(coeffs2, obs)

        coeffs_expected = tf.Variable([1.0, -2.0])
        H = qml.Hamiltonian(coeffs_expected, obs)

        assert H.compare(H1 + H2)

        H1 += H2
        assert H.compare(H1)

    def test_hamiltonian_sub(self):
        """Tests that Hamiltonians are subtracted correctly"""
        tf = pytest.importorskip("tensorflow")

        coeffs = tf.Variable([1.0, -2.0])
        obs = [qml.PauliX(0), qml.PauliY(1)]
        H1 = qml.Hamiltonian(coeffs, obs)

        coeffs2 = tf.Variable([0.5, -0.4])
        H2 = qml.Hamiltonian(coeffs2, obs)

        coeffs_expected = tf.Variable([0.5, -1.6])
        H = qml.Hamiltonian(coeffs_expected, obs)

        assert H.compare(H1 - H2)

        H1 -= H2
        assert H.compare(H1)

    def test_hamiltonian_matmul(self):
        """Tests that Hamiltonians are tensored correctly"""
        tf = pytest.importorskip("tensorflow")

        coeffs = tf.Variable([1.0, 2.0])
        obs = [qml.PauliX(0), qml.PauliY(1)]
        H1 = qml.Hamiltonian(coeffs, obs)

        coeffs2 = tf.Variable([-1.0, -2.0])
        obs2 = [qml.PauliX(2), qml.PauliY(3)]
        H2 = qml.Hamiltonian(coeffs2, obs2)

        coeffs_expected = tf.Variable([-4.0, -2.0, -2.0, -1.0])
        obs_expected = [
            qml.PauliY(1) @ qml.PauliY(3),
            qml.PauliX(0) @ qml.PauliY(3),
            qml.PauliX(2) @ qml.PauliY(1),
            qml.PauliX(0) @ qml.PauliX(2),
        ]
        H = qml.Hamiltonian(coeffs_expected, obs_expected)

        assert H.compare(H1 @ H2)


class TestHamiltonianArithmeticTorch:
    """Tests creation of Hamiltonians using arithmetic
    operations with torch tensor coefficients."""

    def test_hamiltonian_equal(self):
        """Tests equality"""
        torch = pytest.importorskip("torch")

        coeffs = torch.tensor([0.5, -1.6])
        obs = [qml.PauliX(0), qml.PauliY(1)]
        H1 = qml.Hamiltonian(coeffs, obs)

        coeffs2 = torch.tensor([-1.6, 0.5])
        obs2 = [qml.PauliY(1), qml.PauliX(0)]
        H2 = qml.Hamiltonian(coeffs2, obs2)

        assert H1.compare(H2)

    def test_hamiltonian_add(self):
        """Tests that Hamiltonians are added correctly"""
        torch = pytest.importorskip("torch")

        coeffs = torch.tensor([0.5, -1.6])
        obs = [qml.PauliX(0), qml.PauliY(1)]
        H1 = qml.Hamiltonian(coeffs, obs)

        coeffs2 = torch.tensor([0.5, -0.4])
        H2 = qml.Hamiltonian(coeffs2, obs)

        coeffs_expected = torch.tensor([1.0, -2.0])
        H = qml.Hamiltonian(coeffs_expected, obs)

        assert H.compare(H1 + H2)

        H1 += H2
        assert H.compare(H1)

    def test_hamiltonian_sub(self):
        """Tests that Hamiltonians are subtracted correctly"""
        torch = pytest.importorskip("torch")

        coeffs = torch.tensor([1.0, -2.0])
        obs = [qml.PauliX(0), qml.PauliY(1)]
        H1 = qml.Hamiltonian(coeffs, obs)

        coeffs2 = torch.tensor([0.5, -0.4])
        H2 = qml.Hamiltonian(coeffs2, obs)

        coeffs_expected = torch.tensor([0.5, -1.6])
        H = qml.Hamiltonian(coeffs_expected, obs)

        assert H.compare(H1 - H2)

        H1 -= H2
        assert H.compare(H1)

    def test_hamiltonian_matmul(self):
        """Tests that Hamiltonians are tensored correctly"""
        torch = pytest.importorskip("torch")

        coeffs = torch.tensor([1.0, 2.0])
        obs = [qml.PauliX(0), qml.PauliY(1)]
        H1 = qml.Hamiltonian(coeffs, obs)

        coeffs2 = torch.tensor([-1.0, -2.0])
        obs2 = [qml.PauliX(2), qml.PauliY(3)]
        H2 = qml.Hamiltonian(coeffs2, obs2)

        coeffs_expected = torch.tensor([-4.0, -2.0, -2.0, -1.0])
        obs_expected = [
            qml.PauliY(1) @ qml.PauliY(3),
            qml.PauliX(0) @ qml.PauliY(3),
            qml.PauliX(2) @ qml.PauliY(1),
            qml.PauliX(0) @ qml.PauliX(2),
        ]
        H = qml.Hamiltonian(coeffs_expected, obs_expected)

        assert H.compare(H1 @ H2)


class TestHamiltonianArithmeticAutograd:
    """Tests creation of Hamiltonians using arithmetic
    operations with autograd tensor coefficients."""

    def test_hamiltonian_equal(self):
        """Tests equality"""
        coeffs = pnp.array([0.5, -1.6])
        obs = [qml.PauliX(0), qml.PauliY(1)]
        H1 = qml.Hamiltonian(coeffs, obs)

        coeffs2 = pnp.array([-1.6, 0.5])
        obs2 = [qml.PauliY(1), qml.PauliX(0)]
        H2 = qml.Hamiltonian(coeffs2, obs2)

        assert H1.compare(H2)

    def test_hamiltonian_add(self):
        """Tests that Hamiltonians are added correctly"""
        coeffs = pnp.array([0.5, -1.6])
        obs = [qml.PauliX(0), qml.PauliY(1)]
        H1 = qml.Hamiltonian(coeffs, obs)

        coeffs2 = pnp.array([0.5, -0.4])
        H2 = qml.Hamiltonian(coeffs2, obs)

        coeffs_expected = pnp.array([1.0, -2.0])
        H = qml.Hamiltonian(coeffs_expected, obs)

        assert H.compare(H1 + H2)

        H1 += H2
        assert H.compare(H1)

    def test_hamiltonian_sub(self):
        """Tests that Hamiltonians are subtracted correctly"""
        coeffs = pnp.array([1.0, -2.0])
        obs = [qml.PauliX(0), qml.PauliY(1)]
        H1 = qml.Hamiltonian(coeffs, obs)

        coeffs2 = pnp.array([0.5, -0.4])
        H2 = qml.Hamiltonian(coeffs2, obs)

        coeffs_expected = pnp.array([0.5, -1.6])
        H = qml.Hamiltonian(coeffs_expected, obs)

        assert H.compare(H1 - H2)

        H1 -= H2
        assert H.compare(H1)

    def test_hamiltonian_matmul(self):
        """Tests that Hamiltonians are tensored correctly"""
        coeffs = pnp.array([1.0, 2.0])
        obs = [qml.PauliX(0), qml.PauliY(1)]
        H1 = qml.Hamiltonian(coeffs, obs)

        coeffs2 = pnp.array([-1.0, -2.0])
        obs2 = [qml.PauliX(2), qml.PauliY(3)]
        H2 = qml.Hamiltonian(coeffs2, obs2)

        coeffs_expected = pnp.array([-4.0, -2.0, -2.0, -1.0])
        obs_expected = [
            qml.PauliY(1) @ qml.PauliY(3),
            qml.PauliX(0) @ qml.PauliY(3),
            qml.PauliX(2) @ qml.PauliY(1),
            qml.PauliX(0) @ qml.PauliX(2),
        ]
        H = qml.Hamiltonian(coeffs_expected, obs_expected)

        assert H.compare(H1 @ H2)


class TestHamiltonianArithmeticJax:
    """Tests creation of Hamiltonians using arithmetic
    operations with jax tensor coefficients."""

    def test_hamiltonian_equal(self):
        """Tests equality"""
        jax = pytest.importorskip("jax")
        from jax import numpy as jnp

        coeffs = jnp.array([0.5, -1.6])
        obs = [qml.PauliX(0), qml.PauliY(1)]
        H1 = qml.Hamiltonian(coeffs, obs)

        coeffs2 = jnp.array([-1.6, 0.5])
        obs2 = [qml.PauliY(1), qml.PauliX(0)]
        H2 = qml.Hamiltonian(coeffs2, obs2)

        assert H1.compare(H2)

    def test_hamiltonian_add(self):
        """Tests that Hamiltonians are added correctly"""
        jax = pytest.importorskip("jax")
        from jax import numpy as jnp

        coeffs = jnp.array([0.5, -1.6])
        obs = [qml.PauliX(0), qml.PauliY(1)]
        H1 = qml.Hamiltonian(coeffs, obs)

        coeffs2 = jnp.array([0.5, -0.4])
        H2 = qml.Hamiltonian(coeffs2, obs)

        coeffs_expected = jnp.array([1.0, -2.0])
        H = qml.Hamiltonian(coeffs_expected, obs)

        assert H.compare(H1 + H2)

        H1 += H2
        assert H.compare(H1)

    def test_hamiltonian_sub(self):
        """Tests that Hamiltonians are subtracted correctly"""
        jax = pytest.importorskip("jax")
        from jax import numpy as jnp

        coeffs = jnp.array([1.0, -2.0])
        obs = [qml.PauliX(0), qml.PauliY(1)]
        H1 = qml.Hamiltonian(coeffs, obs)

        coeffs2 = jnp.array([0.5, -0.4])
        H2 = qml.Hamiltonian(coeffs2, obs)

        coeffs_expected = jnp.array([0.5, -1.6])
        H = qml.Hamiltonian(coeffs_expected, obs)

        assert H.compare(H1 - H2)

        H1 -= H2
        assert H.compare(H1)

    def test_hamiltonian_matmul(self):
        """Tests that Hamiltonians are tensored correctly"""
        jax = pytest.importorskip("jax")
        from jax import numpy as jnp

        coeffs = jnp.array([1.0, 2.0])
        obs = [qml.PauliX(0), qml.PauliY(1)]
        H1 = qml.Hamiltonian(coeffs, obs)

        coeffs2 = jnp.array([-1.0, -2.0])
        obs2 = [qml.PauliX(2), qml.PauliY(3)]
        H2 = qml.Hamiltonian(coeffs2, obs2)

        coeffs_expected = jnp.array([-4.0, -2.0, -2.0, -1.0])
        obs_expected = [
            qml.PauliY(1) @ qml.PauliY(3),
            qml.PauliX(0) @ qml.PauliY(3),
            qml.PauliX(2) @ qml.PauliY(1),
            qml.PauliX(0) @ qml.PauliX(2),
        ]
        H = qml.Hamiltonian(coeffs_expected, obs_expected)

        assert H.compare(H1 @ H2)


class TestHamiltonianEvaluation:
    """Test the usage of a Hamiltonian as an observable"""

    @pytest.mark.parametrize("coeffs, param, interface", COEFFS_PARAM_INTERFACE)
    def test_vqe_forward_different_coeff_types(self, coeffs, param, interface):
        """Check that manually splitting a Hamiltonian expectation has the same
        result as passing the Hamiltonian as an observable"""
        dev = qml.device("default.qubit", wires=2)
        H = qml.Hamiltonian(coeffs, [qml.PauliX(0), qml.PauliZ(0)])

        @qml.qnode(dev, interface=interface)
        def circuit():
            qml.RX(param, wires=0)
            qml.RY(param, wires=0)
            return qml.expval(H)

        @qml.qnode(dev, interface=interface)
        def circuit1():
            qml.RX(param, wires=0)
            qml.RY(param, wires=0)
            return qml.expval(qml.PauliX(0))

        @qml.qnode(dev, interface=interface)
        def circuit2():
            qml.RX(param, wires=0)
            qml.RY(param, wires=0)
            return qml.expval(qml.PauliZ(0))

        res = circuit()
        res_expected = coeffs[0] * circuit1() + coeffs[1] * circuit2()
        assert np.isclose(res, res_expected)

    def test_simplify_reduces_tape_parameters(self):
        """Test that simplifying a Hamiltonian reduces the number of parameters on a tape"""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit():
            qml.RY(0.1, wires=0)
            return qml.expval(
                qml.Hamiltonian([1.0, 2.0], [qml.PauliX(1), qml.PauliX(1)], simplify=True)
            )

        circuit()
        pars = circuit.qtape.get_parameters(trainable_only=False)
        # simplify worked and added 1. and 2.
        assert pars == [0.1, 3.0]


class TestHamiltonianDifferentiation:
    """Test that the Hamiltonian coefficients are differentiable"""

    @pytest.mark.parametrize("simplify", [True, False])
    def test_vqe_differentiation_paramshift(self, simplify):
        """Test the parameter-shift method by comparing the differentiation of linearly combined subcircuits
        with the differentiation of a Hamiltonian expectation"""
        coeffs = np.array([-0.05, 0.17])
        param = np.array(1.7)

        # differentiating a circuit with measurement expval(H)
        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit(coeffs, param):
            qml.RX(param, wires=0)
            qml.RY(param, wires=0)
            return qml.expval(
                qml.Hamiltonian(coeffs, [qml.PauliX(0), qml.PauliZ(0)], simplify=simplify)
            )

        grad_fn = qml.grad(circuit)
        grad = grad_fn(coeffs, param)

        # differentiating a cost that combines circuits with
        # measurements expval(Pauli)
        half1 = qml.QNode(circuit1, dev, diff_method="parameter-shift")
        half2 = qml.QNode(circuit2, dev, diff_method="parameter-shift")

        def combine(coeffs, param):
            return coeffs[0] * half1(param) + coeffs[1] * half2(param)

        grad_fn_expected = qml.grad(combine)
        grad_expected = grad_fn_expected(coeffs, param)

        assert np.allclose(grad[0], grad_expected[0])
        assert np.allclose(grad[1], grad_expected[1])

    @pytest.mark.parametrize("simplify", [True, False])
    def test_vqe_differentiation_autograd(self, simplify):
        """Test the autograd interface by comparing the differentiation of linearly combined subcircuits
        with the differentiation of a Hamiltonian expectation"""
        coeffs = pnp.array([-0.05, 0.17], requires_grad=True)
        param = pnp.array(1.7, requires_grad=True)

        # differentiating a circuit with measurement expval(H)
        @qml.qnode(dev, interface="autograd")
        def circuit(coeffs, param):
            qml.RX(param, wires=0)
            qml.RY(param, wires=0)
            return qml.expval(
                qml.Hamiltonian(coeffs, [qml.PauliX(0), qml.PauliZ(0)], simplify=simplify)
            )

        grad_fn = qml.grad(circuit)
        grad = grad_fn(coeffs, param)

        # differentiating a cost that combines circuits with
        # measurements expval(Pauli)
        half1 = qml.QNode(circuit1, dev, interface="autograd")
        half2 = qml.QNode(circuit2, dev, interface="autograd")

        def combine(coeffs, param):
            return coeffs[0] * half1(param) + coeffs[1] * half2(param)

        grad_fn_expected = qml.grad(combine)
        grad_expected = grad_fn_expected(coeffs, param)

        assert np.allclose(grad[0], grad_expected[0])
        assert np.allclose(grad[1], grad_expected[1])

    @pytest.mark.parametrize("simplify", [True, False])
    def test_vqe_differentiation_jax(self, simplify):
        """Test the jax interface by comparing the differentiation of linearly combined subcircuits
        with the differentiation of a Hamiltonian expectation"""

        jax = pytest.importorskip("jax")
        jnp = pytest.importorskip("jax.numpy")
        coeffs = jnp.array([-0.05, 0.17])
        param = jnp.array(1.7)

        # differentiating a circuit with measurement expval(H)
        @qml.qnode(dev, interface="jax", diff_method="backprop")
        def circuit(coeffs, param):
            qml.RX(param, wires=0)
            qml.RY(param, wires=0)
            return qml.expval(
                qml.Hamiltonian(coeffs, [qml.PauliX(0), qml.PauliZ(0)], simplify=simplify)
            )

        grad_fn = jax.grad(circuit)
        grad = grad_fn(coeffs, param)

        # differentiating a cost that combines circuits with
        # measurements expval(Pauli)
        half1 = qml.QNode(circuit1, dev, interface="jax", diff_method="backprop")
        half2 = qml.QNode(circuit2, dev, interface="jax", diff_method="backprop")

        def combine(coeffs, param):
            return coeffs[0] * half1(param) + coeffs[1] * half2(param)

        grad_fn_expected = jax.grad(combine)
        grad_expected = grad_fn_expected(coeffs, param)

        assert np.allclose(grad[0], grad_expected[0])
        assert np.allclose(grad[1], grad_expected[1])

    @pytest.mark.parametrize("simplify", [True, False])
    def test_vqe_differentiation_torch(self, simplify):
        """Test the torch interface by comparing the differentiation of linearly combined subcircuits
        with the differentiation of a Hamiltonian expectation"""

        torch = pytest.importorskip("torch")
        coeffs = torch.tensor([-0.05, 0.17], requires_grad=True)
        param = torch.tensor(1.7, requires_grad=True)

        # differentiating a circuit with measurement expval(H)
        @qml.qnode(dev, interface="torch")
        def circuit(coeffs, param):
            qml.RX(param, wires=0)
            qml.RY(param, wires=0)
            return qml.expval(
                qml.Hamiltonian(coeffs, [qml.PauliX(0), qml.PauliZ(0)], simplify=simplify)
            )

        res = circuit(coeffs, param)
        res.backward()
        grad = (coeffs.grad, param.grad)

        # differentiating a cost that combines circuits with
        # measurements expval(Pauli)

        # we need to create new tensors here
        coeffs2 = torch.tensor([-0.05, 0.17], requires_grad=True)
        param2 = torch.tensor(1.7, requires_grad=True)

        half1 = qml.QNode(circuit1, dev, interface="torch")
        half2 = qml.QNode(circuit2, dev, interface="torch")

        def combine(coeffs, param):
            return coeffs[0] * half1(param) + coeffs[1] * half2(param)

        res_expected = combine(coeffs2, param2)
        res_expected.backward()
        grad_expected = (coeffs2.grad, param2.grad)

        assert np.allclose(grad[0], grad_expected[0])
        assert np.allclose(grad[1], grad_expected[1])

    @pytest.mark.parametrize("simplify", [True, False])
    def test_vqe_differentiation_tf(self, simplify):
        """Test the tf interface by comparing the differentiation of linearly combined subcircuits
        with the differentiation of a Hamiltonian expectation"""

        tf = pytest.importorskip("tf")
        coeffs = tf.Variable([-0.05, 0.17], dtype=tf.double)
        param = tf.Variable(1.7, dtype=tf.double)

        # differentiating a circuit with measurement expval(H)
        @qml.qnode(dev, interface="tf", diff_method="backprop")
        def circuit(coeffs, param):
            qml.RX(param, wires=0)
            qml.RY(param, wires=0)
            return qml.expval(
                qml.Hamiltonian(coeffs, [qml.PauliX(0), qml.PauliZ(0)], simplify=simplify)
            )

        with tf.GradientTape() as tape:
            res = circuit(coeffs, param)
        grad = tape.gradient(res, [coeffs, param])

        # differentiating a cost that combines circuits with
        # measurements expval(Pauli)

        # we need to create new tensors here
        coeffs2 = tf.Variable([-0.05, 0.17], dtype=tf.double)
        param2 = tf.Variable(1.7, dtype=tf.double)
        half1 = qml.QNode(circuit1, dev, interface="tf", diff_method="backprop")
        half2 = qml.QNode(circuit2, dev, interface="tf", diff_method="backprop")

        def combine(coeffs, param):
            return coeffs[0] * half1(param) + coeffs[1] * half2(param)

        with tf.GradientTape() as tape2:
            res_expected = combine(coeffs2, param2)
        grad_expected = tape2.gradient(res_expected, [coeffs2, param2])

        assert np.allclose(grad[0], grad_expected[0])
        assert np.allclose(grad[1], grad_expected[1])
