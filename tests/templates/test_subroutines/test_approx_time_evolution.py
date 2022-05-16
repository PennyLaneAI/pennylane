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
Tests for the ApproxTimeEvolution template.
"""
import pytest
import numpy as np
from pennylane import numpy as pnp
import pennylane as qml


class TestDecomposition:
    """Tests that the template defines the correct decomposition."""

    @pytest.mark.parametrize(
        ("time", "hamiltonian", "steps", "expected_queue"),
        [
            (
                2,
                qml.Hamiltonian([1, 1], [qml.PauliX(0), qml.PauliX(1)]),
                2,
                [
                    qml.PauliRot(2.0, "X", wires=[0]),
                    qml.PauliRot(2.0, "X", wires=[1]),
                    qml.PauliRot(2.0, "X", wires=[0]),
                    qml.PauliRot(2.0, "X", wires=[1]),
                ],
            ),
            (
                2,
                qml.Hamiltonian([2, 0.5], [qml.PauliX("a"), qml.PauliZ("b") @ qml.PauliX("a")]),
                2,
                [
                    qml.PauliRot(4.0, "X", wires=["a"]),
                    qml.PauliRot(1.0, "ZX", wires=["b", "a"]),
                    qml.PauliRot(4.0, "X", wires=["a"]),
                    qml.PauliRot(1.0, "ZX", wires=["b", "a"]),
                ],
            ),
            (
                2,
                qml.Hamiltonian([1, 1], [qml.PauliX(0), qml.Identity(0) @ qml.Identity(1)]),
                2,
                [qml.PauliRot(2.0, "X", wires=[0]), qml.PauliRot(2.0, "X", wires=[0])],
            ),
            (
                2,
                qml.Hamiltonian(
                    [2, 0.5, 0.5],
                    [
                        qml.PauliX("a"),
                        qml.PauliZ(-15) @ qml.PauliX("a"),
                        qml.Identity(0) @ qml.PauliY(-15),
                    ],
                ),
                1,
                [
                    qml.PauliRot(8.0, "X", wires=["a"]),
                    qml.PauliRot(2.0, "ZX", wires=[-15, "a"]),
                    qml.PauliRot(2.0, "IY", wires=[0, -15]),
                ],
            ),
        ],
    )
    def test_evolution_operations(self, time, hamiltonian, steps, expected_queue):
        """Tests that the sequence of gates implemented in the ApproxTimeEvolution template is correct"""

        op = qml.ApproxTimeEvolution(hamiltonian, time, steps)
        queue = op.expand().operations

        for expected_gate, gate in zip(expected_queue, queue):
            prep = [gate.parameters, gate.wires]
            target = [expected_gate.parameters, expected_gate.wires]

            assert prep == target

    @pytest.mark.parametrize(
        ("time", "hamiltonian", "steps", "expectation"),
        [
            (np.pi, qml.Hamiltonian([1, 1], [qml.PauliX(0), qml.PauliX(1)]), 2, [1.0, 1.0]),
            (
                np.pi / 2,
                qml.Hamiltonian([0.5, 1], [qml.PauliY(0), qml.Identity(0) @ qml.PauliX(1)]),
                1,
                [0.0, -1.0],
            ),
            (
                np.pi / 4,
                qml.Hamiltonian(
                    [1, 1, 1], [qml.PauliX(0), qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliX(1)]
                ),
                1,
                [0.0, 0.0],
            ),
            (
                1,
                qml.Hamiltonian([1, 1], [qml.PauliX(0), qml.PauliX(1)]),
                2,
                [-0.41614684, -0.41614684],
            ),
            (
                2,
                qml.Hamiltonian(
                    [1, 1, 1, 1],
                    [qml.PauliX(0), qml.PauliY(0), qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliY(1)],
                ),
                2,
                [-0.87801124, 0.51725747],
            ),
        ],
    )
    def test_evolution_output(self, time, hamiltonian, steps, expectation):
        """Tests that the output from the ApproxTimeEvolution template is correct"""

        n_wires = 2
        dev = qml.device("default.qubit", wires=n_wires)

        @qml.qnode(dev)
        def circuit():
            qml.ApproxTimeEvolution(hamiltonian, time, steps)
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_wires)]

        assert np.allclose(circuit(), expectation)

    def test_custom_wire_labels(self, tol):
        """Test that template can deal with non-numeric, nonconsecutive wire labels."""
        hamiltonian = qml.Hamiltonian([1, 1, 1], [qml.PauliX(0), qml.PauliX(1), qml.PauliX(2)])
        hamiltonian2 = qml.Hamiltonian(
            [1, 1, 1], [qml.PauliX("z"), qml.PauliX("a"), qml.PauliX("k")]
        )

        dev = qml.device("default.qubit", wires=3)
        dev2 = qml.device("default.qubit", wires=["z", "a", "k"])

        @qml.qnode(dev)
        def circuit():
            qml.ApproxTimeEvolution(hamiltonian, 0.5, 2)
            return qml.expval(qml.Identity(0))

        @qml.qnode(dev2)
        def circuit2():
            qml.ApproxTimeEvolution(hamiltonian2, 0.5, 2)
            return qml.expval(qml.Identity("z"))

        circuit()
        circuit2()

        assert np.allclose(dev.state, dev2.state, atol=tol, rtol=0)


class TestInputs:
    """Test inputs and pre-processing."""

    def test_hamiltonian_error(self):
        """Tests if the correct error is thrown when hamiltonian is not a pennylane.Hamiltonian object"""

        n_wires = 2
        dev = qml.device("default.qubit", wires=n_wires)

        hamiltonian = np.array([[1, 1], [1, 1]])

        @qml.qnode(dev)
        def circuit():
            qml.ApproxTimeEvolution(hamiltonian, 2, 3)
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_wires)]

        with pytest.raises(ValueError, match="hamiltonian must be of type pennylane.Hamiltonian"):
            circuit()

    @pytest.mark.parametrize(
        ("hamiltonian"),
        [
            qml.Hamiltonian([1, 1], [qml.PauliX(0), qml.Hadamard(0)]),
            qml.Hamiltonian(
                [1, 1],
                [qml.PauliX(0) @ qml.Hermitian(np.array([[1, 1], [1, 1]]), 1), qml.PauliX(0)],
            ),
        ],
    )
    def test_non_pauli_error(self, hamiltonian):
        """Tests if the correct errors are thrown when the user attempts to input a matrix with non-Pauli terms"""

        n_wires = 2
        dev = qml.device("default.qubit", wires=n_wires)

        @qml.qnode(dev)
        def circuit():
            qml.ApproxTimeEvolution(hamiltonian, 2, 3)
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_wires)]

        with pytest.raises(
            ValueError, match="hamiltonian must be written in terms of Pauli matrices"
        ):
            circuit()

    def test_id(self):
        """Tests that the id attribute can be set."""
        h = qml.Hamiltonian([1, 1], [qml.PauliX(0), qml.PauliY(0)])
        template = qml.ApproxTimeEvolution(h, 2, 3, id="a")
        assert template.id == "a"

    def test_wire_indices(self):
        """Tests that correct wires are set."""
        wire_indices = [0, 1]
        H = (
            qml.PauliX(wire_indices[0])
            + qml.PauliZ(wire_indices[1])
            + 0.5 * qml.PauliX(wire_indices[0]) @ qml.PauliX(wire_indices[1])
        )
        qml.ApproxTimeEvolution(H, 0.5, 2)
        assert wire_indices[0] in H.wires
        assert wire_indices[1] in H.wires


# test data for gradient tests

hamiltonian = qml.Hamiltonian([1, 1], [qml.PauliX(0), qml.PauliX(1)])
n = 2


def circuit_template(time):
    qml.ApproxTimeEvolution(hamiltonian, time, n)
    return qml.expval(qml.PauliZ(0))


def circuit_decomposed(time):
    qml.PauliRot(time, "X", wires=[0])
    qml.PauliRot(time, "X", wires=[1])
    qml.PauliRot(time, "X", wires=[0])
    qml.PauliRot(time, "X", wires=[1])
    return qml.expval(qml.PauliZ(0))


class TestInterfaces:
    """Tests that the template is compatible with all interfaces, including the computation
    of gradients."""

    def test_float(self, tol):
        """Tests float as input."""

        time = 0.5

        dev = qml.device("default.qubit", wires=3)

        circuit = qml.QNode(circuit_template, dev)
        circuit2 = qml.QNode(circuit_decomposed, dev)

        res = circuit(time)
        res2 = circuit2(time)
        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

    @pytest.mark.autograd
    def test_autograd(self, tol):
        """Tests the autograd interface."""

        time = pnp.array(0.5, requires_grad=True)

        dev = qml.device("default.qubit", wires=3)

        circuit = qml.QNode(circuit_template, dev)
        circuit2 = qml.QNode(circuit_decomposed, dev)

        res = circuit(time)
        res2 = circuit2(time)
        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

        grad_fn = qml.grad(circuit)
        grads = grad_fn(time)

        grad_fn2 = qml.grad(circuit2)
        grads2 = grad_fn2(time)

        assert np.allclose(grads, grads2, atol=tol, rtol=0)

    @pytest.mark.jax
    def test_jax(self, tol):
        """Tests the jax interface."""

        import jax
        import jax.numpy as jnp

        time = jnp.array(0.5)

        dev = qml.device("default.qubit", wires=3)

        circuit = qml.QNode(circuit_template, dev, interface="jax")
        circuit2 = qml.QNode(circuit_decomposed, dev, interface="jax")

        res = circuit(time)
        res2 = circuit2(time)
        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

        grad_fn = jax.grad(circuit)
        grads = grad_fn(time)

        grad_fn2 = jax.grad(circuit2)
        grads2 = grad_fn2(time)

        assert np.allclose(grads, grads2, atol=tol, rtol=0)

    @pytest.mark.tf
    def test_tf(self, tol):
        """Tests the tf interface."""

        import tensorflow as tf

        time = tf.Variable(0.5)

        dev = qml.device("default.qubit", wires=3)

        circuit = qml.QNode(circuit_template, dev, interface="tf")
        circuit2 = qml.QNode(circuit_decomposed, dev, interface="tf")

        res = circuit(time)
        res2 = circuit2(time)
        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

        with tf.GradientTape() as tape:
            res = circuit(time)
        grads = tape.gradient(res, [time])

        with tf.GradientTape() as tape2:
            res2 = circuit2(time)
        grads2 = tape2.gradient(res2, [time])

        assert np.allclose(grads[0], grads2[0], atol=tol, rtol=0)

    @pytest.mark.torch
    def test_torch(self, tol):
        """Tests the torch interface."""

        import torch

        time = torch.tensor(0.5, requires_grad=True)

        dev = qml.device("default.qubit", wires=3)

        circuit = qml.QNode(circuit_template, dev, interface="torch")
        circuit2 = qml.QNode(circuit_decomposed, dev, interface="torch")

        res = circuit(time)
        res2 = circuit2(time)
        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

        res = circuit(time)
        res.backward()
        grads = [time.grad]

        res2 = circuit2(time)
        res2.backward()
        grads2 = [time.grad]

        assert np.allclose(grads[0], grads2[0], atol=tol, rtol=0)


@pytest.mark.autograd
@pytest.mark.parametrize(
    "dev_name,diff_method",
    [["default.qubit.autograd", "backprop"], ["default.qubit", qml.gradients.param_shift]],
)
def test_trainable_hamiltonian(dev_name, diff_method):
    """Test that the ApproxTimeEvolution template
    can be differentiated if the Hamiltonian coefficients are trainable"""
    dev = qml.device(dev_name, wires=2)

    obs = [qml.PauliX(0) @ qml.PauliY(1), qml.PauliY(0) @ qml.PauliX(1)]

    def create_tape(coeffs, t):
        H = qml.Hamiltonian(coeffs, obs)

        with qml.tape.QuantumTape() as tape:
            qml.templates.ApproxTimeEvolution(H, t, 2)
            qml.expval(qml.PauliZ(0))

        return tape

    def cost(coeffs, t):
        tape = create_tape(coeffs, t)

        if diff_method is qml.gradients.param_shift:
            tape = dev.expand_fn(tape)

        return qml.execute([tape], dev, diff_method)[0]

    t = pnp.array(0.54, requires_grad=True)
    coeffs = pnp.array([-0.6, 2.0], requires_grad=True)

    res = cost(coeffs, t)
    grad = qml.grad(cost)(coeffs, t)

    assert len(grad) == 2

    assert isinstance(grad[0], np.ndarray)
    assert grad[0].shape == (2,)

    assert isinstance(grad[1], np.ndarray)
    assert grad[1].shape == tuple()

    # compare to finite-differences
    tape = create_tape(coeffs, t)
    g_tapes, fn = qml.gradients.finite_diff(tape, _expand=False, validate_params=False)
    expected = fn(qml.execute(g_tapes, dev, None))[0]

    assert np.allclose(grad[0], expected[0:1])
    assert np.allclose(grad[1], expected[2])
