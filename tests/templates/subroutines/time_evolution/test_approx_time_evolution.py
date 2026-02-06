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
import numpy as np
import pytest

import pennylane as qp
from pennylane import numpy as pnp
from pennylane.ops.functions.assert_valid import _test_decomposition_rule


@pytest.mark.jax
def test_standard_validity():
    """Run standard tests of operation validity."""
    H = 2.0 * qp.PauliX(0) + 3.0 * qp.PauliY(0)
    t = 0.1
    op = qp.ApproxTimeEvolution(H, t, n=20)
    qp.ops.functions.assert_valid(op)


# pylint: disable=protected-access
def test_flatten_unflatten():
    """Tests the _flatten and _unflatten methods."""
    H = 2.0 * qp.PauliX(0) + 3.0 * qp.PauliY(0)
    t = 0.1
    op = qp.ApproxTimeEvolution(H, t, n=20)
    data, metadata = op._flatten()
    assert data[0] is H
    assert data[1] == t
    assert metadata == (20,)

    # check metadata hashable
    assert hash(metadata)

    new_op = type(op)._unflatten(*op._flatten())
    qp.assert_equal(op, new_op)
    assert new_op is not op


def test_queuing():
    """Test that ApproxTimeEvolution de-queues the input hamiltonian."""

    with qp.queuing.AnnotatedQueue() as q:
        H = qp.X(0) + qp.Y(1)
        op = qp.ApproxTimeEvolution(H, 0.1, n=20)

    assert len(q.queue) == 1
    assert q.queue[0] is op


class TestDecomposition:
    """Tests that the template defines the correct decomposition."""

    @pytest.mark.parametrize(
        ("time", "hamiltonian", "steps", "expected_queue"),
        [
            (
                2,
                qp.Hamiltonian([1, 1], [qp.PauliX(0), qp.PauliX(1)]),
                2,
                [
                    qp.PauliRot(2.0, "X", wires=[0]),
                    qp.PauliRot(2.0, "X", wires=[1]),
                    qp.PauliRot(2.0, "X", wires=[0]),
                    qp.PauliRot(2.0, "X", wires=[1]),
                ],
            ),
            (
                2,
                qp.Hamiltonian([2, 0.5], [qp.PauliX("a"), qp.PauliX("a") @ qp.PauliZ("b")]),
                2,
                [
                    qp.PauliRot(4.0, "X", wires=["a"]),
                    qp.PauliRot(1.0, "XZ", wires=["a", "b"]),
                    qp.PauliRot(4.0, "X", wires=["a"]),
                    qp.PauliRot(1.0, "XZ", wires=["a", "b"]),
                ],
            ),
            (
                2,
                qp.Hamiltonian([1, 1], [qp.PauliX(0), qp.Identity(0) @ qp.Identity(1)]),
                2,
                [qp.PauliRot(2.0, "X", wires=[0]), qp.PauliRot(2.0, "X", wires=[0])],
            ),
            (
                2,
                qp.Hamiltonian(
                    [2, 0.5, 0.5],
                    [
                        qp.PauliX("a"),
                        qp.PauliX("a") @ qp.PauliZ(-15),
                        qp.Identity(0) @ qp.PauliY(-15),
                    ],
                ),
                1,
                [
                    qp.PauliRot(8.0, "X", wires=["a"]),
                    qp.PauliRot(2.0, "XZ", wires=["a", -15]),
                    qp.PauliRot(2.0, "Y", wires=[-15]),
                ],
            ),
        ],
    )
    def test_evolution_operations(self, time, hamiltonian, steps, expected_queue):
        """Tests that the sequence of gates implemented in the ApproxTimeEvolution template is correct"""

        op = qp.ApproxTimeEvolution(hamiltonian, time, steps)
        queue = op.decomposition()

        for expected_gate, gate in zip(expected_queue, queue):
            qp.assert_equal(expected_gate, gate)

    @pytest.mark.parametrize(
        ("time", "hamiltonian", "steps", "expectation"),
        [
            (np.pi, qp.Hamiltonian([1, 1], [qp.PauliX(0), qp.PauliX(1)]), 2, [1.0, 1.0]),
            (
                np.pi / 2,
                qp.Hamiltonian([0.5, 1], [qp.PauliY(0), qp.Identity(0) @ qp.PauliX(1)]),
                1,
                [0.0, -1.0],
            ),
            (
                np.pi / 4,
                qp.Hamiltonian(
                    [1, 1, 1], [qp.PauliX(0), qp.PauliZ(0) @ qp.PauliZ(1), qp.PauliX(1)]
                ),
                1,
                [0.0, 0.0],
            ),
            (
                1,
                qp.Hamiltonian([1, 1], [qp.PauliX(0), qp.PauliX(1)]),
                2,
                [-0.41614684, -0.41614684],
            ),
            (
                2,
                qp.Hamiltonian(
                    [1, 1, 1, 1],
                    [qp.PauliX(0), qp.PauliY(0), qp.PauliZ(0) @ qp.PauliZ(1), qp.PauliY(1)],
                ),
                2,
                [-0.87801124, 0.51725747],
            ),
        ],
    )
    def test_evolution_output(self, time, hamiltonian, steps, expectation):
        """Tests that the output from the ApproxTimeEvolution template is correct"""

        n_wires = 2
        dev = qp.device("default.qubit", wires=n_wires)

        @qp.qnode(dev)
        def circuit():
            qp.ApproxTimeEvolution(hamiltonian, time, steps)
            return [qp.expval(qp.PauliZ(wires=i)) for i in range(n_wires)]

        assert np.allclose(circuit(), expectation)

    def test_custom_wire_labels(self, tol):
        """Test that template can deal with non-numeric, nonconsecutive wire labels."""
        hamiltonian = qp.Hamiltonian([1, 1, 1], [qp.PauliX(0), qp.PauliX(1), qp.PauliX(2)])
        hamiltonian2 = qp.Hamiltonian(
            [1, 1, 1], [qp.PauliX("z"), qp.PauliX("a"), qp.PauliX("k")]
        )

        dev = qp.device("default.qubit", wires=3)
        dev2 = qp.device("default.qubit", wires=["z", "a", "k"])

        @qp.qnode(dev)
        def circuit():
            qp.ApproxTimeEvolution(hamiltonian, 0.5, 2)
            return qp.expval(qp.Identity(0)), qp.state()

        @qp.qnode(dev2)
        def circuit2():
            qp.ApproxTimeEvolution(hamiltonian2, 0.5, 2)
            return qp.expval(qp.Identity("z")), qp.state()

        res1, state1 = circuit()
        res2, state2 = circuit2()

        assert np.allclose(res1, res2, atol=tol, rtol=0)
        assert np.allclose(state1, state2, atol=tol, rtol=0)

    DECOMP_PARAMS = [
        (qp.Hamiltonian([1, 1], [qp.PauliX(0), qp.PauliY(0)]), 0.5, 2),
        (
            qp.Hamiltonian(
                [1, 1],
                [qp.PauliX(0) @ qp.PauliZ(1), qp.PauliX(0)],
            ),
            2,
            3,
        ),
    ]

    @pytest.mark.capture
    @pytest.mark.parametrize(("hamiltonian", "time", "steps"), DECOMP_PARAMS)
    def test_decomposition_new(self, hamiltonian, time, steps):
        op = qp.ApproxTimeEvolution(hamiltonian, time, steps)
        for rule in qp.list_decomps(qp.ApproxTimeEvolution):
            _test_decomposition_rule(op, rule)


class TestInputs:
    """Test inputs and pre-processing."""

    def test_hamiltonian_error(self):
        """Tests if the correct error is thrown when hamiltonian is not a pennylane.Hamiltonian object"""

        n_wires = 2
        dev = qp.device("default.qubit", wires=n_wires)

        hamiltonian = np.array([[1, 1], [1, 1]])

        @qp.qnode(dev)
        def circuit():
            qp.ApproxTimeEvolution(hamiltonian, 2, 3)
            return [qp.expval(qp.PauliZ(wires=i)) for i in range(n_wires)]

        with pytest.raises(
            ValueError, match="hamiltonian must be a linear combination of pauli words"
        ):
            circuit()

    @pytest.mark.parametrize(
        ("hamiltonian"),
        [
            qp.Hamiltonian([1, 1], [qp.PauliX(0), qp.Hadamard(0)]),
            qp.Hamiltonian(
                [1, 1],
                [qp.PauliX(0) @ qp.Hermitian(np.array([[1, 1], [1, 1]]), 1), qp.PauliX(0)],
            ),
        ],
    )
    def test_non_pauli_error(self, hamiltonian):
        """Tests if the correct errors are thrown when the user attempts to input a matrix with non-Pauli terms"""

        n_wires = 2
        dev = qp.device("default.qubit", wires=n_wires)

        @qp.qnode(dev)
        def circuit():
            qp.ApproxTimeEvolution(hamiltonian, 2, 3)
            return [qp.expval(qp.PauliZ(wires=i)) for i in range(n_wires)]

        with pytest.raises(
            ValueError, match="hamiltonian must be a linear combination of pauli words"
        ):
            circuit()

    def test_id(self):
        """Tests that the id attribute can be set."""
        h = qp.Hamiltonian([1, 1], [qp.PauliX(0), qp.PauliY(0)])
        template = qp.ApproxTimeEvolution(h, 2, 3, id="a")
        assert template.id == "a"

    def test_wire_indices(self):
        """Tests that correct wires are set."""
        wire_indices = [0, 1]
        H = (
            qp.PauliX(wire_indices[0])
            + qp.PauliZ(wire_indices[1])
            + 0.5 * qp.PauliX(wire_indices[0]) @ qp.PauliX(wire_indices[1])
        )
        qp.ApproxTimeEvolution(H, 0.5, 2)
        assert wire_indices[0] in H.wires
        assert wire_indices[1] in H.wires


# test data for gradient tests

ham = qp.Hamiltonian([1, 1], [qp.PauliX(0), qp.PauliX(1)])
n = 2


def circuit_template(time):
    qp.ApproxTimeEvolution(ham, time, n)
    return qp.expval(qp.PauliZ(0))


def circuit_decomposed(time):
    qp.PauliRot(time, "X", wires=[0])
    qp.PauliRot(time, "X", wires=[1])
    qp.PauliRot(time, "X", wires=[0])
    qp.PauliRot(time, "X", wires=[1])
    return qp.expval(qp.PauliZ(0))


class TestInterfaces:
    """Tests that the template is compatible with all interfaces, including the computation
    of gradients."""

    def test_float(self, tol):
        """Tests float as input."""

        time = 0.5

        dev = qp.device("default.qubit", wires=3)

        circuit = qp.QNode(circuit_template, dev)
        circuit2 = qp.QNode(circuit_decomposed, dev)

        res = circuit(time)
        res2 = circuit2(time)
        assert qp.math.allclose(res, res2, atol=tol, rtol=0)

    @pytest.mark.autograd
    def test_autograd(self, tol):
        """Tests the autograd interface."""

        time = pnp.array(0.5, requires_grad=True)

        dev = qp.device("default.qubit", wires=3)

        circuit = qp.QNode(circuit_template, dev)
        circuit2 = qp.QNode(circuit_decomposed, dev)

        res = circuit(time)
        res2 = circuit2(time)
        assert qp.math.allclose(res, res2, atol=tol, rtol=0)

        grad_fn = qp.grad(circuit)
        grads = grad_fn(time)

        grad_fn2 = qp.grad(circuit2)
        grads2 = grad_fn2(time)

        assert np.allclose(grads, grads2, atol=tol, rtol=0)

    @pytest.mark.jax
    def test_jax(self, tol):
        """Tests the jax interface."""

        import jax
        import jax.numpy as jnp

        time = jnp.array(0.5)

        dev = qp.device("default.qubit", wires=3)

        circuit = qp.QNode(circuit_template, dev)
        circuit2 = qp.QNode(circuit_decomposed, dev)

        res = circuit(time)
        res2 = circuit2(time)
        assert qp.math.allclose(res, res2, atol=tol, rtol=0)

        grad_fn = jax.grad(circuit)
        grads = grad_fn(time)

        grad_fn2 = jax.grad(circuit2)
        grads2 = grad_fn2(time)

        assert np.allclose(grads, grads2, atol=tol, rtol=0)

    @pytest.mark.jax
    def test_jax_jit(self, tol):
        """Tests jit within the jax interface."""

        import jax
        import jax.numpy as jnp

        time = jnp.array(0.5)

        dev = qp.device("default.qubit", wires=3)

        circuit = qp.QNode(circuit_template, dev)
        circuit2 = jax.jit(circuit)

        res = circuit(time)
        res2 = circuit2(time)
        assert qp.math.allclose(res, res2, atol=tol, rtol=0)

        grad_fn = jax.grad(circuit)
        grads = grad_fn(time)

        grad_fn2 = jax.grad(circuit2)
        grads2 = grad_fn2(time)

        assert qp.math.allclose(grads, grads2, atol=tol, rtol=0)

    @pytest.mark.tf
    def test_tf(self, tol):
        """Tests the tf interface."""

        import tensorflow as tf

        time = tf.Variable(0.5)

        dev = qp.device("default.qubit", wires=3)

        circuit = qp.QNode(circuit_template, dev)
        circuit2 = qp.QNode(circuit_decomposed, dev)

        res = circuit(time)
        res2 = circuit2(time)
        assert qp.math.allclose(res, res2, atol=tol, rtol=0)

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

        dev = qp.device("default.qubit", wires=3)

        circuit = qp.QNode(circuit_template, dev)
        circuit2 = qp.QNode(circuit_decomposed, dev)

        res = circuit(time)
        res2 = circuit2(time)
        assert qp.math.allclose(res, res2, atol=tol, rtol=0)

        res = circuit(time)
        res.backward()
        grads = [time.grad]

        res2 = circuit2(time)
        res2.backward()
        grads2 = [time.grad]

        assert np.allclose(grads[0], grads2[0], atol=tol, rtol=0)


# pylint: disable=protected-access, unexpected-keyword-arg
@pytest.mark.autograd
@pytest.mark.parametrize(
    "dev_name,diff_method",
    [["default.qubit", "backprop"], ["default.qubit", qp.gradients.param_shift]],
)
def test_trainable_hamiltonian(dev_name, diff_method):
    """Test that the ApproxTimeEvolution template
    can be differentiated if the Hamiltonian coefficients are trainable"""
    dev = qp.device(dev_name, wires=2)

    obs = [qp.PauliX(0) @ qp.PauliY(1), qp.PauliY(0) @ qp.PauliX(1)]

    def create_tape(coeffs, t):
        H = qp.Hamiltonian(coeffs, obs)

        with qp.queuing.AnnotatedQueue() as q:
            qp.templates.ApproxTimeEvolution(H, t, 2)
            qp.expval(qp.PauliZ(0))

        tape = qp.tape.QuantumScript.from_queue(q)
        return tape

    def cost(coeffs, t):
        tape = create_tape(coeffs, t)

        if diff_method is qp.gradients.param_shift and dev_name != "default.qubit":
            tape = dev.expand_fn(tape)
            return qp.execute([tape], dev, diff_method)[0]
        program = dev.preprocess_transforms()
        return qp.execute([tape], dev, diff_method=diff_method, transform_program=program)[0]

    t = pnp.array(0.54, requires_grad=True)
    coeffs = pnp.array([-0.6, 2.0], requires_grad=True)

    cost(coeffs, t)
    grad = qp.grad(cost)(coeffs, t)

    assert len(grad) == 2

    assert isinstance(grad[0], np.ndarray)
    assert grad[0].shape == (2,)

    assert isinstance(grad[1], np.ndarray)
    assert grad[1].shape == tuple()

    # compare to finite-differences

    @qp.qnode(dev, diff_method="finite-diff")
    def circuit(coeffs, t):
        H = qp.Hamiltonian(coeffs, obs)
        qp.ApproxTimeEvolution(H, t, 2)
        return qp.expval(qp.PauliZ(0))

    expected = qp.grad(circuit)(coeffs, t)
    assert np.allclose(qp.math.hstack(grad), qp.math.hstack(expected))
