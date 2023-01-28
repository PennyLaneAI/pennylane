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
Unit tests for the ParticleConservingU2 template.
"""
import pytest
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


class TestDecomposition:
    """Tests that the template defines the correct decomposition."""

    @pytest.mark.parametrize(
        "layers, qubits, init_state",
        [
            (2, 4, np.array([1, 1, 0, 0])),
            (1, 6, np.array([1, 1, 0, 0, 0, 0])),
            (1, 5, np.array([1, 1, 0, 0, 0])),
        ],
    )
    def test_operations(self, layers, qubits, init_state):
        """Test the correctness of the ParticleConservingU2 template including the gate count
        and order, the wires each operation acts on and the correct use of parameters
        in the circuit."""
        weights = np.random.normal(0, 2 * np.pi, (layers, 2 * qubits - 1))

        n_gates = 1 + (qubits + (qubits - 1) * 3) * layers

        exp_gates = (
            [qml.RZ] * qubits + ([qml.CNOT] + [qml.CRX] + [qml.CNOT]) * (qubits - 1)
        ) * layers

        op = qml.ParticleConservingU2(weights, wires=range(qubits), init_state=init_state)
        queue = op.expand().operations

        # number of gates
        assert len(queue) == n_gates

        # initialization
        assert isinstance(queue[0], qml.BasisEmbedding)

        # order of gates
        for op1, op2 in zip(queue[1:], exp_gates):
            assert isinstance(op1, op2)

        # gate parameter
        params = np.array(
            [queue[i].parameters for i in range(1, n_gates) if queue[i].parameters != []]
        )
        weights[:, qubits:] = weights[:, qubits:] * 2
        assert np.allclose(params.flatten(), weights.flatten())

        # gate wires
        wires = range(qubits)
        nm_wires = [wires[l : l + 2] for l in range(0, qubits - 1, 2)]
        nm_wires += [wires[l : l + 2] for l in range(1, qubits - 1, 2)]

        exp_wires = []
        for _ in range(layers):
            for i in range(qubits):
                exp_wires.append([wires[i]])
            for j in nm_wires:
                exp_wires.append(list(j))
                exp_wires.append(list(j[::-1]))
                exp_wires.append(list(j))

        res_wires = [queue[i].wires.tolist() for i in range(1, n_gates)]

        assert res_wires == exp_wires

    @pytest.mark.parametrize(
        ("init_state", "exp_state"),
        [
            (np.array([0, 0]), np.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j])),
            (
                np.array([0, 1]),
                np.array([0.0 + 0.0j, 0.862093 + 0.0j, 0.0 - 0.506749j, 0.0 + 0.0j]),
            ),
            (
                np.array([1, 0]),
                np.array([0.0 + 0.0j, 0.0 - 0.506749j, 0.862093 + 0.0j, 0.0 + 0.0j]),
            ),
            (np.array([1, 1]), np.array([0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j])),
        ],
    )
    def test_decomposition_u2ex(self, init_state, exp_state, tol):
        """Test the decomposition of the U_{2, ex}` exchange gate by asserting the prepared
        state."""

        N = 2
        wires = range(N)

        weight = 0.53141

        dev = qml.device("default.qubit", wires=N)

        @qml.qnode(dev)
        def circuit(weight):
            qml.BasisState(init_state, wires=wires)
            qml.particle_conserving_u2.u2_ex_gate(weight, wires)
            return qml.expval(qml.PauliZ(0))

        circuit(weight)

        assert np.allclose(circuit.device.state, exp_state, atol=tol)

    def test_custom_wire_labels(self, tol):
        """Test that template can deal with non-numeric, nonconsecutive wire labels."""
        weights = np.random.random(size=(1, 5))
        init_state = np.array([1, 1, 0])

        dev = qml.device("default.qubit", wires=3)
        dev2 = qml.device("default.qubit", wires=["z", "a", "k"])

        @qml.qnode(dev)
        def circuit():
            qml.ParticleConservingU2(weights, wires=range(3), init_state=init_state)
            return qml.expval(qml.Identity(0))

        @qml.qnode(dev2)
        def circuit2():
            qml.ParticleConservingU2(weights, wires=["z", "a", "k"], init_state=init_state)
            return qml.expval(qml.Identity("z"))

        circuit()
        circuit2()

        assert np.allclose(dev.state, dev2.state, atol=tol, rtol=0)


class TestInputs:
    """Test inputs and pre-processing."""

    @pytest.mark.parametrize(
        ("weights", "wires", "msg_match"),
        [
            (
                np.array([[-0.080, 2.629, -0.710, 5.383, 0.646, -2.872, -3.856]]),
                [0],
                "This template requires the number of qubits to be greater than one",
            ),
            (
                np.array([[-0.080, 2.629, -0.710, 5.383]]),
                [0, 1, 2, 3],
                "Weights tensor must",
            ),
            (
                np.array(
                    [
                        [-0.080, 2.629, -0.710, 5.383, 0.646, -2.872],
                        [-0.080, 2.629, -0.710, 5.383, 0.646, -2.872],
                    ]
                ),
                [0, 1, 2, 3],
                "Weights tensor must",
            ),
            (
                np.array([-0.080, 2.629, -0.710, 5.383, 0.646, -2.872]),
                [0, 1, 2, 3],
                "Weights tensor must be 2-dimensional",
            ),
        ],
    )
    def test_exceptions(self, weights, wires, msg_match):
        """Test that ParticleConservingU2 throws an exception if the parameters have illegal
        shapes, types or values."""
        N = len(wires)
        init_state = np.array([1, 1, 0, 0])

        dev = qml.device("default.qubit", wires=N)

        @qml.qnode(dev)
        def circuit():
            qml.ParticleConservingU2(
                weights=weights,
                wires=wires,
                init_state=init_state,
            )
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(ValueError, match=msg_match):
            circuit()

    def test_id(self):
        """Tests that the id attribute can be set."""
        init_state = np.array([1, 1, 0])
        template = qml.ParticleConservingU2(
            weights=np.random.random(size=(1, 5)), wires=range(3), init_state=init_state, id="a"
        )
        assert template.id == "a"


class TestAttributes:
    """Tests additional methods and attributes"""

    @pytest.mark.parametrize(
        "n_layers, n_wires, expected_shape",
        [
            (2, 3, (2, 5)),
            (2, 2, (2, 3)),
            (1, 3, (1, 5)),
        ],
    )
    def test_shape(self, n_layers, n_wires, expected_shape):
        """Test that the shape method returns the correct shape of the weights tensor"""

        shape = qml.ParticleConservingU2.shape(n_layers, n_wires)
        assert shape == expected_shape

    def test_shape_exception_not_enough_qubits(self):
        """Test that the shape function warns if there are not enough qubits."""

        with pytest.raises(ValueError, match="The number of qubits must be greater than one"):
            qml.ParticleConservingU2.shape(3, 1)


def circuit_template(weights):
    qml.ParticleConservingU2(weights, range(2), init_state=np.array([1, 1]))
    return qml.expval(qml.PauliZ(0))


def circuit_decomposed(weights):
    qml.BasisState(np.array([1, 1]), wires=[0, 1])
    qml.RZ(weights[0, 0], wires=[0])
    qml.RZ(weights[0, 1], wires=[1])
    qml.CNOT(wires=[0, 1])
    qml.CRX(weights[0, 2], wires=[1, 0])
    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.PauliZ(0))


class TestInterfaces:
    """Tests that the template is compatible with all interfaces, including the computation
    of gradients."""

    @pytest.mark.autograd
    def test_autograd(self, tol):
        """Tests the autograd interface."""

        weights = np.random.random(size=(1, 3))
        weights = pnp.array(weights, requires_grad=True)

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

        assert np.allclose(grads[0], grads2[0], atol=tol, rtol=0)

    @pytest.mark.jax
    def test_jax(self, tol):
        """Tests the jax interface."""

        import jax
        import jax.numpy as jnp

        weights = jnp.array(np.random.random(size=(1, 3)))

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

        weights = tf.Variable(np.random.random(size=(1, 3)))

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

        weights = torch.tensor(np.random.random(size=(1, 3)), requires_grad=True)

        dev = qml.device("default.qubit", wires=2)

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
