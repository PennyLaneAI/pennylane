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
import numpy as np
import pytest

import pennylane as qp
from pennylane import numpy as pnp
from pennylane.ops.functions.assert_valid import _test_decomposition_rule


@pytest.mark.jax
@pytest.mark.parametrize("init_state", [np.array([1, 1, 0, 0]), None])
def test_standard_validity(init_state):
    """Run standard checks with the assert_valid function."""

    layers = 2
    qubits = 4

    weights = np.random.normal(0, 2 * np.pi, (layers, 2 * qubits - 1))
    op = qp.ParticleConservingU2(weights, wires=range(qubits), init_state=init_state)
    qp.ops.functions.assert_valid(op)


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
            [qp.RZ] * qubits + ([qp.CNOT] + [qp.CRX] + [qp.CNOT]) * (qubits - 1)
        ) * layers

        op = qp.ParticleConservingU2(weights, wires=range(qubits), init_state=init_state)
        queue = op.decomposition()

        # number of gates
        assert len(queue) == n_gates

        # initialization
        assert isinstance(queue[0], qp.BasisEmbedding)

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

    @pytest.mark.usefixtures("enable_and_disable_graph_decomp")
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

        dev = qp.device("default.qubit", wires=N)

        @qp.qnode(dev)
        def circuit(weight):
            qp.BasisState(init_state, wires=wires)
            qp.particle_conserving_u2.u2_ex_gate(weight, wires)
            return qp.state()

        assert np.allclose(circuit(weight), exp_state, atol=tol)

    @pytest.mark.usefixtures("enable_and_disable_graph_decomp")
    def test_custom_wire_labels(self, tol):
        """Test that template can deal with non-numeric, nonconsecutive wire labels."""
        weights = np.random.random(size=(1, 5))
        init_state = np.array([1, 1, 0])

        dev = qp.device("default.qubit", wires=3)
        dev2 = qp.device("default.qubit", wires=["z", "a", "k"])

        @qp.qnode(dev)
        def circuit():
            qp.ParticleConservingU2(weights, wires=range(3), init_state=init_state)
            return qp.expval(qp.Identity(0)), qp.state()

        @qp.qnode(dev2)
        def circuit2():
            qp.ParticleConservingU2(weights, wires=["z", "a", "k"], init_state=init_state)
            return qp.expval(qp.Identity("z")), qp.state()

        res1, state1 = circuit()
        res2, state2 = circuit2()

        assert np.allclose(res1, res2, atol=tol, rtol=0)
        assert np.allclose(state1, state2, atol=tol, rtol=0)

    DECOMP_PARAMS = [
        # 4 layers, 2 qubits
        (
            [
                [-1.9106598048020482, -2.4252293336089434, -7.955118597565304],
                [2.634047947068791, -0.7841414002528393, 0.5580586682816082],
                [-2.5764787187334948, -1.7730285658921874, -1.587972235140794],
                [6.761177598132525, 13.538833429432247, -4.258978933877374],
            ],
            [0, 1],
            [0, 0],
        ),
        # 5 layers, 3 qubits
        (
            [
                [
                    0.3052380620823558,
                    -7.176878021732757,
                    1.8168695431818656,
                    4.0156753512334395,
                    2.472490764521911,
                ],
                [
                    12.946738593766135,
                    -2.2214352299175593,
                    -1.3386463376097575,
                    -2.3116469339577805,
                    3.254816983584464,
                ],
                [
                    3.6543666928558225,
                    -7.954364452146332,
                    4.430610147610802,
                    18.68178317908595,
                    -1.118756203873823,
                ],
                [
                    1.0023162483548982,
                    -11.392850933992039,
                    -0.12811288304772592,
                    2.9087161900770857,
                    2.798327081873103,
                ],
                [
                    -5.398400024254137,
                    -4.3375073879567285,
                    -5.225989835053272,
                    -14.645670306765235,
                    4.187589882604234,
                ],
            ],
            [0, 1, 2],
            [1, 0, 1],
        ),
        # 6 layers, 4 qubits
        (
            [
                [
                    -3.240336931367033,
                    0.486828936777422,
                    1.1408054343765788,
                    3.2995452541375814,
                    -14.384468470984158,
                    0.8377575410680149,
                    3.1007036923646667,
                ],
                [
                    -5.216060057078694,
                    -0.4326847534067299,
                    -2.1685328686500203,
                    0.7536263659502104,
                    -0.09798679822560973,
                    1.822844254814766,
                    7.816781332516206,
                ],
                [
                    -3.8421959401321146,
                    -6.555634909805163,
                    0.9848252156727638,
                    10.517001347723776,
                    4.380088731133282,
                    0.8294915715300569,
                    -2.4883370266135927,
                ],
                [
                    -0.5330004663315265,
                    -5.691354649220492,
                    4.0149034239175645,
                    12.066167350764042,
                    3.420698360621476,
                    -8.730871493925331,
                    -7.787177801409433,
                ],
                [
                    13.373053735184142,
                    -0.2012499751493434,
                    1.8504577491059968,
                    2.889564893963196,
                    -10.92857888908123,
                    -6.175299809001929,
                    6.447686216455785,
                ],
                [
                    -0.7662453160353478,
                    1.621556540147981,
                    -1.7019013453601364,
                    3.4266728305696517,
                    3.026254448320594,
                    -0.34087470179284,
                    -0.32663543530160666,
                ],
            ],
            [0, 1, 2, 3],
            [0, 0, 1, 1],
        ),
    ]

    @pytest.mark.capture
    @pytest.mark.parametrize(("weights", "wires", "init_state"), DECOMP_PARAMS)
    def test_decomposition_new(self, weights, wires, init_state):
        op = qp.ParticleConservingU2(weights, wires=wires, init_state=init_state)
        for rule in qp.list_decomps(qp.ParticleConservingU2):
            _test_decomposition_rule(op, rule)


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

        dev = qp.device("default.qubit", wires=N)

        @qp.qnode(dev)
        def circuit():
            qp.ParticleConservingU2(
                weights=weights,
                wires=wires,
                init_state=init_state,
            )
            return qp.expval(qp.PauliZ(0))

        with pytest.raises(ValueError, match=msg_match):
            circuit()

    def test_id(self):
        """Tests that the id attribute can be set."""
        init_state = np.array([1, 1, 0])
        template = qp.ParticleConservingU2(
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

        shape = qp.ParticleConservingU2.shape(n_layers, n_wires)
        assert shape == expected_shape

    def test_shape_exception_not_enough_qubits(self):
        """Test that the shape function warns if there are not enough qubits."""

        with pytest.raises(ValueError, match="The number of qubits must be greater than one"):
            qp.ParticleConservingU2.shape(3, 1)


def circuit_template(weights):
    qp.ParticleConservingU2(weights, range(2), init_state=np.array([1, 1]))
    return qp.expval(qp.PauliZ(0))


def circuit_decomposed(weights):
    qp.BasisState(np.array([1, 1]), wires=[0, 1])
    qp.RZ(weights[0, 0], wires=[0])
    qp.RZ(weights[0, 1], wires=[1])
    qp.CNOT(wires=[0, 1])
    qp.CRX(weights[0, 2], wires=[1, 0])
    qp.CNOT(wires=[0, 1])
    return qp.expval(qp.PauliZ(0))


class TestInterfaces:
    """Tests that the template is compatible with all interfaces, including the computation
    of gradients."""

    @pytest.mark.autograd
    def test_autograd(self, tol):
        """Tests the autograd interface."""

        weights = np.random.random(size=(1, 3))
        weights = pnp.array(weights, requires_grad=True)

        dev = qp.device("default.qubit", wires=2)

        circuit = qp.QNode(circuit_template, dev)
        circuit2 = qp.QNode(circuit_decomposed, dev)

        res = circuit(weights)
        res2 = circuit2(weights)
        assert qp.math.allclose(res, res2, atol=tol, rtol=0)

        grad_fn = qp.grad(circuit)
        grads = grad_fn(weights)

        grad_fn2 = qp.grad(circuit2)
        grads2 = grad_fn2(weights)

        assert np.allclose(grads[0], grads2[0], atol=tol, rtol=0)

    @pytest.mark.jax
    def test_jax(self, tol):
        """Tests the jax interface."""

        import jax
        import jax.numpy as jnp

        weights = jnp.array(np.random.random(size=(1, 3)))

        dev = qp.device("default.qubit", wires=2)

        circuit = qp.QNode(circuit_template, dev)
        circuit2 = qp.QNode(circuit_decomposed, dev)

        res = circuit(weights)
        res2 = circuit2(weights)
        assert qp.math.allclose(res, res2, atol=tol, rtol=0)

        grad_fn = jax.grad(circuit)
        grads = grad_fn(weights)

        grad_fn2 = jax.grad(circuit2)
        grads2 = grad_fn2(weights)

        assert np.allclose(grads[0], grads2[0], atol=tol, rtol=0)

    @pytest.mark.jax
    def test_jax_jit(self, tol):
        """Tests jit within the jax interface."""

        import jax
        import jax.numpy as jnp

        weights = jnp.array(np.random.random(size=(1, 3)))

        dev = qp.device("default.qubit", wires=2)

        circuit = qp.QNode(circuit_template, dev)
        circuit2 = jax.jit(circuit)

        res = circuit(weights)
        res2 = circuit2(weights)
        assert qp.math.allclose(res, res2, atol=tol, rtol=0)

        grad_fn = jax.grad(circuit)
        grads = grad_fn(weights)

        grad_fn2 = jax.grad(circuit2)
        grads2 = grad_fn2(weights)

        assert qp.math.allclose(grads, grads2, atol=tol, rtol=0)

    @pytest.mark.tf
    def test_tf(self, tol):
        """Tests the tf interface."""

        import tensorflow as tf

        weights = tf.Variable(np.random.random(size=(1, 3)))

        dev = qp.device("default.qubit", wires=2)

        circuit = qp.QNode(circuit_template, dev)
        circuit2 = qp.QNode(circuit_decomposed, dev)

        res = circuit(weights)
        res2 = circuit2(weights)
        assert qp.math.allclose(res, res2, atol=tol, rtol=0)

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

        dev = qp.device("default.qubit", wires=2)

        circuit = qp.QNode(circuit_template, dev)
        circuit2 = qp.QNode(circuit_decomposed, dev)

        res = circuit(weights)
        res2 = circuit2(weights)
        assert qp.math.allclose(res, res2, atol=tol, rtol=0)

        res = circuit(weights)
        res.backward()
        grads = [weights.grad]

        res2 = circuit2(weights)
        res2.backward()
        grads2 = [weights.grad]

        assert np.allclose(grads[0], grads2[0], atol=tol, rtol=0)
