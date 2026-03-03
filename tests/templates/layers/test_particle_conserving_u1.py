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
Unit tests for the ParticleConservingU1 template.
"""
import numpy as np
import pytest

import pennylane as qml
from pennylane import numpy as pnp
from pennylane.ops.functions.assert_valid import _test_decomposition_rule


@pytest.mark.jax
@pytest.mark.parametrize("init_state", [np.array([1, 1, 0]), None])
def test_standard_validity(init_state):
    """Check the operation using the assert_valid function."""

    weights = np.random.random(size=(1, 2, 2))
    op = qml.ParticleConservingU1(weights, wires=range(3), init_state=init_state)

    qml.ops.functions.assert_valid(op)


class TestDecomposition:
    """Tests that the template defines the correct decomposition."""

    @staticmethod
    def _wires_gates_u1(wires):
        """Auxiliary function giving the wires that the elementary gates decomposing
        ``u1_ex_gate`` act on."""

        exp_wires = [
            wires,
            wires,
            [wires[1]],
            wires,
            [wires[1]],
            wires,
            [wires[0]],
            wires[::-1],
            wires[::-1],
            wires,
            wires,
            [wires[1]],
            wires,
            [wires[1]],
            wires,
            [wires[0]],
        ]

        return exp_wires

    def test_operations(self):
        """Test the correctness of the ParticleConservingU1 template including the gate count
        and order, the wires each operation acts on and the correct use of parameters
        in the circuit."""

        qubits = 4
        layers = 2
        weights = np.random.random(size=(layers, qubits - 1, 2))

        gates_per_u1 = 16
        gates_per_layer = gates_per_u1 * (qubits - 1)
        gate_count = layers * gates_per_layer + 1

        gates_u1 = [
            qml.CZ,
            qml.CRot,
            qml.PhaseShift,
            qml.CNOT,
            qml.PhaseShift,
            qml.CNOT,
            qml.PhaseShift,
        ]
        gates_ent = gates_u1 + [qml.CZ, qml.CRot] + gates_u1

        wires = list(range(qubits))

        nm_wires = [wires[l : l + 2] for l in range(0, qubits - 1, 2)]
        nm_wires += [wires[l : l + 2] for l in range(1, qubits - 1, 2)]

        op = qml.ParticleConservingU1(weights, wires, init_state=np.array([1, 1, 0, 0]))
        queue = op.decomposition()

        assert gate_count == len(queue)

        # check initialization of the qubit register
        assert isinstance(queue[0], qml.BasisEmbedding)

        # check all quantum operations
        idx_CRot = 8
        for l in range(layers):
            for i in range(qubits - 1):
                exp_wires = self._wires_gates_u1(nm_wires[i])

                phi = weights[l, i, 0]
                theta = weights[l, i, 1]

                for j, exp_gate in enumerate(gates_ent):
                    idx = gates_per_layer * l + gates_per_u1 * i + j + 1

                    # check that the gates are applied in the right order
                    assert isinstance(queue[idx], exp_gate)

                    # check the wires the gates act on
                    assert queue[idx].wires.tolist() == exp_wires[j]

                    # check that parametrized gates take the parameters \phi and \theta properly
                    if exp_gate is qml.CRot:
                        if j < idx_CRot:
                            exp_params = [-phi, np.pi, phi]
                        elif j > idx_CRot:
                            exp_params = [phi, np.pi, -phi]
                        else:
                            exp_params = [0, 2 * theta, 0]

                        assert queue[idx].parameters == exp_params

                    elif exp_gate is qml.PhaseShift:
                        if j < idx_CRot:
                            exp_params = phi if j == idx_CRot / 2 else -phi
                        if j > idx_CRot:
                            exp_params = -phi if j == (3 * idx_CRot + 2) / 2 else phi

                        assert queue[idx].parameters == exp_params

    def test_custom_wire_labels(self, tol):
        """Test that template can deal with non-numeric, nonconsecutive wire labels."""
        weights = np.random.random(size=(1, 2, 2))
        init_state = np.array([1, 1, 0])

        dev = qml.device("default.qubit", wires=3)
        dev2 = qml.device("default.qubit", wires=["z", "a", "k"])

        @qml.qnode(dev)
        def circuit():
            qml.ParticleConservingU1(weights, wires=range(3), init_state=init_state)
            return qml.expval(qml.Identity(0)), qml.state()

        @qml.qnode(dev2)
        def circuit2():
            qml.ParticleConservingU1(weights, wires=["z", "a", "k"], init_state=init_state)
            return qml.expval(qml.Identity("z")), qml.state()

        res1, state1 = circuit()
        res2, state2 = circuit2()

        assert np.allclose(res1, res2, atol=tol, rtol=0)
        assert np.allclose(state1, state2, atol=tol, rtol=0)

    @pytest.mark.parametrize(
        ("init_state", "exp_state"),
        [
            (np.array([0, 0]), np.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j])),
            (
                np.array([1, 1]),
                np.array(
                    [
                        0.00000000e00 + 0.00000000e00j,
                        3.18686105e-17 - 1.38160066e-17j,
                        1.09332080e-16 - 2.26795885e-17j,
                        1.00000000e00 + 2.77555756e-17j,
                    ]
                ),
            ),
            (
                np.array([0, 1]),
                np.array(
                    [
                        0.00000000e00 + 0.00000000e00j,
                        8.23539739e-01 - 2.77555756e-17j,
                        -5.55434174e-01 - 1.15217954e-01j,
                        3.18686105e-17 + 1.38160066e-17j,
                    ]
                ),
            ),
            (
                np.array([1, 0]),
                np.array(
                    [
                        0.00000000e00 + 0.00000000e00j,
                        -5.55434174e-01 + 1.15217954e-01j,
                        -8.23539739e-01 - 5.55111512e-17j,
                        1.09332080e-16 + 2.26795885e-17j,
                    ]
                ),
            ),
        ],
    )
    def test_decomposition_u1ex(self, init_state, exp_state, tol):
        """Test the decomposition of the U_{1, ex}` exchange gate by asserting the prepared
        state."""

        N = 2
        wires = range(N)
        weights = np.array([[[0.2045368, -0.6031732]]])

        dev = qml.device("default.qubit", wires=N)

        @qml.qnode(dev)
        def circuit(weights):
            qml.ParticleConservingU1(weights, wires, init_state=init_state)
            return qml.state()

        assert np.allclose(circuit(weights), exp_state, atol=tol)

    DECOMP_PARAMS = [
        # 4 layers, 2 qubits
        (
            [
                [[0.11017280164958265, 0.4698218184451487]],
                [[0.7135672121068317, 0.0272660664063894]],
                [[0.5157427097550098, 0.10272892526263466]],
                [[0.9239814091830049, 0.8808136909883172]],
            ],
            [0, 1],
            [0, 0],
        ),
        # 5 layers, 3 qubits
        (
            [
                [
                    [0.30996116164608845, 0.7762565711230485],
                    [0.29871765303474085, 0.24416302786429345],
                ],
                [
                    [0.17436018243610651, 0.2734732726492589],
                    [0.545084231833944, 0.3152339642958135],
                ],
                [
                    [0.5155025591256616, 0.010752411646314797],
                    [0.01299807042587997, 0.28647607092144867],
                ],
                [
                    [0.24446696573419435, 0.3000904265834745],
                    [0.3765987980718698, 0.14682646358276497],
                ],
                [
                    [0.40048766331999774, 0.8765133487227904],
                    [0.4958676000527388, 0.9877614437193426],
                ],
            ],
            [0, 1, 2],
            [1, 0, 1],
        ),
        # 6 layers, 4 qubits
        (
            [
                [
                    [0.26912988693106343, 0.7471399517794667],
                    [0.45243585979670253, 0.4841394533928278],
                    [0.20547086596211728, 0.24306963037339213],
                ],
                [
                    [0.7109658281610554, 0.16071657656855232],
                    [0.8463217509168768, 0.04811307565512346],
                    [0.3028182641489623, 0.130379080742683],
                ],
                [
                    [0.100436673785685, 0.02729620001262867],
                    [0.012189820928570572, 0.9408691015077094],
                    [0.8649719443970864, 0.16911101452497068],
                ],
                [
                    [0.8848486243995226, 0.20487678689411004],
                    [0.7884149466080009, 0.10983556912458681],
                    [0.2273674130906974, 0.07707995281244895],
                ],
                [
                    [0.49691776367871987, 0.15122009425990723],
                    [0.164885189391221, 0.7987235260782912],
                    [0.7153276123829537, 0.6469720977785561],
                ],
                [
                    [0.3924863551874419, 0.9532711994451982],
                    [0.4074478651463257, 0.4393058861225143],
                    [0.3135000703614109, 0.5396793760954518],
                ],
            ],
            [0, 1, 2, 3],
            [0, 0, 1, 1],
        ),
    ]

    @pytest.mark.capture
    @pytest.mark.parametrize(("weights", "wires", "init_state"), DECOMP_PARAMS)
    def test_decomposition_new(self, weights, wires, init_state):
        op = qml.ParticleConservingU1(weights, wires=wires, init_state=init_state)
        for rule in qml.list_decomps(qml.ParticleConservingU1):
            _test_decomposition_rule(op, rule)


class TestInputs:
    """Test inputs and pre-processing."""

    @pytest.mark.parametrize(
        ("weights", "n_wires", "msg_match"),
        [
            (np.ones((4, 3)), 4, "Weights tensor must"),
            (np.ones((4, 2, 2)), 4, "Weights tensor must"),
            (np.ones((4, 3, 1)), 4, "Weights tensor must"),
            (
                np.ones((4, 3, 1)),
                1,
                "Expected the number of qubits",
            ),
        ],
    )
    def test_exceptions(self, weights, n_wires, msg_match):
        """Test that ParticleConservingU1 throws an exception if the parameter array has an illegal
        shape."""

        wires = range(n_wires)
        init_state = np.array([1, 1, 0, 0])
        dev = qml.device("default.qubit", wires=n_wires)

        @qml.qnode(dev)
        def circuit():
            qml.ParticleConservingU1(weights=weights, wires=wires, init_state=init_state)
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(ValueError, match=msg_match):
            circuit()

    @pytest.mark.usefixtures("ignore_id_deprecation")
    def test_id(self):
        """Tests that the id attribute can be set."""
        template = qml.ParticleConservingU1(
            weights=np.array([[[0.2, -0.6]]]), wires=range(2), init_state=np.array([1, 1]), id="a"
        )
        assert template.id == "a"


class TestAttributes:
    """Tests additional methods and attributes"""

    @pytest.mark.parametrize(
        "n_layers, n_wires, expected_shape",
        [
            (2, 3, (2, 2, 2)),
            (2, 2, (2, 1, 2)),
        ],
    )
    def test_shape(self, n_layers, n_wires, expected_shape):
        """Test that the shape method returns the correct shape of the weights tensor"""

        shape = qml.ParticleConservingU1.shape(n_layers, n_wires)
        assert shape == expected_shape

    def test_shape_exception_not_enough_qubits(self):
        """Test that the shape function warns if there are not enough qubits."""

        with pytest.raises(ValueError, match="The number of qubits must be greater than one"):
            qml.ParticleConservingU1.shape(3, 1)


def circuit_template(weights):
    qml.ParticleConservingU1(weights, range(2), init_state=np.array([1, 1]))
    return qml.expval(qml.PauliZ(0))


def circuit_decomposed(weights):
    qml.BasisState(np.array([1, 1]), wires=[0, 1])
    qml.CZ(wires=[0, 1])
    qml.CRot(weights[0, 0, 0], np.pi, weights[0, 0, 0], wires=[0, 1])
    qml.PhaseShift(-weights[0, 0, 0], wires=[1])
    qml.CNOT(wires=[0, 1])
    qml.PhaseShift(weights[0, 0, 0], wires=[1])
    qml.CNOT(wires=[0, 1])
    qml.PhaseShift(-weights[0, 0, 0], wires=[0])
    qml.CZ(wires=[1, 0])
    qml.CRot(0, weights[0, 0, 1], 0, wires=[1, 0])
    qml.CZ(wires=[0, 1])
    qml.CRot(weights[0, 0, 0], np.pi, -weights[0, 0, 0], wires=[0, 1])
    qml.PhaseShift(weights[0, 0, 0], wires=[1])
    qml.CNOT(wires=[0, 1])
    qml.PhaseShift(-weights[0, 0, 0], wires=[1])
    qml.CNOT(wires=[0, 1])
    qml.PhaseShift(weights[0, 0, 0], wires=[0])

    return qml.expval(qml.PauliZ(0))


class TestInterfaces:
    """Tests that the template is compatible with all interfaces, including the computation
    of gradients."""

    @pytest.mark.autograd
    def test_autograd(self, tol):
        """Tests the autograd interface."""

        weights = np.random.random(size=(1, 1, 2))
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

        weights = jnp.array(np.random.random(size=(1, 1, 2)))

        dev = qml.device("default.qubit", wires=2)

        circuit = qml.QNode(circuit_template, dev)
        circuit2 = qml.QNode(circuit_decomposed, dev)

        res = circuit(weights)
        res2 = circuit2(weights)
        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

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

        weights = jnp.array(np.random.random(size=(1, 1, 2)))

        dev = qml.device("default.qubit", wires=2)

        circuit = qml.QNode(circuit_template, dev)
        circuit2 = jax.jit(circuit)

        res = circuit(weights)
        res2 = circuit2(weights)
        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

        grad_fn = jax.grad(circuit)
        grads = grad_fn(weights)

        grad_fn2 = jax.grad(circuit2)
        grads2 = grad_fn2(weights)

        assert qml.math.allclose(grads, grads2, atol=tol, rtol=0)

    @pytest.mark.tf
    def test_tf(self, tol):
        """Tests the tf interface."""

        import tensorflow as tf

        weights = tf.Variable(np.random.random(size=(1, 1, 2)))

        dev = qml.device("default.qubit", wires=2)

        circuit = qml.QNode(circuit_template, dev)
        circuit2 = qml.QNode(circuit_decomposed, dev)

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

        weights = torch.tensor(np.random.random(size=(1, 1, 2)), requires_grad=True)

        dev = qml.device("default.qubit", wires=2)

        circuit = qml.QNode(circuit_template, dev)
        circuit2 = qml.QNode(circuit_decomposed, dev)

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
