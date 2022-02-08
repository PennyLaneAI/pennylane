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
Tests for the k-UpCCGSD template.
"""
from os import killpg
import pytest
import numpy as np
import pennylane as qml


class TestDecomposition:
    """Test that the template defines the correct decomposition."""

    @pytest.mark.parametrize(
        ("k", "delta_sz", "init_state", "wires"),
        [
            (
                1,
                0,
                qml.math.array([1, 1, 0, 0]),
                qml.math.array([0, 1, 2, 3]),
            ),
            (
                1,
                -1,
                qml.math.array([1, 1, 0, 0]),
                qml.math.array([0, 1, 2, 3]),
            ),
            (
                2,
                1,
                qml.math.array([1, 1, 0, 0]),
                qml.math.array([0, 1, 2, 3]),
            ),
            (
                2,
                0,
                qml.math.array([1, 1, 0, 0, 0, 0]),
                qml.math.array([0, 1, 2, 3, 4, 5]),
            ),
            (
                2,
                1,
                qml.math.array([1, 1, 0, 0, 0, 0, 0, 0]),
                qml.math.array([0, 1, 2, 3, 4, 5, 6, 7]),
            ),
        ],
    )
    def test_kupccgsd_operations(self, k, delta_sz, init_state, wires):
        """Test the correctness of the k-UpCCGSD template including the gate count
        and order, the wires the operation acts on and the correct use of parameters
        in the circuit."""

        # wires for generalized single excitation terms
        sz = np.array([0.5 if (i % 2 == 0) else -0.5 for i in range(len(wires))])
        gen_single_terms_wires = [
            wires[r : p + 1] if r < p else wires[p : r + 1][::-1]
            for r in range(len(wires))
            for p in range(len(wires))
            if sz[p] - sz[r] == delta_sz and p != r
        ]

        # wires for generalized pair coupled cluser double exictation terms
        pair_double_terms_wires = [
            [wires[r : r + 2], wires[p : p + 2]]
            for r in range(0, len(wires) - 1, 2)
            for p in range(0, len(wires) - 1, 2)
            if p != r
        ]

        n_excit_terms = len(gen_single_terms_wires) + len(pair_double_terms_wires)
        weights = np.random.normal(0, 2 * np.pi, (k, n_excit_terms))

        n_gates = 1 + n_excit_terms * k
        exp_unitary = [qml.FermionicDoubleExcitation] * len(pair_double_terms_wires)
        exp_unitary += [qml.FermionicSingleExcitation] * len(gen_single_terms_wires)

        op = qml.kUpCCGSD(weights, wires=wires, k=k, delta_sz=delta_sz, init_state=init_state)
        queue = op.expand().operations

        # number of gates
        assert len(queue) == n_gates

        # initialization
        assert isinstance(queue[0], qml.BasisEmbedding)

        # order of gates
        for op1, op2 in zip(queue[1:], exp_unitary):
            assert isinstance(op1, op2)

        # gate parameter
        params = np.zeros((k, n_excit_terms))
        for i in range(1, n_gates):
            gate_index = (i - 1) % n_excit_terms
            if gate_index < len(pair_double_terms_wires):
                gate_index += len(gen_single_terms_wires)
            else:
                gate_index -= len(pair_double_terms_wires)
            params[(i - 1) // n_excit_terms][gate_index] = queue[i].parameters[0]

        assert qml.math.allclose(params.flatten(), weights.flatten())

        # gate wires
        exp_wires = (
            [np.concatenate(w) for w in pair_double_terms_wires] + gen_single_terms_wires
        ) * k
        res_wires = [queue[i].wires.tolist() for i in range(1, n_gates)]
        for wires1, wires2 in zip(exp_wires, res_wires):
            assert np.all(wires1 == wires2)

    def test_custom_wire_labels(self, tol):
        """Test that template can deal with non-numeric, nonconsecutive wire labels."""

        weights = np.random.random(size=(1, 6))

        dev = qml.device("default.qubit", wires=4)
        dev2 = qml.device("default.qubit", wires=["z", "a", "k", "e"])

        @qml.qnode(dev)
        def circuit():
            qml.kUpCCGSD(
                weights,
                wires=range(4),
                k=1,
                delta_sz=0,
                init_state=np.array([0, 1, 0, 1]),
            )
            return qml.expval(qml.Identity(0))

        @qml.qnode(dev2)
        def circuit2():
            qml.kUpCCGSD(
                weights,
                wires=["z", "a", "k", "e"],
                k=1,
                delta_sz=0,
                init_state=np.array([0, 1, 0, 1]),
            )
            return qml.expval(qml.Identity("z"))

        circuit()
        circuit2()

        assert np.allclose(dev.state, dev2.state, atol=tol, rtol=0)

    @pytest.mark.parametrize(
        ("num_qubits", "k", "exp_state"),
        [
            (
                4,
                4,
                qml.math.array(
                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                ),
            ),
            (
                6,
                6,
                qml.math.array(
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.1077,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.686,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        -0.0429,
                        0.0,
                        0.0,
                        -0.0956,
                        0.0,
                        0.0,
                        0.2733,
                        0.0,
                        0.0,
                        -0.6089,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        -0.1777,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0708,
                        0.0,
                        0.0,
                        -0.1577,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ]
                ),
            ),
        ],
    )
    def test_k_layers_upccgsd(self, num_qubits, k, exp_state, tol):
        """Test that the k-UpCCGSD template with multiple layers works correctly asserting the prepared state."""

        wires = range(num_qubits)

        shape = qml.kUpCCGSD.shape(k=k, n_wires=num_qubits, delta_sz=0)
        weight = np.pi / 2 * qml.math.ones(shape)

        dev = qml.device("default.qubit", wires=wires)

        init_state = qml.math.array([1 if x < num_qubits // 2 else 0 for x in wires])

        @qml.qnode(dev)
        def circuit(weight):
            qml.kUpCCGSD(weight, wires=wires, k=k, delta_sz=0, init_state=init_state)
            return qml.state()

        circuit(weight)

        assert qml.math.allclose(circuit.device.state, exp_state, atol=tol)

    @pytest.mark.parametrize(
        ("wires", "delta_sz", "generalized_singles_wires", "generalized_pair_doubles_wires"),
        [
            (
                [0, 1, 2, 3],
                0,
                [[0, 1, 2], [1, 2, 3], [2, 1, 0], [3, 2, 1]],
                [[[0, 1], [2, 3]], [[2, 3], [0, 1]]],
            ),
            (
                [0, 1, 2, 3],
                1,
                [[1, 0], [1, 2], [3, 2, 1, 0], [3, 2]],
                [[[0, 1], [2, 3]], [[2, 3], [0, 1]]],
            ),
            (
                [0, 1, 2, 3, 4, 5],
                -1,
                [
                    [0, 1],
                    [0, 1, 2, 3],
                    [0, 1, 2, 3, 4, 5],
                    [2, 1],
                    [2, 3],
                    [2, 3, 4, 5],
                    [4, 3, 2, 1],
                    [4, 3],
                    [4, 5],
                ],
                [
                    [[0, 1], [2, 3]],
                    [[0, 1], [4, 5]],
                    [[2, 3], [0, 1]],
                    [[2, 3], [4, 5]],
                    [[4, 5], [0, 1]],
                    [[4, 5], [2, 3]],
                ],
            ),
            (
                ["a0", "b1", "c2", "d3", "e4", "f5"],
                1,
                [
                    ["b1", "a0"],
                    ["b1", "c2"],
                    ["b1", "c2", "d3", "e4"],
                    ["d3", "c2", "b1", "a0"],
                    ["d3", "c2"],
                    ["d3", "e4"],
                    ["f5", "e4", "d3", "c2", "b1", "a0"],
                    ["f5", "e4", "d3", "c2"],
                    ["f5", "e4"],
                ],
                [
                    [["a0", "b1"], ["c2", "d3"]],
                    [["a0", "b1"], ["e4", "f5"]],
                    [["c2", "d3"], ["a0", "b1"]],
                    [["c2", "d3"], ["e4", "f5"]],
                    [["e4", "f5"], ["a0", "b1"]],
                    [["e4", "f5"], ["c2", "d3"]],
                ],
            ),
        ],
    )
    def test_excitations_wires_kupccgsd(
        self, wires, delta_sz, generalized_singles_wires, generalized_pair_doubles_wires
    ):
        """Test the correctness of the wire indices for the generalized singles and paired doubles excitaitons
        used by the template."""

        shape = qml.kUpCCGSD.shape(k=1, n_wires=len(wires), delta_sz=delta_sz)
        weights = np.pi / 2 * qml.math.ones(shape)

        ref_state = qml.math.array([1, 1, 0, 0])

        op = qml.kUpCCGSD(weights, wires=wires, k=1, delta_sz=delta_sz, init_state=ref_state)
        gen_singles_wires, gen_doubles_wires = (
            op.hyperparameters["s_wires"],
            op.hyperparameters["d_wires"],
        )

        assert gen_singles_wires == generalized_singles_wires
        assert gen_doubles_wires == generalized_pair_doubles_wires


class TestInputs:
    """Test inputs and pre-processing."""

    @pytest.mark.parametrize(
        ("weights", "wires", "k", "delta_sz", "init_state", "msg_match"),
        [
            (
                np.array([[0.55, 0.72, 0.6, 0.54, 0.42, 0.65]]),
                [0, 1, 2],
                1,
                0,
                np.array([1, 1, 0, 0]),
                "Requires at least four wires",
            ),
            (
                np.array([[0.55, 0.72, 0.6, 0.54, 0.42, 0.65]]),
                [0, 1, 2, 3, 4],
                1,
                0,
                np.array([1, 1, 0, 0]),
                "Requires even number of wires",
            ),
            (
                np.array([[0.55, 0.72, 0.6, 0.54, 0.42, 0.65]]),
                [0, 1, 2, 3],
                0,
                0,
                np.array([1, 1, 0, 0]),
                "Requires k to be at least 1",
            ),
            (
                np.array([[0.55, 0.72, 0.6, 0.54, 0.42, 0.65]]),
                [0, 1, 2, 3],
                1,
                -2,
                np.array([1, 1, 0, 0]),
                "Requires delta_sz to be one of ±1 or 0",
            ),
            (
                np.array([-2.8, 1.6]),
                [0, 1, 2, 3],
                1,
                0,
                np.array([1, 1, 0, 0]),
                "Weights tensor must be of",
            ),
            (
                np.array([-2.8, 1.6]),
                [0, 1, 2, 3, 4, 5],
                2,
                -1,
                np.array([1, 1, 0, 0]),
                "Weights tensor must be of",
            ),
            (
                np.array([[0.55, 0.72, 0.6, 0.54, 0.42, 0.65]]),
                [0, 1, 2, 3],
                1,
                0,
                np.array([1.4, 1.3, 0.0, 0.0]),
                "Elements of 'init_state' must be integers",
            ),
        ],
    )
    def test_kupccgsd_exceptions(self, wires, weights, k, delta_sz, init_state, msg_match):
        """Test that k-UpCCGSD throws an exception if the parameters have illegal
        shapes, types or values."""

        dev = qml.device("default.qubit", wires=len(wires))

        @qml.qnode(dev)
        def circuit():
            qml.kUpCCGSD(
                weights=weights,
                wires=wires,
                k=k,
                delta_sz=delta_sz,
                init_state=init_state,
            )
            return qml.expval(qml.PauliZ(0))

        qnode = qml.QNode(circuit, dev)

        with pytest.raises(ValueError, match=msg_match):
            circuit()

    def test_id(self):
        """Test that the id attribute can be set."""
        template = qml.kUpCCGSD(
            qml.math.array([[0.55, 0.72, 0.6, 0.54, 0.42, 0.65]]),
            wires=range(4),
            k=1,
            delta_sz=0,
            init_state=qml.math.array([1, 1, 0, 0]),
            id="a",
        )
        assert template.id == "a"


class TestAttributes:
    """Test additional methods and attributes"""

    @pytest.mark.parametrize(
        "k, n_wires, delta_sz, expected_shape",
        [
            (2, 4, 0, (2, 6)),
            (2, 6, 0, (2, 18)),
            (2, 8, 0, (2, 36)),
            (2, 4, 1, (2, 6)),
            (2, 6, 1, (2, 15)),
            (2, 8, 1, (2, 28)),
        ],
    )
    def test_shape(self, k, n_wires, delta_sz, expected_shape):
        """Test that the shape method returns the correct shape of the weights tensor."""

        shape = qml.kUpCCGSD.shape(k, n_wires, delta_sz)
        assert shape == expected_shape

    def test_shape_exception_not_enough_qubits(self):
        """Test that the shape function warns if there are not enough qubits."""

        with pytest.raises(
            ValueError, match="This template requires the number of qubits to be greater than four"
        ):
            qml.kUpCCGSD.shape(k=2, n_wires=1, delta_sz=0)

    def test_shape_exception_not_even_qubits(self):
        """Test that the shape function warns if the number of qubits are not even."""

        with pytest.raises(ValueError, match="This template requires an even number of qubits"):
            qml.kUpCCGSD.shape(k=2, n_wires=5, delta_sz=0)


def circuit_template(weights):
    qml.kUpCCGSD(
        weights,
        wires=range(4),
        k=1,
        delta_sz=0,
        init_state=np.array([1, 1, 0, 0]),
    )
    return qml.expval(qml.PauliZ(0))


def circuit_decomposed(weights):
    qml.BasisState(np.array([0, 0, 1, 1]), wires=[0, 1, 2, 3])
    qml.FermionicDoubleExcitation(weights[0][4], wires1=[0, 1], wires2=[2, 3])
    qml.FermionicDoubleExcitation(weights[0][5], wires1=[2, 3], wires2=[0, 1])
    qml.FermionicSingleExcitation(weights[0][0], wires=[0, 1, 2])
    qml.FermionicSingleExcitation(weights[0][1], wires=[1, 2, 3])
    qml.FermionicSingleExcitation(weights[0][2], wires=[2, 1, 0])
    qml.FermionicSingleExcitation(weights[0][3], wires=[3, 2, 1])
    return qml.expval(qml.PauliZ(0))


class TestInterfaces:
    """Test that the template is compatible with all interfaces, including the computation
    of gradients."""

    def test_list_and_tuples(self, tol):
        """Test common iterables as inputs."""

        dev = qml.device("default.qubit", wires=4)

        circuit = qml.QNode(circuit_template, dev)
        circuit2 = qml.QNode(circuit_decomposed, dev)

        weights = [[0.55, 0.72, 0.6, 0.54, 0.42, 0.65]]
        res = circuit(weights)
        res2 = circuit2(weights)
        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

        weights_tuple = [((0.55, 0.72, 0.6, 0.54, 0.42, 0.65))]
        res = circuit(weights_tuple)
        res2 = circuit2(weights_tuple)
        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

    def test_autograd(self, tol):
        """Test the autograd interface."""

        weights = qml.numpy.random.random(size=(1, 6), requires_grad=True)

        dev = qml.device("default.qubit", wires=4)

        circuit = qml.QNode(circuit_template, dev, interface="autograd")
        circuit2 = qml.QNode(circuit_decomposed, dev, interface="autograd")

        res = circuit(weights)
        res2 = circuit2(weights)
        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

        grad_fn = qml.grad(circuit)
        grads = grad_fn(weights)

        grad_fn2 = qml.grad(circuit2)
        grads2 = grad_fn2(weights)

        assert np.allclose(grads, grads2, atol=tol, rtol=0)

    def test_jax(self, tol):
        """Test the jax interface."""

        jax = pytest.importorskip("jax")
        import jax.numpy as jnp

        weights = jnp.array(np.random.random(size=(1, 6)))

        dev = qml.device("default.qubit", wires=4)

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

    def test_tf(self, tol):
        """Test the tf interface."""

        tf = pytest.importorskip("tensorflow")

        weights = tf.Variable(np.random.random(size=(1, 6)))

        dev = qml.device("default.qubit", wires=4)

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

    def test_torch(self, tol):
        """Test the torch interface."""

        torch = pytest.importorskip("torch")

        weights = torch.tensor(np.random.random(size=(1, 6)), requires_grad=True)

        dev = qml.device("default.qubit", wires=4)

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
