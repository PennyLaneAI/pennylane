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
Tests for the UCCSD template.
"""
import numpy as np

# pylint: disable=protected-access,too-many-arguments
import pytest

import pennylane as qml
from pennylane import numpy as pnp
from pennylane.ops.functions.assert_valid import _test_decomposition_rule

test_data_decomposition = [
    (
        [[0, 1, 2]],
        [],
        np.array([3.815]),
        1,
        [
            [0, qml.BasisState, [0, 1, 2, 3, 4, 5], [np.array([1, 1, 0, 0, 0, 0])]],
            [1, qml.RX, [0], [-np.pi / 2]],
            [5, qml.RZ, [2], [1.9075]],
            [6, qml.CNOT, [1, 2], []],
        ],
    ),
    (
        [[0, 1, 2], [1, 2, 3]],
        [],
        np.array([3.815, 4.866]),
        1,
        [
            [2, qml.Hadamard, [2], []],
            [8, qml.RX, [0], [np.pi / 2]],
            [12, qml.CNOT, [0, 1], []],
            [23, qml.RZ, [3], [2.433]],
            [24, qml.CNOT, [2, 3], []],
            [26, qml.RX, [1], [np.pi / 2]],
        ],
    ),
    (
        [],
        [[[0, 1], [2, 3, 4, 5]]],
        np.array([3.815]),
        1,
        [
            [3, qml.RX, [2], [-np.pi / 2]],
            [29, qml.RZ, [5], [0.476875]],
            [73, qml.Hadamard, [0], []],
            [150, qml.RX, [1], [np.pi / 2]],
            [88, qml.CNOT, [3, 4], []],
            [121, qml.CNOT, [2, 3], []],
        ],
    ),
    (
        [],
        [[[0, 1], [2, 3]], [[0, 1], [4, 5]]],
        np.array([3.815, 4.866]),
        1,
        [
            [4, qml.Hadamard, [3], []],
            [16, qml.RX, [0], [-np.pi / 2]],
            [38, qml.RZ, [3], [0.476875]],
            [78, qml.Hadamard, [2], []],
            [107, qml.RX, [1], [-np.pi / 2]],
            [209, qml.Hadamard, [4], []],
            [218, qml.RZ, [5], [-0.60825]],
            [82, qml.CNOT, [2, 3], []],
            [159, qml.CNOT, [4, 5], []],
        ],
    ),
    (
        [[0, 1, 2, 3, 4], [1, 2, 3]],
        [[[0, 1], [2, 3]], [[0, 1], [4, 5]]],
        np.array([3.815, 4.866, 1.019, 0.639]),
        1,
        [
            [16, qml.RX, [0], [-np.pi / 2]],
            [47, qml.Hadamard, [1], []],
            [74, qml.Hadamard, [2], []],
            [83, qml.RZ, [3], [-0.127375]],
            [134, qml.RX, [4], [np.pi / 2]],
            [158, qml.RZ, [5], [0.079875]],
            [188, qml.RZ, [5], [-0.079875]],
            [96, qml.CNOT, [1, 2], []],
            [235, qml.CNOT, [1, 4], []],
        ],
    ),
    (
        [[0, 1, 2]],
        [],
        np.array([[3.815]]),
        1,
        [
            [0, qml.BasisState, [0, 1, 2, 3, 4, 5], [np.array([1, 1, 0, 0, 0, 0])]],
            [1, qml.RX, [0], [-np.pi / 2]],
            [5, qml.RZ, [2], [1.9075]],
            [6, qml.CNOT, [1, 2], []],
        ],
    ),
    (
        [[0, 1, 2]],
        [],
        np.array([[3.815], [1.019]]),
        2,
        [
            [0, qml.BasisState, [0, 1, 2, 3, 4, 5], [np.array([1, 1, 0, 0, 0, 0])]],
            [1, qml.RX, [0], [-np.pi / 2]],
            [5, qml.RZ, [2], [1.9075]],
            [6, qml.CNOT, [1, 2], []],
            [19, qml.RX, [0], [-np.pi / 2]],
            [23, qml.RZ, [2], [0.5095]],
            [24, qml.CNOT, [1, 2], []],
        ],
    ),
]

test_data_decomposition_new = [
    ([[0, 1, 2]], [], np.array([3.815]), 1),
    (
        [[0, 1, 2], [1, 2, 3]],
        [],
        np.array([3.815, 4.866]),
        1,
    ),
    (
        [],
        [[[0, 1], [2, 3]]],
        np.array([3.815]),
        1,
    ),
    (
        [],
        [[[0, 1], [2, 3]], [[0, 1], [4, 5]]],
        np.array([3.815, 4.866]),
        1,
    ),
    (
        [[2, 3, 4], [1, 2, 3]],
        [[[0, 1], [2, 3]], [[0, 1], [4, 5]]],
        np.array([3.815, 4.866, 1.019, 0.639]),
        1,
    ),
    (
        [[0, 1, 2]],
        [],
        np.array([[3.815], [1.019]]),
        2,
    ),
]


@pytest.mark.jax
@pytest.mark.parametrize("s_wires, d_wires, weights, n_repeats, _", test_data_decomposition)
def test_standard_validity(s_wires, d_wires, weights, n_repeats, _):
    """Test standard validity criteria using assert_valid."""
    cnots = 0
    for s_wires_ in s_wires:
        cnots += 4 * (len(s_wires_) - 1)

    for d_wires_ in d_wires:
        cnots += 16 * (len(d_wires_[0]) - 1 + len(d_wires_[1]) - 1 + 1)

    cnots *= n_repeats

    N = 6
    wires = range(N)

    ref_state = np.array([1, 1, 0, 0, 0, 0])

    op = qml.UCCSD(
        weights,
        wires,
        s_wires=s_wires,
        d_wires=d_wires,
        init_state=ref_state,
        n_repeats=n_repeats,
    )
    qml.ops.functions.assert_valid(op)


class TestDecomposition:
    """Tests that the template defines the correct decomposition."""

    @pytest.mark.parametrize(
        "s_wires, d_wires, weights, n_repeats, ref_gates", test_data_decomposition
    )
    def test_uccsd_operations(self, s_wires, d_wires, weights, n_repeats, ref_gates):
        """Test the correctness of the UCCSD template including the gate count
        and order, the wires the operation acts on and the correct use of parameters
        in the circuit."""

        sqg = (10 * len(s_wires) + 72 * len(d_wires)) * n_repeats

        cnots = 0
        for s_wires_ in s_wires:
            cnots += 4 * (len(s_wires_) - 1)

        for d_wires_ in d_wires:
            cnots += 16 * (len(d_wires_[0]) - 1 + len(d_wires_[1]) - 1 + 1)

        cnots *= n_repeats

        N = 6
        wires = range(N)

        ref_state = np.array([1, 1, 0, 0, 0, 0])

        op = qml.UCCSD(
            weights,
            wires,
            s_wires=s_wires,
            d_wires=d_wires,
            init_state=ref_state,
            n_repeats=n_repeats,
        )
        raw_queue = op.decomposition()

        # hack to avoid updating the test data:
        # expand the other templates, which now
        # queue as a single operation
        queue = []
        for op in raw_queue:
            if op.name in ["FermionicSingleExcitation", "FermionicDoubleExcitation"]:
                queue.extend(op.decomposition())
            else:
                queue.append(op)

        assert len(queue) == sqg + cnots + 1

        for gate in ref_gates:
            idx = gate[0]

            exp_gate = gate[1]
            res_gate = queue[idx]
            assert isinstance(res_gate, exp_gate)

            exp_wires = gate[2]
            res_wires = queue[idx].wires
            assert res_wires.tolist() == exp_wires

            exp_weight = gate[3]
            res_weight = queue[idx].parameters
            assert np.allclose(res_weight, exp_weight)

    @pytest.mark.parametrize("s_wires, d_wires, weights, n_repeats", test_data_decomposition_new)
    # Note: UCCSD is not capture compatible
    def test_decomposition_new(
        self, s_wires, d_wires, weights, n_repeats
    ):  # pylint: disable=unused-argument
        """Tests the decomposition rule implemented with the new system."""
        op = qml.UCCSD(
            weights,
            wires=range(6),
            s_wires=s_wires,
            d_wires=d_wires,
            init_state=np.array([1, 1, 0, 0, 0, 0]),
            n_repeats=n_repeats,
        )
        for rule in qml.list_decomps(qml.UCCSD):
            _test_decomposition_rule(op, rule)

    def test_custom_wire_labels(self, tol):
        """Test that template can deal with non-numeric, nonconsecutive wire labels."""

        weights = [0.1, 0.2]
        dev = qml.device("default.qubit", wires=4)
        dev2 = qml.device("default.qubit", wires=["z", "a", "k", "e"])

        @qml.qnode(dev)
        def circuit():
            qml.UCCSD(
                weights,
                wires=range(4),
                s_wires=[[0, 1]],
                d_wires=[[[0, 1], [2, 3]]],
                init_state=np.array([0, 1, 0, 1]),
            )
            return qml.expval(qml.Identity(0)), qml.state()

        @qml.qnode(dev2)
        def circuit2():
            qml.UCCSD(
                weights,
                wires=["z", "a", "k", "e"],
                s_wires=[["z", "a"]],
                d_wires=[[["z", "a"], ["k", "e"]]],
                init_state=np.array([0, 1, 0, 1]),
            )
            return qml.expval(qml.Identity("z")), qml.state()

        res1, state1 = circuit()
        res2, state2 = circuit2()

        assert np.allclose(res1, res2, atol=tol, rtol=0)
        assert np.allclose(state1, state2, atol=tol, rtol=0)


class TestInputs:
    """Test inputs and pre-processing."""

    @pytest.mark.parametrize(
        ("weights", "s_wires", "d_wires", "init_state", "n_repeats", "msg_match"),
        [
            (
                np.array([-2.8]),
                [[0, 1, 2]],
                [],
                np.array([1.2, 1, 0, 0]),
                1,
                "Elements of 'init_state' must be integers",
            ),
            (
                np.array([-2.8]),
                [],
                [],
                np.array([1, 1, 0, 0]),
                1,
                "s_wires and d_wires lists can not be both empty",
            ),
            (
                np.array([-2.8]),
                [],
                [[[0, 1, 2, 3]]],
                np.array([1, 1, 0, 0]),
                1,
                "expected entries of d_wires to be of size 2",
            ),
            (
                np.array([-2.8]),
                [[0, 2]],
                [],
                np.array([1, 1, 0, 0, 0]),
                1,
                "State must be of length 4",
            ),
            (
                np.array([-2.8, 1.6]),
                [[0, 1, 2]],
                [],
                np.array([1, 1, 0, 0]),
                1,
                "For one-dimensional weights tensor",
            ),
            (
                np.array([-2.8, 1.6]),
                [],
                [[[0, 1], [2, 3]]],
                np.array([1, 1, 0, 0]),
                1,
                "For one-dimensional weights tensor",
            ),
            (
                np.array([-2.8, 1.6]),
                [[0, 1, 2], [1, 2, 3]],
                [[[0, 1], [2, 3]]],
                np.array([1, 1, 0, 0]),
                1,
                "For one-dimensional weights tensor",
            ),
            (
                np.array([-2.8]),
                [[0, 1, 2]],
                [],
                np.array([1, 1, 0, 0]),
                0,
                "Requires n_repeats to be at least 1",
            ),
            (
                np.array([-2.8]),
                [[0, 1, 2]],
                [],
                np.array([1, 1, 0, 0]),
                -1,
                "Requires n_repeats to be at least 1",
            ),
            (
                np.array([-2.8]),
                [[0, 1, 2]],
                [],
                np.array([1, 1, 0, 0]),
                2,
                "For one-dimensional weights tensor",
            ),
            (
                np.array([[-2.8], [-1.8]]),
                [[0, 1, 2]],
                [],
                np.array([1, 1, 0, 0]),
                3,
                "Weights tensor must be of",
            ),
        ],
    )
    def test_uccsd_xceptions(self, weights, s_wires, d_wires, init_state, n_repeats, msg_match):
        """Test that UCCSD throws an exception if the parameters have illegal
        shapes, types or values."""
        N = 4
        wires = range(4)
        dev = qml.device("default.qubit", wires=N)

        def circuit(
            weights=weights,
            wires=wires,
            s_wires=s_wires,
            d_wires=d_wires,
            init_state=init_state,
            n_repeats=n_repeats,
        ):
            qml.UCCSD(
                weights=weights,
                wires=wires,
                s_wires=s_wires,
                d_wires=d_wires,
                init_state=init_state,
                n_repeats=n_repeats,
            )
            return qml.expval(qml.PauliZ(0))

        qnode = qml.QNode(circuit, dev)

        with pytest.raises(ValueError, match=msg_match):
            qnode(
                weights=weights,
                wires=wires,
                s_wires=s_wires,
                d_wires=d_wires,
                init_state=init_state,
                n_repeats=n_repeats,
            )

    def test_id(self):
        """Tests that the id attribute can be set."""
        template = qml.UCCSD(
            [0.1, 0.2],
            wires=range(4),
            s_wires=[[0, 1]],
            d_wires=[[[0, 1], [2, 3]]],
            init_state=np.array([0, 1, 0, 1]),
            id="a",
        )
        assert template.id == "a"


def circuit_template(weights, n_repeats=1):
    qml.UCCSD(
        weights,
        wires=range(4),
        s_wires=[[0, 1]],
        d_wires=[[[0, 1], [2, 3]]],
        init_state=np.array([1, 0, 0, 0]),
        n_repeats=n_repeats,
    )
    return qml.expval(qml.PauliZ(0))


def circuit_decomposed(weights, n_repeats=1):
    qml.BasisState(np.array([1, 0, 0, 0]), wires=range(4))

    if n_repeats == 1 and len(qml.math.shape(weights)) == 1:
        weights = qml.math.expand_dims(weights, 0)

    for layer in range(n_repeats):
        qml.FermionicDoubleExcitation(weights[layer][1], wires1=[0, 1], wires2=[2, 3])
        qml.FermionicSingleExcitation(weights[layer][0], wires=[0, 1])
    return qml.expval(qml.PauliZ(0))


class TestInterfaces:
    """Tests that the template is compatible with all interfaces, including the computation
    of gradients."""

    def test_list_and_tuples(self, tol):
        """Tests common iterables as inputs."""

        weights = list(range(2))

        dev = qml.device("default.qubit", wires=4)

        circuit = qml.QNode(circuit_template, dev)
        circuit2 = qml.QNode(circuit_decomposed, dev)

        res = circuit(weights)
        res2 = circuit2(weights)
        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

        # Test with n_repeats=2
        weights = [list(range(2)), list(range(2))]
        res = circuit(weights, n_repeats=2)
        res2 = circuit2(weights, n_repeats=2)
        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

    @pytest.mark.autograd
    def test_autograd(self, tol):
        """Tests the autograd interface."""

        weights = pnp.array(np.random.random(size=(2,)), requires_grad=True)

        dev = qml.device("default.qubit", wires=4)

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

        # Test with n_repeats=2
        weights = pnp.array(
            np.random.random(
                size=(
                    2,
                    2,
                )
            ),
            requires_grad=True,
        )
        res = circuit(weights, n_repeats=2)
        res2 = circuit2(weights, n_repeats=2)
        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

        grad_fn = qml.grad(circuit)
        grads = grad_fn(weights, n_repeats=2)

        grad_fn2 = qml.grad(circuit2)
        grads2 = grad_fn2(weights, n_repeats=2)

        assert np.allclose(grads, grads2, atol=tol, rtol=0)

    @pytest.mark.jax
    @pytest.mark.slow
    def test_jax(self, tol):
        """Tests the jax interface."""

        import jax
        import jax.numpy as jnp

        weights = jnp.array(np.random.random(size=(2,)))
        dev = qml.device("default.qubit", wires=4)

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

        # Test with n_repeats=2
        weights = jnp.array(
            np.random.random(
                size=(
                    2,
                    2,
                )
            )
        )
        res = circuit(weights, n_repeats=2)
        res2 = circuit2(weights, n_repeats=2)
        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

        grad_fn = jax.grad(circuit)
        grads = grad_fn(weights, n_repeats=2)

        grad_fn2 = jax.grad(circuit2)
        grads2 = grad_fn2(weights, n_repeats=2)

        assert np.allclose(grads[0], grads2[0], atol=tol, rtol=0)

    @pytest.mark.jax
    @pytest.mark.slow
    def test_jax_jit(self, tol):
        """Tests the jax interface."""

        import jax
        import jax.numpy as jnp

        weights = jnp.array(np.random.random(size=(2,)))
        dev = qml.device("default.qubit", wires=4)

        circuit = qml.QNode(circuit_template, dev)
        circuit2 = jax.jit(circuit, static_argnames="n_repeats")

        res = circuit(weights)
        res2 = circuit2(weights)
        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

        grad_fn = jax.grad(circuit)
        grads = grad_fn(weights)

        grad_fn2 = jax.grad(circuit2)
        grads2 = grad_fn2(weights)

        assert qml.math.allclose(grads, grads2, atol=tol, rtol=0)

        # Test with n_repeats=2
        weights = jnp.array(np.random.random(size=(2, 2)))
        res = circuit(weights, n_repeats=2)
        res2 = circuit2(weights, n_repeats=2)
        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

        grad_fn = jax.grad(circuit)
        grads = grad_fn(weights, n_repeats=2)

        grad_fn2 = jax.grad(circuit2)
        grads2 = grad_fn2(weights, n_repeats=2)

        assert np.allclose(grads, grads2, atol=tol, rtol=0)

    @pytest.mark.tf
    def test_tf(self, tol):
        """Tests the tf interface."""

        import tensorflow as tf

        weights = tf.Variable(np.random.random(size=(2,)))
        dev = qml.device("default.qubit", wires=4)

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

        # Test with n_repeats=2
        weights = tf.Variable(
            np.random.random(
                size=(
                    2,
                    2,
                )
            )
        )
        res = circuit(weights, n_repeats=2)
        res2 = circuit2(weights, n_repeats=2)
        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

        with tf.GradientTape() as tape:
            res = circuit(weights, n_repeats=2)
        grads = tape.gradient(res, [weights])

        with tf.GradientTape() as tape2:
            res2 = circuit2(weights, n_repeats=2)
        grads2 = tape2.gradient(res2, [weights])

        assert np.allclose(grads[0], grads2[0], atol=tol, rtol=0)

    @pytest.mark.torch
    def test_torch(self, tol):
        """Tests the torch interface."""

        import torch

        weights = torch.tensor(np.random.random(size=(2,)), requires_grad=True)

        dev = qml.device("default.qubit", wires=4)

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

        # Test with n_repeats=2
        weights = torch.tensor(
            np.random.random(
                size=(
                    2,
                    2,
                )
            ),
            requires_grad=True,
        )
        circuit = qml.QNode(circuit_template, dev)
        circuit2 = qml.QNode(circuit_decomposed, dev)

        res = circuit(weights, n_repeats=2)
        res2 = circuit2(weights, n_repeats=2)
        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

        res = circuit(weights, n_repeats=2)
        res.backward()
        grads = [weights.grad]

        res2 = circuit2(weights, n_repeats=2)
        res2.backward()
        grads2 = [weights.grad]

        assert np.allclose(grads[0], grads2[0], atol=tol, rtol=0)
