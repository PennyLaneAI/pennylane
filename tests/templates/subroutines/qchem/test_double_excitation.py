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
Tests for the FermionicDoubleExcitation template.
"""
import numpy as np
import pytest

import pennylane as qp
from pennylane import numpy as pnp
from pennylane.ops.functions.assert_valid import _test_decomposition_rule


@pytest.mark.jax
def test_standard_validity():
    """Run standard tests of operation validity."""
    weight = 0.5
    wires1 = qp.wires.Wires((0, 1))
    wires2 = qp.wires.Wires((2, 3, 4))
    op = qp.FermionicDoubleExcitation(weight, wires1=wires1, wires2=wires2)
    qp.ops.functions.assert_valid(op)


class TestDecomposition:
    """Tests that the template defines the correct decomposition."""

    @pytest.mark.parametrize(
        ("wires1", "wires2", "ref_gates"),
        [
            (
                [0, 1, 2],
                [4, 5, 6],
                [
                    [0, qp.Hadamard, [0], []],
                    [1, qp.Hadamard, [2], []],
                    [2, qp.RX, [4], [-np.pi / 2]],
                    [3, qp.Hadamard, [6], []],
                    [9, qp.RZ, [6], [np.pi / 24]],
                    [15, qp.Hadamard, [0], []],
                    [16, qp.Hadamard, [2], []],
                    [17, qp.RX, [4], [np.pi / 2]],
                    [18, qp.Hadamard, [6], []],
                ],
            ),
            (
                [0, 1],
                [4, 5],
                [
                    [15, qp.RX, [0], [-np.pi / 2]],
                    [16, qp.Hadamard, [1], []],
                    [17, qp.RX, [4], [-np.pi / 2]],
                    [18, qp.RX, [5], [-np.pi / 2]],
                    [22, qp.RZ, [5], [np.pi / 24]],
                    [26, qp.RX, [0], [np.pi / 2]],
                    [27, qp.Hadamard, [1], []],
                    [28, qp.RX, [4], [np.pi / 2]],
                    [29, qp.RX, [5], [np.pi / 2]],
                ],
            ),
            (
                [1, 2, 3],
                [7, 8, 9, 10, 11],
                [
                    [46, qp.Hadamard, [1], []],
                    [47, qp.RX, [3], [-np.pi / 2]],
                    [48, qp.RX, [7], [-np.pi / 2]],
                    [49, qp.RX, [11], [-np.pi / 2]],
                    [57, qp.RZ, [11], [np.pi / 24]],
                    [65, qp.Hadamard, [1], []],
                    [66, qp.RX, [3], [np.pi / 2]],
                    [67, qp.RX, [7], [np.pi / 2]],
                    [68, qp.RX, [11], [np.pi / 2]],
                ],
            ),
            (
                [2, 3, 4],
                [8, 9, 10],
                [
                    [57, qp.Hadamard, [2], []],
                    [58, qp.Hadamard, [4], []],
                    [59, qp.Hadamard, [8], []],
                    [60, qp.RX, [10], [-np.pi / 2]],
                    [66, qp.RZ, [10], [np.pi / 24]],
                    [72, qp.Hadamard, [2], []],
                    [73, qp.Hadamard, [4], []],
                    [74, qp.Hadamard, [8], []],
                    [75, qp.RX, [10], [np.pi / 2]],
                ],
            ),
            (
                [3, 4, 5],
                [11, 12, 13, 14, 15],
                [
                    [92, qp.RX, [3], [-np.pi / 2]],
                    [93, qp.Hadamard, [5], []],
                    [94, qp.Hadamard, [11], []],
                    [95, qp.Hadamard, [15], []],
                    [103, qp.RZ, [15], [-np.pi / 24]],
                    [111, qp.RX, [3], [np.pi / 2]],
                    [112, qp.Hadamard, [5], []],
                    [113, qp.Hadamard, [11], []],
                    [114, qp.Hadamard, [15], []],
                ],
            ),
            (
                [4, 5, 6, 7],
                [9, 10],
                [
                    [95, qp.Hadamard, [4], []],
                    [96, qp.RX, [7], [-np.pi / 2]],
                    [97, qp.Hadamard, [9], []],
                    [98, qp.Hadamard, [10], []],
                    [104, qp.RZ, [10], [-np.pi / 24]],
                    [110, qp.Hadamard, [4], []],
                    [111, qp.RX, [7], [np.pi / 2]],
                    [112, qp.Hadamard, [9], []],
                    [113, qp.Hadamard, [10], []],
                ],
            ),
            (
                [5, 6],
                [10, 11, 12],
                [
                    [102, qp.RX, [5], [-np.pi / 2]],
                    [103, qp.RX, [6], [-np.pi / 2]],
                    [104, qp.RX, [10], [-np.pi / 2]],
                    [105, qp.Hadamard, [12], []],
                    [110, qp.RZ, [12], [-np.pi / 24]],
                    [115, qp.RX, [5], [np.pi / 2]],
                    [116, qp.RX, [6], [np.pi / 2]],
                    [117, qp.RX, [10], [np.pi / 2]],
                    [118, qp.Hadamard, [12], []],
                ],
            ),
            (
                [3, 4, 5, 6],
                [17, 18, 19],
                [
                    [147, qp.RX, [3], [-np.pi / 2]],
                    [148, qp.RX, [6], [-np.pi / 2]],
                    [149, qp.Hadamard, [17], []],
                    [150, qp.RX, [19], [-np.pi / 2]],
                    [157, qp.RZ, [19], [-np.pi / 24]],
                    [164, qp.RX, [3], [np.pi / 2]],
                    [165, qp.RX, [6], [np.pi / 2]],
                    [166, qp.Hadamard, [17], []],
                    [167, qp.RX, [19], [np.pi / 2]],
                ],
            ),
            (
                [6, 7],
                [8, 9],
                [
                    [4, qp.CNOT, [6, 7], []],
                    [5, qp.CNOT, [7, 8], []],
                    [6, qp.CNOT, [8, 9], []],
                    [8, qp.CNOT, [8, 9], []],
                    [9, qp.CNOT, [7, 8], []],
                    [10, qp.CNOT, [6, 7], []],
                ],
            ),
            (
                [4, 5, 6, 7],
                [8, 9, 10, 11, 12, 13],
                [
                    [58, qp.CNOT, [4, 5], []],
                    [59, qp.CNOT, [5, 6], []],
                    [60, qp.CNOT, [6, 7], []],
                    [61, qp.CNOT, [7, 8], []],
                    [62, qp.CNOT, [8, 9], []],
                    [63, qp.CNOT, [9, 10], []],
                    [64, qp.CNOT, [10, 11], []],
                    [65, qp.CNOT, [11, 12], []],
                    [66, qp.CNOT, [12, 13], []],
                    [122, qp.CNOT, [12, 13], []],
                    [123, qp.CNOT, [11, 12], []],
                    [124, qp.CNOT, [10, 11], []],
                    [125, qp.CNOT, [9, 10], []],
                    [126, qp.CNOT, [8, 9], []],
                    [127, qp.CNOT, [7, 8], []],
                    [128, qp.CNOT, [6, 7], []],
                    [129, qp.CNOT, [5, 6], []],
                    [130, qp.CNOT, [4, 5], []],
                ],
            ),
        ],
    )
    def test_double_ex_unitary_operations(self, wires1, wires2, ref_gates):
        """Test the correctness of the FermionicDoubleExcitation template including the gate count
        and order, the wires each operation acts on and the correct use of parameters
        in the circuit."""

        sqg = 72
        cnots = 16 * (len(wires1) - 1 + len(wires2) - 1 + 1)
        weight = np.pi / 3
        op = qp.FermionicDoubleExcitation(weight, wires1=wires1, wires2=wires2)
        queue = op.decomposition()

        assert len(queue) == sqg + cnots

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
            assert res_weight == exp_weight

    @pytest.mark.parametrize(
        ("wires1", "wires2"),
        [
            (
                [0, 1, 2],
                [4, 5, 6],
            ),
            (
                [0, 1],
                [4, 5],
            ),
        ],
    )
    @pytest.mark.capture
    def test_decomposition_new_capture(self, wires1, wires2):
        """Tests the decomposition rule implemented with the new system."""
        op = qp.FermionicDoubleExcitation(np.pi / 3, wires1=wires1, wires2=wires2)

        for rule in qp.list_decomps(qp.FermionicDoubleExcitation):
            _test_decomposition_rule(op, rule)

    @pytest.mark.parametrize(
        ("wires1", "wires2"),
        [
            (
                [0, 1, 2],
                [4, 5, 6],
            ),
            (
                [0, 1],
                [4, 5],
            ),
        ],
    )
    def test_decomposition_new(self, wires1, wires2):
        """Tests the decomposition rule implemented with the new system."""
        op = qp.FermionicDoubleExcitation(np.pi / 3, wires1=wires1, wires2=wires2)

        for rule in qp.list_decomps(qp.FermionicDoubleExcitation):
            _test_decomposition_rule(op, rule)

    def test_custom_wire_labels(self, tol):
        """Test that template can deal with non-numeric, nonconsecutive wire labels."""

        dev = qp.device("default.qubit", wires=5)
        dev2 = qp.device("default.qubit", wires=["z", "a", "k", "t", "s"])

        @qp.qnode(dev)
        def circuit():
            qp.FermionicDoubleExcitation(0.4, wires1=[0, 2], wires2=[1, 4, 3])
            return qp.expval(qp.Identity(0)), qp.state()

        @qp.qnode(dev2)
        def circuit2():
            qp.FermionicDoubleExcitation(0.4, wires1=["z", "k"], wires2=["a", "s", "t"])
            return qp.expval(qp.Identity("z")), qp.state()

        res1, state1 = circuit()
        res2, state2 = circuit2()

        assert np.allclose(res1, res2, atol=tol, rtol=0)
        assert np.allclose(state1, state2, atol=tol, rtol=0)


class TestInputs:
    """Test inputs and pre-processing."""

    @pytest.mark.parametrize(
        ("weight", "wires1", "wires2", "msg_match"),
        [
            (0.2, [0], [1, 2], "expected at least two wires representing the occupied"),
            (0.2, [0, 1], [2], "expected at least two wires representing the unoccupied"),
            (0.2, [0], [1], "expected at least two wires representing the occupied"),
            ([0.2, 1.1], [0, 2], [4, 6], "Weight must be a scalar"),
        ],
    )
    def test_double_excitation_unitary_exceptions(self, weight, wires1, wires2, msg_match):
        """Test exception if ``weight`` or
        ``pphh`` parameter has illegal shapes, types or values."""
        dev = qp.device("default.qubit", wires=10)

        def circuit(weight):
            qp.FermionicDoubleExcitation(weight=weight, wires1=wires1, wires2=wires2)
            return qp.expval(qp.PauliZ(0))

        qnode = qp.QNode(circuit, dev)

        with pytest.raises(ValueError, match=msg_match):
            qnode(weight)

    def test_id(self):
        """Tests that the id attribute can be set."""
        template = qp.FermionicDoubleExcitation(0.4, wires1=[0, 2], wires2=[1, 4, 3], id="a")
        assert template.id == "a"


def circuit_template(weight):
    qp.FermionicDoubleExcitation(weight, wires1=[0, 1], wires2=[2, 3])
    return qp.expval(qp.PauliZ(0))


class TestInterfaces:
    """Tests that the template is compatible with all interfaces, including the computation
    of gradients."""

    @pytest.mark.autograd
    def test_autograd(self):
        """Tests the autograd interface."""

        weight = pnp.array(0.5, requires_grad=True)

        dev = qp.device("default.qubit", wires=4)

        circuit = qp.QNode(circuit_template, dev)

        circuit(weight)
        grad_fn = qp.grad(circuit)

        # since test cases are hard to construct
        # for this template, just check that the gradient is computed
        # without error
        grad_fn(weight)

    @pytest.mark.jax
    @pytest.mark.slow
    def test_jax(self):
        """Tests the jax interface."""

        import jax
        import jax.numpy as jnp

        weight = jnp.array(0.5)
        dev = qp.device("default.qubit", wires=4)

        circuit = qp.QNode(circuit_template, dev)

        circuit(weight)
        grad_fn = jax.grad(circuit)

        # check that the gradient is computed without error
        grad_fn(weight)

    @pytest.mark.jax
    def test_jax_jit(self):
        """Test the template compiles with JAX JIT."""

        import jax
        import jax.numpy as jnp

        weight = jnp.array(0.5)
        dev = qp.device("default.qubit", wires=4)

        circuit = qp.QNode(circuit_template, dev)
        circuit2 = jax.jit(circuit)

        assert qp.math.allclose(circuit(weight), circuit2(weight))

        grad_fn = jax.grad(circuit)
        grad_fn2 = jax.grad(circuit2)

        assert qp.math.allclose(grad_fn(weight), grad_fn2(weight))

    @pytest.mark.tf
    def test_tf(self):
        """Tests the tf interface."""

        import tensorflow as tf

        weight = tf.Variable(0.5)
        dev = qp.device("default.qubit", wires=4)

        circuit = qp.QNode(circuit_template, dev)

        circuit(weight)

        with tf.GradientTape() as tape:
            res = circuit(weight)

        # check that the gradient is computed without error
        tape.gradient(res, [weight])

    @pytest.mark.torch
    def test_torch(self):
        """Tests the torch interface."""

        import torch

        weight = torch.tensor(0.5, requires_grad=True)

        dev = qp.device("default.qubit", wires=4)

        circuit = qp.QNode(circuit_template, dev)

        circuit(weight)

        res = circuit(weight)
        res.backward()
        # check that the gradient is computed without error
        weight.grad  # pylint: disable=pointless-statement
