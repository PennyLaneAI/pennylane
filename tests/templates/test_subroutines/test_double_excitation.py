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
import pytest
import numpy as np
from pennylane import numpy as pnp
import pennylane as qml


class TestDecomposition:
    """Tests that the template defines the correct decomposition."""

    @pytest.mark.parametrize(
        ("wires1", "wires2", "ref_gates"),
        [
            (
                [0, 1, 2],
                [4, 5, 6],
                [
                    [0, qml.Hadamard, [0], []],
                    [1, qml.Hadamard, [2], []],
                    [2, qml.RX, [4], [-np.pi / 2]],
                    [3, qml.Hadamard, [6], []],
                    [9, qml.RZ, [6], [np.pi / 24]],
                    [15, qml.Hadamard, [0], []],
                    [16, qml.Hadamard, [2], []],
                    [17, qml.RX, [4], [np.pi / 2]],
                    [18, qml.Hadamard, [6], []],
                ],
            ),
            (
                [0, 1],
                [4, 5],
                [
                    [15, qml.RX, [0], [-np.pi / 2]],
                    [16, qml.Hadamard, [1], []],
                    [17, qml.RX, [4], [-np.pi / 2]],
                    [18, qml.RX, [5], [-np.pi / 2]],
                    [22, qml.RZ, [5], [np.pi / 24]],
                    [26, qml.RX, [0], [np.pi / 2]],
                    [27, qml.Hadamard, [1], []],
                    [28, qml.RX, [4], [np.pi / 2]],
                    [29, qml.RX, [5], [np.pi / 2]],
                ],
            ),
            (
                [1, 2, 3],
                [7, 8, 9, 10, 11],
                [
                    [46, qml.Hadamard, [1], []],
                    [47, qml.RX, [3], [-np.pi / 2]],
                    [48, qml.RX, [7], [-np.pi / 2]],
                    [49, qml.RX, [11], [-np.pi / 2]],
                    [57, qml.RZ, [11], [np.pi / 24]],
                    [65, qml.Hadamard, [1], []],
                    [66, qml.RX, [3], [np.pi / 2]],
                    [67, qml.RX, [7], [np.pi / 2]],
                    [68, qml.RX, [11], [np.pi / 2]],
                ],
            ),
            (
                [2, 3, 4],
                [8, 9, 10],
                [
                    [57, qml.Hadamard, [2], []],
                    [58, qml.Hadamard, [4], []],
                    [59, qml.Hadamard, [8], []],
                    [60, qml.RX, [10], [-np.pi / 2]],
                    [66, qml.RZ, [10], [np.pi / 24]],
                    [72, qml.Hadamard, [2], []],
                    [73, qml.Hadamard, [4], []],
                    [74, qml.Hadamard, [8], []],
                    [75, qml.RX, [10], [np.pi / 2]],
                ],
            ),
            (
                [3, 4, 5],
                [11, 12, 13, 14, 15],
                [
                    [92, qml.RX, [3], [-np.pi / 2]],
                    [93, qml.Hadamard, [5], []],
                    [94, qml.Hadamard, [11], []],
                    [95, qml.Hadamard, [15], []],
                    [103, qml.RZ, [15], [-np.pi / 24]],
                    [111, qml.RX, [3], [np.pi / 2]],
                    [112, qml.Hadamard, [5], []],
                    [113, qml.Hadamard, [11], []],
                    [114, qml.Hadamard, [15], []],
                ],
            ),
            (
                [4, 5, 6, 7],
                [9, 10],
                [
                    [95, qml.Hadamard, [4], []],
                    [96, qml.RX, [7], [-np.pi / 2]],
                    [97, qml.Hadamard, [9], []],
                    [98, qml.Hadamard, [10], []],
                    [104, qml.RZ, [10], [-np.pi / 24]],
                    [110, qml.Hadamard, [4], []],
                    [111, qml.RX, [7], [np.pi / 2]],
                    [112, qml.Hadamard, [9], []],
                    [113, qml.Hadamard, [10], []],
                ],
            ),
            (
                [5, 6],
                [10, 11, 12],
                [
                    [102, qml.RX, [5], [-np.pi / 2]],
                    [103, qml.RX, [6], [-np.pi / 2]],
                    [104, qml.RX, [10], [-np.pi / 2]],
                    [105, qml.Hadamard, [12], []],
                    [110, qml.RZ, [12], [-np.pi / 24]],
                    [115, qml.RX, [5], [np.pi / 2]],
                    [116, qml.RX, [6], [np.pi / 2]],
                    [117, qml.RX, [10], [np.pi / 2]],
                    [118, qml.Hadamard, [12], []],
                ],
            ),
            (
                [3, 4, 5, 6],
                [17, 18, 19],
                [
                    [147, qml.RX, [3], [-np.pi / 2]],
                    [148, qml.RX, [6], [-np.pi / 2]],
                    [149, qml.Hadamard, [17], []],
                    [150, qml.RX, [19], [-np.pi / 2]],
                    [157, qml.RZ, [19], [-np.pi / 24]],
                    [164, qml.RX, [3], [np.pi / 2]],
                    [165, qml.RX, [6], [np.pi / 2]],
                    [166, qml.Hadamard, [17], []],
                    [167, qml.RX, [19], [np.pi / 2]],
                ],
            ),
            (
                [6, 7],
                [8, 9],
                [
                    [4, qml.CNOT, [6, 7], []],
                    [5, qml.CNOT, [7, 8], []],
                    [6, qml.CNOT, [8, 9], []],
                    [8, qml.CNOT, [8, 9], []],
                    [9, qml.CNOT, [7, 8], []],
                    [10, qml.CNOT, [6, 7], []],
                ],
            ),
            (
                [4, 5, 6, 7],
                [8, 9, 10, 11, 12, 13],
                [
                    [58, qml.CNOT, [4, 5], []],
                    [59, qml.CNOT, [5, 6], []],
                    [60, qml.CNOT, [6, 7], []],
                    [61, qml.CNOT, [7, 8], []],
                    [62, qml.CNOT, [8, 9], []],
                    [63, qml.CNOT, [9, 10], []],
                    [64, qml.CNOT, [10, 11], []],
                    [65, qml.CNOT, [11, 12], []],
                    [66, qml.CNOT, [12, 13], []],
                    [122, qml.CNOT, [12, 13], []],
                    [123, qml.CNOT, [11, 12], []],
                    [124, qml.CNOT, [10, 11], []],
                    [125, qml.CNOT, [9, 10], []],
                    [126, qml.CNOT, [8, 9], []],
                    [127, qml.CNOT, [7, 8], []],
                    [128, qml.CNOT, [6, 7], []],
                    [129, qml.CNOT, [5, 6], []],
                    [130, qml.CNOT, [4, 5], []],
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
        op = qml.FermionicDoubleExcitation(weight, wires1=wires1, wires2=wires2)
        queue = op.expand().operations

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

    def test_custom_wire_labels(self, tol):
        """Test that template can deal with non-numeric, nonconsecutive wire labels."""

        dev = qml.device("default.qubit", wires=5)
        dev2 = qml.device("default.qubit", wires=["z", "a", "k", "t", "s"])

        @qml.qnode(dev)
        def circuit():
            qml.FermionicDoubleExcitation(0.4, wires1=[0, 2], wires2=[1, 4, 3])
            return qml.expval(qml.Identity(0))

        @qml.qnode(dev2)
        def circuit2():
            qml.FermionicDoubleExcitation(0.4, wires1=["z", "k"], wires2=["a", "s", "t"])
            return qml.expval(qml.Identity("z"))

        circuit()
        circuit2()

        assert np.allclose(dev.state, dev2.state, atol=tol, rtol=0)


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
        dev = qml.device("default.qubit", wires=10)

        def circuit(weight):
            qml.FermionicDoubleExcitation(weight=weight, wires1=wires1, wires2=wires2)
            return qml.expval(qml.PauliZ(0))

        qnode = qml.QNode(circuit, dev)

        with pytest.raises(ValueError, match=msg_match):
            qnode(weight)

    def test_id(self):
        """Tests that the id attribute can be set."""
        template = qml.FermionicDoubleExcitation(0.4, wires1=[0, 2], wires2=[1, 4, 3], id="a")
        assert template.id == "a"


def circuit_template(weight):
    qml.FermionicDoubleExcitation(weight, wires1=[0, 1], wires2=[2, 3])
    return qml.expval(qml.PauliZ(0))


class TestInterfaces:
    """Tests that the template is compatible with all interfaces, including the computation
    of gradients."""

    @pytest.mark.autograd
    def test_autograd(self):
        """Tests the autograd interface."""

        weight = pnp.array(0.5, requires_grad=True)

        dev = qml.device("default.qubit", wires=4)

        circuit = qml.QNode(circuit_template, dev)

        circuit(weight)
        grad_fn = qml.grad(circuit)

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
        dev = qml.device("default.qubit", wires=4)

        circuit = qml.QNode(circuit_template, dev, interface="jax")

        circuit(weight)
        grad_fn = jax.grad(circuit)

        # check that the gradient is computed without error
        grad_fn(weight)

    @pytest.mark.tf
    def test_tf(self):
        """Tests the tf interface."""

        import tensorflow as tf

        weight = tf.Variable(0.5)
        dev = qml.device("default.qubit", wires=4)

        circuit = qml.QNode(circuit_template, dev, interface="tf")

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

        dev = qml.device("default.qubit", wires=4)

        circuit = qml.QNode(circuit_template, dev, interface="torch")

        circuit(weight)

        res = circuit(weight)
        res.backward()
        # check that the gradient is computed without error
        [weight.grad]
