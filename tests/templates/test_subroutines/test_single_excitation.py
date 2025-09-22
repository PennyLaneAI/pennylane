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
Tests for the FermionicSingleExcitation template.
"""
import numpy as np
import pytest

import pennylane as qml
from pennylane import numpy as pnp
from pennylane.ops.functions.assert_valid import _test_decomposition_rule


@pytest.mark.jax
def test_standard_validity():
    """Test standard validity criteria using assert_valid."""
    weight = np.pi / 3
    op = qml.FermionicSingleExcitation(weight, wires=[0, 1, 2])
    qml.ops.functions.assert_valid(op)


class TestDecomposition:
    """Tests that the template defines the correct decomposition."""

    @pytest.mark.parametrize(
        ("single_wires", "ref_gates"),
        [
            (
                [0, 1, 2],
                [
                    [0, qml.RX, [0], [-np.pi / 2]],
                    [1, qml.Hadamard, [2], []],
                    [7, qml.RX, [0], [np.pi / 2]],
                    [8, qml.Hadamard, [2], []],
                    [9, qml.Hadamard, [0], []],
                    [10, qml.RX, [2], [-np.pi / 2]],
                    [16, qml.Hadamard, [0], []],
                    [17, qml.RX, [2], [np.pi / 2]],
                    [4, qml.RZ, [2], [np.pi / 6]],
                    [13, qml.RZ, [2], [-np.pi / 6]],
                ],
            ),
            (
                [10, 11],
                [
                    [0, qml.RX, [10], [-np.pi / 2]],
                    [1, qml.Hadamard, [11], []],
                    [12, qml.Hadamard, [10], []],
                    [13, qml.RX, [11], [np.pi / 2]],
                    [3, qml.RZ, [11], [np.pi / 6]],
                    [10, qml.RZ, [11], [-np.pi / 6]],
                ],
            ),
            (
                [1, 2, 3, 4],
                [
                    [2, qml.CNOT, [1, 2], []],
                    [3, qml.CNOT, [2, 3], []],
                    [4, qml.CNOT, [3, 4], []],
                    [6, qml.CNOT, [3, 4], []],
                    [7, qml.CNOT, [2, 3], []],
                    [8, qml.CNOT, [1, 2], []],
                    [13, qml.CNOT, [1, 2], []],
                    [14, qml.CNOT, [2, 3], []],
                    [15, qml.CNOT, [3, 4], []],
                    [17, qml.CNOT, [3, 4], []],
                    [18, qml.CNOT, [2, 3], []],
                    [19, qml.CNOT, [1, 2], []],
                ],
            ),
            (
                [10, 11],
                [
                    [2, qml.CNOT, [10, 11], []],
                    [4, qml.CNOT, [10, 11], []],
                    [9, qml.CNOT, [10, 11], []],
                    [11, qml.CNOT, [10, 11], []],
                ],
            ),
        ],
    )
    def test_single_ex_unitary_operations(self, single_wires, ref_gates):
        """Test the correctness of the FermionicSingleExcitation template including the gate count
        and order, the wires each operation acts on and the correct use of parameters
        in the circuit."""
        # pylint: disable=protected-access

        sqg = 10
        cnots = 4 * (len(single_wires) - 1)
        weight = np.pi / 3
        op = qml.FermionicSingleExcitation(weight, wires=single_wires)
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

    def test_custom_wire_labels(self, tol):
        """Test that template can deal with non-numeric, nonconsecutive wire labels."""

        dev = qml.device("default.qubit", wires=3)
        dev2 = qml.device("default.qubit", wires=["z", "a", "k"])

        @qml.qnode(dev)
        def circuit():
            qml.FermionicSingleExcitation(0.4, wires=[1, 0, 2])
            return qml.expval(qml.Identity(0)), qml.state()

        @qml.qnode(dev2)
        def circuit2():
            qml.FermionicSingleExcitation(0.4, wires=["a", "z", "k"])
            return qml.expval(qml.Identity("z")), qml.state()

        res1, state1 = circuit()
        res2, state2 = circuit2()

        assert np.allclose(res1, res2, atol=tol, rtol=0)
        assert np.allclose(state1, state2, atol=tol, rtol=0)

    @pytest.mark.parametrize(
        ("single_wires"),
        [
            [0, 1, 2],
            [10, 11],
            [1, 2, 3, 4],
        ],
    )
    @pytest.mark.capture
    def test_decomposition_new_capture(self, single_wires):
        """Tests the decomposition rule implemented with the new system."""
        op = qml.FermionicSingleExcitation(
            np.pi / 3,
            wires=single_wires,
        )

        for rule in qml.list_decomps(qml.FermionicSingleExcitation):
            _test_decomposition_rule(op, rule)

    @pytest.mark.parametrize(
        ("single_wires"),
        [
            [0, 1, 2],
            [10, 11],
            [1, 2, 3, 4],
        ],
    )
    def test_decomposition_new(self, single_wires):
        """Tests the decomposition rule implemented with the new system."""
        op = qml.FermionicSingleExcitation(
            np.pi / 3,
            wires=single_wires,
        )

        for rule in qml.list_decomps(qml.FermionicSingleExcitation):
            _test_decomposition_rule(op, rule)


class TestInputs:
    """Test inputs and pre-processing."""

    @pytest.mark.parametrize(
        ("weight", "single_wires", "msg_match"),
        [
            (0.2, [0], "expected at least two wires"),
            (0.2, [], "expected at least two wires"),
            ([0.2, 1.1], [0, 1, 2], "Weight must be a scalar"),
        ],
    )
    def test_single_excitation_unitary_exceptions(self, weight, single_wires, msg_match):
        """Test that FermionicSingleExcitation throws an exception if ``weight`` or
        ``single_wires`` parameter has illegal shapes, types or values."""
        dev = qml.device("default.qubit", wires=5)

        def circuit(weight=weight):
            qml.FermionicSingleExcitation(weight=weight, wires=single_wires)
            return qml.expval(qml.PauliZ(0))

        qnode = qml.QNode(circuit, dev)

        with pytest.raises(ValueError, match=msg_match):
            qnode(weight=weight)

    def test_id(self):
        """Tests that the id attribute can be set."""
        template = qml.FermionicSingleExcitation(0.4, wires=[1, 0, 2], id="a")
        assert template.id == "a"


def circuit_template(weight):
    qml.FermionicSingleExcitation(weight, wires=[0, 1])
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
    def test_jax(self):
        """Tests the jax interface."""

        import jax
        import jax.numpy as jnp

        weight = jnp.array(0.5)
        dev = qml.device("default.qubit", wires=4)

        circuit = qml.QNode(circuit_template, dev)

        circuit(weight)
        grad_fn = jax.grad(circuit)

        # check that the gradient is computed without error
        grad_fn(weight)

    @pytest.mark.jax
    def test_jax_jit(self):
        """Tests jit within the jax interface."""

        import jax
        import jax.numpy as jnp

        weight = jnp.array(0.5)
        dev = qml.device("default.qubit", wires=4)

        circuit = qml.QNode(circuit_template, dev)
        jit_circuit = jax.jit(circuit)
        assert qml.math.allclose(circuit(weight), jit_circuit(weight))

        grad_fn = jax.grad(circuit)
        grad_jit = jax.grad(jit_circuit)
        assert qml.math.allclose(grad_fn(weight), grad_jit(weight))

    @pytest.mark.tf
    def test_tf(self):
        """Tests the tf interface."""

        import tensorflow as tf

        weight = tf.Variable(0.5)
        dev = qml.device("default.qubit", wires=4)

        circuit = qml.QNode(circuit_template, dev)

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

        circuit = qml.QNode(circuit_template, dev)

        circuit(weight)

        res = circuit(weight)
        res.backward()
        # check that the gradient is computed without error
        weight.grad  # pylint: disable=pointless-statement
