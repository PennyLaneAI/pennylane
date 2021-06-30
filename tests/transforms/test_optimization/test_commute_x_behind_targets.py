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

import pytest
from pennylane import numpy as np
import pennylane as qml
from pennylane.wires import Wires

from pennylane.transforms.optimization import commute_x_behind_targets


class TestCommuteXBehindTargets:
    """Test that X rotations are properly pushed behind targets of X-based controlled operations."""

    def test_single_x_after_cnot_gate(self):
        """Test that a single X after a CNOT is pushed behind."""

        def qfunc():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 2])
            qml.PauliX(wires=2)

        transformed_qfunc = commute_x_behind_targets(qfunc)

        ops = qml.transforms.make_tape(transformed_qfunc)().operations

        assert len(ops) == 3

        assert ops[0].name == "Hadamard"
        assert ops[0].wires == Wires(0)

        assert ops[1].name == "PauliX"
        assert ops[1].wires == Wires(2)

        assert ops[2].name == "CNOT"
        assert ops[2].wires == Wires([0, 2])

    def test_multiple_x_after_cnot_gate(self):
        """Test that multiple X rotations after a CNOT both get pushed behind."""

        def qfunc():
            qml.Hadamard(wires="a")
            qml.CNOT(wires=["b", "a"])
            qml.RX(0.2, wires="a")
            qml.PauliX(wires="a")

        transformed_qfunc = commute_x_behind_targets(qfunc)

        ops = qml.transforms.make_tape(transformed_qfunc)().operations

        assert len(ops) == 4

        assert ops[0].name == "Hadamard"
        assert ops[0].wires == Wires("a")

        assert ops[1].name == "RX"
        assert ops[1].parameters[0] == 0.2
        assert ops[1].wires == Wires("a")

        assert ops[2].name == "PauliX"
        assert ops[2].wires == Wires("a")

        assert ops[3].name == "CNOT"
        assert ops[3].wires == Wires(["b", "a"])

    def test_single_x_after_crx_gate(self):
        """Test that a single X rotation after a CRX is pushed behind."""

        def qfunc():
            qml.Hadamard(wires=0)
            qml.CRX(0.1, wires=[0, "a"])
            qml.RX(0.2, wires="a")

        transformed_qfunc = commute_x_behind_targets(qfunc)

        ops = qml.transforms.make_tape(transformed_qfunc)().operations

        assert len(ops) == 3

        assert ops[0].name == "Hadamard"
        assert ops[0].wires == Wires(0)

        assert ops[1].name == "RX"
        assert ops[1].parameters[0] == 0.2
        assert ops[1].wires == Wires("a")

        assert ops[2].name == "CRX"
        assert ops[2].parameters[0] == 0.1
        assert ops[2].wires == Wires([0, "a"])

    def test_multiple_x_after_crx_gate(self):
        """Test that multiple X rotations after a CRX are pushed behind."""

        def qfunc():
            qml.Hadamard(wires="a")
            qml.CRX(0.3, wires=["b", "a"])
            qml.PauliX(wires="a")
            qml.RX(0.1, wires="a")

        transformed_qfunc = commute_x_behind_targets(qfunc)

        ops = qml.transforms.make_tape(transformed_qfunc)().operations

        assert len(ops) == 4

        assert ops[0].name == "Hadamard"
        assert ops[0].wires == Wires("a")

        assert ops[1].name == "PauliX"
        assert ops[1].wires == Wires("a")

        assert ops[2].name == "RX"
        assert ops[2].parameters[0] == 0.1
        assert ops[2].wires == Wires("a")

        assert ops[3].name == "CRX"
        assert ops[3].parameters[0] == 0.3
        assert ops[3].wires == Wires(["b", "a"])

    def test_single_x_after_toffoli_gate(self):
        """Test that a single X rotation after a Toffoli is pushed behind."""

        def qfunc():
            qml.Hadamard(wires=0)
            qml.Toffoli(wires=[0, 3, "a"])
            qml.RX(0.2, wires="a")

        transformed_qfunc = commute_x_behind_targets(qfunc)

        ops = qml.transforms.make_tape(transformed_qfunc)().operations

        assert len(ops) == 3

        assert ops[0].name == "Hadamard"
        assert ops[0].wires == Wires(0)

        assert ops[1].name == "RX"
        assert ops[1].parameters[0] == 0.2
        assert ops[1].wires == Wires("a")

        assert ops[2].name == "Toffoli"
        assert ops[2].wires == Wires([0, 3, "a"])

    def test_multiple_x_after_toffoli_gate(self):
        """Test that multiple X rotations after a Toffoli are pushed behind."""

        def qfunc():
            qml.Hadamard(wires="a")
            qml.Toffoli(wires=["b", "c", "a"])
            qml.RX(0.1, wires="a")
            qml.PauliX(wires="b")
            qml.RX(0.2, wires="a")

        transformed_qfunc = commute_x_behind_targets(qfunc)

        ops = qml.transforms.make_tape(transformed_qfunc)().operations

        assert len(ops) == 5

        assert ops[0].name == "Hadamard"
        assert ops[0].wires == Wires("a")

        assert ops[1].name == "RX"
        assert ops[1].parameters[0] == 0.1
        assert ops[1].wires == Wires("a")

        assert ops[2].name == "RX"
        assert ops[2].parameters[0] == 0.2
        assert ops[2].wires == Wires("a")

        assert ops[3].name == "Toffoli"
        assert ops[3].wires == Wires(["b", "c", "a"])

        assert ops[4].name == "PauliX"
        assert ops[4].wires == Wires(["b"])

    def test_no_commuting_gates_after_crx(self):
        """Test that pushing commuting X gates behind targets is properly 'blocked'."""

        def qfunc():
            qml.Hadamard(wires=0)
            qml.CRX(0.1, wires=[0, "a"])
            # The Hadamard blocks the CRX from moving ahead of the PauliX
            qml.Hadamard(wires="a")
            qml.PauliX(wires="a")

        transformed_qfunc = commute_x_behind_targets(qfunc)

        ops = qml.transforms.make_tape(transformed_qfunc)().operations

        assert len(ops) == 4

        assert ops[0].name == "Hadamard"
        assert ops[0].wires == Wires(0)

        assert ops[1].name == "CRX"
        assert ops[1].parameters[0] == 0.1
        assert ops[1].wires == Wires([0, "a"])

        assert ops[2].name == "Hadamard"
        assert ops[2].wires == Wires("a")

        assert ops[3].name == "PauliX"
        assert ops[3].wires == Wires("a")


# Example QNode and device for interface testing
dev = qml.device("default.qubit", wires=3)


def qfunc(theta):
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    qml.RX(theta[0], wires=1)
    qml.PauliY(wires=1)
    qml.CNOT(wires=[1, 2])
    qml.PauliX(wires=2)
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(2))


transformed_qfunc = commute_x_behind_targets(qfunc)

expected_op_list = ["Hadamard", "RX", "CNOT", "PauliY", "PauliX", "CNOT"]
expected_wires_list = [Wires(0), Wires(1), Wires([0, 1]), Wires(1), Wires(2), Wires([1, 2])]


class TestCommuteXBehindTargetsInterfaces:
    """Test that X gates can be pushed behind X-based targets in all interfaces."""

    def test_commute_x_behind_targets_autograd(self):
        """Test QNode and gradient in autograd interface."""

        original_qnode = qml.QNode(qfunc, dev)
        transformed_qnode = qml.QNode(transformed_qfunc, dev)

        input = np.array([0.1], requires_grad=True)

        # Check that the numerical output is the same
        assert qml.math.allclose(original_qnode(input), transformed_qnode(input))

        # Check that the gradient is the same
        assert qml.math.allclose(
            qml.grad(original_qnode)(input), qml.grad(transformed_qnode)(input)
        )

        # Check operation list
        ops = transformed_qnode.qtape.operations
        assert len(ops) == 6
        assert all([op.name == expected_name for (op, expected_name) in zip(ops, expected_op_list)])
        assert all(
            [op.wires == expected_wires for (op, expected_wires) in zip(ops, expected_wires_list)]
        )

    def test_commute_x_behind_targets_torch(self):
        """Test QNode and gradient in torch interface."""
        torch = pytest.importorskip("torch", minversion="1.8")

        original_qnode = qml.QNode(qfunc, dev, interface="torch")
        transformed_qnode = qml.QNode(transformed_qfunc, dev, interface="torch")

        original_input = torch.tensor([0.1], requires_grad=True)
        transformed_input = torch.tensor([0.1], requires_grad=True)

        original_result = original_qnode(original_input)
        transformed_result = transformed_qnode(transformed_input)

        # Check that the numerical output is the same
        assert qml.math.allclose(original_result, transformed_result)

        # Check that the gradient is the same
        original_result.backward()
        transformed_result.backward()

        assert qml.math.allclose(original_input.grad, transformed_input.grad)

        # Check operation list
        ops = transformed_qnode.qtape.operations
        assert len(ops) == 6
        assert all([op.name == expected_name for (op, expected_name) in zip(ops, expected_op_list)])
        assert all(
            [op.wires == expected_wires for (op, expected_wires) in zip(ops, expected_wires_list)]
        )

    def test_commute_x_behind_targets_tf(self):
        """Test QNode and gradient in tensorflow interface."""
        tf = pytest.importorskip("tensorflow")

        original_qnode = qml.QNode(qfunc, dev, interface="tf")
        transformed_qnode = qml.QNode(transformed_qfunc, dev, interface="tf")

        original_input = tf.Variable([0.1])
        transformed_input = tf.Variable([0.1])

        original_result = original_qnode(original_input)
        transformed_result = transformed_qnode(transformed_input)

        # Check that the numerical output is the same
        assert qml.math.allclose(original_result, transformed_result)

        # Check that the gradient is the same
        with tf.GradientTape() as tape:
            loss = original_qnode(original_input)
        original_grad = tape.gradient(loss, original_input)

        with tf.GradientTape() as tape:
            loss = transformed_qnode(transformed_input)
        transformed_grad = tape.gradient(loss, transformed_input)

        assert qml.math.allclose(original_grad, transformed_grad)

        # Check operation list
        ops = transformed_qnode.qtape.operations
        assert len(ops) == 6
        assert all([op.name == expected_name for (op, expected_name) in zip(ops, expected_op_list)])
        assert all(
            [op.wires == expected_wires for (op, expected_wires) in zip(ops, expected_wires_list)]
        )

    def test_commute_x_behind_targets_jax(self):
        """Test QNode and gradient in JAX interface."""
        jax = pytest.importorskip("jax")
        from jax import numpy as jnp

        original_qnode = qml.QNode(qfunc, dev, interface="jax")
        transformed_qnode = qml.QNode(transformed_qfunc, dev, interface="jax")

        input = jnp.array([0.1], dtype=jnp.float64)

        # Check that the numerical output is the same
        assert qml.math.allclose(original_qnode(input), transformed_qnode(input))

        # Check that the gradient is the same
        assert qml.math.allclose(
            jax.grad(original_qnode)(input), jax.grad(transformed_qnode)(input)
        )

        # Check operation list
        ops = transformed_qnode.qtape.operations
        assert len(ops) == 6
        assert all([op.name == expected_name for (op, expected_name) in zip(ops, expected_op_list)])
        assert all(
            [op.wires == expected_wires for (op, expected_wires) in zip(ops, expected_wires_list)]
        )
