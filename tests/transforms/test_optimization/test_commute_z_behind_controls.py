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
from pennylane.transforms.optimization import commute_z_behind_controls


class TestCommuteZBehindControls:
    """Test that diagonal gates are properly pushed behind X-based target operations."""

    def test_single_z_after_cnot_gate(self):
        """Test that a single Z after a CNOT is pushed behind."""

        def qfunc():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 2])
            qml.PauliZ(wires=0)

        transformed_qfunc = commute_z_behind_controls(qfunc)

        ops = qml.transforms.make_tape(transformed_qfunc)().operations

        assert len(ops) == 3

        assert ops[0].name == "Hadamard"
        assert ops[0].wires == Wires(0)

        assert ops[1].name == "PauliZ"
        assert ops[1].wires == Wires(0)

        assert ops[2].name == "CNOT"
        assert ops[2].wires == Wires([0, 2])

    def test_multiple_z_after_cnot_gate(self):
        """Test that multiple Z rotations after a CNOT both get pushed behind."""

        def qfunc():
            qml.Hadamard(wires="a")
            qml.CNOT(wires=["b", "a"])
            qml.RZ(0.2, wires="b")
            qml.PauliZ(wires="b")

        transformed_qfunc = commute_z_behind_controls(qfunc)

        ops = qml.transforms.make_tape(transformed_qfunc)().operations

        assert len(ops) == 4

        assert ops[0].name == "Hadamard"
        assert ops[0].wires == Wires("a")

        assert ops[1].name == "RZ"
        assert ops[1].parameters[0] == 0.2
        assert ops[1].wires == Wires("b")

        assert ops[2].name == "PauliZ"
        assert ops[2].wires == Wires("b")

        assert ops[3].name == "CNOT"
        assert ops[3].wires == Wires(["b", "a"])

    def test_single_z_after_cry_gate(self):
        """Test that a single Z rotation after a CRY is pushed behind."""

        def qfunc():
            qml.Hadamard(wires=0)
            qml.CRY(0.1, wires=[0, "a"])
            qml.RZ(0.2, wires=0)

        transformed_qfunc = commute_z_behind_controls(qfunc)

        ops = qml.transforms.make_tape(transformed_qfunc)().operations

        assert len(ops) == 3

        assert ops[0].name == "Hadamard"
        assert ops[0].wires == Wires(0)

        assert ops[1].name == "RZ"
        assert ops[1].parameters[0] == 0.2
        assert ops[1].wires == Wires(0)

        assert ops[2].name == "CRY"
        assert ops[2].parameters[0] == 0.1
        assert ops[2].wires == Wires([0, "a"])

    def test_multiple_z_after_crx_gate(self):
        """Test that multiple Z rotations after a CRX are pushed behind."""

        def qfunc():
            qml.Hadamard(wires="a")
            qml.CRX(0.3, wires=["b", "a"])
            qml.PhaseShift(0.2, wires="b")
            qml.T(wires="b")

        transformed_qfunc = commute_z_behind_controls(qfunc)

        ops = qml.transforms.make_tape(transformed_qfunc)().operations

        assert len(ops) == 4

        assert ops[0].name == "Hadamard"
        assert ops[0].wires == Wires("a")

        assert ops[1].name == "PhaseShift"
        assert ops[1].parameters[0] == 0.2
        assert ops[1].wires == Wires("b")

        assert ops[2].name == "T"
        assert ops[2].wires == Wires("b")

        assert ops[3].name == "CRX"
        assert ops[3].parameters[0] == 0.3
        assert ops[3].wires == Wires(["b", "a"])

    def test_no_commuting_gates_after_crx(self):
        """Test that pushing commuting X gates behind targets is properly 'blocked'."""

        def qfunc():
            qml.Hadamard(wires=0)
            qml.CRX(0.1, wires=[0, "a"])
            # The Hadamard blocks the CRX from moving ahead of the PauliX
            qml.Hadamard(wires=0)
            qml.S(wires=0)

        transformed_qfunc = commute_z_behind_controls(qfunc)

        ops = qml.transforms.make_tape(transformed_qfunc)().operations

        assert len(ops) == 4

        assert ops[0].name == "Hadamard"
        assert ops[0].wires == Wires(0)

        assert ops[1].name == "CRX"
        assert ops[1].parameters[0] == 0.1
        assert ops[1].wires == Wires([0, "a"])

        assert ops[2].name == "Hadamard"
        assert ops[2].wires == Wires(0)

        assert ops[3].name == "S"
        assert ops[3].wires == Wires(0)


# Example QNode and device for interface testing
dev = qml.device("default.qubit", wires=3)


def qfunc(theta):
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    qml.RZ(theta[0], wires=0)
    qml.PauliY(wires=1)
    qml.CNOT(wires=[1, 2])
    qml.PauliZ(wires=1)
    return qml.expval(qml.PauliX(0) @ qml.PauliX(2))


transformed_qfunc = commute_z_behind_controls(qfunc)

expected_op_list = ["Hadamard", "RZ", "CNOT", "PauliY", "PauliZ", "CNOT"]
expected_wires_list = [Wires(0), Wires(0), Wires([0, 1]), Wires(1), Wires(1), Wires([1, 2])]


class TestCommuteZBehindControlsInterfaces:
    """Test that Z gates can be pushed behind Z-based targets in all interfaces."""

    def test_commute_z_behind_controls_autograd(self):
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

    def test_commute_z_behind_controls_torch(self):
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

    def test_commute_z_behind_controls_tf(self):
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

    def test_commute_z_behind_controls_jax(self):
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
