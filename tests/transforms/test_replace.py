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
from pennylane.transforms.replace import replace
from pennylane.transforms.optimization import cancel_inverses
from test_optimization.utils import compare_operation_lists


# Various custom decomposition functions
def custom_hadamard(wires):
    return [qml.RY(np.pi / 2, wires=wires), qml.PauliX(wires=wires)]


def custom_cnot(wires):
    return [
        qml.Hadamard(wires=wires[1]),
        qml.CZ(wires=[wires[0], wires[1]]),
        qml.Hadamard(wires=wires[1]),
    ]


def custom_rx(theta, wires):
    return [qml.PauliY(wires=wires), qml.RX(-theta, wires=wires), qml.PauliY(wires=wires)]


class TestReplace:
    """Test that custom definitions of operations are properly replaced."""

    def test_replace_no_custom_ops(self):
        """Test that replacement with no custom ops does nothing to a quantum function."""

        def qfunc():
            qml.Hadamard(wires=0)
            qml.Hadamard(wires=1)

        transformed_qfunc = replace()(qfunc)
        transformed_ops = qml.transforms.make_tape(transformed_qfunc)().operations

        compare_operation_lists(transformed_ops, ["Hadamard"] * 2, [Wires(0), Wires(1)])

    def test_replace_one_instance_custom_op(self):
        """Test that a single operator in a circuit is correctly replaced."""

        custom_ops = {"Hadamard": custom_hadamard}

        def qfunc():
            qml.Hadamard(wires="a")

        transformed_qfunc = replace(custom_ops=custom_ops)(qfunc)
        transformed_ops = qml.transforms.make_tape(transformed_qfunc)().operations

        compare_operation_lists(transformed_ops, ["RY", "PauliX"], [Wires("a"), Wires("a")])

    def test_replace_two_instances_custom_op(self):
        """Test that multiple instances of an operator in a circuit are correctly replaced."""

        custom_ops = {"Hadamard": custom_hadamard}

        def qfunc():
            qml.Hadamard(wires="a")
            qml.CNOT(wires=["a", "b"])
            qml.Hadamard(wires="b")

        transformed_qfunc = replace(custom_ops=custom_ops)(qfunc)
        transformed_ops = qml.transforms.make_tape(transformed_qfunc)().operations

        compare_operation_lists(
            transformed_ops,
            ["RY", "PauliX", "CNOT", "RY", "PauliX"],
            [Wires("a"), Wires("a"), Wires(["a", "b"]), Wires("b"), Wires("b")],
        )

    def test_replace_two_custom_op(self):
        """Test that multiple instances of an operator in a circuit are correctly replaced."""

        custom_ops = {"Hadamard": custom_hadamard, "CNOT": custom_cnot}

        def qfunc():
            qml.Hadamard(wires="a")
            qml.CNOT(wires=["a", "b"])
            qml.Hadamard(wires="b")

        transformed_qfunc = replace(custom_ops=custom_ops)(qfunc)
        transformed_ops = qml.transforms.make_tape(transformed_qfunc)().operations

        compare_operation_lists(
            transformed_ops,
            ["RY", "PauliX", "Hadamard", "CZ", "Hadamard", "RY", "PauliX"],
            [Wires("a"), Wires("a"), Wires("b"), Wires(["a", "b"])] + [Wires("b")] * 3,
        )

    def test_replace_integration(self):
        """Test that replacement plays well with other transforms"""

        @cancel_inverses
        @replace(custom_ops={"CNOT": custom_cnot})
        def qfunc():
            qml.Hadamard(wires="a")
            qml.CNOT(wires=["a", "b"])
            qml.Hadamard(wires="b")

        transformed_ops = qml.transforms.make_tape(qfunc)().operations

        compare_operation_lists(
            transformed_ops, ["Hadamard"] * 2 + ["CZ"], [Wires("a"), Wires("b"), Wires(["a", "b"])]
        )


# Example QNode and device for interface testing
dev = qml.device("default.qubit", wires=["a", "b"])


def qfunc(theta):
    qml.RX(theta[0], wires="a")
    qml.CNOT(wires=["a", "b"])
    qml.Hadamard(wires="b")
    qml.RY(theta[1], wires="b")

    return qml.expval(qml.PauliY("a") @ qml.PauliZ("b"))


custom_ops = {"CNOT": custom_cnot, "RX": custom_rx, "Hadamard": custom_hadamard}

transformed_qfunc = replace(custom_ops=custom_ops)(qfunc)

expected_op_list = ["PauliY", "RX", "PauliY", "Hadamard", "CZ", "Hadamard", "RY", "PauliX", "RY"]
expected_wires_list = [Wires("a")] * 3 + [Wires("b")] + [Wires(["a", "b"])] + [Wires("b")] * 4


class TestReplaceInterfaces:
    """Test that replacement is differentiable in all interfaces."""

    def test_replace_autograd(self):
        """Test QNode and gradient in autograd interface."""

        original_qnode = qml.QNode(qfunc, dev, interface="autograd")
        transformed_qnode = qml.QNode(transformed_qfunc, dev)

        input = np.array([0.1, 0.2], requires_grad=True)

        # Check that the numerical output is the same
        assert qml.math.allclose(original_qnode(input), transformed_qnode(input))

        # Check that the gradient is the same
        assert qml.math.allclose(
            qml.grad(original_qnode)(input), qml.grad(transformed_qnode)(input)
        )

        # Check operation list
        ops = transformed_qnode.qtape.operations
        print(ops)
        compare_operation_lists(ops, expected_op_list, expected_wires_list)

    def test_cancel_inverses_torch(self):
        """Test QNode and gradient in torch interface."""
        torch = pytest.importorskip("torch", minversion="1.8")

        original_qnode = qml.QNode(qfunc, dev, interface="torch")
        transformed_qnode = qml.QNode(transformed_qfunc, dev, interface="torch")

        original_input = torch.tensor([0.4, 0.5], requires_grad=True)
        transformed_input = torch.tensor([0.4, 0.5], requires_grad=True)

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
        compare_operation_lists(ops, expected_op_list, expected_wires_list)

    def test_cancel_inverses_tf(self):
        """Test QNode and gradient in tensorflow interface."""
        tf = pytest.importorskip("tensorflow")

        original_qnode = qml.QNode(qfunc, dev, interface="tf")
        transformed_qnode = qml.QNode(transformed_qfunc, dev, interface="tf")

        original_input = tf.Variable([0.1, 0.2])
        transformed_input = tf.Variable([0.1, 0.2])

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
        compare_operation_lists(ops, expected_op_list, expected_wires_list)

    def test_cancel_inverses_jax(self):
        """Test QNode and gradient in JAX interface."""
        jax = pytest.importorskip("jax")
        from jax import numpy as jnp

        # Enable float64 support
        from jax.config import config

        remember = config.read("jax_enable_x64")
        config.update("jax_enable_x64", True)

        original_qnode = qml.QNode(qfunc, dev, interface="jax")
        transformed_qnode = qml.QNode(transformed_qfunc, dev, interface="jax")

        input = jnp.array([0.1, 0.2], dtype=jnp.float64)

        # Check that the numerical output is the same
        assert qml.math.allclose(original_qnode(input), transformed_qnode(input))

        # Check that the gradient is the same
        assert qml.math.allclose(
            jax.grad(original_qnode)(input), jax.grad(transformed_qnode)(input)
        )

        # Check operation list
        ops = transformed_qnode.qtape.operations
        compare_operation_lists(ops, expected_op_list, expected_wires_list)
