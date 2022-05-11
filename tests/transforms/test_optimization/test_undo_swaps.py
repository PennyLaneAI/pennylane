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
from pennylane.transforms.optimization import undo_swaps
from utils import compare_operation_lists


class TestUndoSwaps:
    """Test that check the main functionalities of the `undo_swaps` transform"""

    def test_one_qubit_gates_transform(self):
        """Test that a single-qubit gate changes correctly with a SWAP."""

        def qfunc():
            qml.Hadamard(wires=0)
            qml.PauliX(wires=1)
            qml.SWAP(wires=[0, 1])
            return qml.probs(1)

        transformed_qfunc = undo_swaps(qfunc)

        tape = qml.transforms.make_tape(transformed_qfunc)()
        res = qml.device("default.qubit", wires=2).execute(tape)
        assert len(tape.operations) == 2
        assert np.allclose(res[0][0], 0.5)

    def test_two_qubits_gates_transform(self):
        """Test that a two-qubit gate changes correctly with a SWAP."""

        def qfunc():
            qml.PauliX(wires=1)
            qml.CNOT(wires=[0, 1])
            qml.SWAP(wires=[0, 1])
            return qml.state()

        transformed_qfunc = undo_swaps(qfunc)

        tape = qml.transforms.make_tape(transformed_qfunc)()
        res = qml.device("default.qubit", wires=2).execute(tape)
        assert len(tape.operations) == 2
        assert np.allclose(res[0][2], 1.0)

    def test_templates_transform(self):
        """Test that a template changes correctly with a SWAP."""

        def qfunc1():
            qml.RX(2, wires=0)
            qml.RY(-3, wires=1)
            qml.QFT(wires=[0, 1, 2])
            qml.SWAP(wires=[1, 2])
            return qml.state()

        def qfunc2():
            qml.RX(2, wires=0)
            qml.RY(-3, wires=2)
            qml.QFT(wires=[0, 2, 1])
            return qml.state()

        transformed_qfunc = undo_swaps(qfunc1)

        tape1 = qml.transforms.make_tape(transformed_qfunc)()
        res1 = qml.device("default.qubit", wires=3).execute(tape1)

        tape2 = qml.transforms.make_tape(qfunc2)()
        res2 = qml.device("default.qubit", wires=3).execute(tape2)

        assert np.allclose(res1, res2)

    def test_multi_swaps(self):
        """Test that transform works with several SWAPs."""

        def qfunc1():
            qml.Hadamard(wires=0)
            qml.PauliX(wires=1)
            qml.SWAP(wires=[0, 1])
            qml.SWAP(wires=[0, 2])
            qml.PauliY(wires=0)
            return qml.expval(qml.PauliZ(0))

        def qfunc2():
            qml.Hadamard(wires=1)
            qml.PauliX(wires=2)
            qml.PauliY(wires=0)
            return qml.expval(qml.PauliZ(0))

        transformed_qfunc = undo_swaps(qfunc1)

        tape1 = qml.transforms.make_tape(transformed_qfunc)()
        res1 = qml.device("default.qubit", wires=3).execute(tape1)

        tape2 = qml.transforms.make_tape(qfunc2)()
        res2 = qml.device("default.qubit", wires=3).execute(tape2)

        assert np.allclose(res1, res2)

    def test_decorator(self):

        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev)
        @undo_swaps
        def qfunc():
            qml.Hadamard(wires=0)
            qml.PauliX(wires=1)
            qml.SWAP(wires=[0, 1])
            qml.SWAP(wires=[0, 2])
            qml.PauliY(wires=0)
            return qml.expval(qml.PauliZ(0))

        qfunc()
        assert len(qfunc.qtape.operations) == 3


# Example QNode and device for interface testing
dev = qml.device("default.qubit", wires=3)

# Test each of single-qubit, two-qubit, and Rot gates
def qfunc(theta):
    qml.Hadamard(wires=0)
    qml.RX(theta[0], wires=1)
    qml.SWAP(wires=[0, 1])
    qml.SWAP(wires=[0, 2])
    qml.RY(theta[1], wires=0)
    return qml.expval(qml.PauliZ(0))


transformed_qfunc = undo_swaps(qfunc)

expected_op_list = ["Hadamard", "RX", "RY"]
expected_wires_list = [
    Wires(1),
    Wires(2),
    Wires(0),
]


class TestUndoSwapsInterfaces:
    """Test that `undo_swaps` transform works in all interfaces."""

    @pytest.mark.autograd
    def test_undo_swaps_autograd(self):
        """Test QNode and gradient in autograd interface."""

        original_qnode = qml.QNode(qfunc, dev)
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
        compare_operation_lists(ops, expected_op_list, expected_wires_list)

    @pytest.mark.torch
    def test_undo_swaps_torch(self):
        """Test QNode and gradient in torch interface."""
        import torch

        original_qnode = qml.QNode(qfunc, dev, interface="torch")
        transformed_qnode = qml.QNode(transformed_qfunc, dev, interface="torch")

        original_input = torch.tensor([0.1, 0.2], requires_grad=True)
        transformed_input = torch.tensor([0.1, 0.2], requires_grad=True)

        original_result = original_qnode(original_input)
        transformed_result = transformed_qnode(transformed_input)

        # Check that the numerical output is the same
        assert qml.math.allclose(original_result, transformed_result.detach().numpy())

        # Check that the gradient is the same
        original_result.backward()
        transformed_result.backward()

        assert qml.math.allclose(original_input.grad, transformed_input.grad)

        # Check operation list
        ops = transformed_qnode.qtape.operations
        compare_operation_lists(ops, expected_op_list, expected_wires_list)

    @pytest.mark.tf
    def test_undo_swaps_tf(self):
        """Test QNode and gradient in tensorflow interface."""
        import tensorflow as tf

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

    @pytest.mark.jax
    def test_undo_swaps_jax(self):
        """Test QNode and gradient in JAX interface."""
        import jax
        from jax import numpy as jnp

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
