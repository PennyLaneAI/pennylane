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
from pennylane.transforms.optimization import merge_amplitude_embedding
from pennylane._device import DeviceError


class TestMergeAmplitudeEmbedding:
    """Test that amplitude embedding gates are combined into a single."""


    def test_multi_amplitude_embedding(self):
        """Test that the transformation is working correctly by joining two amplitudeEmbedding."""

        def qfunc():
            qml.AmplitudeEmbedding([0,1], wires=0)
            qml.AmplitudeEmbedding([0,1], wires=1)
            qml.state()

        transformed_qfunc = merge_amplitude_embedding(qfunc)
        ops = qml.transforms.make_tape(transformed_qfunc)().operations

        assert len(ops) == 1

        # Check that the solution is as expected.

        dev = qml.device("default.qubit", wires = 2)
        assert qml.QNode(transformed_qfunc, dev)()[-1] == 1


    def test_repeated_qubit(self):
        """Check that AmplitudeEmbedding cannot be applied if the qubit has already been used."""

        def qfunc():
            qml.CNOT(wires = [0,1])
            qml.AmplitudeEmbedding([0,1], wires=1)

        transformed_qfunc = merge_amplitude_embedding(qfunc)
        ops = qml.transforms.make_tape(transformed_qfunc)().operations

        dev = qml.device("default.qubit", wires=2)
        qnode = qml.QNode(transformed_qfunc, )

        with pytest.raises(DeviceError, match="applied in the same qubit"):
            qnode()


# Example QNode and device for interface testing
dev = qml.device("default.qubit", wires=3)

# Test each of single-qubit, two-qubit, and Rot gates
def qfunc(amplitude):
    qml.Hadamard(wires = 2)
    qml.AmplitudeEmbedding(amplitude, wires=0)
    qml.AmplitudeEmbedding(amplitude, wires=1)
    qml.CNOT(wires = [0,1])
    return qml.expval(qml.PauliX(0) @ qml.PauliX(2))


transformed_qfunc = merge_amplitude_embedding(qfunc)

expected_op_list = ["Hadamard", "AmplitudeEmbedding", "CNOT"]
expected_wires_list = [
    Wires(2),
    Wires([0,1]),
    Wires([0,1]),
]

'''
class TestMergeRotationsInterfaces:
    """Test that rotation merging works in all interfaces."""

    def test_merge_rotations_autograd(self):
        """Test QNode and gradient in autograd interface."""

        original_qnode = qml.QNode(qfunc, dev)
        transformed_qnode = qml.QNode(transformed_qfunc, dev)

        input = np.array([0.1, 0.2, 0.3, 0.4], requires_grad=True)

        # Check that the numerical output is the same
        assert qml.math.allclose(original_qnode(input), transformed_qnode(input))

        # Check that the gradient is the same
        assert qml.math.allclose(
            qml.grad(original_qnode)(input), qml.grad(transformed_qnode)(input)
        )

        # Check operation list
        ops = transformed_qnode.qtape.operations
        compare_operation_lists(ops, expected_op_list, expected_wires_list)

    def test_merge_rotations_torch(self):
        """Test QNode and gradient in torch interface."""
        torch = pytest.importorskip("torch", minversion="1.8")

        original_qnode = qml.QNode(qfunc, dev, interface="torch")
        transformed_qnode = qml.QNode(transformed_qfunc, dev, interface="torch")

        original_input = torch.tensor([0.1, 0.2, 0.3, 0.4], requires_grad=True)
        transformed_input = torch.tensor([0.1, 0.2, 0.3, 0.4], requires_grad=True)

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

    def test_merge_rotations_tf(self):
        """Test QNode and gradient in tensorflow interface."""
        tf = pytest.importorskip("tensorflow")

        original_qnode = qml.QNode(qfunc, dev, interface="tf")
        transformed_qnode = qml.QNode(transformed_qfunc, dev, interface="tf")

        original_input = tf.Variable([0.1, 0.2, 0.3, 0.4])
        transformed_input = tf.Variable([0.1, 0.2, 0.3, 0.4])

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

    def test_merge_rotations_jax(self):
        """Test QNode and gradient in JAX interface."""
        jax = pytest.importorskip("jax")
        from jax import numpy as jnp

        from jax.config import config

        remember = config.read("jax_enable_x64")
        config.update("jax_enable_x64", True)

        original_qnode = qml.QNode(qfunc, dev, interface="jax")
        transformed_qnode = qml.QNode(transformed_qfunc, dev, interface="jax")

        input = jnp.array([0.1, 0.2, 0.3, 0.4], dtype=jnp.float64)

        # Check that the numerical output is the same
        assert qml.math.allclose(original_qnode(input), transformed_qnode(input))

        # Check that the gradient is the same
        assert qml.math.allclose(
            jax.grad(original_qnode)(input), jax.grad(transformed_qnode)(input)
        )

        # Check operation list
        ops = transformed_qnode.qtape.operations
        compare_operation_lists(ops, expected_op_list, expected_wires_list)
'''
