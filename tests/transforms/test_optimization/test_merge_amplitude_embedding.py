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
Unit tests for the optimization transform ``merge_amplitude_embedding``.
"""

import pytest

import pennylane as qp
from pennylane import numpy as np
from pennylane.exceptions import DeviceError
from pennylane.transforms.optimization import merge_amplitude_embedding


class TestMergeAmplitudeEmbedding:
    """Test that amplitude embedding gates are combined into a single."""

    def test_multi_amplitude_embedding(self):
        """Test that the transformation is working correctly by joining two AmplitudeEmbedding."""

        def qfunc():
            qp.AmplitudeEmbedding([0.0, 1.0], wires=0)
            qp.AmplitudeEmbedding([0.0, 1.0], wires=1)
            qp.Hadamard(wires=0)
            qp.Hadamard(wires=0)
            return qp.state()

        transformed_qfunc = merge_amplitude_embedding(qfunc)
        ops = qp.tape.make_qscript(transformed_qfunc)().operations

        assert len(ops) == 3

        # Check that the solution is as expected.
        dev = qp.device("default.qubit", wires=2)
        assert np.allclose(qp.QNode(transformed_qfunc, dev)()[-1], 1)

    def test_multi_amplitude_embedding_qnode(self):
        """Test that the transformation is working correctly by joining two AmplitudeEmbedding."""

        dev = qp.device("default.qubit", wires=2)

        @merge_amplitude_embedding
        @qp.qnode(device=dev)
        def circuit():
            qp.AmplitudeEmbedding([0.0, 1.0], wires=0)
            qp.AmplitudeEmbedding([0.0, 1.0], wires=1)
            qp.Hadamard(wires=0)
            qp.Hadamard(wires=1)
            return qp.state()

        assert qp.math.allclose(circuit(), np.array([1, -1, -1, 1]) / 2)

    def test_repeated_qubit(self):
        """Check that AmplitudeEmbedding cannot be applied if the qubit has already been used."""

        def qfunc():
            qp.CNOT(wires=[0.0, 1.0])
            qp.AmplitudeEmbedding([0.0, 1.0], wires=1)

        transformed_qfunc = merge_amplitude_embedding(qfunc)

        dev = qp.device("default.qubit", wires=2)
        qnode = qp.QNode(transformed_qfunc, dev)

        with pytest.raises(DeviceError, match="applied in the same qubit"):
            qnode()

    def test_decorator(self):
        """Check that the decorator works."""

        @merge_amplitude_embedding
        def qfunc():
            qp.AmplitudeEmbedding([0, 1, 0, 0], wires=[0, 1])
            qp.AmplitudeEmbedding([0, 1], wires=2)

            return qp.state()

        dev = qp.device("default.qubit", wires=3)
        qnode = qp.QNode(qfunc, dev)
        assert qnode()[3] == 1.0

    def test_broadcasting(self):
        """Test that merging preserves the batch dimension"""
        dev = qp.device("default.qubit", wires=3)

        @qp.transforms.merge_amplitude_embedding
        @qp.qnode(dev)
        def qnode():
            qp.AmplitudeEmbedding([[1, 0], [0, 1]], wires=0)
            qp.AmplitudeEmbedding([1, 0], wires=1)
            qp.AmplitudeEmbedding([[0, 1], [1, 0]], wires=2)
            return qp.state()

        res = qnode()
        tape = qp.workflow.construct_tape(qnode)()
        assert tape.batch_size == 2

        # |001> and |100>
        expected = np.array([[0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0]])
        assert np.allclose(res, expected)


class TestMergeAmplitudeEmbeddingInterfaces:
    """Test that merging amplitude embedding operations works in all interfaces."""

    @pytest.mark.autograd
    def test_merge_amplitude_embedding_autograd(self):
        """Test QNode in autograd interface."""

        def qfunc(amplitude):
            qp.AmplitudeEmbedding(amplitude, wires=0)
            qp.AmplitudeEmbedding(amplitude, wires=1)
            return qp.state()

        dev = qp.device("default.qubit", wires=2)
        optimized_qfunc = qp.transforms.merge_amplitude_embedding(qfunc)
        optimized_qnode = qp.QNode(optimized_qfunc, dev)

        amplitude = np.array([0.0, 1.0], requires_grad=True)
        # Check the state |11> is being generated.
        assert optimized_qnode(amplitude)[-1] == 1

    @pytest.mark.torch
    def test_merge_amplitude_embedding_torch(self):
        """Test QNode in torch interface."""
        import torch

        def qfunc(amplitude):
            qp.AmplitudeEmbedding(amplitude, wires=0)
            qp.AmplitudeEmbedding(amplitude, wires=1)
            return qp.state()

        dev = qp.device("default.qubit", wires=2)
        optimized_qfunc = qp.transforms.merge_amplitude_embedding(qfunc)
        optimized_qnode = qp.QNode(optimized_qfunc, dev)

        amplitude = torch.tensor([0.0, 1.0], requires_grad=True)
        # Check the state |11> is being generated.
        assert optimized_qnode(amplitude)[-1] == 1

    @pytest.mark.tf
    def test_merge_amplitude_embedding_tf(self):
        """Test QNode in tensorflow interface."""
        import tensorflow as tf

        def qfunc(amplitude):
            qp.AmplitudeEmbedding(amplitude, wires=0)
            qp.AmplitudeEmbedding(amplitude, wires=1)
            return qp.state()

        dev = qp.device("default.qubit", wires=2)
        optimized_qfunc = qp.transforms.merge_amplitude_embedding(qfunc)
        optimized_qnode = qp.QNode(optimized_qfunc, dev)

        amplitude = tf.Variable([0.0, 1.0])
        # Check the state |11> is being generated.
        assert optimized_qnode(amplitude)[-1] == 1

    @pytest.mark.jax
    def test_merge_amplitude_embedding_jax(self):
        """Test QNode in JAX interface."""
        from jax import numpy as jnp

        def qfunc(amplitude):
            qp.AmplitudeEmbedding(amplitude, wires=0)
            qp.AmplitudeEmbedding(amplitude, wires=1)
            return qp.state()

        dev = qp.device("default.qubit", wires=2)
        optimized_qfunc = qp.transforms.merge_amplitude_embedding(qfunc)
        optimized_qnode = qp.QNode(optimized_qfunc, dev)

        amplitude = jnp.array([0.0, 1.0], dtype=jnp.float64)
        # Check the state |11> is being generated.
        assert optimized_qnode(amplitude)[-1] == 1
