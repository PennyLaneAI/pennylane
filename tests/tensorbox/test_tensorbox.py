# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for TensorBox subclasses"""
import numpy as np
import pytest

import pennylane as qml
from pennylane.tensorbox.numpy_box import NumpyBox


class TestNumpyBox:
    """Tests for the NumpyBox class"""

    def test_creation_from_list(self):
        """Test that a NumpyBox is automatically created from a list"""
        x = [0.1, 0.2, 0.3]
        res = qml.TensorBox(x)
        assert isinstance(res, NumpyBox)
        assert res.interface == "numpy"
        assert isinstance(res.unbox(), np.ndarray)
        assert np.all(res.unbox() == x)

    def test_creation_from_tuple(self):
        """Test that a NumpyBox is automatically created from a tuple"""
        x = (0.1, 0.2, 0.3)
        res = qml.TensorBox(x)
        assert isinstance(res, NumpyBox)
        assert res.interface == "numpy"
        assert isinstance(res.unbox(), np.ndarray)
        assert np.all(res.unbox() == x)

    def test_creation_from_tensorbox(self):
        """Test that a tensorbox input simply returns itself"""
        x = qml.TensorBox(np.array([0.1, 0.2, 0.3]))
        res = qml.TensorBox(x)
        assert x is res

    def test_unknown_input_type(self):
        """Test that an exception is raised if the input type
        is unknown"""
        with pytest.raises(ValueError, match="Unknown tensor type"):
            qml.TensorBox(True)

    def test_astensor(self):
        """Test conversion of sequences to numpy arrays"""
        x = np.array([0.1, 0.2, 0.3])
        y = [0.4, 0.5, 0.6]

        res = qml.TensorBox(x).astensor(y)
        assert isinstance(res, np.ndarray)
        assert np.all(res == y)

    def test_len(self):
        """Test length"""
        x = np.array([[1, 2], [3, 4]])
        res = qml.TensorBox(x)
        assert len(res) == len(x) == 2

    def test_ufunc_compatibility(self):
        """Test that the NumpyBox class has ufunc compatibility"""
        x = np.array([0.1, 0.2, 0.3])
        res = np.sum(np.sin(qml.TensorBox(x)))
        assert res == np.sin(0.1) + np.sin(0.2) + np.sin(0.3)

    def test_inplace_addition(self):
        """Test that in-place addition works correctly"""
        x = qml.TensorBox(np.array([0., 0., 0.]))
        np.add.at(x, [0, 1, 1], 1)
        assert np.all(x == np.array([1., 2., 0.]))

    def test_multiplication(self):
        """Test multiplication between tensors and arrays"""
        x = np.array([[1, 2], [3, 4]])
        y = np.array([[1, 0], [0, 1]])

        xT = qml.TensorBox(x)
        res = xT * y
        assert np.all(res.unbox() == x * y)

        yT = qml.TensorBox(y)
        res = x * yT
        assert np.all(res.unbox() == x * y)

        res = xT * yT
        assert np.all(res.unbox() == x * y)

    def test_exponentiation(self):
        """Test exponentiation between tensors and arrays"""
        x = np.array([[1, 2], [3, 4]])
        y = np.array([[1, 0], [0, 1]])

        xT = qml.TensorBox(x)
        res = xT ** 2
        assert np.all(res.unbox() == x ** 2)

        yT = qml.TensorBox(y)
        res = 2 ** yT
        assert np.all(res.unbox() == 2 ** y)

        res = xT ** yT
        assert np.all(res.unbox() == x ** y)

    def test_unbox_list(self):
        """Test unboxing a mixed list works correctly"""
        x = np.array([[1, 2], [3, 4]])
        y = np.array([[1, 0], [0, 1]])

        xT = qml.TensorBox(x)
        res = xT.unbox_list([y, xT, x])

        assert np.all(res == [y, x, x])

    def test_numpy(self):
        """Test that calling numpy() returns a NumPy array representation
        of the TensorBox"""
        x = np.array([[1, 2], [3, 4]])
        xT = qml.TensorBox(x)
        assert isinstance(xT.numpy(), np.ndarray)
        assert np.all(xT.numpy() == x)

    def test_shape(self):
        """Test that arrays return the right shape"""
        x = np.array([[[1, 2], [3, 4]]])
        x = qml.TensorBox(x)
        res = x.shape
        assert res == (1, 2, 2)

    def test_expand_dims(self):
        """Test that dimension expansion works"""
        x = np.array([1, 2, 3])
        xT = qml.TensorBox(x)

        res = xT.expand_dims(axis=1)
        expected = np.expand_dims(x, axis=1)
        assert isinstance(res, NumpyBox)
        assert np.all(res == expected)

    def test_ones_like(self):
        """Test that all ones arrays are correctly created"""
        x = np.array([[1, 2, 3], [4, 5, 6]])
        xT = qml.TensorBox(x)

        res = xT.ones_like()
        expected = np.ones_like(x)
        assert isinstance(res, NumpyBox)
        assert np.all(res == expected)

    def test_stack(self):
        """Test that arrays are correctly stacked together"""
        x = np.array([[1, 2], [3, 4]])
        y = np.array([[1, 0], [0, 1]])

        xT = qml.TensorBox(x)
        res = xT.stack([y, xT, x])

        assert np.all(res == np.stack([y, x, x]))

    def test_transpose(self):
        """Test that the transpose is correct"""
        x = np.array([[1, 2], [3, 4]])
        xT = qml.TensorBox(x)

        assert np.all(xT.T == x.T)

    def test_requires_grad(self):
        """Test that the requires grad attribute always returns False"""
        x = np.array([[1, 2], [3, 4]])
        xT = qml.TensorBox(x)
        assert not xT.requires_grad


class TestAutogradBox:
    """Tests for the AutogradBox class"""

    @pytest.fixture
    def autograd(self):
        autograd = pytest.importorskip("autograd")
        return autograd

    def test_creation(self, autograd):
        """Test that a AutogradBox is automatically created from a PennyLane numpy tensor"""
        from pennylane.tensorbox.autograd_box import AutogradBox

        x = qml.numpy.array([0.1, 0.2, 0.3])
        res = qml.TensorBox(x)
        assert isinstance(res, AutogradBox)
        assert res.interface == "autograd"
        assert isinstance(res.unbox(), np.ndarray)
        assert np.all(res == x)

    def test_astensor_list(self, autograd):
        """Test conversion of a list to PennyLane tensors"""
        x = qml.numpy.array([0.1, 0.2, 0.3])
        y = [0.4, 0.5, 0.6]

        res = qml.TensorBox(x).astensor(y)
        assert isinstance(res, qml.numpy.tensor)
        assert np.all(res == y)

    def test_astensor_array(self, autograd):
        """Test conversion of numpy arrays to PennyLane tensors"""
        x = qml.numpy.array([0.1, 0.2, 0.3])
        y = np.array([0.4, 0.5, 0.6])

        res = qml.TensorBox(x).astensor(y)
        assert isinstance(res, qml.numpy.tensor)
        assert np.all(res == y)

    def test_len(self, autograd):
        """Test length"""
        np = qml.numpy
        x = np.array([[1, 2], [3, 4]])
        res = qml.TensorBox(x)
        assert len(res) == len(x) == 2

    def test_ufunc_compatibility(self, autograd):
        """Test that the AutogradBox class has ufunc compatibility"""
        np = qml.numpy
        x = np.array([0.1, 0.2, 0.3])
        res = np.sum(np.sin(qml.TensorBox(x)))
        assert res.unbox().item() == np.sin(0.1) + np.sin(0.2) + np.sin(0.3)

    def test_multiplication(self, autograd):
        """Test multiplication between tensors and arrays"""
        np = qml.numpy
        x = np.array([[1, 2], [3, 4]])
        y = np.array([[1, 0], [0, 1]])

        xT = qml.TensorBox(x)
        res = xT * y
        assert np.all(res == x * y)

        yT = qml.TensorBox(y)
        res = x * yT
        assert np.all(res == x * y)

        res = xT * yT
        assert np.all(res == x * y)

    def test_unbox_list(self, autograd):
        """Test unboxing a mixed list works correctly"""
        np = qml.numpy
        x = np.array([[1, 2], [3, 4]])
        y = np.array([[1, 0], [0, 1]])

        xT = qml.TensorBox(x)
        res = xT.unbox_list([y, xT, x])

        assert np.all(res == [y, x, x])

    def test_numpy(self, autograd):
        """Test that calling numpy() returns a NumPy array representation
        of the TensorBox"""
        x = qml.numpy.array([[1, 2], [3, 4]])
        xT = qml.TensorBox(x)
        assert isinstance(xT.numpy(), np.ndarray)
        assert not isinstance(xT.numpy(), qml.numpy.tensor)
        assert np.all(xT == x)

    def test_shape(self, autograd):
        """Test that arrays return the right shape"""
        np = qml.numpy
        x = np.array([[[1, 2], [3, 4]]])
        x = qml.TensorBox(x)
        res = x.shape
        assert res == (1, 2, 2)

    def test_expand_dims(self, autograd):
        """Test that dimension expansion works"""
        from pennylane.tensorbox.autograd_box import AutogradBox

        np = qml.numpy
        x = np.array([1, 2, 3])
        xT = qml.TensorBox(x)

        res = xT.expand_dims(axis=1)
        expected = np.expand_dims(x, axis=1)
        assert isinstance(res, AutogradBox)
        assert np.all(res == expected)

    def test_ones_like(self, autograd):
        """Test that all ones arrays are correctly created"""
        from pennylane.tensorbox.autograd_box import AutogradBox

        np = qml.numpy
        x = np.array([[1, 2, 3], [4, 5, 6]])
        xT = qml.TensorBox(x)

        res = xT.ones_like()
        expected = np.ones_like(x)
        assert isinstance(res, AutogradBox)
        assert np.all(res == expected)

    def test_stack(self, autograd):
        """Test that arrays are correctly stacked together"""
        np = qml.numpy
        x = np.array([[1, 2], [3, 4]])
        y = np.array([[1, 0], [0, 1]])

        xT = qml.TensorBox(x)
        res = xT.stack([y, xT, x])

        assert np.all(res == np.stack([y, x, x]))

    def test_transpose(self, autograd):
        """Test that the transpose is correct"""
        np = qml.numpy
        x = np.array([[1, 2], [3, 4]])
        xT = qml.TensorBox(x)

        assert np.all(xT.T == x.T)

    def test_autodifferentiation(self, autograd):
        """Test that autodifferentiation is preserved when writing
        a cost function that uses TensorBox method chaining"""
        np = qml.numpy
        x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        cost_fn = lambda a: (qml.TensorBox(a).T ** 2).unbox()[0, 1]
        grad_fn = qml.grad(cost_fn)

        res = grad_fn(x)[0]
        expected = np.array([[0.0, 0.0, 0.0], [8.0, 0.0, 0.0]])
        assert np.all(res == expected)

    def test_requires_grad(self, autograd):
        """Test that the requires grad attribute matches the underlying tensor"""
        np = qml.numpy
        x = np.array([[1, 2], [3, 4]], requires_grad=False)
        xT = qml.TensorBox(x)
        assert not xT.requires_grad

        x = np.array([[1, 2], [3, 4]], requires_grad=True)
        xT = qml.TensorBox(x)
        assert xT.requires_grad


class TestTorchBox:
    """Tests for the TorchBox class"""

    @pytest.fixture
    def torch(self):
        torch = pytest.importorskip("torch")
        return torch

    def test_creation(self, torch):
        """Test that a TorchBox is automatically created from a torch tensor"""
        from pennylane.tensorbox.torch_box import TorchBox

        x = torch.tensor([0.1, 0.2, 0.3])
        res = qml.TensorBox(x)

        assert isinstance(res, TorchBox)
        assert res.interface == "torch"
        assert isinstance(res.unbox(), torch.Tensor)
        assert torch.allclose(res.unbox(), x)

    def test_astensor_list(self, torch):
        """Test conversion of numpy arrays to torch tensors"""
        x = torch.tensor([0.1, 0.2, 0.3])
        y = [0.4, 0.5, 0.6]

        res = qml.TensorBox(x).astensor(y)
        assert isinstance(res, torch.Tensor)
        assert np.all(res.numpy() == np.array(y, dtype=np.float32))

    def test_astensor_array(self, torch):
        """Test conversion of numpy arrays to torch tensors"""
        x = torch.tensor([0.1, 0.2, 0.3])
        y = np.array([0.4, 0.5, 0.6])

        res = qml.TensorBox(x).astensor(y)
        assert isinstance(res, torch.Tensor)
        assert np.all(res.numpy() == y)

    def test_len(self, torch):
        """Test length"""
        x = torch.tensor([[1, 2], [3, 4]])
        res = qml.TensorBox(x)
        assert len(res) == len(x) == 2

    def test_multiplication(self, torch):
        """Test multiplication between tensors and arrays"""
        x = torch.tensor([[1, 2], [3, 4]])
        y = torch.tensor([[1, 0], [0, 1]])

        xT = qml.TensorBox(x)
        res = xT * y
        assert torch.allclose(res.unbox(), x * y)

        yT = qml.TensorBox(y)
        res = x * yT
        assert torch.allclose(res.unbox(), x * y)

        res = xT * yT
        assert torch.allclose(res.unbox(), x * y)

    def test_unbox_list(self, torch):
        """Test unboxing a mixed list works correctly"""
        x = torch.tensor([[1, 2], [3, 4]])
        y = torch.tensor([[1, 0], [0, 1]])

        xT = qml.TensorBox(x)
        res = xT.unbox_list([y, xT, x])

        assert np.all(res == [y, x, x])

    def test_numpy(self, torch):
        """Test that calling numpy() returns a NumPy array representation
        of the TensorBox"""
        x = torch.tensor([[1, 2], [3, 4]])
        xT = qml.TensorBox(x)
        assert isinstance(xT.numpy(), np.ndarray)
        assert not isinstance(xT.numpy(), torch.Tensor)
        assert np.all(xT.numpy() == x.numpy())

    def test_shape(self, torch):
        """Test that arrays return the right shape"""
        x = torch.tensor([[[1, 2], [3, 4]]])
        x = qml.TensorBox(x)
        res = x.shape
        assert res == (1, 2, 2)

    def test_expand_dims(self, torch):
        """Test that dimension expansion works"""
        from pennylane.tensorbox.torch_box import TorchBox

        x = torch.tensor([1, 2, 3])
        xT = qml.TensorBox(x)

        res = xT.expand_dims(axis=1)
        expected = torch.unsqueeze(x, dim=1)
        assert isinstance(res, TorchBox)
        assert torch.allclose(res.unbox(), expected)

    def test_ones_like(self, torch):
        """Test that all ones arrays are correctly created"""
        from pennylane.tensorbox.torch_box import TorchBox

        x = torch.tensor([[1, 2, 3], [4, 5, 6]])
        xT = qml.TensorBox(x)

        res = xT.ones_like()
        expected = torch.ones_like(x)
        assert isinstance(res, TorchBox)
        assert torch.allclose(res.unbox(), expected)

    def test_stack(self, torch):
        """Test that arrays are correctly stacked together"""
        x = torch.tensor([[1, 2], [3, 4]])
        y = torch.tensor([[1, 0], [0, 1]])

        xT = qml.TensorBox(x)
        res = xT.stack([y, xT, x])

        assert torch.allclose(res.unbox(), torch.stack([y, x, x]))

    def test_transpose(self, torch):
        """Test that the transpose is correct"""
        x = torch.tensor([[1, 2], [3, 4]])
        xT = qml.TensorBox(x)

        assert torch.allclose(xT.T.unbox(), x.T)

    def test_autodifferentiation(self, torch):
        """Test that autodifferentiation is preserved when writing
        a cost function that uses TensorBox method chaining"""
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
        y = (qml.TensorBox(x).T ** 2).unbox()[0, 1]
        y.backward()

        res = x.grad
        expected = torch.tensor([[0.0, 0.0, 0.0], [8.0, 0.0, 0.0]])
        assert torch.allclose(res, expected)

    def test_requires_grad(self, torch):
        """Test that the requires grad attribute matches the underlying tensor"""
        x = torch.tensor([[1., 2], [3, 4]], requires_grad=False)
        xT = qml.TensorBox(x)
        assert not xT.requires_grad

        x = torch.tensor([[1., 2], [3, 4]], requires_grad=True)
        xT = qml.TensorBox(x)
        assert xT.requires_grad


class TestTensorFlowBox:
    """Tests for the TensorFlowBox class"""

    @pytest.fixture
    def tf(self):
        tf = pytest.importorskip("tensorflow", minversion="2.0")
        return tf

    def test_should_record_import(self, tf, mocker, monkeypatch):
        """In TensorFlow 2.1, the should_record_backprop function was named
        should_record. Test that the module correctly imports this function
        if available."""
        should_record = tf.python.eager.tape.should_record_backprop
        mocker.patch("tensorflow.python.eager.tape.should_record", should_record, create=True)
        monkeypatch.delattr(tf.python.eager.tape, "should_record_backprop")
        from pennylane.tensorbox.tf_box import TensorFlowBox

        x = tf.Variable([0.1, 0.2, 0.3])
        res = qml.TensorBox(x)
        assert isinstance(res, TensorFlowBox)

    def test_creation(self, tf):
        """Test that a TensorFlowBox is automatically created from a tf tensor"""
        from pennylane.tensorbox.tf_box import TensorFlowBox

        x = tf.Variable([0.1, 0.2, 0.3])
        res = qml.TensorBox(x)
        assert isinstance(res, TensorFlowBox)
        assert res.interface == "tf"
        assert isinstance(res.unbox(), tf.Variable)
        assert np.all(res.unbox() == x)

    def test_astensor_list(self, tf):
        """Test conversion of numpy arrays to tf tensors"""
        x = tf.Variable([0.1, 0.2, 0.3])
        y = [0.4, 0.5, 0.6]

        res = qml.TensorBox(x).astensor(y)
        assert isinstance(res, tf.Tensor)
        assert np.all(res.numpy() == np.asarray(y, dtype=np.float32))

    def test_astensor_array(self, tf):
        """Test conversion of numpy arrays to tf tensors"""
        x = tf.Variable([0.1, 0.2, 0.3])
        y = np.array([0.4, 0.5, 0.6])

        res = qml.TensorBox(x).astensor(y)
        assert isinstance(res, tf.Tensor)
        assert np.all(res.numpy() == y)

    def test_len(self, tf):
        """Test length"""
        x = tf.Variable([[1, 2], [3, 4]])
        res = qml.TensorBox(x)
        assert len(res) == len(x.numpy()) == 2

        x = tf.constant([[1, 2], [3, 4], [5, 6]])
        res = qml.TensorBox(x)
        assert len(res) == len(x.numpy()) == 3

    def test_multiplication(self, tf):
        """Test multiplication between tensors and arrays"""
        x = tf.Variable([[1, 2], [3, 4]])
        y = tf.Variable([[1, 0], [0, 1]])

        xT = qml.TensorBox(x)
        res = xT * y
        assert np.allclose(res.unbox(), x * y)

        yT = qml.TensorBox(y)
        res = x * yT
        assert np.allclose(res.unbox(), x * y)

        res = xT * yT
        assert np.allclose(res.unbox(), x * y)

    def test_unbox_list(self, tf):
        """Test unboxing a mixed list works correctly"""
        x = tf.Variable([[1, 2], [3, 4]])
        y = tf.Variable([[1, 0], [0, 1]])

        xT = qml.TensorBox(x)
        res = xT.unbox_list([y, xT, x])

        assert np.all(res == [y, x, x])

    def test_numpy(self, tf):
        """Test that calling numpy() returns a NumPy array representation
        of the TensorBox"""
        x = tf.Variable([[1, 2], [3, 4]])
        xT = qml.TensorBox(x)
        assert isinstance(xT.numpy(), np.ndarray)
        assert not isinstance(xT.numpy(), tf.Variable)
        assert np.all(xT.numpy() == x.numpy())

    def test_shape(self, tf):
        """Test that arrays return the right shape"""
        x = tf.Variable([[[1, 2], [3, 4]]])
        x = qml.TensorBox(x)
        res = x.shape
        assert res == (1, 2, 2)

    def test_expand_dims(self, tf):
        """Test that dimension expansion works"""
        from pennylane.tensorbox.tf_box import TensorFlowBox

        x = tf.Variable([1, 2, 3])
        xT = qml.TensorBox(x)

        res = xT.expand_dims(axis=1)
        expected = tf.expand_dims(x, 1)
        assert isinstance(res, TensorFlowBox)
        assert np.allclose(res.unbox(), expected)

    def test_ones_like(self, tf):
        """Test that all ones arrays are correctly created"""
        from pennylane.tensorbox.tf_box import TensorFlowBox

        x = tf.Variable([[1, 2, 3], [4, 5, 6]])
        xT = qml.TensorBox(x)

        res = xT.ones_like()
        expected = tf.ones_like(x)
        assert isinstance(res, TensorFlowBox)
        assert np.allclose(res.unbox(), expected)

    def test_stack(self, tf):
        """Test that arrays are correctly stacked together"""
        x = tf.Variable([[1, 2], [3, 4]])
        y = tf.Variable([[1, 0], [0, 1]])

        xT = qml.TensorBox(x)
        res = xT.stack([y, xT, x])

        assert np.allclose(res.unbox(), tf.stack([y, x, x]))

    def test_transpose(self, tf):
        """Test that the transpose is correct"""
        x = tf.Variable([[1, 2], [3, 4]])
        xT = qml.TensorBox(x)

        assert np.allclose(xT.T.unbox(), tf.transpose(x))

    def test_autodifferentiation(self, tf):
        """Test that autodifferentiation is preserved when writing
        a cost function that uses TensorBox method chaining"""
        x = tf.Variable([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        with tf.GradientTape() as tape:
            y = (qml.TensorBox(x).T ** 2).unbox()[0, 1]

        res = tape.gradient(y, x)
        expected = tf.constant([[0.0, 0.0, 0.0], [8.0, 0.0, 0.0]])
        assert np.allclose(res, expected)

    def test_requires_grad(self, tf):
        """Test that the requires grad attribute matches the underlying tensor"""
        x = tf.Variable([[1., 2], [3, 4]])
        xT = qml.TensorBox(x)

        # without the gradient tape, TensorFlow treats all variables
        # as non-differentiable
        assert not xT.requires_grad

        # within a gradient tape, it will be trainable
        with tf.GradientTape() as tape:
            assert xT.requires_grad

        # constants are never trainable
        y = tf.constant([[1., 2], [3, 4]])
        yT = qml.TensorBox(y)

        with tf.GradientTape() as tape:
            assert not yT.requires_grad

        # we can watch the constant tensor to make it trainable
        with tf.GradientTape() as tape:
            tape.watch([yT.unbox()])
            assert yT.requires_grad
