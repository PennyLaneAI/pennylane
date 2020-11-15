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

tf = pytest.importorskip("tensorflow", minversion="2.1")

import pennylane as qml
from pennylane.proc.tf_box import TensorFlowBox


def test_creation():
    """Test that a TensorFlowBox is automatically created from a tf tensor"""

    x = tf.Variable([0.1, 0.2, 0.3])
    res = qml.proc.TensorBox(x)
    assert isinstance(res, TensorFlowBox)
    assert res.interface == "tf"
    assert isinstance(res.unbox(), tf.Variable)
    assert np.all(res.unbox() == x)


def test_astensor_list():
    """Test conversion of numpy arrays to tf tensors"""
    x = tf.Variable([0.1, 0.2, 0.3])
    y = [0.4, 0.5, 0.6]

    res = qml.proc.TensorBox(x).astensor(y)
    assert isinstance(res, tf.Tensor)
    assert np.all(res.numpy() == np.asarray(y, dtype=np.float32))


def test_astensor_array():
    """Test conversion of numpy arrays to tf tensors"""
    x = tf.Variable([0.1, 0.2, 0.3])
    y = np.array([0.4, 0.5, 0.6])

    res = qml.proc.TensorBox(x).astensor(y)
    assert isinstance(res, tf.Tensor)
    assert np.all(res.numpy() == y)


def test_cast():
    """Test that arrays can be cast to different dtypes"""
    x = tf.Variable([1, 2, 3])

    res = qml.proc.TensorBox(x).cast(np.float64)
    expected = tf.Variable([1.0, 2.0, 3.0])
    assert np.all(res.numpy() == expected.numpy())

    res = qml.proc.TensorBox(x).cast(np.dtype("int8"))
    expected = tf.Variable([1, 2, 3], dtype=tf.int8)
    assert np.all(res.numpy() == expected.numpy())

    res = qml.proc.TensorBox(x).cast("complex128")
    expected = tf.Variable([1, 2, 3], dtype=tf.complex128)
    assert np.all(res.numpy() == expected.numpy())

    res = qml.proc.TensorBox(x).cast(tf.complex128)
    expected = tf.Variable([1, 2, 3], dtype=tf.complex128)
    assert np.all(res.numpy() == expected.numpy())


def test_len():
    """Test length"""
    x = tf.Variable([[1, 2], [3, 4]])
    res = qml.proc.TensorBox(x)
    assert len(res) == len(x.numpy()) == 2

    x = tf.constant([[1, 2], [3, 4], [5, 6]])
    res = qml.proc.TensorBox(x)
    assert len(res) == len(x.numpy()) == 3


def test_multiplication():
    """Test multiplication between tensors and arrays"""
    x = tf.Variable([[1, 2], [3, 4]])
    y = tf.Variable([[1, 0], [0, 1]])

    xT = qml.proc.TensorBox(x)
    res = xT * y
    assert np.allclose(res.unbox(), x * y)

    yT = qml.proc.TensorBox(y)
    res = x * yT
    assert np.allclose(res.unbox(), x * y)

    res = xT * yT
    assert np.allclose(res.unbox(), x * y)


def test_unbox_list():
    """Test unboxing a mixed list works correctly"""
    x = tf.Variable([[1, 2], [3, 4]])
    y = tf.Variable([[1, 0], [0, 1]])

    xT = qml.proc.TensorBox(x)
    res = xT.unbox_list([y, xT, x])

    assert np.all(res == [y, x, x])


def test_numpy():
    """Test that calling numpy() returns a NumPy array representation
    of the TensorBox"""
    x = tf.Variable([[1, 2], [3, 4]])
    xT = qml.proc.TensorBox(x)
    assert isinstance(xT.numpy(), np.ndarray)
    assert not isinstance(xT.numpy(), tf.Variable)
    assert np.all(xT.numpy() == x.numpy())


def test_shape():
    """Test that arrays return the right shape"""
    x = tf.Variable([[[1, 2], [3, 4]]])
    x = qml.proc.TensorBox(x)
    res = x.shape
    assert res == (1, 2, 2)
    assert np.shape(x) == (1, 2, 2)


def test_expand_dims():
    """Test that dimension expansion works"""
    from pennylane.proc.tf_box import TensorFlowBox

    x = tf.Variable([1, 2, 3])
    xT = qml.proc.TensorBox(x)

    res = xT.expand_dims(axis=1)
    expected = tf.expand_dims(x, 1)
    assert isinstance(res, TensorFlowBox)
    assert np.allclose(res.unbox(), expected)


def test_ones_like():
    """Test that all ones arrays are correctly created"""
    from pennylane.proc.tf_box import TensorFlowBox

    x = tf.Variable([[1, 2, 3], [4, 5, 6]])
    xT = qml.proc.TensorBox(x)

    res = xT.ones_like()
    expected = tf.ones_like(x)
    assert isinstance(res, TensorFlowBox)
    assert np.allclose(res.unbox(), expected)


def test_stack():
    """Test that arrays are correctly stacked together"""
    x = tf.Variable([[1, 2], [3, 4]])
    y = tf.Variable([[1, 0], [0, 1]])

    xT = qml.proc.TensorBox(x)
    res = xT.stack([y, xT, x])

    assert np.allclose(res.unbox(), tf.stack([y, x, x]))


def test_transpose():
    """Test that the transpose is correct"""
    x = tf.Variable([[1, 2], [3, 4]])
    xT = qml.proc.TensorBox(x)

    assert np.allclose(xT.T.unbox(), tf.transpose(x))


def test_autodifferentiation():
    """Test that autodifferentiation is preserved when writing
    a cost function that uses TensorBox method chaining"""
    x = tf.Variable([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    with tf.GradientTape() as tape:
        y = (qml.proc.TensorBox(x).T ** 2).unbox()[0, 1]

    res = tape.gradient(y, x)
    expected = tf.constant([[0.0, 0.0, 0.0], [8.0, 0.0, 0.0]])
    assert np.allclose(res, expected)


def test_requires_grad():
    """Test that the requires grad attribute matches the underlying tensor"""
    x = tf.Variable([[1.0, 2], [3, 4]])
    xT = qml.proc.TensorBox(x)

    # without the gradient tape, TensorFlow treats all variables
    # as non-differentiable
    assert not xT.requires_grad

    # within a gradient tape, it will be trainable
    with tf.GradientTape() as tape:
        assert xT.requires_grad

    # constants are never trainable
    y = tf.constant([[1.0, 2], [3, 4]])
    yT = qml.proc.TensorBox(y)

    with tf.GradientTape() as tape:
        assert not yT.requires_grad

    # we can watch the constant tensor to make it trainable
    with tf.GradientTape() as tape:
        tape.watch([yT.unbox()])
        assert yT.requires_grad
