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
"""Unit tests for AutogradBox subclass"""
import pytest

autograd = pytest.importorskip("autograd")

import pennylane as qml
from pennylane import numpy as np
from pennylane.math.autograd_box import AutogradBox


def test_creation():
    """Test that a AutogradBox is automatically created from a PennyLane numpy tensor"""
    x = np.array([0.1, 0.2, 0.3])
    res = qml.math.TensorBox(x)
    assert isinstance(res, AutogradBox)
    assert res.interface == "autograd"
    assert isinstance(res.unbox(), np.ndarray)
    assert np.all(res == x)


def test_astensor_list():
    """Test conversion of a list to PennyLane tensors"""
    x = np.array([0.1, 0.2, 0.3])
    y = [0.4, 0.5, 0.6]

    res = qml.math.TensorBox(x).astensor(y)
    assert isinstance(res, np.tensor)
    assert np.all(res == y)


def test_astensor_array():
    """Test conversion of numpy arrays to PennyLane tensors"""
    x = np.array([0.1, 0.2, 0.3])
    y = np.array([0.4, 0.5, 0.6])

    res = qml.math.TensorBox(x).astensor(y)
    assert isinstance(res, np.tensor)
    assert np.all(res == y)


def test_cast():
    """Test that arrays can be cast to different dtypes"""
    x = np.array([1, 2, 3])

    res = qml.math.TensorBox(x).cast(np.float64)
    expected = np.array([1.0, 2.0, 3.0])
    assert np.all(res == expected)
    assert res.numpy().dtype.type is np.float64

    res = qml.math.TensorBox(x).cast(np.dtype("int8"))
    expected = np.array([1, 2, 3], dtype=np.int8)
    assert np.all(res == expected)
    assert res.numpy().dtype == np.dtype("int8")

    res = qml.math.TensorBox(x).cast("complex128")
    expected = np.array([1, 2, 3], dtype=np.complex128)
    assert np.all(res == expected)
    assert res.numpy().dtype.type is np.complex128


@pytest.mark.parametrize(
    "x,expected", [[np.array([]), 0], [np.array([1]), 1], [np.array([[1, 2], [3, 4]]), 2]]
)
def test_len(x, expected):
    """Test length"""
    res = qml.math.TensorBox(x)
    assert len(res) == len(x) == expected


def test_multiplication():
    """Test multiplication between tensors and arrays"""
    x = np.array([[1, 2], [3, 4]])
    y = np.array([[1, 0], [0, 1]])

    xT = qml.math.TensorBox(x)
    res = xT * y
    assert np.all(res == x * y)

    yT = qml.math.TensorBox(y)
    res = x * yT
    assert np.all(res == x * y)

    res = xT * yT
    assert np.all(res == x * y)


def test_unbox_list():
    """Test unboxing a mixed list works correctly"""
    x = np.array([[1, 2], [3, 4]])
    y = np.array([[1, 0], [0, 1]])

    xT = qml.math.TensorBox(x)
    res = xT.unbox_list([y, xT, x])

    assert np.all(res == [y, x, x])


def test_numpy():
    """Test that calling numpy() returns a NumPy array representation
    of the TensorBox"""
    x = np.array([[1, 2], [3, 4]])
    xT = qml.math.TensorBox(x)
    assert isinstance(xT.numpy(), np.ndarray)
    assert not isinstance(xT.numpy(), np.tensor)
    assert np.all(xT == x)


def test_shape():
    """Test that arrays return the right shape"""
    x = np.array([[[1, 2], [3, 4]]])
    x = qml.math.TensorBox(x)
    res = x.shape
    assert res == (1, 2, 2)


def test_expand_dims():
    """Test that dimension expansion works"""
    x = np.array([1, 2, 3])
    xT = qml.math.TensorBox(x)

    res = xT.expand_dims(axis=1)
    expected = np.expand_dims(x, axis=1)
    assert isinstance(res, AutogradBox)
    assert np.all(res == expected)


def test_ones_like():
    """Test that all ones arrays are correctly created"""
    x = np.array([[1, 2, 3], [4, 5, 6]])
    xT = qml.math.TensorBox(x)

    res = xT.ones_like()
    expected = np.ones_like(x)
    assert isinstance(res, AutogradBox)
    assert np.all(res == expected)


def test_stack():
    """Test that arrays are correctly stacked together"""
    x = np.array([[1, 2], [3, 4]])
    y = np.array([[1, 0], [0, 1]])

    xT = qml.math.TensorBox(x)
    res = xT.stack([y, xT, x])

    assert np.all(res == np.stack([y, x, x]))


def test_transpose():
    """Test that the transpose is correct"""
    x = np.array([[1, 2], [3, 4]])
    xT = qml.math.TensorBox(x)

    assert np.all(xT.T() == x.T)


def test_autodifferentiation():
    """Test that autodifferentiation is preserved when writing
    a cost function that uses TensorBox method chaining"""
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    cost_fn = lambda a: (qml.math.TensorBox(a).T() ** 2).unbox()[0, 1]
    grad_fn = qml.grad(cost_fn)

    res = grad_fn(x)[0]
    expected = np.array([[0.0, 0.0, 0.0], [8.0, 0.0, 0.0]])
    assert np.all(res == expected)


def test_requires_grad():
    """Test that the requires grad attribute matches the underlying tensor"""
    x = np.array([[1, 2], [3, 4]], requires_grad=False)
    xT = qml.math.TensorBox(x)
    assert not xT.requires_grad

    x = np.array([[1, 2], [3, 4]], requires_grad=True)
    xT = qml.math.TensorBox(x)
    assert xT.requires_grad
