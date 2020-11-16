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
"""Unit tests for TensorBox class"""
import numpy as np
import pytest

import pennylane as qml
from pennylane.proc.numpy_box import NumpyBox


def test_creation_from_list():
    """Test that a NumpyBox is automatically created from a list"""
    x = [0.1, 0.2, 0.3]
    res = qml.proc.TensorBox(x)
    assert isinstance(res, NumpyBox)
    assert res.interface == "numpy"
    assert isinstance(res.unbox(), np.ndarray)
    assert np.all(res.unbox() == x)


def test_creation_from_tuple():
    """Test that a NumpyBox is automatically created from a tuple"""
    x = (0.1, 0.2, 0.3)
    res = qml.proc.TensorBox(x)
    assert isinstance(res, NumpyBox)
    assert res.interface == "numpy"
    assert isinstance(res.unbox(), np.ndarray)
    assert np.all(res.unbox() == x)


def test_creation_from_tensorbox():
    """Test that a tensorbox input simply returns it"""
    x = qml.proc.TensorBox(np.array([0.1, 0.2, 0.3]))
    res = qml.proc.TensorBox(x)
    assert x is res


def test_unknown_input_type():
    """Test that an exception is raised if the input type
    is unknown"""
    with pytest.raises(ValueError, match="Unknown tensor type"):
        qml.proc.TensorBox(True)


def test_astensor():
    """Test conversion of sequences to numpy arrays"""
    x = np.array([0.1, 0.2, 0.3])
    y = [0.4, 0.5, 0.6]

    res = qml.proc.TensorBox(x).astensor(y)
    assert isinstance(res, np.ndarray)
    assert np.all(res == y)


def test_cast():
    """Test that arrays can be cast to different dtypes"""
    x = np.array([1, 2, 3])

    res = qml.proc.TensorBox(x).cast(np.float64)
    expected = np.array([1.0, 2.0, 3.0])
    assert np.all(res == expected)

    res = qml.proc.TensorBox(x).cast(np.dtype("int8"))
    expected = np.array([1, 2, 3], dtype=np.int8)
    assert np.all(res == expected)

    res = qml.proc.TensorBox(x).cast("complex128")
    expected = np.array([1, 2, 3], dtype=np.complex128)
    assert np.all(res == expected)


def test_len():
    """Test length"""
    x = np.array([[1, 2], [3, 4]])
    res = qml.proc.TensorBox(x)
    assert len(res) == len(x) == 2


def test_ufunc_compatibility():
    """Test that the NumpyBox class has ufunc compatibility"""
    x = np.array([0.1, 0.2, 0.3])
    res = np.sum(np.sin(qml.proc.TensorBox(x)))
    assert res == np.sin(0.1) + np.sin(0.2) + np.sin(0.3)

    x = np.array([0.1, 0.2, 0.3])
    res = np.sum(np.sin(qml.proc.TensorBox(x), out=np.empty([3])))
    assert res == np.sin(0.1) + np.sin(0.2) + np.sin(0.3)


def test_inplace_addition():
    """Test that in-place addition works correctly"""
    x = qml.proc.TensorBox(np.array([0.0, 0.0, 0.0]))
    np.add.at(x, [0, 1, 1], 1)
    assert np.all(x == np.array([1.0, 2.0, 0.0]))


def test_addition():
    """Test addition between tensors and arrays"""
    x = np.array([[1, 2], [3, 4]])
    y = np.array([[1, 0], [0, 1]])

    xT = qml.proc.TensorBox(x)
    res = xT + y
    assert np.all(res.unbox() == x + y)

    yT = qml.proc.TensorBox(y)
    res = x + yT
    assert np.all(res.unbox() == x + y)

    res = xT + yT
    assert np.all(res.unbox() == x + y)


def test_subtraction():
    """Test addition between tensors and arrays"""
    x = np.array([[1, 2], [3, 4]])
    y = np.array([[1, 0], [0, 1]])

    xT = qml.proc.TensorBox(x)
    res = xT - y
    assert np.all(res.unbox() == x - y)

    yT = qml.proc.TensorBox(y)
    res = x - yT
    assert np.all(res.unbox() == x - y)

    res = xT - yT
    assert np.all(res.unbox() == x - y)


def test_multiplication():
    """Test multiplication between tensors and arrays"""
    x = np.array([[1, 2], [3, 4]])
    y = np.array([[1, 0], [0, 1]])

    xT = qml.proc.TensorBox(x)
    res = xT * y
    assert np.all(res.unbox() == x * y)

    yT = qml.proc.TensorBox(y)
    res = x * yT
    assert np.all(res.unbox() == x * y)

    res = xT * yT
    assert np.all(res.unbox() == x * y)


def test_division():
    """Test addition between tensors and arrays"""
    x = np.array([[1, 2], [3, 4]])
    y = np.array([[1, 4], [0.25, 1]])

    xT = qml.proc.TensorBox(x)
    res = xT / y
    assert np.all(res.unbox() == x / y)

    yT = qml.proc.TensorBox(y)
    res = x / yT
    assert np.all(res.unbox() == x / y)

    res = xT / yT
    assert np.all(res.unbox() == x / y)

    res = 5 / yT
    assert np.all(res.unbox() == 5 / y)

    res = yT / 5
    assert np.all(res.unbox() == y / 5)


def test_exponentiation():
    """Test exponentiation between tensors and arrays"""
    x = np.array([[1, 2], [3, 4]])
    y = np.array([[1, 0], [0, 1]])

    xT = qml.proc.TensorBox(x)
    res = xT ** 2
    assert np.all(res.unbox() == x ** 2)

    yT = qml.proc.TensorBox(y)
    res = 2 ** yT
    assert np.all(res.unbox() == 2 ** y)

    res = xT ** yT
    assert np.all(res.unbox() == x ** y)


def test_unbox_list():
    """Test unboxing a mixed list works correctly"""
    x = np.array([[1, 2], [3, 4]])
    y = np.array([[1, 0], [0, 1]])

    xT = qml.proc.TensorBox(x)
    res = xT.unbox_list([y, xT, x])

    assert np.all(res == [y, x, x])


def test_numpy():
    """Test that calling numpy() returns a NumPy array representation
    of the TensorBox"""
    x = np.array([[1, 2], [3, 4]])
    xT = qml.proc.TensorBox(x)
    assert isinstance(xT.numpy(), np.ndarray)
    assert np.all(xT.numpy() == x)


def test_shape():
    """Test that arrays return the right shape"""
    x = np.array([[[1, 2], [3, 4]]])
    x = qml.proc.TensorBox(x)
    res = x.shape
    assert res == (1, 2, 2)


def test_expand_dims():
    """Test that dimension expansion works"""
    x = np.array([1, 2, 3])
    xT = qml.proc.TensorBox(x)

    res = xT.expand_dims(axis=1)
    expected = np.expand_dims(x, axis=1)
    assert isinstance(res, NumpyBox)
    assert np.all(res == expected)


def test_ones_like():
    """Test that all ones arrays are correctly created"""
    x = np.array([[1, 2, 3], [4, 5, 6]])
    xT = qml.proc.TensorBox(x)

    res = xT.ones_like()
    expected = np.ones_like(x)
    assert isinstance(res, NumpyBox)
    assert np.all(res == expected)


def test_stack():
    """Test that arrays are correctly stacked together"""
    x = np.array([[1, 2], [3, 4]])
    y = np.array([[1, 0], [0, 1]])

    xT = qml.proc.TensorBox(x)
    res = xT.stack([y, xT, x])

    assert np.all(res == np.stack([y, x, x]))


def test_transpose():
    """Test that the transpose is correct"""
    x = np.array([[1, 2], [3, 4]])
    xT = qml.proc.TensorBox(x)

    assert np.all(xT.T == x.T)


def test_requires_grad():
    """Test that the requires grad attribute always returns False"""
    x = np.array([[1, 2], [3, 4]])
    xT = qml.proc.TensorBox(x)
    assert not xT.requires_grad
