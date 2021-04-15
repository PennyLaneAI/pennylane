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
"""Unit tests for JaxBox subclass"""
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
import pennylane as qml
import numpy as np
from pennylane.math.jax_box import JaxBox


def test_creation():
    """Test that a JaxBox is automatically created from a PennyLane jax tensor"""
    x = jnp.array([0.1, 0.2, 0.3])
    res = qml.math.TensorBox(x)
    assert isinstance(res, JaxBox)
    assert res.interface == "jax"
    assert np.all(res == x)


def test_astensor_list():
    """Test conversion of a list to PennyLane tensors"""
    x = jnp.array([0.1, 0.2, 0.3])
    y = jnp.array([0.4, 0.5, 0.6])

    res = qml.math.TensorBox(x).astensor(y)
    assert np.all(res == y)


def test_astensor_array():
    """Test conversion of jax arrays to PennyLane tensors"""
    x = jnp.array([0.1, 0.2, 0.3])
    y = jnp.array([0.4, 0.5, 0.6])

    res = qml.math.TensorBox(x).astensor(y)
    assert np.all(res == y)


def test_cast():
    """Test that arrays can be cast to different dtypes"""
    x = jnp.array([1, 2, 3])

    res = qml.math.TensorBox(x).cast(np.float16)
    expected = jnp.array([1.0, 2.0, 3.0])
    assert np.all(res == expected)
    assert res.numpy().dtype.type is np.float16

    res = qml.math.TensorBox(x).cast(np.dtype("int8"))
    expected = jnp.array([1, 2, 3], dtype=np.int8)
    assert np.all(res == expected)
    assert res.numpy().dtype == np.dtype("int8")

    res = qml.math.TensorBox(x).cast("complex64")
    expected = jnp.array([1, 2, 3], dtype=np.complex64)
    assert np.all(res == expected)
    assert res.numpy().dtype.type is np.complex64


@pytest.mark.parametrize(
    "x,expected", [[np.array([]), 0], [np.array([1]), 1], [np.array([[1, 2], [3, 4]]), 2]]
)
def test_len(x, expected):
    """Test length"""
    res = qml.math.TensorBox(x)
    assert len(res) == len(x) == expected


def test_multiplication():
    """Test multiplication between tensors and arrays"""
    x = jnp.array([[1, 2], [3, 4]])
    y = jnp.array([[1, 0], [0, 1]])

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
    x = jnp.array([[1, 2], [3, 4]])
    y = jnp.array([[1, 0], [0, 1]])

    xT = qml.math.TensorBox(x)
    res = xT.unbox_list([y, xT, x])

    assert np.all(res == [y, x, x])


def test_shape():
    """Test that arrays return the right shape"""
    x = jnp.array([[[1, 2], [3, 4]]])
    x = qml.math.TensorBox(x)
    res = x.shape
    assert res == (1, 2, 2)


def test_expand_dims():
    """Test that dimension expansion works"""
    x = jnp.array([1, 2, 3])
    xT = qml.math.TensorBox(x)

    res = xT.expand_dims(axis=1)
    expected = np.expand_dims(x, axis=1)
    assert isinstance(res, JaxBox)
    assert np.all(res == expected)


def test_ones_like():
    """Test that all ones arrays are correctly created"""
    x = jnp.array([[1, 2, 3], [4, 5, 6]])
    xT = qml.math.TensorBox(x)

    res = xT.ones_like()
    expected = np.ones_like(x)
    assert isinstance(res, JaxBox)
    assert np.all(res == expected)


def test_stack():
    """Test that arrays are correctly stacked together"""
    x = jnp.array([[1, 2], [3, 4]])
    y = jnp.array([[1, 0], [0, 1]])

    xT = qml.math.TensorBox(x)
    res = xT.stack([y, xT, x])

    assert np.all(res == np.stack([y, x, x]))


def test_transpose():
    """Test that the transpose is correct"""
    x = jnp.array([[1, 2], [3, 4]])
    xT = qml.math.TensorBox(x)

    assert np.all(xT.T() == x.T)


def test_autodifferentiation():
    """Test that autodifferentiation is preserved when writing
    a cost function that uses TensorBox method chaining"""
    x = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    cost_fn = lambda a: (qml.math.TensorBox(a).T() ** 2).unbox()[0, 1]
    grad_fn = jax.grad(cost_fn)

    res = grad_fn(x)
    expected = jnp.array([[0.0, 0.0, 0.0], [8.0, 0.0, 0.0]])
    np.testing.assert_array_equal(res, expected)
