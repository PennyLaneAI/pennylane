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
"""Unit tests for TorchBox subclass"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

import pennylane as qml
from pennylane.proc.torch_box import TorchBox


def test_creation():
    """Test that a TorchBox is automatically created from a torch tensor"""
    x = torch.tensor([0.1, 0.2, 0.3])
    res = qml.proc.TensorBox(x)

    assert isinstance(res, TorchBox)
    assert res.interface == "torch"
    assert isinstance(res.unbox(), torch.Tensor)
    assert torch.allclose(res.unbox(), x)


def test_astensor_list():
    """Test conversion of numpy arrays to torch tensors"""
    x = torch.tensor([0.1, 0.2, 0.3])
    y = [0.4, 0.5, 0.6]

    res = qml.proc.TensorBox(x).astensor(y)
    assert isinstance(res, torch.Tensor)
    assert np.all(res.numpy() == np.array(y, dtype=np.float32))


def test_astensor_array():
    """Test conversion of numpy arrays to torch tensors"""
    x = torch.tensor([0.1, 0.2, 0.3])
    y = np.array([0.4, 0.5, 0.6])

    res = qml.proc.TensorBox(x).astensor(y)
    assert isinstance(res, torch.Tensor)
    assert np.all(res.numpy() == y)


def test_cast():
    """Test that arrays can be cast to different dtypes"""
    x = torch.tensor([1, 2, 3])

    res = qml.proc.TensorBox(x).cast(np.float64)
    expected = torch.tensor([1.0, 2.0, 3.0])
    assert np.all(res.numpy() == expected.numpy())

    res = qml.proc.TensorBox(x).cast(np.dtype("int8"))
    expected = torch.tensor([1, 2, 3], dtype=torch.int8)
    assert np.all(res.numpy() == expected.numpy())

    res = qml.proc.TensorBox(x).cast("float64")
    expected = torch.tensor([1, 2, 3], dtype=torch.float64)
    assert np.all(res.numpy() == expected.numpy())

    res = qml.proc.TensorBox(x).cast(torch.float64)
    expected = torch.tensor([1, 2, 3], dtype=torch.float64)
    assert np.all(res.numpy() == expected.numpy())


def test_cast_exception():
    """Test that an exception is raised if we are unable to deduce
    the correct Torch dtype"""
    x = torch.tensor([1, 2, 3])

    with pytest.raises(ValueError, match="Unable to convert"):
        qml.proc.TensorBox(x).cast(np.bytes0)


def test_len():
    """Test length"""
    x = torch.tensor([[1, 2], [3, 4]])
    res = qml.proc.TensorBox(x)
    assert len(res) == len(x) == 2


def test_multiplication():
    """Test multiplication between tensors and arrays"""
    x = torch.tensor([[1, 2], [3, 4]])
    y = torch.tensor([[1, 0], [0, 1]])

    xT = qml.proc.TensorBox(x)
    res = xT * y
    assert torch.allclose(res.unbox(), x * y)

    yT = qml.proc.TensorBox(y)
    res = x * yT
    assert torch.allclose(res.unbox(), x * y)

    res = xT * yT
    assert torch.allclose(res.unbox(), x * y)


def test_unbox_list():
    """Test unboxing a mixed list works correctly"""
    x = torch.tensor([[1, 2], [3, 4]])
    y = torch.tensor([[1, 0], [0, 1]])

    xT = qml.proc.TensorBox(x)
    res = xT.unbox_list([y, xT, x])

    assert np.all(res == [y, x, x])


def test_numpy():
    """Test that calling numpy() returns a NumPy array representation
    of the TensorBox"""
    x = torch.tensor([[1, 2], [3, 4]])
    xT = qml.proc.TensorBox(x)
    assert isinstance(xT.numpy(), np.ndarray)
    assert not isinstance(xT.numpy(), torch.Tensor)
    assert np.all(xT.numpy() == x.numpy())


def test_shape():
    """Test that arrays return the right shape"""
    x = torch.tensor([[[1, 2], [3, 4]]])
    x = qml.proc.TensorBox(x)
    res = x.shape
    assert res == (1, 2, 2)


def test_expand_dims():
    """Test that dimension expansion works"""
    x = torch.tensor([1, 2, 3])
    xT = qml.proc.TensorBox(x)

    res = xT.expand_dims(axis=1)
    expected = torch.unsqueeze(x, dim=1)
    assert isinstance(res, TorchBox)
    assert torch.allclose(res.unbox(), expected)


def test_ones_like():
    """Test that all ones arrays are correctly created"""
    x = torch.tensor([[1, 2, 3], [4, 5, 6]])
    xT = qml.proc.TensorBox(x)

    res = xT.ones_like()
    expected = torch.ones_like(x)
    assert isinstance(res, TorchBox)
    assert torch.allclose(res.unbox(), expected)


def test_stack():
    """Test that arrays are correctly stacked together"""
    x = torch.tensor([[1, 2], [3, 4]])
    y = torch.tensor([[1, 0], [0, 1]])

    xT = qml.proc.TensorBox(x)
    res = xT.stack([y, xT, x])

    assert torch.allclose(res.unbox(), torch.stack([y, x, x]))


def test_transpose():
    """Test that the transpose is correct"""
    x = torch.tensor([[1, 2], [3, 4]])
    xT = qml.proc.TensorBox(x)

    assert torch.allclose(xT.T.unbox(), x.T)


def test_autodifferentiation():
    """Test that autodifferentiation is preserved when writing
    a cost function that uses TensorBox method chaining"""
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    y = (qml.proc.TensorBox(x).T ** 2).unbox()[0, 1]
    y.backward()

    res = x.grad
    expected = torch.tensor([[0.0, 0.0, 0.0], [8.0, 0.0, 0.0]])
    assert torch.allclose(res, expected)


def test_requires_grad():
    """Test that the requires grad attribute matches the underlying tensor"""
    x = torch.tensor([[1.0, 2], [3, 4]], requires_grad=False)
    xT = qml.proc.TensorBox(x)
    assert not xT.requires_grad

    x = torch.tensor([[1.0, 2], [3, 4]], requires_grad=True)
    xT = qml.proc.TensorBox(x)
    assert xT.requires_grad
