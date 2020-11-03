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
import pennylane as qml
import numpy as np

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

    def test_len(self):
        """Test length"""
        x = np.array([[1, 2], [3, 4]])
        res = qml.TensorBox(x)
        assert len(res) == len(x) == 2

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

    def test_unbox_list(self):
        """Test unboxing a mixed list works correctly"""
        x = np.array([[1, 2], [3, 4]])
        y = np.array([[1, 0], [0, 1]])

        xT = qml.TensorBox(x)
        res = xT.unbox_list([y, xT, x])

        assert np.all(res == [y, x, x])

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


class TestAutogradBox:
    """Tests for the AutogradBox class"""

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

    def test_len(self):
        """Test length"""
        x = np.array([[1, 2], [3, 4]])
        res = qml.TensorBox(x)
        assert len(res) == len(x) == 2

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

    def test_unbox_list(self):
        """Test unboxing a mixed list works correctly"""
        x = np.array([[1, 2], [3, 4]])
        y = np.array([[1, 0], [0, 1]])

        xT = qml.TensorBox(x)
        res = xT.unbox_list([y, xT, x])

        assert np.all(res == [y, x, x])

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
