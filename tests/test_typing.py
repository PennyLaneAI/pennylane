# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the ``typing.py`` file."""

import numpy as np
import pytest

import pennylane.numpy as pnp
from pennylane.typing import (
    AbstractArray,
    AbstractWires,
    Bool,
    Complex,
    Float,
    Int,
    TensorLike,
    Wire,
    _AbstractTypeFactory,
    _AbstractWireTypeFactory,
)
from pennylane.wires import Wires


class TestTensorLike:
    def test_isinstance_unknown_type(self):
        """Test that an unknown type returns False."""

        # pylint: disable=too-few-public-methods
        class UnknownType:
            """Unknown type."""

        assert not isinstance(UnknownType(), TensorLike)

    def test_isinstance_numpy_array(self):
        """Tests that a numpy array is a Tensor"""
        assert isinstance(np.array(1), TensorLike)

    def test_isinstance_pennylane_tensor(self):
        """Tests that a PennyLane numpy tensor is a Tensor"""
        assert isinstance(pnp.array(1), TensorLike)

    @pytest.mark.jax
    def test_isinstance_jax_array_is_tensor_like(self):
        """Tests that a jax DeviceArray is a Tensor"""
        import jax

        tensor = jax.numpy.array(1)
        assert isinstance(tensor, TensorLike)

    @pytest.mark.torch
    def test_isinstance_torch_tensor_is_tensor_like(self):
        """Tests that a torch Tensor is a Tensor"""
        import torch

        tensor = torch.Tensor(1)
        assert isinstance(tensor, TensorLike)

    @pytest.mark.tf
    def test_isinstance_tf_tensor_is_tensor_like(self):
        """Tests that a tensorflow Tensor is a Tensor"""
        import tensorflow as tf

        tensor = tf.constant([1, 2, 3])
        assert isinstance(tensor, tf.Tensor)
        assert isinstance(tensor, TensorLike)
        var = tf.Variable(9)
        assert isinstance(var, TensorLike)

    def test_subclass_unknown_type(self):
        """Test that an unknown type returns False."""

        # pylint: disable=too-few-public-methods
        class UnknownType:
            """Unknown type."""

        assert not issubclass(UnknownType, TensorLike)

    def test_subclass_numpy_array(self):
        """Tests that a numpy array is a Tensor"""
        assert issubclass(np.ndarray, TensorLike)

    def test_subclass_pennylane_tensor(self):
        """Tests that a PennyLane numpy tensor is a Tensor"""
        assert issubclass(pnp.ndarray, TensorLike)

    @pytest.mark.jax
    def test_subclass_jax_array_is_tensor_like(self):
        """Tests that a jax DeviceArray is a Tensor"""
        import jax

        assert issubclass(jax.numpy.ndarray, TensorLike)

    @pytest.mark.torch
    def test_subclass_torch_tensor_is_tensor_like(self):
        """Tests that a torch Tensor is a Tensor"""
        import torch

        assert issubclass(torch.Tensor, TensorLike)

    @pytest.mark.tf
    def test_subclass_tf_tensor_is_tensor_like(self):
        """Tests that a tensorflow Tensor is a Tensor"""
        import tensorflow as tf

        assert issubclass(tf.Tensor, TensorLike)
        assert issubclass(tf.Variable, TensorLike)


class TestAbstractArray:
    """Tests for the AbstractArray class."""

    def test_basic_instance(self):
        """Test a normal instance of AbstractArray."""

        a = AbstractArray([2, 3], dtype=float)
        assert a.shape == (2, 3)  # converted to tuple
        assert a.dtype == np.float64  # converted to numpy dtype

        assert a.size == 6
        assert a.ndim == 2
        assert a.T.shape == (3, 2)
        assert a.T.dtype == np.float64

        assert len(a) == 2

        with pytest.raises(IndexError, match="Cannot index into an AbstractArray."):
            a[1] = 2

    @pytest.mark.torch
    def test_provide_torch_dtype(self):
        """Test that a torch dtype is converted to a numpy dtype."""

        import torch

        a = AbstractArray((2,), torch.int32)

        assert a.dtype == np.int32

    def test_hash_and_equality(self):
        """Test hash and equality for AbstractArray"""

        a0 = AbstractArray((2,), int)
        a0_1 = AbstractArray((2,), int)

        assert a0 == a0_1
        assert hash(a0) == hash(a0_1)

        a1 = AbstractArray((2, 3), int)
        assert a1 != a0
        assert hash(a1) != hash(a0)

        a2 = AbstractArray((3,), int)
        assert a2 != a0
        assert hash(a2) != hash(a0)

        a3 = AbstractArray((2,), float)
        assert a3 != a0
        assert hash(a3) != hash(a0)

        a4 = AbstractArray((...,), int)
        assert a4 != a0
        assert hash(a4) != hash(a0)

        with pytest.raises(
            TypeError, match=r"Cannot check equality between AbstractArray and <class 'int'>"
        ):
            _ = a3 == 2

    def test_ellipsis_in_shape(self):
        """Test that an Ellipsis can be used in the shape tuple."""

        a = AbstractArray((..., 2), int)

        assert a.shape == (..., 2)
        assert a.T.shape == (2, ...)

        with pytest.raises(TypeError, match="size is undefined for"):
            _ = a.size

        assert a.ndim == 2

    def test_wire_type_factory(self):
        """Test that we can index into a wire type factory to produce a new hint with a size."""

        a = _AbstractWireTypeFactory()

        b = a[2]
        assert isinstance(b, AbstractWires)
        assert b.num_wires == 2

        c = a[...]
        assert isinstance(c, AbstractWires)
        assert c.num_wires == ...

        d = a[-1]
        assert isinstance(d, AbstractWires)
        assert d.num_wires == -1

    def test_type_factory(self):
        """Test that we can index into a type factory to produce a new hint with a size."""

        a = _AbstractTypeFactory(int)

        b = a[2, 3]
        assert isinstance(b, AbstractArray)
        assert b.dtype == np.int64
        assert b.shape == (2, 3)

        c = a[2]
        assert isinstance(c, AbstractArray)
        assert c.shape == (2,)
        assert c.dtype == np.int64

        d = a[...]
        assert isinstance(d, AbstractArray)
        assert d.shape == (...,)
        assert d.dtype == np.int64

        e = a[5, ..., 2]
        assert isinstance(e, AbstractArray)
        assert e.shape == (5, ..., 2)
        assert e.dtype == np.int64

        f = a[-1]
        assert isinstance(f, AbstractArray)
        assert f.dtype == np.int64
        assert f.shape == (-1,)

    def test_error_indexing_into_non_scalar(self):
        """Test an error is raised when indexing into a non-scalar AbstractArray."""

        a = AbstractArray((2,), int)

        with pytest.raises(IndexError, match="Cannot index into an AbstractArray"):
            _ = a[1]

    def test_error_len_on_scalar(self):
        """Test that requesting the len of a scalar results in an error."""
        a = AbstractArray((), int)

        with pytest.raises(TypeError, match=r"len\(\) of unsized object."):
            _ = len(a)

    @pytest.mark.parametrize("bad_index", (5.0, "a", None))
    def test_error_bad_indices(self, bad_index):
        """Test that an error is raised on invalid indices."""

        a = _AbstractTypeFactory(int)
        b = _AbstractWireTypeFactory()

        with pytest.raises(TypeError, match="can only be subscripted with integers and ellipsis."):
            _ = a[bad_index]

        with pytest.raises(TypeError, match="can only be subscripted with integers and ellipsis."):
            _ = b[bad_index]

    @pytest.mark.parametrize(
        "shortcut, dtype",
        [(Int, np.int64), (Float, np.float64), (Bool, np.bool), (Complex, np.complex128)],
    )
    def test_scalar_shortcuts(self, shortcut, dtype):
        """Test for the various available shortcuts."""

        assert shortcut.dtype == dtype
        assert shortcut.shape == ()

        a = shortcut[2, 3, ...]
        assert a.shape == (2, 3, ...)
        assert a.dtype == dtype

    # pylint: disable=isinstance-second-argument-not-valid-type
    def test_instance_check(self):
        """Test that things can be checked to be instances of a AbstractArray instance."""

        a = AbstractArray((4, 2), bool)
        b = AbstractArray((..., 2), bool)

        for variant in (a, b):
            assert isinstance(np.zeros((4, 2), bool), variant)

            assert not isinstance(np.array([0, 0], dtype=bool), variant)

            assert not isinstance(np.ones((4, 2), float), variant)

            assert not isinstance("a", variant)


class TestAbstractWires:
    """Test for the AbstractWires class."""

    def test_basic(self):
        """Basic tests for the AbstractWires class."""

        a = AbstractWires(3)
        assert a.num_wires == 3
        assert len(a) == 3

    def test_comparison(self):
        """Test for equality and comparison."""
        a = AbstractWires(3)
        assert a == AbstractWires(3)
        assert a != AbstractWires(4)
        assert hash(a) == hash(AbstractWires(3))
        assert hash(a) != hash(AbstractWires(4))

        with pytest.raises(
            TypeError, match="Cannot check equality between AbstractWires and an object"
        ):
            _ = a == 2

    def test_ellipsis(self):
        """Test that number of wires can be specified by an ellipsis."""

        a = AbstractWires(...)
        assert a.num_wires == ...

    def test_shape_and_dtype(self):
        """Test that AbstractWires have a shape and dtype."""

        a = AbstractWires(3)
        assert a.shape == (3,)
        assert a.dtype == np.int64

    def test_instance_check(self):
        """Test instance check of Wire."""

        # int wire labels
        for i in range(4):
            w = Wires(list(range(i)))
            assert isinstance(w, Wire[i])
            assert not isinstance(w, Wire[i - 1])

        # str wire labels
        l = ["a", "b", "c"]

        for i in range(len(l)):
            w = Wires(l[:i])
            assert isinstance(w, Wire[i])
            assert not isinstance(w, Wire[i - 1])

        # non-wires
        assert not isinstance({"not": "wires"}, Wire)
