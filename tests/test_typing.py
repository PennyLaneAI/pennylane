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


# pylint: disable=protected-access,too-many-public-methods
class TestAbstractArray:
    """Tests for the AbstractArray class."""

    def test_basic_instance(self):
        """Test a normal instance of AbstractArray."""

        a = AbstractArray([2, 3], dtype=float)
        assert a.shape == (2, 3)  # converted to tuple
        assert a.dtype == np.float64  # converted to numpy dtype
        assert a._weak_type is True
        assert a.shape_fixed is True

        assert a.size == 6
        assert a.ndim == 2
        assert a.T.shape == (3, 2)
        assert a.T.dtype == np.float64

        assert len(a) == 2

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

        a4 = AbstractArray((-1,), int)
        assert a4 != a0
        assert hash(a4) != hash(a0)

        a5 = AbstractArray(..., int)
        assert a5 != a0
        assert hash(a5) != hash(a0)

        with pytest.raises(
            TypeError, match=r"Cannot check equality between AbstractArray and <class 'int'>"
        ):
            _ = a3 == 2

    def test_repr(self):
        """Test that the repr of AbstractArray is as expected."""
        a0 = AbstractArray((1, 2), int)
        assert repr(a0) == "AbstractArray((1, 2), int64, weak_type=True)"

        a1 = AbstractArray((1, 2), np.int32)
        assert repr(a1) == "AbstractArray((1, 2), int32)"

        a2 = AbstractArray((-1, 2), np.int32)
        assert repr(a2) == "AbstractArray((-1, 2), int32)"

        a3 = AbstractArray(..., np.int32)
        assert repr(a3) == "AbstractArray(..., int32)"

    def test_str(self):
        """Test that the str of AbstractArray is as expected."""
        a0 = AbstractArray((1, 2), int)
        assert str(a0) == "AbstractArray((1, 2), int64, weak_type=True)"

        a1 = AbstractArray((1, 2), np.int32)
        assert str(a1) == "AbstractArray((1, 2), int32)"

        a2 = AbstractArray((-1, 2), np.int32)
        assert str(a2) == "AbstractArray((-1, 2), int32)"

        a3 = AbstractArray(..., np.int32)
        assert str(a3) == "AbstractArray(..., int32)"

    def test_ellipsis_shape(self):
        """Test that Ellipsis means the number of axes is unknown."""

        a = AbstractArray(..., int)

        assert a.shape is Ellipsis
        assert a.shape_fixed is False
        assert a.T.shape is Ellipsis

        with pytest.raises(TypeError, match="size is undefined for"):
            _ = a.size

        with pytest.raises(TypeError, match="ndim is undefined for"):
            _ = a.ndim

        with pytest.raises(TypeError, match=r"len\(\) of unsized object."):
            _ = len(a)

    def test_invalid_shape_tuple_error(self):
        """Test that Ellipsis cannot appear inside a shape tuple."""
        with pytest.raises(ValueError, match="Shapes can only be initialized with integer values"):
            AbstractArray(("a", 2), int)

    def test_unknown_axis_size(self):
        """Test that -1 marks an axis with unknown size."""

        a = AbstractArray((-1, 2), int)

        assert a.shape == (-1, 2)
        assert a.shape_fixed is False
        assert a.T.shape == (2, -1)

        with pytest.raises(TypeError, match="size is undefined for"):
            _ = a.size

    def test_wire_type_factory(self):
        """Test that we can index into a wire type factory to produce a new hint with a size."""

        a = _AbstractWireTypeFactory()

        b = a[2]
        assert isinstance(b, AbstractWires)
        assert b.num_wires == 2

        c = a[-1]
        assert isinstance(c, AbstractWires)
        assert c.num_wires == -1

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
        assert d.shape == Ellipsis
        assert d.dtype == np.int64

        e = a[5, -1, 2]
        assert isinstance(e, AbstractArray)
        assert e.shape == (5, -1, 2)
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

        with pytest.raises(IndexError, match="Cannot index into an AbstractArray."):
            a[1] = 2

    def test_error_len_on_scalar(self):
        """Test that requesting the len of a scalar results in an error."""
        a = AbstractArray((), int)

        with pytest.raises(TypeError, match=r"len\(\) of unsized object."):
            _ = len(a)

    def test_error_len_on_dynamic_zeroeth_axis(self):
        """Test that requesting the len of a scalar results in an error."""
        a = AbstractArray((-1,), int)

        with pytest.raises(TypeError, match=r"len\(\) is undefined for"):
            _ = len(a)

    @pytest.mark.parametrize("bad_index", (5.0, "a", None))
    def test_error_bad_indices(self, bad_index):
        """Test that an error is raised on invalid indices."""

        a = _AbstractTypeFactory(int)
        b = _AbstractWireTypeFactory()

        with pytest.raises(TypeError, match="can only be subscripted with integers and ellipsis."):
            _ = a[bad_index]

        with pytest.raises(TypeError, match="can only be subscripted with integers."):
            _ = b[bad_index]

    @pytest.mark.parametrize(
        "shortcut, dtype",
        [(Int, np.int64), (Float, np.float64), (Bool, np.bool), (Complex, np.complex128)],
    )
    def test_scalar_shortcuts(self, shortcut, dtype):
        """Test for the various available shortcuts."""

        assert shortcut.dtype == dtype
        assert shortcut.shape == ()

        a = shortcut[2, 3, -1]
        assert a.shape == (2, 3, -1)
        assert a.dtype == dtype

    # pylint: disable=isinstance-second-argument-not-valid-type
    def test_instance_check(self):
        """Test that things can be checked to be instances of a AbstractArray instance."""

        a = AbstractArray((4, 2), bool)
        b = AbstractArray((-1, 2), bool)

        for variant in (a, b):
            assert isinstance(np.zeros((4, 2), bool), variant)
            assert not isinstance(np.array([0, 0], dtype=bool), variant)
            assert not isinstance(np.ones((4, 2), float), variant)
            assert not isinstance("a", variant)

    def test_instance_check_unknown_rank(self):
        """Test ``isinstance`` when the abstract rank is unknown."""
        aa = AbstractArray(..., np.float64)
        assert isinstance(np.ones((4, 2)), aa)
        assert isinstance(np.array(0.5), aa)
        assert not isinstance(np.ones((4, 2), dtype=np.int64), aa)

    def test_instance_check_rejects_non_array(self):
        """Test that non-array objects without a shape fail ``isinstance`` checks."""
        aa = AbstractArray((2,), np.float64)
        assert not isinstance(0.5, aa)

    def test_weak_type_for_number_dtypes(self):
        """Test that number dtypes are marked as weak types."""
        assert AbstractArray((), float)._weak_type is True
        assert AbstractArray((2, 3), int)._weak_type is True
        assert AbstractArray((2, 3), np.float32)._weak_type is False
        assert AbstractArray((2, 3), np.bool_)._weak_type is False

    def test_is_compatible_with_non_numeric_input(self):
        """Test that ``is_compatible_with`` for non-numeric values returns ``False``."""
        aa = AbstractArray((5,), int)
        assert aa.is_compatible_with("hello") is False

    def test_is_compatible_with_weak_scalar(self):
        """Test ``is_compatible_with`` for scalar values."""
        aa = AbstractArray((), float)

        assert aa.is_compatible_with(0.5)
        assert aa.is_compatible_with(np.array(0.5))
        assert aa.is_compatible_with(1)
        assert not aa.is_compatible_with(np.array([0.5, 0.6]))
        assert not aa.is_compatible_with(1 + 0j)

    def test_is_compatible_with_weak_array(self):
        """Test ``is_compatible_with`` for array values."""
        aa = AbstractArray((2, 3), float)

        assert aa.is_compatible_with(np.ones((2, 3)))
        assert aa.is_compatible_with(np.ones((2, 3), dtype=int))
        assert not aa.is_compatible_with(np.ones((2, 2)))
        assert not aa.is_compatible_with(np.ones((2, 3), dtype=complex))

    def test_is_compatible_with_arbitrary_shape(self):
        """Test that ``shape=Ellipsis`` accepts any rank and size."""
        aa = AbstractArray(..., float)
        assert aa.is_compatible_with(np.ones((4, 2)))
        assert aa.is_compatible_with(np.ones((5, 4, 3, 2)))
        assert aa.is_compatible_with(1)

    def test_is_compatible_with_list(self):
        """Test that list inputs are converted before compatibility checks."""
        aa = AbstractArray((2,), int)
        assert aa.is_compatible_with([1, 2])
        assert not aa.is_compatible_with([1, 2, 3])

    def test_is_compatible_with_unknown_axes(self):
        """Test that -1 in a shape accept any concrete dimension."""
        aa = AbstractArray((-1, 2), float)
        assert aa.is_compatible_with(np.ones((4, 2)))
        assert aa.is_compatible_with(np.ones((7, 2)))
        assert not aa.is_compatible_with(np.ones((4, 3)))

    @pytest.mark.torch
    def test_is_compatible_with_torch_tensor(self):
        """Test compatibility checks against torch tensors."""
        import torch

        aa = AbstractArray((2,), torch.float32)
        assert aa.is_compatible_with(torch.ones(2, dtype=torch.float32))
        assert not aa.is_compatible_with(torch.ones(2, dtype=torch.float64))


class TestAbstractWires:
    """Test for the AbstractWires class."""

    def test_basic(self):
        """Basic tests for the AbstractWires class."""

        a = AbstractWires(3)
        assert a.num_wires == 3
        assert len(a) == 3

    def test_invalid_num_wires(self):
        """Test that an error is raised if the provided ``num_wires`` is not valid."""
        with pytest.raises(TypeError, match="'num_wires' must be"):
            _ = AbstractWires("a")

        with pytest.raises(ValueError, match="'num_wires' must be"):
            _ = AbstractWires(-3)

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

    def test_repr(self):
        """Test that the repr of AbstractWires is correct."""
        a0 = AbstractWires(2)
        assert repr(a0) == "AbstractWires(2)"

        a1 = AbstractWires(-1)
        assert repr(a1) == "AbstractWires(-1)"

    def test_str(self):
        """Test that the str of AbstractWires is correct."""
        a0 = AbstractWires(2)
        assert str(a0) == "AbstractWires(2)"

        a1 = AbstractWires(-1)
        assert str(a1) == "AbstractWires(-1)"

    def test_unknown_num_wires(self):
        """Test that -1 marks an unknown number of wires."""
        a = AbstractWires(-1)
        assert a.num_wires == -1
        assert a.shape == (-1,)

        with pytest.raises(TypeError, match="len\\(\\) is undefined for"):
            _ = len(a)

    def test_shape_and_dtype(self):
        """Test that AbstractWires have a shape and dtype."""

        a = AbstractWires(3)
        assert a.shape == (3,)
        assert a.dtype == np.int64

    def test_instance_check(self):
        """Test instance check of Wire."""
        # pylint: disable=isinstance-second-argument-not-valid-type

        # int wire labels
        for i in range(4):
            w = Wires(list(range(i)))
            assert isinstance(w, Wire[i])
            assert not isinstance(w, Wire[6])

        # str wire labels
        l = ["a", "b", "c"]

        for i in range(len(l)):
            w = Wires(l[:i])
            assert isinstance(w, Wire[i])
            assert isinstance(w, Wire[-1])
            assert not isinstance(w, Wire[6])

        # non-wires
        assert not isinstance({"not": "wires"}, Wire)
