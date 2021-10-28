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
"""Unit tests for the TensorBox functional API in pennylane.fn.fn
"""
import itertools
import numpy as onp
import pytest

import pennylane as qml
from pennylane import numpy as np
from pennylane import math as fn
from autograd.numpy.numpy_boxes import ArrayBox


tf = pytest.importorskip("tensorflow", minversion="2.1")
torch = pytest.importorskip("torch")
jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")


class TestGetMultiTensorbox:
    """Tests for the _multi_dispatch utility function"""

    def test_exception_tensorflow_and_torch(self):
        """Test that an exception is raised if the sequence of tensors contains
        tensors from incompatible dispatch libraries"""
        x = tf.Variable([1.0, 2.0, 3.0])
        y = onp.array([0.5, 0.1])
        z = torch.tensor([0.6])

        with pytest.raises(ValueError, match="Tensors contain mixed types"):
            fn._multi_dispatch([x, y, z])

    def test_warning_tensorflow_and_autograd(self):
        """Test that a warning is raised if the sequence of tensors contains
        both tensorflow and autograd tensors."""
        x = tf.Variable([1.0, 2.0, 3.0])
        y = np.array([0.5, 0.1])

        with pytest.warns(UserWarning, match="Consider replacing Autograd with vanilla NumPy"):
            fn._multi_dispatch([x, y])

    def test_warning_torch_and_autograd(self):
        """Test that a warning is raised if the sequence of tensors contains
        both torch and autograd tensors."""
        x = torch.tensor([1.0, 2.0, 3.0])
        y = np.array([0.5, 0.1])

        with pytest.warns(UserWarning, match="Consider replacing Autograd with vanilla NumPy"):
            fn._multi_dispatch([x, y])

    def test_return_tensorflow_box(self):
        """Test that TensorFlow is correctly identified as the dispatching library."""
        x = tf.Variable([1.0, 2.0, 3.0])
        y = onp.array([0.5, 0.1])

        res = fn._multi_dispatch([y, x])
        assert res == "tensorflow"

    def test_return_torch_box(self):
        """Test that Torch is correctly identified as the dispatching library."""
        x = torch.tensor([1.0, 2.0, 3.0])
        y = onp.array([0.5, 0.1])

        res = fn._multi_dispatch([y, x])
        assert res == "torch"

    def test_return_autograd_box(self):
        """Test that autograd is correctly identified as the dispatching library."""
        x = np.array([1.0, 2.0, 3.0])
        y = [0.5, 0.1]

        res = fn._multi_dispatch([y, x])
        assert res == "autograd"

    def test_return_numpy_box(self):
        """Test that NumPy is correctly identified as the dispatching library."""
        x = onp.array([1.0, 2.0, 3.0])
        y = [0.5, 0.1]

        res = fn._multi_dispatch([y, x])
        assert res == "numpy"


test_abs_data = [
    (1, -2, 3 + 4j),
    [1, -2, 3 + 4j],
    onp.array([1, -2, 3 + 4j]),
    np.array([1, -2, 3 + 4j]),
    torch.tensor([1, -2, 3 + 4j], dtype=torch.complex128),
    tf.Variable([1, -2, 3 + 4j], dtype=tf.complex128),
    tf.constant([1, -2, 3 + 4j], dtype=tf.complex128),
]


@pytest.mark.parametrize("t", test_abs_data)
def test_abs(t):
    """Test that the absolute function works for a variety
    of input"""
    res = fn.abs(t)
    assert fn.allequal(res, [1, 2, 5])


test_data = [
    (1, 2, 3),
    [1, 2, 3],
    onp.array([1, 2, 3]),
    np.array([1, 2, 3]),
    torch.tensor([1, 2, 3]),
    tf.Variable([1, 2, 3]),
    tf.constant([1, 2, 3]),
]


@pytest.mark.parametrize("t1,t2", list(itertools.combinations(test_data, r=2)))
def test_allequal(t1, t2):
    """Test that the allequal function works for a variety of inputs."""
    res = fn.allequal(t1, t2)

    if isinstance(t1, tf.Variable):
        t1 = tf.convert_to_tensor(t1)

    if isinstance(t2, tf.Variable):
        t2 = tf.convert_to_tensor(t2)

    expected = all(float(x) == float(y) for x, y in zip(t1, t2))
    assert res == expected


@pytest.mark.parametrize("t1,t2", list(itertools.combinations(test_data, r=2)))
def test_allclose(t1, t2):
    """Test that the allclose function works for a variety of inputs."""
    res = fn.allclose(t1, t2)

    if isinstance(t1, tf.Variable):
        t1 = tf.convert_to_tensor(t1)

    if isinstance(t2, tf.Variable):
        t2 = tf.convert_to_tensor(t2)

    expected = all(float(x) == float(y) for x, y in zip(t1, t2))
    assert res == expected


test_angle_data = [
    [1.0, 1.0j, 1 + 1j],
    [1.0, 1.0j, 1 + 1j],
    onp.array([1.0, 1.0j, 1 + 1j]),
    np.array([1.0, 1.0j, 1 + 1j]),
    torch.tensor([1.0, 1.0j, 1 + 1j], dtype=torch.complex128),
    tf.Variable([1.0, 1.0j, 1 + 1j], dtype=tf.complex128),
    tf.constant([1.0, 1.0j, 1 + 1j], dtype=tf.complex128),
]


@pytest.mark.parametrize("t", test_angle_data)
def test_angle(t):
    """Test that the angle function works for a variety
    of input"""
    res = fn.angle(t)
    assert fn.allequal(res, [0, np.pi / 2, np.pi / 4])


test_arcsin_data = [
    (1, 0.2, -0.5),
    [1, 0.2, -0.5],
    onp.array([1, 0.2, -0.5]),
    np.array([1, 0.2, -0.5]),
    torch.tensor([1, 0.2, -0.5], dtype=torch.float64),
    tf.Variable([1, 0.2, -0.5], dtype=tf.float64),
    tf.constant([1, 0.2, -0.5], dtype=tf.float64),
]


@pytest.mark.parametrize("t", test_arcsin_data)
def test_arcsin(t):
    """Test that the arcsin function works for a variety
    of input"""
    res = fn.arcsin(t)
    assert fn.allequal(res, np.arcsin([1, 0.2, -0.5]))


test_conj_data = [
    [1.0, 1.0j, 1 + 1j],
    onp.array([1.0, 1.0j, 1 + 1j]),
    np.array([1.0, 1.0j, 1 + 1j]),
    jnp.array([1.0, 1.0j, 1 + 1j]),
    torch.tensor([1.0, 1.0j, 1 + 1j], dtype=torch.complex128),
    tf.Variable([1.0, 1.0j, 1 + 1j], dtype=tf.complex128),
    tf.constant([1.0, 1.0j, 1 + 1j], dtype=tf.complex128),
]


@pytest.mark.parametrize("t", test_conj_data)
def test_conj(t):
    res = fn.conj(t)
    assert fn.allequal(res, np.conj(t))


class TestCast:
    """Tests for the cast function"""

    @pytest.mark.parametrize("t", test_data)
    def test_cast_numpy(self, t):
        """Test that specifying a NumPy dtype results in proper casting
        behaviour"""
        res = fn.cast(t, onp.float64)
        assert fn.get_interface(res) == fn.get_interface(t)

        if hasattr(res, "numpy"):
            # if tensorflow or pytorch, extract view of underlying data
            res = res.numpy()
            t = t.numpy()

        assert onp.issubdtype(onp.asarray(t).dtype, onp.integer)
        assert res.dtype.type is onp.float64

    @pytest.mark.parametrize("t", test_data)
    def test_cast_numpy_dtype(self, t):
        """Test that specifying a NumPy dtype object results in proper casting
        behaviour"""
        res = fn.cast(t, onp.dtype("float64"))
        assert fn.get_interface(res) == fn.get_interface(t)

        if hasattr(res, "numpy"):
            # if tensorflow or pytorch, extract view of underlying data
            res = res.numpy()
            t = t.numpy()

        assert onp.issubdtype(onp.asarray(t).dtype, onp.integer)
        assert res.dtype.type is onp.float64

    @pytest.mark.parametrize("t", test_data)
    def test_cast_numpy_string(self, t):
        """Test that specifying a NumPy dtype via a string results in proper casting
        behaviour"""
        res = fn.cast(t, "float64")
        assert fn.get_interface(res) == fn.get_interface(t)

        if hasattr(res, "numpy"):
            # if tensorflow or pytorch, extract view of underlying data
            res = res.numpy()
            t = t.numpy()

        assert onp.issubdtype(onp.asarray(t).dtype, onp.integer)
        assert res.dtype.type is onp.float64

    def test_cast_tensorflow_dtype(self):
        """If the tensor is a TensorFlow tensor, casting using a TensorFlow dtype
        will also work"""
        t = tf.Variable([1, 2, 3])
        res = fn.cast(t, tf.complex128)
        assert isinstance(res, tf.Tensor)
        assert res.dtype is tf.complex128

    def test_cast_torch_dtype(self):
        """If the tensor is a Torch tensor, casting using a Torch dtype
        will also work"""
        t = torch.tensor([1, 2, 3], dtype=torch.int64)
        res = fn.cast(t, torch.float64)
        assert isinstance(res, torch.Tensor)
        assert res.dtype is torch.float64


cast_like_test_data = [
    (1, 2, 3),
    [1, 2, 3],
    onp.array([1, 2, 3], dtype=onp.int64),
    np.array([1, 2, 3], dtype=np.int64),
    torch.tensor([1, 2, 3], dtype=torch.int64),
    tf.Variable([1, 2, 3], dtype=tf.int64),
    tf.constant([1, 2, 3], dtype=tf.int64),
    (1.0, 2.0, 3.0),
    [1.0, 2.0, 3.0],
    onp.array([1, 2, 3], dtype=onp.float64),
    np.array([1, 2, 3], dtype=np.float64),
    torch.tensor([1, 2, 3], dtype=torch.float64),
    tf.Variable([1, 2, 3], dtype=tf.float64),
    tf.constant([1, 2, 3], dtype=tf.float64),
]


@pytest.mark.parametrize("t1,t2", list(itertools.combinations(cast_like_test_data, r=2)))
def test_cast_like(t1, t2):
    """Test that casting t1 like t2 results in t1 being cast to the same datatype as t2"""
    res = fn.cast_like(t1, t2)

    # if tensorflow or pytorch, extract view of underlying data
    if hasattr(res, "numpy"):
        res = res.numpy()

    if hasattr(t2, "numpy"):
        t2 = t2.numpy()

    assert fn.allequal(res, t1)
    assert onp.asarray(res).dtype.type is onp.asarray(t2).dtype.type


class TestConcatenate:
    """Tests for the concatenate function"""

    def test_concatenate_array(self):
        """Test that concatenate, called without the axis arguments, concatenates across the 0th dimension"""
        t1 = [0.6, 0.1, 0.6]
        t2 = np.array([0.1, 0.2, 0.3])
        t3 = onp.array([5.0, 8.0, 101.0])

        res = fn.concatenate([t1, t2, t3])
        assert isinstance(res, np.ndarray)
        assert np.all(res == np.concatenate([t1, t2, t3]))

    def test_concatenate_jax(self):
        """Test that concatenate, called without the axis arguments, concatenates across the 0th dimension"""
        t1 = jnp.array([5.0, 8.0, 101.0])
        t2 = jnp.array([0.6, 0.1, 0.6])
        t3 = jnp.array([0.1, 0.2, 0.3])

        res = fn.concatenate([t1, t2, t3])
        assert jnp.all(res == jnp.concatenate([t1, t2, t3]))

    def test_stack_tensorflow(self):
        """Test that concatenate, called without the axis arguments, concatenates across the 0th dimension"""
        t1 = tf.constant([0.6, 0.1, 0.6])
        t2 = tf.Variable([0.1, 0.2, 0.3])
        t3 = onp.array([5.0, 8.0, 101.0])

        res = fn.concatenate([t1, t2, t3])
        assert isinstance(res, tf.Tensor)
        assert np.all(res.numpy() == np.concatenate([t1.numpy(), t2.numpy(), t3]))

    def test_stack_torch(self):
        """Test that concatenate, called without the axis arguments, concatenates across the 0th dimension"""
        t1 = onp.array([5.0, 8.0, 101.0], dtype=np.float64)
        t2 = torch.tensor([0.6, 0.1, 0.6], dtype=torch.float64)
        t3 = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)

        res = fn.concatenate([t1, t2, t3])
        assert isinstance(res, torch.Tensor)
        assert np.all(res.numpy() == np.concatenate([t1, t2.numpy(), t3.numpy()]))

    @pytest.mark.parametrize(
        "t1", [onp.array([[1], [2]]), torch.tensor([[1], [2]]), tf.constant([[1], [2]])]
    )
    def test_stack_axis(self, t1):
        """Test that passing the axis argument allows for concatenating along
        a different axis"""
        t2 = onp.array([[3], [4]])
        res = fn.concatenate([t1, t2], axis=1)

        # if tensorflow or pytorch, extract view of underlying data
        if hasattr(res, "numpy"):
            res = res.numpy()

        assert fn.allclose(res, np.array([[1, 3], [2, 4]]))
        assert list(res.shape) == [2, 2]

    @pytest.mark.parametrize(
        "t1", [onp.array([[1], [2]]), torch.tensor([[1], [2]]), tf.constant([[1], [2]])]
    )
    def test_concatenate_flattened_arrays(self, t1):
        """Concatenating arrays with axis=None will result in all arrays being pre-flattened"""
        t2 = onp.array([5])
        res = fn.concatenate([t1, t2], axis=None)

        # if tensorflow or pytorch, extract view of underlying data
        if hasattr(res, "numpy"):
            res = res.numpy()

        assert fn.allclose(res, np.array([1, 2, 5]))
        assert list(res.shape) == [3]


class TestConvertLike:
    """tests for the convert like function"""

    @pytest.mark.parametrize("t1,t2", list(itertools.combinations(test_data, r=2)))
    def test_convert_tensor_like(self, t1, t2):
        """Test that converting t1 like t2 results in t1 being cast to the same tensor type as t2"""
        res = fn.convert_like(t1, t2)

        # if tensorflow or pytorch, extract view of underlying data
        if hasattr(res, "numpy"):
            res = res.numpy()

        if hasattr(t2, "numpy"):
            t2 = t2.numpy()

        assert fn.allequal(res, t1)
        assert isinstance(res, np.ndarray if isinstance(t2, (list, tuple)) else t2.__class__)

    @pytest.mark.parametrize("t_like", [np.array([1]), tf.constant([1]), torch.tensor([1])])
    def test_convert_scalar(self, t_like):
        """Test that a python scalar is converted to a scalar tensor"""
        res = fn.convert_like(5, t_like)
        assert isinstance(res, t_like.__class__)
        assert res.ndim == 0
        assert fn.allequal(res, [5])


class TestDot:
    """Tests for the dot product function"""

    scalar_product_data = [
        [2, 6],
        [np.array(2), np.array(6)],
        [torch.tensor(2), onp.array(6)],
        [torch.tensor(2), torch.tensor(6)],
        [tf.Variable(2), onp.array(6)],
        [tf.constant(2), onp.array(6)],
        [tf.Variable(2), tf.Variable(6)],
        [jnp.array(2), jnp.array(6)],
    ]

    @pytest.mark.parametrize("t1, t2", scalar_product_data)
    def test_scalar_product(self, t1, t2):
        """Test that the dot product of two scalars results in a scalar"""
        res = fn.dot(t1, t2)
        assert fn.allequal(res, 12)

    vector_product_data = [
        [[1, 2, 3], [1, 2, 3]],
        [np.array([1, 2, 3]), np.array([1, 2, 3])],
        [torch.tensor([1, 2, 3]), onp.array([1, 2, 3])],
        [torch.tensor([1, 2, 3]), torch.tensor([1, 2, 3])],
        [tf.Variable([1, 2, 3]), onp.array([1, 2, 3])],
        [tf.constant([1, 2, 3]), onp.array([1, 2, 3])],
        [tf.Variable([1, 2, 3]), tf.Variable([1, 2, 3])],
        [jnp.array([1, 2, 3]), jnp.array([1, 2, 3])],
    ]

    @pytest.mark.parametrize("t1, t2", vector_product_data)
    def test_vector_product(self, t1, t2):
        """Test that the dot product of two vectors results in a scalar"""
        res = fn.dot(t1, t2)
        assert fn.allequal(res, 14)

    matrix_vector_product_data = [
        [[[1, 2], [3, 4]], [6, 7]],
        [np.array([[1, 2], [3, 4]]), np.array([6, 7])],
        [torch.tensor([[1, 2], [3, 4]]), onp.array([6, 7])],
        [torch.tensor([[1, 2], [3, 4]]), torch.tensor([6, 7])],
        [tf.Variable([[1, 2], [3, 4]]), onp.array([6, 7])],
        [tf.constant([[1, 2], [3, 4]]), onp.array([6, 7])],
        [tf.Variable([[1, 2], [3, 4]]), tf.Variable([6, 7])],
        [jnp.array([[1, 2], [3, 4]]), jnp.array([6, 7])],
        [onp.array([[1, 2], [3, 4]]), jnp.array([6, 7])],
    ]

    @pytest.mark.parametrize("t1, t2", matrix_vector_product_data)
    def test_matrix_vector_product(self, t1, t2):
        """Test that the matrix-vector dot product of two vectors results in a vector"""
        res = fn.dot(t1, t2)
        assert fn.allequal(res, [20, 46])

    @pytest.mark.parametrize("t1, t2", matrix_vector_product_data)
    def test_vector_matrix_product(self, t1, t2):
        """Test that the vector-matrix dot product of two vectors results in a vector"""
        res = fn.dot(t2, t1)
        assert fn.allequal(res, [27, 40])

    @pytest.mark.parametrize("t1, t2", matrix_vector_product_data)
    def test_matrix_matrix_product(self, t1, t2):
        """Test that the matrix-matrix dot product of two vectors results in a matrix"""
        res = fn.dot(t1, t1)
        assert fn.allequal(res, np.array([[7, 10], [15, 22]]))

    multidim_product_data = [
        [
            np.array([[[1, 2], [3, 4], [-1, 1]], [[5, 6], [0, -1], [2, 1]]]),
            np.array([[[1, 1], [3, 3]], [[3, 1], [3, 2]]]),
        ],
        [
            torch.tensor([[[1, 2], [3, 4], [-1, 1]], [[5, 6], [0, -1], [2, 1]]]),
            onp.array([[[1, 1], [3, 3]], [[3, 1], [3, 2]]]),
        ],
        [
            torch.tensor([[[1, 2], [3, 4], [-1, 1]], [[5, 6], [0, -1], [2, 1]]]),
            torch.tensor([[[1, 1], [3, 3]], [[3, 1], [3, 2]]]),
        ],
        [
            onp.array([[[1, 2], [3, 4], [-1, 1]], [[5, 6], [0, -1], [2, 1]]]),
            tf.Variable([[[1, 1], [3, 3]], [[3, 1], [3, 2]]]),
        ],
        [
            tf.constant([[[1, 2], [3, 4], [-1, 1]], [[5, 6], [0, -1], [2, 1]]]),
            onp.array([[[1, 1], [3, 3]], [[3, 1], [3, 2]]]),
        ],
        [
            tf.Variable([[[1, 2], [3, 4], [-1, 1]], [[5, 6], [0, -1], [2, 1]]]),
            tf.constant([[[1, 1], [3, 3]], [[3, 1], [3, 2]]]),
        ],
        [
            jnp.array([[[1, 2], [3, 4], [-1, 1]], [[5, 6], [0, -1], [2, 1]]]),
            jnp.array([[[1, 1], [3, 3]], [[3, 1], [3, 2]]]),
        ],
    ]

    @pytest.mark.parametrize("t1, t2", multidim_product_data)
    def test_multidimensional_product(self, t1, t2):
        """Test that the multi-dimensional dot product reduces across the last dimension of the first
        tensor, and the second-to-last dimension of the second tensor."""
        res = fn.dot(t1, t2)
        expected = np.array(
            [
                [[[7, 7], [9, 5]], [[15, 15], [21, 11]], [[2, 2], [0, 1]]],
                [[[23, 23], [33, 17]], [[-3, -3], [-3, -2]], [[5, 5], [9, 4]]],
            ]
        )
        assert fn.allequal(res, expected)


# the following test data is of the form
# [original shape, axis to expand, new shape]
expand_dims_test_data = [
    [tuple(), 0, (1,)],
    [(3,), 0, (1, 3)],
    [(3,), 1, (3, 1)],
    [(2, 2), 0, (1, 2, 2)],
    [(2, 2), 1, (2, 1, 2)],
    [(2, 2), 2, (2, 2, 1)],
]


@pytest.mark.parametrize("shape,axis,new_shape", expand_dims_test_data)
class TestExpandDims:
    """Tests for the expand_dims function"""

    def test_expand_dims_sequence(self, shape, axis, new_shape):
        """Test that expand_dimensions works correctly
        when given a sequence"""
        if not shape:
            pytest.skip("Cannot expand the dimensions of a Python scalar!")

        t1 = np.empty(shape).tolist()
        t2 = fn.expand_dims(t1, axis=axis)
        assert t2.shape == new_shape

    def test_expand_dims_array(self, shape, axis, new_shape):
        """Test that expand_dimensions works correctly
        when given an array"""
        t1 = np.empty(shape)
        t2 = fn.expand_dims(t1, axis=axis)
        assert t2.shape == new_shape
        assert isinstance(t2, np.ndarray)

    def test_expand_dims_torch(self, shape, axis, new_shape):
        """Test that the expand dimensions works correctly
        when given a torch tensor"""
        t1 = torch.empty(shape)
        t2 = fn.expand_dims(t1, axis=axis)
        assert t2.shape == new_shape
        assert isinstance(t2, torch.Tensor)

    def test_expand_dims_tf(self, shape, axis, new_shape):
        """Test that the expand dimensions works correctly
        when given a TF tensor"""
        t1 = tf.ones(shape)
        t2 = fn.expand_dims(t1, axis=axis)
        assert t2.shape == new_shape
        assert isinstance(t2, tf.Tensor)


interface_test_data = [
    [(1, 2, 3), "numpy"],
    [[1, 2, 3], "numpy"],
    [onp.array([1, 2, 3]), "numpy"],
    [np.array([1, 2, 3]), "autograd"],
    [torch.tensor([1, 2, 3]), "torch"],
    [tf.Variable([1, 2, 3]), "tensorflow"],
    [tf.constant([1, 2, 3]), "tensorflow"],
    [jnp.array([1, 2, 3]), "jax"],
]


@pytest.mark.parametrize("t,interface", interface_test_data)
def test_get_interface(t, interface):
    """Test that the interface of a tensor-like object

    is correctly returned."""
    res = fn.get_interface(t)
    assert res == interface


@pytest.mark.parametrize("t", test_data)
def test_toarray(t):
    """Test that the toarray method correctly converts the input
    tensor into a NumPy array."""
    res = fn.toarray(t)
    assert fn.allequal(res, t)
    assert isinstance(res, onp.ndarray)


@pytest.mark.parametrize("t", test_data)
def test_numpy(t):
    """Test that the to_numpy method correctly converts the input
    tensor into a NumPy array."""
    res = fn.to_numpy(t)
    assert fn.allequal(res, t)
    assert isinstance(res, onp.ndarray)


def test_numpy_jax_jit():
    """Test that the to_numpy() method raises an exception
    if used inside the JAX JIT"""

    @jax.jit
    def cost(x):
        fn.to_numpy(x)
        return x

    with pytest.raises(ValueError, match="not supported when using the JAX JIT"):
        cost(jnp.array(0.1))


class TestOnesLike:
    """Tests for the ones_like function"""

    @pytest.mark.parametrize("t", cast_like_test_data)
    def test_ones_like_inferred_dtype(self, t):
        """Test that the ones like function creates the correct
        shape and type tensor."""
        res = fn.ones_like(t)

        if isinstance(t, (list, tuple)):
            t = onp.asarray(t)

        assert res.shape == t.shape
        assert fn.get_interface(res) == fn.get_interface(t)
        assert fn.allclose(res, np.ones(t.shape))

        # if tensorflow or pytorch, extract view of underlying data
        if hasattr(res, "numpy"):
            res = res.numpy()
            t = t.numpy()

        assert onp.asarray(res).dtype.type is onp.asarray(t).dtype.type

    @pytest.mark.parametrize("t", cast_like_test_data)
    def test_ones_like_explicit_dtype(self, t):
        """Test that the ones like function creates the correct
        shape and type tensor."""
        res = fn.ones_like(t, dtype=np.float16)

        if isinstance(t, (list, tuple)):
            t = onp.asarray(t)

        assert res.shape == t.shape
        assert fn.get_interface(res) == fn.get_interface(t)
        assert fn.allclose(res, np.ones(t.shape))

        # if tensorflow or pytorch, extract view of underlying data
        if hasattr(res, "numpy"):
            res = res.numpy()
            t = t.numpy()

        assert onp.asarray(res).dtype.type is np.float16


class TestRequiresGrad:
    """Tests for the requires_grad function"""

    @pytest.mark.parametrize("t", [(1, 2, 3), [1, 2, 3], onp.array([1, 2, 3])])
    def test_numpy(self, t):
        """Vanilla NumPy arrays, sequences, and lists will always return False"""
        assert not fn.requires_grad(t)

    @pytest.mark.slow
    def test_jax(self):
        """JAX DeviceArrays differentiability depends on the argnums argument"""
        res = None

        def cost_fn(t, s):
            nonlocal res
            res = [fn.requires_grad(t), fn.requires_grad(s)]
            return jnp.sum(t * s)

        t = jnp.array([1.0, 2.0, 3.0])
        s = jnp.array([-2.0, -3.0, -4.0])

        jax.grad(cost_fn, argnums=0)(t, s)
        assert res == [True, False]

        jax.grad(cost_fn, argnums=1)(t, s)
        assert res == [False, True]

        jax.grad(cost_fn, argnums=[0, 1])(t, s)
        assert res == [True, True]

    def test_autograd(self):
        """Autograd arrays will simply return their requires_grad attribute"""
        t = np.array([1.0, 2.0], requires_grad=True)
        assert fn.requires_grad(t)

        t = np.array([1.0, 2.0], requires_grad=False)
        assert not fn.requires_grad(t)

    def test_autograd_backwards(self):
        """Autograd trainability corresponds to the requires_grad attribute during the backwards pass."""
        res = None

        def cost_fn(t, s):
            nonlocal res
            res = [fn.requires_grad(t), fn.requires_grad(s)]
            return np.sum(t * s)

        t = np.array([1.0, 2.0, 3.0])
        s = np.array([-2.0, -3.0, -4.0])

        qml.grad(cost_fn)(t, s)
        assert res == [True, True]

        t.requires_grad = False
        qml.grad(cost_fn)(t, s)
        assert res == [False, True]

        t.requires_grad = True
        s.requires_grad = False
        qml.grad(cost_fn)(t, s)
        assert res == [True, False]

        t.requires_grad = False
        s.requires_grad = False

        with pytest.warns(UserWarning, match="Output seems independent of input"):
            qml.grad(cost_fn)(t, s)

        assert res == [False, False]

    def test_torch(self):
        """Torch tensors will simply return their requires_grad attribute"""
        t = torch.tensor([1.0, 2.0], requires_grad=True)
        assert fn.requires_grad(t)

        t = torch.tensor([1.0, 2.0], requires_grad=False)
        assert not fn.requires_grad(t)

    def test_tf(self):
        """TensorFlow tensors will True *if* they are being watched by a gradient tape"""
        t1 = tf.Variable([1.0, 2.0])
        t2 = tf.constant([1.0, 2.0])
        assert not fn.requires_grad(t1)
        assert not fn.requires_grad(t2)

        with tf.GradientTape():
            # variables are automatically watched within a context,
            # but constants are not
            assert fn.requires_grad(t1)
            assert not fn.requires_grad(t2)

        with tf.GradientTape() as tape:
            # watching makes all tensors trainable
            tape.watch([t1, t2])
            assert fn.requires_grad(t1)
            assert fn.requires_grad(t2)

    def test_unknown_interface(self):
        """Test that an error is raised if the interface is unknown"""
        with pytest.raises(ValueError, match="unknown object"):
            fn.requires_grad(type("hello", tuple(), {})())


shape_test_data = [
    tuple(),
    (3,),
    (2, 2),
    (3, 2, 2),
    (2, 1, 1, 2),
]


@pytest.mark.parametrize(
    "interface,create_array",
    [
        ("sequence", lambda shape: np.empty(shape).tolist()),
        ("autograd", np.empty),
        ("torch", torch.empty),
        ("jax", jnp.ones),
        ("tf", tf.ones),
    ],
)
@pytest.mark.parametrize("shape", shape_test_data)
def test_shape(shape, interface, create_array):
    """Test that the shape of tensors is correctly returned"""
    if interface == "sequence" and not shape:
        pytest.skip("Cannot expand the dimensions of a Python scalar!")

    t = create_array(shape)
    assert fn.shape(t) == shape


@pytest.mark.parametrize("t", test_data)
def test_sqrt(t):
    """Test that the square root function works for a variety
    of input"""
    res = fn.sqrt(t)
    assert fn.allclose(res, [1, np.sqrt(2), np.sqrt(3)])


class TestStack:
    """Tests for the stack function"""

    def test_stack_array(self):
        """Test that stack, called without the axis arguments, stacks vertically"""
        t1 = [0.6, 0.1, 0.6]
        t2 = np.array([0.1, 0.2, 0.3])
        t3 = onp.array([5.0, 8.0, 101.0])

        res = fn.stack([t1, t2, t3])
        assert isinstance(res, np.ndarray)
        assert np.all(res == np.stack([t1, t2, t3]))

    def test_stack_array_jax(self):
        """Test that stack, called without the axis arguments, stacks vertically"""
        t1 = onp.array([0.6, 0.1, 0.6])
        t2 = jnp.array([0.1, 0.2, 0.3])
        t3 = jnp.array([5.0, 8.0, 101.0])

        res = fn.stack([t1, t2, t3])
        assert np.all(res == np.stack([t1, t2, t3]))

    def test_stack_tensorflow(self):
        """Test that stack, called without the axis arguments, stacks vertically"""
        t1 = tf.constant([0.6, 0.1, 0.6])
        t2 = tf.Variable([0.1, 0.2, 0.3])
        t3 = onp.array([5.0, 8.0, 101.0])

        res = fn.stack([t1, t2, t3])
        assert isinstance(res, tf.Tensor)
        assert np.all(res.numpy() == np.stack([t1.numpy(), t2.numpy(), t3]))

    def test_stack_torch(self):
        """Test that stack, called without the axis arguments, stacks vertically"""
        t1 = onp.array([5.0, 8.0, 101.0], dtype=np.float64)
        t2 = torch.tensor([0.6, 0.1, 0.6], dtype=torch.float64)
        t3 = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)

        res = fn.stack([t1, t2, t3])
        assert isinstance(res, torch.Tensor)
        assert np.all(res.numpy() == np.stack([t1, t2.numpy(), t3.numpy()]))

    @pytest.mark.parametrize("t1", [onp.array([1, 2]), torch.tensor([1, 2]), tf.constant([1, 2])])
    def test_stack_axis(self, t1):
        """Test that passing the axis argument allows for stacking along
        a different axis"""
        t2 = onp.array([3, 4])
        res = fn.stack([t1, t2], axis=1)

        # if tensorflow or pytorch, extract view of underlying data
        if hasattr(res, "numpy"):
            res = res.numpy()

        assert fn.allclose(res, np.array([[1, 3], [2, 4]]))
        assert list(res.shape) == [2, 2]


class TestSum:
    """Tests for the summation function"""

    def test_array(self):
        """Test that sum, called without the axis arguments, returns a scalar"""
        t = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        res = fn.sum(t)
        assert isinstance(res, np.ndarray)
        assert fn.allclose(res, 2.1)

    def test_tensorflow(self):
        """Test that sum, called without the axis arguments, returns a scalar"""
        t = tf.Variable([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        res = fn.sum(t)
        assert isinstance(res, tf.Tensor)
        assert fn.allclose(res, 2.1)

    def test_torch(self):
        """Test that sum, called without the axis arguments, returns a scalar"""
        t = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        res = fn.sum(t)
        assert isinstance(res, torch.Tensor)
        assert fn.allclose(res, 2.1)

    def test_jax(self):
        """Test that sum, called without the axis arguments, returns a scalar"""
        t = jnp.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        res = fn.sum(t)
        assert fn.allclose(res, 2.1)

    @pytest.mark.parametrize(
        "t1",
        [
            np.array([[[1, 2], [3, 4], [-1, 1]], [[5, 6], [0, -1], [2, 1]]]),
            torch.tensor([[[1, 2], [3, 4], [-1, 1]], [[5, 6], [0, -1], [2, 1]]]),
            tf.constant([[[1, 2], [3, 4], [-1, 1]], [[5, 6], [0, -1], [2, 1]]]),
            jnp.array([[[1, 2], [3, 4], [-1, 1]], [[5, 6], [0, -1], [2, 1]]]),
        ],
    )
    def test_sum_axis(self, t1):
        """Test that passing the axis argument allows for summing along
        a specific axis"""
        res = fn.sum(t1, axis=(0, 2))

        # if tensorflow or pytorch, extract view of underlying data
        if hasattr(res, "numpy"):
            res = res.numpy()

        assert fn.allclose(res, np.array([14, 6, 3]))
        assert res.shape == (3,)

    @pytest.mark.parametrize(
        "t1",
        [
            np.array([[[1, 2], [3, 4], [-1, 1]], [[5, 6], [0, -1], [2, 1]]]),
            torch.tensor([[[1, 2], [3, 4], [-1, 1]], [[5, 6], [0, -1], [2, 1]]]),
            tf.constant([[[1, 2], [3, 4], [-1, 1]], [[5, 6], [0, -1], [2, 1]]]),
            jnp.array([[[1, 2], [3, 4], [-1, 1]], [[5, 6], [0, -1], [2, 1]]]),
        ],
    )
    def test_sum_axis_keepdims(self, t1):
        """Test that passing the axis argument allows for summing along
        a specific axis, while keepdims avoids the summed dimensions from being removed"""
        res = fn.sum(t1, axis=(0, 2), keepdims=True)

        # if tensorflow or pytorch, extract view of underlying data
        if hasattr(res, "numpy"):
            res = res.numpy()

        assert fn.allclose(res, np.array([[[14], [6], [3]]]))
        assert res.shape == (1, 3, 1)


@pytest.mark.parametrize("t", test_data)
def test_T(t):
    """Test the simple transpose (T) function"""
    res = fn.T(t)

    if isinstance(t, (list, tuple)):
        t = onp.asarray(t)

    assert fn.get_interface(res) == fn.get_interface(t)

    # if tensorflow or pytorch, extract view of underlying data
    if hasattr(res, "numpy"):
        res = res.numpy()
        t = t.numpy()

    assert np.all(res.T == t.T)


class TestTake:
    """Tests for the qml.take function"""

    take_data = [
        np.array([[[1, 2], [3, 4], [-1, 1]], [[5, 6], [0, -1], [2, 1]]]),
        torch.tensor([[[1, 2], [3, 4], [-1, 1]], [[5, 6], [0, -1], [2, 1]]]),
        onp.array([[[1, 2], [3, 4], [-1, 1]], [[5, 6], [0, -1], [2, 1]]]),
        tf.constant([[[1, 2], [3, 4], [-1, 1]], [[5, 6], [0, -1], [2, 1]]]),
        tf.Variable([[[1, 2], [3, 4], [-1, 1]], [[5, 6], [0, -1], [2, 1]]]),
        jnp.asarray([[[1, 2], [3, 4], [-1, 1]], [[5, 6], [0, -1], [2, 1]]]),
    ]

    @pytest.mark.parametrize("t", take_data)
    def test_flattened_indexing(self, t):
        """Test that indexing without the axis argument
        will flatten the tensor first"""
        indices = 5
        res = fn.take(t, indices)
        assert fn.allclose(res, 1)

    @pytest.mark.parametrize("t", take_data)
    def test_array_indexing(self, t):
        """Test that indexing with a sequence properly extracts
        the elements from the flattened tensor"""
        indices = [0, 2, 3, 6, -2]
        res = fn.take(t, indices)
        assert fn.allclose(res, [1, 3, 4, 5, 2])

    @pytest.mark.parametrize("t", take_data)
    def test_multidimensional_indexing(self, t):
        """Test that indexing with a multi-dimensional sequence properly extracts
        the elements from the flattened tensor"""
        indices = [[0, 1], [3, 2]]
        res = fn.take(t, indices)
        assert fn.allclose(res, [[1, 2], [4, 3]])

    @pytest.mark.parametrize("t", take_data)
    def test_array_indexing_along_axis(self, t):
        """Test that indexing with a sequence properly extracts
        the elements from the specified tensor axis"""
        indices = [0, 1, -2]
        res = fn.take(t, indices, axis=2)
        expected = np.array(
            [[[1, 2, 1], [3, 4, 3], [-1, 1, -1]], [[5, 6, 5], [0, -1, 0], [2, 1, 2]]]
        )
        assert fn.allclose(res, expected)

    @pytest.mark.parametrize("t", take_data)
    def test_array_indexing_along_axis(self, t):
        """Test that indexing with a sequence properly extracts
        the elements from the specified tensor axis"""
        indices = [0, 1, -2]
        res = fn.take(t, indices, axis=2)
        expected = np.array(
            [[[1, 2, 1], [3, 4, 3], [-1, 1, -1]], [[5, 6, 5], [0, -1, 0], [2, 1, 2]]]
        )
        assert fn.allclose(res, expected)

    @pytest.mark.parametrize("t", take_data)
    def test_multidimensional_indexing_along_axis(self, t):
        """Test that indexing with a sequence properly extracts
        the elements from the specified tensor axis"""
        indices = np.array([[0, 0], [1, 0]])
        res = fn.take(t, indices, axis=1)
        expected = np.array(
            [[[[1, 2], [1, 2]], [[3, 4], [1, 2]]], [[[5, 6], [5, 6]], [[0, -1], [5, 6]]]]
        )
        assert fn.allclose(res, expected)

    def test_multidimensional_indexing_along_axis_autograd(self):
        """Test that indexing with a sequence properly extracts
        the elements from the specified tensor axis"""
        t = np.array([[[1, 2], [3, 4], [-1, 1]], [[5, 6], [0, -1], [2, 1]]])
        indices = np.array([[0, 0], [1, 0]])

        def cost_fn(t):
            return fn.sum(fn.take(t, indices, axis=1))

        res = cost_fn(t)
        expected = np.sum(
            np.array([[[[1, 2], [1, 2]], [[3, 4], [1, 2]]], [[[5, 6], [5, 6]], [[0, -1], [5, 6]]]])
        )
        assert fn.allclose(res, expected)

        grad = qml.grad(cost_fn)(t)
        expected = np.array([[[3, 3], [1, 1], [0, 0]], [[3, 3], [1, 1], [0, 0]]])
        assert fn.allclose(grad, expected)


where_data = [
    np.array([[[1, 2], [3, 4], [-1, 1]], [[5, 6], [0, -1], [2, 1]]]),
    torch.tensor([[[1, 2], [3, 4], [-1, 1]], [[5, 6], [0, -1], [2, 1]]]),
    onp.array([[[1, 2], [3, 4], [-1, 1]], [[5, 6], [0, -1], [2, 1]]]),
    tf.constant([[[1, 2], [3, 4], [-1, 1]], [[5, 6], [0, -1], [2, 1]]]),
    tf.Variable([[[1, 2], [3, 4], [-1, 1]], [[5, 6], [0, -1], [2, 1]]]),
    jnp.array([[[1, 2], [3, 4], [-1, 1]], [[5, 6], [0, -1], [2, 1]]]),
]


@pytest.mark.parametrize("t", where_data)
def test_where(t):
    """Test that the where function works as expected"""
    res = fn.where(t < 0, 100 * fn.ones_like(t), t)
    expected = np.array([[[1, 2], [3, 4], [100, 1]], [[5, 6], [0, 100], [2, 1]]])
    assert fn.allclose(res, expected)


squeeze_data = [
    np.ones((1, 2, 3, 1, 5, 1)),
    torch.ones((1, 2, 3, 1, 5, 1)),
    tf.ones((1, 2, 3, 1, 5, 1)),
    jnp.ones((1, 2, 3, 1, 5, 1)),
    onp.ones((1, 2, 3, 1, 5, 1)),
]


@pytest.mark.parametrize("t", squeeze_data)
def test_squeeze(t):
    """Test that the squeeze function works as expected"""
    res = fn.squeeze(t)
    assert res.shape == (2, 3, 5)


class TestScatterElementAdd:
    """Tests for the scatter_element_add function"""

    def test_array(self):
        """Test that a NumPy array is differentiable when using scatter addition"""
        x = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], requires_grad=True)
        y = np.array(0.56, requires_grad=True)

        def cost(weights):
            return fn.scatter_element_add(weights[0], [1, 2], weights[1] ** 2)

        res = cost([x, y])
        assert isinstance(res, np.ndarray)
        assert fn.allclose(res, onp.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.3136]]))

        grad = qml.grad(lambda weights: cost(weights)[1, 2])([x, y])
        assert fn.allclose(grad[0], onp.array([[0, 0, 0], [0, 0, 1.0]]))
        assert fn.allclose(grad[1], 2 * y)

    def test_array_multi(self):
        """Test that a NumPy array and the addend are differentiable when using
        scatter addition (multi dispatch)."""
        x = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], requires_grad=True)
        y = np.array(0.56, requires_grad=True)

        def cost_multi(weight_0, weight_1):
            return fn.scatter_element_add(weight_0, [1, 2], weight_1 ** 2)

        res = cost_multi(x, y)
        assert isinstance(res, np.ndarray)
        assert fn.allclose(res, onp.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.3136]]))

        jac = qml.jacobian(lambda *weights: cost_multi(*weights))(x, y)
        assert fn.allclose(jac[0], onp.eye(6).reshape((2, 3, 2, 3)))
        assert fn.allclose(jac[1], onp.array([[0, 0, 0], [0, 0, 2 * y]]))

    def test_tensorflow(self):
        """Test that a TF tensor is differentiable when using scatter addition"""
        x = tf.Variable([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
        y = tf.Variable(0.56)

        with tf.GradientTape() as tape:
            res = fn.scatter_element_add(x, [1, 2], y ** 2)
            loss = res[1, 2]

        assert isinstance(res, tf.Tensor)
        assert fn.allclose(res, onp.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.3136]]))

        grad = tape.gradient(loss, [x, y])
        assert fn.allclose(grad[0], onp.array([[0, 0, 0], [0, 0, 1.0]]))
        assert fn.allclose(grad[1], 2 * y)

    def test_torch(self):
        """Test that a torch tensor is differentiable when using scatter addition"""
        x = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], requires_grad=True)
        y = torch.tensor(0.56, requires_grad=True)

        res = fn.scatter_element_add(x, [1, 2], y ** 2)
        loss = res[1, 2]

        assert isinstance(res, torch.Tensor)
        assert fn.allclose(res.detach(), onp.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.3136]]))

        loss.backward()
        assert fn.allclose(x.grad, onp.array([[0, 0, 0], [0, 0, 1.0]]))
        assert fn.allclose(y.grad, 2 * y)

    def test_jax(self):
        """Test that a JAX array is differentiable when using scatter addition"""
        x = jnp.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
        y = jnp.array(0.56)

        def cost(weights):
            return fn.scatter_element_add(weights[0], [1, 2], weights[1] ** 2)

        res = cost([x, y])
        assert isinstance(res, jax.interpreters.xla.DeviceArray)
        assert fn.allclose(res, onp.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.3136]]))

        grad = jax.grad(lambda weights: cost(weights)[1, 2])([x, y])
        assert fn.allclose(grad[0], onp.array([[0, 0, 0], [0, 0, 1.0]]))
        assert fn.allclose(grad[1], 2 * y)

    def test_jax_multi(self):
        """Test that a NumPy array and the addend are differentiable when using
        scatter addition (multi dispatch)."""
        x = jnp.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
        y = jnp.array(0.56)

        def cost_multi(weight_0, weight_1):
            return fn.scatter_element_add(weight_0, [1, 2], weight_1 ** 2)

        res = cost_multi(x, y)
        assert isinstance(res, jax.interpreters.xla.DeviceArray)
        assert fn.allclose(res, onp.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.3136]]))

        jac = jax.jacobian(lambda *weights: cost_multi(*weights), argnums=[0, 1])(x, y)
        assert fn.allclose(jac[0], onp.eye(6).reshape((2, 3, 2, 3)))
        assert fn.allclose(jac[1], onp.array([[0, 0, 0], [0, 0, 2 * y]]))


class TestDiag:
    """Tests for the diag function"""

    @pytest.mark.parametrize(
        "a, interface",
        [
            [np.array(0.5), "autograd"],
            [tf.Variable(0.5), "tensorflow"],
            [torch.tensor(0.5), "torch"],
        ],
    )
    def test_sequence(self, a, interface):
        """Test that a sequence is automatically converted into
        a diagonal tensor"""
        t = [0.1, 0.2, a]
        res = fn.diag(t)
        assert fn.get_interface(res) == interface
        assert fn.allclose(res, onp.diag([0.1, 0.2, 0.5]))

    def test_array(self):
        """Test that a NumPy array is automatically converted into
        a diagonal tensor"""
        t = np.array([0.1, 0.2, 0.3])
        res = fn.diag(t)
        assert isinstance(res, np.ndarray)
        assert fn.allclose(res, onp.diag([0.1, 0.2, 0.3]))

        res = fn.diag(t, k=1)
        assert fn.allclose(res, onp.diag([0.1, 0.2, 0.3], k=1))

    def test_tensorflow(self):
        """Test that a tensorflow tensor is automatically converted into
        a diagonal tensor"""
        t = tf.Variable([0.1, 0.2, 0.3])
        res = fn.diag(t)
        assert isinstance(res, tf.Tensor)
        assert fn.allclose(res, onp.diag([0.1, 0.2, 0.3]))

        res = fn.diag(t, k=1)
        assert fn.allclose(res, onp.diag([0.1, 0.2, 0.3], k=1))

    def test_torch(self):
        """Test that a torch tensor is automatically converted into
        a diagonal tensor"""
        t = torch.tensor([0.1, 0.2, 0.3])
        res = fn.diag(t)
        assert isinstance(res, torch.Tensor)
        assert fn.allclose(res, onp.diag([0.1, 0.2, 0.3]))

        res = fn.diag(t, k=1)
        assert fn.allclose(res, onp.diag([0.1, 0.2, 0.3], k=1))

    def test_jax(self):
        """Test that a jax array is automatically converted into
        a diagonal tensor"""
        t = jnp.array([0.1, 0.2, 0.3])
        res = fn.diag(t)
        assert fn.allclose(res, onp.diag([0.1, 0.2, 0.3]))

        res = fn.diag(t, k=1)
        assert fn.allclose(res, onp.diag([0.1, 0.2, 0.3], k=1))


class TestCovMatrix:
    """Tests for the cov matrix function"""

    obs_list = [qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliY(2)]

    @staticmethod
    def ansatz(weights, wires):
        """Circuit ansatz for testing"""
        qml.RY(weights[0], wires=wires[0])
        qml.RX(weights[1], wires=wires[1])
        qml.RX(weights[2], wires=wires[2])
        qml.CNOT(wires=[wires[0], wires[1]])
        qml.CNOT(wires=[wires[1], wires[2]])

    @staticmethod
    def expected_cov(weights):
        """Analytic covariance matrix for ansatz and obs_list"""
        a, b, c = weights
        return np.array(
            [
                [np.sin(b) ** 2, -np.cos(a) * np.sin(b) ** 2 * np.sin(c)],
                [
                    -np.cos(a) * np.sin(b) ** 2 * np.sin(c),
                    1 - np.cos(a) ** 2 * np.cos(b) ** 2 * np.sin(c) ** 2,
                ],
            ]
        )

    @staticmethod
    def expected_grad(weights):
        """Analytic covariance matrix gradient for ansatz and obs_list"""
        a, b, c = weights
        return np.array(
            [
                np.sin(a) * np.sin(b) ** 2 * np.sin(c),
                -2 * np.cos(a) * np.cos(b) * np.sin(b) * np.sin(c),
                -np.cos(a) * np.cos(c) * np.sin(b) ** 2,
            ]
        )

    def test_weird_wires(self, tol):
        """Test that the covariance matrix computes the correct
        result when weird wires are used"""
        dev = qml.device("default.qubit", wires=["a", -1, "q"])
        obs_list = [qml.PauliZ("a") @ qml.PauliZ(-1), qml.PauliY("q")]

        @qml.qnode(dev, interface="autograd")
        def circuit(weights):
            """Returns the shared probability distribution of ansatz
            in the joint basis for obs_list"""
            self.ansatz(weights, wires=dev.wires)

            for o in obs_list:
                o.diagonalizing_gates()

            return qml.probs(wires=dev.wires)

        def cov(weights):
            probs = circuit(weights)
            return fn.cov_matrix(probs, obs_list, wires=dev.wires)

        weights = np.array([0.1, 0.2, 0.3])
        res = cov(weights)
        expected = self.expected_cov(weights)
        assert np.allclose(res, expected, atol=tol, rtol=0)

        grad_fn = qml.grad(lambda weights: cov(weights)[0, 1])
        res = grad_fn(weights)
        expected = self.expected_grad(weights)
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_autograd(self, tol):
        """Test that the covariance matrix computes the correct
        result, and is differentiable, using the Autograd interface"""
        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev, interface="autograd")
        def circuit(weights):
            """Returns the shared probability distribution of ansatz
            in the joint basis for obs_list"""
            self.ansatz(weights, wires=dev.wires)

            for o in self.obs_list:
                o.diagonalizing_gates()

            return qml.probs(wires=[0, 1, 2])

        def cov(weights):
            probs = circuit(weights)
            return fn.cov_matrix(probs, self.obs_list)

        weights = np.array([0.1, 0.2, 0.3])
        res = cov(weights)
        expected = self.expected_cov(weights)
        assert np.allclose(res, expected, atol=tol, rtol=0)

        grad_fn = qml.grad(lambda weights: cov(weights)[0, 1])
        res = grad_fn(weights)
        expected = self.expected_grad(weights)
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_torch(self, tol):
        """Test that the covariance matrix computes the correct
        result, and is differentiable, using the Torch interface"""
        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev, interface="torch")
        def circuit(weights):
            """Returns the shared probability distribution of ansatz
            in the joint basis for obs_list"""
            self.ansatz(weights, wires=dev.wires)

            for o in self.obs_list:
                o.diagonalizing_gates()

            return qml.probs(wires=[0, 1, 2])

        weights = np.array([0.1, 0.2, 0.3])
        weights_t = torch.tensor(weights, requires_grad=True)
        probs = circuit(weights_t)
        res = fn.cov_matrix(probs, self.obs_list)
        expected = self.expected_cov(weights)
        assert np.allclose(res.detach().numpy(), expected, atol=tol, rtol=0)

        loss = res[0, 1]
        loss.backward()
        res = weights_t.grad
        expected = self.expected_grad(weights)
        assert np.allclose(res.detach().numpy(), expected, atol=tol, rtol=0)

    def test_tf(self, tol):
        """Test that the covariance matrix computes the correct
        result, and is differentiable, using the TF interface"""
        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev, interface="tf")
        def circuit(weights):
            """Returns the shared probability distribution of ansatz
            in the joint basis for obs_list"""
            self.ansatz(weights, wires=dev.wires)

            for o in self.obs_list:
                o.diagonalizing_gates()

            return qml.probs(wires=[0, 1, 2])

        weights = np.array([0.1, 0.2, 0.3])
        weights_t = tf.Variable(weights)

        with tf.GradientTape() as tape:
            probs = circuit(weights_t)
            cov = fn.cov_matrix(probs, self.obs_list)
            loss = cov[0, 1]

        expected = self.expected_cov(weights)
        assert np.allclose(cov, expected, atol=tol, rtol=0)

        grad = tape.gradient(loss, weights_t)
        expected = self.expected_grad(weights)
        assert np.allclose(grad, expected, atol=tol, rtol=0)

    @pytest.mark.slow
    def test_jax(self, tol):
        """Test that the covariance matrix computes the correct
        result, and is differentiable, using the JAX interface"""
        dev = qml.device("default.qubit.jax", wires=3)

        @qml.qnode(dev, interface="jax", diff_method="backprop")
        def circuit(weights):
            """Returns the shared probability distribution of ansatz
            in the joint basis for obs_list"""
            self.ansatz(weights, wires=dev.wires)

            for o in self.obs_list:
                o.diagonalizing_gates()

            return qml.probs(wires=[0, 1, 2])

        def cov(weights):
            probs = circuit(weights)
            return fn.cov_matrix(probs, self.obs_list)

        weights = jnp.array([0.1, 0.2, 0.3])
        res = cov(weights)
        expected = self.expected_cov(weights)
        assert jnp.allclose(res, expected, atol=tol, rtol=0)

        grad_fn = jax.grad(lambda weights: cov(weights)[0, 1])
        res = grad_fn(weights)
        expected = self.expected_grad(weights)
        assert jnp.allclose(res, expected, atol=tol, rtol=0)


block_diag_data = [
    [onp.array([[1, 2], [3, 4]]), torch.tensor([[1, 2], [-1, -6]]), torch.tensor([[5]])],
    [onp.array([[1, 2], [3, 4]]), tf.Variable([[1, 2], [-1, -6]]), tf.constant([[5]])],
    [np.array([[1, 2], [3, 4]]), np.array([[1, 2], [-1, -6]]), np.array([[5]])],
    [jnp.array([[1, 2], [3, 4]]), jnp.array([[1, 2], [-1, -6]]), jnp.array([[5]])],
]


@pytest.mark.parametrize("tensors", block_diag_data)
def test_block_diag(tensors):
    """Tests for the block diagonal function"""
    res = fn.block_diag(tensors)
    expected = np.array(
        [[1, 2, 0, 0, 0], [3, 4, 0, 0, 0], [0, 0, 1, 2, 0], [0, 0, -1, -6, 0], [0, 0, 0, 0, 5]]
    )
    assert fn.allclose(res, expected)


class TestBlockDiagDiffability:

    expected = lambda self, x, y: np.array(
        [
            [[-np.sin(x * y) * y, -np.sin(x * y) * x], [0, 0], [0, 0]],
            [[0, 0], [1.0, 0.0], [0, 1.2]],
            [[0, 0], [2 * x, -1 / 3], [-1 / y, x / y ** 2]],
        ]
    )

    def test_autograd(self):
        """Tests for differentiating the block diagonal function with autograd."""
        tensors = lambda x, y: [
            np.array([[fn.cos(x * y)]]),
            np.array([[x, 1.2 * y], [x ** 2 - y / 3, -x / y]]),
        ]
        f = lambda x, y: fn.block_diag(tensors(x, y))
        x, y = 0.2, 1.5
        res = qml.jacobian(f)(x, y)
        exp = self.expected(x, y)
        # Transposes in the following because autograd behaves strangely
        assert fn.allclose(res[:, :, 0].T, exp[:, :, 0])
        assert fn.allclose(res[:, :, 1].T, exp[:, :, 1])

    def test_jax(self):
        """Tests for differentiating the block diagonal function with JAX."""
        jax = pytest.importorskip("jax")
        tensors = lambda x, y: [
            jnp.array([[fn.cos(x * y)]]),
            jnp.array([[x, 1.2 * y], [x ** 2 - y / 3, -x / y]]),
        ]
        f = lambda x, y: fn.block_diag(tensors(x, y))
        x, y = 0.2, 1.5
        res = jax.jacobian(f, argnums=[0, 1])(x, y)
        exp = self.expected(x, y)
        assert fn.allclose(exp[:, :, 0], res[0])
        assert fn.allclose(exp[:, :, 1], res[1])

    def test_tf(self):
        """Tests for differentiating the block diagonal function with Tensorflow."""
        tf = pytest.importorskip("tensorflow")
        x, y = [tf.Variable([[0.2]]), tf.Variable([[0.1, 0.2], [0.3, 0.4]])]
        with tf.GradientTape() as tape:
            out = fn.block_diag([x, y])
        res = tape.jacobian(out, (x, y))
        exp_0 = np.zeros((3, 3, 1, 1))
        exp_0[0, 0, 0, 0] = 1.0
        exp_1 = np.zeros((3, 3, 2, 2))
        exp_1[1, 1, 0, 0] = exp_1[1, 2, 0, 1] = exp_1[2, 1, 1, 0] = exp_1[2, 2, 1, 1] = 1.0
        assert fn.allclose(exp_0, res[0])
        assert fn.allclose(exp_1, res[1])

    def test_torch(self):
        """Tests for differentiating the block diagonal function with Torch."""
        torch = pytest.importorskip("torch")
        x, y = [torch.tensor([[0.2]]), torch.tensor([[0.1, 0.2], [0.3, 0.4]])]
        f = lambda x, y: fn.block_diag([x, y])
        res = torch.autograd.functional.jacobian(f, (x, y))
        exp_0 = np.zeros((3, 3, 1, 1))
        exp_0[0, 0, 0, 0] = 1.0
        exp_1 = np.zeros((3, 3, 2, 2))
        exp_1[1, 1, 0, 0] = exp_1[1, 2, 0, 1] = exp_1[2, 1, 1, 0] = exp_1[2, 2, 1, 1] = 1.0
        assert fn.allclose(exp_0, res[0])
        assert fn.allclose(exp_1, res[1])


gather_data = [
    torch.tensor([[1, 2, 3], [-1, -6, -3]]),
    tf.Variable([[1, 2, 3], [-1, -6, -3]]),
    jnp.array([[1, 2, 3], [-1, -6, -3]]),
    np.array([[1, 2, 3], [-1, -6, -3]]),
]


@pytest.mark.parametrize("tensor", gather_data)
def test_gather(tensor):
    """Tests for the gather function"""
    indices = [1, 0]
    res = fn.gather(tensor, indices)
    expected = np.array([[-1, -6, -3], [1, 2, 3]])
    assert fn.allclose(res, expected)


class TestCoercion:
    """Test that TensorFlow and PyTorch correctly coerce types"""

    def test_tensorflow_coercion(self):
        """Test tensorflow coercion"""
        tensors = [tf.Variable([0.2]), np.array([1, 2, 3]), tf.constant(1 + 3j, dtype=tf.complex64)]
        res = qml.math.coerce(tensors, like="tensorflow")
        dtypes = [r.dtype for r in res]
        assert all(d is tf.complex64 for d in dtypes)

    def test_torch_coercion(self):
        """Test Torch coercion"""
        tensors = [
            torch.tensor([0.2]),
            np.array([1, 2, 3]),
            torch.tensor(1 + 3j, dtype=torch.complex64),
        ]
        res = qml.math.coerce(tensors, like="torch")
        dtypes = [r.dtype for r in res]
        assert all(d is torch.complex64 for d in dtypes)


class TestUnwrap:
    """Test tensor unwrapping"""

    def test_tensorflow_unwrapping(self):
        """Test that a sequence of TensorFlow values is properly unwrapped"""
        values = [
            onp.array([0.1, 0.2]),
            tf.Variable(0.1, dtype=tf.float64),
            tf.constant([0.5, 0.2]),
        ]
        res = qml.math.unwrap(values)
        expected = [np.array([0.1, 0.2]), 0.1, np.array([0.5, 0.2])]
        assert all(np.allclose(a, b) for a, b in zip(res, expected))

    def test_torch_unwrapping(self):
        """Test that a sequence of Torch values is properly unwrapped"""
        values = [
            onp.array([0.1, 0.2]),
            torch.tensor(0.1, dtype=torch.float64),
            torch.tensor([0.5, 0.2]),
        ]
        res = qml.math.unwrap(values)
        expected = [np.array([0.1, 0.2]), 0.1, np.array([0.5, 0.2])]
        assert all(np.allclose(a, b) for a, b in zip(res, expected))

    def test_autograd_unwrapping_forward(self):
        """Test that a sequence of Autograd values is properly unwrapped
        during the forward pass"""
        unwrapped_params = None

        def cost_fn(params):
            nonlocal unwrapped_params
            unwrapped_params = qml.math.unwrap(params)
            return np.sum(np.sin(params[0] * params[2])) + params[1]

        values = [onp.array([0.1, 0.2]), np.tensor(0.1, dtype=np.float64), np.tensor([0.5, 0.2])]
        cost_fn(values)

        expected = [np.array([0.1, 0.2]), 0.1, np.array([0.5, 0.2])]
        assert all(np.allclose(a, b) for a, b in zip(unwrapped_params, expected))
        assert all(not isinstance(a, np.tensor) for a in unwrapped_params)

    def test_autograd_unwrapping_backward(self):
        """Test that a sequence of Autograd values is properly unwrapped
        during the backward pass"""
        unwrapped_params = None

        def cost_fn(*params):
            nonlocal unwrapped_params
            unwrapped_params = qml.math.unwrap(params)
            return np.sum(np.sin(params[0] * params[2])) + params[1]

        values = [onp.array([0.1, 0.2]), np.tensor(0.1, dtype=np.float64), np.tensor([0.5, 0.2])]
        grad = qml.grad(cost_fn)(*values)

        expected = [np.array([0.1, 0.2]), 0.1, np.array([0.5, 0.2])]
        assert all(np.allclose(a, b) for a, b in zip(unwrapped_params, expected))
        assert all(not isinstance(a, ArrayBox) for a in unwrapped_params)

    def test_autograd_unwrapping_backward_nested(self):
        """Test that a sequence of Autograd values is properly unwrapped
        during multiple backward passes"""
        unwrapped_params = None

        def cost_fn(p, max_depth=None):
            nonlocal unwrapped_params
            unwrapped_params = qml.math.unwrap(p, max_depth)
            return np.sum(np.sin(np.prod(p)))

        values = np.tensor([0.1, 0.2, 0.3])
        hess = qml.jacobian(qml.grad(cost_fn))(values)

        expected = np.array([0.1, 0.2, 0.3])
        assert np.allclose(unwrapped_params, expected)
        assert not isinstance(unwrapped_params, ArrayBox)

        # Specifying max_depth=1 will result in the second backward
        # pass not being unwrapped
        hess = qml.jacobian(qml.grad(cost_fn))(values, max_depth=1)
        assert all(isinstance(a, ArrayBox) for a in unwrapped_params)

    def test_jax_unwrapping(self):
        """Test that a sequence of Autograd values is properly unwrapped
        during the forward pass"""
        unwrapped_params = None

        def cost_fn(params):
            nonlocal unwrapped_params
            unwrapped_params = qml.math.unwrap(params)
            return np.sum(np.sin(params[0])) + params[2]

        values = [jnp.array([0.1, 0.2]), onp.array(0.1, dtype=np.float64), jnp.array([0.5, 0.2])]
        cost_fn(values)

        expected = [np.array([0.1, 0.2]), 0.1, np.array([0.5, 0.2])]
        assert all(np.allclose(a, b) for a, b in zip(unwrapped_params, expected))
        assert all(not isinstance(a, np.tensor) for a in unwrapped_params)


class TestGetTrainable:
    """Tests for getting trainable indices"""

    def test_tensorflow(self):
        """Test that the trainability indices of a sequence of TensorFlow values
        is correctly extracted"""
        values = [
            onp.array([0.1, 0.2]),
            tf.Variable(0.1, dtype=tf.float64),
            tf.constant([0.5, 0.2]),
        ]

        # outside of a gradient tape, no indices are trainable
        res = qml.math.get_trainable_indices(values)
        assert not res

        # within a gradient tape, Variables are automatically watched
        with tf.GradientTape():
            res = qml.math.get_trainable_indices(values)

        assert res == {1}

        # Watching can be set manually
        with tf.GradientTape() as tape:
            tape.watch([values[2]])
            res = qml.math.get_trainable_indices(values)

        assert res == {1, 2}

    def test_torch(self):
        """Test that the trainability indices of a sequence of Torch values
        is correctly extracted"""
        values = [
            onp.array([0.1, 0.2]),
            torch.tensor(0.1, requires_grad=True),
            torch.tensor([0.5, 0.2]),
        ]
        res = qml.math.get_trainable_indices(values)
        assert res == {1}

    def test_autograd(self):
        """Test that the trainability indices of a sequence of Autograd arrays
        is correctly extracted"""
        res = None

        def cost_fn(params):
            nonlocal res
            res = qml.math.get_trainable_indices(params)
            return np.sum(np.sin(params[0] * params[2])) + params[1]

        values = [[0.1, 0.2], np.tensor(0.1, requires_grad=True), np.tensor([0.5, 0.2])]
        cost_fn(values)

        assert res == {1, 2}

    def test_autograd_unwrapping_backward(self):
        """Test that the trainability indices of a sequence of Autograd arrays
        is correctly extracted on the backward pass"""
        res = None

        def cost_fn(*params):
            nonlocal res
            res = qml.math.get_trainable_indices(params)
            return np.sum(np.sin(params[0] * params[2])) + params[1]

        values = [
            np.array([0.1, 0.2]),
            np.tensor(0.1, requires_grad=True),
            np.tensor([0.5, 0.2], requires_grad=False),
        ]
        grad = qml.grad(cost_fn)(*values)

        assert res == {0, 1}


test_sort_data = [
    ([1, 3, 4, 2], [1, 2, 3, 4]),
    (onp.array([1, 3, 4, 2]), onp.array([1, 2, 3, 4])),
    (np.array([1, 3, 4, 2]), np.array([1, 2, 3, 4])),
    (jnp.array([1, 3, 4, 2]), jnp.array([1, 2, 3, 4])),
    (torch.tensor([1, 3, 4, 2]), torch.tensor([1, 2, 3, 4])),
    (tf.Variable([1, 3, 4, 2]), tf.Variable([1, 2, 3, 4])),
    (tf.constant([1, 3, 4, 2]), tf.constant([1, 2, 3, 4])),
]


class TestSortFunction:
    """Test the sort function works across all interfaces"""

    @pytest.mark.parametrize("input, test_output", test_sort_data)
    def test_sort(self, input, test_output):
        """Test the sort method is outputting only sorted values not indices"""
        result = fn.sort(input)

        assert all(result == test_output)
