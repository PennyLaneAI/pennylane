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
from functools import partial
import itertools
import numpy as onp
import pytest

import pennylane as qml
from pennylane import numpy as np
from pennylane import math as fn
from autograd.numpy.numpy_boxes import ArrayBox

pytestmark = pytest.mark.all_interfaces

tf = pytest.importorskip("tensorflow", minversion="2.1")
torch = pytest.importorskip("torch")
jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
sci = pytest.importorskip("scipy")


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

    @pytest.mark.filterwarnings("error:Contains tensors of types {.+}; dispatch will prioritize")
    def test_no_warning_scipy_and_autograd(self):
        """Test that no warning is raised if the sequence of tensors contains
        SciPy sparse matrices and autograd tensors."""
        x = sci.sparse.eye(3)
        y = np.array([0.5, 0.1])

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
        """Test that concatenate, called without the axis arguments,
        concatenates across the 0th dimension"""
        t1 = [0.6, 0.1, 0.6]
        t2 = np.array([0.1, 0.2, 0.3])
        t3 = onp.array([5.0, 8.0, 101.0])

        res = fn.concatenate([t1, t2, t3])
        assert isinstance(res, np.ndarray)
        assert np.all(res == np.concatenate([t1, t2, t3]))

    def test_concatenate_jax(self):
        """Test that concatenate, called without the axis arguments,
        concatenates across the 0th dimension"""
        t1 = jnp.array([5.0, 8.0, 101.0])
        t2 = jnp.array([0.6, 0.1, 0.6])
        t3 = jnp.array([0.1, 0.2, 0.3])

        res = fn.concatenate([t1, t2, t3])
        assert jnp.all(res == jnp.concatenate([t1, t2, t3]))

    def test_concatenate_tensorflow(self):
        """Test that concatenate, called without the axis arguments,
        concatenates across the 0th dimension"""
        t1 = tf.constant([0.6, 0.1, 0.6])
        t2 = tf.Variable([0.1, 0.2, 0.3])
        t3 = onp.array([5.0, 8.0, 101.0])

        res = fn.concatenate([t1, t2, t3])
        assert isinstance(res, tf.Tensor)
        assert np.all(res.numpy() == np.concatenate([t1.numpy(), t2.numpy(), t3]))

    def test_concatenate_torch(self):
        """Test that concatenate, called without the axis arguments,
        concatenates across the 0th dimension"""
        t1 = onp.array([5.0, 8.0, 101.0], dtype=np.float64)
        t2 = torch.tensor([0.6, 0.1, 0.6], dtype=torch.float64)
        t3 = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)

        res = fn.concatenate([t1, t2, t3])
        assert isinstance(res, torch.Tensor)
        assert np.all(res.numpy() == np.concatenate([t1, t2.numpy(), t3.numpy()]))

    @pytest.mark.parametrize(
        "t1", [onp.array([[1], [2]]), torch.tensor([[1], [2]]), tf.constant([[1], [2]])]
    )
    def test_concatenate_axis(self, t1):
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

    def test_matrix_vector_product_tensorflow_autograph(self):
        """Test that the matrix-matrix dot product of two vectors results in a matrix
        when using TensorFlow autograph mode"""
        t1, t2 = tf.Variable([[1, 2], [3, 4]]), tf.Variable([6, 7])

        @tf.function
        def cost(t1, t2):
            return fn.dot(t1, t2)

        with tf.GradientTape() as tape:
            res = cost(t1, t2)

        assert fn.allequal(res, [20, 46])

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


class TestTensordotTorch:
    """Tests for the tensor product function in torch.
    This test is required because the functionality of tensordot for Torch
    is being patched in PennyLane, as compared to autoray."""

    v1 = torch.tensor([0.1, 0.5, -0.9, 1.0, -4.2, 0.1], dtype=torch.float64)
    v2 = torch.tensor([4.3, -1.2, 8.2, 0.6, -4.2, -11.0], dtype=torch.float64)
    _arange = np.arange(0, 54).reshape((9, 6)).astype(np.float64)
    _shuffled_arange = np.array(
        [
            [42.0, 43.0, 44.0, 45.0, 46.0, 47.0],
            [6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
            [30.0, 31.0, 32.0, 33.0, 34.0, 35.0],
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            [48.0, 49.0, 50.0, 51.0, 52.0, 53.0],
            [12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
            [24.0, 25.0, 26.0, 27.0, 28.0, 29.0],
            [18.0, 19.0, 20.0, 21.0, 22.0, 23.0],
            [36.0, 37.0, 38.0, 39.0, 40.0, 41.0],
        ],
        dtype=np.float64,
    )
    M1 = torch.tensor(_arange)
    M2 = torch.tensor(_shuffled_arange)
    T1 = np.arange(0, 3 * 6 * 9 * 2).reshape((3, 6, 9, 2)).astype(np.float64)
    T1 = torch.tensor(np.array([T1[1], T1[0], T1[2]]), dtype=torch.float64)

    v1_dot_v2 = 9.59
    v1_outer_v2 = np.array(
        [
            [0.43, -0.12, 0.82, 0.06, -0.42, -1.1],
            [2.15, -0.6, 4.1, 0.3, -2.1, -5.5],
            [-3.87, 1.08, -7.38, -0.54, 3.78, 9.9],
            [4.3, -1.2, 8.2, 0.6, -4.2, -11.0],
            [-18.06, 5.04, -34.44, -2.52, 17.64, 46.2],
            [0.43, -0.12, 0.82, 0.06, -0.42, -1.1],
        ],
        dtype=np.float64,
    )

    M1_dot_v1 = torch.tensor(
        [-14.6, -35.0, -55.4, -75.8, -96.2, -116.6, -137.0, -157.4, -177.8], dtype=torch.float64
    )
    M1_dot_v2 = torch.tensor(
        [-54.8, -74.6, -94.4, -114.2, -134.0, -153.8, -173.6, -193.4, -213.2], dtype=torch.float64
    )
    M2_dot_v1 = torch.tensor(
        [-157.4, -35.0, -116.6, -14.6, -177.8, -55.4, -96.2, -75.8, -137.0], dtype=torch.float64
    )
    M2_dot_v2 = torch.tensor(
        [-193.4, -74.6, -153.8, -54.8, -213.2, -94.4, -134.0, -114.2, -173.6], dtype=torch.float64
    )
    M1_dot_M2T = torch.tensor(
        [
            [685, 145, 505, 55, 775, 235, 415, 325, 595],
            [2287, 451, 1675, 145, 2593, 757, 1369, 1063, 1981],
            [3889, 757, 2845, 235, 4411, 1279, 2323, 1801, 3367],
            [5491, 1063, 4015, 325, 6229, 1801, 3277, 2539, 4753],
            [7093, 1369, 5185, 415, 8047, 2323, 4231, 3277, 6139],
            [8695, 1675, 6355, 505, 9865, 2845, 5185, 4015, 7525],
            [10297, 1981, 7525, 595, 11683, 3367, 6139, 4753, 8911],
            [11899, 2287, 8695, 685, 13501, 3889, 7093, 5491, 10297],
            [13501, 2593, 9865, 775, 15319, 4411, 8047, 6229, 11683],
        ],
        dtype=torch.float64,
    )
    M1T_dot_M2 = torch.tensor(
        [
            [5256, 5472, 5688, 5904, 6120, 6336],
            [5472, 5697, 5922, 6147, 6372, 6597],
            [5688, 5922, 6156, 6390, 6624, 6858],
            [5904, 6147, 6390, 6633, 6876, 7119],
            [6120, 6372, 6624, 6876, 7128, 7380],
            [6336, 6597, 6858, 7119, 7380, 7641],
        ],
        dtype=torch.float64,
    )

    T1_dot_v1 = torch.tensor(
        [
            [
                [-630.0, -633.4],
                [-636.8, -640.2],
                [-643.6, -647.0],
                [-650.4, -653.8],
                [-657.2, -660.6],
                [-664.0, -667.4],
                [-670.8, -674.2],
                [-677.6, -681.0],
                [-684.4, -687.8],
            ],
            [
                [-262.8, -266.2],
                [-269.6, -273.0],
                [-276.4, -279.8],
                [-283.2, -286.6],
                [-290.0, -293.4],
                [-296.8, -300.2],
                [-303.6, -307.0],
                [-310.4, -313.8],
                [-317.2, -320.6],
            ],
            [
                [-997.2, -1000.6],
                [-1004.0, -1007.4],
                [-1010.8, -1014.2],
                [-1017.6, -1021.0],
                [-1024.4, -1027.8],
                [-1031.2, -1034.6],
                [-1038.0, -1041.4],
                [-1044.8, -1048.2],
                [-1051.6, -1055.0],
            ],
        ],
        dtype=torch.float64,
    )

    T1_dot_v2 = torch.tensor(
        [
            [
                [-1342.8, -1346.1],
                [-1349.4, -1352.7],
                [-1356.0, -1359.3],
                [-1362.6, -1365.9],
                [-1369.2, -1372.5],
                [-1375.8, -1379.1],
                [-1382.4, -1385.7],
                [-1389.0, -1392.3],
                [-1395.6, -1398.9],
            ],
            [
                [-986.4, -989.7],
                [-993.0, -996.3],
                [-999.6, -1002.9],
                [-1006.2, -1009.5],
                [-1012.8, -1016.1],
                [-1019.4, -1022.7],
                [-1026.0, -1029.3],
                [-1032.6, -1035.9],
                [-1039.2, -1042.5],
            ],
            [
                [-1699.2, -1702.5],
                [-1705.8, -1709.1],
                [-1712.4, -1715.7],
                [-1719.0, -1722.3],
                [-1725.6, -1728.9],
                [-1732.2, -1735.5],
                [-1738.8, -1742.1],
                [-1745.4, -1748.7],
                [-1752.0, -1755.3],
            ],
        ],
        dtype=torch.float64,
    )

    T1_dot_M1 = torch.tensor(
        [
            [237546.0, 238977.0],
            [82998.0, 84429.0],
            [392094.0, 393525.0],
        ],
        dtype=torch.float64,
    )

    T1_dot_M2 = torch.tensor(
        [
            [233370.0, 234801.0],
            [78822.0, 80253.0],
            [387918.0, 389349.0],
        ],
        dtype=torch.float64,
    )

    @pytest.mark.parametrize("axes", [[[0], [0]], [[-1], [0]], [[0], [-1]], [[-1], [-1]]])
    def test_tensordot_torch_vector_vector(self, axes):
        assert fn.allclose(fn.tensordot(self.v1, self.v2, axes=axes), self.v1_dot_v2)

    def test_tensordot_torch_outer(self):
        assert fn.allclose(fn.tensordot(self.v1, self.v2, axes=0), self.v1_outer_v2)
        assert fn.allclose(fn.tensordot(self.v2, self.v1, axes=0), qml.math.T(self.v1_outer_v2))

    def test_tensordot_torch_outer_with_old_version(self, monkeypatch):
        with monkeypatch.context() as m:
            m.setattr("torch.__version__", "1.9.0")
            assert fn.allclose(fn.tensordot(self.v1, self.v2, axes=0), self.v1_outer_v2)
            assert fn.allclose(fn.tensordot(self.v2, self.v1, axes=0), qml.math.T(self.v1_outer_v2))

    @pytest.mark.parametrize(
        "M, v, expected",
        [(M1, v1, M1_dot_v1), (M1, v2, M1_dot_v2), (M2, v1, M2_dot_v1), (M2, v2, M2_dot_v2)],
    )
    @pytest.mark.parametrize("axes", [[[1], [0]], [[-1], [0]], [[1], [-1]], [[-1], [-1]]])
    def test_tensordot_torch_matrix_vector(self, M, v, expected, axes):
        assert fn.allclose(fn.tensordot(M, v, axes=axes), expected)

    @pytest.mark.parametrize("axes", [[[1], [0]], [[-1], [0]], [[1], [-2]], [[-1], [-2]]])
    def test_tensordot_torch_matrix_matrix(self, axes):
        assert fn.allclose(fn.tensordot(self.M1, qml.math.T(self.M2), axes=axes), self.M1_dot_M2T)
        assert fn.allclose(
            fn.tensordot(self.M2, qml.math.T(self.M1), axes=axes), qml.math.T(self.M1_dot_M2T)
        )
        assert fn.allclose(fn.tensordot(qml.math.T(self.M1), self.M2, axes=axes), self.M1T_dot_M2)
        assert fn.allclose(
            fn.tensordot(qml.math.T(self.M2), self.M1, axes=axes), qml.math.T(self.M1T_dot_M2)
        )

    @pytest.mark.parametrize("axes", [[[1], [0]], [[-3], [0]], [[1], [-1]], [[-3], [-1]]])
    @pytest.mark.parametrize("v, expected", [(v1, T1_dot_v1), (v2, T1_dot_v2)])
    def test_tensordot_torch_tensor_vector(self, v, expected, axes):
        assert fn.allclose(fn.tensordot(self.T1, v, axes=axes), expected)

    @pytest.mark.parametrize("axes1", [[1, 2], [-3, -2], [1, -2], [-3, 2]])
    @pytest.mark.parametrize("axes2", [[1, 0], [-1, -2], [1, -2]])
    @pytest.mark.parametrize("M, expected", [(M1, T1_dot_M1), (M2, T1_dot_M2)])
    def test_tensordot_torch_tensor_matrix(self, M, expected, axes1, axes2):
        assert fn.allclose(fn.tensordot(self.T1, M, axes=[axes1, axes2]), expected)


class TestTensordotDifferentiability:

    v0 = np.array([0.1, 5.3, -0.9, 1.1])
    v1 = np.array([0.5, -1.7, -2.9, 0.0])
    v2 = np.array([-0.4, 9.1, 1.6])
    exp_shapes = ((len(v0), len(v2), len(v0)), (len(v0), len(v2), len(v2)))
    exp_jacs = (np.zeros(exp_shapes[0]), np.zeros(exp_shapes[1]))
    for i in range(len(v0)):
        exp_jacs[0][i, :, i] = v2
    for i in range(len(v2)):
        exp_jacs[1][:, i, i] = v0

    def test_autograd(self):
        """Tests differentiability of tensordot with Autograd."""
        v0 = np.array(self.v0, requires_grad=True)
        v1 = np.array(self.v1, requires_grad=True)
        v2 = np.array(self.v2, requires_grad=True)

        # Test inner product
        jac = qml.jacobian(partial(fn.tensordot, axes=[0, 0]), argnum=(0, 1))(v0, v1)
        assert all(fn.allclose(jac[i], _v) for i, _v in enumerate([v1, v0]))

        # Test outer product
        jac = qml.jacobian(partial(fn.tensordot, axes=0), argnum=(0, 1))(v0, v2)
        assert all(fn.shape(jac[i]) == self.exp_shapes[i] for i in [0, 1])
        assert all(fn.allclose(jac[i], self.exp_jacs[i]) for i in [0, 1])

    def test_torch(self):
        """Tests differentiability of tensordot with Torch."""
        jac_fn = torch.autograd.functional.jacobian

        v0 = torch.tensor(self.v0, requires_grad=True, dtype=torch.float64)
        v1 = torch.tensor(self.v1, requires_grad=True, dtype=torch.float64)
        v2 = torch.tensor(self.v2, requires_grad=True, dtype=torch.float64)

        # Test inner product
        jac = jac_fn(partial(fn.tensordot, axes=[[0], [0]]), (v0, v1))
        assert all(fn.allclose(jac[i], _v) for i, _v in enumerate([v1, v0]))

        # Test outer product
        jac = jac_fn(partial(fn.tensordot, axes=0), (v0, v2))
        assert all(fn.shape(jac[i]) == self.exp_shapes[i] for i in [0, 1])
        assert all(fn.allclose(jac[i], self.exp_jacs[i]) for i in [0, 1])

    def test_jax(self):
        """Tests differentiability of tensordot with JAX."""
        jac_fn = jax.jacobian

        v0 = jnp.array(self.v0)
        v1 = jnp.array(self.v1)
        v2 = jnp.array(self.v2)

        # Test inner product
        jac = jac_fn(partial(fn.tensordot, axes=[[0], [0]]), argnums=(0, 1))(v0, v1)
        assert all(fn.allclose(jac[i], _v) for i, _v in enumerate([v1, v0]))

        # Test outer product
        jac = jac_fn(partial(fn.tensordot, axes=0), argnums=(0, 1))(v0, v2)
        assert all(fn.shape(jac[i]) == self.exp_shapes[i] for i in [0, 1])
        assert all(fn.allclose(jac[i], self.exp_jacs[i]) for i in [0, 1])

    def test_tensorflow(self):
        """Tests differentiability of tensordot with TensorFlow."""

        def jac_fn(func, args):
            with tf.GradientTape() as tape:
                out = func(*args)
            return tape.jacobian(out, args)

        v0 = tf.Variable(self.v0, dtype=tf.float64)
        v1 = tf.Variable(self.v1, dtype=tf.float64)
        v2 = tf.Variable(self.v2, dtype=tf.float64)

        # Test inner product
        jac = jac_fn(partial(fn.tensordot, axes=[[0], [0]]), (v0, v1))
        assert all(fn.allclose(jac[i], _v) for i, _v in enumerate([v1, v0]))

        # Test outer product
        jac = jac_fn(partial(fn.tensordot, axes=0), (v0, v2))
        assert all(fn.shape(jac[i]) == self.exp_shapes[i] for i in [0, 1])
        assert all(fn.allclose(jac[i], self.exp_jacs[i]) for i in [0, 1])


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
        with pytest.warns(UserWarning, match="Attempted to differentiate a function with no"):
            qml.grad(cost_fn)(t, s)
        assert res == [False, False]

    def test_torch(self):
        """Torch tensors will simply return their requires_grad attribute"""
        t = torch.tensor([1.0, 2.0], requires_grad=True)
        assert fn.requires_grad(t)

        t = torch.tensor([1.0, 2.0], requires_grad=False)
        assert not fn.requires_grad(t)

    def test_tf(self):
        """TensorFlow tensors will True *if* they are being watched by a
        gradient tape or if they have a trainable attribute set to True"""
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

        t3 = tf.Variable([1.0, 2.0], trainable=True)
        assert fn.requires_grad(t3)

    def test_unknown_interface(self):
        """Test that an error is raised if the interface is unknown"""
        with pytest.raises(ValueError, match="unknown object"):
            fn.requires_grad(type("hello", tuple(), {})())


class TestInBackprop:
    """Tests for the in_backprop function"""

    @pytest.mark.slow
    def test_jax(self):
        """The value of in_backprop for JAX DeviceArrays depends on the argnums argument"""
        res = None

        def cost_fn(t, s):
            nonlocal res
            res = [fn.in_backprop(t), fn.in_backprop(s)]
            return jnp.sum(t * s)

        t = jnp.array([1.0, 2.0, 3.0])
        s = jnp.array([-2.0, -3.0, -4.0])

        jax.grad(cost_fn, argnums=0)(t, s)
        assert res == [True, False]

        jax.grad(cost_fn, argnums=1)(t, s)
        assert res == [False, True]

        jax.grad(cost_fn, argnums=[0, 1])(t, s)
        assert res == [True, True]

    def test_autograd_backwards(self):
        """The value of in_backprop for Autograd tensors corresponds to the requires_grad attribute during the backwards pass."""
        res = None

        def cost_fn(t, s):
            nonlocal res
            res = [fn.in_backprop(t), fn.in_backprop(s)]
            return np.sum(t * s)

        t = np.array([1.0, 2.0, 3.0], requires_grad=True)
        s = np.array([-2.0, -3.0, -4.0], requires_grad=True)

        qml.grad(cost_fn)(t, s)
        assert res == [True, True]

        t.requires_grad = False
        s.requires_grad = True
        qml.grad(cost_fn)(t, s)
        assert res == [False, True]

        t.requires_grad = True
        s.requires_grad = False
        qml.grad(cost_fn)(t, s)
        assert res == [True, False]

        t.requires_grad = False
        s.requires_grad = False
        with pytest.warns(UserWarning, match="Attempted to differentiate a function with no"):
            qml.grad(cost_fn)(t, s)
        assert res == [False, False]

    def test_tf(self):
        """The value of in_backprop for TensorFlow tensors is True *if* they are being watched by a gradient tape"""
        t1 = tf.Variable([1.0, 2.0])
        t2 = tf.constant([1.0, 2.0])
        assert not fn.in_backprop(t1)
        assert not fn.in_backprop(t2)

        with tf.GradientTape():
            # variables are automatically watched within a context,
            # but constants are not
            assert fn.in_backprop(t1)
            assert not fn.in_backprop(t2)

        with tf.GradientTape() as tape:
            # watching makes all tensors trainable
            tape.watch([t1, t2])
            assert fn.in_backprop(t1)
            assert fn.in_backprop(t2)

    def test_tf_autograph(self):
        """TensorFlow tensors will True *if* they are being watched by a gradient tape with Autograph."""
        t1 = tf.Variable([1.0, 2.0])
        t2 = tf.constant([1.0, 2.0])
        assert not fn.in_backprop(t1)
        assert not fn.in_backprop(t2)

        @tf.function
        def f_pow(x):
            return tf.math.pow(x, 3)

        with tf.GradientTape():
            # variables are automatically watched within a context,
            # but constants are not
            y = f_pow(t1)
            assert fn.in_backprop(t1)
            assert not fn.in_backprop(t2)

        with tf.GradientTape() as tape:
            # watching makes all tensors trainable
            tape.watch([t1, t2])
            y = f_pow(t1)
            assert fn.in_backprop(t1)
            assert fn.in_backprop(t2)

    @pytest.mark.torch
    def test_unknown_interface_in_backprop(self):
        """Test that an error is raised if the interface is unknown"""
        import torch

        with pytest.raises(ValueError, match="is in backpropagation."):
            fn.in_backprop(torch.tensor([0.1]))

        with pytest.raises(ValueError, match="is in backpropagation."):
            fn.in_backprop(type("hello", tuple(), {})())


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

    def test_array_indexing_autograd(self):
        """Test that indexing with a sequence properly extracts
        the elements from the flattened tensor"""
        t = np.array([[[1, 2], [3, 4], [-1, 1]], [[5, 6], [0, -1], [2, 1]]])
        indices = [0, 2, 3, 6, -2]

        def cost_fn(t):
            return np.sum(fn.take(t, indices))

        grad = qml.grad(cost_fn)(t)
        expected = np.array([[[1, 0], [1, 1], [0, 0]], [[1, 0], [0, 0], [1, 0]]])
        assert fn.allclose(grad, expected)

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
    ("autograd", np.array([[[1, 2], [3, 4], [-1, 1]], [[5, 6], [0, -1], [2, 1]]])),
    ("torch", torch.tensor([[[1, 2], [3, 4], [-1, 1]], [[5, 6], [0, -1], [2, 1]]])),
    ("numpy", onp.array([[[1, 2], [3, 4], [-1, 1]], [[5, 6], [0, -1], [2, 1]]])),
    ("tf", tf.constant([[[1, 2], [3, 4], [-1, 1]], [[5, 6], [0, -1], [2, 1]]])),
    ("tf", tf.Variable([[[1, 2], [3, 4], [-1, 1]], [[5, 6], [0, -1], [2, 1]]])),
    ("jax", jnp.array([[[1, 2], [3, 4], [-1, 1]], [[5, 6], [0, -1], [2, 1]]])),
]


@pytest.mark.parametrize("interface, t", where_data)
def test_where(interface, t):
    """Test that the where function works as expected"""
    # With output values
    res = fn.where(t < 0, 100 * fn.ones_like(t), t)
    expected = np.array([[[1, 2], [3, 4], [100, 1]], [[5, 6], [0, 100], [2, 1]]])
    assert fn.allclose(res, expected)

    # Without output values
    res = fn.where(t > 0)
    expected = (
        [0, 0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 1, 1, 2, 0, 0, 2, 2],
        [0, 1, 0, 1, 1, 0, 1, 0, 1],
    )
    assert all(fn.allclose(_res, _exp) for _res, _exp in zip(res, expected))


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

    x = onp.ones((2, 3), dtype=np.float64)
    y = onp.array(0.56)
    index = [1, 2]
    expected_val = onp.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.3136]])
    expected_grad_x = onp.array([[0, 0, 0], [0, 0, 1.0]])
    expected_grad_y = 2 * y
    expected_jac_x = onp.eye(6).reshape((2, 3, 2, 3))
    expected_jac_y = onp.array([[0, 0, 0], [0, 0, 2 * y]])

    def test_array(self):
        """Test that a NumPy array is differentiable when using scatter addition"""
        x = np.array(self.x, requires_grad=True)
        y = np.array(self.y, requires_grad=True)

        def cost(*weights):
            return fn.scatter_element_add(weights[0], self.index, weights[1] ** 2)

        res = cost(x, y)
        assert isinstance(res, np.ndarray)
        assert fn.allclose(res, self.expected_val)

        grad = qml.grad(lambda *weights: cost(*weights)[self.index[0], self.index[1]])(x, y)
        assert fn.allclose(grad[0], self.expected_grad_x)
        assert fn.allclose(grad[1], self.expected_grad_y)

    def test_array_multi(self):
        """Test that a NumPy array and the addend are differentiable when using
        scatter addition (multi dispatch)."""
        x = np.array(self.x, requires_grad=True)
        y = np.array(self.y, requires_grad=True)

        def cost_multi(weight_0, weight_1):
            return fn.scatter_element_add(weight_0, self.index, weight_1**2)

        res = cost_multi(x, y)
        assert isinstance(res, np.ndarray)
        assert fn.allclose(res, self.expected_val)

        jac = qml.jacobian(lambda *weights: cost_multi(*weights))(x, y)
        assert fn.allclose(jac[0], self.expected_jac_x)
        assert fn.allclose(jac[1], self.expected_jac_y)

    def test_array_batch(self):
        """Test that a NumPy array and the addend are differentiable when using
        scatter addition (multi dispatch)."""
        x = np.ones((2, 3), requires_grad=True)
        y = np.array([0.56, 0.3], requires_grad=True)

        def cost_multi(weight_0, weight_1):
            return fn.scatter_element_add(weight_0, [(0, 1), (1, 2)], weight_1**2)

        res = cost_multi(x, y)
        assert isinstance(res, np.ndarray)
        assert fn.allclose(res, onp.array([[1.0, 1.3136, 1.0], [1.0, 1.0, 1.09]]))

        jac = qml.jacobian(lambda *weights: cost_multi(*weights))(x, y)
        assert fn.allclose(jac[0], self.expected_jac_x)
        exp_jac_y = onp.zeros((2, 3, 2))
        exp_jac_y[0, 1, 0] = 2 * y[0]
        exp_jac_y[1, 2, 1] = 2 * y[1]
        assert fn.allclose(jac[1], exp_jac_y)

    def test_tensorflow(self):
        """Test that a TF tensor is differentiable when using scatter addition"""
        x = tf.Variable(self.x)
        y = tf.Variable(self.y)

        with tf.GradientTape() as tape:
            res = fn.scatter_element_add(x, self.index, y**2)
            loss = res[self.index[0], self.index[1]]

        assert isinstance(res, tf.Tensor)
        assert fn.allclose(res, self.expected_val)

        grad = tape.gradient(loss, [x, y])
        assert fn.allclose(grad[0], self.expected_grad_x)
        assert fn.allclose(grad[1], self.expected_grad_y)

    def test_torch(self):
        """Test that a torch tensor is differentiable when using scatter addition"""
        x = torch.tensor(self.x, requires_grad=True)
        y = torch.tensor(self.y, requires_grad=True)

        res = fn.scatter_element_add(x, self.index, y**2)
        loss = res[self.index[0], self.index[1]]

        assert isinstance(res, torch.Tensor)
        assert fn.allclose(res.detach(), self.expected_val)

        loss.backward()
        assert fn.allclose(x.grad, self.expected_grad_x)
        assert fn.allclose(y.grad, self.expected_grad_y)

    def test_jax(self):
        """Test that a JAX array is differentiable when using scatter addition"""
        x = jnp.array(self.x)
        y = jnp.array(self.y)

        def cost(weights):
            return fn.scatter_element_add(weights[0], self.index, weights[1] ** 2)

        res = cost([x, y])
        assert isinstance(res, jax.interpreters.xla.DeviceArray)
        assert fn.allclose(res, self.expected_val)

        grad = jax.grad(lambda weights: cost(weights)[self.index[0], self.index[1]])([x, y])
        assert fn.allclose(grad[0], self.expected_grad_x)
        assert fn.allclose(grad[1], self.expected_grad_y)

    def test_jax_multi(self):
        """Test that a NumPy array and the addend are differentiable when using
        scatter addition (multi dispatch)."""
        x = jnp.array(self.x)
        y = jnp.array(self.y)

        def cost_multi(weight_0, weight_1):
            return fn.scatter_element_add(weight_0, self.index, weight_1**2)

        res = cost_multi(x, y)
        assert isinstance(res, jax.interpreters.xla.DeviceArray)
        assert fn.allclose(res, self.expected_val)

        jac = jax.jacobian(lambda *weights: cost_multi(*weights), argnums=[0, 1])(x, y)
        assert fn.allclose(jac[0], self.expected_jac_x)
        assert fn.allclose(jac[1], self.expected_jac_y)


class TestScatterElementAddMultiValue:
    """Tests for the scatter_element_add function when adding
    multiple values at multiple positions."""

    x = onp.ones((2, 3), dtype=np.float64)
    y = onp.array(0.56)
    indices = [[1, 0], [2, 1]]
    expected_val = onp.array([[1.0, 1.3136, 1.0], [1.0, 1.0, 1.27636]])
    expected_grad_x = onp.array([[0, 1.0, 0], [0, 0, 1.0]])
    expected_grad_y = 2 * y + onp.cos(y / 2) / 2

    def test_array(self):
        """Test that a NumPy array is differentiable when using scatter addition
        with multiple values."""
        x = np.array(self.x, requires_grad=True)
        y = np.array(self.y, requires_grad=True)

        def cost(*weights):
            return fn.scatter_element_add(
                weights[0], self.indices, [fn.sin(weights[1] / 2), weights[1] ** 2]
            )

        res = cost(x, y)
        assert isinstance(res, np.ndarray)
        assert fn.allclose(res, self.expected_val)

        scalar_cost = (
            lambda *weights: cost(*weights)[self.indices[0][0], self.indices[1][0]]
            + cost(*weights)[self.indices[0][1], self.indices[1][1]]
        )
        grad = qml.grad(scalar_cost)(x, y)
        assert fn.allclose(grad[0], self.expected_grad_x)
        assert fn.allclose(grad[1], self.expected_grad_y)

    def test_tensorflow(self):
        """Test that a TF tensor is differentiable when using scatter addition
        with multiple values."""
        x = tf.Variable(self.x)
        y = tf.Variable(self.y)

        with tf.GradientTape() as tape:
            res = fn.scatter_element_add(x, self.indices, [tf.sin(y / 2), y**2])
            loss = (
                res[self.indices[0][0], self.indices[1][0]]
                + res[self.indices[0][1], self.indices[1][1]]
            )

        assert isinstance(res, tf.Tensor)
        assert fn.allclose(res, self.expected_val)

        grad = tape.gradient(loss, [x, y])
        assert fn.allclose(grad[0], self.expected_grad_x)
        assert fn.allclose(grad[1], self.expected_grad_y)

    def test_torch(self):
        """Test that a torch tensor is differentiable when using scatter addition
        with multiple values."""
        x = torch.tensor(self.x, requires_grad=True)
        y = torch.tensor(self.y, requires_grad=True)

        values = torch.zeros(2)
        values[0] += torch.sin(y / 2)
        values[1] += y**2
        res = fn.scatter_element_add(x, self.indices, values)
        loss = (
            res[self.indices[0][0], self.indices[1][0]]
            + res[self.indices[0][1], self.indices[1][1]]
        )

        assert isinstance(res, torch.Tensor)
        assert fn.allclose(res.detach(), self.expected_val)

        loss.backward()
        assert fn.allclose(x.grad, self.expected_grad_x)
        assert fn.allclose(y.grad, self.expected_grad_y)

    def test_jax(self):
        """Test that a JAX array is differentiable when using scatter addition
        with multiple values."""
        x = jnp.array(self.x)
        y = jnp.array(self.y)

        def cost(weights):
            return fn.scatter_element_add(
                weights[0], self.indices, [fn.sin(weights[1] / 2), weights[1] ** 2]
            )

        res = cost([x, y])
        assert isinstance(res, jax.interpreters.xla.DeviceArray)
        assert fn.allclose(res, self.expected_val)

        scalar_cost = (
            lambda weights: cost(weights)[self.indices[0][0], self.indices[1][0]]
            + cost(weights)[self.indices[0][1], self.indices[1][1]]
        )
        grad = jax.grad(scalar_cost)([x, y])
        assert fn.allclose(grad[0], self.expected_grad_x)
        assert fn.allclose(grad[1], self.expected_grad_y)


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

    expected = lambda self, x, y: (
        [
            [-np.sin(x * y) * y, 0, 0],
            [0, 1.0, 0],
            [0, 2 * x, -1 / y],
        ],
        [
            [-np.sin(x * y) * x, 0, 0],
            [0, 0.0, 1.2],
            [0, -1 / 3, x / y**2],
        ],
    )

    def test_autograd(self):
        """Tests for differentiating the block diagonal function with autograd."""
        tensors = lambda x, y: [
            np.array([[fn.cos(x * y)]]),
            np.array([[x, 1.2 * y], [x**2 - y / 3, -x / y]]),
        ]
        f = lambda x, y: fn.block_diag(tensors(x, y))
        x, y = np.array([0.2, 1.5], requires_grad=True)
        res = qml.jacobian(f)(x, y)
        exp = self.expected(x, y)
        assert fn.allclose(res[0], exp[0])
        assert fn.allclose(res[1], exp[1])

    def test_jax(self):
        """Tests for differentiating the block diagonal function with JAX."""
        jax = pytest.importorskip("jax")
        tensors = lambda x, y: [
            jnp.array([[fn.cos(x * y)]]),
            jnp.array([[x, 1.2 * y], [x**2 - y / 3, -x / y]]),
        ]
        f = lambda x, y: fn.block_diag(tensors(x, y))
        x, y = 0.2, 1.5
        res = jax.jacobian(f, argnums=[0, 1])(x, y)
        exp = self.expected(x, y)
        assert fn.allclose(exp[0], res[0])
        assert fn.allclose(exp[1], res[1])

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

    @pytest.mark.gpu
    def test_torch_coercion_error(self):
        """Test Torch coercion error if multiple devices were specified."""

        if not torch.cuda.is_available():
            pytest.skip("A GPU would be required to run this test, but CUDA is not available.")

        tensors = [
            torch.tensor([0.2], device="cpu"),
            np.array([1, 2, 3]),
            torch.tensor(1 + 3j, dtype=torch.complex64, device="cuda"),
        ]

        with pytest.raises(
            RuntimeError,
            match="Expected all tensors to be on the same device, but found at least two devices",
        ):
            res = qml.math.coerce(tensors, like="torch")


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

        values = [
            onp.array([0.1, 0.2]),
            np.tensor(0.1, dtype=np.float64, requires_grad=True),
            np.tensor([0.5, 0.2], requires_grad=True),
        ]
        grad = qml.grad(cost_fn, argnum=[1, 2])(*values)

        expected = [np.array([0.1, 0.2]), 0.1, np.array([0.5, 0.2])]
        assert all(np.allclose(a, b) for a, b in zip(unwrapped_params, expected))
        assert not any(isinstance(a, ArrayBox) for a in unwrapped_params)

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


class TestExpm:

    _compare_mat = None

    def get_compare_mat(self):
        """Computes expm via taylor expansion."""
        if self._compare_mat is None:
            mat = qml.RX.compute_matrix(0.3)
            out = np.eye(2, dtype=complex)
            coeff = 1
            for i in range(1, 8):
                coeff *= i
                out += np.linalg.matrix_power(mat, i) / coeff

            self._compare_mat = out

        return self._compare_mat

    @pytest.mark.parametrize(
        "phi", [qml.numpy.array(0.3), torch.tensor(0.3), tf.Variable(0.3), jnp.array(0.3)]
    )
    def test_expm(self, phi):
        """Test expm function for all interfaces against taylor expansion approximation."""
        orig_mat = qml.RX.compute_matrix(phi)
        exp_mat = qml.math.expm(orig_mat)

        assert qml.math.allclose(exp_mat, self.get_compare_mat(), atol=1e-4)
