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
"""Unit tests for pennylane.math.single_dispatch"""
# pylint: disable=import-outside-toplevel
import itertools
from functools import partial
from unittest.mock import patch

import numpy as onp
import pytest
import scipy as sp
from autograd.numpy.numpy_boxes import ArrayBox

import pennylane as qml
from pennylane import math as fn
from pennylane import numpy as np
from pennylane.math.single_dispatch import _sparse_matrix_power_bruteforce

pytestmark = pytest.mark.all_interfaces

torch = pytest.importorskip("torch")
jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
sci = pytest.importorskip("scipy")


class TestGetMultiTensorbox:
    """Tests for the get_interface utility function"""

    def test_warning_torch_and_autograd(self):
        """Test that a warning is raised if the sequence of tensors contains
        both torch and autograd tensors."""
        x = torch.tensor([1.0, 2.0, 3.0])
        y = np.array([0.5, 0.1])

        with pytest.warns(UserWarning, match="Consider replacing Autograd with vanilla NumPy"):
            fn.get_interface(x, y)

    def test_warning_jax_and_autograd(self):
        """Test that a warning is raised if the sequence of tensors contains
        both jax and autograd tensors."""
        x = jnp.array([1.0, 2.0, 3.0])
        y = np.array([0.5, 0.1])

        with pytest.warns(UserWarning, match="Consider replacing Autograd with vanilla NumPy"):
            fn.get_interface(x, y)

    @pytest.mark.filterwarnings("error:Contains tensors of types {.+}; dispatch will prioritize")
    def test_no_warning_scipy_and_autograd(self):
        """Test that no warning is raised if the sequence of tensors contains
        SciPy sparse matrices and autograd tensors."""
        x = sci.sparse.eye(3)
        y = np.array([0.5, 0.1])

        fn.get_interface(x, y)

    def test_return_torch_box(self):
        """Test that Torch is correctly identified as the dispatching library."""
        x = torch.tensor([1.0, 2.0, 3.0])
        y = onp.array([0.5, 0.1])

        res = fn.get_interface(y, x)
        assert res == "torch"

    def test_return_autograd_box(self):
        """Test that autograd is correctly identified as the dispatching library."""
        x = np.array([1.0, 2.0, 3.0])
        y = [0.5, 0.1]

        res = fn.get_interface(y, x)
        assert res == "autograd"

    def test_return_jax_box(self):
        """Test that jax is correctly identified as the dispatching library."""
        x = jnp.array([1.0, 2.0, 3.0])
        y = [0.5, 0.1]

        res = fn.get_interface(y, x)
        assert res == "jax"

    def test_return_numpy_box(self):
        """Test that NumPy is correctly identified as the dispatching library."""
        x = onp.array([1.0, 2.0, 3.0])
        y = [0.5, 0.1]

        res = fn.get_interface(y, x)
        assert res == "numpy"

    def test_get_deep_interface(self):
        """Test get_deep_interface returns the interface of deep values."""
        assert fn.get_deep_interface([()]) == "numpy"
        assert fn.get_deep_interface(([1, 2], [3, 4])) == "numpy"
        assert fn.get_deep_interface([[jnp.array(1.1)]]) == "jax"
        assert fn.get_deep_interface([[np.array(1.3, requires_grad=False)]]) == "autograd"


test_abs_data = [
    (1, -2, 3 + 4j),
    [1, -2, 3 + 4j],
    onp.array([1, -2, 3 + 4j]),
    np.array([1, -2, 3 + 4j]),
    torch.tensor([1, -2, 3 + 4j], dtype=torch.complex128),
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
]

unequal_test_data = [
    (2, 2, 3),
    [1, 21, 3],
    onp.array([1, 2, 39]),
    np.array([1, 2, 4]),
    torch.tensor([1, 20, 3]),
]


@pytest.mark.parametrize("t1,t2", list(itertools.combinations(test_data, r=2)))
def test_allequal(t1, t2):
    """Test that the allequal function works for a variety of inputs."""
    res = fn.allequal(t1, t2)

    expected = all(float(x) == float(y) for x, y in zip(t1, t2))
    assert res == expected


test_all_vectors = [
    ((False, False, False), False),
    ((True, True, False), False),
    ((True, True, True), True),
]


@pytest.mark.parametrize("array_fn", [tuple, list, onp.array, np.array, torch.tensor])
@pytest.mark.parametrize("t1, expected", test_all_vectors)
def test_all(array_fn, t1, expected):
    """Test that the all function works for a variety of inputs."""
    res = fn.all(array_fn(t1))

    assert res == expected


test_any_vectors = [
    ((False, False, False), False),
    ((True, True, False), True),
    ((True, True, True), True),
]


@pytest.mark.parametrize("array_fn", [tuple, list, onp.array, np.array, torch.tensor])
@pytest.mark.parametrize("t1, expected", test_any_vectors)
def test_any(array_fn, t1, expected):
    """Test that the any function works for a variety of inputs."""
    res = fn.any(array_fn(t1))

    assert res == expected


@pytest.mark.parametrize(
    "t1,t2",
    list(
        itertools.combinations(test_data + [torch.tensor([1.0, 2.0, 3.0], requires_grad=True)], r=2)
    )
    + [(t1, t2) for t1 in test_data for t2 in unequal_test_data],
)
def test_allclose(t1, t2):
    """Test that the allclose function works for a variety of inputs."""
    res = fn.allclose(t1, t2)

    expected = all(float(x) == float(y) for x, y in zip(t1, t2))
    assert res == expected


class TestAllCloseSparse:
    """Test that the sparse-matrix specialized allclose functions works well"""

    @pytest.mark.parametrize("v", [0, 1, 2.0, 3.0j, 1e-9])
    def test_sparse_scalar(self, v):
        """Test comparing a scalar to a sparse matrix"""
        dense = v

        sparse = sp.sparse.csr_matrix([v] * 10)

        assert fn.allclose(dense, sparse)
        assert fn.allclose(sparse, dense)

        # Shift one element of the sparse matrix
        sparse_wrong = sp.sparse.csr_matrix([v + 0.1] + [v] * 9)
        assert not fn.allclose(dense, sparse_wrong)
        assert not fn.allclose(sparse_wrong, dense)

        # Empty one element of the sparse matrix
        v_nonzero = not np.isclose(v, 0)
        sparse_wrong = sp.sparse.csr_matrix([0 if v_nonzero else 1] + [v] * 9)
        assert not fn.allclose(dense, sparse_wrong)
        assert not fn.allclose(sparse_wrong, dense)

    def test_dense_sparse_small_matrix(self):
        """Test comparing small dense and sparse matrices"""
        dense = np.array([[1, 0, 2], [0, 3, 0]])
        sparse = sp.sparse.csr_matrix(dense)

        assert fn.allclose(dense, sparse)
        assert fn.allclose(sparse, dense)

    def test_dense_sparse_zero_nnz(self):
        """Test comparing dense matrix with empty sparse matrix"""
        dense = np.zeros((2, 3))
        sparse = sp.sparse.csr_matrix(dense)

        assert fn.allclose(dense, sparse)
        assert fn.allclose(sparse, dense)

    def test_dense_sparse_different_shapes(self):
        """Test comparing matrices with different shapes"""
        dense = np.array([[1, 2], [3, 4]])
        sparse = sp.sparse.csr_matrix(np.array([[1, 2]]))

        assert not fn.allclose(dense, sparse)
        assert not fn.allclose(sparse, dense)

    def test_dense_sparse_large_matrix(self):
        """Test comparing large dense and sparse matrices"""
        n = 200
        dense = np.eye(n)
        sparse = sp.sparse.eye(n)

        assert fn.allclose(dense, sparse)
        assert fn.allclose(sparse, dense)

        # When size is large enough, a very small perturbation
        # will override the tolerance.
        dense[-1, 0] = np.finfo(float).eps
        assert not fn.allclose(dense, sparse)
        assert not fn.allclose(sparse, dense)

    def test_sparse_sparse_large_matrix(self):
        """Test comparing large dense and sparse matrices"""
        n = 200
        dense = np.eye(n)
        sparse0 = sp.sparse.csr_matrix(dense)
        sparse = sp.sparse.eye(n)

        assert fn.allclose(sparse0, sparse)

        dense[-1, 0] = 0.001

        sparse0 = sp.sparse.csr_matrix(dense)

        assert not fn.allclose(sparse0, sparse)

    def test_dense_sparse_different_nonzero(self):
        """Test comparing matrices with different nonzero patterns"""
        dense = np.array([[1, 0], [0, 1]])
        sparse = sp.sparse.csr_matrix(np.array([[1, 1], [0, 1]]))

        assert not fn.allclose(dense, sparse)
        assert not fn.allclose(sparse, dense)

    @pytest.mark.parametrize("rtol,atol", [(1e-7, 1e-8), (1e-5, 1e-6)])
    def test_dense_sparse_tolerances(self, rtol, atol):
        """Test comparing matrices with different tolerances"""
        dense = np.array([[1.0, 0.0], [0.0, 1.0 + 1e-7]])
        sparse = sp.sparse.csr_matrix(np.array([[1.0, 0.0], [0.0, 1.0]]))

        allclose_result = fn.allclose(dense, sparse, rtol=rtol, atol=atol)
        expected = np.allclose(dense, sparse.toarray(), rtol=rtol, atol=atol)
        assert allclose_result == expected


test_angle_data = [
    [1.0, 1.0j, 1 + 1j],
    [1.0, 1.0j, 1 + 1j],
    onp.array([1.0, 1.0j, 1 + 1j]),
    np.array([1.0, 1.0j, 1 + 1j]),
    torch.tensor([1.0, 1.0j, 1 + 1j], dtype=torch.complex128),
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
]


@pytest.mark.parametrize("t", test_conj_data)
def test_conj(t):
    """Test the qml.math.conj function."""
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
    (1.0, 2.0, 3.0),
    [1.0, 2.0, 3.0],
    onp.array([1, 2, 3], dtype=onp.float64),
    np.array([1, 2, 3], dtype=np.float64),
    torch.tensor([1, 2, 3], dtype=torch.float64),
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

    def test_concatenate_torch(self):
        """Test that concatenate, called without the axis arguments,
        concatenates across the 0th dimension"""
        t1 = onp.array([5.0, 8.0, 101.0], dtype=np.float64)
        t2 = torch.tensor([0.6, 0.1, 0.6], dtype=torch.float64)
        t3 = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)

        res = fn.concatenate([t1, t2, t3])
        assert isinstance(res, torch.Tensor)
        assert np.all(res.numpy() == np.concatenate([t1, t2.numpy(), t3.numpy()]))

    @pytest.mark.parametrize("t1", [onp.array([[1], [2]]), torch.tensor([[1], [2]])])
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

    @pytest.mark.parametrize("t1", [onp.array([[1], [2]]), torch.tensor([[1], [2]])])
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

    @pytest.mark.parametrize("t_like", [np.array([1]), torch.tensor([1])])
    def test_convert_scalar(self, t_like):
        """Test that a python scalar is converted to a scalar tensor"""
        res = fn.convert_like(5, t_like)
        assert isinstance(res, t_like.__class__)
        assert res.ndim == 0
        assert fn.allequal(res, [5])

    def test_convert_like_sparse(self):
        """Test that a numpy array can be converted to a scipy array."""

        np_array = np.array([[1, 0], [1, 0]])
        sp_array = sci.sparse.csr_matrix([[0, 1], [1, 0]])
        out = qml.math.convert_like(np_array, sp_array)
        assert isinstance(out, sci.sparse.csr_matrix)
        assert qml.math.allclose(out.todense(), np_array)


class TestDot:
    """Tests for the dot product function"""

    scalar_product_data = [
        [2, 6],
        [np.array(2), np.array(6)],
        [torch.tensor(2), onp.array(6)],
        [torch.tensor(2), torch.tensor(6)],
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
        res = fn.dot(t2, t2)
        assert fn.allequal(res, 85)

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
        """Test tensordot vector-vector product with PyTorch."""
        assert fn.allclose(fn.tensordot(self.v1, self.v2, axes=axes), self.v1_dot_v2)

    def test_tensordot_torch_outer(self):
        """Test tensordot outer product with PyTorch."""
        assert fn.allclose(fn.tensordot(self.v1, self.v2, axes=0), self.v1_outer_v2)
        assert fn.allclose(fn.tensordot(self.v2, self.v1, axes=0), qml.math.T(self.v1_outer_v2))

    def test_tensordot_torch_outer_with_old_version(self, monkeypatch):
        """Test tensordot outer product with an old version of PyTorch."""
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
        """Test tensordot matrix-vector product with PyTorch."""
        assert fn.allclose(fn.tensordot(M, v, axes=axes), expected)

    @pytest.mark.parametrize("axes", [[[1], [0]], [[-1], [0]], [[1], [-2]], [[-1], [-2]]])
    def test_tensordot_torch_matrix_matrix(self, axes):
        """Test tensordot matrix-matrix product with PyTorch."""
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
        """Test tensordot tensor-vector product with PyTorch."""
        assert fn.allclose(fn.tensordot(self.T1, v, axes=axes), expected)

    @pytest.mark.parametrize("axes1", [[1, 2], [-3, -2], [1, -2], [-3, 2]])
    @pytest.mark.parametrize("axes2", [[1, 0], [-1, -2], [1, -2]])
    @pytest.mark.parametrize("M, expected", [(M1, T1_dot_M1), (M2, T1_dot_M2)])
    def test_tensordot_torch_tensor_matrix(self, M, expected, axes1, axes2):
        """Test tensordot tensor-matrix product with PyTorch."""
        assert fn.allclose(fn.tensordot(self.T1, M, axes=[axes1, axes2]), expected)


class TestTensordotDifferentiability:
    """Test the differentiability of qml.math.tensordot."""

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


interface_test_data = [
    [(1, 2, 3), "numpy"],
    [[1, 2, 3], "numpy"],
    [onp.array([1, 2, 3]), "numpy"],
    [np.array([1, 2, 3]), "autograd"],
    [torch.tensor([1, 2, 3]), "torch"],
    [jnp.array([1, 2, 3]), "jax"],
]


@pytest.mark.parametrize("t,interface", interface_test_data)
def test_get_interface(t, interface):
    """Test that the interface of a tensor-like object
    is correctly returned."""
    res = fn.get_interface(t)
    assert res == interface


class TestScipySparse:
    """Test the scipy.sparse objects get correctly dispatched"""

    matrix = [sci.sparse.csr_matrix([[0, 1], [1, 0]])]

    matrix_4 = [sci.sparse.csr_matrix(np.eye(4))]

    dispatched_linalg_methods = [
        fn.linalg.expm,
        fn.linalg.inv,
        fn.linalg.norm,
    ]

    dispatched_linalg_methods_factorization = [
        fn.linalg.eigs,
        fn.linalg.eigsh,
        fn.linalg.svds,
    ]

    dispatched_linalg_methods_linear_solver = [
        fn.linalg.spsolve,
    ]

    @pytest.mark.parametrize("matrix", matrix)
    def test_get_interface_scipy(self, matrix):
        """Test that the interface of a scipy sparse matrix is correctly returned."""

        assert fn.get_interface(matrix) == "scipy"
        assert fn.get_interface(matrix, matrix) == "scipy"

    @pytest.mark.parametrize("matrix", matrix)
    @pytest.mark.parametrize("method", dispatched_linalg_methods)
    def test_dispatched_linalg_methods_single(self, method, matrix):
        """Test that the dispatched single function works"""
        method(matrix)

    @pytest.mark.parametrize("matrix", matrix_4)
    @pytest.mark.parametrize("method", dispatched_linalg_methods_factorization)
    def test_dispatched_linalg_methods_factorization(self, method, matrix):
        """Test that the dispatched single function works"""
        method(matrix, 1)

    @pytest.mark.parametrize("matrix", matrix_4)
    @pytest.mark.parametrize("method", dispatched_linalg_methods_linear_solver)
    def test_dispatched_linalg_methods_linear_solver(self, method, matrix):
        """Test that the dispatched single function works"""
        method(matrix, sci.sparse.eye(matrix.shape[0]))

    @pytest.mark.parametrize("matrix", matrix + matrix_4)
    def test_dispatched_linalg_methods_matrix_power(self, matrix):
        """Test that the matrix power method dispatched"""
        _sparse_matrix_power_bruteforce(matrix, 2)

    def test_matrix_power(self):
        """Test our customized matrix power function"""
        A = sci.sparse.csr_matrix([[2, 0], [0, 2]])

        # Test n = 0 (identity matrix)
        result = _sparse_matrix_power_bruteforce(A, 0)
        expected = sci.sparse.eye(2, dtype=A.dtype, format=A.format)
        assert np.allclose(result.toarray(), expected.toarray())

        # Test n = 1 (should be the same matrix)
        result = _sparse_matrix_power_bruteforce(A, 1)
        assert np.allclose(result.toarray(), A.toarray())

        # Test n = 2 (square of matrix)
        result = _sparse_matrix_power_bruteforce(A, 2)
        expected = A @ A
        assert np.allclose(result.toarray(), expected.toarray())

        # Test n = 3 (cube of matrix)
        result = _sparse_matrix_power_bruteforce(A, 3)
        expected = A @ A @ A
        assert np.allclose(result.toarray(), expected.toarray())

        # Simple benchmark with the dispatcher
        result0 = fn.linalg.matrix_power(A, 3)
        assert np.allclose(result0.toarray(), expected.toarray())

        # Test negative exponent (should raise an error)
        with pytest.raises(ValueError):
            _sparse_matrix_power_bruteforce(A, -1)

        # Test non-integer exponent (should raise an error)
        with pytest.raises(ValueError, match="exponent must be an integer"):
            _sparse_matrix_power_bruteforce(A, 1.5)


# pylint: disable=too-few-public-methods
class TestInterfaceEnum:
    """Test the Interface enum class"""

    @pytest.mark.parametrize("user_input", [None, "numpy", "scipy"])
    def test_numpy(self, user_input):
        """Test that the numpy interface is correctly returned"""
        assert fn.Interface(user_input) == fn.Interface.NUMPY

    def test_autograd(self):
        """Test that the autograd interface is correctly returned"""
        assert fn.Interface("autograd") == fn.Interface.AUTOGRAD

    @pytest.mark.parametrize("user_input", ["torch", "pytorch"])
    def test_torch(self, user_input):
        """Test that the torch interface is correctly returned"""
        assert fn.Interface(user_input) == fn.Interface.TORCH

    @pytest.mark.parametrize("user_input", ["JAX", "jax", "jax-python"])
    def test_jax(self, user_input):
        """Test that the jax interface is correctly returned"""
        assert fn.Interface(user_input) == fn.Interface.JAX

    def test_jax_jit(self):
        """Test that the jax-jit interface is correctly returned"""
        assert fn.Interface("jax-jit") == fn.Interface.JAX_JIT

    def test_auto(self):
        """Test that the auto interface is correctly returned"""
        assert fn.Interface("auto") == fn.Interface.AUTO

    def test_eq(self):
        """Test that an error is raised if comparing to string"""
        assert fn.Interface.NUMPY == fn.Interface.NUMPY
        with pytest.raises(TypeError, match="Cannot compare Interface with str"):
            # pylint: disable=pointless-statement
            fn.Interface.NUMPY == "numpy"


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


@pytest.mark.parametrize("t", test_data)
def test_numpy_arraybox(t):
    """Test that the to_numpy method correctly converts the input
    ArrayBox into a NumPy array."""
    val = np.array(5.0)
    t = ArrayBox(val, None, None)
    res = fn.to_numpy(t)
    assert res == val
    assert isinstance(res, type(val))


def test_numpy_jax_jit():
    """Test that the to_numpy() method raises an exception
    if used inside the JAX JIT"""

    @jax.jit
    def cost(x):
        fn.to_numpy(x)
        return x

    with pytest.raises(ValueError, match="not supported when using the JAX JIT"):
        cost(jnp.array(0.1))


def test_numpy_torch():
    """Test that the to_numpy method correctly converts the input
    Torch tensor into a NumPy array."""
    x = torch.tensor([1.0, 2.0, 3.0])
    fn.to_numpy(x)


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
        """JAX Arrays differentiability depends on the argnums argument"""
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

    @pytest.mark.slow
    def test_jax_jit(self):
        """JAX Arrays differentiability does not depends on the argnums argument with Jitting because it is
        differentiability is set in the custom jvp."""
        res = None

        def cost_fn(t, s):
            nonlocal res
            res = [fn.requires_grad(t), fn.requires_grad(s)]
            return jnp.sum(t * s)

        t = jnp.array([1.0, 2.0, 3.0])
        s = jnp.array([-2.0, -3.0, -4.0])

        jax.jit(jax.grad(cost_fn, argnums=0))(t, s)
        assert res == [True, True]

        jax.jit(jax.grad(cost_fn, argnums=1))(t, s)
        assert res == [True, True]

        jax.jit(jax.grad(cost_fn, argnums=[0, 1]))(t, s)
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

    def test_unknown_interface(self):
        """Test that an error is raised if the interface is unknown"""
        with pytest.raises(ValueError, match="unknown object"):
            fn.requires_grad(type("hello", tuple(), {})())


class TestInBackprop:
    """Tests for the in_backprop function"""

    @pytest.mark.slow
    def test_jax(self):
        """The value of in_backprop for JAX Arrays depends on the argnums argument"""
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

    @pytest.mark.torch
    def test_unknown_interface_in_backprop(self):
        """Test that an error is raised if the interface is unknown"""

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
    ],
)
@pytest.mark.parametrize("shape", shape_test_data)
def test_shape(shape, interface, create_array):
    """Test that the shape of tensors is correctly returned"""
    if interface == "sequence" and not shape:
        pytest.skip("Cannot expand the dimensions of a Python scalar!")

    t = create_array(shape)
    assert fn.shape(t) == shape


@pytest.mark.parametrize("interface", ["numpy", "autograd", "jax", "torch"])
def test_shape_and_ndim_deep(interface):
    val = [[fn.asarray(1, like=interface)]]
    assert fn.shape(val) == (1, 1)
    assert fn.ndim(val) == 2


@pytest.mark.parametrize(
    "x, expected",
    (
        (1.0, "float64"),
        (1, "int64"),
        (onp.array(0.5), "float64"),
        (onp.array(1.0, dtype="float32"), "float32"),
        (ArrayBox(1, "a", "b"), "int64"),
        (np.array(0.5), "float64"),
        (np.array(0.5, dtype="complex64"), "complex64"),
        # skip jax as output is dependent on global configuration
        (torch.tensor(0.1, dtype=torch.float32), "float32"),
        (torch.tensor(0.5, dtype=torch.float64), "float64"),
        (torch.tensor(0.1, dtype=torch.complex128), "complex128"),
    ),
)
def test_get_dtype_name(x, expected):
    """Test that get_dtype_name returns the a string for the datatype."""
    assert fn.get_dtype_name(x) == expected


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

    def test_stack_torch(self):
        """Test that stack, called without the axis arguments, stacks vertically"""
        t1 = onp.array([5.0, 8.0, 101.0], dtype=np.float64)
        t2 = torch.tensor([0.6, 0.1, 0.6], dtype=torch.float64)
        t3 = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)

        res = fn.stack([t1, t2, t3])
        assert isinstance(res, torch.Tensor)
        assert np.all(res.numpy() == np.stack([t1, t2.numpy(), t3.numpy()]))

    @pytest.mark.parametrize("t1", [onp.array([1, 2]), torch.tensor([1, 2])])
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
        res = fn.take(t, indices, mode="wrap")
        assert fn.allclose(res, [1, 3, 4, 5, 2])

    def test_array_indexing_autograd(self):
        """Test that indexing with a sequence properly extracts
        the elements from the flattened tensor"""
        t = np.array([[[1, 2], [3, 4], [-1, 1]], [[5, 6], [0, -1], [2, 1]]], dtype=np.float64)
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
        t = np.array([[[1, 2], [3, 4], [-1, 1]], [[5, 6], [0, -1], [2, 1]]], dtype=np.float64)
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

    @pytest.mark.torch
    def test_last_axis_support_torch(self):
        """Test that _torch_take correctly sets the last axis"""
        x = fn.arange(8, like="torch").reshape((2, 4))
        assert np.array_equal(fn.take(x, indices=3, axis=-1), [3, 7])


where_data = [
    np.array([[[1, 2], [3, 4], [-1, 1]], [[5, 6], [0, -1], [2, 1]]]),
    torch.tensor([[[1, 2], [3, 4], [-1, 1]], [[5, 6], [0, -1], [2, 1]]]),
    onp.array([[[1, 2], [3, 4], [-1, 1]], [[5, 6], [0, -1], [2, 1]]]),
    jnp.array([[[1, 2], [3, 4], [-1, 1]], [[5, 6], [0, -1], [2, 1]]]),
]


@pytest.mark.parametrize("t", where_data)
def test_where(t):
    """Test that the qml.math.where function works as expected"""
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

        def grab_cost_entry(*weights):
            """Return a single entry of the cost"""
            return cost(*weights)[self.index[0], self.index[1]]

        grad = qml.grad(grab_cost_entry)(x, y)
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

        jac = qml.jacobian(cost_multi)(x, y)
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

        jac = qml.jacobian(cost_multi)(x, y)
        assert fn.allclose(jac[0], self.expected_jac_x)
        exp_jac_y = onp.zeros((2, 3, 2))
        exp_jac_y[0, 1, 0] = 2 * y[0]
        exp_jac_y[1, 2, 1] = 2 * y[1]
        assert fn.allclose(jac[1], exp_jac_y)

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
        assert isinstance(res, jax.Array)
        assert fn.allclose(res, self.expected_val)

        def grab_cost_entry(weights):
            """Return a single entry of the cost"""
            return cost(weights)[self.index[0], self.index[1]]

        grad = jax.grad(grab_cost_entry)([x, y])
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
        assert isinstance(res, jax.Array)
        assert fn.allclose(res, self.expected_val)

        jac = jax.jacobian(cost_multi, argnums=[0, 1])(x, y)
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

        def add_cost_entries(*weights):
            """Add two entries of the cost."""
            c = cost(*weights)
            return (
                c[self.indices[0][0], self.indices[1][0]]
                + c[self.indices[0][1], self.indices[1][1]]
            )

        grad = qml.grad(add_cost_entries)(x, y)
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
        assert isinstance(res, jax.Array)
        assert fn.allclose(res, self.expected_val)

        def add_cost_entries(weights):
            """Add two entries of the cost."""
            c = cost(weights)
            return (
                c[self.indices[0][0], self.indices[1][0]]
                + c[self.indices[0][1], self.indices[1][1]]
            )

        grad = jax.grad(add_cost_entries)([x, y])
        assert fn.allclose(grad[0], self.expected_grad_x)
        assert fn.allclose(grad[1], self.expected_grad_y)


class TestDiag:
    """Tests for the diag function"""

    @pytest.mark.parametrize(
        "a, interface",
        [
            [np.array(0.5), "autograd"],
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

        def grab_cov_entry(weights):
            """Grab an entry of the cov output."""
            return cov(weights)[0, 1]

        grad_fn = qml.grad(grab_cov_entry)
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

        def grab_cov_entry(weights):
            """Grab an entry of the cov output."""
            return cov(weights)[0, 1]

        grad_fn = qml.grad(grab_cov_entry)
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

    @pytest.mark.slow
    def test_jax(self, tol):
        """Test that the covariance matrix computes the correct
        result, and is differentiable, using the JAX interface"""
        dev = qml.device("default.qubit", wires=3)

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

        def grab_cov_entry(weights):
            """Grab an entry of the cov output."""
            return cov(weights)[0, 1]

        grad_fn = jax.grad(grab_cov_entry)
        res = grad_fn(weights)
        expected = self.expected_grad(weights)
        assert jnp.allclose(res, expected, atol=tol, rtol=0)


block_diag_data = [
    [onp.array([[1, 2], [3, 4]]), torch.tensor([[1, 2], [-1, -6]]), torch.tensor([[5]])],
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
    """Test differentiability of qml.math.block_diag."""

    @staticmethod
    def expected(x, y):
        """Return the expected Jacobian of block_diag."""
        return (
            [
                [-fn.sin(x * y) * y, 0, 0],
                [0, 1.0, 0],
                [0, 2 * x, -1 / y],
            ],
            [
                [-fn.sin(x * y) * x, 0, 0],
                [0, 0.0, 1.2],
                [0, -1 / 3, x / y**2],
            ],
        )

    def test_autograd(self):
        """Tests for differentiating the block diagonal function with autograd."""

        def f(x, y):
            return fn.block_diag(
                [
                    np.array([[fn.cos(x * y)]]),
                    np.array([[x, 1.2 * y], [x**2 - y / 3, -x / y]]),
                ]
            )

        x, y = np.array([0.2, 1.5], requires_grad=True)
        res = qml.jacobian(f)(x, y)
        exp = self.expected(x, y)
        assert fn.allclose(res[0], exp[0])
        assert fn.allclose(res[1], exp[1])

    def test_jax(self):
        """Tests for differentiating the block diagonal function with JAX."""

        def f(x, y):
            return fn.block_diag(
                [
                    jnp.array([[fn.cos(x * y)]]),
                    jnp.array([[x, 1.2 * y], [x**2 - y / 3, -x / y]]),
                ]
            )

        x, y = 0.2, 1.5
        res = jax.jacobian(f, argnums=[0, 1])(x, y)
        exp = self.expected(x, y)
        assert fn.allclose(exp[0], res[0])
        assert fn.allclose(exp[1], res[1])

    def test_torch(self):
        """Tests for differentiating the block diagonal function with Torch."""
        x, y = [torch.tensor([[0.2]]), torch.tensor([[0.1, 0.2], [0.3, 0.4]])]

        def f(x, y):
            return fn.block_diag([x, y])

        res = torch.autograd.functional.jacobian(f, (x, y))
        exp_0 = np.zeros((3, 3, 1, 1))
        exp_0[0, 0, 0, 0] = 1.0
        exp_1 = np.zeros((3, 3, 2, 2))
        exp_1[1, 1, 0, 0] = exp_1[1, 2, 0, 1] = exp_1[2, 1, 1, 0] = exp_1[2, 2, 1, 1] = 1.0
        assert fn.allclose(exp_0, res[0])
        assert fn.allclose(exp_1, res[1])


gather_data = [
    torch.tensor([[1, 2, 3], [-1, -6, -3]]),
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
    """Test that qml.math.coerce works for all supported interfaces."""

    @pytest.mark.parametrize("coercion_interface", ["jax", "autograd", "scipy"])
    def test_trivial_coercions(self, coercion_interface):
        """Test coercion is trivial for JAX, Autograd, and Scipy."""
        tensors = [
            jnp.array([0.2]),
            onp.array([1, 2, 3]),
            torch.tensor(1 + 3j, dtype=torch.complex64),
            np.array([1, 2, 3]),
        ]
        expected_interfaces = ["jax", "numpy", "torch", "autograd"]
        res = qml.math.coerce(tensors, like=coercion_interface)
        for tensor, interface in zip(res, expected_interfaces, strict=True):
            assert fn.get_interface(tensor) == interface

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
        # pylint: disable=not-an-iterable
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
        # pylint: disable=not-an-iterable
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
        _ = qml.grad(cost_fn, argnum=[1, 2])(*values)

        expected = [np.array([0.1, 0.2]), 0.1, np.array([0.5, 0.2])]
        assert all(np.allclose(a, b) for a, b in zip(unwrapped_params, expected))
        assert not any(isinstance(a, ArrayBox) for a in unwrapped_params)

    def test_autograd_unwrapping_backward_nested(self):
        """Test that a sequence of Autograd values is properly unwrapped
        during multiple backward passes"""
        # pylint: disable=not-an-iterable
        unwrapped_params = None

        def cost_fn(p, max_depth=None):
            nonlocal unwrapped_params
            unwrapped_params = qml.math.unwrap(p, max_depth)
            return np.sum(np.sin(np.prod(p)))

        values = np.tensor([0.1, 0.2, 0.3])
        _ = qml.jacobian(qml.grad(cost_fn))(values)

        expected = np.array([0.1, 0.2, 0.3])
        assert np.allclose(unwrapped_params, expected)
        assert not isinstance(unwrapped_params, ArrayBox)

        # Specifying max_depth=1 will result in the second backward
        # pass not being unwrapped
        _ = qml.jacobian(qml.grad(cost_fn))(values, max_depth=1)
        assert all(isinstance(a, ArrayBox) for a in unwrapped_params)

    def test_jax_unwrapping(self):
        """Test that a sequence of Autograd values is properly unwrapped
        during the forward pass"""
        # pylint: disable=not-an-iterable
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
        _ = qml.grad(cost_fn)(*values)

        assert res == {0, 1}


test_sort_data = [
    ([1, 3, 4, 2], [1, 2, 3, 4]),
    (onp.array([1, 3, 4, 2]), onp.array([1, 2, 3, 4])),
    (np.array([1, 3, 4, 2]), np.array([1, 2, 3, 4])),
    (jnp.array([1, 3, 4, 2]), jnp.array([1, 2, 3, 4])),
    (torch.tensor([1, 3, 4, 2]), torch.tensor([1, 2, 3, 4])),
]


class TestSortFunction:
    """Test the sort function works across all interfaces"""

    # pylint: disable=too-few-public-methods

    @pytest.mark.parametrize("input, test_output", test_sort_data)
    def test_sort(self, input, test_output):
        """Test the sort method is outputting only sorted values not indices"""
        result = fn.sort(input)

        assert all(result == test_output)


class TestExpm:
    """Test the expm function works across all interfaces"""

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

    @pytest.mark.parametrize("phi", [qml.numpy.array(0.3), torch.tensor(0.3), jnp.array(0.3)])
    def test_expm(self, phi):
        """Test expm function for all interfaces against taylor expansion approximation."""
        orig_mat = qml.RX.compute_matrix(phi)
        exp_mat = qml.math.expm(orig_mat)

        assert qml.math.allclose(exp_mat, self.get_compare_mat(), atol=1e-4)


class TestSize:
    """Test qml.math.size method."""

    # pylint: disable=too-few-public-methods

    array_and_size = [
        ([], 0),
        (1, 1),
        ([0, 1, 2, 3, 4, 5], 6),
        ([[0, 1, 2], [3, 4, 5]], 6),
        ([[0, 1], [2, 3], [4, 5]], 6),
        ([[0], [1], [2], [3], [4], [5]], 6),
    ]

    @pytest.mark.parametrize(
        "interface",
        [
            pytest.param("torch", marks=pytest.mark.torch),
        ],
    )
    @pytest.mark.parametrize(("array", "size"), array_and_size)
    def test_size_torch_and_tf(self, array, size, interface):
        """Test size function with the torch and tf interfaces."""
        r = fn.size(fn.asarray(array, like=interface))
        assert r == size


@pytest.mark.parametrize("name", ["fft", "ifft", "fft2", "ifft2"])
class TestFft:
    """Test qml.math.fft functions and their differentiability."""

    arg = {
        "fft": onp.sin(onp.linspace(0, np.pi, 5)) - onp.cos(onp.linspace(0, np.pi, 5)) / 2,
        "ifft": onp.linspace(0, np.pi, 5),
        "fft2": onp.outer(
            0.4 * onp.sin(onp.linspace(0, onp.pi, 3)), np.cos(onp.linspace(onp.pi, 0, 2)) / 2
        ),
        "ifft2": onp.outer(
            0.4 * onp.sin(onp.linspace(0, onp.pi, 3)), np.cos(onp.linspace(onp.pi, 0, 2)) / 2
        ),
    }

    exp_fft = {
        "fft": [
            2.414213,
            -1.903347 + 0.17493j,
            -0.55376019 + 0.02838791j,
            -0.55376019 - 0.02838791j,
            -1.9033466 - 0.17493416j,
        ],
        "ifft": [
            1.57079633 + 0.0j,
            -0.39269908 - 0.54050392j,
            -0.39269908 - 0.12759567j,
            -0.39269908 + 0.12759567j,
            -0.39269908 + 0.54050392j,
        ],
        "fft2": [[0, -0.4], [0, 0.2 + 0.34641016j], [0, 0.2 - 0.34641016j]],
        "ifft2": [[0, -1 / 15 + 0.0j], [0, 1 / 30 - 0.05773503j], [0, 1 / 30 + 0.05773503j]],
    }

    exp_jac_fft = {
        "fft": [
            [1, 1, 1, 1, 1],
            [
                1,
                0.30901699 - 0.95105652j,
                -0.80901699 - 0.58778525j,
                -0.80901699 + 0.58778525j,
                0.30901699 + 0.95105652j,
            ],
            [
                1,
                -0.80901699 - 0.58778525j,
                0.30901699 + 0.95105652j,
                0.30901699 - 0.95105652j,
                -0.80901699 + 0.58778525j,
            ],
            [
                1,
                -0.80901699 + 0.58778525j,
                0.30901699 - 0.95105652j,
                0.30901699 + 0.95105652j,
                -0.80901699 - 0.58778525j,
            ],
            [
                1,
                0.30901699 + 0.95105652j,
                -0.80901699 + 0.58778525j,
                -0.80901699 - 0.58778525j,
                0.30901699 - 0.95105652j,
            ],
        ],
        "ifft": [
            [0.2 + 0.0j, 0.2 + 0.0j, 0.2 + 0.0j, 0.2 + 0.0j, 0.2 + 0.0j],
            [
                0.2 + 0.0j,
                0.0618034 + 0.1902113j,
                -0.1618034 + 0.11755705j,
                -0.1618034 - 0.11755705j,
                0.0618034 - 0.1902113j,
            ],
            [
                0.2 + 0.0j,
                -0.1618034 + 0.11755705j,
                0.0618034 - 0.1902113j,
                0.0618034 + 0.1902113j,
                -0.1618034 - 0.11755705j,
            ],
            [
                0.2 + 0.0j,
                -0.1618034 - 0.11755705j,
                0.0618034 + 0.1902113j,
                0.0618034 - 0.1902113j,
                -0.1618034 + 0.11755705j,
            ],
            [
                0.2 + 0.0j,
                0.0618034 - 0.1902113j,
                -0.1618034 - 0.11755705j,
                -0.1618034 + 0.11755705j,
                0.0618034 + 0.1902113j,
            ],
        ],
        "fft2": [
            [[[1, 1], [1, 1], [1, 1]], [[1, -1], [1, -1], [1, -1]]],
            [
                [[1, 1], [-0.5 - (x := 0.8660254j), -0.5 - x], [-0.5 + x, -0.5 + x]],
                [[1, -1], [-0.5 - x, 0.5 + x], [-0.5 + x, 0.5 - x]],
            ],
            [
                [[1, 1], [-0.5 + x, -0.5 + x], [-0.5 - x, -0.5 - x]],
                [[1, -1], [-0.5 + x, 0.5 - x], [-0.5 - x, 0.5 + x]],
            ],
        ],
        "ifft2": [
            [
                [[1 / 6, 1 / 6], [1 / 6, 1 / 6], [1 / 6, 1 / 6]],
                [[1 / 6, -1 / 6], [1 / 6, -1 / 6], [1 / 6, -1 / 6]],
            ],
            [
                [
                    [1 / 6, 1 / 6],
                    [-1 / 12 + (x := 0.14433757j), -1 / 12 + x],
                    [-1 / 12 - x, -1 / 12 - x],
                ],
                [[1 / 6, -1 / 6], [-1 / 12 + x, 1 / 12 - x], [-1 / 12 - x, 1 / 12 + x]],
            ],
            [
                [[1 / 6, 1 / 6], [-1 / 12 - x, -1 / 12 - x], [-1 / 12 + x, -1 / 12 + x]],
                [[1 / 6, -1 / 6], [-1 / 12 - x, 1 / 12 + x], [-1 / 12 + x, 1 / 12 - x]],
            ],
        ],
    }

    @staticmethod
    def fft_real(x, func=None):
        """Compute the real part of an FFT function output."""
        return qml.math.real(func(x))

    @staticmethod
    def fft_imag(x, func=None):
        """Compute the imag part of an FFT function output."""
        return qml.math.imag(func(x))

    def test_numpy(self, name):
        """Test that the functions are available in Numpy."""
        func = getattr(qml.math.fft, name)
        out = func(self.arg[name])
        assert qml.math.allclose(out, self.exp_fft[name])

    @pytest.mark.autograd
    def test_autograd(self, name):
        """Test that the functions are available in Autograd."""
        func = getattr(qml.math.fft, name)
        arg = np.array(self.arg[name], requires_grad=True)
        out = func(arg)
        assert qml.math.allclose(out, self.exp_fft[name])
        jac_real = qml.jacobian(self.fft_real)(arg, func=func)
        jac_imag = qml.jacobian(self.fft_imag)(arg, func=func)
        assert qml.math.allclose(jac_real + 1j * jac_imag, self.exp_jac_fft[name])

    @pytest.mark.jax
    def test_jax(self, name):
        """Test that the functions are available in JAX."""
        func = getattr(qml.math.fft, name)
        arg = jax.numpy.array(self.arg[name], dtype=jax.numpy.complex64)
        out = func(arg)
        assert qml.math.allclose(out, self.exp_fft[name])
        jac = jax.jacobian(func, holomorphic=True)(arg)
        assert qml.math.allclose(jac, self.exp_jac_fft[name])

    @pytest.mark.torch
    def test_torch(self, name):
        """Test that the functions are available in PyTorch."""
        func = getattr(qml.math.fft, name)
        arg = torch.tensor(self.arg[name], requires_grad=True)
        out = func(arg)
        assert qml.math.allclose(out, self.exp_fft[name])
        jac_real = torch.autograd.functional.jacobian(partial(self.fft_real, func=func), arg)
        jac_imag = torch.autograd.functional.jacobian(partial(self.fft_imag, func=func), arg)
        print(jac_real + 1j * jac_imag)
        print(self.exp_jac_fft[name])
        assert qml.math.allclose(jac_real + 1j * jac_imag, self.exp_jac_fft[name])


def test_jax_ndim():
    """Test that qml.math.ndim dispatches to jax.numpy.ndim."""
    with patch("jax.numpy.ndim") as mock_ndim:
        _ = qml.math.ndim(jax.numpy.array(3))

    mock_ndim.assert_called_once_with(3)


class TestSetIndex:
    """Test the set_index method."""

    @pytest.mark.parametrize(
        "array", [qml.numpy.zeros((2, 2)), torch.zeros((2, 2)), jnp.zeros((2, 2))]
    )
    def test_set_index_jax_2d_array(self, array):
        """Test that an array can be created that is a copy of the
        original array, with the value at the specified index updated"""

        array2 = qml.math.set_index(array, (1, 1), 3)
        assert qml.math.allclose(array2, np.array([[0, 0], [0, 3]]))
        # since idx and val have no interface, we expect the returned array type to match initial type
        assert isinstance(array2, type(array))

    @pytest.mark.parametrize("array", [qml.numpy.zeros(4), torch.zeros(4), jnp.zeros(4)])
    def test_set_index_jax_1d_array(self, array):
        """Test that an array can be created that is a copy of the
        original array, with the value at the specified index updated"""

        array2 = qml.math.set_index(array, 3, 3)
        assert qml.math.allclose(array2, np.array([[0, 0, 0, 3]]))
        # since idx and val have no interface, we expect the returned array type to match initial type
        assert isinstance(array2, type(array))

    @pytest.mark.parametrize(
        "array",
        [jnp.array([[1, 2], [3, 4]]), onp.array([[1, 2], [3, 4]]), np.array([[1, 2], [3, 4]])],
    )
    def test_set_index_with_val_tracer(self, array):
        """Test that for both jax and numpy arrays, if the val to set is a tracer,
        the set_index function succeeds and returns an updated jax array"""
        from jax.interpreters.partial_eval import DynamicJaxprTracer

        @jax.jit
        def jitted_function(x):
            assert isinstance(x, DynamicJaxprTracer)
            return qml.math.set_index(array, (0, 0), x)

        val = jnp.array(7)
        array2 = jitted_function(val)

        assert qml.math.allclose(array2, jnp.array([[7, 2], [3, 4]]))
        assert isinstance(array2, jnp.ndarray)

    @pytest.mark.parametrize(
        "array",
        [jnp.array([[1, 2], [3, 4]]), onp.array([[1, 2], [3, 4]]), np.array([[1, 2], [3, 4]])],
    )
    def test_set_index_with_idx_tracer_2D_array(self, array):
        """Test that for both jax and numpy 2d arrays, if the idx to set is a tracer,
        the set_index function succeeds and returns an updated jax array"""
        from jax.interpreters.partial_eval import DynamicJaxprTracer

        @jax.jit
        def jitted_function(y):
            assert isinstance(y, DynamicJaxprTracer)
            return qml.math.set_index(array, (1 + y, y), 7)

        val = jnp.array(0)
        array2 = jitted_function(val)

        assert qml.math.allclose(array2, jnp.array([[1, 2], [7, 4]]))
        assert isinstance(array2, jnp.ndarray)

    @pytest.mark.parametrize(
        "array", [jnp.array([1, 2, 3, 4]), onp.array([1, 2, 3, 4]), np.array([1, 2, 3, 4])]
    )
    def test_set_index_with_idx_tracer_1D_array(self, array):
        """Test that for both jax and numpy 1d arrays, if the idx to set is a tracer,
        the set_index function succeeds and returns an updated jax array"""
        from jax.interpreters.partial_eval import DynamicJaxprTracer

        @jax.jit
        def jitted_function(y):
            assert isinstance(y, DynamicJaxprTracer)
            return qml.math.set_index(array, y, 7)

        val = jnp.array(0)
        array2 = jitted_function(val)

        assert qml.math.allclose(array2, jnp.array([[7, 2, 3, 4]]))
        assert isinstance(array2, jnp.ndarray)


class TestScatter:
    """Tests for qml.math.scatter functionality"""

    @pytest.mark.all_interfaces
    @pytest.mark.parametrize("interface", ["numpy", "jax", "torch"])
    def test_scatter_basic(self, interface):
        """Test basic scatter operation - placing values at specific indices in a zero array"""
        indices = [0, 2, 4]
        updates = [1.0, 2.0, 3.0]
        shape = [6]

        updates = qml.math.asarray(updates, like=interface)
        indices = qml.math.asarray(indices, like=interface)

        result = qml.math.scatter(indices, updates, shape)
        expected = qml.math.asarray([1.0, 0.0, 2.0, 0.0, 3.0, 0.0], like=interface)

        assert qml.math.allclose(result, expected)

    @pytest.mark.all_interfaces
    @pytest.mark.parametrize("interface", ["numpy", "jax", "torch"])
    def test_scatter_complex(self, interface):
        """Test scatter with complex values"""
        indices = [1, 3]
        updates = [1.0 + 1.0j, 2.0 - 1.0j]
        shape = [4]

        updates = qml.math.asarray(updates, like=interface)
        indices = qml.math.asarray(indices, like=interface)

        result = qml.math.scatter(indices, updates, shape)
        expected = qml.math.asarray(
            [0.0 + 0.0j, 1.0 + 1.0j, 0.0 + 0.0j, 2.0 - 1.0j], like=interface
        )

        assert qml.math.allclose(result, expected)

    @pytest.mark.all_interfaces
    @pytest.mark.parametrize("interface", ["numpy", "jax", "torch"])
    def test_scatter_multidimensional(self, interface):
        """Test scatter with multidimensional target shape"""
        indices = [0, 2]
        updates = [[1.0, 2.0], [3.0, 4.0]]
        shape = [3, 2]  # 3x2 target array

        updates = qml.math.asarray(updates, like=interface)
        indices = qml.math.asarray(indices, like=interface)

        result = qml.math.scatter(indices, updates, shape)
        expected = qml.math.asarray([[1.0, 2.0], [0.0, 0.0], [3.0, 4.0]], like=interface)

        assert qml.math.allclose(result, expected)
