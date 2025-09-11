# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Autoray registrations"""

# pylint: disable=protected-access,import-outside-toplevel,disable=unnecessary-lambda
from importlib import import_module

# pylint: disable=wrong-import-order
import autoray as ar
import numpy as np
import scipy as sp
from packaging.version import Version
from scipy.linalg import block_diag as _scipy_block_diag

from .interface_utils import get_deep_interface
from .utils import is_abstract


def _i(name):
    """Convenience function to import PennyLane
    interfaces via a string pattern"""
    if name == "tf":  # pragma: no cover (TensorFlow tests were disabled during deprecation)
        return import_module("tensorflow")

    if name == "qml":
        return import_module("pennylane")

    return import_module(name)


# ------------------------------- Builtins -------------------------------- #


def _builtins_ndim(x):
    interface = get_deep_interface(x)
    x = ar.numpy.asarray(x, like=interface)
    return ar.ndim(x)


def _builtins_shape(x):
    interface = get_deep_interface(x)
    x = ar.numpy.asarray(x, like=interface)
    return ar.shape(x)


ar.register_function("builtins", "ndim", _builtins_ndim)
ar.register_function("builtins", "shape", _builtins_shape)
ar.register_function("builtins", "coerce", lambda x: x)
ar.register_function("builtins", "logical_xor", lambda x, y: x ^ y)

# -------------------------------- SciPy --------------------------------- #
# the following is required to ensure that SciPy sparse Hamiltonians passed to
# qml.SparseHamiltonian are not automatically 'unwrapped' to dense NumPy arrays.
ar.register_function("scipy", "to_numpy", lambda x: x)
ar.register_function("scipy", "coerce", lambda x: x)
ar.register_function("scipy", "array", lambda x: x)
ar.register_function("scipy", "shape", np.shape)
ar.register_function("scipy", "dot", np.dot)
ar.register_function("scipy", "conj", np.conj)
ar.register_function("scipy", "transpose", np.transpose)
ar.register_function("scipy", "ndim", np.ndim)


# -------------------------------- SciPy Sparse --------------------------------- #
# the following is required to ensure that general SciPy sparse matrices are
# not automatically 'unwrapped' to dense NumPy arrays. Note that we assume
# that whenever the backend is 'scipy', the input is a SciPy sparse matrix.


def sparse_matrix_power(A, n):
    """Dispatch to the appropriate sparse matrix power function."""
    try:  # pragma: no cover
        # pylint: disable=import-outside-toplevel
        from scipy.sparse.linalg import matrix_power

        # added in scipy 1.12.0

        return matrix_power(A, n)
    except ImportError:  # pragma: no cover
        return _sparse_matrix_power_bruteforce(A, n)


def _sparse_matrix_power_bruteforce(A, n):
    """
    Compute the power of a sparse matrix using brute-force matrix multiplication.
    Supports only non-negative integer exponents.

    Parameters:
    A : scipy.sparse matrix
        The sparse matrix to be exponentiated.
    n : int
        The exponent (must be non-negative).

    Returns:
    scipy.sparse matrix
        The matrix A raised to the power n.
    """

    if n < 0:
        raise ValueError("This function only supports non-negative integer exponents.")

    if n == 0:
        return sp.sparse.eye(A.shape[0], dtype=A.dtype, format=A.format)  # Identity matrix

    try:
        matmul_range = range(n - 1)
    except Exception as e:
        raise ValueError("exponent must be an integer") from e

    result = A.copy()
    for _ in matmul_range:
        result = result @ A  # Native matmul operation

    return result


ar.register_function("scipy", "linalg.inv", sp.sparse.linalg.inv)
ar.register_function("scipy", "linalg.expm", sp.sparse.linalg.expm)
ar.register_function("scipy", "linalg.matrix_power", sparse_matrix_power)
ar.register_function("scipy", "linalg.norm", sp.sparse.linalg.norm)
ar.register_function("scipy", "linalg.spsolve", sp.sparse.linalg.spsolve)
ar.register_function("scipy", "linalg.eigs", sp.sparse.linalg.eigs)
ar.register_function("scipy", "linalg.eigsh", sp.sparse.linalg.eigsh)
ar.register_function("scipy", "linalg.svds", sp.sparse.linalg.svds)


ar.register_function("scipy", "trace", lambda x: x.trace())
ar.register_function("scipy", "reshape", lambda x, new_shape: x.reshape(new_shape))
ar.register_function("scipy", "real", lambda x: x.real)
ar.register_function("scipy", "imag", lambda x: x.imag)
ar.register_function("scipy", "size", lambda x: np.prod(x.shape))
ar.register_function("scipy", "eye", sp.sparse.eye)
ar.register_function("scipy", "zeros", sp.sparse.csr_matrix)
ar.register_function("scipy", "hstack", sp.sparse.hstack)
ar.register_function("scipy", "vstack", sp.sparse.vstack)

# -------------------------------- NumPy --------------------------------- #

ar.register_function("numpy", "flatten", lambda x: x.flatten())
ar.register_function("numpy", "coerce", lambda x: x)
ar.register_function("numpy", "block_diag", lambda x: _scipy_block_diag(*x))
ar.register_function("builtins", "block_diag", lambda x: _scipy_block_diag(*x))
ar.register_function("numpy", "gather", lambda x, indices: x[np.array(indices)])
ar.register_function("numpy", "unstack", list)

ar.register_function("builtins", "unstack", list)


def _scatter_numpy(indices, array, shape):
    new_array = np.zeros(shape, dtype=array.dtype.type)
    new_array[indices] = array
    return new_array


def _scatter_element_add_numpy(tensor, index, value, **_):
    """In-place addition of a multidimensional value over various
    indices of a tensor."""
    new_tensor = tensor.copy()
    new_tensor[tuple(index)] += value
    return new_tensor


ar.register_function("numpy", "scatter", _scatter_numpy)
ar.register_function("numpy", "scatter_element_add", _scatter_element_add_numpy)
ar.register_function("numpy", "eigvalsh", np.linalg.eigvalsh)
ar.register_function("numpy", "entr", lambda x: -np.sum(x * np.log(x), axis=-1))


def _cond(pred, true_fn, false_fn, args):
    if pred:
        return true_fn(*args)

    return false_fn(*args)


ar.register_function("numpy", "cond", _cond)
ar.register_function("builtins", "cond", _cond)

ar.register_function("numpy", "gamma", lambda x: _i("scipy").special.gamma(x))

ar.register_function("builtins", "gamma", lambda x: _i("scipy").special.gamma(x))

# -------------------------------- Autograd --------------------------------- #


# When autoray inspects PennyLane NumPy tensors, they will be associated with
# the 'pennylane' module, and not autograd. Set an alias so it understands this is
# simply autograd.
ar.autoray._BACKEND_ALIASES["pennylane"] = "autograd"

# When dispatching to autograd, ensure that autoray will instead call
# qml.numpy rather than autograd.numpy, to take into account our autograd modification.
ar.autoray._MODULE_ALIASES["autograd"] = "pennylane.numpy"

ar.register_function("autograd", "ndim", lambda x: _i("autograd").numpy.ndim(x))
ar.register_function("autograd", "shape", lambda x: _i("autograd").numpy.shape(x))
ar.register_function("autograd", "flatten", lambda x: x.flatten())
ar.register_function("autograd", "coerce", lambda x: x)
ar.register_function("autograd", "gather", lambda x, indices: x[np.array(indices)])
ar.register_function("autograd", "unstack", list)


def autograd_get_dtype_name(x):
    """A autograd version of get_dtype_name that can handle array boxes."""
    # this function seems to only get called with x is an arraybox.
    return ar.get_dtype_name(x._value)


ar.register_function("autograd", "get_dtype_name", autograd_get_dtype_name)


def _block_diag_autograd(tensors):
    """Autograd implementation of scipy.linalg.block_diag"""
    _np = _i("qml").numpy
    tensors = [t.reshape((1, len(t))) if len(t.shape) == 1 else t for t in tensors]
    rsizes, csizes = _np.array([t.shape for t in tensors]).T
    all_zeros = [[_np.zeros((rsize, csize)) for csize in csizes] for rsize in rsizes]

    res = _np.hstack([tensors[0], *all_zeros[0][1:]])
    for i, t in enumerate(tensors[1:], start=1):
        row = _np.hstack([*all_zeros[i][:i], t, *all_zeros[i][i + 1 :]])
        res = _np.vstack([res, row])

    return res


ar.register_function("autograd", "block_diag", _block_diag_autograd)


def _unwrap_arraybox(arraybox, max_depth=None, _n=0):
    if max_depth is not None and _n == max_depth:
        return arraybox

    val = getattr(arraybox, "_value", arraybox)

    if hasattr(val, "_value"):
        return _unwrap_arraybox(val, max_depth=max_depth, _n=_n + 1)

    return val


def _to_numpy_autograd(x, max_depth=None, _n=0):
    if hasattr(x, "_value"):
        # Catches the edge case where the data is an Autograd arraybox,
        # which only occurs during backpropagation.
        return _unwrap_arraybox(x, max_depth=max_depth, _n=_n)

    return x.numpy()


ar.register_function("autograd", "to_numpy", _to_numpy_autograd)


def _scatter_element_add_autograd(tensor, index, value, **_):
    """In-place addition of a multidimensional value over various
    indices of a tensor. Since Autograd doesn't support indexing
    assignment, we have to be clever and use ravel_multi_index."""
    pnp = _i("qml").numpy
    size = tensor.size
    flat_index = pnp.ravel_multi_index(index, tensor.shape)
    if pnp.isscalar(flat_index):
        flat_index = [flat_index]
    if pnp.isscalar(value) or len(pnp.shape(value)) == 0:
        value = [value]
    t = [0] * size
    for _id, val in zip(flat_index, value, strict=True):
        t[_id] = val
    return tensor + pnp.array(t).reshape(tensor.shape)


ar.register_function("autograd", "scatter_element_add", _scatter_element_add_autograd)


def _take_autograd(tensor, indices, axis=None):
    indices = _i("qml").numpy.asarray(indices)

    if axis is None:
        return tensor.flatten()[indices]

    fancy_indices = [slice(None)] * axis + [indices]
    return tensor[tuple(fancy_indices)]


ar.register_function("autograd", "take", _take_autograd)
ar.register_function("autograd", "eigvalsh", lambda x: _i("autograd").numpy.linalg.eigh(x)[0])
ar.register_function(
    "autograd",
    "entr",
    lambda x: -_i("autograd").numpy.sum(x * _i("autograd").numpy.log(x), axis=-1),
)

ar.register_function("autograd", "diagonal", lambda x, *args: _i("qml").numpy.diag(x))
ar.register_function("autograd", "cond", _cond)

ar.register_function("autograd", "gamma", lambda x: _i("autograd.scipy").special.gamma(x))


# -------------------------------- TensorFlow --------------------------------- #


ar.autoray._SUBMODULE_ALIASES["tensorflow", "angle"] = "tensorflow.math"
ar.autoray._SUBMODULE_ALIASES["tensorflow", "arcsin"] = "tensorflow.math"
ar.autoray._SUBMODULE_ALIASES["tensorflow", "arccos"] = "tensorflow.math"
ar.autoray._SUBMODULE_ALIASES["tensorflow", "arctan"] = "tensorflow.math"
ar.autoray._SUBMODULE_ALIASES["tensorflow", "arctan2"] = "tensorflow.math"
ar.autoray._SUBMODULE_ALIASES["tensorflow", "mod"] = "tensorflow.math"
ar.autoray._SUBMODULE_ALIASES["tensorflow", "logical_and"] = "tensorflow.math"
ar.autoray._SUBMODULE_ALIASES["tensorflow", "kron"] = "tensorflow.experimental.numpy"
ar.autoray._SUBMODULE_ALIASES["tensorflow", "moveaxis"] = "tensorflow.experimental.numpy"
ar.autoray._SUBMODULE_ALIASES["tensorflow", "sinc"] = "tensorflow.experimental.numpy"
ar.autoray._SUBMODULE_ALIASES["tensorflow", "isclose"] = "tensorflow.experimental.numpy"
ar.autoray._SUBMODULE_ALIASES["tensorflow", "atleast_1d"] = "tensorflow.experimental.numpy"
ar.autoray._SUBMODULE_ALIASES["tensorflow", "all"] = "tensorflow.experimental.numpy"
ar.autoray._SUBMODULE_ALIASES["tensorflow", "ravel"] = "tensorflow.experimental.numpy"
ar.autoray._SUBMODULE_ALIASES["tensorflow", "vstack"] = "tensorflow.experimental.numpy"
ar.autoray._SUBMODULE_ALIASES["tensorflow", "unstack"] = "tensorflow"
ar.autoray._SUBMODULE_ALIASES["tensorflow", "gather"] = "tensorflow"
ar.autoray._SUBMODULE_ALIASES["tensorflow", "concat"] = "tensorflow"


tf_fft_functions = [
    "fft",
    "ifft",
    "fft2",
    "ifft2",
]


for fn in tf_fft_functions:
    ar.autoray._SUBMODULE_ALIASES["tensorflow", "fft." + fn] = "tensorflow.signal"

ar.autoray._FUNC_ALIASES["tensorflow", "fft.fft2"] = "fft2d"
ar.autoray._FUNC_ALIASES["tensorflow", "fft.ifft2"] = "ifft2d"

ar.autoray._FUNC_ALIASES["tensorflow", "arcsin"] = "asin"
ar.autoray._FUNC_ALIASES["tensorflow", "arccos"] = "acos"
ar.autoray._FUNC_ALIASES["tensorflow", "arctan"] = "atan"
ar.autoray._FUNC_ALIASES["tensorflow", "arctan2"] = "atan2"


def _coerce_tensorflow_diag(
    x, **kwargs
):  # pragma: no cover (TensorFlow tests were disabled during deprecation)
    x = _tf_convert_to_tensor(x)
    tf = _i("tf")
    nd = len(x.shape)
    if nd == 2:
        return tf.linalg.diag_part(x, **kwargs)
    if nd == 1:
        return tf.linalg.diag(x, **kwargs)
    raise ValueError("Input must be 1- or 2-d.")


ar.register_function("tensorflow", "diag", _coerce_tensorflow_diag)


def _tensorflow_allclose(
    a, b, **kwargs
):  # pragma: no cover (TensorFlow tests were disabled during deprecation)
    if is_abstract(a):
        a = ar.to_numpy(a)
    return ar.autoray.allclose(a, b, **kwargs)


ar.register_function("tensorflow", "allclose", _tensorflow_allclose)

ar.register_function(
    "tensorflow",
    "finfo",
    lambda *args, **kwargs: _i("tf").experimental.numpy.finfo(*args, **kwargs),
)


def _tf_convert_to_tensor(
    x, requires_grad=False, **kwargs
):  # pragma: no cover (TensorFlow tests were disabled during deprecation)
    if isinstance(x, _i("tf").Tensor) and "dtype" in kwargs:
        out = _i("tf").cast(x, **kwargs)
    else:
        out = _i("tf").convert_to_tensor(x, **kwargs)
    return _i("tf").Variable(out) if requires_grad else out


ar.register_function("tensorflow", "asarray", _tf_convert_to_tensor)
ar.register_function(
    "tensorflow",
    "hstack",
    lambda *args, **kwargs: _i("tf").experimental.numpy.hstack(*args),
)

ar.register_function("tensorflow", "flatten", lambda x: _i("tf").reshape(x, [-1]))
ar.register_function("tensorflow", "shape", lambda x: tuple(x.shape))
ar.register_function(
    "tensorflow",
    "full",
    lambda shape, fill_value, **kwargs: _i("tf").fill(
        shape if isinstance(shape, (tuple, list)) else (shape), fill_value, **kwargs
    ),
)
ar.register_function(
    "tensorflow",
    "isnan",
    lambda tensor, **kwargs: _i("tf").math.is_nan(_i("tf").math.real(tensor), **kwargs)
    | _i("tf").math.is_nan(_i("tf").math.imag(tensor), **kwargs),
)
ar.register_function(
    "tensorflow", "any", lambda tensor, **kwargs: _i("tf").reduce_any(tensor, **kwargs)
)
ar.register_function(
    "tensorflow",
    "sqrt",
    lambda x: _i("tf").math.sqrt(
        _i("tf").cast(x, "float64") if x.dtype.name in ("int64", "int32") else x
    ),
)


def _ifft2_tf(
    a, s=None, axes=(-2, -1), norm=None
):  # pragma: no cover (TensorFlow tests were disabled during deprecation)
    if axes != (-2, -1):
        raise ValueError(
            "TensorFlow does not support passing axes; the ifft "
            "will always be performed over the inner-most 2 dimensions."
        )

    if norm is not None:
        raise ValueError("TensorFlow does not support the 'norm' keyword argument.")
    if s is not None:
        raise ValueError("TensorFlow does not support the 's' keyword argument.")

    # TensorFlow only supports FFT of complex tensors
    if a.dtype not in [_i("tf").complex64, _i("tf").complex128]:
        if a.dtype is _i("tf").float64:
            a = _i("tf").cast(a, dtype=_i("tf").complex128)
        else:
            a = _i("tf").cast(a, dtype=_i("tf").complex64)

    return _i("tf").signal.ifft2d(input=a)


ar.register_function("tensorflow", "fft.ifft2", _ifft2_tf)


def _round_tf(
    tensor, decimals=0
):  # pragma: no cover (TensorFlow tests were disabled during deprecation)
    """Implement a TensorFlow version of np.round"""
    tf = _i("tf")
    tol = 10**decimals
    return tf.round(tensor * tol) / tol


ar.register_function("tensorflow", "round", _round_tf)


def _ndim_tf(tensor):  # pragma: no cover (TensorFlow tests were disabled during deprecation)
    return len(tensor.shape)


ar.register_function("tensorflow", "ndim", _ndim_tf)


def _take_tf(
    tensor, indices, axis=None, **kwargs
):  # pragma: no cover (TensorFlow tests were disabled during deprecation)
    tf = _i("tf")
    return tf.experimental.numpy.take(tensor, indices, axis=axis, **kwargs)


ar.register_function("tensorflow", "take", _take_tf)


def _coerce_types_tf(
    tensors,
):  # pragma: no cover (TensorFlow tests were disabled during deprecation)
    """Coerce the dtypes of a list of tensors so that they
    all share the same dtype, without any reduction in information."""
    tf = _i("tf")
    tensors = [tf.convert_to_tensor(t) for t in tensors]
    dtypes = {i.dtype for i in tensors}

    if len(dtypes) == 1:
        return tensors

    complex_priority = [tf.complex64, tf.complex128]
    float_priority = [tf.float16, tf.float32, tf.float64]
    int_priority = [tf.int8, tf.int16, tf.int32, tf.int64]

    complex_type = [i for i in complex_priority if i in dtypes]
    float_type = [i for i in float_priority if i in dtypes]
    int_type = [i for i in int_priority if i in dtypes]

    cast_type = complex_type or float_type or int_type
    cast_type = list(cast_type)[-1]

    return [tf.cast(t, cast_type) for t in tensors]


ar.register_function("tensorflow", "coerce", _coerce_types_tf)


def _block_diag_tf(tensors):  # pragma: no cover (TensorFlow tests were disabled during deprecation)
    """TensorFlow implementation of scipy.linalg.block_diag"""
    tf = _i("tf")
    int_dtype = None

    if tensors[0].dtype in (tf.int32, tf.int64):
        int_dtype = tensors[0].dtype
        tensors = [tf.cast(t, tf.float32) for t in tensors]

    linop_blocks = [tf.linalg.LinearOperatorFullMatrix(block) for block in tensors]
    linop_block_diagonal = tf.linalg.LinearOperatorBlockDiag(linop_blocks)
    res = linop_block_diagonal.to_dense()

    if int_dtype is None:
        return res

    return tf.cast(res, int_dtype)


ar.register_function("tensorflow", "block_diag", _block_diag_tf)


def _scatter_tf(
    indices, array, new_dims
):  # pragma: no cover (TensorFlow tests were disabled during deprecation)
    import tensorflow as tf

    indices = np.expand_dims(indices, 1)
    return tf.scatter_nd(indices, array, new_dims)


def _scatter_element_add_tf(
    tensor, index, value, **_
):  # pragma: no cover (TensorFlow tests were disabled during deprecation)
    """In-place addition of a multidimensional value over various
    indices of a tensor."""
    import tensorflow as tf

    if not isinstance(index[0], int):
        index = tuple(zip(*index, strict=True))
    indices = tf.expand_dims(index, 0)
    value = tf.cast(tf.expand_dims(value, 0), tensor.dtype)
    return tf.tensor_scatter_nd_add(tensor, indices, value)


ar.register_function("tensorflow", "scatter", _scatter_tf)
ar.register_function("tensorflow", "scatter_element_add", _scatter_element_add_tf)


def _transpose_tf(
    a, axes=None
):  # pragma: no cover (TensorFlow tests were disabled during deprecation)
    import tensorflow as tf

    return tf.transpose(a, perm=axes)


ar.register_function("tensorflow", "transpose", _transpose_tf)
ar.register_function("tensorflow", "diagonal", lambda x, *args: _i("tf").linalg.diag_part(x))
ar.register_function("tensorflow", "outer", lambda a, b: _i("tf").tensordot(a, b, axes=0))

# for some reason Autoray modifies the default behaviour, so we change it back here
ar.register_function("tensorflow", "where", lambda *args, **kwargs: _i("tf").where(*args, **kwargs))


def _eigvalsh_tf(
    density_matrix,
):  # pragma: no cover (TensorFlow tests were disabled during deprecation)
    evs = _i("tf").linalg.eigvalsh(density_matrix)
    evs = _i("tf").math.real(evs)
    return evs


ar.register_function("tensorflow", "eigvalsh", _eigvalsh_tf)
ar.register_function(
    "tensorflow", "entr", lambda x: -_i("tf").math.reduce_sum(x * _i("tf").math.log(x), axis=-1)
)


def _kron_tf(a, b):  # pragma: no cover (TensorFlow tests were disabled during deprecation)
    import tensorflow as tf

    a_shape = a.shape
    b_shape = b.shape

    if len(a_shape) == 1:
        a = a[:, tf.newaxis]
        b = b[tf.newaxis, :]
        return tf.reshape(a * b, (a_shape[0] * b_shape[0],))

    a = a[:, tf.newaxis, :, tf.newaxis]
    b = b[tf.newaxis, :, tf.newaxis, :]
    return tf.reshape(a * b, (a_shape[0] * b_shape[0], a_shape[1] * b_shape[1]))


ar.register_function("tensorflow", "kron", _kron_tf)


def _cond_tf(
    pred, true_fn, false_fn, args
):  # pragma: no cover (TensorFlow tests were disabled during deprecation)
    import tensorflow as tf

    return tf.cond(pred, lambda: true_fn(*args), lambda: false_fn(*args))


ar.register_function("tensorflow", "cond", _cond_tf)


ar.register_function(
    "tensorflow",
    "vander",
    lambda *args, **kwargs: _i("tf").experimental.numpy.vander(*args, **kwargs),
)
ar.register_function("tensorflow", "size", lambda x: _i("tf").size(x))


# -------------------------------- Torch --------------------------------- #

ar.autoray._FUNC_ALIASES["torch", "unstack"] = "unbind"


def _to_numpy_torch(x):
    if getattr(x, "is_conj", False) and x.is_conj():  # pragma: no cover
        # The following line is only covered if using Torch <v1.10.0
        x = x.resolve_conj()

    return x.detach().cpu().numpy()


ar.register_function("torch", "to_numpy", _to_numpy_torch)


def _asarray_torch(x, dtype=None, requires_grad=False, **kwargs):
    import torch

    dtype_map = {
        np.int8: torch.int8,
        np.int16: torch.int16,
        np.int32: torch.int32,
        np.int64: torch.int64,
        np.float16: torch.float16,
        np.float32: torch.float32,
        np.float64: torch.float64,
        np.complex64: torch.complex64,
        np.complex128: torch.complex128,
        "float64": torch.float64,
    }
    dtype = dtype_map.get(dtype, dtype)
    if requires_grad:
        return torch.tensor(x, dtype=dtype, **kwargs, requires_grad=True)

    return torch.as_tensor(x, dtype=dtype, **kwargs)


ar.register_function("torch", "asarray", _asarray_torch)
ar.register_function("torch", "diag", lambda x, k=0: _i("torch").diag(x, diagonal=k))
ar.register_function("torch", "expand_dims", lambda x, axis: _i("torch").unsqueeze(x, dim=axis))
ar.register_function("torch", "shape", lambda x: tuple(x.shape))
ar.register_function("torch", "gather", lambda x, indices: x[indices])
ar.register_function("torch", "equal", lambda x, y: _i("torch").eq(x, y))
ar.register_function("torch", "mod", lambda x, y: x % y)

ar.register_function(
    "torch",
    "fft.ifft2",
    lambda a, s=None, axes=(-2, -1), norm=None: _i("torch").fft.ifft2(
        input=a, s=s, dim=axes, norm=norm
    ),
)


ar.register_function(
    "torch",
    "sqrt",
    lambda x: _i("torch").sqrt(
        x.to(_i("torch").float64) if x.dtype in (_i("torch").int64, _i("torch").int32) else x
    ),
)

ar.autoray._SUBMODULE_ALIASES["torch", "arctan2"] = "torch"
ar.autoray._FUNC_ALIASES["torch", "arctan2"] = "atan2"


def _round_torch(tensor, decimals=0):
    """Implement a Torch version of np.round"""
    torch = _i("torch")
    tol = 10**decimals
    return torch.round(tensor * tol) / tol


ar.register_function("torch", "round", _round_torch)


def _take_torch(tensor, indices, axis=None, **_):
    """Torch implementation of np.take"""
    torch = _i("torch")

    if not isinstance(indices, torch.Tensor):
        indices = torch.as_tensor(indices)

    if axis is None:
        return tensor.flatten()[indices]

    if indices.ndim == 1:
        if (indices < 0).any():
            # index_select doesn't allow negative indices
            dim_length = tensor.size()[0] if axis is None else tensor.shape[axis]
            indices = torch.where(indices >= 0, indices, indices + dim_length)

        return torch.index_select(tensor, dim=axis, index=indices)

    if axis == -1:
        return tensor[..., indices]
    fancy_indices = [slice(None)] * axis + [indices]
    return tensor[fancy_indices]


ar.register_function("torch", "take", _take_torch)


def _coerce_types_torch(tensors):
    """Coerce a list of tensors to all have the same dtype
    without any loss of information."""
    torch = _i("torch")

    # Extract existing set devices, if any
    device_set = set()
    dev_indices = set()
    for t in tensors:
        if isinstance(t, torch.Tensor):
            device_set.add(t.device.type)
            dev_indices.add(t.device.index)
        else:
            device_set.add("cpu")
            dev_indices.add(None)

    if len(device_set) > 1:  # pragma: no cover
        # If data exists on two separate GPUs, outright fail
        if len([i for i in dev_indices if i is not None]) > 1:
            device_names = ", ".join(str(d) for d in device_set)

            raise RuntimeError(
                f"Expected all tensors to be on the same device, but found at least two devices, {device_names}!"
            )
        # Otherwise, automigrate data from CPU to GPU and carry on.
        dev_indices.remove(None)
        dev_id = dev_indices.pop()
        tensors = [
            torch.as_tensor(t, device=torch.device(f"cuda:{dev_id}"))
            for t in tensors  # pragma: no cover
        ]
    else:
        device = device_set.pop()
        dev_id = dev_indices.pop() if dev_indices else None
        torch_device = torch.device(f"{device}:{dev_id}" if dev_id is not None else device)
        tensors = [torch.as_tensor(t, device=torch_device) for t in tensors]

    dtypes = {i.dtype for i in tensors}

    if len(dtypes) == 1:
        return tensors

    complex_priority = [torch.complex64, torch.complex128]
    float_priority = [torch.float16, torch.float32, torch.float64]
    int_priority = [torch.int8, torch.int16, torch.int32, torch.int64]

    complex_type = [i for i in complex_priority if i in dtypes]
    float_type = [i for i in float_priority if i in dtypes]
    int_type = [i for i in int_priority if i in dtypes]

    cast_type = complex_type or float_type or int_type
    cast_type = list(cast_type)[-1]

    return [t.to(cast_type) for t in tensors]


ar.register_function("torch", "coerce", _coerce_types_torch)


def _block_diag_torch(tensors):
    """Torch implementation of scipy.linalg.block_diag"""
    torch = _i("torch")
    sizes = np.array([t.shape for t in tensors])
    shape = np.sum(sizes, axis=0).tolist()
    res = torch.zeros(shape, dtype=tensors[0].dtype, device=tensors[0].device)

    # get the diagonal indices at which new block
    # diagonals need to be inserted
    p = np.cumsum(sizes, axis=0)

    # converted the diagonal indices to row and column indices
    ridx, cidx = np.stack([p - sizes, p]).T

    for t, r, c in zip(tensors, ridx, cidx, strict=True):
        row = np.arange(*r).reshape(-1, 1)
        col = np.arange(*c).reshape(1, -1)
        res[row, col] = t

    return res


ar.register_function("torch", "block_diag", _block_diag_torch)


def _scatter_torch(indices, tensor, new_dimensions):
    import torch

    new_tensor = torch.zeros(new_dimensions, dtype=tensor.dtype, device=tensor.device)
    new_tensor[indices] = tensor
    return new_tensor


def _scatter_element_add_torch(tensor, index, value, **_):
    """In-place addition of a multidimensional value over various
    indices of a tensor. Note that Torch only supports index assignments
    on non-leaf nodes; if the node is a leaf, we must clone it first."""
    if tensor.is_leaf:
        tensor = tensor.clone()
    tensor[tuple(index)] += value
    return tensor


ar.register_function("torch", "scatter", _scatter_torch)
ar.register_function("torch", "scatter_element_add", _scatter_element_add_torch)


def _sort_torch(tensor):
    """Update handling of sort to return only values not indices."""
    sorted_tensor = _i("torch").sort(tensor)
    return sorted_tensor.values


ar.register_function("torch", "sort", _sort_torch)


def _tensordot_torch(tensor1, tensor2, axes):
    torch = _i("torch")
    if Version(torch.__version__) < Version("1.10.0") and axes == 0:
        return torch.outer(tensor1, tensor2)
    return torch.tensordot(tensor1, tensor2, axes)


ar.register_function("torch", "tensordot", _tensordot_torch)


def _ndim_torch(tensor):
    return tensor.dim()


def _size_torch(tensor):
    return tensor.numel()


ar.register_function("torch", "ndim", _ndim_torch)
ar.register_function("torch", "size", _size_torch)


ar.register_function("torch", "eigvalsh", lambda x: _i("torch").linalg.eigvalsh(x))
ar.register_function(
    "torch", "entr", lambda x: _i("torch").sum(_i("torch").special.entr(x), dim=-1)
)


def _sum_torch(tensor, axis=None, keepdims=False, dtype=None):
    import torch

    if axis is None:
        return torch.sum(tensor, dtype=dtype)

    if not isinstance(axis, int) and len(axis) == 0:
        return tensor

    return torch.sum(tensor, dim=axis, keepdim=keepdims, dtype=dtype)


ar.register_function("torch", "sum", _sum_torch)
ar.register_function("torch", "cond", _cond)


# -------------------------------- JAX --------------------------------- #


def _to_numpy_jax(x):
    from jax.core import concrete_or_error
    from jax.errors import ConcretizationTypeError, TracerArrayConversionError

    try:
        x = concrete_or_error(None, x)
        return np.array(x)
    except (ConcretizationTypeError, TracerArrayConversionError) as e:
        raise ValueError(
            "Converting a JAX array to a NumPy array not supported when using the JAX JIT."
        ) from e


ar.register_function("jax", "flatten", lambda x: x.flatten())
ar.register_function(
    "jax",
    "take",
    lambda x, indices, axis=None, **kwargs: _i("jax").numpy.take(
        x, _i("jax").numpy.asarray(indices), axis=axis, **kwargs
    ),
)
ar.register_function("jax", "coerce", lambda x: x)
ar.register_function("jax", "to_numpy", _to_numpy_jax)
ar.register_function("jax", "block_diag", lambda x: _i("jax").scipy.linalg.block_diag(*x))
ar.register_function("jax", "gather", lambda x, indices: x[np.array(indices)])


# pylint: disable=unused-argument
def _asarray_jax(x, dtype=None, requires_grad=False, **kwargs):
    return _i("jax").numpy.array(x, dtype=dtype, **kwargs)


ar.register_function("jax", "asarray", _asarray_jax)


def _ndim_jax(x):
    import jax.numpy as jnp

    return jnp.ndim(x)


ar.register_function("jax", "ndim", lambda x: _ndim_jax(x))


def _scatter_jax(indices, array, new_dimensions):
    from jax import numpy as jnp

    new_array = jnp.zeros(new_dimensions, dtype=array.dtype.type)
    new_array = new_array.at[indices].set(array)
    return new_array


ar.register_function("jax", "scatter", _scatter_jax)
ar.register_function(
    "jax",
    "scatter_element_add",
    lambda x, index, value, **kwargs: x.at[tuple(index)].add(value, **kwargs),
)
ar.register_function("jax", "unstack", list)
# pylint: disable=unnecessary-lambda
ar.register_function("jax", "eigvalsh", lambda x: _i("jax").numpy.linalg.eigvalsh(x))
ar.register_function(
    "jax", "entr", lambda x: _i("jax").numpy.sum(_i("jax").scipy.special.entr(x), axis=-1)
)

ar.register_function(
    "jax",
    "cond",
    lambda pred, true_fn, false_fn, args: _i("jax").lax.cond(pred, true_fn, false_fn, *args),
)

ar.register_function(
    "jax", "gamma", lambda x: _i("jax").numpy.exp(_i("jax").scipy.special.gammaln(x))
)
