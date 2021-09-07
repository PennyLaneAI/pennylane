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
# pylint:disable=protected-access,import-outside-toplevel,wrong-import-position
from importlib import import_module
import numbers

import autoray as ar
import numpy as np


def _i(name):
    """Convenience function to import PennyLane
    interfaces via a string pattern"""
    if name == "tf":
        return import_module("tensorflow")

    if name == "qml":
        return import_module("pennylane")

    return import_module(name)


# -------------------------------- NumPy --------------------------------- #
from scipy.linalg import block_diag as _scipy_block_diag

ar.register_function("numpy", "flatten", lambda x: x.flatten())
ar.register_function("numpy", "coerce", lambda x: x)
ar.register_function("numpy", "block_diag", lambda x: _scipy_block_diag(*x))
ar.register_function("builtins", "block_diag", lambda x: _scipy_block_diag(*x))
ar.register_function("numpy", "gather", lambda x, indices: x[np.array(indices)])
ar.register_function("numpy", "unstack", list)


def _scatter_element_add_numpy(tensor, index, value):
    """In-place addition of a multidimensional value over various
    indices of a tensor."""
    tensor[tuple(index)] += value
    return tensor


ar.register_function("numpy", "scatter_element_add", _scatter_element_add_numpy)


# -------------------------------- Autograd --------------------------------- #


# When autoray inspects PennyLane NumPy tensors, they will be associated with
# the 'pennylane' module, and not autograd. Set an alias so it understands this is
# simply autograd.
ar.autoray._BACKEND_ALIASES["pennylane"] = "autograd"

# When dispatching to autograd, ensure that autoray will instead call
# qml.numpy rather than autograd.numpy, to take into account our autograd modification.
ar.autoray._MODULE_ALIASES["autograd"] = "pennylane.numpy"


ar.register_function("autograd", "flatten", lambda x: x.flatten())
ar.register_function("autograd", "coerce", lambda x: x)
ar.register_function("autograd", "block_diag", lambda x: _scipy_block_diag(*x))
ar.register_function("autograd", "gather", lambda x, indices: x[np.array(indices)])
ar.register_function("autograd", "unstack", list)


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


def _scatter_element_add_autograd(tensor, index, value):
    """In-place addition of a multidimensional value over various
    indices of a tensor. Since Autograd doesn't support indexing
    assignment, we have to be clever and use ravel_multi_index."""
    size = tensor.size
    flat_index = _i("qml").numpy.ravel_multi_index(index, tensor.shape)
    t = [0] * size
    t[flat_index] = value
    return tensor + _i("qml").numpy.array(t).reshape(tensor.shape)


ar.register_function("autograd", "scatter_element_add", _scatter_element_add_autograd)


def _take_autograd(tensor, indices, axis=None):
    indices = _i("qml").numpy.asarray(indices)

    if axis is None:
        return tensor.flatten()[indices]

    fancy_indices = [slice(None)] * axis + [indices]
    return tensor[tuple(fancy_indices)]


ar.register_function("autograd", "take", _take_autograd)


# -------------------------------- TensorFlow --------------------------------- #


ar.autoray._SUBMODULE_ALIASES["tensorflow", "angle"] = "tensorflow.math"
ar.autoray._SUBMODULE_ALIASES["tensorflow", "arcsin"] = "tensorflow.math"
ar.autoray._SUBMODULE_ALIASES["tensorflow", "arccos"] = "tensorflow.math"
ar.autoray._SUBMODULE_ALIASES["tensorflow", "arctan"] = "tensorflow.math"
ar.autoray._SUBMODULE_ALIASES["tensorflow", "arctan2"] = "tensorflow.math"
ar.autoray._SUBMODULE_ALIASES["tensorflow", "diag"] = "tensorflow.linalg"
ar.autoray._SUBMODULE_ALIASES["tensorflow", "kron"] = "tensorflow.experimental.numpy"


ar.autoray._FUNC_ALIASES["tensorflow", "arcsin"] = "asin"
ar.autoray._FUNC_ALIASES["tensorflow", "arccos"] = "acos"
ar.autoray._FUNC_ALIASES["tensorflow", "arctan"] = "atan"
ar.autoray._FUNC_ALIASES["tensorflow", "arctan2"] = "atan2"
ar.autoray._FUNC_ALIASES["tensorflow", "diag"] = "diag"


ar.register_function("tensorflow", "asarray", lambda x: _i("tf").convert_to_tensor(x))
ar.register_function("tensorflow", "flatten", lambda x: _i("tf").reshape(x, [-1]))
ar.register_function("tensorflow", "shape", lambda x: tuple(x.shape))
ar.register_function(
    "tensorflow",
    "sqrt",
    lambda x: _i("tf").math.sqrt(
        _i("tf").cast(x, "float64") if x.dtype.name in ("int64", "int32") else x
    ),
)


def _take_tf(tensor, indices, axis=None):
    """Implement a TensorFlow version of np.take"""
    tf = _i("tf")

    if isinstance(indices, numbers.Number):
        indices = [indices]

    indices = tf.convert_to_tensor(indices)

    if np.any(indices < 0):
        # Unlike NumPy, TensorFlow doesn't support negative indices.
        dim_length = tf.size(tensor).numpy() if axis is None else tf.shape(tensor)[axis]
        indices = tf.where(indices >= 0, indices, indices + dim_length)

    if axis is None:
        # Unlike NumPy, if axis=None TensorFlow defaults to the first
        # dimension rather than flattening the array.
        data = tf.reshape(tensor, [-1])
        return tf.gather(data, indices)

    return tf.gather(tensor, indices, axis=axis)


ar.register_function("tensorflow", "take", _take_tf)


def _coerce_types_tf(tensors):
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


def _block_diag_tf(tensors):
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


def _scatter_element_add_tf(tensor, index, value):
    """In-place addition of a multidimensional value over various
    indices of a tensor."""
    import tensorflow as tf

    indices = tf.expand_dims(index, 0)
    value = tf.cast(tf.expand_dims(value, 0), tensor.dtype)
    return tf.tensor_scatter_nd_add(tensor, indices, value)


ar.register_function("tensorflow", "scatter_element_add", _scatter_element_add_tf)


# -------------------------------- Torch --------------------------------- #

ar.autoray._FUNC_ALIASES["torch", "unstack"] = "unbind"

ar.register_function(
    "torch", "asarray", lambda x, device=None: _i("torch").as_tensor(x, device=device)
)
ar.register_function("torch", "diag", lambda x, k=0: _i("torch").diag(x, diagonal=k))
ar.register_function("torch", "expand_dims", lambda x, axis: _i("torch").unsqueeze(x, dim=axis))
ar.register_function("torch", "shape", lambda x: tuple(x.shape))
ar.register_function("torch", "gather", lambda x, indices: x[indices])

ar.register_function(
    "torch",
    "sqrt",
    lambda x: _i("torch").sqrt(
        x.to(_i("torch").float64) if x.dtype in (_i("torch").int64, _i("torch").int32) else x
    ),
)

ar.autoray._SUBMODULE_ALIASES["torch", "arctan2"] = "torch"
ar.autoray._FUNC_ALIASES["torch", "arctan2"] = "atan2"


def _take_torch(tensor, indices, axis=None):
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

    fancy_indices = [slice(None)] * axis + [indices]
    return tensor[fancy_indices]


ar.register_function("torch", "take", _take_torch)


def _coerce_types_torch(tensors):
    """Coerce a list of tensors to all have the same dtype
    without any loss of information."""
    torch = _i("torch")
    tensors = [torch.as_tensor(t) for t in tensors]
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
    res = torch.zeros(shape, dtype=tensors[0].dtype)

    # get the diagonal indices at which new block
    # diagonals need to be inserted
    p = np.cumsum(sizes, axis=0)

    # converted the diagonal indices to row and column indices
    ridx, cidx = np.stack([p - sizes, p]).T

    for t, r, c in zip(tensors, ridx, cidx):
        row = np.arange(*r).reshape(-1, 1)
        col = np.arange(*c).reshape(1, -1)
        res[row, col] = t

    return res


ar.register_function("torch", "block_diag", _block_diag_torch)


def _scatter_element_add_torch(tensor, index, value):
    """In-place addition of a multidimensional value over various
    indices of a tensor. Note that Torch only supports index assignments
    on non-leaf nodes; if the node is a leaf, we must clone it first."""
    if tensor.is_leaf:
        tensor = tensor.clone()
    tensor[tuple(index)] += value
    return tensor


ar.register_function("torch", "scatter_element_add", _scatter_element_add_torch)


# -------------------------------- JAX --------------------------------- #


def _to_numpy_jax(x):
    from jax.errors import TracerArrayConversionError

    try:
        return np.array(getattr(x, "val", x))
    except TracerArrayConversionError as e:
        raise ValueError(
            "Converting a JAX array to a NumPy array not supported when using the JAX JIT."
        ) from e


ar.register_function("jax", "flatten", lambda x: x.flatten())
ar.register_function(
    "jax",
    "take",
    lambda x, indices, axis=None: _i("jax").numpy.take(x, indices, axis=axis, mode="wrap"),
)
ar.register_function("jax", "coerce", lambda x: x)
ar.register_function("jax", "to_numpy", _to_numpy_jax)
ar.register_function("jax", "block_diag", lambda x: _i("jax").scipy.linalg.block_diag(*x))
ar.register_function("jax", "gather", lambda x, indices: x[np.array(indices)])
ar.register_function(
    "jax",
    "scatter_element_add",
    lambda x, index, value: _i("jax").ops.index_add(x, tuple(index), value),
)
ar.register_function("jax", "unstack", list)
