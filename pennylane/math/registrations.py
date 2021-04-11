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
# pylint:disable=protected-access
import numbers

import autoray as ar
from autoray import numpy as np
import numpy as _np


# -------------------------------- NumPy --------------------------------- #
from scipy.linalg import block_diag as _scipy_block_diag

ar.register_function("numpy", "flatten", lambda x: x.flatten())
ar.register_function("numpy", "coerce", lambda x: x)
ar.register_function("numpy", "block_diag", lambda x: _scipy_block_diag(*x))
ar.register_function("builtins", "block_diag", lambda x: _scipy_block_diag(*x))
ar.register_function("numpy", "gather", lambda x, indices: x[_np.array(indices)])


def _scatter_element_add_numpy(tensor, index, value):
    tensor[tuple(index)] += value
    return tensor


ar.register_function("numpy", "scatter_element_add", _scatter_element_add_numpy)


# -------------------------------- Autograd --------------------------------- #


ar.autoray._MODULE_ALIASES["autograd"] = "pennylane.numpy"
ar.register_function("autograd", "flatten", lambda x: x.flatten())
ar.register_function("autograd", "coerce", lambda x: x)
ar.register_function("autograd", "block_diag", lambda x: _scipy_block_diag(*x))
ar.register_function("autograd", "gather", lambda x, indices: x[_np.array(indices)])


def _to_numpy_autograd(x):
    if hasattr(x, "_value"):
        # Catches the edge case where the data is an Autograd arraybox,
        # which only occurs during backpropagation.
        return x._value

    return x.numpy()


ar.register_function("autograd", "to_numpy", _to_numpy_autograd)


def _scatter_element_add_autograd(tensor, index, value):
    size = tensor.size
    flat_index = __import__("pennylane").numpy.ravel_multi_index(index, tensor.shape)
    t = [0] * size
    t[flat_index] = value
    return tensor + __import__("pennylane").numpy.array(t).reshape(tensor.shape)


ar.register_function("autograd", "scatter_element_add", _scatter_element_add_autograd)


# -------------------------------- TensorFlow --------------------------------- #


ar.autoray._SUBMODULE_ALIASES["tensorflow", "angle"] = "tensorflow.math"
ar.autoray._SUBMODULE_ALIASES["tensorflow", "arcsin"] = "tensorflow.math"
ar.autoray._SUBMODULE_ALIASES["tensorflow", "arccos"] = "tensorflow.math"
ar.autoray._SUBMODULE_ALIASES["tensorflow", "arctan"] = "tensorflow.math"
ar.autoray._SUBMODULE_ALIASES["tensorflow", "diag"] = "tensorflow.linalg"


ar.autoray._FUNC_ALIASES["tensorflow", "arcsin"] = "asin"
ar.autoray._FUNC_ALIASES["tensorflow", "diag"] = "diag"


ar.register_function(
    "tensorflow", "asarray", lambda x: __import__("tensorflow").convert_to_tensor(x)
)
ar.register_function("tensorflow", "flatten", lambda x: np.reshape(x, [-1]))
ar.register_function("tensorflow", "flatten", lambda x: np.reshape(x, [-1]))
ar.register_function("tensorflow", "shape", lambda x: tuple(x.shape))
ar.register_function(
    "tensorflow",
    "sqrt",
    lambda x: __import__("tensorflow").math.sqrt(
        __import__("tensorflow").cast(x, "float64") if x.dtype.name in ("int64", "int32") else x
    ),
)


def _take_tf(tensor, indices, axis=None):
    tf = __import__("tensorflow")

    if isinstance(indices, numbers.Number):
        indices = [indices]

    indices = tf.convert_to_tensor(indices)

    if _np.any(indices < 0):
        # Unlike NumPy, TensorFlow doesn't support negative indices.
        dim_length = tf.size(tensor).numpy() if axis is None else np.shape(tensor)[axis]
        indices = tf.where(indices >= 0, indices, indices + dim_length)

    if axis is None:
        # Unlike NumPy, if axis=None TensorFlow defaults to the first
        # dimension rather than flattening the array.
        data = tf.reshape(tensor, [-1])
        return tf.gather(data, indices)

    return tf.gather(tensor, indices, axis=axis)


ar.register_function("tensorflow", "take", _take_tf)


def _coerce_types_tf(tensors):
    tf = __import__("tensorflow")
    tensors = [tf.convert_to_tensor(t) for t in tensors]
    dtypes = {i.dtype for i in tensors}

    if len(dtypes) == 1:
        return tensors

    complex_type = dtypes.intersection({tf.complex64, tf.complex128})
    float_type = dtypes.intersection({tf.float16, tf.float32, tf.float64})
    int_type = dtypes.intersection({tf.int8, tf.int16, tf.int32, tf.int64})

    cast_type = complex_type or float_type or int_type
    cast_type = list(cast_type)[-1]

    return [tf.cast(t, cast_type) for t in tensors]


ar.register_function("tensorflow", "coerce", _coerce_types_tf)


def _block_diag_tf(tensors):
    tf = __import__("tensorflow")
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
    import tensorflow as tf

    indices = tf.expand_dims(index, 0)
    value = tf.cast(tf.expand_dims(value, 0), tensor.dtype)
    return tf.tensor_scatter_nd_add(tensor, indices, value)


ar.register_function("tensorflow", "scatter_element_add", _scatter_element_add_tf)


# -------------------------------- Torch --------------------------------- #


ar.register_function("torch", "asarray", lambda x: __import__("torch").as_tensor(x))
ar.register_function("torch", "diag", lambda x, k=0: __import__("torch").diag(x, diagonal=k))
ar.register_function("torch", "expand_dims", lambda x, axis: np.unsqueeze(x, dim=axis))
ar.register_function("torch", "shape", lambda x: tuple(x.shape))
ar.register_function("torch", "gather", lambda x, indices: x[indices])


def _take_torch(tensor, indices, axis=None):
    torch = __import__("torch")

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
    torch = __import__("torch")
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
    torch = __import__("torch")
    sizes = _np.array([t.shape for t in tensors])
    res = torch.zeros(_np.sum(sizes, axis=0).tolist(), dtype=tensors[0].dtype)

    p = np.cumsum(sizes, axis=0)
    ridx, cidx = _np.stack([p - sizes, p]).T

    for t, r, c in zip(tensors, ridx, cidx):
        row = _np.arange(*r).reshape(-1, 1)
        col = _np.arange(*c).reshape(1, -1)
        res[row, col] = t

    return res


ar.register_function("torch", "block_diag", _block_diag_torch)


def _scatter_element_add_torch(tensor, index, value):
    if tensor.is_leaf:
        tensor = tensor.clone()
    tensor[tuple(index)] += value
    return tensor


ar.register_function("torch", "scatter_element_add", _scatter_element_add_torch)


# -------------------------------- JAX --------------------------------- #


ar.register_function("jax", "flatten", lambda x: x.flatten())
ar.register_function(
    "jax",
    "take",
    lambda x, indices, axis=None: __import__("jax").numpy.take(x, indices, axis=axis, mode="wrap"),
)
ar.register_function("jax", "coerce", lambda x: x)
ar.register_function("jax", "to_numpy", lambda x: x)
ar.register_function("jax", "block_diag", lambda x: __import__("jax").scipy.linalg.block_diag(*x))
ar.register_function("jax", "gather", lambda x, indices: x[_np.array(indices)])
ar.register_function(
    "jax",
    "scatter_element_add",
    lambda x, index, value: __import__("jax").ops.index_add(x, tuple(index), value),
)
