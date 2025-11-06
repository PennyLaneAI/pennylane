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
"""
This package contains unified functions for framework-agnostic tensor and array
manipulation. Given the input tensor-like object, the call is dispatched
to the corresponding array manipulation framework, allowing for end-to-end
differentiation to be preserved.

.. warning::

    These functions are experimental, and only a subset of common functionality is supported.
    Furthermore, the names and behaviour of these functions may differ from similar
    functions in common frameworks; please refer to the function docstrings for more details.

The following frameworks are currently supported:

* NumPy
* Autograd
* TensorFlow
* PyTorch
* JAX

"""
import autoray as ar

from .is_independent import is_independent
from .matrix_manipulation import (
    expand_matrix,
    expand_vector,
    reduce_matrices,
    get_batch_size,
    convert_to_su2,
    convert_to_su4,
)
from .multi_dispatch import (
    add,
    array,
    block_diag,
    concatenate,
    detach,
    diag,
    dot,
    einsum,
    expm,
    eye,
    frobenius_inner_product,
    gammainc,
    get_trainable_indices,
    iscomplex,
    kron,
    matmul,
    multi_dispatch,
    norm,
    svd,
    ones_like,
    scatter,
    scatter_element_add,
    set_index,
    stack,
    tensordot,
    unwrap,
    where,
)
from .quantum import (
    cov_matrix,
    dm_from_state_vector,
    expectation_value,
    marginal_prob,
    mutual_info,
    partial_trace,
    purity,
    reduce_dm,
    reduce_statevector,
    relative_entropy,
    sqrt_matrix,
    sqrt_matrix_sparse,
    vn_entropy,
    vn_entanglement_entropy,
    max_entropy,
    min_entropy,
    trace_distance,
    choi_matrix,
)
from .fidelity import fidelity, fidelity_statevector
from .utils import (
    allclose,
    allequal,
    cast,
    cast_like,
    convert_like,
    in_backprop,
    requires_grad,
    is_abstract,
    binary_finite_reduced_row_echelon,
)
from .interface_utils import (
    SUPPORTED_INTERFACE_NAMES,
    get_deep_interface,
    get_interface,
    Interface,
)
from .grad import grad, jacobian
from . import decomposition

sum = ar.numpy.sum
conj = ar.numpy.conj
transpose = ar.numpy.transpose
sqrt = ar.numpy.sqrt
zeros = ar.numpy.zeros
moveaxis = ar.numpy.moveaxis
mean = ar.numpy.mean
round = ar.numpy.round
shape = ar.numpy.shape
flatten = ar.numpy.flatten
reshape = ar.numpy.reshape
multiply = ar.numpy.multiply
toarray = ar.numpy.to_numpy
T = ar.numpy.transpose


def get_dtype_name(x) -> str:
    """An interface independent way of getting the name of the datatype.

    >>> x = tf.Variable(0.1)
    >>> qml.math.get_dtype_name(tf.Variable(0.1))
    'float32'
    """
    return ar.get_dtype_name(x)


def is_real_obj_or_close(obj):
    """Convert an array to its real part if it is close to being real-valued, and afterwards
    return whether the resulting data type is real.

    Args:
        obj (array): Array to check for being (close to) real.

    Returns:
        bool: Whether the array ``obj``, after potentially converting it to a real matrix,
        has a real data type. This is obtained by checking whether the data type name starts with
        ``"complex"`` and returning the negated result of this.

    >>> x = jnp.array(0.4)
    >>> qml.math.is_real_obj_or_close(x)
    True

    >>> x = tf.Variable(0.4+0.2j)
    >>> qml.math.is_real_obj_or_close(x)
    False

    >>> x = torch.tensor(0.4+1e-13j)
    >>> qml.math.is_real_obj_or_close(x)
    True

    Default absolute and relative tolerances of
    ``qml.math.allclose`` are used to determine whether the
    input is close to real-valued.
    """
    if not is_abstract(obj) and allclose(ar.imag(obj), 0.0):
        obj = ar.real(obj)
    return not get_dtype_name(obj).startswith("complex")


class NumpyMimic(ar.autoray.AutoNamespace):
    """Subclass of the Autoray NumpyMimic class in order to support
    the NumPy fft submodule"""

    # pylint: disable=too-few-public-methods

    def __getattribute__(self, fn):
        if fn == "fft":
            return numpy_fft
        return super().__getattribute__(fn)


numpy_mimic = NumpyMimic()
numpy_fft = ar.autoray.AutoNamespace(submodule="fft")

# small constant for numerical stability that the user can modify
eps = 1e-14


def __getattr__(name):
    return getattr(numpy_mimic, name)


__all__ = [
    "add",
    "allclose",
    "allequal",
    "array",
    "block_diag",
    "cast",
    "cast_like",
    "concatenate",
    "convert_like",
    "convert_to_su2",
    "convert_to_su4",
    "cov_matrix",
    "detach",
    "diag",
    "dm_from_state_vector",
    "dot",
    "einsum",
    "expand_matrix",
    "expand_vector",
    "expectation_value",
    "expm",
    "eye",
    "fidelity",
    "fidelity_statevector",
    "frobenius_inner_product",
    "gammainc",
    "get_dtype_name",
    "get_interface",
    "get_batch_size",
    "get_deep_interface",
    "get_trainable_indices",
    "grad",
    "in_backprop",
    "is_abstract",
    "is_independent",
    "is_real_obj_or_close",
    "iscomplex",
    "jacobian",
    "kron",
    "Interface",
    "matmul",
    "marginal_prob",
    "max_entropy",
    "min_entropy",
    "multi_dispatch",
    "mutual_info",
    "norm",
    "ones_like",
    "partial_trace",
    "purity",
    "reduce_dm",
    "reduce_matrices",
    "reduce_statevector",
    "binary_finite_reduced_row_echelon",
    "relative_entropy",
    "requires_grad",
    "scatter",
    "scatter_element_add",
    "set_index",
    "sqrt_matrix",
    "stack",
    "svd",
    "tensordot",
    "trace_distance",
    "unwrap",
    "vn_entropy",
    "vn_entanglement_entropy",
    "where",
    "choi_matrix",
]
