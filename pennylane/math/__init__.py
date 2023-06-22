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
from .matrix_manipulation import expand_matrix, reduce_matrices
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
    jax_argnums_to_tape_trainable,
    kron,
    matmul,
    multi_dispatch,
    norm,
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
    fidelity,
    marginal_prob,
    mutual_info,
    purity,
    reduced_dm,
    reduce_dm,
    reduce_statevector,
    relative_entropy,
    sqrt_matrix,
    vn_entropy,
    max_entropy,
    trace_distance,
)
from .utils import (
    allclose,
    allequal,
    cast,
    cast_like,
    convert_like,
    get_interface,
    in_backprop,
    is_abstract,
    requires_grad,
)

sum = ar.numpy.sum
toarray = ar.numpy.to_numpy
T = ar.numpy.transpose


class NumpyMimic(ar.autoray.NumpyMimic):
    """Subclass of the Autoray NumpyMimic class in order to support
    the NumPy fft submodule"""

    # pylint: disable=too-few-public-methods

    def __getattribute__(self, fn):
        if fn == "fft":
            return numpy_fft
        return super().__getattribute__(fn)


numpy_mimic = NumpyMimic()
numpy_fft = ar.autoray.NumpyMimic("fft")

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
    "cov_matrix",
    "detach",
    "diag",
    "dm_from_state_vector",
    "dot",
    "einsum",
    "expand_matrix",
    "eye",
    "fidelity",
    "frobenius_inner_product",
    "get_interface",
    "get_trainable_indices",
    "in_backprop",
    "is_abstract",
    "is_independent",
    "iscomplex",
    "marginal_prob",
    "max_entropy",
    "multi_dispatch",
    "mutual_info",
    "ones_like",
    "purity",
    "reduce_dm",
    "reduced_dm",
    "relative_entropy",
    "reduce_statevector",
    "requires_grad",
    "sqrt_matrix",
    "scatter_element_add",
    "stack",
    "tensordot",
    "trace_distance",
    "unwrap",
    "vn_entropy",
    "where",
]
