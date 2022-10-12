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

from .multi_dispatch import (
    multi_dispatch,
    array,
    block_diag,
    concatenate,
    diag,
    dot,
    einsum,
    eye,
    frobenius_inner_product,
    get_trainable_indices,
    ones_like,
    scatter,
    scatter_element_add,
    stack,
    tensordot,
    unwrap,
    where,
    add,
    iscomplex,
    expm,
)

from .quantum import cov_matrix, marginal_prob
from .quantum import reduced_dm, vn_entropy, mutual_info, sqrt_matrix, fidelity, relative_entropy

from .utils import (
    allclose,
    allequal,
    cast,
    cast_like,
    in_backprop,
    is_abstract,
    convert_like,
    get_interface,
    requires_grad,
)

from .is_independent import is_independent

from .matrix_manipulation import expand_matrix, reduce_matrices

sum = ar.numpy.sum
toarray = ar.numpy.to_numpy
T = ar.numpy.transpose


# small constant for numerical stability that the user can modify
eps = 1e-14


def __getattr__(name):
    return getattr(ar.numpy, name)


__all__ = [
    "multi_dispatch",
    "allclose",
    "allequal",
    "array",
    "block_diag",
    "cast",
    "cast_like",
    "concatenate",
    "convert_like",
    "cov_matrix",
    "diag",
    "dot",
    "einsum",
    "eye",
    "fidelity",
    "frobenius_inner_product",
    "get_interface",
    "get_trainable_indices",
    "in_backprop",
    "is_abstract",
    "is_independent",
    "marginal_prob",
    "mutual_info",
    "ones_like",
    "reduced_dm",
    "relative_entropy",
    "requires_grad",
    "sqrt_matrix",
    "scatter_element_add",
    "stack",
    "tensordot",
    "unwrap",
    "vn_entropy",
    "where",
    "add",
    "iscomplex",
    "expand_matrix",
]
