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
    _multi_dispatch,
    multi_dispatch,
    block_diag,
    concatenate,
    diag,
    dot,
    frobenius_inner_product,
    get_trainable_indices,
    ones_like,
    safe_squeeze,
    scatter_element_add,
    stack,
    tensordot,
    unwrap,
    where,
)

from .quantum import cov_matrix, marginal_prob

from .utils import (
    allclose,
    allequal,
    cast,
    cast_like,
    is_abstract,
    convert_like,
    get_interface,
    requires_grad,
)

from .is_independent import is_independent

sum = ar.numpy.sum
toarray = ar.numpy.to_numpy
T = ar.numpy.transpose


def __getattr__(name):
    return getattr(ar.numpy, name)


__all__ = [
    "_multi_dispatch",
    "multi_dispatch",
    "allclose",
    "allequal",
    "block_diag",
    "cast",
    "cast_like",
    "concatenate",
    "convert_like",
    "cov_matrix",
    "diag",
    "dot",
    "frobenius_inner_product",
    "get_interface",
    "get_trainable_indices",
    "is_abstract",
    "is_independent",
    "marginal_prob",
    "ones_like",
    "requires_grad",
    "safe_squeeze",
    "scatter_element_add",
    "stack",
    "tensordot",
    "unwrap",
    "where",
]
