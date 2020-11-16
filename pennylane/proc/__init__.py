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
"""
This package contains unified functions for framework-agnostic tensor and array
manipulation. Given the input tensor-like object, the call is dispatched
to the corresponding array manipulation framework, allowing for end-to-end
differentiation to be preserved.

.. warning::

    These functions are experimental, and only a subset of common functionality is supported.
    Furthermore, the names and behaviour of these functions may differ from similar
    functions in common frameworks; please refer to the function docstrings for more details.

Internally, these functions dispatch by using the :class:`TensorBox` class, a container and API for
array-like objects that allows array manipulation to be performed in a unified manner for supported
tensor/array manipulation frameworks.

The following frameworks are currently supported:

* NumPy
* Autograd
* TensorFlow
* PyTorch
"""
from .fn import (
    T,
    allclose,
    allequal,
    cast,
    cast_like,
    convert_like,
    expand_dims,
    get_interface,
    ones_like,
    requires_grad,
    shape,
    stack,
    toarray,
)
from .tensorbox import TensorBox
