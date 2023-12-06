# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

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
An internal module for working with pytrees.
"""

from typing import Callable, Tuple, Any

has_jax = True
try:
    import jax.tree_util as jax_tree_util
except ImportError:
    has_jax = False

Leaves = Any
Metadata = Any

FlattenFn = Callable[[Any], Tuple[Leaves, Metadata]]
UnflattenFn = Callable[[Leaves, Metadata], Any]


def _register_pytree_with_jax(pytree_type: type, flatten_fn: FlattenFn, unflatten_fn: UnflattenFn):
    def jax_unflatten(aux, parameters):
        return unflatten_fn(parameters, aux)

    jax_tree_util.register_pytree_node(pytree_type, flatten_fn, jax_unflatten)


def register_pytree(pytree_type: type, flatten_fn: FlattenFn, unflatten_fn: UnflattenFn):
    """Register a type with all available pytree backends.

    Current backends is jax.
    Args:
        pytree_type (type): the type to register, such as ``qml.RX``
        flatten_fn (Callable): a function that splits an object into trainable leaves and hashable metadata.
        unflatten_fn (Callable): a function that reconstructs an object from its leaves and metadata.

    Returns:
        None

    Side Effects:
        ``pytree`` type becomes registered with available backends.

    """

    if has_jax:
        _register_pytree_with_jax(pytree_type, flatten_fn, unflatten_fn)
