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

"""

has_jax = True
try:
    import jax.tree_util as jax_tree_util
except ImportError:
    has_jax = False

has_torch = True
try:
    from torch.utils._pytree import _register_pytree_node as torch_register_pytree
except ImportError:
    has_torch = False


def _register_pytree_with_jax(pytree_type, flatten_fn, unflatten_fn):
    def jax_unflatten(aux, parameters):
        return unflatten_fn(parameters, aux)

    jax_tree_util.register_pytree_node(pytree_type, flatten_fn, jax_unflatten)
    return


def _register_pytree_with_torch(pytree_type, flatten_fn, unflatten_fn):

    torch_register_pytree(pytree_type, flatten_fn, unflatten_fn)
    return


def register_pytree(pytree_type, flatten_fn, unflatten_fn):

    if has_jax:
        _register_pytree_with_jax(pytree_type, flatten_fn, unflatten_fn)

    if has_torch:
        _register_pytree_with_torch(pytree_type, flatten_fn, unflatten_fn)

    return
