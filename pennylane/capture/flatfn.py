# Copyright 2024 Xanadu Quantum Technologies Inc.

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
Defines a utility for capturing higher order primitives that return pytrees.
"""
from functools import update_wrapper

has_jax = True
try:
    import jax
except ImportError:
    has_jax = False


# pylint: disable=too-few-public-methods
class FlatFn:
    """Wrap a function so that is accepts flat arguments and returns flat outputs. Caches
    the pytree shape of the output into the ``out_tree`` property, so that the results can
    be repacked later.

    >>> def f(x):
    ...     return {"y": 2+x["x"]}
    >>> args = ({"x": 0},)
    >>> flat_args, in_tree = jax.tree_util.tree_flatten(args)
    >>> flat_f = FlatFn(f, in_tree)
    >>> res = flat_f(*flat_args)
    >>> res
    [2]
    >>> jax.tree_util.tree_unflatten(flat_f.out_tree, res)
    {'y': 2.5}

    """

    def __init__(self, f, in_tree):
        self.f = f
        self.in_tree = in_tree
        self.out_tree = None
        update_wrapper(self, f)

    def __call__(self, *flat_args):
        args = jax.tree_util.tree_unflatten(self.in_tree, flat_args)
        out = self.f(*args)
        out_flat, out_tree = jax.tree_util.tree_flatten(out)
        self.out_tree = out_tree
        return out_flat
