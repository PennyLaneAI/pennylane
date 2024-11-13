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
    """Wrap a function so that it caches the pytree shape of the output into the ``out_tree``
    property, so that the results can be repacked later. It also returns flattened results
    instead of the original result object.

    If an ``in_tree`` is provided, the function accepts flattened inputs instead of the
    original inputs with tree structure given by ``in_tree``.

    **Example**

    >>> import jax
    >>> from pennylane.capture.flatfn import FlatFn
    >>> def f(x):
    ...     return {"y": 2+x["x"]}
    >>> flat_f = FlatFn(f)
    >>> arg = {"x": 0.5}
    >>> res = flat_f(arg)
    >>> res
    [2.5]
    >>> jax.tree_util.tree_unflatten(flat_f.out_tree, res)
    {'y': 2.5}

    If we want to use a fully flattened function that also takes flat inputs instead of
    the original inputs with tree structure, we can provide the ``PyTreeDef`` for this input
    structure:

    >>> flat_args, in_tree = jax.tree_util.tree_flatten((arg,))
    >>> flat_f = FlatFn(f, in_tree)
    >>> res = flat_f(*flat_args)
    >>> res
    [2.5]
    >>> jax.tree_util.tree_unflatten(flat_f.out_tree, res)
    {'y': 2.5}

    Note that the ``in_tree`` has to be  created by flattening a tuple of all input
    arguments, even if there is only a single argument.
    """

    def __init__(self, f, in_tree=None):
        self.f = f
        self.in_tree = in_tree
        self.out_tree = None
        update_wrapper(self, f)

    def __call__(self, *args, **kwargs):
        if self.in_tree is not None:
            args = jax.tree_util.tree_unflatten(self.in_tree, args)
        out = self.f(*args, **kwargs)
        out_flat, out_tree = jax.tree_util.tree_flatten(out)
        self.out_tree = out_tree
        return out_flat
