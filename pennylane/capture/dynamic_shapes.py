# Copyright 2025 Xanadu Quantum Technologies Inc.

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
Contains a utility for handling inputs with dynamically shaped arrays.
"""
from string import ascii_lowercase

has_jax = True
try:
    import jax
except ImportError:
    has_jax = False


def determine_abstracted_axes(args):
    """Computed the abstracted axes and extracing the abstract shapes from the arguments.

    Args:
        args (tuple): the arguments for a higher order primitive

    Returns:
        tuple, tuple: the corresponding abstracted axes and dynamic shapes

    See the ``intro_to_dynamic_shapes.md`` document for more information on how dynamic shapes work.

    To make jaxpr from arguments with dynamic shapes, the ``abstracted_axes`` keyword argument must be set.
    Then, when calling the jaxpr, variables for the dynamic shapes must be passed.

    ```
    def f(n):
        x = jax.numpy.ones((n,))
        abstracted_axes, abstract_shapes = qml.capture.determine_abstracted_axes((x,))
        jaxpr = jax.make_jaxpr(jax.numpy.sum, abstracted_axes=abstracted_axes)(x)
        return jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, *abstract_shapes, x)
    ```

    """
    if not has_jax:
        raise ImportError("jax must be installed to use determine_abstracted_axes")
    if not jax.config.jax_dynamic_shapes:
        return None, tuple()

    args, structure = jax.tree_util.tree_flatten(args)
    abstracted_axes = []
    abstract_shapes = []
    for l in args:
        l_shape = []
        for s in getattr(l, "shape", ()):
            if isinstance(s, int):  # not abstract
                l_shape.append(())
            else:
                l_shape.append(ascii_lowercase[len(abstract_shapes)])
                if all(s is not x for x in abstract_shapes):
                    # not already added
                    abstract_shapes.append(s)
        abstracted_axes.append(tuple(l_shape))

    if not abstract_shapes:
        return None, ()
    abstracted_axes = jax.tree_util.tree_unflatten(structure, abstracted_axes)
    return abstracted_axes, abstract_shapes
