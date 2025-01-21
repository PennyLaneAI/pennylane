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
from functools import lru_cache
from string import ascii_lowercase as letters

has_jax = True
try:
    import jax
except ImportError:  # pragma: no cover
    has_jax = False  # pragma: no cover


@lru_cache
def _get_letter(ind: int) -> str:
    if ind < 26:
        return letters[ind]
    if ind < 702:
        return letters[ind // 26 - 1] + letters[ind % 26]
    raise NotImplementedError("we only support up to 702 dynamic axes")  # pragma: no cover


def _get_shape_for_array(x, abstract_shapes: list) -> dict:
    """
    Populate the dictionary of abstract axes for a single tensorlike.

    This dictionary has dimensions as keys, and a string marker as the value.

    Examples of shape -> abstract axes:

    * `(3,4) -> {}`
    * `(tracer1, ) -> {0: "a"}`
    * `(tracer1, tracer1) -> {0: "a", 1: "a"}`
    * `(3, tracer1) -> {1: "a"}`
    * `(tracer1, 2, tracer2) -> {0: "a", 2: "b"}`

    `abstract_shapes` contains all the tracers found in shapes.

    """
    abstract_axes = {}
    for i, s in enumerate(getattr(x, "shape", ())):
        if not isinstance(s, int):  #  if not int, then abstract
            found = False
            # check if the shape tracer is one we have already encountered
            for previous_idx, previous_shape in enumerate(abstract_shapes):
                if s is previous_shape:
                    abstract_axes[i] = _get_letter(previous_idx)
                    found = True
                    break
            # haven't encountered it, so add it to abstract_axes
            # and use new letter designation
            if not found:
                abstract_axes[i] = _get_letter(len(abstract_shapes))
                abstract_shapes.append(s)

    return abstract_axes


def determine_abstracted_axes(args):
    """Computed the abstracted axes and extracting the abstract shapes from the arguments.

    Args:
        args (tuple): the arguments for a higher order primitive

    Returns:
        tuple, tuple: the corresponding abstracted axes and dynamic shapes

    Note that "dynamic shapes" only refers to the size of dimensions, but not the number of dimensions.
    Even with dynamic shapes mode enabled, we cannot change the number of dimensions.

    See the ``intro_to_dynamic_shapes.md`` document for more information on how dynamic shapes work.

    To make jaxpr from arguments with dynamic shapes, the ``abstracted_axes`` keyword argument must be set.
    Then, when calling the jaxpr, variables for the dynamic shapes must be passed.

    .. code-block:: python

        jax.config.update("jax_dynamic_shapes", True)

        def f(n):
            x = jax.numpy.ones((n,))
            abstracted_axes, abstract_shapes = qml.capture.determine_abstracted_axes((x,))
            jaxpr = jax.make_jaxpr(jax.numpy.sum, abstracted_axes=abstracted_axes)(x)
            return jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, *abstract_shapes, x)

    """
    if not has_jax:  # pragma: no cover
        raise ImportError("jax must be installed to use determine_abstracted_axes")
    if not jax.config.jax_dynamic_shapes:  # pylint: disable=no-member
        return None, tuple()

    args, structure = jax.tree_util.tree_flatten(args)

    abstract_shapes = []
    # note: this function in-place mutates abstract_shapes
    # adding any additional abstract shapes found
    abstracted_axes = [_get_shape_for_array(a, abstract_shapes) for a in args]

    if not abstract_shapes:
        return None, ()
    abstracted_axes = jax.tree_util.tree_unflatten(structure, abstracted_axes)
    return abstracted_axes, abstract_shapes
