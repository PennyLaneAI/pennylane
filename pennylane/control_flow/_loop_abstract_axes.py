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
Contains utilities for handling abstracted axes for for_loop and while_loop.

Note that these are located in the ``control_flow`` module and not the ``capture`` module
as they are specific to just ``for_loop`` and ``while_loop``.
``capture.determine_abstacted_axes`` applies to any higher order primitive.

"""

from collections import namedtuple
from typing import Any

from pennylane.typing import TensorLike

AbstractShapeLocation = namedtuple("AbstractShapeLocation", ("arg_idx", "shape_idx"))


# pylint: disable=too-few-public-methods
class CalculateAbstractedAxes:
    """A helper class for accumulating information about abstract axes for loop functions."""

    def __init__(self, allow_array_resizing: bool = False):
        self.allow_array_resizing = allow_array_resizing
        self.abstract_shapes = []
        self.abstracted_axes = []
        self.shape_locations = []

    def add_arg(self, x_idx: int, x):
        """Process a new argument"""
        arg_abstracted_axes = {}

        for shape_idx, s in enumerate(getattr(x, "shape", ())):
            if not isinstance(s, int):  #  if not int, then abstract
                found = False
                if not self.allow_array_resizing:
                    for previous_idx, previous_shape in enumerate(self.abstract_shapes):
                        if s is previous_shape:
                            arg_abstracted_axes[shape_idx] = previous_idx
                            self.shape_locations[previous_idx].append(
                                AbstractShapeLocation(x_idx, shape_idx)
                            )
                            found = True
                            break
                # haven't encountered it, so add it to abstract_axes
                # and use new letter designation
                if not found:
                    arg_abstracted_axes[shape_idx] = len(self.abstract_shapes)
                    self.shape_locations.append([AbstractShapeLocation(x_idx, shape_idx)])
                    self.abstract_shapes.append(s)
        self.abstracted_axes.append(arg_abstracted_axes)


# pylint: disable=import-outside-toplevel
def loop_determine_abstracted_axes(
    args, allow_array_resizing: bool = False
) -> tuple[Any, list[TensorLike], list[list[AbstractShapeLocation]]]:
    """Determine the abstract axes for arguments that will be used in a loop context.

    Args:
        args (Any): Arguments to determine the abstracted axes for
        allow_array_resizing=False (bool): If True, each abstracted axis should be treated as
            an independent axis

    Returns:
        abstracted_axes, abstract_shapes, locations for shapes

    .. code-block:: python

        from pennylane.capture import loop_determine_abstracted_axes
        from functools import partial

        def f(*args, allow_array_resizing):
            abstracted_axes, abstract_shapes, locations = loop_determine_abstracted_axes(args, allow_array_resizing)
            print(abstracted_axes)
            print(abstract_shapes)
            print(locations)

        args = (0, jnp.ones(3), jnp.zeros((3, 3)))
        jax.make_jaxpr(partial(f, allow_array_resizing=False), abstracted_axes=({}, {0:"a"}, {1:"a"}))(*args)

    .. code-block::

        ({}, {0: 'a'}, {1: 'a'})
        [Traced<ShapedArray(int32[], weak_type=True)>with<DynamicJaxprTrace(level=1/0)>]
        [[AbstractShapeLocation(arg_idx=1, shape_idx=0), AbstractShapeLocation(arg_idx=2, shape_idx=1)]]

    Here we can that the abstracted axes out match what we put in. The returned ``abstract_shapes`` is the single
    abstract shape that occurs in both variables.  The locations array tells us that we can locate the first
    abstract shape in the ``1`` argument at shape position ``0``, and in the ``2`` argument at shape position ``1``.

    If we instead specify ``allow_array_resizing=True``, we can see the difference.

    ... code-block:: python

        jax.make_jaxpr(partial(f, allow_array_resizing=True), abstracted_axes=({}, {0:"a"}, {1:"a"}))(*args)

    .. code-block::

        ({}, {0: 'a'}, {1: 'b'})
        [Traced<ShapedArray(int32[], weak_type=True)>with<DynamicJaxprTrace(level=1/0)>, Traced<ShapedArray(int32[], weak_type=True)>with<DynamicJaxprTrace(level=1/0)>]
        [[AbstractShapeLocation(arg_idx=1, shape_idx=0)], [AbstractShapeLocation(arg_idx=2, shape_idx=1)]]

    Now the abstracted axes treat the two abstracted axes as different, even though they are the same tracer in the input
    arguments. The abstract shapes has two elements. By looking at the locations, we can see that we can find
    the first abstract shape in argument ``1`` at shape ``0``, and we can find the second abstract shape in
    argument ``2`` at shape ``1``.


    """
    import jax

    args, structure = jax.tree_util.tree_flatten(args)
    calculator = CalculateAbstractedAxes(allow_array_resizing=allow_array_resizing)
    _ = [calculator.add_arg(x_idx, x) for x_idx, x in enumerate(args)]

    if not any(calculator.abstracted_axes):
        return None, [], []

    abstracted_axes = jax.tree_util.tree_unflatten(structure, calculator.abstracted_axes)
    return abstracted_axes, calculator.abstract_shapes, calculator.shape_locations
