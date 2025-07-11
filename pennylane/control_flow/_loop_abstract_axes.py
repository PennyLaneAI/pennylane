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
``capture.determine_abstracted_axes`` applies to any higher order primitive.

"""

from collections import namedtuple
from collections.abc import Callable
from typing import Any

from pennylane.typing import TensorLike

AbstractShapeLocation = namedtuple("AbstractShapeLocation", ("arg_idx", "shape_idx"))


def add_abstract_shapes(f, shape_locations: list[list[AbstractShapeLocation]]):
    """Add the abstract shapes at the specified locations to the output of f.

    Here we can see that the shapes at argument 0, shape index 0 and
    argument 1, shape index 1 are returned alongside the results of ``f``.

    .. code-block:: python
        import jax.numpy as jnp
        def f(x, y): return [x, y]

        loc1 = AbstractShapeLocation(arg_idx=0, shape_idx=0)
        loc2 = AbstractShapeLocation(arg_idx=1, shape_idx=1)
        repeat_loc = AbstractShapeLocation(arg_idx=0, shape_idx=1)
        locations = [[loc1, repeat_loc], [loc2]]

        add_abstract_shapes(f, locations)(jnp.zeros((1,1)), jnp.zeros((3,4)))

    .. code-block::

        [1,
        4,
        Array([[0.]], dtype=float32),
        Array([[0., 0., 0., 0.],
                [0., 0., 0., 0.],
                [0., 0., 0., 0.]], dtype=float32)]

    """

    def new_f(*args, **kwargs):
        results = f(*args, **kwargs)
        new_shapes = [results[loc[0].arg_idx].shape[loc[0].shape_idx] for loc in shape_locations]
        return new_shapes + results

    return new_f


def get_dummy_arg(arg):
    """If any axes are abstract, replace them with an empty numpy array.

    Even if abstracted_axes specifies two dimensions as having different dynamic shapes,
    if the dimension is the same tracer, jax will still treat them as the same shape.

    .. code-block:: python

        def f(a, b): return 0

        def w(i0):
            a = jnp.arange(i0)
            b = jnp.arange(i0)
            jaxpr = jax.make_jaxpr(f, abstracted_axes=({0:0}, {0:1}))(a, b)
            print(jaxpr)

        _ = jax.make_jaxpr(w)(2)

    .. code-block::

        { lambda ; a:i32[] b:i32[a] c:i32[a]. let  in (0,) }

    So we need to override this behavior by not giving them any abstract shapes to focus on.
    Instead, we just pass in an empty numpy array with all abstract dimensions replaced with ``2``.
    We use numpy instead of jax so the creation of the array will not show up in the jaxpr.

    """
    if all(isinstance(s, int) for s in arg.shape):
        return arg
    # add small, non-trivial size 2 as a concrete stand-in for dynamic axes
    shape = tuple(s if isinstance(s, int) else 2 for s in arg.shape)
    from jax.numpy import empty  # pylint: disable=import-outside-toplevel

    return empty(shape=shape, dtype=arg.dtype)


def validate_no_resizing_returns(
    jaxpr: "jax.extend.core.Jaxpr",
    locations: list[list[AbstractShapeLocation]],
    name: str = "while_loop",
) -> str | None:
    """Validate that all jaxpr outputs that should have the same shape as specified in ``locations``
    continue to have the same shape.  Returns a string with an error message so we can
    either decide to raise the error, or try again with different settings.
    """
    offset = len(locations)  # number of abstract shapes. We start from the first normal arg.

    for locations_list in locations:
        loc0 = locations_list[0]
        first_var = jaxpr.outvars[loc0.arg_idx + offset].aval.shape[loc0.shape_idx]
        for compare_loc in locations_list[1:]:
            compare_var = jaxpr.outvars[compare_loc.arg_idx + offset].aval.shape[
                compare_loc.shape_idx
            ]
            if compare_var is not first_var:
                return (
                    "Detected dynamically shaped arrays being resized independently. "
                    f"\nReturned variables at {loc0.arg_idx} and {compare_loc.arg_idx} must keep the same size "
                    "with allow_array_resizing=False."
                    f"\nPlease specify allow_array_resizing=True to `qml.{name}` to allow "
                    "dynamically shaped arrays to be reshaped independently. "
                )

    return None


def _has_dynamic_shape(val):
    return any(not isinstance(s, int) for s in getattr(val, "shape", ()))


def handle_jaxpr_error(
    e: ValueError, fns: tuple[Callable, ...], allow_array_resizing, name: str = "while_loop"
):
    """Handle any ValueError's raised by the creation of the jaxpr, adding information to any error
    about 'Incompatible shapes for broadcasting'."""
    import jax  # pylint: disable=import-outside-toplevel

    if "Incompatible shapes for broadcasting" in str(e) and jax.config.jax_dynamic_shapes:
        closures = sum(((fn.__closure__ or ()) for fn in fns), ())
        if any(_has_dynamic_shape(i.cell_contents) for i in closures):
            msg = (
                "Detected an attempt to combine arrays with different dynamic shapes. "
                "\nThis also may be due to a closure variable with a dynamic shape."
                " Try promoting the closure variable with the dynamic shape to being an explicit argument. "
            )
            if allow_array_resizing is True:
                msg += "\nThis may also be due to allow_array_resizing=True. Try with allow_array_resizing=False instead."
        else:
            msg = (
                "Detected an attempt to combine arrays with two different dynamic shapes. "
                f"To keep dynamic shapes matching, please specify `allow_array_resizing=False` to `qml.{name}`."
            )
        raise ValueError(msg) from e
    raise e


# pylint: disable=too-few-public-methods
class _CalculateLoopAbstractedAxes:
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
                # and use new number designation
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
        allow_array_resizing (bool): If True, each abstracted axis should be treated as
            an independent axis. Defaults to False.

    Returns:
        abstracted_axes, abstract_shapes, locations for shapes

    .. code-block:: python

        from pennylane.control_flow._loop_abstract_axes import loop_determine_abstracted_axes
        from functools import partial

        def f(*args, allow_array_resizing):
            abstracted_axes, abstract_shapes, locations = loop_determine_abstracted_axes(args, allow_array_resizing)
            print(abstracted_axes)
            print(abstract_shapes)
            print(locations)

        args = (0, jnp.ones(3), jnp.zeros((3, 3)))
        jax.make_jaxpr(partial(f, allow_array_resizing=False), abstracted_axes=({}, {0:"a"}, {1:"a"}))(*args)

    .. code-block::

        ({}, {0: 0}, {1: 0})
        [Traced<ShapedArray(int32[], weak_type=True)>with<DynamicJaxprTrace(level=1/0)>]
        [[AbstractShapeLocation(arg_idx=1, shape_idx=0), AbstractShapeLocation(arg_idx=2, shape_idx=1)]]

    Here we can verify that the output abstracted axes match what we put in. The returned ``abstract_shapes`` is the single
    abstract shape that occurs in both variables.  The locations array tells us that we can locate the first
    abstract shape in the ``1`` argument at shape position ``0``, and in the ``2`` argument at shape position ``1``.

    If we instead specify ``allow_array_resizing=True``, we can see the difference.

    ... code-block:: python

        jax.make_jaxpr(partial(f, allow_array_resizing=True), abstracted_axes=({}, {0:"a"}, {1:"a"}))(*args)

    .. code-block::

        ({}, {0: 0}, {1: 1})
        [Traced<ShapedArray(int32[], weak_type=True)>with<DynamicJaxprTrace(level=1/0)>, Traced<ShapedArray(int32[], weak_type=True)>with<DynamicJaxprTrace(level=1/0)>]
        [[AbstractShapeLocation(arg_idx=1, shape_idx=0)], [AbstractShapeLocation(arg_idx=2, shape_idx=1)]]

    Now the abstracted axes treat the two abstracted axes as different, even though they are the same tracer in the input
    arguments. The abstract shapes have two elements. By looking at the locations, we can see that we can find
    the first abstract shape in argument ``1`` at shape position ``0``, and we can find the second abstract shape in
    argument ``2`` at shape position ``1``.


    """
    import jax

    args, structure = jax.tree_util.tree_flatten(args)
    calculator = _CalculateLoopAbstractedAxes(allow_array_resizing=allow_array_resizing)
    _ = [calculator.add_arg(x_idx, x) for x_idx, x in enumerate(args)]

    if not any(calculator.abstracted_axes):
        return None, [], []

    abstracted_axes = jax.tree_util.tree_unflatten(structure, calculator.abstracted_axes)
    return abstracted_axes, calculator.abstract_shapes, calculator.shape_locations
