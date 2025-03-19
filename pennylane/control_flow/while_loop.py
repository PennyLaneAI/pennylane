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
"""While loop."""
import functools
from collections.abc import Callable
from typing import Literal, Optional

import numpy as np

from pennylane import capture
from pennylane.capture import FlatFn, enabled
from pennylane.capture.dynamic_shapes import register_custom_staging_rule
from pennylane.compiler.compiler import AvailableCompilers, active_compiler

from ._loop_abstract_axes import AbstractShapeLocation, loop_determine_abstracted_axes


def while_loop(cond_fn, allow_array_resizing: Literal["auto", True, False] = "auto"):
    """A :func:`~.qjit` compatible for-loop for PennyLane programs. When
    used without :func:`~.qjit` or program capture, this function will fall back to a standard
    Python for loop.

    This decorator provides a functional version of the traditional while loop,
    similar to `jax.lax.while_loop <https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.while_loop.html>`__.
    That is, any variables that are modified across iterations need to be provided as
    inputs and outputs to the loop body function:

    - Input arguments contain the value of a variable at the start of an
      iteration

    - Output arguments contain the value at the end of the iteration. The
      outputs are then fed back as inputs to the next iteration.

    The final iteration values are also returned from the transformed function.

    The semantics of ``while_loop`` are given by the following Python pseudocode:

    .. code-block:: python

        def while_loop(cond_fn, body_fn, *args):
            while cond_fn(*args):
                args = body_fn(*args)
            return args

    Args:
        cond_fn (Callable): the condition function in the while loop
        allow_array_resizing (Literal["auto", True, False]): How to handle arrays
            with dynamic shapes that change between iterations

    Returns:
        Callable: A wrapper around the while-loop function.

    Raises:
        CompileError: if the compiler is not installed

    .. seealso:: :func:`~.for_loop`, :func:`~.qjit`

    **Example**

    .. code-block:: python

        dev = qml.device("lightning.qubit", wires=1)

        @qml.qnode(dev)
        def circuit(x: float):

            @qml.while_loop(lambda x: x < 2.0)
            def loop_rx(x):
                # perform some work and update (some of) the arguments
                qml.RX(x, wires=0)
                return x ** 2

            # apply the while loop
            loop_rx(x)

            return qml.expval(qml.Z(0))

    >>> circuit(1.6)
    -0.02919952

    ``while_loop`` is also :func:`~.qjit` compatible; when used with the
    :func:`~.qjit` decorator, the while loop will not be unrolled, and instead
    will be captured as-is during compilation and executed during runtime:

    >>> qml.qjit(circuit)(1.6)
    Array(-0.02919952, dtype=float64)

    .. details::
        :title: Usage Details

        A dynamicly shaped array is an array whose shape depends on an abstract value. This is
        an experimental jax mode that can be turned on with:

        >>> jax.config.update("jax_dynamic_shapes", True)

        Note that new dynamic dimensions cannot yet be created inside a loop.  Only things
        that already have a dynamic dimension can have that dynamic dimension change.
        For example, this is **not** a viable ``while_loop``, as ``x`` is initialized
        with an array with a concrete size.

        .. code-block:: python

            def w():
                @qml.while_loop(lambda i, x: i < 5)
                def f(i, x):
                    return i + 1, jax.numpy.append(x, i)

                return f(0, jnp.array([]))

        When working with dynamic shapes in a ``while_loop``, we have two possible
        options. In ``allow_array_resizing=True`` treats every dynamic dimension as indepedent.

        .. code-block:: python

            @qml.while_loop(lambda a, b: jax.numpy.sum(a) < 10, allow_array_array_resizing=True)
            def f(a, b):
                return jax.numpy.hstack([a, b]), 2*b

            def w(i0):
                a0, b0 = jnp.ones(i0), jnp.ones(i0)
                return f(a0, b0)

        Even though ``a`` and ``b`` are initialized with the same shape, the shapes no longer match
        after one iteration. In this circumstance, ``a`` and ``b`` can no longer be combined
        with operations like ``a * b``, as they do not have matching shapes.

        With ``allow_array_resizing=False``, anything that starts with the same dynamic dimension
        must keep the same shape throughout the loop.

        .. code-block:: python

            @qml.while_loop(lambda a, b: jax.numpy.sum(a) < 10, allow_array_resizing=False)
            def f(x, y):
                return x * y, 2*y

            def workflow(i0):
                x0 = jnp.ones(i0)
                y0 = jnp.ones(i0)
                return f(x0, y0)

        Note that with ``allow_array_resizing=False``, all arrays can still be resizing together, as
        long as the pattern still matches:

        .. code-block:: python

            @qml.while_loop(lambda a, b: jax.numpy.sum(a) < 10, allow_array_resizing=False)
            def f(x, y):
                x = jnp.hstack([x, y])
                return x, 2*x

            def workflow(i0):
                x0 = jnp.ones(i0)
                y0 = jnp.ones(i0)
                return f(x0, y0)

        ``allow_array_resizing="auto"`` will try and choose between these two options, but will
        add some additional overhead if ``allow_array_resizing=True`` is the best fit.

    """

    if active_jit := active_compiler():
        compilers = AvailableCompilers.names_entrypoints
        ops_loader = compilers[active_jit]["ops"].load()
        return ops_loader.while_loop(cond_fn)

    # if there is no active compiler, simply interpret the while loop
    # via the Python interpretor.
    def _decorator(body_fn: Callable) -> Callable:
        """Transform that will call the input ``body_fn`` until the closure variable ``cond_fn`` is met.

        Args:
            body_fn (Callable):

        Closure Variables:
            cond_fn (Callable):

        Returns:
            Callable: a callable with the same signature as ``body_fn`` and ``cond_fn``.
        """
        return WhileLoopCallable(cond_fn, body_fn, allow_array_resizing=allow_array_resizing)

    return _decorator


@functools.lru_cache
def _get_while_loop_qfunc_prim():
    """Get the while_loop primitive for quantum functions."""

    # pylint: disable=import-outside-toplevel
    from pennylane.capture.custom_primitives import NonInterpPrimitive

    while_loop_prim = NonInterpPrimitive("while_loop")
    while_loop_prim.multiple_results = True
    while_loop_prim.prim_type = "higher_order"
    register_custom_staging_rule(while_loop_prim, lambda params: params["jaxpr_body_fn"].outvars)

    # pylint: disable=too-many-arguments
    @while_loop_prim.def_impl
    def _(
        *args,
        jaxpr_body_fn,
        jaxpr_cond_fn,
        body_slice,
        cond_slice,
        args_slice,
    ):

        jaxpr_consts_body = args[body_slice]
        jaxpr_consts_cond = args[cond_slice]
        init_state = args[args_slice]
        # If cond_fn(*init_state) is False, return the initial state
        fn_res = init_state
        while capture.eval_jaxpr(jaxpr_cond_fn, jaxpr_consts_cond, *fn_res)[0]:
            fn_res = capture.eval_jaxpr(jaxpr_body_fn, jaxpr_consts_body, *fn_res)

        return fn_res

    @while_loop_prim.def_abstract_eval
    def _(*args, args_slice, **__):
        return args[args_slice]

    return while_loop_prim


def _add_abstract_shapes(f, shape_locations):
    def new_f(*args, **kwargs):
        results = f(*args, **kwargs)
        new_shapes = [results[loc[0].arg_idx].shape[loc[0].shape_idx] for loc in shape_locations]
        return new_shapes + results

    return new_f


def _has_dynamic_shape(val):
    return any(not isinstance(s, int) for s in getattr(val, "shape", ()))


def _get_dummy_arg(arg):
    """If any axes are abstract, replace with an empty numpy array."""
    if all(isinstance(s, int) for s in arg.shape):
        return arg
    shape = tuple(s if isinstance(s, int) else 2 for s in arg.shape)
    return np.empty(shape=shape, dtype=arg.dtype)


def _validate_no_resizing_returns(
    jaxpr: "jax.core.Jaxpr", locations: list[list[AbstractShapeLocation]]
) -> Optional[str]:
    offset = len(locations)

    for locations_list in locations:
        l0 = locations_list[0]
        first_var = jaxpr.outvars[l0.arg_idx + offset].aval.shape[l0.shape_idx]
        for compare_loc in locations_list[1:]:
            compare_var = jaxpr.outvars[compare_loc.arg_idx + offset].aval.shape[
                compare_loc.shape_idx
            ]
            if compare_var is not first_var:
                return (
                    "Detected dynamically shaped arrays being resized indepedently. "
                    f"\nReturned variables at {l0.arg_idx} and {compare_loc.arg_idx} must keep the same size "
                    "with allow_array_resizing=False."
                    "\nPlease specify allow_array_resizing=True to `qml.for_loop` to allow "
                    "dynamically shaped arrays to be reshaped indepdendently. "
                )

    return None


def _validate_static_shapes_dtypes(jaxpr, abstracted_axes, offset=0, is_for=False) -> None:
    if abstracted_axes:
        abstracted_axes = abstracted_axes[is_for:]
    else:
        abstracted_axes = (() for _ in jaxpr.invars[offset + is_for :])
    for invar, outvar, aa in zip(
        jaxpr.invars[offset + is_for :], jaxpr.outvars[offset:], abstracted_axes, strict=True
    ):
        if getattr(invar.aval, "dtype", ()) != getattr(outvar.aval, "dtype", ()):
            raise ValueError(
                "dtype of the output variable must match the dtype of the corresponding input."
                f" Got {invar.aval.dtype} for the input variable and {outvar.aval.dtype} for the output variable."
            )
        shape1 = getattr(invar.aval, "shape", ())
        shape2 = getattr(outvar.aval, "shape", ())
        if len(shape1) != len(shape2):
            raise ValueError(
                "The shape of the output variable must match the shape of the input variable."
                f" Got {shape1} and {shape2} for corresponding variables."
            )
        for i, (s1, s2) in enumerate(zip(shape1, shape2)):
            if i not in aa and s1 != s2:
                raise ValueError(
                    "The shape of the output variable must match the shape of the input variable."
                    f" Got {shape1} and {shape2} for corresponding variables."
                )


class WhileLoopCallable:  # pylint:disable=too-few-public-methods
    """Base class to represent a while loop. This class
    when called with an initial state will execute the while
    loop via the Python interpreter.

    Args:
        cond_fn (Callable): the condition function in the while loop
        body_fn (Callable): the function that is executed within the while loop
        allow_array_resizing (Literal["auto", True, False])
    """

    def __init__(
        self, cond_fn, body_fn, allow_array_resizing: Literal["auto", True, False] = "auto"
    ):
        self.cond_fn = cond_fn
        self.body_fn = body_fn
        self.allow_array_resizing = allow_array_resizing

    def _call_capture_disabled(self, *init_state):
        args = init_state
        fn_res = args if len(args) > 1 else args[0] if len(args) == 1 else None

        while self.cond_fn(*args):
            fn_res = self.body_fn(*args)
            args = fn_res if len(args) > 1 else (fn_res,) if len(args) == 1 else ()

        return fn_res

    def _handle_error(self, e: ValueError):
        """Handle any ValueError's raised by the creation of the jaxpr, adding information to any error
        about 'Incompatible shapes for broadcasting'."""
        import jax  # pylint: disable=import-outside-toplevel

        if "Incompatible shapes for broadcasting" in str(e) and jax.config.jax_dynamic_shapes:
            closures = (self.body_fn.__closure__ or ()) + (self.cond_fn.__closure__ or ())
            if any(_has_dynamic_shape(i.cell_contents) for i in closures):
                msg = (
                    "Detected an attempt to combine arrays with different dynamic shapes. "
                    "\nThis also may be due to a closure variable with a dynamic shape."
                    " Try promoting the closure variable with the dynamic shape to being an explicit argument. "
                )
                if self.allow_array_resizing is True:
                    msg += "\nThis may also be due to allow_array_resizing=True. Try with allow_array_resizing=False instead."
            else:
                msg = (
                    "Detected an attempt to combine arrays with two different dynamic shapes. "
                    "To keep dynamic shapes matching, please specify ``allow_array_resizing=False`` to ``qml.for_loop``."
                )
            raise ValueError(msg) from e
        raise e

    def _get_jaxprs(self, init_state, allow_array_resizing):
        import jax  # pylint: disable=import-outside-toplevel

        flat_args, in_tree = jax.tree_util.tree_flatten(init_state)
        tmp_array_resizing = False if allow_array_resizing == "auto" else allow_array_resizing
        abstracted_axes, abstract_shapes, shape_locations = loop_determine_abstracted_axes(
            tuple(flat_args), allow_array_resizing=tmp_array_resizing
        )

        flat_body_fn = FlatFn(self.body_fn, in_tree=in_tree)
        flat_cond_fn = FlatFn(self.cond_fn, in_tree=in_tree)

        if abstracted_axes:
            new_body_fn = _add_abstract_shapes(flat_body_fn, shape_locations)
            dummy_init_state = [_get_dummy_arg(arg) for arg in flat_args]
        else:
            new_body_fn = flat_body_fn
            dummy_init_state = flat_args

        try:
            jaxpr_body_fn = jax.make_jaxpr(new_body_fn, abstracted_axes=abstracted_axes)(
                *dummy_init_state
            )
            jaxpr_cond_fn = jax.make_jaxpr(flat_cond_fn, abstracted_axes=abstracted_axes)(
                *dummy_init_state
            )
        except ValueError as e:
            self._handle_error(e)

        _validate_static_shapes_dtypes(
            jaxpr_body_fn.jaxpr, abstracted_axes, offset=len(abstract_shapes)
        )
        validation = _validate_no_resizing_returns(jaxpr_body_fn.jaxpr, shape_locations)
        if validation:
            if allow_array_resizing == "auto":
                return self._get_jaxprs(init_state, allow_array_resizing=True)
            raise ValueError(validation)

        assert flat_body_fn.out_tree is not None, "Should be set when constructing the jaxpr"
        out_tree = flat_body_fn.out_tree

        return jaxpr_body_fn, jaxpr_cond_fn, abstract_shapes + flat_args, out_tree

    def _call_capture_enabled(self, *init_state):

        import jax  # pylint: disable=import-outside-toplevel

        while_loop_prim = _get_while_loop_qfunc_prim()

        jaxpr_body_fn, jaxpr_cond_fn, all_args, out_tree = self._get_jaxprs(
            init_state, allow_array_resizing=self.allow_array_resizing
        )

        body_consts = slice(0, len(jaxpr_body_fn.consts))
        cond_consts = slice(body_consts.stop, body_consts.stop + len(jaxpr_cond_fn.consts))
        args_slice = slice(cond_consts.stop, None)

        results = while_loop_prim.bind(
            *jaxpr_body_fn.consts,
            *jaxpr_cond_fn.consts,
            *all_args,
            jaxpr_body_fn=jaxpr_body_fn.jaxpr,
            jaxpr_cond_fn=jaxpr_cond_fn.jaxpr,
            body_slice=body_consts,
            cond_slice=cond_consts,
            args_slice=args_slice,
        )

        results = results[-out_tree.num_leaves :]
        return jax.tree_util.tree_unflatten(out_tree, results)

    def __call__(self, *init_state):

        if enabled():
            return self._call_capture_enabled(*init_state)

        return self._call_capture_disabled(*init_state)
