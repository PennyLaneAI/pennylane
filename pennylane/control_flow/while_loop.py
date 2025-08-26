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
from typing import Literal

from pennylane import capture
from pennylane.capture import FlatFn, enabled
from pennylane.capture.dynamic_shapes import register_custom_staging_rule
from pennylane.compiler.compiler import AvailableCompilers, active_compiler

from ._loop_abstract_axes import (
    add_abstract_shapes,
    get_dummy_arg,
    handle_jaxpr_error,
    loop_determine_abstracted_axes,
    validate_no_resizing_returns,
)


def while_loop(cond_fn, allow_array_resizing: Literal["auto", True, False] = "auto"):
    """A :func:`~.qjit` compatible while-loop for PennyLane programs. When
    used without :func:`~.qjit` or program capture, this function will fall back to a standard
    Python while loop.

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
            with dynamic shapes that change between iterations. Defaults to `"auto"`.

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

        .. note::

            The following examples may yield different outputs depending on how the
            workflow function is executed. For instance, the function can be run
            directly as:

            >>> arg = 2
            >>> workflow(arg)

            Alternatively, the function can be traced with ``jax.make_jaxpr`` to produce a JAXPR representation,
            which captures the abstract computational graph for the given input and generates the abstract shapes.
            The resulting JAXPR can then be evaluated using ``qml.capture.eval_jaxpr``:

            >>> jaxpr = jax.make_jaxpr(workflow)(arg)
            >>> qml.capture.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, arg)

        A dynamically shaped array is an array whose shape depends on an abstract value. This is
        an experimental jax mode that can be turned on with:

        >>> import jax
        >>> import jax.numpy as jnp
        >>> jax.config.update("jax_dynamic_shapes", True)
        >>> qml.capture.enable()

        ``allow_array_resizing="auto"`` will try and choose between the following two possible modes.
        If the needed mode is ``allow_array_resizing=False``, then this will require re-capturing
        the loop, potentially taking more time.

        When working with dynamic shapes in a ``while_loop``, we have two possible
        options. ``allow_array_resizing=True`` treats every dynamic dimension as independent.

        .. code-block:: python

            @qml.while_loop(lambda a, b: jnp.sum(a) < 10, allow_array_resizing=True)
            def f(x, y):
                return jnp.hstack([x, y]), 2*y

            def workflow(i0):
                x0, y0 = jnp.ones(i0), jnp.ones(i0)
                return f(x0, y0)


        Even though ``x`` and ``y`` are initialized with the same shape, the shapes no longer match
        after one iteration. In this circumstance, ``x`` and ``y`` can no longer be combined
        with operations like ``x * y``, as they do not have matching shapes.

        With ``allow_array_resizing=False``, anything that starts with the same dynamic dimension
        must keep the same shape pattern throughout the loop.

        .. code-block:: python

            @qml.while_loop(lambda a, b: jnp.sum(a) < 10, allow_array_resizing=False)
            def f(x, y):
                return x * y, 2*y

            def workflow(i0):
                x0 = jnp.ones(i0)
                y0 = jnp.ones(i0)
                return f(x0, y0)


        Note that with ``allow_array_resizing=False``, all arrays can still be resized together, as
        long as the pattern still matches. For example, here both ``x`` and ``y`` start with the
        same shape, and keep the same shape as each other for each iteration.

        .. code-block:: python

            @qml.while_loop(lambda a, b: jnp.sum(a) < 10, allow_array_resizing=False)
            def f(x, y):
                x = jnp.hstack([x, y])
                return x, 2*x

            def workflow(i0):
                x0 = jnp.ones(i0)
                y0 = jnp.ones(i0)
                return f(x0, y0)

        Note that new dynamic dimensions cannot yet be created inside a loop.  Only things
        that already have a dynamic dimension can have that dynamic dimension change.
        For example, this is **not** a viable ``while_loop``, as ``x`` is initialized
        with an array with a concrete size.

        .. code-block:: python

            def w():
                @qml.while_loop(lambda i, x: i < 5)
                def f(i, x):
                    return i + 1, jnp.append(x, i)

                return f(0, jnp.array([]))

    """

    if active_jit := active_compiler():
        compilers = AvailableCompilers.names_entrypoints
        ops_loader = compilers[active_jit]["ops"].load()
        return ops_loader.while_loop(cond_fn)

    # if there is no active compiler, simply interpret the while loop
    # via the Python interpreter.
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
    from pennylane.capture.custom_primitives import QmlPrimitive

    while_loop_prim = QmlPrimitive("while_loop")
    while_loop_prim.multiple_results = True
    while_loop_prim.prim_type = "higher_order"
    register_custom_staging_rule(while_loop_prim, lambda params: params["jaxpr_body_fn"].outvars)

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
        self,
        cond_fn: Callable,
        body_fn: Callable,
        allow_array_resizing: Literal["auto", True, False] = "auto",
    ):
        self.cond_fn: Callable = cond_fn
        self.body_fn: Callable = body_fn
        self.allow_array_resizing = allow_array_resizing

    def _call_capture_disabled(self, *init_state):
        args = init_state
        fn_res = args if len(args) > 1 else args[0] if len(args) == 1 else None

        while self.cond_fn(*args):
            fn_res = self.body_fn(*args)
            args = fn_res if len(args) > 1 else (fn_res,) if len(args) == 1 else ()

        return fn_res

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
            new_body_fn = add_abstract_shapes(flat_body_fn, shape_locations)
            dummy_init_state = [get_dummy_arg(arg) for arg in flat_args]
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
            handle_jaxpr_error(e, (self.cond_fn, self.body_fn), self.allow_array_resizing)

        error_msg = validate_no_resizing_returns(jaxpr_body_fn.jaxpr, shape_locations)
        if error_msg:
            if allow_array_resizing == "auto":
                return self._get_jaxprs(init_state, allow_array_resizing=True)
            raise ValueError(error_msg)

        assert flat_body_fn.out_tree is not None, "Should be set when constructing the jaxpr"
        return jaxpr_body_fn, jaxpr_cond_fn, abstract_shapes + flat_args, flat_body_fn.out_tree

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
