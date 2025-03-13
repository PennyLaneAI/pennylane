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
from typing import Sequence

import numpy as np

from pennylane import capture
from pennylane.capture import FlatFn, enabled
from pennylane.capture.dynamic_shapes import loop_determine_abstracted_axes
from pennylane.compiler.compiler import AvailableCompilers, active_compiler


def while_loop(cond_fn, allow_array_resizing=False):
    """A :func:`~.qjit` compatible for-loop for PennyLane programs. When
    used without :func:`~.qjit`, this function will fall back to a standard
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
    _register_custom_staging_rule(while_loop_prim)

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


def _register_custom_staging_rule(while_prim):

    # see https://github.com/jax-ml/jax/blob/9e62994bce7c7fcbb2f6a50c9ef89526cd2c2be6/jax/_src/lax/lax.py#L3538
    # and https://github.com/jax-ml/jax/blob/9e62994bce7c7fcbb2f6a50c9ef89526cd2c2be6/jax/_src/lax/lax.py#L208
    # for reference to how jax is handling staging rules for dynamic shapes in v0.4.28
    # see also capture/intro_to_dynamic_shapes.md

    import jax  # pylint: disable=import-outside-toplevel
    from jax._src.interpreters import partial_eval as pe  # pylint: disable=import-outside-toplevel

    def _tracer_and_outvar(
        jaxpr_trace: pe.DynamicJaxprTrace,
        outvar: jax.core.Var,
        env: dict[jax.core.Var, jax.core.Var],
    ) -> tuple[pe.DynamicJaxprTracer, jax.core.Var]:
        """
        Create a new tracer and returned var from the true branch outvar
        returned vars are cached in env for use in future shapes
        """
        if not hasattr(outvar.aval, "shape"):
            out_tracer = pe.DynamicJaxprTracer(jaxpr_trace, outvar.aval)
            return out_tracer, jaxpr_trace.makevar(out_tracer)

        new_shape = [s if isinstance(s, int) else env[s] for s in outvar.aval.shape]
        new_aval = jax.core.DShapedArray(tuple(new_shape), outvar.aval.dtype)
        out_tracer = pe.DynamicJaxprTracer(jaxpr_trace, new_aval)
        new_var = jaxpr_trace.makevar(out_tracer)

        if not isinstance(outvar, jax.core.Literal):
            env[outvar] = new_var
        return out_tracer, new_var

    def custom_staging_rule(
        jaxpr_trace: pe.DynamicJaxprTrace, *tracers: pe.DynamicJaxprTracer, **params
    ) -> Sequence[pe.DynamicJaxprTracer] | pe.DynamicJaxprTracer:
        """
        Add new jaxpr equation to the jaxpr_trace and return new tracers.
        """
        if not jax.config.jax_dynamic_shapes:
            # fallback to normal behavior
            return jaxpr_trace.default_process_primitive(while_prim, tracers, params)
        body_outvars = params["jaxpr_body_fn"].outvars

        env: dict[jax.core.Var, jax.core.Var] = {}  # branch var to new equation var
        out_tracers, returned_vars = tuple(
            zip(*(_tracer_and_outvar(jaxpr_trace, var, env) for var in body_outvars), strict=True)
        )

        invars = [jaxpr_trace.getvar(x) for x in tracers]
        eqn = pe.new_jaxpr_eqn(
            invars,
            returned_vars,
            while_prim,
            params,
            jax.core.no_effects,
        )
        jaxpr_trace.frame.add_eqn(eqn)
        return out_tracers

    pe.custom_staging_rules[while_prim] = custom_staging_rule


def _add_abstract_shapes(f, shape_recipes):
    def new_f(*args, **kwargs):
        results = f(*args, **kwargs)
        if shape_recipes:
            new_shapes = [results[arg_idx].shape[shape_idx] for arg_idx, shape_idx in shape_recipes]
            return new_shapes + results
        return results

    return new_f


class WhileLoopCallable:  # pylint:disable=too-few-public-methods
    """Base class to represent a while loop. This class
    when called with an initial state will execute the while
    loop via the Python interpreter.

    Args:
        cond_fn (Callable): the condition function in the while loop
        body_fn (Callable): the function that is executed within the while loop
    """

    def __init__(self, cond_fn, body_fn, allow_array_resizing=False):
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

    def _call_capture_enabled(self, *init_state):

        import jax  # pylint: disable=import-outside-toplevel

        while_loop_prim = _get_while_loop_qfunc_prim()

        flat_args, in_tree = jax.tree_util.tree_flatten(init_state)
        abstracted_axes, abstract_shapes, shape_recipes = loop_determine_abstracted_axes(
            init_state, allow_array_resizing=self.allow_array_resizing
        )
        flat_body_fn = FlatFn(self.body_fn, in_tree=in_tree)
        flat_cond_fn = FlatFn(self.cond_fn, in_tree=in_tree)

        dummy_init_state = []
        for arg in flat_args:
            if all(isinstance(s, int) for s in arg.shape):
                dummy_init_state.append(arg)
            else:
                shape = tuple(s if isinstance(s, int) else 5 for s in arg.shape)
                dummy_init_state.append(np.empty(shape=shape, dtype=arg.dtype))

        if abstracted_axes:
            new_body_fn = _add_abstract_shapes(flat_body_fn, shape_recipes)
            new_cond_fn = _add_abstract_shapes(flat_cond_fn, None)
        else:
            new_body_fn = flat_body_fn
            new_cond_fn = self.cond_fn

        jaxpr_body_fn = jax.make_jaxpr(new_body_fn, abstracted_axes=abstracted_axes)(
            *dummy_init_state
        )
        jaxpr_cond_fn = jax.make_jaxpr(new_cond_fn, abstracted_axes=abstracted_axes)(
            *dummy_init_state
        )

        body_consts = slice(0, len(jaxpr_body_fn.consts))
        cond_consts = slice(body_consts.stop, body_consts.stop + len(jaxpr_cond_fn.consts))
        args_slice = slice(cond_consts.stop, None)

        results = while_loop_prim.bind(
            *jaxpr_body_fn.consts,
            *jaxpr_cond_fn.consts,
            *abstract_shapes,
            *flat_args,
            jaxpr_body_fn=jaxpr_body_fn.jaxpr,
            jaxpr_cond_fn=jaxpr_cond_fn.jaxpr,
            body_slice=body_consts,
            cond_slice=cond_consts,
            args_slice=args_slice,
        )
        assert flat_body_fn.out_tree is not None, "Should be set when constructing the jaxpr"
        results = results[-flat_body_fn.out_tree.num_leaves :]
        return jax.tree_util.tree_unflatten(flat_body_fn.out_tree, results)

    def __call__(self, *init_state):

        if enabled():
            return self._call_capture_enabled(*init_state)

        return self._call_capture_disabled(*init_state)
