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
"""For loop."""
import functools

from pennylane import capture
from pennylane.capture import FlatFn, determine_abstracted_axes, enabled
from pennylane.compiler.compiler import AvailableCompilers, active_compiler


def for_loop(start, stop=None, step=1):
    """for_loop([start, ]stop[, step])
    A :func:`~.qjit` compatible for-loop for PennyLane programs. When
    used without :func:`~.qjit`, this function will fall back to a standard
    Python for loop.

    This decorator provides a functional version of the traditional
    for-loop, similar to `jax.cond.fori_loop <https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.fori_loop.html>`__.
    That is, any variables that are modified across iterations need to be provided
    as inputs/outputs to the loop body function:

    - Input arguments contain the value of a variable at the start of an
      iteration.

    - output arguments contain the value at the end of the iteration. The
      outputs are then fed back as inputs to the next iteration.

    The final iteration values are also returned from the transformed
    function.

    The semantics of ``for_loop`` are given by the following Python pseudo-code:

    .. code-block:: python

        def for_loop(start, stop, step, loop_fn, *args):
            for i in range(start, stop, step):
                args = loop_fn(i, *args)
            return args

    Unlike ``jax.cond.fori_loop``, the step can be negative if it is known at tracing time
    (i.e., constant). If a non-constant negative step is used, the loop will produce no iterations.

    .. note::

        This function can be used in the following different ways:

        1. ``for_loop(stop)``:  Values are generated within the interval ``[0, stop)``
        2. ``for_loop(start, stop)``: Values are generated within the interval ``[start, stop)``
        3. ``for_loop(start, stop, step)``: Values are generated within the interval ``[start, stop)``,
           with spacing between the values given by ``step``

    Args:
        start (int, optional): starting value of the iteration index.
            The default start value is ``0``
        stop (int): upper bound of the iteration index
        step (int, optional): increment applied to the iteration index at the end of
            each iteration. The default step size is ``1``

    Returns:
        Callable[[int, ...], ...]: A wrapper around the loop body function.
        Note that the loop body function must always have the iteration index as its first
        argument, which can be used arbitrarily inside the loop body. As the value of the index
        across iterations is handled automatically by the provided loop bounds, it must not be
        returned from the function.

    .. seealso:: :func:`~.while_loop`, :func:`~.qjit`

    **Example**

    .. code-block:: python

        dev = qml.device("lightning.qubit", wires=1)

        @qml.qnode(dev)
        def circuit(n: int, x: float):

            @qml.for_loop(0, n, 1)
            def loop_rx(i, x):
                # perform some work and update (some of) the arguments
                qml.RX(x, wires=0)

                # update the value of x for the next iteration
                return jnp.sin(x)

            # apply the for loop
            final_x = loop_rx(x)

            return qml.expval(qml.Z(0))

    >>> circuit(7, 1.6)
    array(0.97926626)

    ``for_loop`` is also :func:`~.qjit` compatible; when used with the
    :func:`~.qjit` decorator, the for loop will not be unrolled, and instead
    will be captured as-is during compilation and executed during runtime:

    >>> qml.qjit(circuit)(7, 1.6)
    Array(0.97926626, dtype=float64)

    .. note::

        Please see the Catalyst :doc:`quickstart guide <catalyst:dev/quick_start>`,
        as well as the :doc:`sharp bits and debugging tips <catalyst:dev/sharp_bits>`
        page for an overview of using quantum just-in-time compilation.

    """
    if stop is None:
        start, stop = 0, start

    if active_jit := active_compiler():
        compilers = AvailableCompilers.names_entrypoints
        ops_loader = compilers[active_jit]["ops"].load()
        return ops_loader.for_loop(start, stop, step)

    # if there is no active compiler, simply interpret the for loop
    # via the Python interpretor.
    def _decorator(body_fn):
        """Transform that will call the input ``body_fn`` within a for loop defined by the closure variables start, stop, and step.

        Args:
            body_fn (Callable): The function called within the for loop. Note that the loop body
                function must always have the iteration index as its first
                argument, which can be used arbitrarily inside the loop body. As the value of the index
                across iterations is handled automatically by the provided loop bounds, it must not be
                returned from the function.

        Closure Variables:
            start (int): starting value of the iteration index
            stop (int): (exclusive) upper bound of the iteration index
            step (int): increment applied to the iteration index at the end of each iteration

        Returns:
            Callable: a callable with the same signature as ``body_fn``
        """
        return ForLoopCallable(start, stop, step, body_fn)

    return _decorator


@functools.lru_cache
def _get_for_loop_qfunc_prim():
    """Get the loop_for primitive for quantum functions."""

    # pylint: disable=import-outside-toplevel
    from pennylane.capture.custom_primitives import NonInterpPrimitive

    for_loop_prim = NonInterpPrimitive("for_loop")
    for_loop_prim.multiple_results = True
    for_loop_prim.prim_type = "higher_order"

    # pylint: disable=too-many-arguments
    @for_loop_prim.def_impl
    def _(start, stop, step, *args, jaxpr_body_fn, consts_slice, args_slice, abstract_shapes_slice):

        consts = args[consts_slice]
        init_state = args[args_slice]
        abstract_shapes = args[abstract_shapes_slice]

        # in case start >= stop, return the initial state
        fn_res = init_state

        for i in range(start, stop, step):
            fn_res = capture.eval_jaxpr(jaxpr_body_fn, consts, *abstract_shapes, i, *fn_res)

        return fn_res

    # pylint: disable=unused-argument
    @for_loop_prim.def_abstract_eval
    def _(start, stop, step, *args, args_slice, **_):
        return args[args_slice]

    return for_loop_prim


class ForLoopCallable:  # pylint:disable=too-few-public-methods
    """Base class to represent a for loop. This class
    when called with an initial state will execute the while
    loop via the Python interpreter.

    Args:
        start (int): starting value of the iteration index
        stop (int): (exclusive) upper bound of the iteration index
        step (int): increment applied to the iteration index at the end of each iteration
        body_fn (Callable): The function called within the for loop. Note that the loop body
            function must always have the iteration index as its first
            argument, which can be used arbitrarily inside the loop body. As the value of the index
            across iterations is handled automatically by the provided loop bounds, it must not be
            returned from the function.
    """

    def __init__(self, start, stop, step, body_fn):
        self.start = start
        self.stop = stop
        self.step = step
        self.body_fn = body_fn

    def _call_capture_disabled(self, *init_state):
        args = init_state
        fn_res = args if len(args) > 1 else args[0] if len(args) == 1 else None

        for i in range(self.start, self.stop, self.step):
            fn_res = self.body_fn(i, *args)
            args = fn_res if len(args) > 1 else (fn_res,) if len(args) == 1 else ()

        return fn_res

    def _call_capture_enabled(self, *init_state):

        import jax  # pylint: disable=import-outside-toplevel

        for_loop_prim = _get_for_loop_qfunc_prim()

        abstracted_axes, abstract_shapes = determine_abstracted_axes((0, *init_state))

        flat_fn = FlatFn(self.body_fn)
        jaxpr_body_fn = jax.make_jaxpr(flat_fn, abstracted_axes=abstracted_axes)(0, *init_state)

        consts_slice = slice(0, len(jaxpr_body_fn.consts))
        abstract_shapes_slice = slice(consts_slice.stop, consts_slice.stop + len(abstract_shapes))
        args_slice = slice(abstract_shapes_slice.stop, None)

        flat_args, _ = jax.tree_util.tree_flatten(init_state)

        results = for_loop_prim.bind(
            self.start,
            self.stop,
            self.step,
            *jaxpr_body_fn.consts,
            *abstract_shapes,
            *flat_args,
            jaxpr_body_fn=jaxpr_body_fn.jaxpr,
            consts_slice=consts_slice,
            args_slice=args_slice,
            abstract_shapes_slice=abstract_shapes_slice,
        )
        assert flat_fn.out_tree is not None
        return jax.tree_util.tree_unflatten(flat_fn.out_tree, results)

    def __call__(self, *init_state):

        if enabled():
            return self._call_capture_enabled(*init_state)

        return self._call_capture_disabled(*init_state)
