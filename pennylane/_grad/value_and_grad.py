# Copyright 2026 Xanadu Quantum Technologies Inc.

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
Defines qp.value_and_grad.
"""

import inspect
from functools import lru_cache, wraps
from importlib.util import find_spec

from pennylane import capture
from pennylane.compiler import compiler
from pennylane.exceptions import CompileError

from .grad import _args_and_argnums, _setup_h, _setup_method, _ShapedArray

_has_jax = find_spec("jax") is not None


# pylint: disable=unused-argument, too-many-arguments
@lru_cache
def _get_value_and_grad_prim():
    """Create a primitive for value and gradient computations."""
    if not _has_jax:  # pragma: no cover
        return None

    import jax  # pylint: disable=import-outside-toplevel

    value_and_grad_prim = capture.QpPrimitive("value_and_grad")
    value_and_grad_prim.multiple_results = True
    value_and_grad_prim.prim_type = "higher_order"

    @value_and_grad_prim.def_impl
    def _value_and_grad_impl(*args, argnums, jaxpr, method, h, fn):
        if method != "auto":  # pragma: no cover
            raise ValueError(f"Invalid value '{method=}' without QJIT.")

        def func(*inner_args):
            res = jax.core.eval_jaxpr(jaxpr, [], *inner_args)
            return res[0]

        res = jax.value_and_grad(func, argnums=argnums)(*args)
        return jax.tree_util.tree_leaves(res)

    # pylint: disable=unused-argument
    @value_and_grad_prim.def_abstract_eval
    def _value_and_grad_abstract(*args, argnums, jaxpr, method, h, fn):
        in_avals = tuple(args[i] for i in argnums)
        grad_avals = (
            _ShapedArray(in_aval.shape, in_aval.dtype, weak_type=in_aval.weak_type)
            for in_aval in in_avals
        )
        return [jaxpr.outvars[0].aval, *grad_avals]

    return value_and_grad_prim


def _capture_value_and_grad(func, *, argnums=0, method=None, h=None):
    # mostly a copy-paste of _capture_diff, but a few minor things needed to get updated
    # Could also find a way to remove code duplication

    import jax  # pylint: disable=import-outside-toplevel

    # pylint: disable=import-outside-toplevel
    from jax.tree_util import tree_flatten, tree_leaves, tree_unflatten

    h = _setup_h(h)
    method = _setup_method(method)
    _argnums = argnums  # somehow renaming stops it from being unbound?

    @wraps(func)
    def new_func(*args, **kwargs):
        flat_args, flat_argnums, full_in_tree, trainable_in_tree = _args_and_argnums(args, _argnums)

        # Create fully flattened function (flat inputs & outputs)
        flat_fn = capture.FlatFn(func, full_in_tree)

        abstracted_axes, abstract_shapes = capture.determine_abstracted_axes(tuple(flat_args))
        jaxpr = jax.make_jaxpr(flat_fn, abstracted_axes=abstracted_axes)(*flat_args, **kwargs)

        if len(jaxpr.out_avals) > 1 or jaxpr.out_avals[0].shape != ():
            raise TypeError(
                f"value_and_grad only defined for scalar-output functions. Got {jaxpr.out_avals}"
            )

        num_abstract_shapes = len(abstract_shapes)
        shift = num_abstract_shapes + len(jaxpr.consts)
        shifted_argnums = tuple(a + shift for a in flat_argnums)
        j = jaxpr.jaxpr
        no_consts_jaxpr = j.replace(constvars=(), invars=j.constvars + j.invars)

        flat_inputs, _ = tree_flatten((args, kwargs))
        prim_kwargs = {
            "argnums": shifted_argnums,
            "jaxpr": no_consts_jaxpr,
            "fn": func,
            "method": method,
            "h": h,
        }
        out_flat = _get_value_and_grad_prim().bind(
            *jaxpr.consts,
            *abstract_shapes,
            *flat_inputs,
            **prim_kwargs,
        )

        res_flat, *grad_flat = out_flat
        # flatten once more to go from 2D derivative structure (outputs, args) to flat structure
        grad_flat = tree_leaves(grad_flat)
        assert flat_fn.out_tree is not None, "out_tree should be set after executing flat_fn"
        res_nested = tree_unflatten(flat_fn.out_tree, [res_flat])

        # The derivative output tree is the composition of output tree and trainable input trees
        combined_tree = flat_fn.out_tree.compose(trainable_in_tree)
        grad_nested = tree_unflatten(combined_tree, grad_flat)

        return res_nested, grad_nested

    return new_func


# pylint: disable=too-few-public-methods
class value_and_grad:
    """A :func:`~.qjit`-compatible transformation for returning the result and jacobian of a
    function.

    This function allows the value and the gradient of a hybrid quantum-classical function to be
    computed within the compiled program.

    Note that ``value_and_grad`` can be more efficient, and reduce overall quantum executions,
    compared to separately executing the function and then computing its gradient.

    .. warning::

        Currently, higher-order differentiation is only supported by the finite-difference
        method.

    Args:
        fn (Callable): a function or a function object to differentiate
        method (str): The method used for differentiation, which can be any of ``["auto", "fd"]``,
                      where:

                      - ``"auto"`` represents deferring the quantum differentiation to the method
                        specified by the QNode, while the classical computation is differentiated
                        using traditional auto-diff. Catalyst supports ``"parameter-shift"`` and
                        ``"adjoint"`` on internal QNodes. Notably, QNodes with
                        ``diff_method="finite-diff"`` are not supported with ``"auto"``.

                      - ``"fd"`` represents first-order finite-differences for the entire hybrid
                        function.

        h (float): the step-size value for the finite-difference (``"fd"``) method
        argnums (Tuple[int, List[int]]): the argument indices to differentiate

    Returns:
        Callable: A callable object that computes the value and gradient of the wrapped function
        for the given arguments.

    Raises:
        ValueError: Invalid method or step size parameters.
        DifferentiableCompilerError: Called on a function that doesn't return a single scalar.

    .. note::

        Any JAX-compatible optimization library, such as `Optax
        <https://optax.readthedocs.io/en/stable/index.html>`_, can be used
        alongside ``value_and_grad`` for JIT-compatible variational workflows.
        See the :doc:`quickstart guide <catalyst:dev/quick_start>` for examples.

    .. seealso:: :func:`~.grad`, :func:`~.jacobian`

    **Example 1 (Classical preprocessing)**

    .. code-block:: python

        dev = qp.device("lightning.qubit", wires=1)

        @qp.qjit
        def workflow(x):
            @qp.qnode(dev)
            def circuit(x):
                qp.RX(jnp.pi * x, wires=0)
                return qp.expval(qp.PauliY(0))
            return qp.value_and_grad(circuit)(x)

    >>> workflow(0.2)
    (Array(-0.58778525, dtype=float64), Array(-2.54160185, dtype=float64))

    **Example 2 (Classical preprocessing and postprocessing)**

    .. code-block:: python

        dev = qp.device("lightning.qubit", wires=1)

        @qp.qjit
        def value_and_grad_loss(theta):
            @qp.qnode(dev, diff_method="adjoint")
            def circuit(theta):
                qp.RX(jnp.exp(theta ** 2) / jnp.cos(theta / 4), wires=0)
                return qp.expval(qp.PauliZ(wires=0))

            def loss(theta):
                return jnp.pi / jnp.tanh(circuit(theta))

            return qp.value_and_grad(loss, method="auto")(theta)

    >>> value_and_grad_loss(1.0)
    (Array(-4.2622289, dtype=float64), Array(5.04324559, dtype=float64))

    **Example 3 (Purely classical functions)**

    .. code-block:: python

        def square(x: float):
            return x ** 2

        @qp.qjit
        def dsquare(x: float):
            return qp.value_and_grad(square)(x)

    >>> dsquare(2.3)
    (Array(5.29, dtype=float64), Array(4.6, dtype=float64))
    """

    def __init__(self, func, argnums=0, method=None, h=None):
        self._func = func
        self._argnums = argnums
        self._method = method
        self._h = h
        # need to preserve input signature for use in catalyst AOT compilation, but
        # get rid of return annotation to placate autograd
        self.__signature__ = inspect.signature(self._func).replace(
            return_annotation=inspect.Signature.empty
        )
        fn_name = getattr(self._func, "__name__", repr(self._func))
        self.__name__ = f"<value_and_grad: {fn_name}>"

    def __call__(self, *args, **kwargs):
        if active_jit := compiler.active_compiler():
            available_eps = compiler.AvailableCompilers.names_entrypoints
            ops_loader = available_eps[active_jit]["ops"].load()
            return ops_loader.value_and_grad(
                self._func, method=self._method, h=self._h, argnums=self._argnums
            )(*args, **kwargs)

        if capture.enabled():
            g = _capture_value_and_grad(
                self._func, argnums=self._argnums, method=self._method, h=self._h
            )
            return g(*args, **kwargs)

        raise CompileError("PennyLane does not support the value_and_grad function without QJIT.")
