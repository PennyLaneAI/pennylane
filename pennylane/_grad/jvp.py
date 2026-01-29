# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

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
Defines qml.jvp
"""
from collections.abc import Sequence
from functools import lru_cache
from importlib.util import find_spec

from pennylane import capture
from pennylane.compiler import compiler
from pennylane.exceptions import CompileError

from .grad import _args_and_argnums, _setup_h, _setup_method

has_jax = find_spec("jax") is not None


def _get_shape(x):
    import jax  # pylint: disable=import-outside-toplevel

    return getattr(x, "shape", jax.numpy.shape(x))


# pylint: disable=unused-argument
@lru_cache
def _get_jvp_prim():
    if not has_jax:  # pragma: no cover
        return None

    import jax  # pylint: disable=import-outside-toplevel

    jvp_prim = capture.QmlPrimitive("jvp")
    jvp_prim.multiple_results = True
    jvp_prim.prim_type = "higher_order"

    @jvp_prim.def_impl
    def _jvp_impl(*args, jaxpr, fn, method, h, argnums):
        params = list(args[: len(jaxpr.invars)])
        dparams = list(args[len(jaxpr.invars) :])

        for i, p in enumerate(params):
            if i not in argnums:
                dparams.insert(i, 0 * p)

        def func(*inner_args):
            return jax.core.eval_jaxpr(jaxpr, [], *inner_args)

        results, dresults = jax.jvp(func, params, dparams)
        return (*results, *dresults)

    @jvp_prim.def_abstract_eval
    def _jvp_abstract_eval(*args, jaxpr, fn, method, h, argnums):
        return 2 * [v.aval for v in jaxpr.outvars]

    return jvp_prim


def _validate_tangents(params, dparams, argnums):
    from jax._src.api import _dtype  # pylint: disable=import-outside-toplevel

    if len(dparams) != len(argnums):
        raise TypeError(
            "number of tangents and number of differentiable parameters in qml.jvp do not "
            "match; the number of parameters must be equal. "
            f"Got {len(argnums)} differentiable parameters and so expected "
            f"as many tangents, but got {len(dparams)} instead."
        )

    for i, dx in zip(argnums, dparams):
        x = params[i]
        if _dtype(x) != _dtype(dx):
            raise TypeError(
                "function params and tangents arguments to qml.jvp do not match; "
                "dtypes must be equal. "
                f"Got function params dtype {_dtype(x)} and expected tangent dtype "
                f"to match, but got tangent dtype {_dtype(dx)} instead."
            )

        if _get_shape(x) != _get_shape(dx):
            raise ValueError(
                "qml.jvp called with different function params and tangent "
                f"shapes; got function params shape {_get_shape(x)} and tangent shape "
                f"{_get_shape(dx)}"
            )


# pylint: disable=too-many-arguments
def _capture_jvp(func, params, dparams, *, argnums=None, method=None, h=None):
    import jax  # pylint: disable=import-outside-toplevel
    from jax.tree_util import tree_leaves, tree_unflatten  # pylint: disable=import-outside-toplevel

    if not isinstance(params, Sequence):
        raise ValueError(f"params must be a Sequence in qml.jvp. Got type {type(params)}.")
    if not isinstance(dparams, Sequence):
        raise ValueError(f"tangents must be a Sequence in qml.jvp. Got type {type(params)}.")

    h = _setup_h(h)
    method = _setup_method(method)
    flat_args, flat_argnums, _, _ = _args_and_argnums(params, argnums)
    flat_dargs = tree_leaves(dparams)

    _validate_tangents(flat_args, flat_dargs, flat_argnums)

    flat_fn = capture.FlatFn(func)
    jaxpr = jax.make_jaxpr(flat_fn)(*params)
    j = jaxpr.jaxpr
    no_consts_jaxpr = j.replace(constvars=(), invars=j.constvars + j.invars)
    shifted_argnums = tuple(i + len(jaxpr.consts) for i in flat_argnums)

    prim_kwargs = {
        "fn": func,
        "method": method,
        "h": h,
        "argnums": shifted_argnums,
        "jaxpr": no_consts_jaxpr,
    }
    out_flat = _get_jvp_prim().bind(*jaxpr.consts, *flat_args, *flat_dargs, **prim_kwargs)
    flat_results, flat_dresults = out_flat[: len(j.outvars)], out_flat[len(j.outvars) :]

    results = tree_unflatten(flat_fn.out_tree, flat_results)
    dresults = tree_unflatten(flat_fn.out_tree, flat_dresults)
    return results, dresults

from .grad import _args_and_argnums, _setup_h, _setup_method

has_jax = True
try:
    import jax
except ImportError:
    has_jax = False


def _get_shape(x):
    return x.shape if hasattr(x, "shape") else jax.numpy.shape(x)


# pylint: disable=unused-argument
@lru_cache
def _get_vjp_prim():
    if not has_jax:  # pragma: no cover
        return None

    vjp_prim = capture.QmlPrimitive("vjp")
    vjp_prim.multiple_results = True
    vjp_prim.prim_type = "higher_order"

    @vjp_prim.def_impl
    def _vjp_impl(*args, jaxpr, fn, method, h, argnums):
        params = args[: len(jaxpr.invars)]
        dy = list(args[len(jaxpr.invars) :])

        def func(*inner_args):
            return jax.core.eval_jaxpr(jaxpr, [], *inner_args)

        res, vjp_fn = jax.vjp(func, *params)
        dparams = vjp_fn(dy)
        return res + [dparams[i] for i in argnums]

    @vjp_prim.def_abstract_eval
    def _vjp_abstract_eval(*args, jaxpr, fn, method, h, argnums):
        return [v.aval for v in jaxpr.outvars] + [jaxpr.invars[i].aval for i in argnums]

    return vjp_prim


@lru_cache
def _get_jvp_prim():
    if not has_jax:  # pragma: no cover
        return None

    jvp_prim = capture.QmlPrimitive("jvp")
    jvp_prim.multiple_results = True
    jvp_prim.prim_type = "higher_order"

    @jvp_prim.def_impl
    def _jvp_impl(*args, jaxpr, fn, method, h, argnums):
        params = list(args[: len(jaxpr.invars)])
        dparams = list(args[len(jaxpr.invars) :])

        print(params, dparams)
        for i, p in enumerate(params):
            if i not in argnums:
                dparams.insert(i, 0 * p)

        def func(*inner_args):
            return jax.core.eval_jaxpr(jaxpr, [], *inner_args)

        results, dresults = jax.jvp(func, params, dparams)
        print(results, dresults)
        return (*results, *dresults)

    @jvp_prim.def_abstract_eval
    def _jvp_abstract_eval(*args, jaxpr, fn, method, h, argnums):
        return 2 * [v.aval for v in jaxpr.outvars]

    return jvp_prim


def _validate_tangents(params, dparams, argnums):
    from jax._src.api import _dtype  # pylint: disable=import-outside-toplevel

    if len(dparams) != len(argnums):
        raise TypeError(
            "number of tangent and number of differentiable parameters in qml.jvp do not "
            "match; the number of parameters must be equal. "
            f"Got {len(argnums)} differentiable parameters and so expected "
            f"as many tangents, but got {len(dparams)} instead."
        )

    for i, dx in zip(argnums, dparams):
        x = params[i]
        if _dtype(x) != _dtype(dx):
            raise TypeError(
                "function params and tangents arguments to qml.jvp do not match; "
                "dtypes must be equal. "
                f"Got function params dtype {_dtype(x)} and so expected tangent dtype "
                f"{_dtype(x)}, but got tangent dtype {_dtype(dx)} instead."
            )

        if _get_shape(x) != _get_shape(dx):
            raise ValueError(
                "qml.jvp called with different function params and tangent "
                f"shapes; got function params shape {_get_shape(x)} and tangent shape "
                f"{_get_shape(dx)}"
            )


# pylint: disable=too-many-arguments
def _capture_jvp(func, params, dparams, *, argnums=None, method=None, h=None):
    from jax.tree_util import tree_leaves, tree_unflatten  # pylint: disable=import-outside-toplevel

    if not isinstance(params, Sequence):
        raise ValueError(f"params must be a Sequence in qml.jvp. Got type {type(params)}.")
    if not isinstance(dparams, Sequence):
        raise ValueError(f"tangents must be a Sequence in qml.jvp. Got type {type(params)}.")

    h = _setup_h(h)
    method = _setup_method(method)
    flat_args, flat_argnums, _, _ = _args_and_argnums(params, argnums)
    flat_dargs = tree_leaves(dparams)

    _validate_tangents(flat_args, flat_dargs, flat_argnums)

    flat_fn = capture.FlatFn(func)
    jaxpr = jax.make_jaxpr(flat_fn)(*params)
    j = jaxpr.jaxpr
    no_consts_jaxpr = j.replace(constvars=(), invars=j.constvars + j.invars)
    shifted_argnums = tuple(i + len(jaxpr.consts) for i in flat_argnums)

    prim_kwargs = {
        "fn": func,
        "method": method,
        "h": h,
        "argnums": shifted_argnums,
        "jaxpr": no_consts_jaxpr,
    }
    out_flat = _get_jvp_prim().bind(*jaxpr.consts, *flat_args, *flat_dargs, **prim_kwargs)
    flat_results, flat_dresults = out_flat[: len(j.outvars)], out_flat[len(j.outvars) :]

    results = tree_unflatten(flat_fn.out_tree, flat_results)
    dresults = tree_unflatten(flat_fn.out_tree, flat_dresults)
    return results, dresults


# pylint: disable=too-many-arguments, too-many-positional-arguments
def jvp(f, params, tangents, method=None, h=None, argnums=None):
    """A :func:`~.qjit` compatible Jacobian-vector product of PennyLane programs.

    This function allows the Jacobian-vector Product of a hybrid quantum-classical function to be
    computed within the compiled program.

    .. warning::

        ``jvp`` is intended to be used with :func:`~.qjit` only.

    .. note::

        When used with :func:`~.qjit`, this function only supports the Catalyst compiler;
        see :func:`catalyst.jvp` for more details.

        Please see the Catalyst :doc:`quickstart guide <catalyst:dev/quick_start>`,
        as well as the :doc:`sharp bits and debugging tips <catalyst:dev/sharp_bits>`
        page for an overview of the differences between Catalyst and PennyLane.

    Args:
        f (Callable): Function-like object to calculate JVP for
        params (List[Array]): List (or a tuple) of the function arguments specifying the point
                              to calculate JVP at. A subset of these parameters are declared as
                              differentiable by listing their indices in the ``argnums`` parameter.
        tangents(List[Array]): List (or a tuple) of tangent values to use in JVP. The list size and
                               shapes must match the ones of differentiable params.
        method(str): Differentiation method to use, same as in :func:`~.grad`.
        h (float): the step-size value for the finite-difference (``"fd"``) method
        argnums (Union[int, List[int]]): the params' indices to differentiate.

    Returns:
        Tuple[Array]: Return values of ``f`` paired with the JVP values.

    Raises:
        TypeError: invalid parameter types
        ValueError: invalid parameter values

    .. seealso:: :func:`~.grad`, :func:`~.vjp`, :func:`~.jacobian`

    **Example 1 (basic usage)**

    .. code-block:: python

        @qml.qjit
        def jvp(params, tangent):
          def f(x):
              y = [jnp.sin(x[0]), x[1] ** 2, x[0] * x[1]]
              return jnp.stack(y)

          return qml.jvp(f, [params], [tangent])

    >>> x = jnp.array([0.1, 0.2])
    >>> tangent = jnp.array([0.3, 0.6])
    >>> jvp(x, tangent)
    (Array([0.09983342, 0.04      , 0.02      ], dtype=float64), Array([0.29850125, 0.24      , 0.12      ], dtype=float64))

    **Example 2 (argnums usage)**

    Here we show how to use ``argnums`` to ignore the non-differentiable parameter ``n`` of the
    target function. Note that the length and shapes of tangents must match the length and shape of
    primal parameters, which we mark as differentiable by passing their indices to ``argnums``.

    .. code-block:: python

        @qml.qjit
        @qml.qnode(qml.device("lightning.qubit", wires=2))
        def circuit(n, params):
            qml.RX(params[n, 0], wires=n)
            qml.RY(params[n, 1], wires=n)
            return qml.expval(qml.Z(1))

        @qml.qjit
        def workflow(primals, tangents):
            return qml.jvp(circuit, [1, primals], [tangents], argnums=[1])

    >>> params = jnp.array([[0.54, 0.3154], [0.654, 0.123]])
    >>> dy = jnp.array([[1.0, 1.0], [1.0, 1.0]])
    >>> workflow(params, dy)
    (Array(0.78766064, dtype=float64), Array(-0.70114352, dtype=float64))
    """
    if capture.enabled():
        return _capture_jvp(f, params, tangents, method=method, h=h, argnums=argnums)

    if capture.enabled():
        return _capture_jvp(f, params, tangents, method=method, h=h, argnums=argnums)

    if active_jit := compiler.active_compiler():
        available_eps = compiler.AvailableCompilers.names_entrypoints
        ops_loader = available_eps[active_jit]["ops"].load()
        return ops_loader.jvp(f, params, tangents, method=method, h=h, argnums=argnums)

    raise CompileError("PennyLane does not support the JVP function without QJIT.")
