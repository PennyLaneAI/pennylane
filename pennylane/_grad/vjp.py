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
Defines qml.vjp
"""
from functools import lru_cache
from importlib.util import find_spec

from pennylane import capture
from pennylane.compiler import compiler
from pennylane.exceptions import CompileError

from .grad import _args_and_argnums, _setup_h, _setup_method

has_jax = find_spec("qpjax") is not None


# pylint: disable=unused-argument
@lru_cache
def _get_vjp_prim():
    if not has_jax:  # pragma: no cover
        return None

    import qpjax  # pylint: disable=import-outside-toplevel

    vjp_prim = capture.QmlPrimitive("vjp")
    vjp_prim.multiple_results = True
    vjp_prim.prim_type = "higher_order"

    @vjp_prim.def_impl
    def _vjp_impl(*args, jaxpr, fn, method, h, argnums):
        params = args[: len(jaxpr.invars)]
        dy = list(args[len(jaxpr.invars) :])

        def func(*inner_args):
            return qpjax.core.eval_jaxpr(jaxpr, [], *inner_args)

        res, vjp_fn = qpjax.vjp(func, *params)
        dparams = vjp_fn(dy)
        return res + [dparams[i] for i in argnums]

    @vjp_prim.def_abstract_eval
    def _vjp_abstract_eval(*args, jaxpr, fn, method, h, argnums):
        return [v.aval for v in jaxpr.outvars] + [jaxpr.invars[i].aval for i in argnums]

    return vjp_prim


def _validate_cotangents(cotangents, out_avals):
    import qpjax  # pylint: disable=import-outside-toplevel
    from qpjax._src.api import _dtype  # pylint: disable=import-outside-toplevel

    def get_shape(x):
        return getattr(x, "shape", qpjax.numpy.shape(x))

    if len(cotangents) != len(out_avals):
        raise ValueError(
            "The length of cotangents must match the number of"
            " outputs of the function with qml.vjp."
        )
    for p, t in zip(cotangents, out_avals):
        if _dtype(p) != _dtype(t):
            raise TypeError(
                "function output params and cotangents arguments to qml.vjp do not match; "
                "dtypes must be equal. "
                f"Got function output params dtype {_dtype(p)} and expected matching cotangent dtype, "
                f"but got cotangent dtype {_dtype(t)} instead."
            )

        if get_shape(p) != get_shape(t):
            raise ValueError(
                "qml.vjp called with different function output params and cotangent "
                f"shapes; got function output params shape {get_shape(p)} and cotangent shape "
                f"{get_shape(t)}"
            )


# pylint: disable=too-many-arguments
def _capture_vjp(func, params, cotangents, *, argnums=None, method=None, h=None):
    import qpjax  # pylint: disable=import-outside-toplevel
    from qpjax.tree_util import tree_leaves, tree_unflatten  # pylint: disable=import-outside-toplevel

    h = _setup_h(h)
    method = _setup_method(method)
    flat_args, flat_argnums, _, trainable_in_tree = _args_and_argnums(params, argnums)
    flat_cotangents = tree_leaves(cotangents)
    flat_fn = capture.FlatFn(func)
    jaxpr = qpjax.make_jaxpr(flat_fn)(*params)
    j = jaxpr.jaxpr
    no_consts_jaxpr = j.replace(constvars=(), invars=j.constvars + j.invars)
    shifted_argnums = tuple(i + len(jaxpr.consts) for i in flat_argnums)

    _validate_cotangents(flat_cotangents, jaxpr.out_avals)

    prim_kwargs = {
        "fn": func,
        "method": method,
        "h": h,
        "argnums": shifted_argnums,
        "jaxpr": no_consts_jaxpr,
    }
    out_flat = _get_vjp_prim().bind(*jaxpr.consts, *flat_args, *flat_cotangents, **prim_kwargs)
    assert flat_fn.out_tree is not None, "out_tree should be set after executing flat_fn"
    num_outputs = len(no_consts_jaxpr.outvars)
    flat_results = out_flat[:num_outputs]
    flat_dparams = out_flat[num_outputs:]

    results = tree_unflatten(flat_fn.out_tree, flat_results)
    dparams = tree_unflatten(trainable_in_tree, flat_dparams)
    return results, dparams


# pylint: disable=too-many-arguments, too-many-positional-arguments
def vjp(f, params, cotangents, method=None, h=None, argnums=None):
    """A :func:`~.qjit` compatible Vector-Jacobian product of PennyLane programs.

    This function allows the Vector-Jacobian Product of a hybrid quantum-classical function to be
    computed within the compiled program.

    .. warning::

        ``vjp`` is intended to be used with :func:`~.qjit` only.

    .. note::

        When used with :func:`~.qjit`, this function only supports the Catalyst compiler.
        See :func:`catalyst.vjp` for more details.

        Please see the Catalyst :doc:`quickstart guide <catalyst:dev/quick_start>`,
        as well as the :doc:`sharp bits and debugging tips <catalyst:dev/sharp_bits>`
        page for an overview of the differences between Catalyst and PennyLane.

    Args:
        f(Callable): Function-like object to calculate VJP for
        params(Sequence[Pytree[Array]]): List (or a tuple) of arguments for `f` specifying the point to calculate
                             VJP at. A subset of these parameters are declared as
                             differentiable by listing their indices in the ``argnums`` parameter.
        cotangents(Pytree[Array]): Cotangent values to use in VJP. Should match the pytree
            structure of the functions output.
        method(str): Differentiation method to use, same as in :func:`~.grad`.
        h (float): the step-size value for the finite-difference (``"fd"``) method
        argnums (Union[int, List[int]]): the params' indices to differentiate.

    Returns:
        Tuple[Array]: Return values of ``f`` paired with the VJP values.

    .. seealso:: :func:`~.grad`, :func:`~.jvp`, :func:`~.jacobian`

    .. note::

        While ``qpjax.vjp`` has no ``argnums`` and treats all params as trainable as default, we
        default to only the first argument as trainable by default.

    **Example**

    .. code-block:: python

        @qml.qjit(static_argnames="argnums")
        def calculate_vjp_qjit(x, y, cotangent, argnums):
          def f(x, y):
              return x * y

          return qml.vjp(f, (x, y), cotangent, argnums=argnums)

    >>> params = (jnp.array([1.0, 2.0]), jnp.array([2.0, 3.0]))
    >>> dy = jnp.array([10.0, 20.0])
    >>> results, dparams = calculate_vjp_qjit(*params, dy, 0)
    >>> results
    Array([2., 6.], dtype=float64)
    >>> dparams # doctest: +SKIP
    Array([20., 60.], dtype=float64)

    Similar to ``grad`` and ``jacobian``, if ``argnums`` is an array, the ``dparams``
    gains an additional dimension that is squeezed out when ``argnums`` is an integer:

    >>> calculate_vjp_qjit(*params, dy, (0,))[1]
    (Array([20., 60.], dtype=float64),)


    """

    if capture.enabled():
        return _capture_vjp(f, params, cotangents, argnums=argnums, method=method, h=h)

    if active_jit := compiler.active_compiler():
        available_eps = compiler.AvailableCompilers.names_entrypoints
        ops_loader = available_eps[active_jit]["ops"].load()
        return ops_loader.vjp(f, params, cotangents, method=method, h=h, argnums=argnums)

    raise CompileError("PennyLane does not support the VJP function without QJIT.")
