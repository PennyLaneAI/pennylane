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
Defines qp.jvp
"""

import warnings

from pennylane.compiler import compiler
from pennylane.exceptions import CompileError, PennyLaneDeprecationWarning


# pylint: disable=too-many-arguments, too-many-positional-arguments
def jvp(f, params, tangents, method=None, h=None, argnums=None, *, argnum=None):
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

    .. warning::

        The argument ``argnum`` has been renamed to ``argnums`` to match Catalyst and JAX.
        The ability to use ``argnum`` will be removed in v0.45.

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

        @qp.qjit
        def jvp(params, tangent):
          def f(x):
              y = [jnp.sin(x[0]), x[1] ** 2, x[0] * x[1]]
              return jnp.stack(y)

          return qp.jvp(f, [params], [tangent])

    >>> x = jnp.array([0.1, 0.2])
    >>> tangent = jnp.array([0.3, 0.6])
    >>> jvp(x, tangent)
    (Array([0.09983342, 0.04      , 0.02      ], dtype=float64), Array([0.29850125, 0.24      , 0.12      ], dtype=float64))

    **Example 2 (argnums usage)**

    Here we show how to use ``argnums`` to ignore the non-differentiable parameter ``n`` of the
    target function. Note that the length and shapes of tangents must match the length and shape of
    primal parameters, which we mark as differentiable by passing their indices to ``argnums``.

    .. code-block:: python

        @qp.qjit
        @qp.qnode(qp.device("lightning.qubit", wires=2))
        def circuit(n, params):
            qp.RX(params[n, 0], wires=n)
            qp.RY(params[n, 1], wires=n)
            return qp.expval(qp.Z(1))

        @qp.qjit
        def workflow(primals, tangents):
            return qp.jvp(circuit, [1, primals], [tangents], argnums=[1])

    >>> params = jnp.array([[0.54, 0.3154], [0.654, 0.123]])
    >>> dy = jnp.array([[1.0, 1.0], [1.0, 1.0]])
    >>> workflow(params, dy)
    (Array(0.78766064, dtype=float64), Array(-0.70114352, dtype=float64))
    """

    argnums = argnums if argnums is not None else argnum
    if argnum is not None:
        warnings.warn(
            "argnum in qp.jvp has been renamed to argnums to match jax and catalyst.",
            PennyLaneDeprecationWarning,
        )

    if active_jit := compiler.active_compiler():
        available_eps = compiler.AvailableCompilers.names_entrypoints
        ops_loader = available_eps[active_jit]["ops"].load()
        return ops_loader.jvp(f, params, tangents, method=method, h=h, argnums=argnums)

    raise CompileError("Pennylane does not support the JVP function without QJIT.")
