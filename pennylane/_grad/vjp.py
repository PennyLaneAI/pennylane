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


from pennylane.compiler import compiler
from pennylane.exceptions import CompileError


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
        params(List[Array]): List (or a tuple) of arguments for `f` specifying the point to calculate
                             VJP at. A subset of these parameters are declared as
                             differentiable by listing their indices in the ``argnums`` parameter.
        cotangents(List[Array]): List (or a tuple) of tangent values to use in VJP. The list size
                                 and shapes must match the size and shape of ``f`` outputs.
        method(str): Differentiation method to use, same as in :func:`~.grad`.
        h (float): the step-size value for the finite-difference (``"fd"``) method
        argnums (Union[int, List[int]]): the params' indices to differentiate.

    Returns:
        Tuple[Array]: Return values of ``f`` paired with the VJP values.

    Raises:
        TypeError: invalid parameter types
        ValueError: invalid parameter values

    .. seealso:: :func:`~.grad`, :func:`~.jvp`, :func:`~.jacobian`

    **Example**

    .. code-block:: python

        @qml.qjit
        def vjp(params, cotangent):
          def f(x):
              y = [jnp.sin(x[0]), x[1] ** 2, x[0] * x[1]]
              return jnp.stack(y)

          return qml.vjp(f, [params], [cotangent])

    >>> x = jnp.array([0.1, 0.2])
    >>> dy = jnp.array([-0.5, 0.1, 0.3])
    >>> vjp(x, dy)
    (Array([0.09983342, 0.04      , 0.02      ], dtype=float64), (Array([-0.43750208,  0.07      ], dtype=float64),))
    """

    if active_jit := compiler.active_compiler():
        available_eps = compiler.AvailableCompilers.names_entrypoints
        ops_loader = available_eps[active_jit]["ops"].load()
        return ops_loader.vjp(f, params, cotangents, method=method, h=h, argnums=argnums)

    raise CompileError("Pennylane does not support the VJP function without QJIT.")
