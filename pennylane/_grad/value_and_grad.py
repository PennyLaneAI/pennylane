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

from pennylane.compiler import compiler
from pennylane.exceptions import CompileError


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

        raise CompileError("PennyLane does not support the value_and_grad function without QJIT.")
