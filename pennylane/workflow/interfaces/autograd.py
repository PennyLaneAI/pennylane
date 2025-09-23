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
This module contains functions for adding the Autograd interface
to a PennyLane Device class.

**How to bind a custom derivative with autograd.**

Suppose I have a function ``f`` that I want to change how autograd takes the derivative of.

I need to:

1) Mark it as an autograd primitive with ``@autograd.extend.primitive``
2) Register its VJP with ``autograd.extend.defvjp``

.. code-block:: python

    @autograd.extend.primitive
    def f(x, exponent=2):
        return x**exponent

    def vjp(ans, x, exponent=2):
        def grad_fn(dy):
            print(f"Calculating the gradient with {x}, {dy}")
            return dy * exponent * x**(exponent-1)
        return grad_fn

    autograd.extend.defvjp(f, vjp, argnums=[0])


>>> autograd.grad(f)(autograd.numpy.array(2.0))
Calculating the gradient with 2.0, 1.0
4.0

The above code told autograd how to differentiate the first argument of ``f``.

We have an additional problem that autograd does not understand that a :class:`~.QuantumTape`
contains parameters we want to differentiate. So in order to match the ``vjp`` function with
the correct parameters, we need to extract them from the batch of tapes, and pass them as is
as the first argument to the primitive. Even though the primitive function
does not use the parameters, that is how we communicate to autograd what parameters the derivatives
belong to.

**Jacobian Calculations and the need for caching:**

Suppose we use the above function with an array and take the jacobian:

>>> x = autograd.numpy.array([1.0, 2.0])
>>> autograd.jacobian(f)(x)
Calculating the gradient with [1. 2.], [1. 0.]
Calculating the gradient with [1. 2.], [0. 1.]
array([[2., 0.],
       [0., 4.]])

Here, the ``grad_fn`` was called once for each output quantity. Each time ``grad_fn``
is called, we are forced to reproduce the calculation for ``exponent * x ** (exponent-1)``,
only to multiply it by a different vector. When executing quantum circuits, that quantity
can potentially be quite expensive. Autograd would naively
request independent vjps for each entry in the output, even though the internal circuits will be
exactly the same.

When caching is enabled, the expensive part (re-executing identical circuits) is
avoided, but when normal caching is turned off, the above can lead to an explosion
in the number of required circuit executions.

To avoid this explosion in the number of executed circuits when caching is turned off, we will instead internally
cache the full jacobian so that is is reused between different calls to the same ``grad_fn``. This behaviour is toggled
by the ``cache_full_jacobian`` keyword argument to :class:`~.TransformJacobianProducts`.

Other interfaces are capable of calculating the full jacobian in one call, so this patch is only present for autograd.

"""

import logging
from collections.abc import Callable

import autograd
from autograd.numpy.numpy_boxes import ArrayBox

import pennylane as qml
from pennylane.tape import QuantumScriptBatch

ExecuteFn = Callable[[QuantumScriptBatch], qml.typing.ResultBatch]

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# pylint: disable=unused-argument
def autograd_execute(
    tapes: QuantumScriptBatch,
    execute_fn: ExecuteFn,
    jpc: qml.workflow.jacobian_products.JacobianProductCalculator,
    device=None,
):
    """Execute a batch of tapes with Autograd parameters on a device.

    Args:
        tapes (Sequence[.QuantumTape]): batch of tapes to execute
        execute_fn (Callable[[Sequence[.QuantumTape]], ResultBatch]): a function that turns a batch of circuits into results
        jpc (JacobianProductCalculator): a class that can compute the vector Jacobian product (VJP)
            for the input tapes.

    Returns:
        TensorLike: A nested tuple of tape results. Each element in
        the returned tuple corresponds in order to the provided tapes.

    **Example:**

    >>> from pennylane.workflow.jacobian_products import DeviceDerivatives
    >>> from pennylane.workflow.autograd import autograd_execute
    >>> execute_fn = qml.device('default.qubit').execute
    >>> config = qml.devices.ExecutionConfig(gradient_method="adjoint", use_device_gradient=True)
    >>> jpc = DeviceDerivatives(qml.device('default.qubit'), config)
    >>> def f(x):
    ...     tape = qml.tape.QuantumScript([qml.RX(x, 0)], [qml.expval(qml.Z(0))])
    ...     batch = (tape, )
    ...     return autograd_execute(batch, execute_fn, jpc)
    >>> qml.grad(f)(qml.numpy.array(0.1))
    -0.09983341664682815

    """
    tapes = tuple(tapes)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("Entry with (tapes=%s, execute_fn=%s, jpc=%s)", tapes, execute_fn, jpc)
    for tape in tapes:
        # set the trainable parameters
        params = tape.get_parameters(trainable_only=False)
        tape.trainable_params = qml.math.get_trainable_indices(params)

    parameters = autograd.builtins.tuple(
        [autograd.builtins.list(t.get_parameters()) for t in tapes]
    )
    return _execute(parameters, tuple(tapes), execute_fn, jpc)


def _to_autograd(result: qml.typing.ResultBatch) -> qml.typing.ResultBatch:
    """Converts an arbitrary result batch to one with autograd arrays.
    Args:
        result (ResultBatch): a nested structure of lists, tuples, dicts, and numpy arrays
    Returns:
        ResultBatch: a nested structure of tuples, dicts, and jax arrays
    """
    if isinstance(result, dict):
        return result
    if isinstance(result, (list, tuple, autograd.builtins.tuple, autograd.builtins.list)):
        return tuple(_to_autograd(r) for r in result)
    return autograd.numpy.array(result)


@autograd.extend.primitive
def _execute(
    parameters,
    tapes,
    execute_fn,
    jpc,
):
    """Autodifferentiable wrapper around a way of executing tapes.

    Args:
        parameters (list[list[Any]]): Nested list of the quantum tape parameters.
            This argument should be generated from the provided list of tapes.
        tapes (Sequence[.QuantumTape]): batch of tapes to execute
        execute_fn (Callable[[Sequence[.QuantumTape]], ResultBatch]): a function that turns a batch of circuits into results
        jpc (JacobianProductCalculator): a class that can compute the vector Jacobian product (VJP)
            for the input tapes.

    """
    return _to_autograd(execute_fn(tapes))


# pylint: disable=unused-argument
def vjp(
    ans,
    parameters,
    tapes,
    execute_fn,
    jpc,
):
    """Returns the vector-Jacobian product operator for a batch of quantum tapes.

    Args:
        ans (array): the result of the batch tape execution
        parameters (list[list[Any]]): Nested list of the quantum tape parameters.
            This argument should be generated from the provided list of tapes.
        tapes (Sequence[.QuantumTape]): batch of tapes to execute
        execute_fn (Callable[[Sequence[.QuantumTape]], ResultBatch]): a function that turns a batch of circuits into results
        jpc (JacobianProductCalculator): a class that can compute the vector Jacobian product (VJP)
            for the input tapes.


    Returns:
        function: this function accepts the backpropagation
        gradient output vector, and computes the vector-Jacobian product
    """

    def grad_fn(dy):
        """Returns the vector-Jacobian product with given
        parameter values and output gradient dy"""
        vjps = jpc.compute_vjp(tapes, dy)
        return tuple(
            qml.math.to_numpy(v, max_depth=1) if isinstance(v, ArrayBox) else v for v in vjps
        )

    return grad_fn


autograd.extend.defvjp(_execute, vjp, argnums=[0])
