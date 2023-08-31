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
as the first argument to the primitive. Even though the primitive function (``f``/ ``_execute``)
does not use the parameters, that is how we communicate to autograd what parameters the derivatves
belong to.


"""
# pylint: disable=too-many-arguments
import logging
from typing import Tuple, Callable

import autograd
from autograd.numpy.numpy_boxes import ArrayBox

import pennylane as qml

Batch = Tuple[qml.tape.QuantumTape]
ExecuteFn = Callable[[Batch], qml.typing.ResultBatch]

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def autograd_execute(
    tapes: Batch,
    execute_fn: ExecuteFn,
    jpc: qml.interfaces.jacobian_products.JacobianProductCalculator,
):
    """Execute a batch of tapes with Autograd parameters on a device.

    Args:
        tapes (Sequence[.QuantumTape]): batch of tapes to execute
        execute_fn (Callable[[Sequence[.QuantumTape]], ResultBatch]): a function that turns a batch of circuits into results
        jpc (JacobianProductCalculator): a class that can compute the vector jacobian product for the input tapes.

    Returns:
        TensorLike: A nested tuple of tape results. Each element in
        the returned tuple corresponds in order to the provided tapes.
    """
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("Entry with (tapes=%s, execute_fn=%s, jpc=%s)", tapes, execute_fn, jpc)
    # pylint: disable=unused-argument
    for tape in tapes:
        # set the trainable parameters
        params = tape.get_parameters(trainable_only=False)
        tape.trainable_params = qml.math.get_trainable_indices(params)

    # pylint misidentifies autograd.builtins as a dict
    # pylint: disable=no-member
    parameters = autograd.builtins.tuple(
        [autograd.builtins.list(t.get_parameters()) for t in tapes]
    )
    return _execute(parameters, tapes, execute_fn, jpc)


@autograd.extend.primitive
def _execute(
    parameters,
    tapes,
    execute_fn,
    jpc,
):  # pylint: disable=dangerous-default-value,unused-argument
    """Autodifferentiable wrapper around ``Device.batch_execute``.

    Args:
        parameters (list[list[Any]]): Nested list of the quantum tape parameters.
            This argument should be generated from the provided list of tapes.
        tapes (Sequence[.QuantumTape]): batch of tapes to execute
        execute_fn (Callable[[Sequence[.QuantumTape]], ResultBatch]): a function that turns a batch of circuits into results
        jpc (JacobianProductCalculator): a class that can compute the vector jacobian product for the input tapes.
    """
    return execute_fn(tapes)


# pylint: disable=dangerous-default-value,unused-argument
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
        jpc (JacobianProductCalculator): a class that can compute the vector jacobian product for the input tapes.

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
