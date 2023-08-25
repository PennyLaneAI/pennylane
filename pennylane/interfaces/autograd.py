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
"""
# pylint: disable=too-many-arguments
import logging

import autograd
from autograd.numpy.numpy_boxes import ArrayBox

import pennylane as qml


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def execute(tapes, execute_fn, jpc):
    """Execute a batch of tapes with Autograd parameters on a device.

    Args:
        parameters (list[list[Any]]): Nested list of the quantum tape parameters.
            This argument should be generated from the provided list of tapes.
        tapes (Sequence[.QuantumTape]): batch of tapes to execute
        execute_fn (Callable[[Batch], ResultBatch])
        jpc (JacobianProductCalculator)

    Returns:
        TensorLike: A nested tuple of tape results. Each element in
        the returned tuple corresponds in order to the provided tapes.
    """
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("autograd boundary entry.")
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
    tapes=None,
    execute_fn=None,
    jpc=None,
):  # pylint: disable=dangerous-default-value,unused-argument
    """Autodifferentiable wrapper around ``Device.batch_execute``.

    Args:
        parameters (list[list[Any]]): Nested list of the quantum tape parameters.
            This argument should be generated from the provided list of tapes.
        tapes (Sequence[.QuantumTape]): batch of tapes to execute
        execute_fn (Callable[[Batch], ResultBatch])
        jpc (JacobianProductCalculator)
    """

    return execute_fn(tapes)


def vjp(
    ans,
    parameters,
    tapes=None,
    execute_fn=None,
    jpc=None,
):  # pylint: disable=dangerous-default-value,unused-argument
    """Returns the vector-Jacobian product operator for a batch of quantum tapes.

    Args:
        ans (array): the result of the batch tape execution
        parameters (list[list[Any]]): Nested list of the quantum tape parameters.
            This argument should be generated from the provided list of tapes.
        tapes (Sequence[.QuantumTape]): batch of tapes to execute
        execute_fn (Callable[[Batch], ResultBatch])
        jpc (JacobianProductCalculator)

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
