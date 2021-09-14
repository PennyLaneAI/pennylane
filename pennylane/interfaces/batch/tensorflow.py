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
This module contains functions for adding the TensorFlow interface
to a PennyLane Device class.
"""
# pylint: disable=too-many-arguments,too-many-branches
import numpy as np
import tensorflow as tf
from tensorflow.python.eager import context

import pennylane as qml


def _compute_vjp(dy, jacs):
    # compute the vector-Jacobian product dy @ jac
    # for a list of dy's and Jacobian matrices.
    vjps = []

    for d, jac in zip(dy, jacs):
        vjp = qml.gradients.compute_vjp(d, jac)

        if not context.executing_eagerly():
            vjp = qml.math.unstack(vjp)

        vjps.extend(vjp)

    return vjps


def execute(tapes, device, execute_fn, gradient_fn, gradient_kwargs, _n=1, max_diff=2):
    """Execute a batch of tapes with TensorFlow parameters on a device.

    Args:
        tapes (Sequence[.QuantumTape]): batch of tapes to execute
        device (.Device): Device to use to execute the batch of tapes.
            If the device does not provide a ``batch_execute`` method,
            by default the tapes will be executed in serial.
        execute_fn (callable): The execution function used to execute the tapes
            during the forward pass. This function must return a tuple ``(results, jacobians)``.
            If ``jacobians`` is an empty list, then ``gradient_fn`` is used to
            compute the gradients during the backwards pass.
        gradient_kwargs (dict): dictionary of keyword arguments to pass when
            determining the gradients of tapes
        gradient_fn (callable): the gradient function to use to compute quantum gradients
        _n (int): a positive integer used to track nesting of derivatives, for example
            if the nth-order derivative is requested.
        max_diff (int): If ``gradient_fn`` is a gradient transform, this option specifies
            the maximum number of derivatives to support. Increasing this value allows
            for higher order derivatives to be extracted, at the cost of additional
            (classical) computational overhead during the backwards pass.

    Returns:
        list[list[tf.Tensor]]: A nested list of tape results. Each element in
        the returned list corresponds in order to the provided tapes.
    """

    parameters = []
    params_unwrapped = []

    for i, tape in enumerate(tapes):
        # store the trainable parameters
        params = tape.get_parameters(trainable_only=False)
        tape.trainable_params = qml.math.get_trainable_indices(params)

        parameters += [p for i, p in enumerate(params) if i in tape.trainable_params]

        # store all unwrapped parameters
        params_unwrapped.append(
            [i.numpy() if isinstance(i, (tf.Variable, tf.Tensor)) else i for i in params]
        )

    with qml.tape.Unwrap(*tapes, set_trainable=False):
        # Forward pass: execute the tapes
        res, jacs = execute_fn(tapes, **gradient_kwargs)

    for i, tape in enumerate(tapes):
        # convert output to TensorFlow tensors
        r = np.hstack(res[i]) if res[i].dtype == np.dtype("object") else res[i]
        res[i] = tf.convert_to_tensor(r)

    @tf.custom_gradient
    def _execute(*parameters):  # pylint:disable=unused-argument
        def grad_fn(*dy, **tfkwargs):
            """Returns the vector-Jacobian product with given
            parameter values and output gradient dy"""

            dy = [qml.math.T(d) for d in dy]

            if jacs:
                # Jacobians were computed on the forward pass (mode="forward")
                # No additional quantum evaluations needed; simply compute the VJPs directly.
                vjps = _compute_vjp(dy, jacs)

            else:
                # Need to compute the Jacobians on the backward pass (accumulation="backward")

                if isinstance(gradient_fn, qml.gradients.gradient_transform):
                    # Gradient function is a gradient transform.

                    # Generate and execute the required gradient tapes
                    if _n == max_diff or not context.executing_eagerly():

                        with qml.tape.Unwrap(*tapes, params=params_unwrapped, set_trainable=False):
                            vjp_tapes, processing_fn = qml.gradients.batch_vjp(
                                tapes,
                                dy,
                                gradient_fn,
                                reduction=lambda vjps, x: vjps.extend(qml.math.unstack(x)),
                                gradient_kwargs=gradient_kwargs,
                            )

                            vjps = processing_fn(execute_fn(vjp_tapes)[0])

                    else:
                        vjp_tapes, processing_fn = qml.gradients.batch_vjp(
                            tapes,
                            dy,
                            gradient_fn,
                            reduction="extend",
                            gradient_kwargs=gradient_kwargs,
                        )

                        # This is where the magic happens. Note that we call ``execute``.
                        # This recursion, coupled with the fact that the gradient transforms
                        # are differentiable, allows for arbitrary order differentiation.
                        vjps = processing_fn(
                            execute(
                                vjp_tapes,
                                device,
                                execute_fn,
                                gradient_fn,
                                gradient_kwargs,
                                _n=_n + 1,
                                max_diff=max_diff,
                            )
                        )

                else:
                    # Gradient function is not a gradient transform
                    # (e.g., it might be a device method).
                    # Note that unlike the previous branch:
                    #
                    # - there is no recursion here
                    # - gradient_fn is not differentiable
                    #
                    # so we cannot support higher-order derivatives.
                    with qml.tape.Unwrap(*tapes, params=params_unwrapped, set_trainable=False):
                        vjps = _compute_vjp(dy, gradient_fn(tapes, **gradient_kwargs))

            variables = tfkwargs.get("variables", None)
            return (vjps, variables) if variables is not None else vjps

        return res, grad_fn

    return _execute(*parameters)
