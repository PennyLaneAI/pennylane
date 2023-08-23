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
import inspect
import logging

import numpy as np
import tensorflow as tf
from tensorflow.python.eager import context

import pennylane as qml
from pennylane.interfaces import InterfaceUnsupportedError
from pennylane.measurements import CountsMP, Shots
from pennylane.transforms import convert_to_numpy_parameters

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def _set_copy_and_unwrap_tape(t, a, unwrap=True):
    """Copy a given tape with operations and set parameters"""
    tc = t.bind_new_parameters(a, list(range(len(a))))
    return convert_to_numpy_parameters(tc) if unwrap else tc


def set_parameters_on_copy_and_unwrap(tapes, params, unwrap=True):
    """Copy a set of tapes with operations and set parameters"""
    return tuple(_set_copy_and_unwrap_tape(t, a, unwrap=unwrap) for t, a in zip(tapes, params))


def _compute_vjp_legacy(dy, jacs):
    # compute the vector-Jacobian product dy @ jac
    # for a list of dy's and Jacobian matrices.
    vjps = []

    for d, jac in zip(dy, jacs):
        vjp = qml.gradients.compute_vjp(d, jac)

        if not context.executing_eagerly():
            vjp = qml.math.unstack(vjp)

        vjps.extend(vjp)

    return vjps


def _compute_vjp(dy, jacs, multi_measurements, has_partitioned_shots):
    # compute the vector-Jacobian product dy @ jac
    # for a list of dy's and Jacobian matrices.

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "Entry with args=(dy=%s, jacs=%s, multi_measurements=%s, shots=%s) called by=%s",
            dy,
            jacs,
            multi_measurements,
            has_partitioned_shots,
            "::L".join(str(i) for i in inspect.getouterframes(inspect.currentframe(), 2)[1][1:3]),
        )

    vjps = []

    for dy_, jac_, multi in zip(dy, jacs, multi_measurements):
        dy_ = dy_ if has_partitioned_shots else (dy_,)
        jac_ = jac_ if has_partitioned_shots else (jac_,)

        shot_vjps = []
        for d, j in zip(dy_, jac_):
            if multi:
                shot_vjps.append(qml.gradients.compute_vjp_multi(d, j))
            else:
                shot_vjps.append(qml.gradients.compute_vjp_single(d, j))

        vjp = qml.math.sum(qml.math.stack(shot_vjps), 0)

        if not context.executing_eagerly():
            vjp = qml.math.unstack(vjp)

        vjps.extend(vjp)

    return vjps


def _to_tensors(x):
    """
    Convert a nested tuple structure of arrays into a nested tuple
    structure of TF tensors
    """
    if isinstance(x, dict) or isinstance(x, list) and all(isinstance(i, dict) for i in x):
        # qml.counts returns a dict (list of dicts when broadcasted), can't form a valid tensor
        return x

    if isinstance(x, tuple):
        return tuple(_to_tensors(x_) for x_ in x)

    return tf.convert_to_tensor(x)


def _res_restructured(res, tapes):
    """
    Reconstruct the nested tuple structure of the output of a list of tapes
    """
    start = 0
    res_nested = []
    for tape in tapes:
        tape_shots = tape.shots or Shots(1)
        shot_res_nested = []
        num_meas = len(tape.measurements)

        for _ in range(tape_shots.num_copies):
            shot_res = tuple(res[start : start + num_meas])
            shot_res_nested.append(shot_res[0] if num_meas == 1 else shot_res)
            start += num_meas

        res_nested.append(
            tuple(shot_res_nested) if tape_shots.has_partitioned_shots else shot_res_nested[0]
        )

    return tuple(res_nested)


def _jac_restructured(jacs, tapes):
    """
    Reconstruct the nested tuple structure of the jacobian of a list of tapes
    """
    start = 0
    jacs_nested = []
    for tape in tapes:
        num_meas = len(tape.measurements)
        num_params = len(tape.trainable_params)

        tape_jacs = tuple(jacs[start : start + num_meas * num_params])
        tape_jacs = tuple(
            tuple(tape_jacs[i * num_params : (i + 1) * num_params]) for i in range(num_meas)
        )

        while isinstance(tape_jacs, tuple) and len(tape_jacs) == 1:
            tape_jacs = tape_jacs[0]

        jacs_nested.append(tape_jacs)
        start += num_meas * num_params

    return tuple(jacs_nested)


def _execute_legacy(
    tapes, device, execute_fn, gradient_fn, gradient_kwargs, _n=1, max_diff=2, mode=None
):
    """Execute a batch of tapes with TensorFlow parameters on a device.

    Args:
        tapes (Sequence[.QuantumTape]): batch of tapes to execute
        device (pennylane.Device): Device to use to execute the batch of tapes.
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
        mode (str): Whether the gradients should be computed on the forward
            pass (``forward``) or the backward pass (``backward``).

    Returns:
        list[list[tf.Tensor]]: A nested list of tape results. Each element in
        the returned list corresponds in order to the provided tapes.
    """
    # pylint: disable=unused-argument

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

    unwrapped_tapes = tuple(convert_to_numpy_parameters(t) for t in tapes)
    # Forward pass: execute the tapes
    res, jacs = execute_fn(unwrapped_tapes, **gradient_kwargs)

    for i, tape in enumerate(tapes):
        # convert output to TensorFlow tensors

        if any(isinstance(m, CountsMP) for m in tape.measurements):
            if tape.batch_size is not None:
                raise InterfaceUnsupportedError(
                    "Broadcasted circuits with counts return types are only supported with "
                    "the new return system. Use qml.enable_return() to turn it on."
                )
            continue

        if isinstance(res[i], np.ndarray):
            # For backwards compatibility, we flatten ragged tape outputs
            # when there is no sampling
            r = np.hstack(res[i]) if res[i].dtype == np.dtype("object") else res[i]
            res[i] = tf.convert_to_tensor(r)

        elif isinstance(res[i], tuple):
            res[i] = tuple(tf.convert_to_tensor(r) for r in res[i])

        else:
            res[i] = tf.convert_to_tensor(qml.math.toarray(res[i]))

    @tf.custom_gradient
    def _execute(*parameters):  # pylint:disable=unused-argument
        def grad_fn(*dy, **tfkwargs):
            """Returns the vector-Jacobian product with given
            parameter values and output gradient dy"""

            dy = [qml.math.T(d) for d in dy]

            if jacs:
                # Jacobians were computed on execution
                # No additional quantum evaluations needed; simply compute the VJPs directly.
                vjps = _compute_vjp_legacy(dy, jacs)

            else:
                # Need to compute the Jacobians on the backward pass (accumulation="backward")

                if isinstance(gradient_fn, qml.gradients.gradient_transform):
                    # Gradient function is a gradient transform.

                    # Generate and execute the required gradient tapes
                    if _n == max_diff or not context.executing_eagerly():
                        new_tapes = set_parameters_on_copy_and_unwrap(tapes, params_unwrapped)
                        vjp_tapes, processing_fn = qml.gradients.batch_vjp(
                            new_tapes,
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
                            _execute_legacy(
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
                    new_tapes = set_parameters_on_copy_and_unwrap(tapes, params_unwrapped)
                    vjps = _compute_vjp_legacy(dy, gradient_fn(new_tapes, **gradient_kwargs))

            variables = tfkwargs.get("variables")
            return (vjps, variables) if variables is not None else vjps

        return res, grad_fn

    return _execute(*parameters)


def execute(tapes, device, execute_fn, gradient_fn, gradient_kwargs, _n=1, max_diff=2):
    """Execute a batch of tapes with TensorFlow parameters on a device.

    Args:
        tapes (Sequence[.QuantumTape]): batch of tapes to execute
        device (pennylane.Device): Device to use to execute the batch of tapes.
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
    # pylint: disable=unused-argument

    parameters = []
    params_unwrapped = []

    # assumes all tapes have the same shot vector
    has_partitioned_shots = tapes[0].shots.has_partitioned_shots

    for i, tape in enumerate(tapes):
        # store the trainable parameters
        params = tape.get_parameters(trainable_only=False)
        tape.trainable_params = qml.math.get_trainable_indices(params)

        parameters += [p for i, p in enumerate(params) if i in tape.trainable_params]

        # store all unwrapped parameters
        params_unwrapped.append(
            [i.numpy() if isinstance(i, (tf.Variable, tf.Tensor)) else i for i in params]
        )
    res, jacs = execute_fn(tapes, **gradient_kwargs)
    res = tuple(_to_tensors(r) for r in res)  # convert output to TensorFlow tensors

    @tf.custom_gradient
    def _execute(*parameters):  # pylint:disable=unused-argument
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Entry with args=(parameters=%s) called by=%s",
                parameters,
                "::L".join(
                    str(i) for i in inspect.getouterframes(inspect.currentframe(), 2)[1][1:3]
                ),
            )

        def grad_fn(*dy, **tfkwargs):
            """Returns the vector-Jacobian product with given
            parameter values and output gradient dy"""
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "Entry with args=(dy=%s, tfkwargs=%s) called by=%s",
                    dy,
                    tfkwargs,
                    "::L".join(
                        str(i) for i in inspect.getouterframes(inspect.currentframe(), 2)[1][1:3]
                    ),
                )

            # whether the tapes contain multiple measurements
            multi_measurements = [len(tape.measurements) > 1 for tape in tapes]

            # reconstruct the nested structure of dy
            dy = _res_restructured(dy, tapes)

            if jacs:
                # Jacobians were computed on execution
                # No additional quantum evaluations needed; simply compute the VJPs directly.
                vjps = _compute_vjp(dy, jacs, multi_measurements, has_partitioned_shots)

            else:
                # Need to compute the Jacobians on the backward pass (accumulation="backward")

                if isinstance(gradient_fn, qml.gradients.gradient_transform):
                    # Gradient function is a gradient transform.

                    # Generate and execute the required gradient tapes
                    if _n == max_diff or not context.executing_eagerly():
                        new_tapes = set_parameters_on_copy_and_unwrap(
                            tapes, params_unwrapped, unwrap=False
                        )
                        vjp_tapes, processing_fn = qml.gradients.batch_vjp(
                            new_tapes,
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
                    new_tapes = set_parameters_on_copy_and_unwrap(
                        tapes, params_unwrapped, unwrap=False
                    )
                    jac = gradient_fn(new_tapes, **gradient_kwargs)

                    vjps = _compute_vjp(dy, jac, multi_measurements, has_partitioned_shots)

            # filter out untrainable parameters if they happen to appear in the vjp
            vjps = [vjp for vjp in vjps if 0 not in qml.math.shape(vjp)]

            variables = tfkwargs.get("variables")
            return (vjps, variables) if variables is not None else vjps

        return res, grad_fn

    return _execute(*parameters)
