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
This module contains functions for adding the TensorFlow Autograph interface
to a PennyLane Device class.
"""
# pylint: disable=too-many-arguments,too-many-branches,too-many-statements
from functools import reduce

import numpy as np
import tensorflow as tf

import pennylane as qml
from pennylane.measurements import SampleMP, StateMP

from .tensorflow import (
    _compute_vjp,
    _compute_vjp_legacy,
    _jac_restructured,
    _res_restructured,
    _to_tensors,
    set_parameters_on_copy_and_unwrap,
)


def _flatten_nested_list(x):
    """
    Recursively flatten the list
    """
    if not isinstance(x, (tuple, list)):
        return [x]

    return reduce(lambda a, y: a + _flatten_nested_list(y), x, [])


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

    all_params = []
    parameters = []
    lens = []
    trainable = []
    output_types = []

    for tape in tapes:
        # store the trainable parameters
        params = tape.get_parameters(trainable_only=False)
        tape.trainable_params = qml.math.get_trainable_indices(params)

        parameters += [p for i, p in enumerate(params) if i in tape.trainable_params]
        all_params += params
        trainable += (np.array(list(tape.trainable_params)) + sum(lens)).tolist()

        lens.append(len(params))

        if tape.all_sampled:
            output_types.append(tf.int64)
        elif isinstance(tape.measurements[0], StateMP):
            output_types.append(tf.complex128)
        else:
            output_types.append(tf.float64)

    if mode == "forward":
        output_types += [tf.float64] * len(tapes)

    output_types += [tf.int32] * len(tapes)

    def _nest_params(all_params):
        count = 0
        params_unwrapped = []

        for s in lens:
            params_unwrapped.append(all_params[count : count + s])
            count += s

        return params_unwrapped

    def _forward(*all_params):
        params_unwrapped = _nest_params(all_params)
        output_sizes = []

        new_tapes = set_parameters_on_copy_and_unwrap(tapes, params_unwrapped)
        # Forward pass: execute the tapes
        res, jacs = execute_fn(new_tapes, **gradient_kwargs)

        for i, _ in enumerate(tapes):
            # convert output to TensorFlow tensors

            # For backwards compatibility, we flatten ragged tape outputs
            # when there is no sampling
            r = np.hstack(res[i]) if res[i].dtype == np.dtype("object") else res[i]

            res[i] = tf.convert_to_tensor(r)
            output_sizes.append(tf.size(res[i]))

        return res + jacs + output_sizes

    @tf.custom_gradient
    def _execute(*all_params):  # pylint:disable=unused-argument
        res = tf.numpy_function(func=_forward, inp=all_params, Tout=output_types)
        output_sizes = res[-len(tapes) :]

        if mode == "forward":
            jacs = res[len(tapes) : 2 * len(tapes)]

        res = res[: len(tapes)]

        def grad_fn(*dy, **tfkwargs):
            """Returns the vector-Jacobian product with given
            parameter values and output gradient dy"""

            dy = [qml.math.T(d) for d in dy[: len(res)]]

            if mode == "forward":
                # Jacobians were computed on the forward pass (mode="forward")
                # No additional quantum evaluations needed; simply compute the VJPs directly.
                len_dy = len(dy)
                vjps = tf.numpy_function(
                    func=lambda *args: _compute_vjp_legacy(args[:len_dy], args[len_dy:]),
                    inp=dy + jacs,
                    Tout=[tf.float64] * len(parameters),
                )

            else:
                # Need to compute the Jacobians on the backward pass (accumulation="backward")
                if isinstance(gradient_fn, qml.gradients.gradient_transform):
                    # Gradient function is a gradient transform.

                    # Generate and execute the required gradient tapes
                    if _n == max_diff:
                        len_all_params = len(all_params)

                        def _backward(*all_params):
                            dy = all_params[len_all_params:]
                            all_params = all_params[:len_all_params]
                            params_unwrapped = _nest_params(all_params)

                            new_tapes = set_parameters_on_copy_and_unwrap(tapes, params_unwrapped)
                            vjp_tapes, processing_fn = qml.gradients.batch_vjp(
                                new_tapes,
                                dy,
                                gradient_fn,
                                reduction=lambda vjps, x: vjps.extend(qml.math.unstack(x)),
                                gradient_kwargs=gradient_kwargs,
                            )

                            return processing_fn(execute_fn(vjp_tapes)[0])

                        vjps = tf.py_function(
                            func=_backward,
                            inp=list(all_params) + dy,
                            Tout=[tf.float64] * len(parameters),
                        )

                    else:
                        vjp_tapes, processing_fn = qml.gradients.batch_vjp(
                            tapes,
                            dy,
                            gradient_fn,
                            reduction="append",
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
                                mode=mode,
                            ),
                            nums=output_sizes,
                        )

                        vjps = tf.unstack(tf.concat(vjps, 0), num=len(parameters))

                else:
                    # Gradient function is not a gradient transform
                    # (e.g., it might be a device method).
                    # Note that unlike the previous branch:
                    #
                    # - there is no recursion here
                    # - gradient_fn is not differentiable
                    #
                    # so we cannot support higher-order derivatives.
                    len_all_params = len(all_params)

                    def _backward(*all_params):
                        dy = all_params[len_all_params:]
                        all_params = all_params[:len_all_params]
                        params_unwrapped = _nest_params(all_params)

                        new_tapes = set_parameters_on_copy_and_unwrap(tapes, params_unwrapped)
                        return _compute_vjp_legacy(dy, gradient_fn(new_tapes, **gradient_kwargs))

                    vjps = tf.numpy_function(
                        func=_backward,
                        inp=list(all_params) + dy,
                        Tout=[tf.float64] * len(parameters),
                    )

            vjps = iter(vjps)
            vjps = [next(vjps) if x in trainable else None for x in range(len(all_params))]

            variables = tfkwargs.get("variables")
            return (vjps, variables) if variables is not None else vjps

        return res, grad_fn

    return _execute(*all_params)


def execute(
    tapes,
    device,
    execute_fn,
    gradient_fn,
    gradient_kwargs,
    _n=1,
    max_diff=2,
    grad_on_execution=None,
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
        grad_on_execution (bool): Whether the gradients should be computed on execution or not.

    Returns:
        list[list[tf.Tensor]]: A nested list of tape results. Each element in
        the returned list corresponds in order to the provided tapes.
    """
    if not qml.active_return():
        mode = "forward" if grad_on_execution else "backward"

        return _execute_legacy(
            tapes,
            device,
            execute_fn,
            gradient_fn,
            gradient_kwargs,
            _n=_n,
            max_diff=max_diff,
            mode=mode,
        )

    all_params = []
    parameters = []
    lens = []
    trainable = []
    output_types = []

    if isinstance(device, qml.devices.experimental.Device):  # pragma: no-cover
        # assumes all tapes have the same shot vector
        has_partitioned_shots = tapes[0].shots.has_partitioned_shots
        num_shot_copies = tapes[0].shots.num_copies or 1
        vjp_shots = legacy_shots = None
    else:
        has_partitioned_shots = vjp_shots = device.shot_vector
        legacy_shots = qml.measurements.Shots(device.shot_vector or 1)
        num_shot_copies = legacy_shots.num_copies

    for tape in tapes:
        # store the trainable parameters
        params = tape.get_parameters(trainable_only=False)
        tape.trainable_params = qml.math.get_trainable_indices(params)

        parameters += [p for i, p in enumerate(params) if i in tape.trainable_params]
        all_params += params
        trainable += (np.array(list(tape.trainable_params)) + sum(lens)).tolist()

        lens.append(len(params))

        o_types = []
        for m in tape.measurements:
            if isinstance(m, SampleMP):
                o_types.append(tf.int64)
            elif isinstance(m, StateMP):
                o_types.append(tf.complex128)
            else:
                o_types.append(tf.float64)

        output_types.extend(o_types * num_shot_copies)
    total_measurements = sum(len(tape.measurements) for tape in tapes)

    if grad_on_execution:
        output_types += [tf.float64] * len(trainable)

    output_types += [tf.int32] * (total_measurements * num_shot_copies)

    def _nest_params(all_params):
        count = 0
        params_unwrapped = []

        for s in lens:
            params_unwrapped.append(all_params[count : count + s])
            count += s

        return params_unwrapped

    def _forward(*all_params):
        params_unwrapped = _nest_params(all_params)
        output_sizes = []

        new_tapes = set_parameters_on_copy_and_unwrap(tapes, params_unwrapped)
        # Forward pass: execute the tapes
        res, jacs = execute_fn(new_tapes, **gradient_kwargs)

        # flatten the results
        res = _flatten_nested_list(res)

        for i, r in enumerate(res):
            # convert output to TensorFlow tensors
            res[i] = _to_tensors(r)
            output_sizes.append(tf.size(res[i]))

        # flatten the jacobians
        if jacs:
            jacs = _flatten_nested_list(jacs)
            for i, jac in enumerate(jacs):
                jacs[i] = tf.convert_to_tensor(jac)
        else:
            jacs = []

        return res + jacs + output_sizes

    @tf.custom_gradient
    def _execute(*all_params):  # pylint:disable=unused-argument
        res = tf.numpy_function(func=_forward, inp=all_params, Tout=output_types)
        output_sizes = res[-total_measurements * num_shot_copies :]

        if grad_on_execution:
            jacs = res[total_measurements * num_shot_copies : -total_measurements * num_shot_copies]

        res = res[: total_measurements * num_shot_copies]

        # reconstruct the nested structure of res
        res = _res_restructured(res, tapes, legacy_shots=legacy_shots)

        def grad_fn(*dy, **tfkwargs):
            """Returns the vector-Jacobian product with given
            parameter values and output gradient dy"""

            # whether the tapes contain multiple measurements
            multi_measurements = [len(tape.measurements) > 1 for tape in tapes]
            dy = list(dy[: total_measurements * num_shot_copies])

            if grad_on_execution:
                # Jacobians were computed on the forward pass (grad_on_execution=True)
                # No additional quantum evaluations needed; simply compute the VJPs directly.

                def _backward(*args):
                    dy = args[: total_measurements * num_shot_copies]
                    jacs = args[total_measurements * num_shot_copies : -len(tapes)]
                    multi_measurements = args[-len(tapes) :]

                    dy = _res_restructured(dy, tapes, legacy_shots=legacy_shots)
                    jacs = _jac_restructured(jacs, tapes)

                    return _compute_vjp(dy, jacs, multi_measurements, has_partitioned_shots)

                vjps = tf.numpy_function(
                    func=_backward,
                    inp=dy + list(jacs) + multi_measurements,
                    Tout=[tf.float64] * len(parameters),
                )

            else:
                # Need to compute the Jacobians on the backward pass (accumulation="backward")
                if isinstance(gradient_fn, qml.gradients.gradient_transform):
                    # Gradient function is a gradient transform.

                    # Generate and execute the required gradient tapes
                    if _n == max_diff:
                        len_all_params = len(all_params)

                        def _backward(*all_params):
                            dy = all_params[len_all_params:]
                            all_params = all_params[:len_all_params]
                            params_unwrapped = _nest_params(all_params)

                            dy = _res_restructured(dy, tapes, legacy_shots=legacy_shots)

                            new_tapes = set_parameters_on_copy_and_unwrap(tapes, params_unwrapped)
                            vjp_tapes, processing_fn = qml.gradients.batch_vjp(
                                new_tapes,
                                dy,
                                gradient_fn,
                                shots=vjp_shots,
                                reduction=lambda vjps, x: vjps.extend(qml.math.unstack(x)),
                                gradient_kwargs=gradient_kwargs,
                            )

                            return processing_fn(execute_fn(vjp_tapes)[0])

                        vjps = tf.py_function(
                            func=_backward,
                            inp=list(all_params) + dy,
                            Tout=[tf.float64] * len(parameters),
                        )

                    else:
                        dy = _res_restructured(dy, tapes, legacy_shots=legacy_shots)

                        vjp_tapes, processing_fn = qml.gradients.batch_vjp(
                            tapes,
                            dy,
                            gradient_fn,
                            shots=vjp_shots,
                            reduction="append",
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
                                grad_on_execution=grad_on_execution,
                            ),
                            nums=output_sizes,
                        )

                        vjps = tf.unstack(tf.concat(vjps, 0), num=len(parameters))

                else:
                    # Gradient function is not a gradient transform
                    # (e.g., it might be a device method).
                    # Note that unlike the previous branch:
                    #
                    # - there is no recursion here
                    # - gradient_fn is not differentiable
                    #
                    # so we cannot support higher-order derivatives.
                    len_all_params = len(all_params)

                    def _backward(*all_params):
                        dy = all_params[len_all_params : -len(tapes)]
                        multi_measurements = all_params[-len(tapes) :]
                        all_params = all_params[:len_all_params]
                        params_unwrapped = _nest_params(all_params)

                        new_tapes = set_parameters_on_copy_and_unwrap(tapes, params_unwrapped)
                        jac = gradient_fn(new_tapes, **gradient_kwargs)

                        vjps = _compute_vjp(dy, jac, multi_measurements, has_partitioned_shots)
                        return vjps

                    vjps = tf.numpy_function(
                        func=_backward,
                        inp=list(all_params) + dy + multi_measurements,
                        Tout=[tf.float64] * len(parameters),
                    )

            if not isinstance(vjps, (list, tuple)):
                vjps = [vjps]

            vjps = iter(vjps)
            vjps = [next(vjps) if x in trainable else None for x in range(len(all_params))]

            variables = tfkwargs.get("variables")
            return (vjps, variables) if variables is not None else vjps

        return res, grad_fn

    return _execute(*all_params)
