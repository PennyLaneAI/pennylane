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
import numpy as np
import tensorflow as tf

import pennylane as qml


from .tensorflow import _compute_vjp, _compute_vjp_new


def execute(tapes, device, execute_fn, gradient_fn, gradient_kwargs, _n=1, max_diff=2, mode=None):
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
        mode (str): Whether the gradients should be computed on the forward
            pass (``forward``) or the backward pass (``backward``).

    Returns:
        list[list[tf.Tensor]]: A nested list of tape results. Each element in
        the returned list corresponds in order to the provided tapes.
    """
    if qml.active_return():
        return _execute_new(
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
        elif tape.measurements[0].return_type is qml.measurements.State:
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

        with qml.tape.Unwrap(*tapes, params=params_unwrapped):
            # Forward pass: execute the tapes
            res, jacs = execute_fn(tapes, **gradient_kwargs)

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
                    func=lambda *args: _compute_vjp(args[:len_dy], args[len_dy:]),
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

                            with qml.tape.Unwrap(*tapes, params=params_unwrapped):
                                vjp_tapes, processing_fn = qml.gradients.batch_vjp(
                                    tapes,
                                    dy,
                                    gradient_fn,
                                    reduction=lambda vjps, x: vjps.extend(qml.math.unstack(x)),
                                    gradient_kwargs=gradient_kwargs,
                                )

                                vjps = processing_fn(execute_fn(vjp_tapes)[0])
                            return vjps

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
                            execute(
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

                        with qml.tape.Unwrap(*tapes, params=params_unwrapped):
                            vjps = _compute_vjp(dy, gradient_fn(tapes, **gradient_kwargs))

                        return vjps

                    vjps = tf.numpy_function(
                        func=_backward,
                        inp=list(all_params) + dy,
                        Tout=[tf.float64] * len(parameters),
                    )

            vjps = iter(vjps)
            vjps = [next(vjps) if x in trainable else None for x in range(len(all_params))]

            variables = tfkwargs.get("variables", None)
            return (vjps, variables) if variables is not None else vjps

        return res, grad_fn

    return _execute(*all_params)


def _execute_new(
    tapes, device, execute_fn, gradient_fn, gradient_kwargs, _n=1, max_diff=2, mode=None
):
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
        elif tape.measurements[0].return_type is qml.measurements.State:
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

        with qml.tape.Unwrap(*tapes, params=params_unwrapped):
            # Forward pass: execute the tapes
            res, jacs = execute_fn(tapes, **gradient_kwargs)

        for i, tape in enumerate(tapes):
            # convert output to TensorFlow tensors
            res[i] = tf.convert_to_tensor(res[i])

            if tape.all_sampled:
                res[i] = tf.cast(res[i], tf.int64)

            output_sizes.append(tf.size(res[i]))

        if jacs:
            for i, jac in enumerate(jacs):
                jacs[i] = tf.convert_to_tensor(jac)

        return res + jacs + output_sizes

    @tf.custom_gradient
    def _execute(*all_params):  # pylint:disable=unused-argument

        res = tf.numpy_function(func=_forward, inp=all_params, Tout=output_types)
        output_sizes = res[-len(tapes) :]

        if mode == "forward":
            jacs = res[len(tapes) : -len(tapes)]

        res = res[: len(tapes)]

        def grad_fn(*dy, **tfkwargs):
            """Returns the vector-Jacobian product with given
            parameter values and output gradient dy"""

            # whether the tapes contain multiple measurements
            multi_measurements = [len(tape.measurements) > 1 for tape in tapes]
            dy = dy[: len(tapes)]

            if mode == "forward":
                # Jacobians were computed on the forward pass (mode="forward")
                # No additional quantum evaluations needed; simply compute the VJPs directly.
                len_dy = len(dy)

                def _backward(*args):
                    dy = args[:len_dy]
                    jacs = args[len_dy:-len_dy]
                    multi_measurements = args[-len_dy:]

                    new_jacs = []
                    for i, jac in enumerate(jacs):
                        if tf.rank(jac) > 0:
                            new_jacs.append(tuple(tf.unstack(jac)))
                        else:
                            new_jacs.append(jac)

                    return _compute_vjp_new(dy, tuple(new_jacs), multi_measurements)

                vjps = tf.numpy_function(
                    func=_backward,
                    inp=list(dy) + list(jacs) + multi_measurements,
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

                            with qml.tape.Unwrap(*tapes, params=params_unwrapped):
                                vjp_tapes, processing_fn = qml.gradients.batch_vjp(
                                    tapes,
                                    dy,
                                    gradient_fn,
                                    reduction=lambda vjps, x: vjps.extend(qml.math.unstack(x)),
                                    gradient_kwargs=gradient_kwargs,
                                )

                                vjps = processing_fn(execute_fn(vjp_tapes)[0])
                            return vjps

                        vjps = tf.py_function(
                            func=_backward,
                            inp=list(all_params) + list(dy),
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
                            execute(
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
                        dy = all_params[len_all_params : -len(tapes)]
                        multi_measurements = all_params[-len(tapes) :]
                        all_params = all_params[:len_all_params]
                        params_unwrapped = _nest_params(all_params)

                        with qml.tape.Unwrap(*tapes, params=params_unwrapped):
                            jac = gradient_fn(tapes, **gradient_kwargs)

                        vjps = _compute_vjp_new(dy, jac, multi_measurements)
                        return vjps

                    vjps = tf.numpy_function(
                        func=_backward,
                        inp=list(all_params) + list(dy) + multi_measurements,
                        Tout=[tf.float64] * len(parameters),
                    )

            vjps = iter(vjps)
            vjps = [next(vjps) if x in trainable else None for x in range(len(all_params))]

            variables = tfkwargs.get("variables", None)
            return (vjps, variables) if variables is not None else vjps

        return res, grad_fn

    return _execute(*all_params)
