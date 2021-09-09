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
This module contains functions for adding the JAX interface
to a PennyLane Device class.
"""
# pylint: disable=too-many-arguments
import jax
from jax.experimental import host_callback
import jax.numpy as jnp
import numpy as np
import inspect

import pennylane as qml

dtype = jnp.float32


def execute(tapes, device, execute_fn, gradient_fn, gradient_kwargs, _n=1, max_diff=2):
    """Execute a batch of tapes with JAX parameters on a device.

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

    Returns:
        list[list[float]]: A nested list of tape results. Each element in
        the returned list corresponds in order to the provided tapes.
    """
    for tape in tapes:
        # set the trainable parameters
        params = tape.get_parameters(trainable_only=False)
        tape.trainable_params = qml.math.get_trainable_indices(params)

    parameters = tuple(list(t.get_parameters()) for t in tapes)

    return _execute(
        parameters,
        tapes=tapes,
        device=device,
        execute_fn=execute_fn,
        gradient_fn=gradient_fn,
        gradient_kwargs=gradient_kwargs,
        _n=_n,
    )


def _execute(
    params,
    tapes=None,
    device=None,
    execute_fn=None,
    gradient_fn=None,
    gradient_kwargs=None,
    _n=1,
):  # pylint: disable=dangerous-default-value,unused-argument
    jacs = None
    total_size = np.sum([t.output_dim for t in tapes])
    total_params = np.sum([len(p) for p in params])

    @jax.custom_vjp
    def wrapped_exec(params):

        def wrapper(p):
            new_tapes = []

            for t, a in zip(tapes, p):
                new_tapes.append(t.copy(copy_operations=True))
                new_tapes[-1].set_parameters(a)

            with qml.tape.Unwrap(*new_tapes):
                res, jacs = execute_fn(new_tapes, **gradient_kwargs)

            return np.stack(res)

        res = host_callback.call(
            wrapper, params, result_shape=jax.ShapeDtypeStruct((total_size, 1), jnp.float32)
        )
        return res

    def wrapped_exec_fwd(params):
        return wrapped_exec(params), params

    def wrapped_exec_bwd(params, g):

        # TODO: forward mode
        if False:
            return []
        else:
            module_name = getattr(inspect.getmodule(gradient_fn), "__name__", "")
            if "pennylane.gradients" in module_name:

                def non_diff_wrapper(args):
                    """The derivative order is at the maximum. Compute the VJP
                    in a non-differentiable manner to reduce overhead."""
                    new_tapes = []
                    p = args[:-1]
                    dy = args[-1]

                    for t, a in zip(tapes, p):
                        new_tapes.append(t.copy(copy_operations=True))
                        new_tapes[-1].set_parameters(a)
                        new_tapes[-1].trainable_params = t.trainable_params

                    vjp_tapes, processing_fn = qml.gradients.batch_vjp(
                        new_tapes,
                        dy,
                        gradient_fn,
                        reduction="append",
                        gradient_kwargs=gradient_kwargs,
                    )

                    partial_res = execute_fn(vjp_tapes)[0]
                    res = processing_fn(partial_res)
                    return np.concatenate(res)

                args = tuple(params) + (g,)
                vjps = host_callback.call(
                    non_diff_wrapper, args, result_shape=jax.ShapeDtypeStruct((total_params,), jnp.float32)
                )

                start = 0
                res = []

                for p in params:
                    res.append(vjps[start : start + len(p)])
                    start += len(p)
                    if len(p) == 1:
                        res[-1] = res[-1][0]

                if res[0].ndim != 0:
                    res = [[jnp.array(p) for p in res[0]]]
                return (tuple(res),)

            elif (
                hasattr(gradient_fn, "fn")
                and inspect.ismethod(gradient_fn.fn)
                and gradient_fn.fn.__self__ is device
            ):
                # Gradient function is a device method.
                # Note that unlike the previous branch:
                #
                # - there is no recursion here
                # - gradient_fn is not differentiable
                #
                # so we cannot support higher-order derivatives.

                with qml.tape.Unwrap(*tapes):
                    jacs = gradient_fn(tapes, **gradient_kwargs)

                vjps = [qml.gradients.compute_vjp(d, jac) for d, jac in zip(g, jacs)]
                res = [[jnp.array(p) for p in vjps[0]]]
                return (tuple(res),)

            else:

                raise ValueError("Unknown gradient function.")

    wrapped_exec.defvjp(wrapped_exec_fwd, wrapped_exec_bwd)
    return wrapped_exec(params)
