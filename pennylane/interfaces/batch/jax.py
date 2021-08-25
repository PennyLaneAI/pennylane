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
import inspect

from functools import partial
import jax
import jax.experimental.host_callback as host_callback
import jax.numpy as jnp
import numpy as np

from pennylane.interfaces.jax import JAXInterface
from pennylane.queuing import AnnotatedQueue
from pennylane.operation import Variance, Expectation
import pennylane as qml

dtype = jnp.float32

def execute(tapes, device, execute_fn, gradient_fn, gradient_kwargs, _n=1, max_diff=2):
    """Execute a batch of tapes with Autograd parameters on a device.

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

    parameters = tuple( list(t.get_parameters()) for t in tapes)

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

    for tape in tapes:
        # TODO (chase): Add support for more than 1 measured observable.
        if len(tape.observables) != 1:
            raise ValueError(
                "The JAX interface currently only supports quantum nodes with a single return type."
            )
        return_type = tape.observables[0].return_type
        if return_type is not Variance and return_type is not Expectation:
            raise ValueError(
                f"Only Variance and Expectation returns are supported for the JAX interface, given {return_type}."
            )

    @jax.custom_vjp
    def wrapped_exec(params):

        def wrapper(p):
            new_tapes = [t.copy() for t in tapes]

            for tape, tape_params in zip(new_tapes, p):
                tape.set_parameters(tape_params, trainable_only=False)

            res = execute_fn(new_tapes)
            return jnp.asarray(res)

        return host_callback.call( wrapper, params, result_shape=jax.ShapeDtypeStruct((len(tapes), 1), dtype),)

    def wrapped_exec_fwd(params):
        return wrapped_exec(params), params

    def wrapped_exec_bwd(params, g):
        # The derivative order is at the maximum. Compute the VJP
        # in a non-differentiable manner to reduce overhead.
        with qml.tape.Unwrap(*tapes):
            vjp_tapes, processing_fn = qml.gradients.batch_vjp(
                tapes,
                g,
                gradient_fn,
                reduction="extend",
                gradient_kwargs=gradient_kwargs,
            )

        # 1. Vanilla computations (to be removed)
        vjps = processing_fn(execute_fn(vjp_tapes))
        vjps = tuple([v] for v in vjps),

        vjps = g.reshape((-1,)) * jnp.asarray(vjps)

        # 2. host_callback.call (todo)

        # def wrapper(p):

        #     new_tapes = [t.copy() for t in tapes]

        #     for tape, tape_params in zip(new_tapes, p):
        #         tape.set_parameters(tape_params, trainable_only=False)

        #     vjp_tapes, processing_fn = qml.gradients.batch_vjp(
        #         new_tapes,
        #         g,
        #         gradient_fn,
        #         reduction="extend",
        #         gradient_kwargs=gradient_kwargs,
        #     )
        #     res = processing_fn(execute_fn(vjp_tapes))
        #     return jnp.asarray(res)

        # vjps = host_callback.call(
        #     wrapper,
        #     params,
        #     result_shape=jax.ShapeDtypeStruct((1,1), dtype),
        # )
        return tuple(vjps)  # Comma is on purpose.

    wrapped_exec.defvjp(wrapped_exec_fwd, wrapped_exec_bwd)
    return wrapped_exec(params)

