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

dtype = jnp.float64

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
    )[0]


# @autograd.extend.primitive
# def _execute(
#     parameters,
#     tapes=None,
#     device=None,
#     execute_fn=None,
#     gradient_fn=None,
#     gradient_kwargs=None,
#     _n=1,
# ):  # pylint: disable=dangerous-default-value,unused-argument
#     """Autodifferentiable wrapper around ``Device.batch_execute``.
# 
#     The signature of this function is designed to work around Autograd restrictions.
#     Note that the ``parameters`` argument is dependent on the ``tapes`` argument;
#     this function should always be called as follows:
# 
#     >>> parameters = [autograd.builtins.list(t.get_parameters()) for t in tapes])
#     >>> parameters = autograd.builtins.tuple(parameters)
#     >>> _execute(parameters, tapes=tapes, device=device)
# 
#     In particular:
# 
#     - ``parameters`` is dependent on the provided tapes: always extract them as above
#     - ``tapes`` is a *required* argument
#     - ``device`` is a *required* argument
# 
#     The private argument ``_n`` is used to track nesting of derivatives, for example
#     if the nth-order derivative is requested. Do not set this argument unless you
#     understand the consequences!
#     """
#     with qml.tape.Unwrap(*tapes):
#         res, jacs = execute_fn(tapes, **gradient_kwargs)
# 
#     for i, r in enumerate(res):
#         res[i] = jax.numpy.array(r)
# 
#         # TODO: need?
#         # if r.dtype == np.dtype("object"):
#         #     # For backwards compatibility, we flatten ragged tape outputs
#         #     res[i] = np.hstack(r)
# 
#     return res, jacs


def vjp(
    ans,
    parameters,
    tapes=None,
    device=None,
    execute_fn=None,
    gradient_fn=None,
    gradient_kwargs=None,
    _n=1,
):  # pylint: disable=dangerous-default-value,unused-argument
    """Returns the vector-Jacobian product operator for a batch of quantum tapes.

    Args:
        ans (array): the result of the batch tape execution
        parameters (list[list[Any]]): Nested list of the quantum tape parameters.
            This argument should be generated from the provided list of tapes.
        tapes (Sequence[.QuantumTape]): batch of tapes to execute
        device (.Device): Device to use to execute the batch of tapes.
            If the device does not provide a ``batch_execute`` method,
            by default the tapes will be executed in serial.
        execute_fn (callable): The execution function used to execute the tapes
            during the forward pass. This function must return a tuple ``(results, jacobians)``.
            If ``jacobians`` is an empty list, then ``gradient_fn`` is used to
            compute the gradients during the backwards pass.
        gradient_fn (callable): the gradient function to use to compute quantum gradients
        gradient_kwargs (dict): dictionary of keyword arguments to pass when
            determining the gradients of tapes
        _n (int): a positive integer used to track nesting of derivatives, for example
            if the nth-order derivative is requested.

    Returns:
        function: this function accepts the backpropagation
        gradient output vector, and computes the vector-Jacobian product
    """

    def grad_fn(dy):
        """Returns the vector-Jacobian product with given
        parameter values and output gradient dy"""

        dy = dy[0]
        jacs = ans[1]

        if jacs:
            # Jacobians were computed on the forward pass (mode="forward")
            # No additional quantum evaluations needed; simply compute the VJPs directly.
            vjps = [qml.gradients.compute_vjp(d, jac) for d, jac in zip(dy, jacs)]

        else:
            # Need to compute the Jacobians on the backward pass (accumulation="backward")

            # Temporary: check if the gradient function is a differentiable transform.
            # For the moment, simply check if it is part of the `qml.gradients` package.
            # Longer term, we should have a way of checking this directly
            # (e.g., isinstance(gradient_fn, GradientTransform))
            module_name = getattr(inspect.getmodule(gradient_fn), "__name__", "")

            if "pennylane.gradients" in module_name:

                # Generate and execute the required gradient tapes
                vjp_tapes, processing_fn = qml.gradients.batch_vjp(
                    tapes, dy, gradient_fn, reduction="append", gradient_kwargs=gradient_kwargs
                )

                # This is where the magic happens. Note that we call ``execute``.
                # This recursion, coupled with the fact that the gradient transforms
                # are differentiable, allows for arbitrary order differentiation.
                vjps = processing_fn(
                    execute(vjp_tapes, device, execute_fn, gradient_fn, gradient_kwargs, _n=_n + 1)
                )

            elif inspect.ismethod(gradient_fn) and gradient_fn.__self__ is device:
                # Gradient function is a device method.
                # Note that unlike the previous branch:
                #
                # - there is no recursion here
                # - gradient_fn is not differentiable
                #
                # so we cannot support higher-order derivatives.

                with qml.tape.Unwrap(*tapes):
                    jacs = gradient_fn(tapes, **gradient_kwargs)

                vjps = [qml.gradients.compute_vjp(d, jac) for d, jac in zip(dy, jacs)]

            else:
                raise ValueError("Unknown gradient function.")

        return [qml.math.to_numpy(v, max_depth=_n) if isinstance(v, ArrayBox) else v for v in vjps]

    return grad_fn

# TODO: from the interface as of now
#def _execute(tape, , device):
def _execute(
    params,
    tapes=None,
    device=None,
    execute_fn=None,
    gradient_fn=None,
    gradient_kwargs=None,
    _n=1,
):  # pylint: disable=dangerous-default-value,unused-argument

    tape = tapes[0]

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
        if not isinstance(params, list):
            params = [params]

        exec_fn = partial(tape.execute_device, device=device)
        return host_callback.call(
            exec_fn,
            params,
            result_shape=jax.ShapeDtypeStruct((1,), dtype),
        )

    def wrapped_exec_fwd(params):
        return wrapped_exec(params), params

    def wrapped_exec_bwd(params, g):
        if not isinstance(params, list):
            params = [params]

        # def jacobian(params):
        #     new_tape = tape.copy()
        #     new_tape.set_parameters(params)
        #     return new_tape.jacobian(device, **new_tape.jacobian_options)

        # val = g.reshape((-1,)) * host_callback.call(
        #     jacobian,
        #     params,
        #     result_shape=jax.ShapeDtypeStruct((1,1), dtype),
        # )

        bwd_fn = partial(vjp, ans=g, tapes=[tape], device=device, gradient_fn=gradient_fn)
        val = g.reshape((-1,)) * host_callback.call(
            bwd_fn,
            params,
            result_shape=jax.ShapeDtypeStruct((1,1), dtype),
        )
        return (val,)  # Comma is on purpose.

    wrapped_exec.defvjp(wrapped_exec_fwd, wrapped_exec_bwd)
    return wrapped_exec(params)

