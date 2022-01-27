# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

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
import jax.numpy as jnp
from jax.experimental import host_callback

import numpy as np
import pennylane as qml
from pennylane.operation import Variance, Expectation
from pennylane.interfaces.batch import InterfaceUnsupportedError

dtype = jnp.float64


def execute(tapes, device, execute_fn, gradient_fn, gradient_kwargs, _n=1, max_diff=1, mode=None):
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
        max_diff (int): If ``gradient_fn`` is a gradient transform, this option specifies
            the maximum order of derivatives to support. Increasing this value allows
            for higher order derivatives to be extracted, at the cost of additional
            (classical) computational overhead during the backwards pass.
        mode (str): Whether the gradients should be computed on the forward
            pass (``forward``) or the backward pass (``backward``).

    Returns:
        list[list[float]]: A nested list of tape results. Each element in
        the returned list corresponds in order to the provided tapes.
    """
    # pylint: disable=unused-argument
    if max_diff > 1:
        raise InterfaceUnsupportedError("The JAX interface only supports first order derivatives.")

    _validate_tapes(tapes)

    for tape in tapes:
        # set the trainable parameters
        params = tape.get_parameters(trainable_only=False)
        tape.trainable_params = qml.math.get_trainable_indices(params)

    parameters = tuple(list(t.get_parameters()) for t in tapes)

    if gradient_fn is None:
        return _execute_with_fwd(
            parameters,
            tapes=tapes,
            device=device,
            execute_fn=execute_fn,
            gradient_kwargs=gradient_kwargs,
            _n=_n,
        )

    return _execute(
        parameters,
        tapes=tapes,
        device=device,
        execute_fn=execute_fn,
        gradient_fn=gradient_fn,
        gradient_kwargs=gradient_kwargs,
        _n=_n,
    )


def _validate_tapes(tapes):
    """Validates that the input tapes are compatible with JAX support.

    Raises:
        ValueError: if tapes with non-scalar outputs were provided or a return
        type other than variance and expectation value was used
    """
    for t in tapes:

        if len(t.observables) != 1:
            raise InterfaceUnsupportedError(
                "The jittable JAX interface currently only supports quantum nodes with a single return type."
            )

        for o in t.observables:
            return_type = o.return_type
            if return_type is not Variance and return_type is not Expectation:
                raise InterfaceUnsupportedError(
                    f"Only Variance and Expectation returns are supported for the jittable JAX interface, given {return_type}."
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

    # Only have scalar outputs
    total_size = len(tapes)
    total_params = np.sum([len(p) for p in params])

    @jax.custom_vjp
    def wrapped_exec(params):
        def wrapper(p):
            """Compute the forward pass."""
            new_tapes = []

            for t, a in zip(tapes, p):
                new_tapes.append(t.copy(copy_operations=True))
                new_tapes[-1].set_parameters(a)

            with qml.tape.Unwrap(*new_tapes):
                res, _ = execute_fn(new_tapes, **gradient_kwargs)

            return res

        shapes = [jax.ShapeDtypeStruct((1,), dtype) for _ in range(total_size)]
        res = host_callback.call(wrapper, params, result_shape=shapes)
        return res

    def wrapped_exec_fwd(params):
        return wrapped_exec(params), params

    def wrapped_exec_bwd(params, g):

        if isinstance(gradient_fn, qml.gradients.gradient_transform):

            def non_diff_wrapper(args):
                """Compute the VJP in a non-differentiable manner."""
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
                non_diff_wrapper,
                args,
                result_shape=jax.ShapeDtypeStruct((total_params,), dtype),
            )

            param_idx = 0
            res = []

            # Group the vjps based on the parameters of the tapes
            for p in params:
                param_vjp = vjps[param_idx : param_idx + len(p)]
                res.append(param_vjp)
                param_idx += len(p)

            # Unwrap partial results into ndim=0 arrays to allow
            # differentiability with JAX
            # E.g.,
            # [DeviceArray([-0.9553365], dtype=float32), DeviceArray([0., 0.],
            # dtype=float32)]
            # is mapped to
            # [[DeviceArray(-0.9553365, dtype=float32)], [DeviceArray(0.,
            # dtype=float32), DeviceArray(0., dtype=float32)]].
            need_unstacking = any(r.ndim != 0 for r in res)
            if need_unstacking:
                res = [qml.math.unstack(x) for x in res]

            return (tuple(res),)

        # Gradient function is a device method.
        with qml.tape.Unwrap(*tapes):
            jacs = gradient_fn(tapes, **gradient_kwargs)

        vjps = [qml.gradients.compute_vjp(d, jac) for d, jac in zip(g, jacs)]
        res = [[jnp.array(p) for p in v] for v in vjps]
        return (tuple(res),)

    wrapped_exec.defvjp(wrapped_exec_fwd, wrapped_exec_bwd)
    return wrapped_exec(params)


# The execute function in forward mode
def _execute_with_fwd(
    params,
    tapes=None,
    device=None,
    execute_fn=None,
    gradient_kwargs=None,
    _n=1,
):  # pylint: disable=dangerous-default-value,unused-argument

    # Only have scalar outputs
    total_size = len(tapes)

    @jax.custom_vjp
    def wrapped_exec(params):
        def wrapper(p):
            """Compute the forward pass by returning the jacobian too."""
            new_tapes = []

            for t, a in zip(tapes, p):
                new_tapes.append(t.copy(copy_operations=True))
                new_tapes[-1].set_parameters(a)

            res, jacs = execute_fn(new_tapes, **gradient_kwargs)

            # On the forward execution return the jacobian too
            return res, jacs

        fwd_shapes = [jax.ShapeDtypeStruct((1,), dtype) for _ in range(total_size)]
        jacobian_shape = [jax.ShapeDtypeStruct((1, len(p)), dtype) for p in params]
        res, jacs = host_callback.call(
            wrapper,
            params,
            result_shape=tuple([fwd_shapes, jacobian_shape]),
        )
        return res, jacs

    def wrapped_exec_fwd(params):
        res, jacs = wrapped_exec(params)
        return res, tuple([jacs, params])

    def wrapped_exec_bwd(params, g):

        # Use the jacobian that was computed on the forward pass
        jacs, params = params

        # Adjust the structure of how the jacobian is returned to match the
        # non-forward mode cases
        # E.g.,
        # [DeviceArray([[ 0.06695931,  0.01383095, -0.46500877]], dtype=float32)]
        # is mapped to
        # [[DeviceArray(0.06695931, dtype=float32), DeviceArray(0.01383095,
        # dtype=float32), DeviceArray(-0.46500877, dtype=float32)]]
        res_jacs = []
        for j in jacs:
            this_j = []
            for i in range(j.shape[1]):
                this_j.append(j[0, i])
            res_jacs.append(this_j)
        return tuple([tuple(res_jacs)])

    wrapped_exec.defvjp(wrapped_exec_fwd, wrapped_exec_bwd)
    res = wrapped_exec(params)

    tracing = any(isinstance(r, jax.interpreters.ad.JVPTracer) for r in res)

    # When there are no tracers (not differentiating), we have the result of
    # the forward pass and the jacobian, but only need the result of the
    # forward pass
    if len(res) == 2 and not tracing:
        res = res[0]

    return res
