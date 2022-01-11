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
import jax.numpy as jnp
from jax.experimental import host_callback

import numpy as np
import pennylane as qml
from pennylane.operation import Variance, Expectation

dtype = jnp.float64


def _execute_id_tap(
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
        result = []

        def wrapper(p, t, device=None):
            """Compute the forward pass."""
            new_tapes = []

            for t, a in zip(tapes, p):
                new_tapes.append(t.copy(copy_operations=True))
                new_tapes[-1].set_parameters(a)

            with qml.tape.Unwrap(*new_tapes):
                res, _ = execute_fn(new_tapes, **gradient_kwargs)

            # Put the array back to the device as we're using id_tap
            res = [jax.device_put(jnp.array(r), device) for r in res]
            result.append(res)

        host_callback.id_tap(wrapper, params, tap_with_device=True)
        if isinstance(result, list) and len(result) > 0:
            return result[0]
        return result

    def wrapped_exec_fwd(params):
        return wrapped_exec(params), params

    def wrapped_exec_bwd(params, g):
        result = []

        if isinstance(gradient_fn, qml.gradients.gradient_transform):

            def non_diff_wrapper(args, t, device=None):
                """Compute the VJP in a non-differentiable manner."""
                new_tapes = []
                p = args

                for t, a in zip(tapes, p):
                    new_tape = t.copy(copy_operations=True)
                    new_tapes.append(new_tape)
                    new_tapes[-1].set_parameters(a)
                    new_tapes[-1].trainable_params = t.trainable_params

                # Note: the cotangent (g) is pulled from the outside because it
                # needs to stay a Traced JAX BatchTrace. Otherwise issues with
                # the output shape of this function arises as jax.vmap is being
                # used.
                g_on_device = jax.device_put(g, device)

                vjp_tapes, processing_fn = qml.gradients.batch_vjp(
                    new_tapes,
                    g_on_device,
                    gradient_fn,
                    reduction="append",
                    gradient_kwargs=gradient_kwargs,
                )

                partial_res = execute_fn(vjp_tapes)[0]
                res = processing_fn(partial_res)

                # Put the array back to the device as we're using id_tap
                res = [jax.device_put(jnp.array(r), device) for r in res]
                result.extend(res)

            args = tuple(params)
            host_callback.id_tap(non_diff_wrapper, args, tap_with_device=True)
            res = result

            # param_idx = 0
            # res = []

            # # Group the vjps based on the parameters of the tapes
            # for p in params:
            #     param_vjp = vjps[param_idx : param_idx + len(p)]
            #     res.append(param_vjp)
            #     param_idx += len(p)

            # Unwrap partial results into ndim=0 arrays to allow
            # differentiability with JAX
            # E.g.,
            # [DeviceArray([-0.9553365], dtype=float32), DeviceArray([0., 0.],
            # dtype=float32)]
            # is mapped to
            # [[DeviceArray(-0.9553365, dtype=float32)], [DeviceArray(0.,
            # dtype=float32), DeviceArray(0., dtype=float32)]].

            need_unwrapping = any(r.ndim != 0 for r in res)
            if need_unwrapping:
                unwrapped_res = []
                for r in res:
                    if r.ndim != 0:
                        r = [jnp.array(p) for p in r]
                    unwrapped_res.append(r)

                res = unwrapped_res

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
def _execute_with_fwd_id_tap(
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

        result = []
        jacobian = []

        def wrapper(p, t, device):
            """Compute the forward pass by returning the jacobian too."""
            new_tapes = []

            for t, a in zip(tapes, p):
                new_tapes.append(t.copy(copy_operations=True))
                new_tapes[-1].set_parameters(a)

            res, jacs = execute_fn(new_tapes, **gradient_kwargs)

            # Put the arrays back to the device as we're using id_tap
            res = [jax.device_put(jnp.array(r), device) for r in res]
            result.extend(res)

            # On the forward execution return the jacobian too
            jacs = [jax.device_put(jnp.array(r), device) for r in jacs]
            jacobian.extend(jacs)

        fwd_shapes = [jax.ShapeDtypeStruct((1,), dtype) for _ in range(total_size)]
        jacobian_shape = [jax.ShapeDtypeStruct((1, len(p)), dtype) for p in params]
        host_callback.id_tap(wrapper, params, tap_with_device=True)
        return result, jacobian

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
