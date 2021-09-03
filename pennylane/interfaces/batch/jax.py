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
from pennylane import math
import numpy as np

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

    parameters = tuple(list(t.get_parameters()) for t in tapes)

    return _execute(
        parameters,
        tapes=tapes,
        device=device,
        execute_fn=execute_fn,
        gradient_fn=gradient_fn,
        gradient_kwargs=gradient_kwargs,
        _n=_n,
        max_diff=max_diff,
    )


def _execute(
    params,
    tapes=None,
    device=None,
    execute_fn=None,
    gradient_fn=None,
    gradient_kwargs=None,
    _n=1,
    max_diff=2,
):  # pylint: disable=dangerous-default-value,unused-argument
    jacs = None

    @jax.custom_vjp
    def wrapped_exec(params):
        nonlocal jacs

        with qml.tape.Unwrap(*tapes):
            res, jacs = execute_fn(tapes, **gradient_kwargs)

        for i, r in enumerate(res):
            res[i] = jnp.asarray(r)

            if r.dtype == jnp.dtype("object"):
                # For backwards compatibility, we flatten ragged tape outputs
                res[i] = jnp.hstack(r)

        return res

    def wrapped_exec_fwd(params):
        return wrapped_exec(params), params

    def wrapped_exec_bwd(params, g):

        # Generate and execute the required gradient tapes
        if _n == max_diff:
            with qml.tape.Unwrap(*tapes):

                if any(["BatchTracer" in str(type(tracer)) for tracer in g]):

                    # 1. step: extract only one of the batches
                    # 2. step: batch_vjp
                    # 3. step: execute, process
                    # 4. step: batch back together
                    # 5. step: return in the same shape as the output of the exec_fwd

                    # Unpack batched traces
                    final_vjps = []

                    #TODO: no hardcode
                    batch_size = 2
                    # print("batch_size", batch_size)
                    # print("G 0", g[0].val.shape, g[0].val)
                    for i in range(batch_size):
                        g_batch = [b.val[i,:] for b in g]
                        print("G batch: ", g_batch)
                        vjp_tapes, processing_fn = qml.gradients.batch_vjp(
                            tapes,
                            g_batch,
                            gradient_fn,
                            reduction="append",
                            gradient_kwargs=gradient_kwargs,
                        )
                        vjp_tapes, processing_fn
                        partial_res = execute_fn(vjp_tapes)[0]
                        partial_vjps = [[jnp.asarray(s) for s in r] for r in processing_fn(partial_res)]
                        final_vjps.append(partial_vjps)

                    # vjps = []
                    # for i,v enumerate(final_vjps):
                    #     for a in v:
                    #         vjps.append(jnp.sum([b for c, d in enumerate(b)]) for b in final_vjps]))

                    #print("Final vjps: ", final_vjps)
                    vjps = final_vjps[0]
                else:
                    vjp_tapes, processing_fn = qml.gradients.batch_vjp(
                        tapes,
                        g,
                        gradient_fn,
                        reduction="append",
                        gradient_kwargs=gradient_kwargs,
                    )
                    partial_res = execute_fn(vjp_tapes)[0]
                    vjps = [[jnp.asarray(s) for s in r] for r in processing_fn(partial_res)]
            vjps = (tuple(vjps),)

        else:
            print("Type of g", type(g))
            vjp_tapes, processing_fn = qml.gradients.batch_vjp(
                tapes, g, gradient_fn, reduction="append", gradient_kwargs=gradient_kwargs
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
            vjps = [[jnp.asarray(s).primal for s in r] for r in vjps]
            vjps = (tuple(vjps),)
        return vjps

    wrapped_exec.defvjp(wrapped_exec_fwd, wrapped_exec_bwd)
    return wrapped_exec(params)
