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
from pennylane.interfaces import InterfaceUnsupportedError
from pennylane.interfaces.jax import _raise_vector_valued_fwd
from pennylane import math

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

    for tape in tapes:
        # set the trainable parameters
        params = tape.get_parameters(trainable_only=False)
        tape.trainable_params = qml.math.get_trainable_indices(params)
        print(tape.trainable_params)

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


def _numeric_type_to_dtype(numeric_type):
    """Auxiliary function for converting from Python numeric types to JAX
    dtypes based on the precision defined for the interface."""

    single_precision = dtype is jnp.float32
    if numeric_type is int:
        return jnp.int32 if single_precision else jnp.int64

    if numeric_type is float:
        return jnp.float32 if single_precision else jnp.float64

    # numeric_type is complex
    return jnp.complex64 if single_precision else jnp.complex128


def _extract_shape_dtype_structs(tapes, device):
    """Auxiliary function for defining the jax.ShapeDtypeStruct objects given
    the tapes and the device.

    The host_callback.call function expects jax.ShapeDtypeStruct objects to
    describe the output of the function call.
    """
    shape_dtypes = []

    for t in tapes:
        shape = t.shape(device)

        tape_dtype = _numeric_type_to_dtype(t.numeric_type)
        shape_and_dtype = jax.ShapeDtypeStruct(tuple(shape), tape_dtype)

        shape_dtypes.append(shape_and_dtype)

    return shape_dtypes

def compute_vjp(dy, jac, num=None):
    if jac is None:
        return None

    dy_row = math.reshape(dy, [-1])

    if num is None:
        num = math.shape(dy_row)[0]

    if not isinstance(dy_row, np.ndarray):
        jac = math.convert_like(jac, dy_row)
        jac = math.cast(jac, dy_row.dtype)

    jac = math.reshape(jac, [num, -1])

    try:
        if math.allclose(dy, 0):
            # If the dy vector is zero, then the
            # corresponding element of the VJP will be zero.
            num_params = jac.shape[1]
            res = math.convert_like(np.zeros([num_params]), dy)
            return math.cast(res, dy.dtype)
    except (AttributeError, TypeError):
        pass

    return math.tensordot(jac, dy_row, [[0], [0]])

def vjp(tape, dy, gradient_fn, gradient_kwargs=None):
    gradient_kwargs = gradient_kwargs or {}
    num_params = len(tape.trainable_params)

    if num_params == 0:
        # The tape has no trainable parameters; the VJP
        # is simply none.
        return [], lambda _, num=None: None

    try:
        if math.allclose(dy, 0):
            # If the dy vector is zero, then the
            # corresponding element of the VJP will be zero,
            # and we can avoid a quantum computation.

            def func(_, num=None):  # pylint: disable=unused-argument
                res = math.convert_like(np.zeros([num_params]), dy)
                return math.cast(res, dy.dtype)

            return [], func
    except (AttributeError, TypeError):
        pass

    gradient_tapes, fn = gradient_fn(tape, **gradient_kwargs)

    def processing_fn(results, num=None):
        # postprocess results to compute the Jacobian
        jac = fn(results)
        return compute_vjp(dy, jac, num=num)

    return gradient_tapes, processing_fn

def compute_jacobian(tape, gradient_fn, gradient_kwargs=None):
    gradient_kwargs = gradient_kwargs or {}
    num_params = len(tape.trainable_params)

    #TODO: how to simplify if dy is zero? Check g before hostcallback call

    if num_params == 0:
        # The tape has no trainable parameters; the VJP
        # is simply none.
        return [], lambda _, num=None: None

    # Compute the gradient related tapes and func
    gradient_tapes, fn = gradient_fn(tape, **gradient_kwargs)

    def processing_fn(results, num=None):

        # postprocess results to compute the Jacobian
        return fn(results)

    return gradient_tapes, processing_fn


def batch_jacobians(jacobians, gradient_fn, reduction="append", gradient_kwargs=None):
    # TODO: instead of tapes, just get in the jacobians

    gradient_kwargs = gradient_kwargs or {}
    reshape_info = []
    gradient_tapes = []
    processing_fns = []

    # TODO: need to reshape info based on g_tapes
    #    reshape_info.append(len(g_tapes))

    # Loop through the tapes and dys vector
    for tape in zip(jacobians):

        g_tapes, fn = gradient_fn(tape, **gradient_kwargs)

        reshape_info.append(len(g_tapes))
        processing_fns.append(fn)
        gradient_tapes.extend(g_tapes)

    def processing_fn(results, nums=None):
        vjps = []
        start = 0

        if nums is None:
            nums = [None] * len(jacobians)

        for t_idx in range(len(jacobians)):
            # extract the correct results from the flat list
            res_len = reshape_info[t_idx]
            res_t = results[start : start + res_len]
            start += res_len

            # postprocess results to compute the VJP
            vjp_ = processing_fns[t_idx](res_t, num=nums[t_idx])

            if vjp_ is None:
                if reduction == "append":
                    vjps.append(None)
                continue

            if isinstance(reduction, str):
                getattr(vjps, reduction)(vjp_)
            elif callable(reduction):
                reduction(vjps, vjp_)

        return vjps

    return gradient_tapes, processing_fn


def _execute(
    params,
    tapes=None,
    device=None,
    execute_fn=None,
    gradient_fn=None,
    gradient_kwargs=None,
    _n=1,
):  # pylint: disable=dangerous-default-value,unused-argument

    total_params = np.sum([len(p) for p in params])
    shape_dtype_structs = _extract_shape_dtype_structs(tapes, device)

    # Copy a given tape with operations and set parameters
    def cp_tape(t, a):
        tc = t.copy(copy_operations=True)
        tc.set_parameters(a)
        return tc

    @jax.custom_vjp
    def wrapped_exec(params):
        def wrapper(p):
            """Compute the forward pass."""
            new_tapes = [cp_tape(t, a) for t, a in zip(tapes, p)]
            with qml.tape.Unwrap(*new_tapes):
                res, _ = execute_fn(new_tapes, **gradient_kwargs)
            return res

        res = host_callback.call(wrapper, params, result_shape=shape_dtype_structs)
        return res

    def wrapped_exec_fwd(params):
        return wrapped_exec(params), params

    def wrapped_exec_bwd(params, g):

        if isinstance(gradient_fn, qml.gradients.gradient_transform):

            def non_diff_wrapper(args):
                """Compute the VJP in a non-differentiable manner."""
                p = args

                new_tapes = [cp_tape(t, a) for t, a in zip(tapes, p)]

                gradient_tapes = []
                processing_fns = []

                jacs = []
                # ---------------------
                for t in new_tapes:

                    g_tapes, fn = gradient_fn(t, **gradient_kwargs)

                    with qml.tape.Unwrap(*g_tapes):
                        res, _ = execute_fn(g_tapes)
                        r = fn(res)

                    res = jnp.hstack(r)
                    print(args, res)
                    jacs.append(res)
                # ---------------------

                return jacs

            args = tuple(params)

            # Break up batch_vjp:
            #
            # Compute the Jacobian
            # A. Hostcallback
            # 1. We gather the vjp_tapes
            # 2. Compute the Jacobian

            # shape = tapes[0].shape(device)
            # sh_lst = list(shape)
            # shape = tuple([sh_lst[1], sh_lst[0]])

            #jax_shape = shape_dtype_structs
            jax_shape = jax.ShapeDtypeStruct(tuple([1,4]), dtype)
            # for idx in range(len(jax_shape)):

            #     this_shape = jax_shape[idx]

            #     # If Probability, disregard the extra (1,) item in the shape
            #     # and consider only the second component
            #     if len(this_shape.shape) == 2 and this_shape.shape[0] == 1:
            #         jax_shape[idx].shape = (this_shape.shape[1],)

            #print("result: ", non_diff_wrapper(args))
            jacobians = host_callback.call(
                non_diff_wrapper,
                args,
                result_shape=jax_shape,
            )
            #jacobians = non_diff_wrapper(args)

            # B. Simple call
            # 3. Use batch_jacobian (that takes in the jacobians)

            # TODO: why do we need to squeeze?

            assert len(jacobians) == len(g)

            vjps = []
            for j, e in zip(jacobians, g):
                e = jnp.squeeze(e)
                vjps.append(j)# @ e)

            final_res = vjps[0][0]
            print(final_res, final_res.shape, jnp.array(final_res))
            return tuple([final_res])

            # vjp_tapes, processing_fn = compute_jacobian(
            #     new_tapes,
            #     gradient_fn,
            #     reduction="append",
            #     gradient_kwargs=gradient_kwargs,
            # )

            #res = processing_fn(partial_res)

            # Use g to recreate the VJP

            # param_idx = 0

            #res = vjps

            # Group the vjps based on the parameters of the tapes
            #for p in params:
            #    param_vjp = vjps[param_idx : param_idx + len(p)]
            #    res.append(param_vjp)
            #    param_idx += len(p)

            # Unwrap partial results into ndim=0 arrays to allow
            # differentiability with JAX
            # E.g.,
            # [DeviceArray([-0.9553365], dtype=float32), DeviceArray([0., 0.],
            # dtype=float32)]
            # is mapped to
            # [[DeviceArray(-0.9553365, dtype=float32)], [DeviceArray(0.,
            # dtype=float32), DeviceArray(0., dtype=float32)]].
            # need_unstacking = any(r.ndim != 0 for r in res)
            # if need_unstacking:
            #     res = [qml.math.unstack(x) for x in res]

            # return (tuple(res),)

        def jacs_wrapper(p):
            """Compute the jacs"""
            new_tapes = [cp_tape(t, a) for t, a in zip(tapes, p)]
            with qml.tape.Unwrap(*new_tapes):
                jacs = gradient_fn(new_tapes, **gradient_kwargs)
            return jacs

        shapes = [
            jax.ShapeDtypeStruct((len(t.measurements), len(p)), dtype)
            for t, p in zip(tapes, params)
        ]
        jacs = host_callback.call(jacs_wrapper, params, result_shape=shapes)
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

        fwd_shape_dtype_struct = _extract_shape_dtype_structs(tapes, device)

        # params should be the parameters for each tape queried prior, assert
        # to double-check
        assert len(tapes) == len(params)

        jacobian_shape = []
        for t, p in zip(tapes, params):
            shape = t.shape(device) + (len(p),)
            _dtype = _numeric_type_to_dtype(t.numeric_type)
            shape = [shape] if isinstance(shape, int) else shape
            o = jax.ShapeDtypeStruct(tuple(shape), _dtype)
            jacobian_shape.append(o)

        res, jacs = host_callback.call(
            wrapper,
            params,
            result_shape=tuple([fwd_shape_dtype_struct, jacobian_shape]),
        )
        return res, jacs

    def wrapped_exec_fwd(params):
        res, jacs = wrapped_exec(params)
        return res, tuple([jacs, params])

    def wrapped_exec_bwd(params, g):

        _raise_vector_valued_fwd(tapes)

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
