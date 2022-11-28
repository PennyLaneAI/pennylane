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

import numpy as np
import semantic_version
import pennylane as qml
from pennylane.interfaces import InterfaceUnsupportedError
from pennylane.interfaces.jax import _raise_vector_valued_fwd
from pennylane.interfaces.jax_jit import _numeric_type_to_dtype 

dtype = jnp.float64


def _extract_shape_dtype_structs_new(tapes, device):
    """Auxiliary function for defining the jax.ShapeDtypeStruct objects given
    the tapes and the device.

    The jax.pure_callback function expects jax.ShapeDtypeStruct objects to
    describe the output of the function call.
    """
    shape_dtypes = []

    def process_single_shape(shape, tape_dtype):
        return jax.ShapeDtypeStruct(tuple(shape), tape_dtype)

    for t in tapes:
        num_measurements = len(t.measurements)

        shape = t.shape(device)
        if num_measurements == 1:
            tape_dtype = _numeric_type_to_dtype(t.numeric_type)
            shape_and_dtype = process_single_shape(shape, tape_dtype)
        else:
            tape_dtype = tuple(_numeric_type_to_dtype(elem) for elem in t.numeric_type)
            shape_and_dtype = tuple(process_single_shape(s, d) for s, d in zip(shape, tape_dtype))

        shape_dtypes.append(shape_and_dtype)
    return shape_dtypes

# TODO: refactor!
def _extract_jac_shape(tapes, device):
    """Auxiliary function for defining the jax.ShapeDtypeStruct objects given
    the tapes and the device.

    The jax.pure_callback function expects jax.ShapeDtypeStruct objects to
    describe the output of the function call.
    """
    shape_dtypes = []

    def process_single_shape(shape, tape_dtype):
        return jax.ShapeDtypeStruct(tuple(shape), tape_dtype)

    for t in tapes:
        num_measurements = len(t.measurements)

        shape = t.shape(device)
        if num_measurements == 1:
            tape_dtype = _numeric_type_to_dtype(t.numeric_type)
            shape_and_dtype = process_single_shape(shape, tape_dtype)
        else:
            tape_dtype = tuple(_numeric_type_to_dtype(elem) for elem in t.numeric_type)
            shape_and_dtype = tuple(process_single_shape(s, d) for s, d in zip(shape, tape_dtype))

        if len(t.trainable_params) == 1:
            shape_dtypes.append(shape_and_dtype)
        else:
            s = [shape_and_dtype] * len(t.trainable_params)
            shape_dtypes.append(tuple(s))

    if len(tapes) == 1:
        return shape_dtypes[0]

    return tuple(shape_dtypes)

def jac_device_shape(tapes, dtype, num_params):

    # TODO: generalize
    shape_dtype_structs = jax.ShapeDtypeStruct((), dtype)
    if num_params == 1:
        abc = []
        for t in tapes:
            num_meas = len(t.measurements)
            shape = [shape_dtype_structs] if num_meas == 1 else tuple([shape_dtype_structs] * num_meas)
            abc.append(shape)
    else:
        abc = []
        for t in tapes:
            num_params = len(t.trainable_params)
            num_meas = len(t.measurements)
            if num_meas == 1:
                shape = tuple([shape_dtype_structs] * num_params)
            else:
                shape = [tuple([shape_dtype_structs] * num_params)] * num_meas
            abc.append(shape)

        if len(abc) == 1:
            abc = abc[0]
    return abc


def execute_new(tapes, device, execute_fn, gradient_fn, gradient_kwargs, _n=1, max_diff=1):
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

    parameters = tuple(list(t.get_parameters()) for t in tapes)

    if gradient_fn is None:
        return _execute_fwd_new(
            parameters,
            tapes=tapes,
            device=device,
            execute_fn=execute_fn,
            gradient_kwargs=gradient_kwargs,
            _n=_n,
        )

    return _execute_bwd_new(
        parameters,
        tapes=tapes,
        device=device,
        execute_fn=execute_fn,
        gradient_fn=gradient_fn,
        gradient_kwargs=gradient_kwargs,
        _n=_n,
    )


def _execute_bwd_new(
    params,
    tapes=None,
    device=None,
    execute_fn=None,
    gradient_fn=None,
    gradient_kwargs=None,
    _n=1,
):  # pylint: disable=dangerous-default-value,unused-argument

    # Copy a given tape with operations and set parameters
    def _copy_tape(t, a):
        tc = t.copy(copy_operations=True)
        tc.set_parameters(a)
        return tc

    @jax.custom_jvp
    def execute_wrapper(params):
        def wrapper(p):
            """Compute the forward pass."""
            new_tapes = [_copy_tape(t, a) for t, a in zip(tapes, p)]
            with qml.tape.Unwrap(*new_tapes):
                res, _ = execute_fn(new_tapes, **gradient_kwargs)
            return res

        shape_dtype_structs = _extract_shape_dtype_structs_new(tapes, device)
        res = jax.pure_callback(wrapper, shape_dtype_structs, params)
        return res

    def callback_fun_device(params, num_params):
        # Backward: Gradient function is a device method.
        def wrapper(params):
            new_tapes = [_copy_tape(t, a) for t, a in zip(tapes, params)]
            with qml.tape.Unwrap(*new_tapes):
                return gradient_fn(new_tapes, **gradient_kwargs)

        shape_dtype_structs = jac_device_shape(tapes, dtype, num_params)
        return jax.pure_callback(wrapper, shape_dtype_structs, params)

    def post_proc_res(jvps, multi_measurements, multi_params):
        res_jvps = []
        for m, p, j in zip(multi_measurements, multi_params, jvps):
            if m and p:
                res_jvps.append(tuple(jnp.squeeze(j_comp) for j_comp in j))
            else:
                res_jvps.append(jnp.squeeze(j))
        return res_jvps

    @jax.custom_jvp
    def callback_fun(params, num_tapes):
        def wrapper(params):
            new_tapes = [_copy_tape(t, a) for t, a in zip(tapes, params)]

            with qml.tape.Unwrap(*new_tapes):
                all_jacs = []
                for new_t in new_tapes:
                    jvp_tapes, fn = gradient_fn(new_t, shots=device.shots, **gradient_kwargs)

                    jacs = execute_fn(jvp_tapes)[0]
                    jacs = fn(jacs)
                    all_jacs.append(jacs)

            if len(all_jacs) == 1:
                return all_jacs[0]

            return all_jacs

        # TODO: This works for test_gradient:
        #abc = tuple([shape_dtype_structs] for _ in range(2))
        expected_shapes = _extract_jac_shape(tapes, device)
        return jax.pure_callback(wrapper, expected_shapes, params)

    @callback_fun.defjvp
    def callback_fun_jvp(primals, tangents):
        return primals, tangents

    @execute_wrapper.defjvp
    def execute_wrapper_jvp(primals, tangents):
        params = primals[0]

        new_tapes = [_copy_tape(t, a) for t, a in zip(tapes, params)]
        num_params = np.sum([len(p) for p in params])

        # Execution: execute the function first
        evaluation_results = execute_wrapper(params)
        multi_measurements = [len(tape.measurements) > 1 for tape in new_tapes]
        multi_params = [len(tape.trainable_params) > 1 for tape in new_tapes]

        # Backward: branch off based on the gradient function is a device method.
        if isinstance(gradient_fn, qml.gradients.gradient_transform):
            # Gradient function is a gradient transform
            multi_measurements = [len(tape.measurements) > 1 for tape in tapes]
            num_tapes = len(new_tapes)

            # If there are no tapes, no need to execute the device; sensible outputs should be generated by the
            # post-processing function
            if num_tapes == 0:
                jvps = processing_fn([])
            else:
                # Note: we have to split up the tape Unwrapping and the JVP computation because:
                # 1. Tape unwrapping has to happen in the callback: otherwise jitting is broken and Tracer objects are
                # converted to NumPy, something that raises an error (as of jax and jaxlib 0.3.25)
                #
                # 2. Passing in the tangents as an argument to the function called by the callback seems to raise a 
                # ValueError: Pure callbacks do not support transpose. Please use jax.custom_vjp to use callbacks while taking gradients.
                #
                # Solution: Use the callback to compute the jacobian and then separately compute the JVP using the
                # tangent
                res_from_callback = callback_fun(params, num_params)
                if len(tapes) == 1:
                    res_from_callback = [res_from_callback]

                #jvps = post_proc_res(res_from_callback, multi_measurements, multi_params)
                jvps = _compute_jvps(res_from_callback, tangents[0], multi_measurements)
                jvps = [jnp.squeeze(j) for j in jvps]

            # TODO: need?
        else:
            # Gradient function is a device method
            jacs = callback_fun_device(params, num_params)
            if len(tapes) == 1:
                jacs = [jacs]

            jvps = _compute_jvps(jacs, tangents[0], multi_measurements)
            jvps = post_proc_res(jvps, multi_measurements, multi_params)
            #jvps = [tuple([jnp.squeeze(j_comp) for j in jvps for j_comp in j])]

        return evaluation_results, jvps

    return execute_wrapper(params)


# The execute function in forward mode
def _execute_fwd_new(
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

        fwd_shape_dtype_struct = _extract_shape_dtype_structs_new(tapes, device)

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

        res, jacs = jax.pure_callback(
            wrapper, params, tuple([fwd_shape_dtype_struct, jacobian_shape])
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


def _compute_jvps(jacs, tangents, multi_measurements):
    """Compute the jvps of multiple tapes, directly for a Jacobian and tangents."""
    jvps = []
    for i, multi in enumerate(multi_measurements):
        if multi:
            jvps.append(qml.gradients.compute_jvp_multi(tangents[i], jacs[i], jitting=True))
        else:
            jvps.append(qml.gradients.compute_jvp_single(tangents[i], jacs[i], jitting=True))
    return jvps


def _to_jax(res):
    res_ = []
    for r in res:
        if not isinstance(r, tuple):
            res_.append(jnp.array(r))
        else:
            sub_r = []
            for r_i in r:
                sub_r.append(jnp.array(r_i))
            res_.append(tuple(sub_r))
    return res_
