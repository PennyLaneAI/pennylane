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
from functools import partial

import jax
import jax.numpy as jnp

import pennylane as qml
from pennylane.interfaces.jax import _compute_jvps
from pennylane.interfaces.jax_jit import _numeric_type_to_dtype
from pennylane.transforms import convert_to_numpy_parameters

dtype = jnp.float64
Zero = jax.custom_derivatives.SymbolicZero


def _set_copy_and_unwrap_tape(t, a, unwrap=True):
    """Copy a given tape with operations and set parameters"""
    tc = t.copy(copy_operations=True)
    tc.set_parameters(a, trainable_only=False)
    return convert_to_numpy_parameters(tc) if unwrap else tc


def set_parameters_on_copy_and_unwrap(tapes, params, unwrap=True):
    """Copy a set of tapes with operations and set parameters"""
    return tuple(_set_copy_and_unwrap_tape(t, a, unwrap=unwrap) for t, a in zip(tapes, params))


def _create_shape_dtype_struct(tape, device):
    """Auxiliary function for creating the shape and dtype object structure
    given a tape."""

    def process_single_shape(shape, tape_dtype):
        return jax.ShapeDtypeStruct(tuple(shape), tape_dtype)

    num_measurements = len(tape.measurements)
    shape = tape.shape(device)
    if num_measurements == 1:
        tape_dtype = _numeric_type_to_dtype(tape.numeric_type)
        return process_single_shape(shape, tape_dtype)

    tape_dtype = tuple(_numeric_type_to_dtype(elem) for elem in tape.numeric_type)
    return tuple(process_single_shape(s, d) for s, d in zip(shape, tape_dtype))


def _tapes_shape_dtype_tuple(tapes, device):
    """Auxiliary function for defining the jax.ShapeDtypeStruct objects given
    the tapes and the device.

    The jax.pure_callback function expects jax.ShapeDtypeStruct objects to
    describe the output of the function call.
    """
    shape_dtypes = []

    for t in tapes:
        shape_and_dtype = _create_shape_dtype_struct(t, device)
        shape_dtypes.append(shape_and_dtype)
    return shape_dtypes


def _jac_shape_dtype_tuple(tapes, device):
    """Auxiliary function for defining the jax.ShapeDtypeStruct objects when
    computing the jacobian associated with the tapes and the device.

    The jax.pure_callback function expects jax.ShapeDtypeStruct objects to
    describe the output of the function call.
    """
    shape_dtypes = []

    for t in tapes:
        shape_and_dtype = _create_shape_dtype_struct(t, device)

        if len(t.trainable_params) == 1:
            shape_dtypes.append(shape_and_dtype)
        else:
            num_measurements = len(t.measurements)
            if num_measurements == 1:
                s = [shape_and_dtype for _ in range(len(t.trainable_params))]
                shape_dtypes.append(tuple(s))
            else:
                s = [tuple(_s for _ in range(len(t.trainable_params))) for _s in shape_and_dtype]
                shape_dtypes.append(tuple(s))

    if len(tapes) == 1:
        return shape_dtypes[0]

    return tuple(shape_dtypes)


def _filter_zeros_tangents(tangents):
    non_zeros_tangents = [[t for t in tangent if isinstance(t, Zero)] for tangent in tangents]

    return non_zeros_tangents


def execute(tapes, device, execute_fn, gradient_fn, gradient_kwargs, _n=1, max_diff=1):
    """Execute a batch of tapes with JAX parameters on a device.

    Args:
        tapes (Sequence[.QuantumTape]): batch of tapes to execute
        device (pennylane.Device): Device to use to execute the batch of tapes.
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
        max_diff (int): If ``gradient_fn`` is a gradient transform, this option specifies
            the maximum order of derivatives to support. Increasing this value allows
            for higher order derivatives to be extracted, at the cost of additional
            (classical) computational overhead during the backwards pass.

    Returns:
        list[list[float]]: A nested list of tape results. Each element in
        the returned list corresponds in order to the provided tapes.
    """
    # pylint: disable=unused-argument

    if any(
        m.return_type in (qml.measurements.Counts, qml.measurements.AllCounts)
        for t in tapes
        for m in t.measurements
    ):
        # Obtaining information about the shape of the Counts measurements is
        # not implemented and is required for the callback logic
        raise NotImplementedError("The JAX-JIT interface doesn't support qml.counts.")

    parameters = tuple(list(t.get_parameters(trainable_only=False)) for t in tapes)

    if gradient_fn is None:
        return _execute_fwd(
            parameters,
            tapes=tapes,
            device=device,
            execute_fn=execute_fn,
            gradient_kwargs=gradient_kwargs,
            _n=_n,
        )

    return _execute_bwd(
        parameters,
        tapes=tapes,
        device=device,
        execute_fn=execute_fn,
        gradient_fn=gradient_fn,
        gradient_kwargs=gradient_kwargs,
        _n=_n,
        max_diff=max_diff,
    )


def _execute_bwd(
    params,
    tapes=None,
    device=None,
    execute_fn=None,
    gradient_fn=None,
    gradient_kwargs=None,
    _n=1,
    max_diff=2,
):  # pylint: disable=dangerous-default-value,unused-argument
    @jax.custom_jvp
    def execute_wrapper(params):
        shape_dtype_structs = _tapes_shape_dtype_tuple(tapes, device)

        def wrapper(p):
            """Compute the forward pass."""
            new_tapes = set_parameters_on_copy_and_unwrap(tapes, p)
            res, _ = execute_fn(new_tapes, **gradient_kwargs)

            # When executed under `jax.vmap` the `result_shapes_dtypes` will contain
            # the shape without the vmap dimensions, while the function here will be
            # executed with objects containing the vmap dimensions. So res[i].ndim
            # will have an extra dimension for every `jax.vmap` wrapping this execution.
            #
            # The execute_fn will return an object with shape `(n_observables, batches)`
            # but the default behaviour for `jax.pure_callback` is to add the extra
            # dimension at the beginning, so `(batches, n_observables)`. So in here
            # we detect with the heuristic above if we are executing under vmap and we
            # swap the order in that case.
            res = jax.tree_map(lambda r, s: r.T if r.ndim > s.ndim else r, res, shape_dtype_structs)
            return res

        res = jax.pure_callback(wrapper, shape_dtype_structs, params, vectorized=True)
        return res

    @partial(execute_wrapper.defjvp, symbolic_zeros=True)
    def execute_wrapper_jvp(primals, tangents):
        # pylint: disable=unused-variable
        params = primals[0]

        # Select the trainable params. Non-trainable params contribute a 0 gradient.
        list_trainable_parameters = [
            [idx for idx, t in enumerate(tangent) if isinstance(t, Zero)] for tangent in tangents[0]
        ]
        for trainable_params, tape in zip(list_trainable_parameters, tapes):
            tape.trainable_params = trainable_params

        multi_measurements = [len(tape.measurements) > 1 for tape in tapes]

        # Execution: execute the function first
        evaluation_results = execute_wrapper(params)

        # Backward: branch off based on the gradient function is a device method.
        if isinstance(gradient_fn, qml.gradients.gradient_transform):
            # Gradient function is a gradient transform
            if _n == max_diff:
                jacobians_from_callback = _grad_transform_jac_via_callback(params, device)

                if len(tapes) == 1:
                    jacobians_from_callback = [jacobians_from_callback]

                tangents_trainable = _filter_zeros_tangents(tangents[0])

                jvps = _compute_jvps(
                    jacobians_from_callback, tangents_trainable, multi_measurements
                )

            else:
                new_tapes = set_parameters_on_copy_and_unwrap(tapes, params, unwrap=False)
                all_jacs = []
                for new_t in new_tapes:
                    jvp_tapes, res_processing_fn = gradient_fn(
                        new_t, shots=device.shots, **gradient_kwargs
                    )
                    jacs = execute(
                        jvp_tapes,
                        device,
                        execute_fn,
                        gradient_fn,
                        gradient_kwargs,
                        _n=_n + 1,
                        max_diff=max_diff,
                    )
                    jacs = res_processing_fn(jacs)
                    all_jacs.append(jacs)

                tangents_trainable = _filter_zeros_tangents(tangents[0])

                jvps = _compute_jvps(all_jacs, tangents_trainable, multi_measurements)
        else:
            # Gradient function is a device method
            res_from_callback = _device_method_jac_via_callback(params, device)

            if len(tapes) == 1:
                res_from_callback = [res_from_callback]

            tangents_trainable = _filter_zeros_tangents(tangents[0])

            jvps = _compute_jvps(res_from_callback, tangents_trainable, multi_measurements)

        return evaluation_results, jvps

    def _grad_transform_jac_via_callback(params, device):
        """Perform a callback to compute the jacobian of tapes using a gradient transform (e.g., parameter-shift or
        finite differences grad transform).

        Note: we are not using the batch_jvp pipeline and rather split the steps of unwrapping tapes and the JVP
        computation because:

        1. Tape unwrapping has to happen in the callback: otherwise jitting is broken and Tracer objects
        are converted to NumPy, something that raises an error;

        2. Passing in the tangents as an argument to the wrapper function called by the jax.pure_callback raises an
        error (as of jax and jaxlib 0.3.25):
        ValueError: Pure callbacks do not support transpose. Please use jax.custom_vjp to use callbacks while
        taking gradients.

        Solution: Use the callback to compute the jacobian and then separately compute the JVP using the
        tangent.
        """

        def wrapper(params):
            new_tapes = set_parameters_on_copy_and_unwrap(tapes, params)
            all_jacs = []
            for new_t in new_tapes:
                jvp_tapes, res_processing_fn = gradient_fn(
                    new_t, shots=device.shots, **gradient_kwargs
                )
                jacs = execute_fn(jvp_tapes)[0]
                jacs = res_processing_fn(jacs)
                all_jacs.append(jacs)

            if len(all_jacs) == 1:
                return all_jacs[0]

            return all_jacs

        expected_shapes = _jac_shape_dtype_tuple(tapes, device)
        res = jax.pure_callback(wrapper, expected_shapes, params)
        return res

    def _device_method_jac_via_callback(params, device):
        """Perform a callback to compute the jacobian of tapes using a device method (e.g., adjoint).

        Note: we are not using the batch_jvp pipeline and rather split the steps of unwrapping tapes and the JVP
        computation because:

        1. Tape unwrapping has to happen in the callback: otherwise jitting is broken and Tracer objects
        are converted to NumPy, something that raises an error;

        2. Passing in the tangents as an argument to the wrapper function called by the jax.pure_callback raises an
        error (as of jax and jaxlib 0.3.25):
        ValueError: Pure callbacks do not support transpose. Please use jax.custom_vjp to use callbacks while
        taking gradients.

        Solution: Use the callback to compute the jacobian and then separately compute the JVP using the
        tangent.
        """

        def wrapper(params):
            new_tapes = set_parameters_on_copy_and_unwrap(tapes, params)
            return gradient_fn(new_tapes, **gradient_kwargs)

        shape_dtype_structs = _jac_shape_dtype_tuple(tapes, device)
        return jax.pure_callback(wrapper, shape_dtype_structs, params)

    return execute_wrapper(params)


# The execute function in forward mode
def _execute_fwd(
    params,
    tapes=None,
    device=None,
    execute_fn=None,
    gradient_kwargs=None,
    _n=1,
):  # pylint: disable=dangerous-default-value,unused-argument
    """The auxiliary execute function for cases when the user requested
    jacobians to be computed in forward mode (e.g. adjoint) or when no gradient function was
    provided. This function does not allow multiple derivatives. It currently does not support shot vectors
    because adjoint jacobian for default qubit does not support it."""

    # pylint: disable=unused-variable
    @jax.custom_jvp
    def execute_wrapper(params):
        def wrapper(p):
            """Compute the forward pass."""
            new_tapes = set_parameters_on_copy_and_unwrap(tapes, p)
            res, jacs = execute_fn(new_tapes, **gradient_kwargs)
            return res, jacs

        shape_dtype_structs = _tapes_shape_dtype_tuple(tapes, device)
        jac_shape_dtype_structs = _jac_shape_dtype_tuple(tapes, device)
        res, jacs = jax.pure_callback(
            wrapper, (shape_dtype_structs, jac_shape_dtype_structs), params
        )
        return res, jacs

    @partial(execute_wrapper.defjvp, symbolic_zeros=True)
    def execute_wrapper_jvp(primals, tangents):
        """Primals[0] are parameters as Jax tracers and tangents[0] is a list of tangent vectors as Jax tracers."""
        list_trainable_parameters = [
            [idx for idx, t in enumerate(tangent) if isinstance(t, Zero)] for tangent in tangents[0]
        ]

        for trainable_params, tape in zip(list_trainable_parameters, tapes):
            tape.trainable_params = trainable_params

        res, jacs = execute_wrapper(primals[0])
        multi_measurements = [len(tape.measurements) > 1 for tape in tapes]

        jacs_ = [jacs] if len(tapes) == 1 else jacs

        tangents = _filter_zeros_tangents(tangents[0])
        jvps = _compute_jvps(jacs_, tangents, multi_measurements)

        return (res, jacs), (jvps, jacs)

    res = execute_wrapper(params)

    return res[0]
