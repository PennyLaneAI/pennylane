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
from typing import Union, Tuple

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
    tc = t.bind_new_parameters(a, list(range(len(a))))
    return convert_to_numpy_parameters(tc) if unwrap else tc


def set_parameters_on_copy_and_unwrap(tapes, params, unwrap=True):
    """Copy a set of tapes with operations and set parameters"""
    return tuple(_set_copy_and_unwrap_tape(t, a, unwrap=unwrap) for t, a in zip(tapes, params))


def _create_shape_dtype_struct(
    tape: "qml.tape.QuantumScript", device: Union["qml.Device", "qml.devices.experimental.Device"]
) -> Union[Tuple, jax.ShapeDtypeStruct]:
    """Auxiliary function for creating the shape and dtype object structure
    given a tape."""

    shape = tape.shape(device)
    if len(tape.measurements) == 1:
        tape_dtype = _numeric_type_to_dtype(tape.numeric_type)
        return jax.ShapeDtypeStruct(tuple(shape), tape_dtype)

    tape_dtype = tuple(_numeric_type_to_dtype(elem) for elem in tape.numeric_type)
    return tuple(jax.ShapeDtypeStruct(tuple(s), d) for s, d in zip(shape, tape_dtype))


def _jac_shape_dtype_struct(
    tape: "qml.tape.QuantumScript", device: Union["qml.Device", "qml.devices.experimental.Device"]
) -> Union[Tuple, jax.ShapeDtypeStruct]:
    """The shape of a jacobian for a single tape given a device.

    Args:
        tape (QuantumTape): the tape who's output we want to determine
        device (Device): the device used to execute the tape.

    >>> tape = qml.tape.QuantumScript([qml.RX(1.0, wires=0)], [qml.expval(qml.PauliX(0)), qml.probs(wires=(0,1))])
    >>> dev = qml.devices.experimental.DefaultQubit2()
    >>> _jac_shape_dtype_struct(tape, dev)
    (ShapeDtypeStruct(shape=(), dtype=float64),
    ShapeDtypeStruct(shape=(4,), dtype=float64))
    """
    shape_and_dtype = _create_shape_dtype_struct(tape, device)
    if len(tape.trainable_params) == 1:
        return shape_and_dtype
    if len(tape.measurements) == 1:
        return tuple(shape_and_dtype for _ in tape.trainable_params)
    return tuple(tuple(_s for _ in tape.trainable_params) for _s in shape_and_dtype)


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
    is_experimental_device = isinstance(device, qml.devices.experimental.Device)

    @jax.custom_jvp
    def execute_wrapper(params):
        shape_dtype_structs = tuple(_create_shape_dtype_struct(t, device) for t in tapes)

        def wrapper(p):
            """Compute the forward pass."""
            new_tapes = set_parameters_on_copy_and_unwrap(tapes, p)
            res, _ = execute_fn(new_tapes, **gradient_kwargs)
            res = tuple(res)

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
            return jax.tree_map(
                lambda r, s: r.T if r.ndim > s.ndim else r, res, shape_dtype_structs
            )

        return jax.pure_callback(wrapper, shape_dtype_structs, params, vectorized=True)

    @partial(execute_wrapper.defjvp, symbolic_zeros=True)
    def execute_wrapper_jvp(primals, tangents):
        """Calculate the jvp in such a way that we can bind it to jax.

        Closure Variables:
            tapes, device, gradient_fn, gradient_kwargs, execute_fn, _n, max_diff
        """
        params = primals[0]

        # Select the trainable params. Non-trainable params contribute a 0 gradient.
        for tangent, tape in zip(tangents[0], tapes):
            trainable_params = tuple(
                idx for idx, t in enumerate(tangent) if not isinstance(t, Zero)
            )
            tape.trainable_params = trainable_params

        if not isinstance(gradient_fn, qml.gradients.gradient_transform):
            # is a device method
            jacobians = _device_method_jac_via_callback(params, device)
        elif _n == max_diff:
            jacobians = _grad_transform_jac_via_callback(params, device)
        else:
            new_tapes = set_parameters_on_copy_and_unwrap(tapes, params, unwrap=False)
            jacobians = []
            for new_t in new_tapes:
                jvp_tapes, res_processing_fn = gradient_fn(
                    new_t,
                    shots=new_t.shots if is_experimental_device else device.shots,
                    **gradient_kwargs
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
                jacobians.append(jacs)

        tangents_trainable = tuple(
            tuple(t for t in tangent if not isinstance(t, Zero)) for tangent in tangents[0]
        )
        multi_measurements = [len(tape.measurements) > 1 for tape in tapes]
        jvps = _compute_jvps(jacobians, tangents_trainable, multi_measurements)

        results = execute_wrapper(params)
        return results, jvps

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

        Closure Variables:
            tapes, gradient_fn, gradient_kwargs
        """

        def wrapper(params):
            new_tapes = set_parameters_on_copy_and_unwrap(tapes, params)
            all_jacs = []
            for new_t in new_tapes:
                jvp_tapes, res_processing_fn = gradient_fn(
                    new_t,
                    shots=new_t.shots if is_experimental_device else device.shots,
                    **gradient_kwargs
                )
                jacs = execute_fn(jvp_tapes)[0]
                jacs = res_processing_fn(jacs)
                all_jacs.append(jacs)

            return tuple(all_jacs)

        expected_shapes = tuple(_jac_shape_dtype_struct(t, device) for t in tapes)
        return jax.pure_callback(wrapper, expected_shapes, params)

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

        Closure Variables:
            tapes, gradient_fn, gradient_kwargs
        """

        def wrapper(params):
            new_tapes = set_parameters_on_copy_and_unwrap(tapes, params)
            return gradient_fn(new_tapes, **gradient_kwargs)

        shape_dtype_structs = tuple(_jac_shape_dtype_struct(t, device) for t in tapes)
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
            return execute_fn(new_tapes, **gradient_kwargs)

        shape_dtype_structs = tuple(_create_shape_dtype_struct(t, device) for t in tapes)

        jac_shape_dtype_structs = tuple(_jac_shape_dtype_struct(t, device) for t in tapes)
        jac_shape_dtype_structs = (
            jac_shape_dtype_structs[0] if len(tapes) == 1 else jac_shape_dtype_structs
        )

        return jax.pure_callback(wrapper, (shape_dtype_structs, jac_shape_dtype_structs), params)

    @partial(execute_wrapper.defjvp, symbolic_zeros=True)
    def execute_wrapper_jvp(primals, tangents):
        """Primals[0] are parameters as Jax tracers and tangents[0] is a list of tangent vectors as Jax tracers."""

        # Get the original trainable parameters and the trainable parameters from symbolic zeros
        original_trainable_parameters = [tape.trainable_params for tape in tapes]

        switch_trainable = []
        for tangent, tape in zip(tangents[0], tapes):
            new_trainable_params = [idx for idx, t in enumerate(tangent) if not isinstance(t, Zero)]
            if tape.trainable_params != new_trainable_params:
                tape.trainable_params = new_trainable_params
                switch_trainable.append(True)
            else:
                switch_trainable.append(False)

        # Forward execution with the right trainable parameters
        res, jacs = execute_wrapper(primals[0])

        jacs_ = [jacs] if len(tapes) == 1 else jacs

        updated_jacs = []

        # Add zeros in the jacobians if the trainable params were switched
        for tape_index, (tape, switch) in enumerate(zip(tapes, switch_trainable)):
            if not switch:
                updated_jacs.append(jacs_[tape_index])
            else:
                intermediate_jacs = []

                shape_dtype = _jac_shape_dtype_struct(tape, device)

                if len(tape.measurements) > 1:
                    if isinstance(shape_dtype[0][0], tuple):
                        shape_dtype = shape_dtype[0]
                    jac_empty = tuple(
                        jnp.zeros(shape=tensor.shape, dtype=tensor.dtype) for tensor in shape_dtype
                    )
                else:
                    # not exactly sure why we need this line
                    if isinstance(shape_dtype, tuple):
                        shape_dtype = shape_dtype[0]
                    jac_empty = jnp.zeros(shape_dtype.shape, shape_dtype.dtype)

                for param_idx in original_trainable_parameters[tape_index]:
                    p_jac = jacs_[tape_index] if param_idx in tape.trainable_params else jac_empty
                    intermediate_jacs.append(p_jac)
                updated_jacs.append(tuple(intermediate_jacs))

        updated_jacs = updated_jacs[0] if len(tapes) == 1 else tuple(updated_jacs)

        # Get the jvps
        multi_measurements = [len(tape.measurements) > 1 for tape in tapes]
        tangents = [[t for t in tangent if not isinstance(t, Zero)] for tangent in tangents[0]]
        jvps = _compute_jvps(jacs_, tangents, multi_measurements)

        return (res, updated_jacs), (jvps, updated_jacs)

    return execute_wrapper(params)[0]
