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
# pylint: disable=too-many-arguments, no-member
from functools import partial
from typing import Union, Tuple
import jax
import jax.numpy as jnp

import pennylane as qml
from pennylane.interfaces.jax import _compute_jvps
from pennylane.transforms import convert_to_numpy_parameters

from .jax import set_parameters_on_copy_and_unwrap

dtype = jnp.float64
Zero = jax.custom_derivatives.SymbolicZero


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


def _set_copy_and_unwrap_tape(t, a, unwrap=True):
    """Copy a given tape with operations and set parameters"""
    tc = t.bind_new_parameters(a, list(range(len(a))))
    return convert_to_numpy_parameters(tc) if unwrap else tc


def _create_shape_dtype_struct(tape: "qml.tape.QuantumScript", device: "qml.Device"):
    """Auxiliary function for creating the shape and dtype object structure
    given a tape."""

    shape = tape.shape(device)
    if len(tape.measurements) == 1:
        tape_dtype = _numeric_type_to_dtype(tape.numeric_type)
        return jax.ShapeDtypeStruct(tuple(shape), tape_dtype)

    tape_dtype = tuple(_numeric_type_to_dtype(elem) for elem in tape.numeric_type)
    return tuple(jax.ShapeDtypeStruct(tuple(s), d) for s, d in zip(shape, tape_dtype))


def _jac_shape_dtype_struct(tape: "qml.tape.QuantumScript", device: "qml.Device"):
    """The shape of a jacobian for a single tape given a device.

    Args:
        tape (QuantumTape): the tape who's output we want to determine
        device (Device): the device used to execute the tape.

    >>> tape = qml.tape.QuantumScript([qml.RX(1.0, wires=0)], [qml.expval(qml.PauliX(0)), qml.probs(0)])
    >>> dev = qml.devices.experimental.DefaultQubit2()
    >>> _jac_shape_dtype_struct(tape, dev)
    (ShapeDtypeStruct(shape=(), dtype=float64),
    ShapeDtypeStruct(shape=(2,), dtype=float64))
    >>> tapes, fn = qml.gradients.param_shift(tape)
    >>> fn(dev.execute(tapes))
    (array(0.), array([-0.42073549,  0.42073549]))
    """
    shape_and_dtype = _create_shape_dtype_struct(tape, device)
    if len(tape.trainable_params) == 1:
        return shape_and_dtype
    if len(tape.measurements) == 1:
        return tuple(shape_and_dtype for _ in tape.trainable_params)
    return tuple(tuple(_s for _ in tape.trainable_params) for _s in shape_and_dtype)


def _empty_jacs_for_shape(
    shape_dtype: Union[Tuple, jax.ShapeDtypeStruct]
) -> Union[Tuple, jnp.array]:
    """Converts a nested tuple containing ``jax.ShapeDtypeStruct`` into a nested tuples
    containing zeros arrays of the appropriate shape.

    """
    if isinstance(shape_dtype, tuple):
        return tuple(_empty_jacs_for_shape(struct) for struct in shape_dtype)
    return jnp.zeros(shape_dtype.shape, shape_dtype.dtype)


def _switched_jacobian(tape, device, original_trainable_parameters, jacs):
    """Adds in jacobians with of all zeros for parameters that had zero tangent.

    Used in ``execute_wrapper_jvp`` for grad_on_execution=True with device derivatives.

    Note that this adds an additional nesting dimension to ``jacs``, with one jacobian
    for each original trainable parameter. I am unsure why this works.

    >>> dev = qml.devices.experimental.DefaultQubit2()
    >>> config = qml.devices.experimental.ExecutionConfig(gradient_method="adjoint")
    >>> tape = qml.tape.QuantumTape([qml.RY(1.0, 0), qml.RX(0.6, 0), qml.RX(0.7, 0)], [qml.expval(qml.PauliZ(0))])
    >>> tape.trainable_params = [0, 2]
    >>> jac = dev.compute_derivatives(tape, config)
    >>> jac
    (array(-0.2250925), array(-0.52061271))
    >>> _switched_jacobian(tape, dev, [0,1,2], jac)
    ((array(-0.2250925), array(-0.52061271)),
    (Array(0., dtype=float64), Array(0., dtype=float64)),
    (array(-0.2250925), array(-0.52061271)))

    """
    intermediate_jacs = []

    shape_dtype = _jac_shape_dtype_struct(tape, device)

    for param_idx in original_trainable_parameters:
        if param_idx in tape.trainable_params:
            p_jac = jacs
        else:
            p_jac = _empty_jacs_for_shape(shape_dtype)
        intermediate_jacs.append(p_jac)
    return tuple(intermediate_jacs)


# pylint: disable=unused-argument
def _device_method_jac_via_callback(
    params, tapes, device, execute_fn, gradient_fn, gradient_kwargs, _n, max_diff
):
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

    def wrapper(inner_params):
        new_tapes = set_parameters_on_copy_and_unwrap(tapes, inner_params, unwrap=False)
        return gradient_fn(new_tapes, **gradient_kwargs)

    shape_dtype_structs = tuple(_jac_shape_dtype_struct(t, device) for t in tapes)
    return jax.pure_callback(wrapper, shape_dtype_structs, params)


# pylint: disable=unused-argument
def _grad_transform_jac_via_callback(
    params, tapes, device, execute_fn, gradient_fn, gradient_kwargs, _n, max_diff
):
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

    def wrapper(inner_params):
        new_tapes = set_parameters_on_copy_and_unwrap(tapes, inner_params, unwrap=False)
        all_jacs = []
        for new_t in new_tapes:
            jvp_tapes, res_processing_fn = gradient_fn(new_t, **gradient_kwargs)
            jacs = execute_fn(jvp_tapes)[0]
            jacs = res_processing_fn(jacs)
            all_jacs.append(jacs)

        return tuple(all_jacs)

    expected_shapes = tuple(_jac_shape_dtype_struct(t, device) for t in tapes)
    return jax.pure_callback(wrapper, expected_shapes, params)


def _grad_transform_jac_no_callback(
    params, tapes, device, execute_fn, gradient_fn, gradient_kwargs, _n, max_diff
):
    """Performs the gradient transform in a differentiable manner."""
    new_tapes = set_parameters_on_copy_and_unwrap(tapes, params, unwrap=False)
    jacobians = []
    for new_t in new_tapes:
        jvp_tapes, res_processing_fn = gradient_fn(new_t, **gradient_kwargs)
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
    return tuple(jacobians)


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
    def execute_wrapper(inner_params):
        shape_dtype_structs = tuple(_create_shape_dtype_struct(t, device) for t in tapes)

        def wrapper(p):
            """Compute the forward pass."""
            new_tapes = set_parameters_on_copy_and_unwrap(tapes, p, unwrap=False)
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

        return jax.pure_callback(wrapper, shape_dtype_structs, inner_params, vectorized=True)

    # execute_wrapper_jvp merely registered, not used.
    # pylint: disable=unused-variable
    @partial(execute_wrapper.defjvp, symbolic_zeros=True)
    def execute_wrapper_jvp(primals, tangents):
        """Calculate the jvp in such a way that we can bind it to jax.

        Closure Variables:
            tapes, device, gradient_fn, gradient_kwargs, execute_fn, _n, max_diff
        """
        params = primals[0]

        # Select the trainable params. Non-trainable params contribute a 0 gradient.
        for tangent, tape in zip(tangents[0], tapes):
            tape.trainable_params = tuple(
                idx for idx, t in enumerate(tangent) if not isinstance(t, Zero)
            )

        if not isinstance(gradient_fn, qml.gradients.gradient_transform):
            jacobians_func = _device_method_jac_via_callback
        elif _n == max_diff:
            jacobians_func = _grad_transform_jac_via_callback
        else:
            jacobians_func = _grad_transform_jac_no_callback

        jacobians = jacobians_func(
            params, tapes, device, execute_fn, gradient_fn, gradient_kwargs, _n, max_diff
        )

        tangents_trainable = tuple(
            tuple(t for t in tangent if not isinstance(t, Zero)) for tangent in tangents[0]
        )
        multi_measurements = [len(tape.measurements) > 1 for tape in tapes]
        jvps = _compute_jvps(jacobians, tangents_trainable, multi_measurements)

        results = execute_wrapper(params)
        return results, jvps

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
            new_tapes = set_parameters_on_copy_and_unwrap(tapes, p, unwrap=False)
            return execute_fn(new_tapes, **gradient_kwargs)

        shape_dtype_structs = tuple(_create_shape_dtype_struct(t, device) for t in tapes)

        jac_shape = tuple(_jac_shape_dtype_struct(t, device) for t in tapes)
        jac_shape = jac_shape[0] if len(tapes) == 1 else jac_shape

        return jax.pure_callback(wrapper, (shape_dtype_structs, jac_shape), params)

    @partial(execute_wrapper.defjvp, symbolic_zeros=True)
    def execute_wrapper_jvp(primals, tangents):
        """Primals[0] are parameters as Jax tracers and tangents[0] is a list of tangent vectors as Jax tracers.

        Closure Variables:
            tapes, execute_wrapper
        """

        # Get the original trainable parameters and the trainable parameters from symbolic zeros
        original_trainable_parameters = [tape.trainable_params for tape in tapes]

        for tangent, tape in zip(tangents[0], tapes):
            tape.trainable_params = [
                idx for idx, t in enumerate(tangent) if not isinstance(t, Zero)
            ]

        # Forward execution with the right trainable parameters
        res, jacs = execute_wrapper(primals[0])

        jacs_ = [jacs] if len(tapes) == 1 else jacs

        updated_jacs = []

        for tape, tape_jac, orig_trainable in zip(tapes, jacs_, original_trainable_parameters):
            if tape.trainable_params != orig_trainable:
                # Add zeros in the jacobians if the trainable params were switched
                updated_jacs.append(_switched_jacobian(tape, device, orig_trainable, tape_jac))
            else:
                updated_jacs.append(tape_jac)

        updated_jacs = updated_jacs[0] if len(tapes) == 1 else tuple(updated_jacs)

        # Get the jvps
        multi_measurements = [len(tape.measurements) > 1 for tape in tapes]
        tangents = [[t for t in tangent if not isinstance(t, Zero)] for tangent in tangents[0]]
        jvps = _compute_jvps(jacs_, tangents, multi_measurements)

        return (res, updated_jacs), (jvps, updated_jacs)

    return execute_wrapper(params)[0]
