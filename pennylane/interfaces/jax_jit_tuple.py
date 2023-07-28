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
    return tuple(shape_dtypes)


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
    non_zeros_tangents = [[t for t in tangent if not isinstance(t, Zero)] for tangent in tangents]

    return non_zeros_tangents


def execute(tapes, execute_fn, vjp_fn, device=None):
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

    return _execute_bwd(
        parameters,
        tapes,
        execute_fn,
        vjp_fn,
        device,
    )


def _execute_bwd(
    parameters,
    tapes,
    execute_fn,
    vjp_fn,
    device,
):
    @jax.custom_jvp
    def execute_wrapper(params):
        new_tapes = set_parameters_on_copy_and_unwrap(tapes, params, unwrap=False)
        return execute_fn(new_tapes)

    @partial(execute_wrapper.defjvp, symbolic_zeros=True)
    def execute_wrapper_jvp(primals, tangents):
        for tangent, tape in zip(tangents[0], tapes):
            tape.trainable_params = tuple(
                idx for idx, t in enumerate(tangent) if not isinstance(t, Zero)
            )
        new_tapes = set_parameters_on_copy_and_unwrap(tapes, primals[0], unwrap=False)
        res, jvps = vjp_fn.execute_and_compute_jvp(new_tapes, tangents[0])
        return res, jvps

    return execute_wrapper(parameters)
