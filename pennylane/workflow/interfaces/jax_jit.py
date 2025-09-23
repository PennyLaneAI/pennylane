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
This module contains functions for binding JVPs or VJPs to JAX when using JIT.

For information on registering VJPs and JVPs, please see the module documentation for ``jax.py``.

When using JAX-JIT, we cannot convert arrays to numpy or act on their concrete values without
using ``jax.pure_callback``.

For example:

>>> def f(x):
...     return qml.math.unwrap(x)
>>> x = jax.numpy.array(1.0)
>>> jax.jit(f)(x)
ValueError: Converting a JAX array to a NumPy array not supported when using the JAX JIT.
>>> def g(x):
...     expected_output_shape = jax.ShapeDtypeStruct((), jax.numpy.float64)
...     return jax.pure_callback(f, expected_output_shape, x, vmap_method="sequential")
>>> jax.jit(g)(x)
Array(1., dtype=float64)

Note that we must provide the expected output shape for the function to use pure callbacks.

"""
# pylint: disable=unused-argument,too-many-arguments
from functools import partial

import jax
import jax.numpy as jnp

import pennylane as qml
from pennylane.typing import ResultBatch
from pennylane.workflow.jacobian_products import _compute_jvps

from .jax import _NonPytreeWrapper

Zero = jax.custom_derivatives.SymbolicZero


def _to_jax(result: qml.typing.ResultBatch) -> qml.typing.ResultBatch:
    """Converts an arbitrary result batch to one with jax arrays.
    Args:
        result (ResultBatch): a nested structure of lists, tuples, and numpy arrays
    Returns:
        ResultBatch: a nested structure of tuples, and jax arrays

    """
    if isinstance(result, dict):
        return {key: _to_jax(value) for key, value in result.items()}
    if isinstance(result, (list, tuple)):
        return tuple(_to_jax(r) for r in result)
    return jnp.array(result)


def _set_all_parameters_on_copy(tapes, params):
    """Copy a set of tapes with operations and set all parameters"""
    return tuple(
        t.bind_new_parameters(a, list(range(len(a)))) for t, a in zip(tapes, params, strict=True)
    )


def _set_trainable_parameters_on_copy(tapes, params):
    """Copy a set of tapes with operations and set all trainable parameters"""
    return tuple(
        t.bind_new_parameters(a, t.trainable_params) for t, a in zip(tapes, params, strict=True)
    )


def _jax_dtype(m_type):
    if m_type == int:
        return jnp.int64 if jax.config.jax_enable_x64 else jnp.int32
    if m_type == float:
        return jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
    if m_type == complex:
        return jnp.complex128 if jax.config.jax_enable_x64 else jnp.complex64
    return jnp.dtype(m_type)


def _get_counts_shape(mp: "qml.measurements.CountsMP", num_device_wires=0):
    num_wires = len(mp.wires) if mp.wires else num_device_wires
    outcome_counts = {}
    binary_pattern = "{0:0" + str(num_wires) + "b}"
    for outcome in range(2**num_wires):
        outcome_binary = binary_pattern.format(outcome)
        outcome_counts[outcome_binary] = jax.core.ShapedArray((), _jax_dtype(int))

    return outcome_counts


def _result_shape_dtype_struct(tape: "qml.tape.QuantumScript", device: "qml.devices.Device"):
    """Auxiliary function for creating the shape and dtype object structure
    given a tape."""

    num_device_wires = len(device.wires) if device.wires else len(tape.wires)

    def struct(mp, shots):
        # depends on num_device_wires and tape.batch_size from closure
        if isinstance(mp, qml.measurements.CountsMP):
            counts_shape = _get_counts_shape(mp, num_device_wires=num_device_wires)
            if tape.batch_size:
                return tuple(counts_shape for _ in range(tape.batch_size))
            return counts_shape

        mp_shape = mp.shape(shots=shots, num_device_wires=num_device_wires)
        if tape.batch_size:
            mp_shape = (tape.batch_size, *mp_shape)
        return jax.ShapeDtypeStruct(mp_shape, _jax_dtype(mp.numeric_type))

    shape = []
    for s in tape.shots if tape.shots else [None]:
        shots_shape = tuple(struct(mp, s) for mp in tape.measurements)

        shots_shape = shots_shape[0] if len(shots_shape) == 1 else tuple(shots_shape)
        shape.append(shots_shape)

    return tuple(shape) if tape.shots.has_partitioned_shots else shape[0]


def _jac_shape_dtype_struct(tape: "qml.tape.QuantumScript", device: "qml.devices.Device"):
    """The shape of a jacobian for a single tape given a device.

    Args:
        tape (QuantumTape): the tape who's output we want to determine
        device (Device): the device used to execute the tape.

    >>> tape = qml.tape.QuantumScript([qml.RX(1.0, wires=0)], [qml.expval(qml.X(0)), qml.probs(0)])
    >>> dev = qml.devices.DefaultQubit()
    >>> _jac_shape_dtype_struct(tape, dev)
    (ShapeDtypeStruct(shape=(), dtype=float64),
    ShapeDtypeStruct(shape=(2,), dtype=float64))
    >>> tapes, fn = qml.gradients.param_shift(tape)
    >>> fn(dev.execute(tapes))
    (array(0.), array([-0.42073549,  0.42073549]))
    """
    shape_and_dtype = _result_shape_dtype_struct(tape, device)
    if len(tape.trainable_params) == 1:
        return shape_and_dtype
    if len(tape.measurements) == 1:
        return tuple(shape_and_dtype for _ in tape.trainable_params)
    return tuple(tuple(_s for _ in tape.trainable_params) for _s in shape_and_dtype)


def _pytree_shape_dtype_struct(pytree):
    """Creates a shape structure that matches the types and shapes for the provided pytree."""
    leaves, struct = jax.tree_util.tree_flatten(pytree)
    new_leaves = [jax.ShapeDtypeStruct(jnp.shape(l), l.dtype) for l in leaves]
    return jax.tree_util.tree_unflatten(struct, new_leaves)


def _execute_wrapper_inner(params, tapes, execute_fn, _, device, is_vjp=False) -> ResultBatch:
    """
    Execute tapes using a pure-callback.
    """
    shape_dtype_structs = tuple(_result_shape_dtype_struct(t, device) for t in tapes.vals)
    _set_fn = _set_trainable_parameters_on_copy if is_vjp else _set_all_parameters_on_copy

    def pure_callback_wrapper(p):
        new_tapes = _set_fn(tapes.vals, p)
        return _to_jax(execute_fn(new_tapes))

    # first order way of determining native parameter broadcasting support
    # will be inaccurate when inclusion of broadcast_expand depends on ExecutionConfig values (like adjoint)
    device_supports_vectorization = (
        qml.transforms.broadcast_expand not in device.preprocess_transforms()
    )

    vmap_method = "legacy_vectorized" if device_supports_vectorization else "sequential"

    out = jax.pure_callback(
        pure_callback_wrapper, shape_dtype_structs, params, vmap_method=vmap_method
    )
    return out


_execute_wrapper = partial(_execute_wrapper_inner, is_vjp=False)
_execute_wrapper_vjp = partial(_execute_wrapper_inner, is_vjp=True)


def _execute_and_compute_jvp(tapes, execute_fn, jpc, device, primals, tangents):
    """
    Compute the results and jvps using a pure callback around
    :meth:`~.JacobianProductCalculator.execute_and_compute_jacobian`.

    Note that we must query the full jacobian inside the pure-callback so that jax can trace the JVP
    calculation.
    """
    # Select the trainable params. Non-trainable params contribute a 0 gradient.
    for tangent, tape in zip(tangents[0], tapes.vals, strict=True):
        tape.trainable_params = tuple(
            idx for idx, t in enumerate(tangent) if not isinstance(t, Zero)
        )
    tangents_trainable = tuple(
        tuple(t for t in tangent if not isinstance(t, Zero)) for tangent in tangents[0]
    )

    def wrapper(inner_params):
        new_tapes = _set_all_parameters_on_copy(tapes.vals, inner_params)
        return _to_jax(jpc.execute_and_compute_jacobian(new_tapes))

    res_struct = tuple(_result_shape_dtype_struct(t, device) for t in tapes.vals)
    jac_struct = tuple(_jac_shape_dtype_struct(t, device) for t in tapes.vals)
    results, jacobians = jax.pure_callback(
        wrapper, (res_struct, jac_struct), primals[0], vmap_method="sequential"
    )

    jvps = _compute_jvps(jacobians, tangents_trainable, tapes.vals)

    return results, jvps


def _vjp_fwd(params, tapes, execute_fn, jpc, device):
    """Perform the forward pass execution, return results and the parameters as residuals."""
    return _execute_wrapper_vjp(params, tapes, execute_fn, jpc, device), params


def _vjp_bwd(tapes, execute_fn, jpc, device, params, dy):
    """Perform the backward pass of a vjp calculation, returning the vjp."""

    def wrapper(inner_params, inner_dy):
        new_tapes = _set_trainable_parameters_on_copy(tapes.vals, inner_params)
        return _to_jax(jpc.compute_vjp(new_tapes, inner_dy))

    vjp_shape = _pytree_shape_dtype_struct(params)
    return (jax.pure_callback(wrapper, vjp_shape, params, dy, vmap_method="legacy_vectorized"),)


_execute_jvp_jit = jax.custom_jvp(_execute_wrapper, nondiff_argnums=[1, 2, 3, 4])
_execute_jvp_jit.defjvp(_execute_and_compute_jvp, symbolic_zeros=True)

_execute_vjp_jit = jax.custom_vjp(_execute_wrapper_vjp, nondiff_argnums=[1, 2, 3, 4])
_execute_vjp_jit.defvjp(_vjp_fwd, _vjp_bwd)


def jax_jit_jvp_execute(tapes, execute_fn, jpc, device):
    """Execute a batch of tapes with JAX parameters using JVP derivatives.

    Args:
        tapes (Sequence[.QuantumTape]): batch of tapes to execute
        execute_fn (Callable[[Sequence[.QuantumTape]], ResultBatch]): a function that turns a batch of circuits into results
        jpc (JacobianProductCalculator): a class that can compute the Jacobian for the input tapes.
        device (pennylane.devices.Device): The device used for execution. Used to determine the shapes of outputs for
            pure callback calls.

    Returns:
        TensorLike: A nested tuple of tape results. Each element in
        the returned tuple corresponds in order to the provided tapes.

    """

    if any(
        isinstance(m, qml.measurements.CountsMP) and not (m.all_outcomes)
        for t in tapes
        for m in t.measurements
    ):
        # Obtaining information about the shape of the Counts measurements is
        # not implemented and is required for the callback logic
        raise NotImplementedError("The JAX-JIT interface doesn't support qml.counts.")

    parameters = tuple(tuple(t.get_parameters(trainable_only=False)) for t in tapes)

    return _execute_jvp_jit(parameters, _NonPytreeWrapper(tuple(tapes)), execute_fn, jpc, device)


def jax_jit_vjp_execute(tapes, execute_fn, jpc, device=None):
    """Execute a batch of tapes with JAX parameters using VJP derivatives.

    Args:
        tapes (Sequence[.QuantumTape]): batch of tapes to execute
        execute_fn (Callable[[Sequence[.QuantumTape]], ResultBatch]): a function that turns a batch of circuits into results
        jpc (JacobianProductCalculator): a class that can compute the vector Jacobian product (VJP)
            for the input tapes.
        device (pennylane.devices.Device): The device used for execution. Used to determine the shapes of outputs for
            pure callback calls.

    Returns:
        TensorLike: A nested tuple of tape results. Each element in
        the returned tuple corresponds in order to the provided tapes.

    """
    if any(
        isinstance(m, qml.measurements.CountsMP) and not (m.all_outcomes)
        for t in tapes
        for m in t.measurements
    ):
        # Obtaining information about the shape of the Counts measurements is
        # not implemented and is required for the callback logic
        raise NotImplementedError("The JAX-JIT interface doesn't support qml.counts.")

    parameters = tuple(tuple(t.get_parameters()) for t in tapes)

    return _execute_vjp_jit(parameters, _NonPytreeWrapper(tuple(tapes)), execute_fn, jpc, device)
