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
from pennylane.transforms import convert_to_numpy_parameters
from pennylane.typing import ResultBatch

from .jacobian_products import _compute_jvps

from .jax import _NonPytreeWrapper

dtype = jnp.float64
Zero = jax.custom_derivatives.SymbolicZero


def _to_jax(result: qml.typing.ResultBatch) -> qml.typing.ResultBatch:
    """Converts an arbitrary result batch to one with jax arrays.
    Args:
        result (ResultBatch): a nested structure of lists, tuples, dicts, and numpy arrays
    Returns:
        ResultBatch: a nested structure of tuples, dicts, and jax arrays
    """
    if isinstance(result, dict):
        return result
    if isinstance(result, (list, tuple)):
        return tuple(_to_jax(r) for r in result)
    return jnp.array(result)


def _set_copy_and_unwrap_tape(t, a, unwrap=True):
    """Copy a given tape with operations and set parameters"""
    tc = t.bind_new_parameters(a, list(range(len(a))))
    return convert_to_numpy_parameters(tc) if unwrap else tc


def set_parameters_on_copy_and_unwrap(tapes, params, unwrap=True):
    """Copy a set of tapes with operations and set parameters"""
    return tuple(_set_copy_and_unwrap_tape(t, a, unwrap=unwrap) for t, a in zip(tapes, params))


def _numeric_type_to_dtype(numeric_type):
    """Auxiliary function for converting from Python numeric types to JAX
    dtypes based on the precision defined for the interface."""

    if numeric_type is int:
        return jnp.int64

    if numeric_type is float:
        return jnp.float64

    # numeric_type is complex
    return jnp.complex128


def _create_shape_dtype_struct(tape: "qml.tape.QuantumScript", device: "qml.Device"):
    """Auxiliary function for creating the shape and dtype object structure
    given a tape."""

    shape = tape.shape(device)
    if len(tape.measurements) == 1:
        tape_dtype = _numeric_type_to_dtype(tape.numeric_type)
        if tape.shots.has_partitioned_shots:
            return tuple(jax.ShapeDtypeStruct(s, tape_dtype) for s in shape)
        return jax.ShapeDtypeStruct(tuple(shape), tape_dtype)

    tape_dtype = tuple(_numeric_type_to_dtype(elem) for elem in tape.numeric_type)
    if tape.shots.has_partitioned_shots:
        return tuple(
            tuple(jax.ShapeDtypeStruct(tuple(s), d) for s, d in zip(si, tape_dtype)) for si in shape
        )
    return tuple(jax.ShapeDtypeStruct(tuple(s), d) for s, d in zip(shape, tape_dtype))


def _jac_shape_dtype_struct(tape: "qml.tape.QuantumScript", device: "qml.Device"):
    """The shape of a jacobian for a single tape given a device.

    Args:
        tape (QuantumTape): the tape who's output we want to determine
        device (Device): the device used to execute the tape.

    >>> tape = qml.tape.QuantumScript([qml.RX(1.0, wires=0)], [qml.expval(qml.PauliX(0)), qml.probs(0)])
    >>> dev = qml.devices.DefaultQubit()
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


def _execute_wrapper(params, tapes, execute_fn, _, device) -> ResultBatch:
    shape_dtype_structs = tuple(_create_shape_dtype_struct(t, device) for t in tapes.vals)

    def pure_callback_wrapper(p):
        new_tapes = set_parameters_on_copy_and_unwrap(tapes.vals, p, unwrap=False)
        res = execute_fn(new_tapes)
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
        return jax.tree_map(lambda r, s: r.T if r.ndim > s.ndim else r, res, shape_dtype_structs)

    out = jax.pure_callback(pure_callback_wrapper, shape_dtype_structs, params, vectorized=True)
    return _to_jax(out)


def _execute_and_compute_jvp(tapes, execute_fn, jpc, device, primals, tangents):

    # Select the trainable params. Non-trainable params contribute a 0 gradient.
    for tangent, tape in zip(tangents[0], tapes.vals):
        tape.trainable_params = tuple(
            idx for idx, t in enumerate(tangent) if not isinstance(t, Zero)
        )
    tangents_trainable = tuple(
        tuple(t for t in tangent if not isinstance(t, Zero)) for tangent in tangents[0]
    )

    def wrapper(inner_params):
        new_tapes = set_parameters_on_copy_and_unwrap(tapes.vals, inner_params, unwrap=False)
        return _to_jax(jpc.execute_and_compute_jacobian(new_tapes))

    res_struct = tuple(_create_shape_dtype_struct(t, device) for t in tapes.vals)
    jac_struct = tuple(_jac_shape_dtype_struct(t, device) for t in tapes.vals)
    results, jacobians = jax.pure_callback(wrapper, (res_struct, jac_struct), primals[0])

    jvps = _compute_jvps(jacobians, tangents_trainable, tapes.vals)

    return _to_jax(results), _to_jax(jvps)


def _vjp_fwd(params, tapes, execute_fn, jpc):
    """Perform the forward pass execution, return results and empty residuals."""
    return _execute_wrapper(params, tapes, execute_fn, jpc), params


def _vjp_bwd(tapes, execute_fn, jpc, params, dy):
    """Perform the backward pass of a vjp calculation, returning the vjp."""

    def wrapper(inner_params):
        new_tapes = set_parameters_on_copy_and_unwrap(tapes.vals, inner_params, unwrap=False)
        return _to_jax(jpc.compute_vjp(new_tapes, dy))

    vjp_shape = None  # TODO
    vjp = jax.pure_callback(wrapper, vjp_shape, params)
    return (_to_jax(vjp),)


_execute_jvp_jit = jax.custom_jvp(_execute_wrapper, nondiff_argnums=[1, 2, 3, 4])
_execute_jvp_jit.defjvp(_execute_and_compute_jvp, symbolic_zeros=True)

_execute_vjp_jit = jax.custom_vjp(_execute_wrapper, nondiff_argnums=[1, 2, 3, 4])
_execute_vjp_jit.defvjp(_vjp_fwd, _vjp_bwd)


def jax_jvp_jit_execute(tapes, execute_fn, jpc, device):

    if any(
        m.return_type in (qml.measurements.Counts, qml.measurements.AllCounts)
        for t in tapes
        for m in t.measurements
    ):
        # Obtaining information about the shape of the Counts measurements is
        # not implemented and is required for the callback logic
        raise NotImplementedError("The JAX-JIT interface doesn't support qml.counts.")

    parameters = tuple(tuple(t.get_parameters(trainable_only=False)) for t in tapes)

    return _execute_jvp_jit(parameters, _NonPytreeWrapper(tuple(tapes)), execute_fn, jpc, device)


def jax_vjp_jit_execute(tapes, execute_fn, jpc, device=None):
    """Execute a batch of tapes with JAX parameters using VJP derivatives.

    Args:
        tapes (Sequence[.QuantumTape]): batch of tapes to execute
        execute_fn (Callable[[Sequence[.QuantumTape]], ResultBatch]): a function that turns a batch of circuits into results
        jpc (JacobianProductCalculator): a class that can compute the vector Jacobian product (VJP)
            for the input tapes.

    Returns:
        TensorLike: A nested tuple of tape results. Each element in
        the returned tuple corresponds in order to the provided tapes.

    """
    if any(
        m.return_type in (qml.measurements.Counts, qml.measurements.AllCounts)
        for t in tapes
        for m in t.measurements
    ):
        # Obtaining information about the shape of the Counts measurements is
        # not implemented and is required for the callback logic
        raise NotImplementedError("The JAX-JIT interface doesn't support qml.counts.")

    parameters = tuple(tuple(t.get_parameters()) for t in tapes)

    return _execute_vjp_jit(parameters, _NonPytreeWrapper(tuple(tapes)), execute_fn, jpc, device)
