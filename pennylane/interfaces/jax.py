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
# pylint: disable=unused-argument
import logging
from typing import Tuple, Callable

import dataclasses

import jax
import jax.numpy as jnp

import pennylane as qml
from pennylane.transforms import convert_to_numpy_parameters
from pennylane.typing import ResultBatch

dtype = jnp.float64

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


Batch = Tuple[qml.tape.QuantumTape]
ExecuteFn = Callable[[Batch], qml.typing.ResultBatch]


@dataclasses.dataclass
class NonPytreeWrapper:
    """We aren't quite ready to switch to having tapes as pytrees as our
    differentiable argument due to:

    * Operators that aren't valid pytrees: ex. ParametrizedEvolution, ParametrizedHamiltonian, HardwareHamiltonian
    * Validation checks on initialization: see BasisStateProjector, StatePrep that does not allow the operator to store the cotangents
    * Jitting non-jax parametrized circuits.  Numpy parameters turn into abstract parameters during the pytree process.

    ``jax.custom_vjp`` forbids any non-differentiable argument to be a pytree, so we need to wrap it in a non-pytree type.

    When the above issues are fixed, we can treat ``tapes`` as the differentiable argument.

    """

    vals: Batch = None


def _set_copy_and_unwrap_tape(t, a, unwrap=True):
    """Copy a given tape with operations and set parameters"""
    tc = t.bind_new_parameters(a, t.trainable_params)
    return convert_to_numpy_parameters(tc) if unwrap else tc


def set_parameters_on_copy_and_unwrap(tapes, params, unwrap=True):
    """Copy a set of tapes with operations and set parameters"""
    return tuple(_set_copy_and_unwrap_tape(t, a, unwrap=unwrap) for t, a in zip(tapes, params))


def get_jax_interface_name(tapes):
    """Check all parameters in each tape and output the name of the suitable
    JAX interface.

    This function checks each tape and determines if any of the gate parameters
    was transformed by a JAX transform such as ``jax.jit``. If so, it outputs
    the name of the JAX interface with jit support.

    Note that determining if jit support should be turned on is done by
    checking if parameters are abstract. Parameters can be abstract not just
    for ``jax.jit``, but for other JAX transforms (vmap, pmap, etc.) too. The
    reason is that JAX doesn't have a public API for checking whether or not
    the execution is within the jit transform.

    Args:
        tapes (Sequence[.QuantumTape]): batch of tapes to execute

    Returns:
        str: name of JAX interface that fits the tape parameters, "jax" or
        "jax-jit"
    """
    for t in tapes:
        for op in t:
            # Unwrap the observable from a MeasurementProcess
            op = op.obs if hasattr(op, "obs") else op
            if op is not None:
                # Some MeasurementProcess objects have op.obs=None
                for param in op.data:
                    if qml.math.is_abstract(param):
                        return "jax-jit"

    return "jax"


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


def execute_wrapper(params, tapes, execute_fn, jpc) -> ResultBatch:
    """Executes ``tapes`` with ``params`` via ``execute_fn``"""
    new_tapes = set_parameters_on_copy_and_unwrap(tapes.vals, params, unwrap=False)
    return _to_jax(execute_fn(new_tapes))


def jax_execute_and_compute_jvp(tapes, execute_fn, jpc, primals, tangents):
    """Compute the results and jvps for ``tapes`` with ``primals[0]`` parameters via
    ``jpc``.
    """
    new_tapes = set_parameters_on_copy_and_unwrap(tapes.vals, primals[0], unwrap=False)
    res, jvps = jpc.execute_and_compute_jvp(new_tapes, tangents[0])
    return _to_jax(res), _to_jax(jvps)


def vjp_fwd(params, tapes, execute_fn, jpc):
    """Perform the forward pass execution, return results and empty residuals."""
    new_tapes = set_parameters_on_copy_and_unwrap(tapes.vals, params, unwrap=False)
    return _to_jax(execute_fn(new_tapes)), None


def vjp_bwd(tapes, execute_fn, jpc, _, dy):
    """Perform the backward pass of a vjp calculation, returning the vjp."""
    vjp = jpc.compute_vjp(tapes.vals, dy)
    return (_to_jax(vjp),)


_execute_jvp = jax.custom_jvp(execute_wrapper, nondiff_argnums=[1, 2, 3])
_execute_jvp.defjvp(jax_execute_and_compute_jvp)

_execute_vjp = jax.custom_vjp(execute_wrapper, nondiff_argnums=[1, 2, 3])
_execute_vjp.defvjp(vjp_fwd, vjp_bwd)


def jax_jvp_execute(tapes: Batch, execute_fn: ExecuteFn, jpc):
    """Execute a batch of tapes with Jax parameters using JVP derivatives.

    Args:
        tapes (Sequence[.QuantumTape]): batch of tapes to execute
        execute_fn (Callable[[Sequence[.QuantumTape]], ResultBatch]): a function that turns a batch of circuits into results
        jpc (JacobianProductCalculator): a class that can compute the vector Jacobian product (VJP)
            for the input tapes.

    Returns:
        TensorLike: A nested tuple of tape results. Each element in
        the returned tuple corresponds in order to the provided tapes.

    """
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("dsfjlfs")

    for tape in tapes:
        params = tape.get_parameters(trainable_only=False)
        tape.trainable_params = qml.math.get_trainable_indices(params)

    parameters = tuple(tuple(t.get_parameters()) for t in tapes)

    return _execute_jvp(parameters, NonPytreeWrapper(tuple(tapes)), execute_fn, jpc)


def jax_vjp_execute(tapes: Batch, execute_fn: ExecuteFn, jpc):
    """Execute a batch of tapes with Jax parameters using VJP derivatives.

    Args:
        tapes (Sequence[.QuantumTape]): batch of tapes to execute
        execute_fn (Callable[[Sequence[.QuantumTape]], ResultBatch]): a function that turns a batch of circuits into results
        jpc (JacobianProductCalculator): a class that can compute the vector Jacobian product (VJP)
            for the input tapes.

    Returns:
        TensorLike: A nested tuple of tape results. Each element in
        the returned tuple corresponds in order to the provided tapes.

    """
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("dsfjlfs")

    for tape in tapes:
        params = tape.get_parameters(trainable_only=False)
        tape.trainable_params = qml.math.get_trainable_indices(params)

    parameters = tuple(tuple(t.get_parameters()) for t in tapes)

    return _execute_vjp(parameters, NonPytreeWrapper(tuple(tapes)), execute_fn, jpc)
