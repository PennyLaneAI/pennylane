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
import logging

import jax
import jax.numpy as jnp

from pennylane.math import is_abstract
from pennylane.typing import ResultBatch

dtype = jnp.float64

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


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
                    if is_abstract(param):
                        return "jax-jit"

    return "jax"


def _to_jax(result: ResultBatch) -> ResultBatch:
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


def jax_execute(tapes, execute_fn, jpc) -> ResultBatch:
    """Wrapper that execute tapes and converts the results back to jax.

    Args:
        tapes (Tuple[.QuantumTape]): batch of tapes to execute
        execute_fn (Callable[[Tuple[.QuantumTape]], ResultBatch]): a function that turns a batch of circuits into results
        jpc (JacobianProductCalculator): a class that can compute the vector Jacobian product (VJP)
            for the input tapes.

    Returns:
        ResultBatch: a nested structure of tuples, dicts, and jax arrays.
    """
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("Entry with (tapes=%s, execute_fn=%s, jpc=%s)", tapes, execute_fn, jpc)
    return _to_jax(execute_fn(tapes))


def jax_execute_and_compute_jvp(_, jpc, primals, tangents):
    """Compute the results and jacobian vector products for a batch of tapes.

    Args:
        execute_fn (Callable[[Tuple[.QuantumTape]], ResultBatch]): a function that turns a batch of circuits into results
        jpc (JacobianProductCalculator): a class that can compute the vector Jacobian product (VJP)
            for the input tapes.
        primals: the differentiable arguments to ``jax_jvp_execute``. ``primals[0]`` is equivalent to the ``tapes``
        tangents: the tangents for the differentiable arguments to ``jax_jvp_execute``. ``tangents[0]`` is a batch of tapes
            whose parameters are the tangents for the corresponding data in ``primals[0]``

    Returns:
        ResultBatch, TensorLike: The results of executing the tapes and the jacobian vector products.

    """
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("Entry with (tapes=%s, jpc=%s)", primals[0], jpc)
    tangents = tuple(t.get_parameters() for t in tangents[0])
    return _to_jax(jpc.execute_and_compute_jvp(primals[0], tangents))


def vjp_fwd(tapes, execute_fn, jpc):
    """Perform the forward pass in a vjp calculation.

    Args:
        tapes (Tuple[.QuantumTape]): batch of tapes to execute
        execute_fn (Callable[[Tuple[.QuantumTape]], ResultBatch]): a function that turns a batch of circuits into results
        jpc (JacobianProductCalculator): a class that can compute the vector Jacobian product (VJP)
            for the input tapes.

    Returns:
        ResultBatch, tapes : The result of executing the tapes and the tapes for use on the backward pass

    """
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("Entry with (tapes=%s, execute_fn=%s, jpc=%s)", tapes, execute_fn, jpc)
    return _to_jax(execute_fn(tapes)), tapes


def vjp_bwd(_, jpc, tapes, dy):
    """Perform the backward pass of a vjp calculation.

        Args:
        execute_fn (Callable[[Tuple[.QuantumTape]], ResultBatch]): a function that turns a batch of circuits into results
        jpc (JacobianProductCalculator): a class that can compute the vector Jacobian product (VJP)
            for the input tapes.
        tapes (Tuple[.QuantumTape]): the residuals returned as the second output of ``vjp_fwd``
        dy (ResultBatch): The derivatives of the output

    Returns:
        TensorLike: the vector jacobian product.

    """
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("Entry with (tapes=%s, jpc=%s)", tapes, jpc)
    return _to_jax(jpc.compute_vjp(tapes, dy))


jax_jvp_execute = jax.custom_jvp(jax_execute, nondiff_argnums=[1, 2])
jax_vjp_execute = jax.custom_vjp(jax_execute, nondiff_argnums=[1, 2])


jax_jvp_execute.defjvp(jax_execute_and_compute_jvp)
jax_vjp_execute.defvjp(vjp_fwd, vjp_bwd)
