# Copyright 2026 Xanadu Quantum Technologies Inc.

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
Pure function implementations for the Phox simulator.
"""
import math
from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike


@dataclass
class CircuitConfig:  # unused for now
    """Configuration data for an IQP circuit simulation."""
    n_samples: int
    n_qubits: int
    generators: list[list[int]]
    observables: list[list[str]]
    key: int
    init_state: ArrayLike = None
    phase_layer: Callable = None


def bitflip_expval(
    generators: ArrayLike, params: ArrayLike, ops: ArrayLike
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute expectation value for the Bitflip noise model.

    Args:
        generators (ArrayLike): Binary matrix of shape ``(n_generators, n_qubits)``.
        params (ArrayLike): Error probabilities/parameters $\theta$.
        ops (ArrayLike): Binary matrix representing Pauli Z operators.

    Returns:
        tuple[jnp.ndarray, jnp.ndarray]: A tuple containing:
            - Expectation values.
            - A zero array for standard error (since this is analytical).
    """
    probs = jnp.cos(2 * params)

    # Indicator = (A . G^T) % 2
    indicator = (ops @ generators.T) % 2
    X = probs * indicator

    # Use jnp.where to replace 0s with 1.0s before product to avoid zeroing out
    result = jnp.prod(jnp.where(X == 0, 1.0, X), axis=1)

    return result, jnp.zeros(ops.shape[0])


def _parse_iqp_dict(circuit_def: dict[int, list[list[int]]], n_qubits: int):
    """
    Converts dictionary circuit definition into JAX-ready matrices.

    Args:
        circuit_def (dict[int, list[list[int]]]): Dictionary mapping parameter indices to lists of qubit indices.
        n_qubits (int): Total number of qubits.

    Returns:
        tuple[jnp.ndarray, jnp.ndarray]: Tuple containing:
            - Binary matrix of generators.
            - Integer array mapping parameters to generators.
    """
    flat_gates = []
    param_indices = []

    for param_idx in sorted(circuit_def.keys()):
        gates_for_this_param = circuit_def[param_idx]
        for gate in gates_for_this_param:
            flat_gates.append(gate)
            param_indices.append(param_idx)

    n_gates = len(flat_gates)
    generators = np.zeros((n_gates, n_qubits), dtype=int)

    for i, qubits in enumerate(flat_gates):
        generators[i, qubits] = 1
    param_map = jnp.array(param_indices, dtype=int)
    return jnp.array(generators), param_map


def _obs_to_numeric(obs: list[list[str]] | list[str]) -> jnp.ndarray:
    """
    Helper to convert observable strings to numeric encoding:
    I -> 0, X -> 1, Y -> 2, Z -> 3.
    """
    if isinstance(obs, list):
        if not obs:
            return jnp.array([], dtype=jnp.int32)
        if isinstance(obs[0], str): # Handle Single observable case
             obs = [obs]
    
    mapping = {"I": 0, "X": 1, "Y": 2, "Z": 3}
    
    numeric_batch = []
    for row in obs:
        if isinstance(row, str) and len(row) > 1:
            numeric_row = [mapping[gate] for gate in row]
        else:
            numeric_row = [mapping[gate] for gate in row]
        numeric_batch.append(numeric_row)
        
    return jnp.array(numeric_batch, dtype=jnp.int32)


def iqp_expval(
    gates: dict[int, list[list[int]]],
    n_qubits: int,
    init_state: tuple[ArrayLike, ArrayLike] | None = None,
):
    """
    Factory that returns a function for computing expectation values.

    Args:
        gates (dict[int, list[list[int]]]): Circuit structure.
        n_qubits (int): Number of qubits.
        init_state (tuple): Initial state.

    Returns:
        Callable: A function ``execute(params, ops, n_samples, key)``.
    """
    generators, param_map = _parse_iqp_dict(gates, n_qubits)

    if init_state is not None:
        X, P = init_state
        P = P[:, jnp.newaxis]
        use_init_state = True
    else:
        X = jnp.empty((0, 0))
        P = jnp.empty((0, 0))
        use_init_state = False

    def expval_execution(params, ops_numeric, n_samples, key):
        samples = jax.random.randint(key, (n_samples, n_qubits), 0, 2)
        
        is_z_or_y = (ops_numeric == 3) | (ops_numeric == 2)
        bitflips = is_z_or_y.astype(jnp.int32)
        
        count_y = jnp.sum(ops_numeric == 2, axis=1)
        
        is_x_or_y = (ops_numeric == 1) | (ops_numeric == 2)
        dot_xy_samples = is_x_or_y.astype(jnp.int32) @ samples.T
        
        phase_y_base = (-1j) ** count_y
        phase_bit_dep = (-1) ** dot_xy_samples
        
        phases = phase_y_base[:, None] * phase_bit_dep
        
        B = (-1) ** (samples @ generators.T)
        C = 1 - ((-1) ** (bitflips @ generators.T))

        expanded_params = params if param_map is None else jnp.asarray(params)[param_map]

        E = C @ (expanded_params[:, None] * B.T)

        M = phases * jnp.exp(1j * E)

        if use_init_state:
            F = jnp.broadcast_to(P, (P.shape[0], samples.shape[0])) * ((-1) ** (X @ samples.T))

            H1 = ((-1) ** (bitflips @ X.T)) @ F
            col_sums = jnp.sum(F.conj(), axis=0, keepdims=True)
            H2 = jnp.broadcast_to(col_sums, (bitflips.shape[0], samples.shape[0]))
            H = H1 * H2
            M = M * H

        expvals = jnp.real(M)
        std_err = jnp.std(expvals, axis=-1, ddof=1) / jnp.sqrt(samples.shape[0])
        return jnp.mean(expvals, axis=1), std_err

    # Compile the inner kernel
    jitted_kernel = jax.jit(expval_execution, static_argnames=["n_samples"])

    # Wrapper to handle string -> numeric conversion
    def execute(params, ops, n_samples, key):
        return jitted_kernel(params, _obs_to_numeric(ops), n_samples, key)

    return execute
