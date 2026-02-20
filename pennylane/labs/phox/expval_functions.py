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
Pure function implementations for the expectation value functions.
"""
import math
from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike


@dataclass
class CircuitConfig:
    """
    Configuration data for an IQP circuit simulation.

    Args:
        gates (dict[int, list[list[int]]]): Circuit structure mapping parameters to gates.
        observables (list[list[str]]): List of Pauli observables.
        n_samples (int): Number of stochastic samples.
        key (ArrayLike): Random key for JAX.
        n_qubits (int): Number of qubits.
        init_state (tuple[ArrayLike, ArrayLike] | None): Initial state configuration (X, P).
        phase_layer (Callable | None): Optional phase layer function.
    """

    gates: dict[int, list[list[int]]]
    observables: list[list[str]]
    n_samples: int
    key: ArrayLike
    n_qubits: int
    init_state: tuple[ArrayLike, ArrayLike] | None = None
    phase_layer: Callable | None = None


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

    indicator = (ops @ generators.T) % 2
    X = probs * indicator

    # Use jnp.where to replace 0s with 1.0s before product to avoid zeroing out
    result = jnp.prod(jnp.where(X == 0, 1.0, X), axis=1)

    return result, jnp.zeros(ops.shape[0])


def _parse_generator_dict(circuit_def: dict[int, list[list[int]]], n_qubits: int):
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


def build_expval_func(
    config: CircuitConfig,
) -> Callable[[ArrayLike, ArrayLike | None], tuple[jnp.ndarray, jnp.ndarray]]:
    """
    Factory that returns a function for computing expectation values.
    """
    generators, param_map = _parse_generator_dict(config.gates, config.n_qubits)

    samples = jax.random.randint(config.key, (config.n_samples, config.n_qubits), 0, 2)
    obs_arr = np.array(config.observables, dtype=str)

    is_Y = obs_arr == "Y"
    is_Z = obs_arr == "Z"
    is_X = obs_arr == "X"

    bitflips = jnp.array(is_Z | is_Y, dtype=jnp.int32)
    mask_XY = jnp.array(is_X | is_Y, dtype=jnp.int32)
    count_Y = jnp.array(is_Y.sum(axis=1), dtype=jnp.int32)
    
    sign_flip = 1 - 2 * ((mask_XY @ samples.T) % 2)
    y_phase = (-1j) ** count_Y[:, jnp.newaxis]
    phases = sign_flip * y_phase

    vmapped_phase_func = None
    if config.phase_layer is not None:
        def compute_phase(p_params, sample, b_flips):
            return config.phase_layer(p_params, sample) - config.phase_layer(
                p_params, (sample + b_flips) % 2
            )
        
        vmapped_phase_func = jax.vmap(
            jax.vmap(compute_phase, in_axes=(None, 0, None)), 
            in_axes=(None, None, 0)
        )

    def expval_execution(params, phase_params=None):
        B = 1 - 2 * ((samples @ generators.T) % 2)
        
        C = 2 * ((bitflips @ generators.T) % 2)

        expanded_params = jnp.asarray(params)[param_map]
        E = C @ (expanded_params[:, None] * B.T)

        if vmapped_phase_func is not None:
            E += vmapped_phase_func(phase_params, samples, bitflips)

        M = phases * jnp.exp(1j * E)

        if config.init_state is not None:
            X, P = config.init_state
            
            F = P[:, jnp.newaxis] * (1 - 2 * ((X @ samples.T) % 2))
            
            H1 = (1 - 2 * ((bitflips @ X.T) % 2)) @ F
            col_sums = jnp.sum(F.conj(), axis=0, keepdims=True)
            H = H1 * col_sums
            M = M * H

        expvals = jnp.real(M)
        std_err = jnp.std(expvals, axis=-1, ddof=1) / jnp.sqrt(samples.shape[0])
        
        return jnp.mean(expvals, axis=1), std_err

    return expval_execution
