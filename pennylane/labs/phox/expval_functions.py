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

from pennylane.labs.tests.conftest import init_state


@dataclass
class CircuitConfig:
    """
    Configuration data for an IQP circuit simulation.

    Args:
        gates (dict[int, list[list[int]]]): Circuit structure mapping parameters to gates.
        observables (ArrayLike): List of Pauli observables mapped to integers (I=0, X=1, Y=2, Z=3).
        n_samples (int): Number of stochastic samples.
        key (ArrayLike): Random key for JAX.
        n_qubits (int): Number of qubits.
        init_state (tuple[ArrayLike, ArrayLike] | None): Initial state configuration (X, P).
        phase_layer (Callable | None): Optional phase layer function.
    """

    gates: dict[int, list[list[int]]]
    observables: ArrayLike
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


def _compute_samples(key: ArrayLike, n_samples: int, n_qubits: int) -> jnp.ndarray:
    """Generates the stochastic sample matrix."""
    return jax.random.randint(key, (n_samples, n_qubits), 0, 2)


def _prep_observables(observables_int: ArrayLike) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Converts integer observables (I=0, X=1, Y=2, Z=3)
    into precomputed bitmasks and y_phases.
    """
    obs_arr = jnp.asarray(observables_int, dtype=jnp.int32)

    is_X = obs_arr == 1
    is_Y = obs_arr == 2
    is_Z = obs_arr == 3

    bitflips = jnp.array(is_Z | is_Y, dtype=jnp.int32)
    mask_XY = jnp.array(is_X | is_Y, dtype=jnp.int32)
    count_Y = jnp.array(is_Y.sum(axis=1), dtype=jnp.int32)

    y_phase = (-1j) ** count_Y[:, jnp.newaxis]

    return bitflips, mask_XY, y_phase


def _core_expval_execution(
    params: ArrayLike,
    phase_params: ArrayLike | None,
    samples: jnp.ndarray,
    obs_data: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    init_state: tuple[ArrayLike, ArrayLike] | None,
    generators: jnp.ndarray,
    param_map: jnp.ndarray,
    vmapped_phase_func: Callable | None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """The pure mathematical core of the expectation value computation."""
    bitflips, mask_XY, y_phase = obs_data

    sign_flip = 1 - 2 * ((mask_XY @ samples.T) % 2)
    phases = sign_flip * y_phase

    B = 1 - 2 * ((samples @ generators.T) % 2)
    C = 2 * ((bitflips @ generators.T) % 2)

    expanded_params = jnp.asarray(params)[param_map]
    E = (C * expanded_params) @ B.T
    if vmapped_phase_func is not None:
        E += vmapped_phase_func(phase_params, samples, bitflips)

    if init_state is None:
        expvals = jnp.real(phases) * jnp.cos(E) - jnp.imag(phases) * jnp.sin(E)
    else:
        M = phases * jnp.exp(1j * E)
        X, P = init_state
        F = P[:, jnp.newaxis] * (1 - 2 * ((X @ samples.T) % 2))
        H1 = (1 - 2 * ((bitflips @ X.T) % 2)) @ F
        col_sums = jnp.sum(F.conj(), axis=0, keepdims=True)
        H = H1 * col_sums
        M = M * H
        expvals = jnp.real(M)

    std_err = jnp.std(expvals, axis=-1, ddof=1) / jnp.sqrt(samples.shape[0])

    return jnp.mean(expvals, axis=1), std_err


def build_expval_func(
    config: CircuitConfig,
) -> Callable:
    """
    Factory that returns a flexible pure function for computing expectation values.
    The returned closure can optionally take runtime overrides for key, observables, etc.
    """
    generators, param_map = _parse_generator_dict(config.gates, config.n_qubits)

    vmapped_phase_func = None
    if config.phase_layer is not None:

        def compute_phase(p_params, sample, b_flips):
            return config.phase_layer(p_params, sample) - config.phase_layer(
                p_params, (sample + b_flips) % 2
            )

        vmapped_phase_func = jax.vmap(
            jax.vmap(compute_phase, in_axes=(None, 0, None)), in_axes=(None, None, 0)
        )

    default_samples = _compute_samples(config.key, config.n_samples, config.n_qubits)
    default_obs_data = _prep_observables(config.observables)

    def expval_execution(
        params: ArrayLike,
        phase_params: ArrayLike | None = None,
        observables: ArrayLike | None = None,
        key: ArrayLike | None = None,
        n_samples: int | None = None,
        init_state: tuple[ArrayLike, ArrayLike] | None = None,
    ):
        if key is not None or n_samples is not None:
            _key = key if key is not None else config.key
            _n = n_samples if n_samples is not None else config.n_samples
            samples = _compute_samples(_key, _n, config.n_qubits)
        else:
            samples = default_samples

        obs_data = default_obs_data if observables is None else _prep_observables(observables)
        state = config.init_state if init_state is None else init_state

        return _core_expval_execution(
            params,
            phase_params,
            samples,
            obs_data,
            state,
            generators,
            param_map,
            vmapped_phase_func,
        )

    return expval_execution
