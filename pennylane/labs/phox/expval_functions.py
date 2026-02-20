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
    Contains ONLY static, compile-time properties (Type 1 variables).

    Args:
        gates (dict[int, list[list[int]]]): Circuit structure mapping parameters to gates.
        n_qubits (int): Number of qubits.
        phase_layer (Callable | None): Optional phase layer function.
    """

    gates: dict[int, list[list[int]]]
    n_qubits: int
    phase_layer: Callable | None = None


def bitflip_expval(
    generators: ArrayLike, params: ArrayLike, ops: ArrayLike
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute expectation value for the Bitflip noise model.

    Args:
        generators (ArrayLike): Binary matrix of shape (n_generators, n_qubits).
        params (ArrayLike): Error probabilities/parameters.
        ops (ArrayLike): Binary matrix representing Pauli Z operators.

    Returns:
        tuple[jnp.ndarray, jnp.ndarray]: Expectation values and standard error.
    """
    probs = jnp.cos(2 * params)

    indicator = (ops @ generators.T) % 2
    X = probs * indicator

    result = jnp.prod(jnp.where(X == 0, 1.0, X), axis=1)

    return result, jnp.zeros(ops.shape[0])


def _parse_generator_dict(circuit_def: dict[int, list[list[int]]], n_qubits: int):
    """
    Converts dictionary circuit definition into JAX-ready matrices.
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
) -> Callable:
    """
    Factory that returns a pure, un-jitted function for computing expectation values.
    """
    generators, param_map = _parse_generator_dict(config.gates, config.n_qubits)

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

    def expval_execution(
        params: tuple,
        data: tuple,
        n_samples: int
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Executes the expectation value calculation.

        Args:
            params: Tuple of differentiable arrays (circuit_params, phase_params, init_state_amps P).
            data: Tuple of non-differentiable arrays (key, init_state_elems X, observables).
                  Observables should be an integer array (0=I, 1=X, 2=Y, 3=Z).
            n_samples: Static integer defining the shape of stochastic sampling.
        """
        circuit_params, phase_params, P = params
        key, X, observables = data

        is_X = observables == 1
        is_Y = observables == 2
        is_Z = observables == 3

        bitflips = jnp.array(is_Z | is_Y, dtype=jnp.int32)
        mask_XY = jnp.array(is_X | is_Y, dtype=jnp.int32)
        count_Y = jnp.array(is_Y.sum(axis=1), dtype=jnp.int32)

        samples = jax.random.randint(key, (n_samples, config.n_qubits), 0, 2)

        sign_flip = 1 - 2 * ((mask_XY @ samples.T) % 2)
        y_phase = (-1j) ** count_Y[:, jnp.newaxis]
        phases = sign_flip * y_phase

        B = 1 - 2 * ((samples @ generators.T) % 2)
        C = 2 * ((bitflips @ generators.T) % 2)

        expanded_params = jnp.asarray(circuit_params)[param_map]
        E = C @ (expanded_params[:, None] * B.T)

        if vmapped_phase_func is not None:
            E += vmapped_phase_func(phase_params, samples, bitflips)

        M = phases * jnp.exp(1j * E)

        if X is not None and P is not None:
            F = P[:, jnp.newaxis] * (1 - 2 * ((X @ samples.T) % 2))
            
            H1 = (1 - 2 * ((bitflips @ X.T) % 2)) @ F
            col_sums = jnp.sum(F.conj(), axis=0, keepdims=True)
            H = H1 * col_sums
            M = M * H

        expvals = jnp.real(M)
        std_err = jnp.std(expvals, axis=-1, ddof=1) / jnp.sqrt(n_samples)
        
        return jnp.mean(expvals, axis=1), std_err

    return expval_execution
