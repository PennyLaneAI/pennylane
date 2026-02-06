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

def _phase(pauli: str, qubit: int) -> complex:
    """
    For a Pauli P return the phase applied by Conjugation with Hadamard (HPH).
    Specifically, we have the relations:
    HXH = Z
    HYH = -Y
    HZH = X
    Args:
        pauli (str): The Pauli operator ("I", "X", "Y", "Z").
        qubit (int): The qubit index (0 or 1, although only checked for parity).
    Returns:
        complex: The phase factor.
    Raises:
        ValueError: If pauli is not one of "I", "X", "Y", "Z".
    """

    if pauli in ("I", "Z"):
        return 1

    if pauli == "Y":
        if qubit == 0:
            return -1j
        return 1j

    if pauli == "X":
        if qubit == 0:
            return 1
        return -1

    raise ValueError(f"Expected Pauli I, X, Y, or Z, got {pauli}.")


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


def iqp_expval(config: CircuitConfig):
    """
    Factory that returns a function for computing expectation values.

    Args:
        config (CircuitConfig): Configuration object containing circuit details.

    Returns:
        Callable: A function ``execute(params, phase_params=None)``.
    """
    generators, param_map = _parse_iqp_dict(config.gates, config.n_qubits)

    samples = jax.random.randint(config.key, (config.n_samples, config.n_qubits), 0, 2)
    bitflips = jnp.array(
        [[1 if g in ("Z", "Y") else 0 for g in op] for op in config.observables], dtype=jnp.int32
    )

    phases = jnp.array(
        [
            [
                math.prod([_phase(gate, qubit) for gate, qubit in zip(op, sample)])
                for sample in samples
            ]
            for op in config.observables
        ]
    )

    @jax.jit
    def expval_execution(params, phase_params=None):
        B = (-1) ** (samples @ generators.T)
        C = 1 - ((-1) ** (bitflips @ generators.T))

        expanded_params = params if param_map is None else jnp.asarray(params)[param_map]

        E = C @ (expanded_params[:, None] * B.T)

        if config.phase_layer is not None:
            def compute_phase(p_params, sample, b_flips):
                return config.phase_layer(p_params, sample) - config.phase_layer(
                    p_params, (sample + b_flips) % 2
                )

            phase_matrix = jax.vmap(compute_phase, in_axes=(None, 0, None))
            phase_matrix = jax.vmap(phase_matrix, in_axes=(None, None, 0))

            E += phase_matrix(phase_params, samples, bitflips)

        M = phases * jnp.exp(1j * E)

        if config.init_state is not None:
            X, P = config.init_state
            P = P[:, jnp.newaxis]
            F = jnp.broadcast_to(P, (P.shape[0], samples.shape[0])) * ((-1) ** (X @ samples.T))

            H1 = ((-1) ** (bitflips @ X.T)) @ F
            col_sums = jnp.sum(F.conj(), axis=0, keepdims=True)
            H2 = jnp.broadcast_to(col_sums, (bitflips.shape[0], samples.shape[0]))
            H = H1 * H2
            M = M * H

        expvals = jnp.real(M)
        std_err = jnp.std(expvals, axis=-1, ddof=1) / jnp.sqrt(samples.shape[0])
        return jnp.mean(expvals, axis=1), std_err

    return expval_execution
