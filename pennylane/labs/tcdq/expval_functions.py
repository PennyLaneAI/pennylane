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
"""Expectation-value estimator for qubit IQP circuits.

This module estimates Pauli expectation values for IQP circuits without
simulating the full quantum state.
"""

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike


@dataclass
class CircuitConfig:  # pylint: disable=too-many-instance-attributes
    """Description of a qubit IQP circuit for classical expectation-value estimation.

    This dataclass bundles all the information needed to build an expectation-value
    estimator via :func:`build_expval_func`: the gate structure, the observables to
    measure, sampling parameters, and an optional non-standard initial state.

    Args:
        gates (dict[int, list[list[int]]]): Circuit structure mapping each
            trainable parameter index to a list of gates. Each gate is itself a
            list of qubit indices that participate in a Pauli-Z tensor-product
            generator. For example, ``{0: [[0, 1]], 1: [[2]]}`` defines two
            parameters: parameter 0 drives a ZZ gate on qubits 0 and 1, and
            parameter 1 drives a Z gate on qubit 2. Use
            :func:`~pennylane.labs.tcdq.create_local_gates` or
            :func:`~pennylane.labs.tcdq.create_lattice_gates` to generate
            these automatically.
        n_samples (int): Number of random bitstrings drawn for the
            estimation.
        key (ArrayLike): JAX PRNG key for random bitstring generation.
        n_qubits (int): Total number of qubits in the circuit.
        observables (ArrayLike | None): Integer array of shape
            ``(n_observables, n_qubits)`` encoding Pauli operators (I=0, X=1,
            Y=2, Z=3). Each row is one observable. If ``None``, observables must
            be passed at call time to the function returned by
            :func:`build_expval_func`.
        init_state_elems (ArrayLike | None): Binary array of shape ``(N, n_qubits)``
            listing the computational-basis states with non-zero amplitude in a
            custom initial state. Use together with ``init_state_amps``. If
            ``None`` (default), the circuit starts in the uniform superposition
            state :math:`H^{\\otimes n}|0\\rangle`.
        init_state_amps (ArrayLike | None): Complex array of shape ``(N,)`` with
            the amplitudes corresponding to ``init_state_elems``.
        phase_fn (Callable | None): Optional custom phase function
            ``phase_fn(params, bitstring)`` applied as an extra diagonal layer.
            Defaults to ``None``.

    **Example**

    >>> import jax
    >>> from pennylane.labs.tcdq import CircuitConfig, create_local_gates
    >>> gates = create_local_gates(n_qubits=4, max_weight=2)
    >>> config = CircuitConfig(
    ...     gates=gates,
    ...     n_samples=2000,
    ...     key=jax.random.PRNGKey(42),
    ...     n_qubits=4,
    ...     observables=[[3, 3, 0, 0], [0, 0, 3, 3]],  # ZZ on (0,1) and ZZ on (2,3)
    ... )

    .. seealso::

        `IQPopt: Fast optimization of instantaneous quantum polynomial circuits in JAX <https://arxiv.org/abs/2501.04776>`_
    """

    #: Circuit structure mapping parameter indices to lists of gates.
    gates: dict[int, list[list[int]]] = None
    #: Number of random bitstrings drawn for the estimation.
    n_samples: int = None
    #: JAX PRNG key for random bitstring generation.
    key: ArrayLike = None
    #: Total number of qubits in the circuit.
    n_qubits: int = None
    #: Pauli observables encoded as an integer array, or ``None``.
    observables: ArrayLike | None = None
    #: Computational-basis states with non-zero amplitude, or ``None``.
    init_state_elems: ArrayLike | None = None
    #: Amplitudes for the custom initial state, or ``None``.
    init_state_amps: ArrayLike | None = None
    #: Optional custom phase function applied as an extra diagonal layer.
    phase_fn: Callable | None = None
    #: Controls the memory usage of the generator matrix during the phase difference computation. Higher block size decreases memory usage.
    block_size: int = 1 << 17


def _parse_generator_dict(circuit_def: dict[int, list[list[int]]], n_qubits: int):
    """Convert a gate dictionary into a binary generator matrix and parameter map.

    Args:
        circuit_def (dict[int, list[list[int]]]): Dictionary mapping parameter indices to
            lists of qubit indices.
        n_qubits (int): Total number of qubits.

    Returns:
        tuple[jnp.ndarray, jnp.ndarray]: Tuple containing:
            - Integer array of shape ``(n_gates, max_weight)`` of qubit indices, padded with ``n_qubits``.
            - Integer array mapping each generator to its parameter index.
    """
    flat_gates = []
    param_indices = []

    for param_idx in sorted(circuit_def.keys()):
        gates_for_this_param = circuit_def[param_idx]
        for gate in gates_for_this_param:
            flat_gates.append(gate)
            param_indices.append(param_idx)

    n_gates = len(flat_gates)
    lengths = {len(gate) for gate in flat_gates}
    width = max(max(lengths, default=0), 1)
    rows = np.full((n_gates, width), n_qubits)

    for i, qubits in enumerate(flat_gates):
        unique = np.asarray(qubits).reshape(-1)
        unique = np.unique(np.where(unique < 0, unique + n_qubits, unique))

        if unique.size and (unique.min() < 0 or unique.max() >= n_qubits):
            raise IndexError(f"Qubit index out of range for a {n_qubits}-qubit circuit")

        rows[i, : unique.size] = unique

    gate_indices = jnp.asarray(np.ascontiguousarray(rows))

    return gate_indices, jnp.array(param_indices, dtype=int)


def _xor_gather_rows(bits: jnp.ndarray, gate_indices: jnp.ndarray) -> jnp.ndarray:
    parity = bits[gate_indices[:, 0]]

    for slot in range(1, gate_indices.shape[1]):
        parity = parity ^ bits[gate_indices[:, slot]]

    return parity


def _pad_sentinel_row(bits: jnp.ndarray) -> jnp.ndarray:
    return jnp.concatenate(
        [bits.astype(jnp.uint8), jnp.zeros((1,) + bits.shape[1:], jnp.uint8)], axis=0
    )


def _parity_dot(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    a = jnp.asarray(a)
    b = jnp.asarray(b)
    dims = (((a.ndim - 1,), (b.ndim - 1,)), ((), ()))

    product = jax.lax.dot_general(a, b, dims)
    return product % 2


def _parity_signs(parity: jnp.ndarray) -> jnp.ndarray:
    return 1 - 2 * parity.astype(jnp.float32)


def _phase_differences(
    gate_params: jnp.ndarray,
    samples_t: jnp.ndarray,
    bitflips_t: jnp.ndarray,
    gate_indices: jnp.ndarray,
    param_map: jnp.ndarray,
    block_size: int,
) -> jnp.ndarray:

    n_gates, max_weight = gate_indices.shape

    def block_contribution(block_indices: jnp.ndarray, block_params: jnp.ndarray) -> jnp.ndarray:
        theta = gate_params[block_params][:, jnp.newaxis]
        b_bits = _xor_gather_rows(samples_t, block_indices)
        q_bits = _xor_gather_rows(bitflips_t, block_indices)
        b_scaled = jnp.where(b_bits.astype(bool), -theta, theta)

        return jax.lax.dot_general(q_bits, b_scaled, (((0,), (0,)), ((), ())))

    if n_gates <= block_size:
        return 2 * block_contribution(gate_indices, param_map)

    n_blocks = -(-n_gates // block_size)
    n_pad = n_blocks * block_size - n_gates

    if n_pad:
        sentinel = samples_t.shape[0] - 1
        gate_indices = jnp.concatenate(
            [gate_indices, jnp.full((n_pad, max_weight), sentinel, gate_indices.dtype)]
        )
        param_map = jnp.concatenate([param_map, jnp.zeros((n_pad,), param_map.dtype)])

    def accumulate(total, block):
        return total + block_contribution(*block), None

    zero = jnp.zeros((bitflips_t.shape[1], samples_t.shape[1]))
    total, _ = jax.lax.scan(
        accumulate,
        zero,
        (
            gate_indices.reshape(n_blocks, block_size, max_weight),
            param_map.reshape(n_blocks, block_size),
        ),
    )

    return 2 * total


def _compute_samples(key: ArrayLike, n_samples: int, n_qubits: int) -> jnp.ndarray:
    """Generate the random bitstrings used by the Monte Carlo estimator."""
    n_bytes = (n_qubits + 7) // 8
    random_bytes = jax.random.bits(key, shape=(n_samples, n_bytes), dtype=jnp.uint8)
    unpacked_bits = jnp.unpackbits(random_bytes, axis=-1)
    return unpacked_bits[:, :n_qubits]


def _prep_observables(observables_int: ArrayLike) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Precompute masks and phase factors for integer-encoded Pauli observables."""
    obs_arr = jnp.asarray(observables_int, dtype=jnp.int32)

    is_X = obs_arr == 1
    is_Y = obs_arr == 2
    is_Z = obs_arr == 3

    bitflips = jnp.array(is_Z | is_Y, dtype=jnp.uint8)
    mask_XY = jnp.array(is_X | is_Y, dtype=jnp.uint8)

    count_Y = is_Y.sum(axis=1, dtype=jnp.int32) & 3
    y_real = jnp.where(count_Y == 0, 1.0, jnp.where(count_Y == 2, -1.0, 0.0))[:, jnp.newaxis]
    y_imag = jnp.where(count_Y == 1, -1.0, jnp.where(count_Y == 3, 1.0, 0.0))[:, jnp.newaxis]

    return bitflips, mask_XY, y_real, y_imag


# pylint: disable=too-many-arguments
def _core_expval_execution(
    gates_params: ArrayLike,
    phase_fn_params: ArrayLike | None,
    samples: jnp.ndarray,
    obs_data: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    init_state_elems: ArrayLike | None,
    init_state_amps: ArrayLike | None,
    gate_indices: jnp.ndarray,
    param_map: jnp.ndarray,
    vmapped_phase_func: Callable | None,
    block_size: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Evaluate the Monte Carlo integrand and return expectation values and variances."""
    bitflips, mask_XY, y_real, y_imag = obs_data

    samples_t = _pad_sentinel_row(samples.T)
    bitflips_t = _pad_sentinel_row(bitflips.T)

    gates_params = jnp.asarray(gates_params)
    E = _phase_differences(gates_params, samples_t, bitflips_t, gate_indices, param_map, block_size)

    if vmapped_phase_func is not None:
        E += vmapped_phase_func(phase_fn_params, samples, bitflips)

    cos_E = jnp.cos(E)
    sin_E = jnp.sin(E)

    sign_flip = _parity_signs(_parity_dot(mask_XY, samples))
    phase_re = sign_flip * (y_real * cos_E - y_imag * sin_E)
    phase_im = sign_flip * (y_real * sin_E + y_imag * cos_E)

    if init_state_elems is None or init_state_amps is None:
        integrand = phase_re
    else:
        state_elems = jnp.asarray(init_state_elems)
        amps = jnp.asarray(init_state_amps)

        g_signs = _parity_signs(_parity_dot(state_elems, samples))
        w_signs = _parity_signs(_parity_dot(bitflips, state_elems))

        amps_re = jnp.real(amps)
        amps_im = jnp.imag(amps)

        stacked = jnp.concatenate(
            [amps_re[:, jnp.newaxis] * g_signs, amps_im[:, jnp.newaxis] * g_signs], axis=1
        )

        overlap = jax.lax.dot_general(w_signs, stacked, (((1,), (0,)), ((), ())))

        n_samples = samples.shape[0]
        overlap_re = overlap[:, :n_samples]
        overlap_im = overlap[:, n_samples:]

        col_re = amps_re @ g_signs
        col_im = amps_im @ g_signs

        h_re = overlap_re * col_re + overlap_im * col_im
        h_im = overlap_re * col_im - overlap_im * col_re

        integrand = phase_re * h_re - phase_im * h_im

    expvals = jnp.mean(integrand, axis=1)
    variances = jnp.var(integrand, axis=-1, ddof=1) / samples.shape[0]

    return expvals, variances


def build_expval_func(
    config: CircuitConfig,
) -> Callable:
    """Build an estimator for Pauli expectation values of a qubit IQP circuit.

    Returns a pure function that estimates the expectation value of each Pauli
    observable specified in ``config.observables`` or passed at call time.

    The returned function captures precomputed data from ``config`` (generator
    matrices, default samples, preprocessed observables) so that repeated
    evaluations with different parameters are fast.

    Args:
        config (CircuitConfig): Full circuit description including gate
            structure, observables, and sampling parameters. See
            :class:`CircuitConfig` for details on how to construct one.

    Returns:
        Callable: A function with signature::

            expval_fn(
                gates_params,
                phase_fn_params=None,
                observables=None,
                key=None,
                n_samples=None,
                init_state_elems=None,
                init_state_amps=None,
            ) -> (expvals, variances)

        where ``expvals`` is a real array of shape ``(n_observables,)`` and
        ``variances`` contains the estimated variance of each expectation-value
        estimator.

    **Example**

    >>> import jax
    >>> import jax.numpy as jnp
    >>> from pennylane.labs.tcdq import CircuitConfig, build_expval_func, create_local_gates
    >>> n_qubits = 4
    >>> gates = create_local_gates(n_qubits, max_weight=2)
    >>> config = CircuitConfig(
    ...     gates=gates,
    ...     n_samples=5000,
    ...     key=jax.random.PRNGKey(0),
    ...     n_qubits=n_qubits,
    ...     observables=[[3, 3, 0, 0], [0, 0, 3, 3]],  # ZZ on (0,1) and (2,3)
    ... )
    >>> expval_fn = jax.jit(build_expval_func(config))
    >>> params = jnp.zeros(len(gates))
    >>> expvals, variances = expval_fn(params)
    >>> expvals.shape
    (2,)

    .. seealso::

        :class:`~pennylane.labs.tcdq.CircuitConfig`,
        `IQPopt: Fast optimization of instantaneous quantum polynomial circuits in JAX <https://arxiv.org/abs/2501.04776>`_
    """
    gate_indices, param_map = _parse_generator_dict(config.gates, config.n_qubits)

    vmapped_phase_func = None
    if config.phase_fn is not None:

        def compute_phase(p_params, sample, b_flips):
            return config.phase_fn(p_params, sample) - config.phase_fn(
                p_params, (sample + b_flips) % 2
            )

        vmapped_phase_func = jax.vmap(
            jax.vmap(compute_phase, in_axes=(None, 0, None)), in_axes=(None, None, 0)
        )

    default_samples = _compute_samples(config.key, config.n_samples, config.n_qubits)
    default_obs_data = None if config.observables is None else _prep_observables(config.observables)

    # pylint: disable=too-many-arguments
    def expval_execution(
        gates_params: ArrayLike,
        phase_fn_params: ArrayLike | None = None,
        observables: ArrayLike | None = None,
        key: ArrayLike | None = None,
        n_samples: int | None = None,
        init_state_elems: ArrayLike | None = None,
        init_state_amps: ArrayLike | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Execute the estimator with optional runtime overrides.

        This closure captures the precomputed matrices and defaults from the
        CircuitConfig, while allowing dynamic injection of new parameters,
        observables, or sampling configurations at execution time.

        Args:
            gates_params (ArrayLike): Trainable parameters $\\theta$ for the circuit gates.
            phase_fn_params (ArrayLike | None, optional): Trainable parameters for the
                custom phase function. Defaults to None.
            observables (ArrayLike | None, optional): Runtime override for the Pauli
                observables (I=0, X=1, Y=2, Z=3). Defaults to None.
            key (ArrayLike | None, optional): Runtime override for the JAX PRNG key
                used for sampling. Defaults to None.
            n_samples (int | None, optional): Runtime override for the number of
                samples. Defaults to None.
            init_state_elems (ArrayLike | None, optional): Runtime override for the
                discrete elements of the initial state (X). Defaults to None.
            init_state_amps (ArrayLike | None, optional): Runtime override for the
                continuous amplitudes of the initial state (P). Defaults to None.

        Returns:
            tuple[jnp.ndarray, jnp.ndarray]: Estimated expectation values and
            the estimated variances of those estimators.
        """
        if key is not None or n_samples is not None:
            _key = key if key is not None else config.key
            _n = n_samples if n_samples is not None else config.n_samples
            samples = _compute_samples(_key, _n, config.n_qubits)
        else:
            samples = default_samples

        if observables is not None:
            obs_data = _prep_observables(observables)
        elif default_obs_data is not None:
            obs_data = default_obs_data
        else:
            raise ValueError(
                "No observables specified. Provide them in CircuitConfig "
                "or pass at call time via the observables argument."
            )

        state_elems = config.init_state_elems if init_state_elems is None else init_state_elems
        state_amps = config.init_state_amps if init_state_amps is None else init_state_amps

        return _core_expval_execution(
            gates_params,
            phase_fn_params,
            samples,
            obs_data,
            state_elems,
            state_amps,
            gate_indices,
            param_map,
            vmapped_phase_func,
            config.block_size,
        )

    return expval_execution
