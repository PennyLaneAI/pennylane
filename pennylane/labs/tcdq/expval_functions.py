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
from jax import lax
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


def _flatten_gate_dict(circuit_def: dict[int, list[list[int]]]):
    """Flatten a gate dictionary into a gate list and a matching parameter-index list."""
    flat_gates = []
    param_indices = []

    for param_idx in sorted(circuit_def.keys()):
        gates_for_this_param = circuit_def[param_idx]
        for gate in gates_for_this_param:
            flat_gates.append(gate)
            param_indices.append(param_idx)

    return flat_gates, param_indices


def _parse_generator_dict(circuit_def: dict[int, list[list[int]]], n_qubits: int):
    """Convert a gate dictionary into a padded array of gate qubit indices.

    This is the sparse counterpart of :func:`_parse_generator_dict`. Instead of a dense
    ``(n_gates, n_qubits)`` binary matrix it returns the qubit indices touched by each
    gate, right-padded with the sentinel index ``n_qubits``. Downstream code appends a
    zero row/column at that sentinel position, so padded entries are neutral for the
    parity (XOR) reductions. Generator parities then cost ``O(max_weight)`` gathers per
    gate rather than an ``O(n_qubits)`` inner product.

    Args:
        circuit_def (dict[int, list[list[int]]]): Dictionary mapping parameter indices to
            lists of qubit indices.
        n_qubits (int): Total number of qubits.

    Returns:
        tuple[jnp.ndarray, jnp.ndarray]: Tuple containing:
            - Integer array of shape ``(n_gates, max_weight)`` of qubit indices,
              padded with ``n_qubits``.
            - Integer array mapping each generator to its parameter index.
    """
    flat_gates, param_indices = _flatten_gate_dict(circuit_def)
    n_gates = len(flat_gates)

    lengths = {len(gate) for gate in flat_gates}
    rows = None
    if len(lengths) == 1 and n_gates:
        # Fast path: every gate has the same weight, so the qubit lists form a matrix.
        candidate = np.asarray(flat_gates, dtype=np.int64).reshape(n_gates, -1)
        candidate = np.where(candidate < 0, candidate + n_qubits, candidate)
        no_duplicates = candidate.shape[1] < 2 or np.all(
            np.diff(np.sort(candidate, axis=1), axis=1) != 0
        )
        if no_duplicates:
            rows = candidate

    if rows is None:
        # General path: ragged weights and/or repeated qubits within a gate.
        width = max(max(lengths, default=0), 1)
        rows = np.full((n_gates, width), n_qubits, dtype=np.int64)
        for i, qubits in enumerate(flat_gates):
            unique = np.asarray(qubits, dtype=np.int64).reshape(-1)
            unique = np.unique(np.where(unique < 0, unique + n_qubits, unique))
            rows[i, : unique.size] = unique

    if n_gates and rows.size:
        if rows.min() < 0 or rows.max() > n_qubits:
            raise IndexError(f"Qubit index out of range for a {n_qubits}-qubit circuit")

    gate_indices = jnp.asarray(np.ascontiguousarray(rows, dtype=np.int32))
    return gate_indices, jnp.array(param_indices, dtype=int)


def _xor_gather_rows(bits: jnp.ndarray, gate_indices: jnp.ndarray) -> jnp.ndarray:
    """XOR-reduce rows of ``bits`` over each gate support.

    Args:
        bits: ``(n_qubits + 1, n_cols)`` array of bits whose last row is zero.
        gate_indices: ``(n_gates, max_weight)`` padded qubit indices.

    Returns:
        ``(n_gates, n_cols)`` parity bits.
    """
    out = bits[gate_indices[:, 0]]
    for slot in range(1, gate_indices.shape[1]):
        out = out ^ bits[gate_indices[:, slot]]
    return out


def _xor_gather_cols(bits: jnp.ndarray, gate_indices: jnp.ndarray) -> jnp.ndarray:
    """XOR-reduce columns of ``bits`` over each gate support.

    Args:
        bits: ``(n_rows, n_qubits + 1)`` array of bits whose last column is zero.
        gate_indices: ``(n_gates, max_weight)`` padded qubit indices.

    Returns:
        ``(n_rows, n_gates)`` parity bits.
    """
    out = bits[:, gate_indices[:, 0]]
    for slot in range(1, gate_indices.shape[1]):
        out = out ^ bits[:, gate_indices[:, slot]]
    return out


#: Target number of entries per generator-block operand. The blocked contraction below
#: keeps both operands of the inner matrix product around this size so that they are
#: built and consumed in cache instead of being streamed through main memory.
_BLOCK_ELEMENTS = 2_000_000


def _block_size(n_gates: int, n_obs: int, n_samples: int) -> int:
    """Choose how many generators to process per block of the phase contraction."""
    width = max(n_obs, n_samples, 1)
    return int(min(max(_BLOCK_ELEMENTS // width, 1024), 32768, max(n_gates, 1)))


def _phase_differences(
    gates_params: jnp.ndarray,
    samples_t: jnp.ndarray,
    bitflips_padded: jnp.ndarray,
    gate_indices: jnp.ndarray,
    param_map: jnp.ndarray,
) -> jnp.ndarray:
    r"""Accumulate the phase difference matrix of the IQP estimator.

    Computes, for every observable ``o`` and sample ``s``,

    .. math:: E_{os} = 2 \sum_g q_{og}\,\theta_g\,(-1)^{s \cdot S_g}

    where :math:`q_{og}` is one when the observable's bitflip string overlaps generator
    :math:`S_g` in an odd number of qubits. The generator axis is processed in blocks so
    that both operands of the inner matrix product are built and consumed in cache.
    """
    n_gates, width = gate_indices.shape
    n_obs = bitflips_padded.shape[0]
    n_samples = samples_t.shape[1]
    dtype = jnp.result_type(jnp.asarray(gates_params).dtype, jnp.float32)

    def block(gidx, pmap):
        params = jnp.asarray(gates_params)[pmap].astype(dtype)[:, jnp.newaxis]
        b_bits = _xor_gather_rows(samples_t, gidx)
        q_bits = _xor_gather_cols(bitflips_padded, gidx)
        b_scaled = jnp.where(b_bits.astype(bool), -params, params)
        return q_bits.astype(dtype) @ b_scaled

    size = _block_size(n_gates, n_obs, n_samples)
    n_blocks = -(-n_gates // size) if n_gates else 1

    if n_blocks <= 1:
        return 2 * block(gate_indices, param_map)

    # Pad the generator axis so it splits evenly. Padded entries carry the sentinel
    # qubit index, hence zero bitflip overlap, hence no contribution to the sum.
    pad = n_blocks * size - n_gates
    gidx = jnp.concatenate(
        [gate_indices, jnp.full((pad, width), samples_t.shape[0] - 1, gate_indices.dtype)]
    ).reshape(n_blocks, size, width)
    pmap = jnp.concatenate([param_map, jnp.zeros((pad,), param_map.dtype)]).reshape(n_blocks, size)

    def step(acc, xs):
        return acc + block(*xs), None

    total, _ = lax.scan(step, jnp.zeros((n_obs, n_samples), dtype), (gidx, pmap))
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

    bitflips = jnp.array(is_Z | is_Y, dtype=jnp.int32)
    mask_XY = jnp.array(is_X | is_Y, dtype=jnp.int32)
    count_Y = jnp.array(is_Y.sum(axis=1), dtype=jnp.int32)

    y_phase = (-1j) ** count_Y[:, jnp.newaxis]

    return bitflips, mask_XY, y_phase


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
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Evaluate the Monte Carlo integrand and return expectation values and variances."""
    bitflips, mask_XY, y_phase = obs_data

    n_samples = samples.shape[0]
    n_obs = bitflips.shape[0]

    dtype = jnp.result_type(jnp.asarray(gates_params).dtype, jnp.float32)

    s_f = samples.astype(dtype)
    sign_flip = 1 - 2 * ((mask_XY.astype(dtype) @ s_f.T) % 2)
    phases = sign_flip * y_phase

    # Generator parities are XOR reductions over each gate's support. Pad the sample
    # bits and the observable bitflips with a zero at the sentinel index so that padded
    # slots of ``gate_indices`` contribute nothing to the parity.
    samples_t = jnp.concatenate(
        [samples.astype(jnp.uint8).T, jnp.zeros((1, n_samples), dtype=jnp.uint8)], axis=0
    )
    bitflips_padded = jnp.concatenate(
        [bitflips.astype(jnp.uint8), jnp.zeros((n_obs, 1), dtype=jnp.uint8)], axis=1
    )

    E = _phase_differences(gates_params, samples_t, bitflips_padded, gate_indices, param_map)

    if vmapped_phase_func is not None:
        E += vmapped_phase_func(phase_fn_params, samples, bitflips)

    if init_state_elems is None or init_state_amps is None:
        integrand = jnp.real(phases) * jnp.cos(E) - jnp.imag(phases) * jnp.sin(E)
    else:
        M = phases * jnp.exp(1j * E)
        X = jnp.asarray(init_state_elems)
        P = jnp.asarray(init_state_amps)
        x_f = X.astype(dtype)
        F = P[:, jnp.newaxis] * (1 - 2 * ((x_f @ s_f.T) % 2))
        H1 = (1 - 2 * ((bitflips.astype(dtype) @ x_f.T) % 2)) @ F
        col_sums = jnp.sum(F.conj(), axis=0, keepdims=True)
        H = H1 * col_sums
        M = M * H
        integrand = jnp.real(M)

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
        )

    return expval_execution
