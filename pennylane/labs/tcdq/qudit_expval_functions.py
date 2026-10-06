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
"""Expectation-value estimator for qudit IQP circuits.

This module extends :mod:`~pennylane.labs.tcdq.expval_functions` from qubits
to qudits. It estimates Heisenberg-Weyl moments without building the full
quantum state.

The estimator samples random dit-strings, evaluates an observable-dependent
phase, evaluates a circuit-dependent phase difference, and averages the
resulting complex integrand.

For further information, see
`Section 2, Classically Estimating Expectation Values <https://github.com/PennyLaneAI/pennylane/blob/port_tcdq_docs_pr/pennylane/labs/tcdq/notes.md#2-classically-estimating-expectation-values>`_,
`Section 3, General Input States <https://github.com/PennyLaneAI/pennylane/blob/port_tcdq_docs_pr/pennylane/labs/tcdq/notes.md#3-general-input-states>`_,
and `Section 4, Monte Carlo Statistics <https://github.com/PennyLaneAI/pennylane/blob/port_tcdq_docs_pr/pennylane/labs/tcdq/notes.md#4-monte-carlo-statistics>`_
of the technical notes.
"""

import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

_REAL_DTYPE = jnp.float32
_INDEX_DTYPE = jnp.int32
_SQRT2 = float(np.sqrt(2.0))


@dataclass
class QuditCircuitConfig:  # pylint: disable=too-many-instance-attributes
    r"""A class to store qudit IQP circuit configurations.

    This class stores the description of a qudit IQP circuit to compute its expectation value with respect to a
    Heisenberg-Weyl (HW) observable. See `arXiv:2607.06675 <https://arxiv.org/abs/2607.06675>`_ for theoretical details.

    A qudit IQP circuit is in the form :math:`U(\mathbf{\theta}) = \left( F^{\otimes n} \right)^\dagger D(\mathbf{\theta}) F^{\otimes n}`
    where :math:`F` is the Fourier transform and :math:`D(\mathbf{\theta})` is a diagonal phase unitary on `n` qudits.
    The diagonal phase unitary is given by a gate set :math:`\mathcal{G}`,

    .. math::

        D(\mathbf{\theta}) = \prod_{\mathbf{g} \in \mathcal{G}} \exp \left( i \theta_\mathbf{g} \mathcal{Q}_\mathbf{g} \right)

    where :math:`\mathbf{\theta}_\mathbf{g}` is a vector parameterizing the gate :math:`\mathbf{g}` and :math:`\mathcal{Q}_\mathbf{g}` is
    the Hermitian counterpart to an HW observable. Optionally, one can specify an additional trainable phase layer
    :math:`D'(\mathbf{\xi})\vert z \rangle = \exp \left( i f_{\mathbf{\xi}}(z) \right) \vert z \rangle`
    where :math:`f_{\mathbf{\xi}}(z)` is a trainable function parameterized by :math:`\mathbf{\xi}`.
    After including the phase layer, the final trainable circuit becomes
    :math:`\left( F^{\otimes n} \right)^\dagger D'(\mathbf{\xi}) D(\mathbf{\theta}) F^{\otimes n}`.

    This dataclass collects the circuit data needed by
    :func:`build_qudit_expval_func`. It is the qudit analogue of
    :class:`~pennylane.labs.tcdq.CircuitConfig`.

    Args:
        dims (int | Sequence[int]): Local qudit dimension(s). Either a single
            ``int`` (e.g., 2 for qubits, 3 for qutrits), which is broadcast to
            every qudit, or a sequence of length ``n_qudits`` giving a distinct
            dimension :math:`d_j` per qudit.
        n_qudits (int): Number of qudits in the circuit.
        gates (dict[int, list[list[tuple[int, int]]]]): Circuit structure mapping
            each trainable-parameter index to a list of generators. Each generator
            is a sparse list of ``(qudit, power)`` pairs, where ``power`` in
            :math:`\{1, \ldots, d_j-1\}` is the power of :math:`Z` acting on that
            qudit; qudits that are not listed act as the identity. For example,
            with ``d=3`` and ``n_qudits=2``,
            ``{0: [[(0, 1)]], 1: [[(1, 1)]], 2: [[(0, 1), (1, 1)]]}`` defines
            three gates: :math:`Z^1` on qudit 0, :math:`Z^1` on qudit 1, and
            :math:`Z^1 \otimes Z^1` on both. The memory used by this dictionary
            scales with the number of active qudits per gate, not with
            ``n_qudits``. Dense generator vectors of length ``n_qudits`` (for the
            example above ``{0: [[1, 0]], 1: [[0, 1]], 2: [[1, 1]]}``) are also
            accepted and converted internally.
        n_samples (int): Number of random dit-strings drawn for the
            estimation.
        key (ArrayLike): JAX PRNG key for random dit-string generation.
        observables (tuple[ArrayLike, ArrayLike] | None): A pair
            ``(l_vecs, m_vecs)`` specifying the Heisenberg–Weyl displacement
            operators :math:`O(\mathbf{l}, \mathbf{m})` to measure.
            Each is an integer array of shape ``(n_obs, n_qudits)`` with entries
            in :math:`\{0, \ldots, d-1\}`. If ``None``, observables must be
            supplied at call time (e.g., when used inside
            :func:`~pennylane.labs.tcdq.build_mmd_loss_hw`).
        init_state_elems (ArrayLike | None): Support of a custom initial state.
            Integer array of shape ``(N, n_qudits)`` with entries in
            :math:`\{0, \ldots, d-1\}`, where ``N`` is the number of non-zero
            amplitudes. Defaults to ``None`` (uniform superposition via QFT).
        init_state_amps (ArrayLike | None): Complex amplitudes of shape ``(N,)``
            for the custom initial state. Must be provided together with
            ``init_state_elems``.
        phase_fn (Callable | None): Optional phase layer with trainable parameters. The phase layer
            :math:`D'(\mathbf{\xi})` is given by a ``Callable`` with signature ``(params: ArrayLike, z: ArrayLike) -> scalar``
            where ``z`` is a dit-string of shape ``(n_qudits, )`` with entries in :math:`\{0, \dots, d-1\}` and
            ``params`` has shape matching :math:`\mathbf{\xi}`.
        block_size (int): Number of gates processed per step when accumulating the
            circuit phase. Circuits with at most ``block_size`` gates are evaluated in a
            single step with cached sample-side factors; larger circuits are accumulated
            block by block, so peak memory scales with
            ``block_size * 2**max_weight * (n_samples + n_obs)`` instead of with the total
            number of gates. Defaults to ``8192``.

    **Example**

    >>> import jax
    >>> import jax.numpy as jnp
    >>> from pennylane.labs.tcdq import QuditCircuitConfig
    >>> config = QuditCircuitConfig(
    ...     dims=3,
    ...     n_qudits=4,
    ...     gates={0: [[(0, 1)]], 1: [[(1, 1)]], 2: [[(0, 1), (1, 1)]]},
    ...     observables=(
    ...         jnp.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=jnp.int32),
    ...         jnp.zeros((2, 4), dtype=jnp.int32),
    ...     ),
    ...     n_samples=5000,
    ...     key=jax.random.PRNGKey(42),
    ... )
    """

    #: Local qudit dimension(s): an int (uniform) or list (per-qudit sequence).
    dims: int | Sequence[int] = None
    #: Number of qudits in the circuit.
    n_qudits: int = None
    #: Circuit structure mapping parameter indices to sparse generators.
    gates: dict[int, list[list[tuple[int, int]]]] = None
    #: Number of random dit-strings drawn for the estimation.
    n_samples: int = None
    #: JAX PRNG key for random dit-string generation.
    key: ArrayLike = None
    #: Heisenberg–Weyl observables ``(l_vecs, m_vecs)``, or ``None``.
    observables: tuple[ArrayLike, ArrayLike] | None = None
    #: Support of a custom initial state, or ``None``.
    init_state_elems: ArrayLike | None = None
    #: Amplitudes for the custom initial state, or ``None``.
    init_state_amps: ArrayLike | None = None
    #: Learnable phase layer
    phase_fn: Callable | None = None
    #: Number of gates per block in the phase-difference accumulation. Higher values increase memory usage.
    block_size: int = 1 << 13


def _dims_to_numpy(dims: int | Sequence[int], n_qudits: int) -> np.ndarray:
    """Normalize the ``dims`` field to an integer array of per-qudit dimensions.

    Accepts either a scalar ``int`` (broadcast to all qudits, the uniform case)
    or a sequence of length ``n_qudits`` (mixed-dimension case), and always
    returns a NumPy integer array of shape ``(n_qudits,)``.

    Raises:
        ValueError: If ``dims`` is a sequence whose length is not ``n_qudits``.
    """
    if isinstance(dims, int):
        return np.full((n_qudits,), int(dims), dtype=int)

    normalized_dims = np.asarray(dims, dtype=int)
    if normalized_dims.shape != (n_qudits,):
        raise ValueError(
            f"d given as a sequence must have length n_qudits={n_qudits}, "
            f"got shape {normalized_dims.shape}."
        )

    return normalized_dims


def _gate_pairs_from_array(gate: ArrayLike, n_qudits: int) -> list[tuple[int, int]]:
    """Convert an array-like generator (dense vector or array of pairs) into ``(qudit, power)`` pairs.

    Raises:
        ValueError: If the generator is malformed, non-integer, or a dense vector whose length
            is not ``n_qudits``.
    """
    try:
        arr = np.asarray(gate)
    except ValueError as exc:
        raise ValueError(
            "Generator must be a sequence of (qudit, power) pairs or a dense vector of "
            f"length n_qudits={n_qudits}; got {gate!r}."
        ) from exc

    if arr.dtype.kind not in "iu":
        rounded = np.asarray(np.round(arr), dtype=np.int64) if arr.size else arr.astype(np.int64)
        if arr.dtype.kind == "f" and np.array_equal(rounded, arr):
            arr = rounded
        else:
            raise ValueError(f"Generator entries must be integers; got {gate!r}.")

    if arr.size == 0:
        return []

    if arr.ndim == 1:  # dense vector of powers
        if arr.shape[0] != n_qudits:
            raise ValueError(
                f"Dense generator has length {arr.shape[0]}, expected n_qudits={n_qudits}. "
                "Sparse generators must be given as (qudit, power) pairs."
            )
        return [(q, p) for q, p in enumerate(arr.tolist()) if p != 0]

    if arr.ndim == 2 and arr.shape[1] == 2:  # sparse (qudit, power) pairs
        return [tuple(pair) for pair in arr.tolist()]

    raise ValueError(
        "Generator must be a sequence of (qudit, power) pairs or a dense vector of "
        f"length n_qudits={n_qudits}; got an array of shape {arr.shape}."
    )


def _normalize_qudit_gate(gate: ArrayLike, n_qudits: int) -> tuple[list[int], list[int]]:
    """Convert one generator (sparse or dense) into sorted ``(qudit_indices, powers)`` lists.

    A sparse generator is a sequence of ``(qudit, power)`` pairs; a dense generator is a
    vector of length ``n_qudits`` whose entries are the powers of :math:`Z` on each qudit.
    Entries with power ``0`` are dropped in both cases. Sequences of integer pairs are handled
    without NumPy so that parsing millions of gates stays cheap.

    Raises:
        ValueError: If the generator is malformed, non-integer, has a negative power, lists the
            same qudit twice, or is a dense vector whose length is not ``n_qudits``.
        IndexError: If a qudit index is out of range for an ``n_qudits``-qudit circuit.
    """
    if isinstance(gate, (list, tuple)) and all(
        isinstance(pair, (list, tuple))
        and len(pair) == 2
        and all(isinstance(entry, (int, np.integer)) for entry in pair)
        for pair in gate
    ):
        pairs = [(int(q), int(p)) for q, p in gate]
    else:
        pairs = _gate_pairs_from_array(gate, n_qudits)

    indices = [q + n_qudits if q < 0 else q for q, _ in pairs]
    if any(q < 0 or q >= n_qudits for q in indices):
        raise IndexError(f"Qudit index out of range for a {n_qudits}-qudit circuit: {gate!r}.")
    if len(set(indices)) != len(indices):
        raise ValueError(f"Generator lists the same qudit more than once: {gate!r}.")
    if any(p < 0 for _, p in pairs):
        raise ValueError(f"Generator powers must be non-negative: {gate!r}.")

    kept = sorted((q, p) for q, (_, p) in zip(indices, pairs) if p != 0)
    return [q for q, _ in kept], [p for _, p in kept]


def _parse_qudit_generator_dict(
    circuit_def: dict[int, list[list[tuple[int, int]]]], n_qudits: int
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Convert a qudit gate dictionary into padded index/power arrays and a parameter map.

    Each generator is stored sparsely as the qudits it acts on and the power of :math:`Z`
    applied to each of them. Generators lighter than ``max_weight`` are padded with the
    sentinel index ``n_qudits`` and power ``0`` (the identity), so all returned rows have
    the same width and the memory footprint does not depend on ``n_qudits``.

    Args:
        circuit_def (dict[int, list[list[tuple[int, int]]]]): Maps parameter indices to
            lists of generators. Each generator is a sequence of ``(qudit, power)`` pairs;
            dense vectors of length ``n_qudits`` are also accepted.
        n_qudits (int): Number of qudits.

    Returns:
        tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]: Tuple containing:
            - Integer array of shape ``(n_gates, max_weight)`` of qudit indices, padded
              with ``n_qudits``.
            - Integer array of shape ``(n_gates, max_weight)`` of :math:`Z` powers, padded
              with ``0``.
            - Integer array of shape ``(n_gates,)`` mapping each gate to its parameter index.

    Raises:
        ValueError: If a generator is malformed (see :func:`_normalize_qudit_gate`).
        IndexError: If a qudit index is out of range.
    """
    param_keys = sorted(circuit_def.keys())
    n_gates = sum(len(circuit_def[key]) for key in param_keys)
    # The number of entries of a generator bounds its weight; the arrays are allocated once
    # with that bound and trimmed to the largest actual weight afterwards.
    width_bound = max((len(gate) for key in param_keys for gate in circuit_def[key]), default=0)
    width_bound = max(width_bound, 1)

    gate_indices = np.full((n_gates, width_bound), n_qudits, dtype=np.int32)
    gate_powers = np.zeros((n_gates, width_bound), dtype=np.int32)
    param_indices = np.empty((n_gates,), dtype=np.int32)

    row = 0
    width = 1
    for param_idx in param_keys:
        for gate in circuit_def[param_idx]:
            indices, powers = _normalize_qudit_gate(gate, n_qudits)
            weight = len(indices)
            gate_indices[row, :weight] = indices
            gate_powers[row, :weight] = powers
            param_indices[row] = param_idx
            width = max(width, weight)
            row += 1

    return (
        jnp.asarray(np.ascontiguousarray(gate_indices[:, :width]), dtype=_INDEX_DTYPE),
        jnp.asarray(np.ascontiguousarray(gate_powers[:, :width]), dtype=_INDEX_DTYPE),
        jnp.asarray(param_indices, dtype=_INDEX_DTYPE),
    )


class _BlockedGates(NamedTuple):
    """Gate arrays arranged for the blocked phase-difference accumulation.

    When ``use_scan`` is ``False`` the arrays have shapes ``(n_gates, max_weight)`` and
    ``(n_gates,)`` and are processed in a single step. Otherwise they are padded and
    reshaped to ``(n_blocks, block_size, max_weight)`` and ``(n_blocks, block_size)`` and
    processed with :func:`jax.lax.scan`.
    """

    #: Qudit indices of each gate, padded with the sentinel index ``n_qudits``.
    indices: jnp.ndarray
    #: Powers of :math:`Z` for each gate, padded with ``0``.
    powers: jnp.ndarray
    #: Parameter index of each gate.
    param_map: jnp.ndarray
    #: Whether the arrays are split into blocks that must be accumulated with a scan.
    use_scan: bool


def _block_gate_arrays(
    gate_indices: jnp.ndarray,
    gate_powers: jnp.ndarray,
    param_map: jnp.ndarray,
    block_size: int,
    sentinel_index: int,
) -> _BlockedGates:
    """Pad and reshape the gate arrays into ``block_size`` blocks for the scan.

    Padding gates have all powers equal to ``0`` and are masked out of the accumulation,
    so they contribute exactly nothing.
    """
    if block_size < 1:
        raise ValueError(f"block_size must be a positive integer, got {block_size}.")

    n_gates, width = gate_indices.shape
    if n_gates <= block_size:
        return _BlockedGates(gate_indices, gate_powers, param_map, use_scan=False)

    n_blocks = -(-n_gates // block_size)
    n_pad = n_blocks * block_size - n_gates

    indices = np.asarray(gate_indices)
    powers = np.asarray(gate_powers)
    params = np.asarray(param_map)
    if n_pad:
        indices = np.concatenate([indices, np.full((n_pad, width), sentinel_index, indices.dtype)])
        powers = np.concatenate([powers, np.zeros((n_pad, width), powers.dtype)])
        params = np.concatenate([params, np.zeros((n_pad,), params.dtype)])

    return _BlockedGates(
        indices=jnp.asarray(indices.reshape(n_blocks, block_size, width)),
        powers=jnp.asarray(powers.reshape(n_blocks, block_size, width)),
        param_map=jnp.asarray(params.reshape(n_blocks, block_size)),
        use_scan=True,
    )


def _compute_qudit_samples(
    key: ArrayLike, num_samples: int, n_qudits: int, dims: ArrayLike
) -> jnp.ndarray:
    """Generates uniformly random dit-strings from the product Z_{d_1} x ... x Z_{d_n}."""

    maxval = jnp.asarray(dims, dtype=jnp.int32)[jnp.newaxis, :]  # (1, n_qudits)
    return jax.random.randint(key, shape=(num_samples, n_qudits), minval=0, maxval=maxval)


def _pad_sentinel_row(vectors: ArrayLike) -> jnp.ndarray:
    """Transpose ``(n_vectors, n_qudits)`` dit-strings and append a zero sentinel row.

    The result has shape ``(n_qudits + 1, n_vectors)``; row ``n_qudits`` is gathered by the
    padding slots of light gates, whose power ``0`` makes their contribution the identity.
    """
    vectors_t = jnp.asarray(vectors, dtype=_INDEX_DTYPE).T
    sentinel = jnp.zeros((1, vectors_t.shape[1]), dtype=_INDEX_DTYPE)
    return jnp.concatenate([vectors_t, sentinel], axis=0)


def _block_scale(
    block_powers: jnp.ndarray, block_indices: jnp.ndarray, inv_d: jnp.ndarray
) -> jnp.ndarray:
    r"""Angle scale :math:`2\pi p / d_q` of every gate slot, shape ``(block, max_weight, 1)``.

    Padding slots have power ``0`` and therefore scale ``0``.
    """
    g = block_powers.astype(_REAL_DTYPE)
    return ((2 * jnp.pi) * g * inv_d[block_indices].astype(_REAL_DTYPE))[:, :, jnp.newaxis]


def _trig_factors(
    scale: jnp.ndarray, values: jnp.ndarray, shift: float, amplitude: float
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """``(amplitude * cos, amplitude * sin)`` of ``scale * values + shift``.

    Args:
        scale (jnp.ndarray): Output of :func:`_block_scale`, shape ``(block, max_weight, 1)``.
        values (jnp.ndarray): Dit-string entries at the gate slots, shape
            ``(block, max_weight, n_vectors)``.
        shift (float): Constant angle offset.
        amplitude (float): Prefactor of both factors.
    """
    angle = scale * values.astype(_REAL_DTYPE) + shift
    return amplitude * jnp.cos(angle), amplitude * jnp.sin(angle)


def _sample_factors(
    block_powers: jnp.ndarray,
    block_indices: jnp.ndarray,
    samples_t: jnp.ndarray,
    inv_d: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""Sample-side factors :math:`\sqrt{2}\cos(2\pi p z_q/d_q + \pi/4)` and the matching sines.

    Args:
        block_powers (jnp.ndarray): :math:`Z` powers, shape ``(block, max_weight)``.
        block_indices (jnp.ndarray): Qudit indices, shape ``(block, max_weight)``.
        samples_t (jnp.ndarray): Padded transposed samples, shape ``(n_qudits + 1, n_samples)``.
        inv_d (jnp.ndarray): Padded inverse local dimensions, shape ``(n_qudits + 1,)``.

    Returns:
        tuple[jnp.ndarray, jnp.ndarray]: Arrays of shape ``(block, max_weight, n_samples)``.
    """
    scale = _block_scale(block_powers, block_indices, inv_d)
    return _trig_factors(scale, samples_t[block_indices], jnp.pi / 4, _SQRT2)


def _obs_factors(
    block_powers: jnp.ndarray, block_indices: jnp.ndarray, l_t: jnp.ndarray, inv_d: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""Observable-side factors :math:`\cos(2\pi p l_q/d_q)` and :math:`\sin(2\pi p l_q/d_q)`.

    Same arguments as :func:`_sample_factors` with the padded transposed observable ``l``
    vectors of shape ``(n_qudits + 1, n_obs)`` in place of the samples.
    """
    scale = _block_scale(block_powers, block_indices, inv_d)
    return _trig_factors(scale, l_t[block_indices], 0.0, 1.0)


def _sigma_products(factors: tuple[jnp.ndarray, jnp.ndarray]) -> list[jnp.ndarray]:
    r"""Products over the gate slots for every cos/sin assignment :math:`\sigma`.

    Expanding :math:`\prod_k \sqrt{2}\cos(a_k - b_k + \pi/4)` with the angle-addition formula
    gives :math:`2^{\omega}` terms, each a product over slots of either the cosine or the sine
    factor. The first entry (all cosines) is the unshifted gate eigenvalue.

    Args:
        factors (tuple[jnp.ndarray, jnp.ndarray]): ``(cos, sin)`` factors of shape
            ``(block, max_weight, n_vectors)``.

    Returns:
        list[jnp.ndarray]: ``2**max_weight`` arrays of shape ``(block, n_vectors)``.
    """
    width = factors[0].shape[1]
    products = []
    for sigma in itertools.product((0, 1), repeat=width):
        product = factors[sigma[0]][:, 0, :]
        for k in range(1, width):
            product = product * factors[sigma[k]][:, k, :]
        products.append(product)
    return products


def _block_phase_contribution(
    theta: jnp.ndarray, sample_products: list[jnp.ndarray], obs_products: list[jnp.ndarray]
) -> jnp.ndarray:
    r"""Phase difference :math:`\sum_g \theta_g [v_g(z) - v_g(z - l)]` for one block of gates.

    Here :math:`v_g(z) = \prod_{k} \sqrt{2}\cos(2\pi g_k z_k / d_k + \pi/4)` is the eigenvalue
    of the gate generator, and the shifted term is the sum over :math:`\sigma` of the products
    of a sample-side and an observable-side factor, contracted over the gates.

    Args:
        theta (jnp.ndarray): Gate parameters, shape ``(block,)``.
        sample_products (list[jnp.ndarray]): Output of :func:`_sigma_products` for the samples,
            arrays of shape ``(block, n_samples)``.
        obs_products (list[jnp.ndarray]): Output of :func:`_sigma_products` for the observable
            ``l`` vectors, arrays of shape ``(block, n_obs)``.

    Returns:
        jnp.ndarray: Phase differences of shape ``(n_obs, n_samples)``.
    """
    theta_col = theta[:, jnp.newaxis]
    total = (theta @ sample_products[0])[jnp.newaxis, :]  # (1, n_samples)
    for B, C in zip(sample_products, obs_products):
        # (n_obs, n_samples) = sum over gates of theta_g * C[g, obs] * B[g, sample]
        total = total - jax.lax.dot_general(C * theta_col, B, (((0,), (0,)), ((), ())))
    return total


# pylint: disable=too-many-arguments
def _phase_differences(
    gates_params: ArrayLike,
    samples_t: jnp.ndarray,
    l_t: jnp.ndarray,
    inv_d: jnp.ndarray,
    blocked: _BlockedGates,
    *,
    sample_products: list[jnp.ndarray] | None = None,
) -> jnp.ndarray:
    """Accumulate the circuit phase differences over all gates.

    Circuits with at most ``block_size`` gates are processed in one step, optionally reusing
    ``sample_products`` precomputed for the default samples. Larger circuits are accumulated
    block by block with :func:`jax.lax.scan`; each block is rematerialized in the backward pass
    (:func:`jax.checkpoint`) so that peak memory in both the forward and the gradient
    computation is set by the block size, not by the number of gates.

    Returns:
        jnp.ndarray: Phase differences of shape ``(n_obs, n_samples)``.
    """
    gates_params = jnp.asarray(gates_params).astype(_REAL_DTYPE)

    def gate_parameters(block_powers, block_params):
        # Gates without any active qudit (padding, or explicit identities) are masked out.
        active = jnp.any(block_powers != 0, axis=1)
        return jnp.where(active, gates_params[block_params], 0.0)

    if not blocked.use_scan:
        if sample_products is None:
            sample_products = _sigma_products(
                _sample_factors(blocked.powers, blocked.indices, samples_t, inv_d)
            )
        obs_products = _sigma_products(_obs_factors(blocked.powers, blocked.indices, l_t, inv_d))
        theta = gate_parameters(blocked.powers, blocked.param_map)
        return _block_phase_contribution(theta, sample_products, obs_products)

    @jax.checkpoint
    def block_contribution(block_indices, block_powers, block_params):
        block_samples = _sigma_products(
            _sample_factors(block_powers, block_indices, samples_t, inv_d)
        )
        block_obs = _sigma_products(_obs_factors(block_powers, block_indices, l_t, inv_d))
        theta = gate_parameters(block_powers, block_params)
        return _block_phase_contribution(theta, block_samples, block_obs)

    def accumulate(total, block):
        return total + block_contribution(*block), None

    zero = jnp.zeros((l_t.shape[1], samples_t.shape[1]), dtype=_REAL_DTYPE)
    total, _ = jax.lax.scan(accumulate, zero, (blocked.indices, blocked.powers, blocked.param_map))
    return total


class _PrecomputedObsData(NamedTuple):
    """Bundled precomputed observable data from the factory."""

    l_vecs: jnp.ndarray
    l_f: jnp.ndarray
    m_f: jnp.ndarray
    l_t: jnp.ndarray
    obs_phase_matrix: jnp.ndarray


def _obs_phase_matrix(
    samples: jnp.ndarray, m_f: jnp.ndarray, l_f: jnp.ndarray, dims: ArrayLike
) -> jnp.ndarray:
    """Compute the observable phase matrix.

    :math:`J[i, j] = \\exp(i\\pi \\sum_k m_{ik} (2 z_{jk} - l_{ik}) / d_k)`.
    """
    s_f = samples.astype(jnp.float32)
    inv_d = (1.0 / jnp.asarray(dims, dtype=jnp.float32))[jnp.newaxis, :]  # (1, n_qudits)
    m_scaled = m_f * inv_d  # (n_obs, n_qudits)
    return jnp.exp(
        1j * jnp.pi * (2 * m_scaled @ s_f.T - jnp.sum(m_scaled * l_f, axis=1, keepdims=True))
    )


def _compute_initial_state_correction(
    samples: jnp.ndarray,
    l_f: jnp.ndarray,
    state_elems: ArrayLike,
    state_amps: ArrayLike,
    dims: ArrayLike,
) -> jnp.ndarray:
    """Compute the correction factor for a non-standard initial state."""
    s_f = samples.astype(jnp.float32)
    X_state = jnp.asarray(state_elems).astype(jnp.float32)  # (N, n)
    Psi = jnp.asarray(state_amps)  # (N,)
    inv_d = (1.0 / jnp.asarray(dims, dtype=jnp.float32))[jnp.newaxis, :]  # (1, n)

    # ω^{Z·X^T} where ω_j = exp(2πi/d_j) — shape (s, N)
    omega_ZX = jnp.exp(2j * jnp.pi * ((s_f * inv_d) @ X_state.T))

    # Ψ̃^(2) = ω^{Z·X^T} · Ψ — shape (s,)
    psi_tilde_2 = omega_ZX @ Psi

    # F = Ψ* · 1_{1×s} ⊙ ω^{-X·Z^T} — shape (N, s)
    F_mat = Psi.conj()[:, jnp.newaxis] * omega_ZX.conj().T

    # Ψ̃^(1) = ω^{L·X^T} · F — shape (l, s)
    omega_LX = jnp.exp(2j * jnp.pi * ((l_f * inv_d) @ X_state.T))  # (l, N)
    psi_tilde_1 = omega_LX @ F_mat

    # H = Ψ̃^(1) ⊙ (1_{l×1} · (Ψ̃^(2))^T) — shape (l, s)
    return psi_tilde_1 * psi_tilde_2[jnp.newaxis, :]


def _compute_mc_statistics(
    integrand: jnp.ndarray, n_samples: int
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute the Monte Carlo mean and covariance from the integrand.

    Returns ``(expvals, cov)`` where ``cov`` is the per-observable covariance
    matrix of the mean estimator, shape ``(n_obs, 2, 2)``.
    """
    expvals = jnp.mean(integrand, axis=1)

    re = jnp.real(integrand)
    im = jnp.imag(integrand)
    re_centered = re - jnp.mean(re, axis=1, keepdims=True)
    im_centered = im - jnp.mean(im, axis=1, keepdims=True)
    var_re = jnp.sum(re_centered**2, axis=1) / (n_samples - 1) / n_samples
    var_im = jnp.sum(im_centered**2, axis=1) / (n_samples - 1) / n_samples
    cov_re_im = jnp.sum(re_centered * im_centered, axis=1) / (n_samples - 1) / n_samples
    cov = jnp.stack(
        [
            jnp.stack([var_re, cov_re_im], axis=-1),
            jnp.stack([cov_re_im, var_im], axis=-1),
        ],
        axis=-2,
    )  # (n_obs, 2, 2)
    return expvals, cov


def build_qudit_expval_func(  # pylint: disable=too-many-statements
    config: QuditCircuitConfig,
) -> Callable:
    """Build an estimator for expectation values of a qudit IQP circuit.

    Returns a pure function that estimates the complex expectation value
    :math:`\\langle O(\\mathbf{l}, \\mathbf{m}) \\rangle` for each
    observable by averaging over randomly sampled dit-strings.

    The returned function captures precomputed data from ``config`` (sparse gate
    arrays, default samples, preprocessed observables) so that repeated
    evaluations with different parameters are fast. The circuit phase is
    accumulated over blocks of ``config.block_size`` gates, so the memory used
    by the estimator does not grow with the number of gates.

    Args:
        config (QuditCircuitConfig): Full circuit description including gate
            structure, observables, and sampling parameters. See
            :class:`QuditCircuitConfig` for details on how to construct one.

    Returns:
        Callable: A function with signature::

            expval_fn(
                gates_params,
                phase_fn_params=None,
                key=None,
                n_samples=None,
                observables=None,
                init_state_elems=None,
                init_state_amps=None,
            ) -> (expvals, cov)

        where ``expvals`` is a complex array of shape ``(n_obs,)`` containing
        the estimated moments, and ``cov`` has shape ``(n_obs, 2, 2)``
        providing the real/imaginary covariance matrix of the mean estimator
        for each observable.

        When ``config.phase_fn`` is set, the returned callable requires ``phase_fn_params`` to be
        passed as the second argument (the trainable parameters of the phase layer).

    Raises:
        ValueError: If no observables are provided either in ``config`` or at
            call time.

    **Example**

    >>> import jax
    >>> import jax.numpy as jnp
    >>> from pennylane.labs.tcdq import QuditCircuitConfig, build_qudit_expval_func
    >>> config = QuditCircuitConfig(
    ...     dims=3,
    ...     n_qudits=2,
    ...     gates={0: [[(0, 1)]], 1: [[(1, 1)]]},
    ...     n_samples=512,
    ...     key=jax.random.PRNGKey(0),
    ...     observables=(
    ...         jnp.array([[1, 0], [0, 1]], dtype=jnp.int32),
    ...         jnp.zeros((2, 2), dtype=jnp.int32),
    ...     ),
    ... )
    >>> expval_fn = build_qudit_expval_func(config)
    >>> params = jnp.array([0.2, -0.1])
    >>> expvals, cov = expval_fn(params)
    >>> expvals.shape, cov.shape
    ((2,), (2, 2, 2))

    .. seealso::

        `Spectral Born machines: classically trainable quantum generative models for discrete data <https://arxiv.org/pdf/2607.06675>`_.
    """
    gate_indices, gate_powers, param_map = _parse_qudit_generator_dict(
        config.gates, config.n_qudits
    )

    n = config.n_qudits
    dims = _dims_to_numpy(config.dims, n)
    default_samples = _compute_qudit_samples(config.key, config.n_samples, n, dims)
    default_samples_t = _pad_sentinel_row(default_samples)

    # Inverse local dimensions with a trailing sentinel slot for the padding index ``n``.
    inv_d = jnp.concatenate(
        [1.0 / jnp.asarray(dims, dtype=_REAL_DTYPE), jnp.zeros((1,), dtype=_REAL_DTYPE)]
    )
    blocked = _block_gate_arrays(
        gate_indices, gate_powers, param_map, config.block_size, sentinel_index=n
    )
    # For circuits that fit in one block, the sample-side factor products for the default
    # samples are precomputed once (2**max_weight arrays of shape (n_gates, n_samples)).
    default_sample_products = None
    if not blocked.use_scan:
        default_sample_products = _sigma_products(
            _sample_factors(blocked.powers, blocked.indices, default_samples_t, inv_d)
        )

    vmapped_phase_func = None
    if config.phase_fn is not None:
        dims_j = jnp.asarray(dims)

        def compute_phase_diff(p_params, sample, l_vec):
            return config.phase_fn(p_params, sample) - config.phase_fn(
                p_params, (sample - l_vec) % dims_j
            )

        vmapped_phase_func = jax.vmap(
            jax.vmap(compute_phase_diff, in_axes=(None, 0, None)),
            in_axes=(None, None, 0),
        )

    if config.observables is not None:
        l_vecs = jnp.array(config.observables[0], dtype=jnp.int32)
        m_vecs = jnp.array(config.observables[1], dtype=jnp.int32)
        l_f = l_vecs.astype(jnp.float32)
        m_f = m_vecs.astype(jnp.float32)
        defaults = _PrecomputedObsData(
            l_vecs=l_vecs,
            l_f=l_f,
            m_f=m_f,
            l_t=_pad_sentinel_row(l_vecs),
            obs_phase_matrix=_obs_phase_matrix(default_samples, m_f, l_f, dims),
        )
    else:
        defaults = None

    def qudit_expval_batched(
        gates_params: ArrayLike,
        phase_fn_params: ArrayLike | None = None,
        key: ArrayLike | None = None,
        n_samples: int | None = None,
        observables: tuple[ArrayLike, ArrayLike] | None = None,
        init_state_elems: ArrayLike | None = None,
        init_state_amps: ArrayLike | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:  # pylint: disable=too-many-arguments
        """Compute batched expectation values for the configured circuit.

        Args:
            gates_params (ArrayLike): 1-D array of gate parameters.
            phase_fn_params (ArrayLike | None, optional): Trainable parameters for the
                custom phase function. Defaults to ``None``.
            key (ArrayLike | None, optional): Runtime override for the JAX PRNG key
                used for sampling. Defaults to None.
            n_samples (int | None, optional): Runtime override for the number of
                samples. Defaults to None.
            observables (tuple[ArrayLike, ArrayLike] | None, optional): Runtime override
                for the displacement-operator observables ``(l_vecs, m_vecs)``.
                Defaults to None.
            init_state_elems (ArrayLike | None, optional): Runtime override for the
                support elements of the initial state. Array of shape ``(N, n_qudits)``
                with integer entries in ``{0, ..., d-1}``. Defaults to None.
            init_state_amps (ArrayLike | None, optional): Runtime override for the
                complex amplitudes of the initial state. Array of shape ``(N,)``.
                Defaults to None.

        Returns:
            tuple[jnp.ndarray, jnp.ndarray]: Returns ``(expvals, cov)`` where
            ``expvals`` are the estimated complex expectation values, shape
            ``(n_obs,)``, and ``cov`` stores the real-imaginary covariance matrices
            of the mean estimator, shape ``(n_obs, 2, 2)``.
        """
        if observables is not None:
            l_vecs = jnp.array(observables[0], dtype=jnp.int32)
            l_f = l_vecs.astype(jnp.float32)
            m_f = jnp.array(observables[1], dtype=jnp.int32).astype(jnp.float32)
            l_t = _pad_sentinel_row(l_vecs)
        elif defaults is not None:
            l_vecs, l_f, m_f, l_t = defaults.l_vecs, defaults.l_f, defaults.m_f, defaults.l_t
        else:
            raise ValueError(
                "No observables specified. Provide them in QuditCircuitConfig "
                "or pass at call time via the observables argument."
            )

        if key is not None or n_samples is not None:
            _key = key if key is not None else config.key
            _n = n_samples if n_samples is not None else config.n_samples
            samples = _compute_qudit_samples(_key, _n, n, dims)
            samples_t = _pad_sentinel_row(samples)
            sample_products = None
        else:
            _n = config.n_samples
            samples = default_samples
            samples_t = default_samples_t
            sample_products = default_sample_products

        use_cached = (
            key is None and n_samples is None and observables is None and defaults is not None
        )
        if use_cached:
            obs_pm = defaults.obs_phase_matrix
        else:
            obs_pm = _obs_phase_matrix(samples, m_f, l_f, dims)

        accumulated_phase_diffs = _phase_differences(
            gates_params, samples_t, l_t, inv_d, blocked, sample_products=sample_products
        )
        if vmapped_phase_func is not None:
            accumulated_phase_diffs = accumulated_phase_diffs + vmapped_phase_func(
                phase_fn_params, samples, l_vecs
            )

        state_elems = config.init_state_elems if init_state_elems is None else init_state_elems
        state_amps = config.init_state_amps if init_state_amps is None else init_state_amps

        integrand = obs_pm * jnp.exp(1j * accumulated_phase_diffs)
        if state_elems is not None and state_amps is not None:
            H = _compute_initial_state_correction(samples, l_f, state_elems, state_amps, dims)
            integrand = integrand * H

        return _compute_mc_statistics(integrand, _n)

    return qudit_expval_batched
