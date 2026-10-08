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
        gates (dict[int, list[dict[int, int]]]): Circuit structure mapping each
            trainable-parameter index to a list of gates. Each gate is a sparse
            generator given as a ``dict`` mapping a qudit index to the power of
            :math:`Z` applied on that qudit; qudits that are not listed are
            acted on trivially. Powers are reduced modulo the local dimension
            :math:`d_j`, and entries that reduce to zero are dropped. For
            example, with ``dims=3`` and ``n_qudits=2``,
            ``{0: [{0: 1}], 1: [{1: 1}], 2: [{0: 1, 1: 1}]}`` defines three
            gates: :math:`Z^1` on qudit 0, :math:`Z^1` on qudit 1, and
            :math:`Z^1 \otimes Z^1` on both. Only the active qudits of each
            gate are stored, so the cost of describing a :math:`k`-local
            circuit scales with :math:`k` rather than with ``n_qudits``.
        n_samples (int): Number of random dit-strings drawn for the
            estimation.
        key (ArrayLike): JAX PRNG key for random dit-string generation.
        observables (tuple[ArrayLike, ArrayLike] | None): A pair
            ``(l_vecs, m_vecs)`` specifying the Heisenberg–Weyl displacement
            operators :math:`O(\mathbf{l}, \mathbf{m})` to measure.
            Each is an integer array of shape ``(n_obs, n_qudits)`` with entries
            in :math:`\{0, \ldots, d-1\}`. If ``None``, observables must be
            supplied at call time (e.g., when used inside
            :func:`~pennylane.labs.tcdq.build_qudit_mmd_loss`).
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

    **Example**

    >>> import jax
    >>> import jax.numpy as jnp
    >>> from pennylane.labs.tcdq import QuditCircuitConfig
    >>> config = QuditCircuitConfig(
    ...     dims=3,
    ...     n_qudits=4,
    ...     gates={0: [{0: 1}], 1: [{1: 1}], 2: [{0: 1, 1: 1}]},
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
    #: Circuit structure mapping parameter indices to sparse gates ``{qudit: power}``.
    gates: dict[int, list[dict[int, int]]] = None
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


class SparseGateGroup(NamedTuple):
    """Gates of equal weight :math:`\\omega` stored in sparse ``(support, power)`` form.

    Only the active qudits of each gate are kept, so the storage cost is
    ``O(n_gates * omega)`` instead of ``O(n_gates * n_qudits)``.

    Args:
        omega (int): Number of active qudits shared by every gate in the group.
        supports (np.ndarray): Active qudit indices, shape ``(n_gates, omega)``,
            sorted in increasing order along the last axis.
        powers (np.ndarray): Power of :math:`Z` at each support position, shape
            ``(n_gates, omega)``, with entries in ``{1, ..., d_j - 1}``.
        param_indices (jnp.ndarray): Index into ``gates_params`` for each gate,
            shape ``(n_gates,)``.
    """

    #: Number of active qudits per gate in this group.
    omega: int
    #: Active qudit indices, shape ``(n_gates, omega)``.
    supports: np.ndarray
    #: Power of :math:`Z` at each support position, shape ``(n_gates, omega)``.
    powers: np.ndarray
    #: Parameter index of each gate, shape ``(n_gates,)``.
    param_indices: jnp.ndarray


def _normalize_sparse_gate(
    gate: dict[int, int], n_qudits: int, dims: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Validate one sparse gate and return its sorted ``(support, powers)`` arrays.

    Qudit indices may be negative (Python-style, counted from the end). Powers
    are reduced modulo the local dimension of their qudit and zero powers are
    dropped, so the returned support contains only genuinely active qudits.

    Raises:
        TypeError: If ``gate`` is not a ``dict`` (e.g. a dense length-``n_qudits`` vector).
        IndexError: If a qudit index is outside ``[-n_qudits, n_qudits)``.
    """
    if not isinstance(gate, dict):
        raise TypeError(
            "Each qudit gate must be a dict mapping qudit index to Z power, e.g. {0: 1, 2: 2}; "
            f"got {type(gate).__name__}. Dense length-n_qudits generator vectors are not accepted."
        )

    support: list[int] = []
    powers: list[int] = []
    for qudit, power in gate.items():
        q = int(qudit)
        if q < -n_qudits or q >= n_qudits:
            raise IndexError(f"Qudit index {q} out of range for a {n_qudits}-qudit circuit.")
        q = q % n_qudits
        p = int(power) % int(dims[q])
        if p == 0:
            continue
        support.append(q)
        powers.append(p)

    order = np.argsort(support, kind="stable")
    return np.asarray(support, dtype=int)[order], np.asarray(powers, dtype=int)[order]


def _parse_qudit_gate_dict(
    circuit_def: dict[int, list[dict[int, int]]], n_qudits: int, dims: ArrayLike
) -> list[SparseGateGroup]:
    """Convert a sparse qudit gate dictionary into weight-grouped support/power arrays.

    This is the qudit analogue of
    :func:`~pennylane.labs.tcdq.expval_functions._parse_generator_dict`. It never
    materialises a dense ``(n_gates, n_qudits)`` generator matrix: each gate is
    stored through its active qudits only, and gates are bucketed by weight so
    that the downstream :math:`2^\\omega` angle-addition expansion can be
    vectorised within each bucket.

    Gates whose support is empty after reducing powers modulo ``dims`` act as
    the identity and are dropped. Their parameter index is still a valid slot
    in ``gates_params``; it simply receives zero gradient.

    Args:
        circuit_def (dict[int, list[dict[int, int]]]): Maps parameter indices to lists
            of sparse gates ``{qudit_index: z_power}``.
        n_qudits (int): Number of qudits.
        dims (ArrayLike): Per-qudit dimensions, shape ``(n_qudits,)``.

    Returns:
        list[SparseGateGroup]: One group per distinct non-zero weight, sorted by weight.

    Raises:
        TypeError: If a gate is not a ``dict``.
        IndexError: If a qudit index is out of range.
    """
    dims = np.asarray(dims, dtype=int)
    buckets: dict[int, tuple[list[np.ndarray], list[np.ndarray], list[int]]] = {}

    for param_idx in sorted(circuit_def.keys()):
        for gate in circuit_def[param_idx]:
            support, powers = _normalize_sparse_gate(gate, n_qudits, dims)
            omega = support.size
            if omega == 0:
                continue
            sup_list, pow_list, pidx_list = buckets.setdefault(omega, ([], [], []))
            sup_list.append(support)
            pow_list.append(powers)
            pidx_list.append(int(param_idx))

    return [
        SparseGateGroup(
            omega=omega,
            supports=np.stack(sup_list),
            powers=np.stack(pow_list),
            param_indices=jnp.array(pidx_list, dtype=int),
        )
        for omega, (sup_list, pow_list, pidx_list) in sorted(buckets.items())
    ]


def _gates_to_dense_generators(
    circuit_def: dict[int, list[dict[int, int]]], n_qudits: int, dims: ArrayLike
) -> tuple[np.ndarray, np.ndarray]:
    """Expand sparse gates into a dense generator matrix (for brute-force references).

    This is intentionally *not* used by the estimator; it exists so that
    exhaustive reference implementations and tests can consume the same
    ``gates`` dictionary. Gates are flattened in the same order as
    :func:`_parse_qudit_gate_dict` visits them (sorted parameter index, then
    list order) and identity gates are kept so that the returned
    ``param_map`` has one entry per gate in ``circuit_def``.

    Returns:
        tuple[np.ndarray, np.ndarray]: ``(generators, param_map)`` with shapes
        ``(n_gates, n_qudits)`` and ``(n_gates,)``.
    """
    dims = np.asarray(dims, dtype=int)
    rows: list[np.ndarray] = []
    param_map: list[int] = []
    for param_idx in sorted(circuit_def.keys()):
        for gate in circuit_def[param_idx]:
            support, powers = _normalize_sparse_gate(gate, n_qudits, dims)
            row = np.zeros((n_qudits,), dtype=int)
            row[support] = powers
            rows.append(row)
            param_map.append(int(param_idx))

    generators = np.stack(rows) if rows else np.zeros((0, n_qudits), dtype=int)
    return generators, np.asarray(param_map, dtype=int)


def _compute_qudit_samples(
    key: ArrayLike, num_samples: int, n_qudits: int, dims: ArrayLike
) -> jnp.ndarray:
    """Generates uniformly random dit-strings from the product Z_{d_1} x ... x Z_{d_n}."""

    maxval = jnp.asarray(dims, dtype=jnp.int32)[jnp.newaxis, :]  # (1, n_qudits)
    return jax.random.randint(key, shape=(num_samples, n_qudits), minval=0, maxval=maxval)


class WeightGroupData(NamedTuple):
    """Precomputed factor matrices for gates sharing the same weight (number of active qudits).

    Gates are grouped by weight :math:`\\omega` (number of non-zero entries in
    the generator vector) so that the :math:`2^\\omega`-term angle-addition
    expansion can be vectorised over gates within each group.

    Args:
        param_indices (jnp.ndarray): Maps each gate in this group to its parameter
            index in the global ``gates_params`` array, shape ``(n_gates,)``.
        samples_matrices (list[jnp.ndarray]): :math:`2^\\omega` matrices of shape
            ``(n_gates, n_samples)`` giving the sample-side factor for each
            angle-addition term.
        obs_matrices (list[jnp.ndarray]): :math:`2^\\omega` matrices of shape
            ``(n_gates, n_obs)`` giving the observable-side factor for each
            angle-addition term.
    """

    #: Maps each gate to its parameter index, shape ``(n_gates,)``.
    param_indices: jnp.ndarray
    #: Sample-side factor matrices for each angle-addition term.
    samples_matrices: list[jnp.ndarray]
    #: Observable-side factor matrices for each angle-addition term.
    obs_matrices: list[jnp.ndarray]


def _gather_support_values(
    vectors: ArrayLike, supports: np.ndarray, target_dim: int, n_gates: int, omega: int
) -> jnp.ndarray:
    """Extract values at the active qudit positions for every gate, for every vector.

    Each gate acts on ``omega`` qudits (its *support*).  Given a batch of
    full-length vectors (e.g. Monte Carlo samples or observable ``l``-vectors),
    this function selects only the entries at each gate's support positions and
    arranges them into shape ``(n_gates, omega, target_dim)`` so downstream
    trigonometric computations can be vectorised over gates and positions.

    Args:
        vectors (ArrayLike): Input array of shape ``(target_dim, n_qudits)`` —
            either the Monte Carlo samples (``target_dim = n_samples``) or the
            observable ``l``-vectors (``target_dim = n_obs``).
        supports (np.ndarray): Active qudit indices for each gate, shape
            ``(n_gates, omega)``.
        target_dim (int): Number of vectors (rows in ``vectors``).
        n_gates (int): Number of gates in this weight group.
        omega (int): Number of active qudits per gate.

    Returns:
        jnp.ndarray: Values at support positions, shape ``(n_gates, omega, target_dim)``.
    """
    flat_supports = supports.reshape(-1)
    return (
        jnp.array(vectors)[:, flat_supports].reshape(target_dim, n_gates, omega).transpose(1, 2, 0)
    )


def _compute_trigonometric_building_blocks(
    gate_vals: np.ndarray,
    z_at_support: jnp.ndarray,
    l_at_support: jnp.ndarray,
    d_at_support: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute (state_cos, state_sin, obs_cos, obs_sin) trig factors over the gate support."""
    g = jnp.array(gate_vals, dtype=jnp.float32)[:, :, jnp.newaxis]
    d_s = jnp.asarray(d_at_support, dtype=jnp.float32)[:, :, jnp.newaxis]
    angle_z = 2 * jnp.pi * g * z_at_support.astype(jnp.float32) / d_s + jnp.pi / 4
    angle_l = 2 * jnp.pi * g * l_at_support.astype(jnp.float32) / d_s
    return (
        jnp.sqrt(2.0) * jnp.cos(angle_z),
        jnp.sqrt(2.0) * jnp.sin(angle_z),
        jnp.cos(angle_l),
        jnp.sin(angle_l),
    )


def _expand_angle_addition(
    state_cos: jnp.ndarray,
    state_sin: jnp.ndarray,
    obs_cos: jnp.ndarray,
    obs_sin: jnp.ndarray,
) -> tuple[list[jnp.ndarray], list[jnp.ndarray]]:
    """Enumerate all :math:`2^\\omega` angle-addition terms to build the factor matrices.

    Each term corresponds to a binary choice (cos or sin) at each active
    qudit position, producing paired sample-side and observable-side factors.
    """
    n_gates, omega, num_samples = state_cos.shape
    n_obs = obs_cos.shape[2]
    state_factors = [state_cos, state_sin]
    obs_factors = [obs_cos, obs_sin]
    samples_list: list[jnp.ndarray] = []
    obs_list: list[jnp.ndarray] = []
    for sigma in itertools.product([0, 1], repeat=omega):
        B = jnp.ones((n_gates, num_samples), dtype=jnp.float32)
        C = jnp.ones((n_gates, n_obs), dtype=jnp.float32)
        for k, choice in enumerate(sigma):
            B *= state_factors[choice][:, k, :]
            C *= obs_factors[choice][:, k, :]
        samples_list.append(B)
        obs_list.append(C)
    return samples_list, obs_list


def _build_weight_group(
    group: SparseGateGroup,
    samples: jnp.ndarray,
    l_vecs: jnp.ndarray,
    dims: np.ndarray,
) -> WeightGroupData:
    """Precompute the factor matrices for a group of gates with the same weight.

    The supports and powers come straight from the sparse gate description, so
    no dense generator matrix is scanned here.
    """
    n_gates, omega = group.supports.shape
    num_samples = samples.shape[0]
    n_obs = l_vecs.shape[0]
    d_at_support = np.asarray(dims)[group.supports]  # (n_gates, omega)

    z_at_support = _gather_support_values(samples, group.supports, num_samples, n_gates, omega)
    l_at_support = _gather_support_values(l_vecs, group.supports, n_obs, n_gates, omega)

    state_cos, state_sin, obs_cos, obs_sin = _compute_trigonometric_building_blocks(
        group.powers, z_at_support, l_at_support, d_at_support
    )
    samples_matrices, obs_matrices = _expand_angle_addition(state_cos, state_sin, obs_cos, obs_sin)
    return WeightGroupData(
        param_indices=group.param_indices,
        samples_matrices=samples_matrices,
        obs_matrices=obs_matrices,
    )


class _PrecomputedObsData(NamedTuple):
    """Bundled precomputed observable data from the factory."""

    l_vecs: jnp.ndarray
    n_obs: int
    l_f: jnp.ndarray
    m_f: jnp.ndarray
    weight_data: list
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


def _build_all_weight_groups(
    gate_groups: list[SparseGateGroup],
    samples: jnp.ndarray,
    l_vecs: jnp.ndarray,
    dims: np.ndarray,
) -> list[WeightGroupData]:
    """Build :class:`WeightGroupData` for each sparse gate group."""
    return [_build_weight_group(group, samples, l_vecs, dims) for group in gate_groups]


# pylint: disable=too-many-arguments
def _accumulate_phase_diffs(
    gates_params: ArrayLike,
    weight_data: list[WeightGroupData],
    n_obs: int,
    n_samples: int,
    vmapped_phase_func: Callable | None,
    phase_fn_params: ArrayLike | None,
    samples: ArrayLike,
    l_vecs: ArrayLike,
) -> jnp.ndarray:
    """Assemble the accumulated phase-difference matrix from all weight groups."""
    accumulated = jnp.zeros((n_obs, n_samples))
    for group in weight_data:
        theta_w = jnp.asarray(gates_params)[group.param_indices]
        accumulated = accumulated + (theta_w @ group.samples_matrices[0])[jnp.newaxis, :]
        for B_sigma, C_sigma in zip(group.samples_matrices, group.obs_matrices):
            accumulated = accumulated - (C_sigma.T * theta_w) @ B_sigma

    if vmapped_phase_func is not None:
        accumulated += vmapped_phase_func(phase_fn_params, samples, l_vecs)

    return accumulated


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

    The returned function captures precomputed data from ``config`` (generator
    matrices, default samples, preprocessed observables) so that repeated
    evaluations with different parameters are fast.

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
    ...     gates={0: [{0: 1}], 1: [{1: 1}]},
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
    n = config.n_qudits
    dims = _dims_to_numpy(config.dims, n)
    gate_groups = _parse_qudit_gate_dict(config.gates, n, dims)
    default_samples = _compute_qudit_samples(config.key, config.n_samples, n, dims)

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
        n_obs = l_vecs.shape[0]
        defaults = _PrecomputedObsData(
            l_vecs=l_vecs,
            n_obs=n_obs,
            l_f=l_f,
            m_f=m_f,
            weight_data=_build_all_weight_groups(gate_groups, default_samples, l_vecs, dims),
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
            n_obs = l_vecs.shape[0]
            l_f = l_vecs.astype(jnp.float32)
            m_f = jnp.array(observables[1], dtype=jnp.int32).astype(jnp.float32)
        elif defaults is not None:
            l_vecs, n_obs, l_f, m_f = defaults.l_vecs, defaults.n_obs, defaults.l_f, defaults.m_f
        else:
            raise ValueError(
                "No observables specified. Provide them in QuditCircuitConfig "
                "or pass at call time via the observables argument."
            )

        if key is not None or n_samples is not None:
            _key = key if key is not None else config.key
            _n = n_samples if n_samples is not None else config.n_samples
            samples = _compute_qudit_samples(_key, _n, n, dims)
        else:
            _n = config.n_samples
            samples = default_samples

        use_cached = (
            key is None and n_samples is None and observables is None and defaults is not None
        )
        if use_cached:
            obs_pm = defaults.obs_phase_matrix
            w_data = defaults.weight_data
        else:
            obs_pm = _obs_phase_matrix(samples, m_f, l_f, dims)
            w_data = _build_all_weight_groups(gate_groups, samples, l_vecs, dims)

        accumulated_phase_diffs = _accumulate_phase_diffs(
            gates_params, w_data, n_obs, _n, vmapped_phase_func, phase_fn_params, samples, l_vecs
        )

        state_elems = config.init_state_elems if init_state_elems is None else init_state_elems
        state_amps = config.init_state_amps if init_state_amps is None else init_state_amps

        integrand = obs_pm * jnp.exp(1j * accumulated_phase_diffs)
        if state_elems is not None and state_amps is not None:
            H = _compute_initial_state_correction(samples, l_f, state_elems, state_amps, dims)
            integrand = integrand * H

        return _compute_mc_statistics(integrand, _n)

    return qudit_expval_batched
