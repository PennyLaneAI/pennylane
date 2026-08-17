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
        d (int | Sequence[int]): Local qudit dimension(s). Either a single
            ``int`` (e.g., 2 for qubits, 3 for qutrits), which is broadcast to
            every qudit, or a sequence of length ``n_qudits`` giving a distinct
            dimension :math:`d_j` per qudit. All
            per-qudit index sets are then :math:`\{0, \ldots, d_j - 1\}`.
        n_qudits (int): Number of qudits in the circuit.
        gates (dict[int, list[list[int]]]): Circuit structure mapping each
            trainable-parameter index to a list of generator vectors. Each
            generator vector has length ``n_qudits`` with integer entries in
            :math:`\{0, \ldots, d_j-1\}` that specify the power of :math:`Z` on
            each qudit. For example, with ``d=3`` and ``n_qudits=2``,
            ``{0: [[1, 0]], 1: [[0, 1]], 2: [[1, 1]]}`` defines three gates:
            :math:`Z^1` on qudit 0, :math:`Z^1` on qudit 1, and
            :math:`Z^1 \otimes Z^1` on both.
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
        control_variate (bool): If ``True``, reduces the Monte Carlo variance using an
            order-2 Taylor-expansion control variate: the integrand with
            :math:`e^{i \Delta_\mathbf{l}}` expanded to second order in the phase
            difference about :math:`\mathbf{\theta} = 0`, keeping the observable phase
            and input-state factors exact. The control mean is evaluated in closed form
            via character orthogonality on :math:`\mathbb{Z}_{d_1} \times \cdots \times
            \mathbb{Z}_{d_n}`, so the resulting estimator stays unbiased. Cost is
            :math:`O(T^2)` per observable with :math:`T = \sum_\mathbf{g}
            2^{\vert \text{supp}(\mathbf{g}) \vert}`, so it is efficient for low-weight
            gate sets. Not compatible with ``phase_fn``. Defaults to ``False``.

    **Example**

    >>> import jax
    >>> import jax.numpy as jnp
    >>> from pennylane.labs.tcdq import QuditCircuitConfig
    >>> config = QuditCircuitConfig(
    ...     d=3,
    ...     n_qudits=4,
    ...     gates={0: [[1, 0, 0, 0]], 1: [[0, 1, 0, 0]], 2: [[1, 1, 0, 0]]},
    ...     observables=(
    ...         jnp.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=jnp.int32),
    ...         jnp.zeros((2, 4), dtype=jnp.int32),
    ...     ),
    ...     n_samples=5000,
    ...     key=jax.random.PRNGKey(42),
    ... )
    """

    #: Local qudit dimension: an int (uniform) or per-qudit sequence.
    d: int | Sequence[int] = None
    #: Number of qudits in the circuit.
    n_qudits: int = None
    #: Circuit structure mapping parameter indices to generator vectors.
    gates: dict[int, list[list[int]]] = None
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
    #: If ``True``, use the order-2 Taylor-expansion control variate.
    control_variate: bool = False


def _dims_to_numpy(d: int | Sequence[int], n_qudits: int) -> np.ndarray:
    """Normalize the ``d`` field to an integer array of per-qudit dimensions.

    Accepts either a scalar ``int`` (broadcast to all qudits, the uniform case)
    or a sequence of length ``n_qudits`` (mixed-dimension case), and always
    returns a NumPy integer array of shape ``(n_qudits,)``.

    Raises:
        ValueError: If ``d`` is a sequence whose length is not ``n_qudits``.
    """
    if np.ndim(d) == 0:
        return np.full((n_qudits,), int(d), dtype=int)

    dims = np.asarray(d, dtype=int)
    if dims.shape != (n_qudits,):
        raise ValueError(
            f"d given as a sequence must have length n_qudits={n_qudits}, "
            f"got shape {dims.shape}."
        )

    return dims


def _parse_qudit_generator_dict(circuit_def: dict[int, list[list[int]]], n_qudits: int):
    """Convert a qudit gate dictionary into a generator matrix and parameter map.

    Unlike the qubit version, generator vectors are provided explicitly (not as wire
    indices), so each inner list must already have length ``n_qudits`` with integer entries
    in ``{0, ..., d-1}``.

    Args:
        circuit_def (dict[int, list[list[int]]]): Maps parameter indices to lists of
            generator vectors of length ``n_qudits``.
        n_qudits (int): Number of qudits.

    Returns:
        tuple[jnp.ndarray, jnp.ndarray]: Tuple containing:
            - Integer generator matrix of shape ``(n_gates, n_qudits)``.
            - Integer array mapping each gate to its parameter index.

    Raises:
        ValueError: If any generator vector has length != ``n_qudits``.
    """
    flat_gates = []
    param_indices = []

    for param_idx in sorted(circuit_def.keys()):
        for gate in circuit_def[param_idx]:
            if len(gate) != n_qudits:
                raise ValueError(f"Generator has length {len(gate)}, expected {n_qudits}.")
            flat_gates.append(gate)
            param_indices.append(param_idx)

    n_gates = len(flat_gates)
    if n_gates == 0:
        generators = np.zeros((0, n_qudits), dtype=int)
    else:
        generators = np.array(flat_gates, dtype=int)

    param_map = jnp.array(param_indices, dtype=int)
    return jnp.array(generators), param_map


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
    generators_w: np.ndarray,
    param_indices: jnp.ndarray,
    samples: jnp.ndarray,
    l_vecs: jnp.ndarray,
    dims: np.ndarray,
) -> WeightGroupData:
    """Precompute the factor matrices for a group of gates with the same weight."""
    n_gates = len(generators_w)
    num_samples = samples.shape[0]
    n_obs = l_vecs.shape[0]
    omega = int(np.count_nonzero(generators_w[0]))
    supports = np.array([np.where(g != 0)[0] for g in generators_w])  # (n_gates, omega)
    gate_vals = np.array([g[s] for g, s in zip(generators_w, supports)])  # (n_gates, omega)
    d_at_support = np.asarray(dims)[supports]  # (n_gates, omega)

    z_at_support = _gather_support_values(samples, supports, num_samples, n_gates, omega)
    l_at_support = _gather_support_values(l_vecs, supports, n_obs, n_gates, omega)

    state_cos, state_sin, obs_cos, obs_sin = _compute_trigonometric_building_blocks(
        gate_vals, z_at_support, l_at_support, d_at_support
    )
    samples_matrices, obs_matrices = _expand_angle_addition(state_cos, state_sin, obs_cos, obs_sin)
    return WeightGroupData(
        param_indices=param_indices, samples_matrices=samples_matrices, obs_matrices=obs_matrices
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


# pylint: disable=too-many-arguments
def _build_all_weight_groups(
    gen_np: np.ndarray,
    pm_np: np.ndarray,
    gate_weights: np.ndarray,
    samples: jnp.ndarray,
    l_vecs: jnp.ndarray,
    dims: np.ndarray,
) -> list[WeightGroupData]:
    """Build :class:`WeightGroupData` for each non-zero gate weight."""
    weight_data: list[WeightGroupData] = []
    for omega in sorted(set(gate_weights)):
        if omega == 0:
            continue
        gate_indices = np.where(gate_weights == omega)[0]
        weight_data.append(
            _build_weight_group(
                generators_w=gen_np[gate_indices],
                param_indices=jnp.array(pm_np[gate_indices]),
                samples=samples,
                l_vecs=l_vecs,
                dims=dims,
            )
        )
    return weight_data


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


def _real_dtype():
    """Widest available real dtype, respecting JAX's ``jax_enable_x64`` setting."""
    return jnp.float64 if jax.config.read("jax_enable_x64") else jnp.float32


def _complex_dtype():
    """Widest available complex dtype, respecting JAX's ``jax_enable_x64`` setting."""
    return jnp.complex128 if jax.config.read("jax_enable_x64") else jnp.complex64


class CharacterExpansionData(NamedTuple):
    """Static character-expansion data for the order-2 Taylor control variate.

    The phase difference admits the exact finite character expansion

    .. math::

        \\Delta_\\mathbf{l}(\\mathbf{z}) = \\sum_{t=1}^{T} A_{\\mathbf{l},t}\\,
        \\chi_{\\mathbf{f}_t}(\\mathbf{z}), \\qquad
        A_{\\mathbf{l},t} = \\theta_t c_t \\left( 1 - \\omega^{-\\mathbf{f}_t \\cdot \\mathbf{l}} \\right),

    where :math:`t` indexes ``(gate, sign-pattern)`` pairs obtained by expanding each
    :math:`\\sqrt{2}\\cos(\\cdot)` factor of :math:`\\mathcal{Q}_\\mathbf{g}` into two
    characters. Weight-zero gates are dropped since they contribute
    :math:`\\mathcal{Q}_\\mathbf{g} \\equiv 1` and cancel in the phase difference.

    Only the :math:`\\mathbf{\\theta}`- and observable-independent pieces are stored
    here; they depend solely on the gate set and are therefore computed once at
    build time.

    Args:
        freqs (jnp.ndarray): Character frequencies :math:`\\mathbf{f}_t` reduced
            modulo ``dims``, shape ``(T, n_qudits)``.
        coeffs (jnp.ndarray): Sign-pattern coefficients :math:`c_t`, shape ``(T,)``.
        param_indices (jnp.ndarray): Maps each term to its parameter index in
            ``gates_params``, shape ``(T,)``.
        pair_match (jnp.ndarray): Boolean table selecting, for each ordered pair
            ``(t, u)``, the reduced sum frequency bucket used by the order-2 term.
            Shape ``(T, T, n_freq_unique)``.
        unique_freqs (jnp.ndarray): The distinct frequencies appearing either as
            :math:`\\mathbf{f}_t` or as :math:`\\mathbf{f}_t + \\mathbf{f}_u`,
            shape ``(n_freq_unique, n_qudits)``.
        single_match (jnp.ndarray): Boolean table mapping each term ``t`` to its
            bucket in ``unique_freqs``, shape ``(T, n_freq_unique)``.
    """

    #: Character frequencies, shape ``(T, n_qudits)``.
    freqs: jnp.ndarray
    #: Sign-pattern coefficients, shape ``(T,)``.
    coeffs: jnp.ndarray
    #: Parameter index for each term, shape ``(T,)``.
    param_indices: jnp.ndarray
    #: Pair-sum frequency bucket table, shape ``(T, T, n_freq_unique)``.
    pair_match: jnp.ndarray
    #: Distinct frequencies, shape ``(n_freq_unique, n_qudits)``.
    unique_freqs: jnp.ndarray
    #: Single-term frequency bucket table, shape ``(T, n_freq_unique)``.
    single_match: jnp.ndarray


def _build_character_expansion(
    gen_np: np.ndarray, pm_np: np.ndarray, dims: np.ndarray
) -> CharacterExpansionData:
    """Enumerate the character expansion of the phase difference.

    Each gate of weight :math:`\\omega` contributes :math:`2^\\omega` terms, one per
    sign pattern :math:`\\mathbf{\\sigma} \\in \\{\\pm 1\\}^{\\text{supp}(\\mathbf{g})}`,
    with frequency :math:`(\\mathbf{f}_{\\mathbf{g},\\mathbf{\\sigma}})_k = \\sigma_k g_k
    \\bmod d_k` and coefficient :math:`c_\\mathbf{\\sigma} = \\prod_k \\chi` or
    :math:`\\bar{\\chi}` according to :math:`\\sigma_k`, where
    :math:`\\chi = (1 + i)/2`.

    Frequencies are reduced modulo ``dims`` on construction, and pair sums are
    reduced again before matching: the delta of the orthogonality lemma holds
    modulo ``d``, and order-2 sum frequencies routinely exceed :math:`d_k`.

    Args:
        gen_np (np.ndarray): Generator matrix, shape ``(n_gates, n_qudits)``.
        pm_np (np.ndarray): Parameter index for each gate, shape ``(n_gates,)``.
        dims (np.ndarray): Per-qudit dimensions, shape ``(n_qudits,)``.

    Returns:
        CharacterExpansionData: Static expansion data.
    """
    chi = 0.5 + 0.5j
    n_qudits = gen_np.shape[1]

    freqs: list[np.ndarray] = []
    coeffs: list[complex] = []
    param_indices: list[int] = []

    for gate, p_idx in zip(gen_np, pm_np):
        support = np.nonzero(gate)[0]
        if len(support) == 0:
            # Weight-zero gates have Q_g = 1 and cancel in the phase difference.
            continue
        for sigma in itertools.product([1, -1], repeat=len(support)):
            f = np.zeros(n_qudits, dtype=int)
            c = 1.0 + 0.0j
            for k, s in zip(support, sigma):
                f[k] = (s * gate[k]) % dims[k]
                c *= chi if s == 1 else np.conj(chi)
            freqs.append(f)
            coeffs.append(c)
            param_indices.append(int(p_idx))

    if len(freqs) == 0:
        empty_f = np.zeros((0, n_qudits), dtype=int)
        return CharacterExpansionData(
            freqs=jnp.array(empty_f),
            coeffs=jnp.zeros((0,), dtype=_complex_dtype()),
            param_indices=jnp.zeros((0,), dtype=int),
            pair_match=jnp.zeros((0, 0, 0), dtype=bool),
            unique_freqs=jnp.array(empty_f),
            single_match=jnp.zeros((0, 0), dtype=bool),
        )

    freqs_np = np.array(freqs, dtype=int)
    pair_sums = (freqs_np[:, None, :] + freqs_np[None, :, :]) % dims  # (T, T, n_qudits)

    # Buckets: every frequency that can be selected by the moment functional.
    stacked = np.concatenate([freqs_np, pair_sums.reshape(-1, n_qudits)], axis=0)
    unique_freqs = np.unique(stacked, axis=0)

    single_match = np.all(freqs_np[:, None, :] == unique_freqs[None, :, :], axis=2)
    pair_match = np.all(pair_sums[:, :, None, :] == unique_freqs[None, None, :, :], axis=3)

    return CharacterExpansionData(
        freqs=jnp.array(freqs_np),
        coeffs=jnp.array(np.array(coeffs, dtype=complex)),
        param_indices=jnp.array(np.array(param_indices, dtype=int)),
        pair_match=jnp.array(pair_match),
        unique_freqs=jnp.array(unique_freqs),
        single_match=jnp.array(single_match),
    )


def _character_amplitudes(
    gates_params: ArrayLike,
    char_data: CharacterExpansionData,
    l_f: jnp.ndarray,
    dims: ArrayLike,
) -> jnp.ndarray:
    """Compute the observable-dependent amplitudes :math:`A_{\\mathbf{l},t}`.

    Terms with :math:`\\mathbf{f}_t \\cdot \\mathbf{l} \\equiv 0` get zero amplitude
    and drop out automatically; in particular :math:`\\mathbf{l} = \\mathbf{0}` gives
    :math:`\\Delta_\\mathbf{l} \\equiv 0`, making the truncation exact.

    Returns:
        jnp.ndarray: Amplitudes of shape ``(n_obs, T)``.
    """
    theta_t = jnp.asarray(gates_params)[char_data.param_indices]  # (T,)
    inv_d = 1.0 / jnp.asarray(dims, dtype=_real_dtype())
    # omega^{-f.l} = exp(-2 pi i sum_k f_k l_k / d_k) -> (n_obs, T)
    fl = (l_f.astype(_real_dtype()) * inv_d[jnp.newaxis, :]) @ char_data.freqs.astype(
        _real_dtype()
    ).T
    shift = jnp.exp(-2j * jnp.pi * fl)
    return (theta_t * char_data.coeffs)[jnp.newaxis, :] * (1.0 - shift)


def _control_variate_integrand(
    obs_pm: jnp.ndarray,
    accumulated_phase_diffs: jnp.ndarray,
    H: jnp.ndarray | None,
) -> jnp.ndarray:
    """Order-2 Taylor approximation control variate.

    Expands only the nonlinear factor :math:`e^{i\\Delta_\\mathbf{l}}` to second
    order, keeping the observable phase and the input-state correction exact.

    Args:
        obs_pm (jnp.ndarray): Observable phase matrix, shape ``(n_obs, n_samples)``.
        accumulated_phase_diffs (jnp.ndarray): Phase difference
            :math:`\\Delta_\\mathbf{l}(\\mathbf{z})`, shape ``(n_obs, n_samples)``.
        H (jnp.ndarray | None): Input-state correction, shape ``(n_obs, n_samples)``,
            or ``None`` for the default input state.

    Returns:
        jnp.ndarray: Control-variate integrand, shape ``(n_obs, n_samples)``.
    """
    D = accumulated_phase_diffs
    taylor = 1.0 + 1j * D - 0.5 * D**2
    out = obs_pm * taylor
    if H is not None:
        out = out * H
    return out


def _moment_functional(amps_lt: jnp.ndarray, char_data: CharacterExpansionData) -> jnp.ndarray:
    r"""Evaluate the moment functional :math:`\Sigma(\nu)` on every frequency bucket.

    Implements

    .. math::

        \Sigma(\mathbf{\nu}) = \delta_{\mathbf{\nu}, \mathbf{0}}
        + i \sum_t A_{\mathbf{l},t} \delta_{\mathbf{f}_t, \mathbf{\nu}}
        - \frac{1}{2} \sum_{t,u} A_{\mathbf{l},t} A_{\mathbf{l},u}
          \delta_{\mathbf{f}_t + \mathbf{f}_u, \mathbf{\nu}},

    i.e. it retains exactly those Taylor terms whose total character frequency
    equals :math:`\mathbf{\nu}`; everything else averages to zero by orthogonality.

    Returns:
        jnp.ndarray: Values of shape ``(n_obs, n_freq_unique)``, aligned with
        ``char_data.unique_freqs``.
    """
    n_obs = amps_lt.shape[0]
    n_unique = char_data.unique_freqs.shape[0]

    if n_unique == 0:
        return jnp.zeros((n_obs, 0), dtype=_complex_dtype())

    zero_bucket = jnp.all(char_data.unique_freqs == 0, axis=1).astype(_complex_dtype())
    order0 = jnp.broadcast_to(zero_bucket[jnp.newaxis, :], (n_obs, n_unique))

    single = char_data.single_match.astype(_complex_dtype())  # (T, n_unique)
    order1 = 1j * (amps_lt @ single)  # (n_obs, n_unique)

    pair = char_data.pair_match.astype(_complex_dtype())  # (T, T, n_unique)
    aa = amps_lt[:, :, jnp.newaxis] * amps_lt[:, jnp.newaxis, :]  # (n_obs, T, T)
    order2 = -0.5 * jnp.einsum("otu,tuv->ov", aa, pair)

    return order0 + order1 + order2


def _select_sigma(
    sigma_vals: jnp.ndarray, target: jnp.ndarray, char_data: CharacterExpansionData, dims: ArrayLike
) -> jnp.ndarray:
    """Look up :math:`\\Sigma` at target frequencies, reducing modulo ``dims``.

    Targets not present among the expansion's frequency buckets contribute only the
    order-0 delta, so they evaluate to 1 when the reduced target is zero and 0
    otherwise.

    Args:
        sigma_vals (jnp.ndarray): Bucket values, shape ``(n_obs, n_freq_unique)``.
        target (jnp.ndarray): Target frequencies, shape ``(n_obs, ..., n_qudits)``.
        char_data (CharacterExpansionData): Static expansion data.
        dims (ArrayLike): Per-qudit dimensions.

    Returns:
        jnp.ndarray: Values of shape ``target.shape[:-1]``.
    """
    dims_i = jnp.asarray(dims, dtype=jnp.int32)
    tgt = jnp.mod(jnp.asarray(target, dtype=jnp.int32), dims_i)

    n_obs = sigma_vals.shape[0]
    lead = tgt.shape[:-1]
    if lead[0] != n_obs:
        raise ValueError(
            f"Leading axis of target must be n_obs={n_obs}, got shape {tuple(target.shape)}."
        )

    n_qudits = tgt.shape[-1]
    inner = int(np.prod(lead[1:])) if len(lead) > 1 else 1
    flat = tgt.reshape(n_obs, inner, n_qudits)

    # Order-0 fallback for targets absent from the frequency buckets.
    is_zero = jnp.all(flat == 0, axis=-1).astype(_complex_dtype())  # (n_obs, inner)

    n_unique = char_data.unique_freqs.shape[0]
    if n_unique == 0:
        return is_zero.reshape(lead)

    match = jnp.all(
        flat[:, :, jnp.newaxis, :] == char_data.unique_freqs[jnp.newaxis, jnp.newaxis, :, :],
        axis=-1,
    )  # (n_obs, inner, n_unique)
    found = jnp.any(match, axis=-1)  # (n_obs, inner)

    # Each observable reads its own bucket values.
    from_bucket = jnp.einsum("ov,oiv->oi", sigma_vals, match.astype(_complex_dtype()))
    out = jnp.where(found, from_bucket, is_zero)
    return out.reshape(lead)


# pylint: disable=too-many-arguments
def _control_variate_expected_value(
    gates_params: ArrayLike,
    char_data: CharacterExpansionData,
    l_f: jnp.ndarray,
    m_f: jnp.ndarray,
    dims: ArrayLike,
    state_elems: ArrayLike | None,
    state_amps: ArrayLike | None,
) -> jnp.ndarray:
    r"""Analytic expectation value of the order-2 Taylor control variate.

    For the default input state (:math:`H_\mathbf{l} \equiv 1`) the surviving Taylor
    frequency must cancel that of the observable phase,

    .. math::

        \tau_\mathbf{l} = P_{\mathbf{l}\mathbf{m}} \Sigma(-\mathbf{m}), \qquad
        P_{\mathbf{l}\mathbf{m}} = \exp \left( -i\pi \sum_k \frac{m_k l_k}{d_k} \right).

    For a general sparse input state each pair :math:`(a, b)` contributes the
    character :math:`\chi_{\mathbf{m} + \mathbf{x}_a - \mathbf{x}_b}`, so the required
    Taylor frequency becomes pair-dependent,

    .. math::

        \tau_\mathbf{l} = P_{\mathbf{l}\mathbf{m}} \sum_{a,b} \Psi_a \bar{\Psi}_b\,
        \omega^{\mathbf{l} \cdot \mathbf{x}_b}\,
        \Sigma \left( -(\mathbf{m} + \mathbf{x}_a - \mathbf{x}_b) \right).

    Because this mean is exact, the control-variate estimator is unbiased for any
    choice of the coefficient :math:`c`.

    Returns:
        jnp.ndarray: Complex control mean of shape ``(n_obs,)``.
    """
    amps_lt = _character_amplitudes(gates_params, char_data, l_f, dims)  # (n_obs, T)
    sigma_vals = _moment_functional(amps_lt, char_data)  # (n_obs, n_unique)

    inv_d = 1.0 / jnp.asarray(dims, dtype=_real_dtype())
    P_lm = jnp.exp(
        -1j
        * jnp.pi
        * jnp.sum(m_f.astype(_real_dtype()) * l_f.astype(_real_dtype()) * inv_d, axis=1)
    )  # (n_obs,)

    m_i = jnp.asarray(m_f, dtype=jnp.int32)

    if state_elems is None or state_amps is None:
        sigma = _select_sigma(sigma_vals, -m_i, char_data, dims)  # (n_obs,)
        return P_lm * sigma

    X = jnp.asarray(state_elems, dtype=jnp.int32)  # (N, n_qudits)
    Psi = jnp.asarray(state_amps)  # (N,)

    x_diff = X[:, jnp.newaxis, :] - X[jnp.newaxis, :, :]  # (N, N, n_qudits)
    target = -(m_i[:, jnp.newaxis, jnp.newaxis, :] + x_diff[jnp.newaxis])  # (n_obs, N, N, n)
    sigma = _select_sigma(sigma_vals, target, char_data, dims)  # (n_obs, N, N)

    # omega^{l.x_b} -> (n_obs, N)
    omega_lx = jnp.exp(
        2j
        * jnp.pi
        * ((l_f.astype(_real_dtype()) * inv_d[jnp.newaxis, :]) @ X.astype(_real_dtype()).T)
    )
    amp_outer = Psi[:, jnp.newaxis] * jnp.conj(Psi)[jnp.newaxis, :]  # (N, N)

    total = jnp.einsum("ob,ab,oab->o", omega_lx, amp_outer, sigma)
    return P_lm * total


def _compute_mc_statistics(
    integrand: jnp.ndarray, n_samples: int
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute Monte Carlo mean, covariance, and mean squared magnitude from the integrand.

    Returns ``(expvals, cov, mean_y_sq)`` where ``cov`` is the per-observable
    covariance matrix of the mean estimator, shape ``(n_obs, 2, 2)``.
    """
    expvals = jnp.mean(integrand, axis=1)
    mean_y_sq = jnp.mean(jnp.abs(integrand) ** 2, axis=1)  # (n_obs,)

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
    return expvals, cov, mean_y_sq


def _compute_cv_mc_statistics(
    integrand: jnp.ndarray,
    cv_integrand: jnp.ndarray,
    cv_mean: jnp.ndarray,
    n_samples: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    r"""Monte Carlo statistics for the control-variate estimator.

    Forms :math:`y + c(\tilde{y} - \tau)` with the variance-minimising coefficient
    :math:`c^\star = -\text{Cov}(y, \tilde{y}) / \text{Var}(\tilde{y})`, which gives
    :math:`\text{Var} = (1 - \rho^2)` times the plain variance. Since the integrand is
    complex, a separate real coefficient is fitted for the real and imaginary parts;
    each keeps the estimator unbiased because :math:`\mathbb{E}[\tilde{y} - \tau] = 0`
    holds componentwise.

    ``mean_y_sq`` is deliberately computed from the raw integrand, since it is the
    mean of :math:`\vert y \vert^2` and not a quantity the control variate rescales.

    Returns:
        tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]: ``(expvals, cov, mean_y_sq)``
        matching the layout of :func:`_compute_mc_statistics`.
    """
    mean_y_sq = jnp.mean(jnp.abs(integrand) ** 2, axis=1)  # (n_obs,)

    def fit(y_part: jnp.ndarray, cv_part: jnp.ndarray, tau_part: jnp.ndarray) -> jnp.ndarray:
        y_c = y_part - jnp.mean(y_part, axis=1, keepdims=True)
        cv_c = cv_part - jnp.mean(cv_part, axis=1, keepdims=True)
        cov = jnp.sum(y_c * cv_c, axis=1)
        var_cv = jnp.sum(cv_c**2, axis=1)
        # The denominator is sanitized *before* dividing: a bare
        # jnp.where(var_cv > 0, -cov / var_cv, 0) still evaluates the division on the
        # untaken branch under reverse-mode AD and yields NaN gradients whenever the
        # control is exactly constant (e.g. l = 0, where Delta vanishes identically).
        safe_var = jnp.where(var_cv > 0, var_cv, 1.0)
        c = jnp.where(var_cv > 0, -cov / safe_var, 0.0)
        return y_part + c[:, jnp.newaxis] * (cv_part - tau_part[:, jnp.newaxis])

    re = fit(jnp.real(integrand), jnp.real(cv_integrand), jnp.real(cv_mean))
    im = fit(jnp.imag(integrand), jnp.imag(cv_integrand), jnp.imag(cv_mean))

    expvals = jnp.mean(re, axis=1) + 1j * jnp.mean(im, axis=1)

    re_c = re - jnp.mean(re, axis=1, keepdims=True)
    im_c = im - jnp.mean(im, axis=1, keepdims=True)
    var_re = jnp.sum(re_c**2, axis=1) / (n_samples - 1) / n_samples
    var_im = jnp.sum(im_c**2, axis=1) / (n_samples - 1) / n_samples
    cov_re_im = jnp.sum(re_c * im_c, axis=1) / (n_samples - 1) / n_samples
    cov = jnp.stack(
        [
            jnp.stack([var_re, cov_re_im], axis=-1),
            jnp.stack([cov_re_im, var_im], axis=-1),
        ],
        axis=-2,
    )  # (n_obs, 2, 2)
    return expvals, cov, mean_y_sq


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
                return_mean_y_sq=False,
            ) -> (expvals, cov) or (expvals, cov, mean_y_sq)

        where ``expvals`` is a complex array of shape ``(n_obs,)`` containing
        the estimated moments, and ``cov`` has shape ``(n_obs, 2, 2)``
        providing the real/imaginary covariance matrix of the mean estimator
        for each observable. When ``return_mean_y_sq=True``, also returns the
        per-observable mean of :math:`|y|^2` (needed internally by the MMD
        loss).

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
    ...     d=3,
    ...     n_qudits=2,
    ...     gates={0: [[1, 0]], 1: [[0, 1]]},
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
    if config.control_variate and config.phase_fn is not None:
        raise ValueError("Phase layers are not compatible with control variates.")

    generators, param_map = _parse_qudit_generator_dict(config.gates, config.n_qudits)

    n = config.n_qudits
    dims = _dims_to_numpy(config.d, n)
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

    gen_np, pm_np = np.array(generators), np.array(param_map)
    gate_weights = np.sum(gen_np != 0, axis=1)

    # Depends only on the gate set, so it is built once regardless of observables.
    char_data = _build_character_expansion(gen_np, pm_np, dims) if config.control_variate else None

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
            weight_data=_build_all_weight_groups(
                gen_np, pm_np, gate_weights, default_samples, l_vecs, dims
            ),
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
        return_mean_y_sq: bool = False,
    ) -> (
        tuple[jnp.ndarray, jnp.ndarray] | tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
    ):  # pylint: disable=too-many-arguments
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
            return_mean_y_sq (bool, optional): If ``True``, also return the
                per-observable mean of ``|y_r|^2``. Defaults to ``False``.

        Returns:
            tuple[jnp.ndarray, jnp.ndarray] | tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            By default returns ``(expvals, cov)`` where ``expvals`` are the estimated
            complex expectation values, shape ``(n_obs,)``, and ``cov`` stores the
            real-imaginary covariance matrices of the mean estimator, shape
            ``(n_obs, 2, 2)``.

            When ``return_mean_y_sq=True``, also returns ``mean_y_sq`` with shape
            ``(n_obs,)``. This equals 1 when the per-sample integrand has unit
            modulus (default input state, diagonal observables).
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
            w_data = _build_all_weight_groups(gen_np, pm_np, gate_weights, samples, l_vecs, dims)

        accumulated_phase_diffs = _accumulate_phase_diffs(
            gates_params, w_data, n_obs, _n, vmapped_phase_func, phase_fn_params, samples, l_vecs
        )

        state_elems = config.init_state_elems if init_state_elems is None else init_state_elems
        state_amps = config.init_state_amps if init_state_amps is None else init_state_amps

        H = None
        if state_elems is not None and state_amps is not None:
            H = _compute_initial_state_correction(samples, l_f, state_elems, state_amps, dims)

        integrand = obs_pm * jnp.exp(1j * accumulated_phase_diffs)
        if H is not None:
            integrand = integrand * H

        if not config.control_variate:
            expvals, cov, mean_y_sq = _compute_mc_statistics(integrand, _n)
        else:
            cv_integrand = _control_variate_integrand(obs_pm, accumulated_phase_diffs, H)
            cv_mean = _control_variate_expected_value(
                gates_params, char_data, l_f, m_f, dims, state_elems, state_amps
            )
            expvals, cov, mean_y_sq = _compute_cv_mc_statistics(
                integrand, cv_integrand, cv_mean, _n
            )

        if return_mean_y_sq:
            return expvals, cov, mean_y_sq
        return expvals, cov

    return qudit_expval_batched
