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

This module extends :mod:`~pennylane.labs.tcdq.iqp` from qubits to qudits. It
estimates Heisenberg-Weyl moments without building the full quantum state.

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

from .base import ObservableAlgebra, TCDQSimulator, estimator


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

    .. warning::

        ``QuditCircuitConfig`` and :func:`build_qudit_expval_func` are
        superseded by :class:`QuditIQPSimulator`, which exposes the same
        estimator through the :class:`~pennylane.labs.tcdq.TCDQSimulator`
        interface. They are kept for backwards compatibility and will be
        removed.

    Args:
        dims (int | Sequence[int]): Local qudit dimension(s). Either a single
            ``int`` (e.g., 2 for qubits, 3 for qutrits), which is broadcast to
            every qudit, or a sequence of length ``n_qudits`` giving a distinct
            dimension :math:`d_j` per qudit.
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

    **Example**

    >>> import jax
    >>> import jax.numpy as jnp
    >>> from pennylane.labs.tcdq import QuditCircuitConfig
    >>> config = QuditCircuitConfig(
    ...     dims=3,
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

    #: Local qudit dimension(s): an int (uniform) or list (per-qudit sequence).
    dims: int | Sequence[int] = None
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


class QuditIQPSimulator(TCDQSimulator):
    r"""Qudit IQP circuit with Monte Carlo estimation of Heisenberg-Weyl moments.

    The qudit analogue of :class:`~pennylane.labs.tcdq.IQPSimulator`. A qudit
    IQP circuit has the form
    :math:`U(\mathbf{\theta}) = (F^{\otimes n})^\dagger D(\mathbf{\theta}) F^{\otimes n}`,
    where :math:`F` is the Fourier transform and :math:`D(\mathbf{\theta})` is
    a diagonal phase unitary. This simulator estimates the complex expectation
    values :math:`\langle O(\mathbf{l}, \mathbf{m}) \rangle` of Heisenberg-Weyl
    displacement operators by averaging over randomly sampled dit-strings.

    Args:
        dims (int | Sequence[int]): Local qudit dimension(s). Either a single
            ``int`` broadcast to every qudit, or a sequence of length
            ``n_qudits`` giving a distinct dimension :math:`d_j` per qudit.
        n_qudits (int): Number of qudits in the circuit.
        gates (dict[int, list[list[int]]]): Circuit structure mapping each
            trainable-parameter index to a list of generator vectors. Each
            generator vector has length ``n_qudits`` with integer entries in
            :math:`\{0, \ldots, d_j-1\}` specifying the power of :math:`Z` on
            each qudit.
        n_samples (int): Default number of random dit-strings drawn per estimate.
        key (ArrayLike): Default JAX PRNG key for dit-string generation.
        init_state (tuple[ArrayLike, ArrayLike] | None): Optional
            ``(elements, amplitudes)`` pair describing a custom initial state,
            where ``elements`` has shape ``(N, n_qudits)`` with entries in
            :math:`\{0, \ldots, d-1\}` and ``amplitudes`` is complex of shape
            ``(N,)``. Defaults to ``None``, the uniform superposition via the
            Fourier transform.
        phase_fn (Callable | None): Optional diagonal phase layer
            ``phase_fn(params, ditstring)``. Defaults to ``None``.

    Raises:
        ValueError: If ``n_samples`` is not greater than 1.

    **Example**

    >>> import jax
    >>> import jax.numpy as jnp
    >>> from pennylane.labs.tcdq import QuditIQPSimulator
    >>> sim = QuditIQPSimulator(
    ...     dims=3,
    ...     n_qudits=2,
    ...     gates={0: [[1, 0]], 1: [[0, 1]]},
    ...     n_samples=512,
    ...     key=jax.random.PRNGKey(0),
    ... )
    >>> expval = sim.build_estimator("hw_expval")
    >>> l_vecs = jnp.array([[1, 0], [0, 1]], dtype=jnp.int32)
    >>> values, cov = expval(jnp.array([0.2, -0.1]), (l_vecs, jnp.zeros_like(l_vecs)))
    >>> values.shape, cov.shape
    ((2,), (2, 2, 2))

    .. seealso::

        :class:`~pennylane.labs.tcdq.TCDQSimulator`,
        :func:`~pennylane.labs.tcdq.build_qudit_mmd_loss`,
        `Spectral Born machines: classically trainable quantum generative models for discrete data <https://arxiv.org/abs/2607.06675>`_
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        dims: int | Sequence[int],
        n_qudits: int,
        gates: dict[int, list[list[int]]],
        n_samples: int,
        key: ArrayLike,
        init_state: tuple[ArrayLike, ArrayLike] | None = None,
        phase_fn: Callable | None = None,
    ):
        if n_samples <= 1:
            raise ValueError("n_samples must be greater than 1")

        self.gates = gates
        self._dims = _dims_to_numpy(dims, n_qudits)
        self.n_samples = n_samples
        self.key = key
        self.init_state = init_state
        self.phase_fn = phase_fn

    @property
    def local_dims(self) -> tuple[int, ...]:
        """tuple[int, ...]: Local dimension of each qudit."""
        return tuple(int(x) for x in self._dims)

    @estimator("hw_expval", algebra=ObservableAlgebra.HEISENBERG_WEYL)
    def _build_hw_expval(self) -> Callable:
        """Build the Monte Carlo estimator for Heisenberg-Weyl moments.

        Returns:
            Callable: A function with signature::

                expval(params, observables, *, key=None, n_samples=None,
                       phase_params=None) -> (values, cov)
        """
        dims = self._dims
        n = len(dims)
        generators, param_map = _parse_qudit_generator_dict(self.gates, n)
        gen_np, pm_np = np.array(generators), np.array(param_map)
        gate_weights = np.sum(gen_np != 0, axis=1)
        default_samples = _compute_qudit_samples(self.key, self.n_samples, n, dims)
        init_elems, init_amps = self.init_state if self.init_state is not None else (None, None)

        vmapped_phase_func = None
        if self.phase_fn is not None:
            dims_j = jnp.asarray(dims)
            phase_fn = self.phase_fn

            def compute_phase_diff(p_params, sample, l_vec):
                return phase_fn(p_params, sample) - phase_fn(p_params, (sample - l_vec) % dims_j)

            vmapped_phase_func = jax.vmap(
                jax.vmap(compute_phase_diff, in_axes=(None, 0, None)),
                in_axes=(None, None, 0),
            )

        # pylint: disable=too-many-arguments
        def hw_expval(
            params: ArrayLike,
            observables: tuple[ArrayLike, ArrayLike],
            *,
            key: ArrayLike | None = None,
            n_samples: int | None = None,
            phase_params: ArrayLike | None = None,
            init_state: tuple[ArrayLike, ArrayLike] | None = None,
        ) -> tuple[jnp.ndarray, jnp.ndarray]:
            """Estimate Heisenberg-Weyl moments and their real-imaginary covariance.

            Args:
                params (ArrayLike): Trainable gate parameters, shape ``(n_params,)``.
                observables (tuple[ArrayLike, ArrayLike]): Pair ``(l_vecs, m_vecs)``
                    of integer arrays of shape ``(n_obs, n_qudits)``.
                key (ArrayLike | None): Override for the dit-string sampling key.
                n_samples (int | None): Override for the number of dit-strings.
                phase_params (ArrayLike | None): Trainable parameters of the
                    phase layer, required when the simulator has a ``phase_fn``.
                init_state (tuple[ArrayLike, ArrayLike] | None): Override for the
                    simulator's ``init_state``.

            Returns:
                tuple[jnp.ndarray, jnp.ndarray]: ``(values, cov)`` where ``values``
                is complex of shape ``(n_obs,)`` and ``cov`` is real of shape
                ``(n_obs, 2, 2)``, the real-imaginary covariance of each mean.
            """
            l_vecs = jnp.array(observables[0], dtype=jnp.int32)
            l_f = l_vecs.astype(jnp.float32)
            m_f = jnp.array(observables[1], dtype=jnp.int32).astype(jnp.float32)
            n_obs = l_vecs.shape[0]

            if key is None and n_samples is None:
                samples, n_eff = default_samples, self.n_samples
            else:
                n_eff = self.n_samples if n_samples is None else n_samples
                samples = _compute_qudit_samples(self.key if key is None else key, n_eff, n, dims)

            weight_data = _build_all_weight_groups(
                gen_np, pm_np, gate_weights, samples, l_vecs, dims
            )
            accumulated_phase_diffs = _accumulate_phase_diffs(
                params,
                weight_data,
                n_obs,
                n_eff,
                vmapped_phase_func,
                phase_params,
                samples,
                l_vecs,
            )

            elems, amps = (init_elems, init_amps) if init_state is None else init_state

            integrand = _obs_phase_matrix(samples, m_f, l_f, dims) * jnp.exp(
                1j * accumulated_phase_diffs
            )
            if elems is not None and amps is not None:
                integrand = integrand * _compute_initial_state_correction(
                    samples, l_f, elems, amps, dims
                )

            return _compute_mc_statistics(integrand, n_eff)

        return hw_expval


def _simulator_from_config(config: QuditCircuitConfig) -> QuditIQPSimulator:
    """Build a :class:`QuditIQPSimulator` from a legacy :class:`QuditCircuitConfig`."""
    elems, amps = config.init_state_elems, config.init_state_amps

    return QuditIQPSimulator(
        dims=config.dims,
        n_qudits=config.n_qudits,
        gates=config.gates,
        n_samples=config.n_samples,
        key=config.key,
        init_state=None if elems is None or amps is None else (elems, amps),
        phase_fn=config.phase_fn,
    )


def build_qudit_expval_func(
    config: QuditCircuitConfig,
) -> Callable:
    """Build an estimator for expectation values of a qudit IQP circuit.

    .. warning::

        This function is superseded by :class:`QuditIQPSimulator`. Prefer
        ``QuditIQPSimulator(...).build_estimator("hw_expval")``, which returns
        an :class:`~pennylane.labs.tcdq.Estimator` carrying the metadata that
        loss functions use to check compatibility. This function is kept for
        backwards compatibility and will be removed.

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

        :class:`~pennylane.labs.tcdq.QuditIQPSimulator`,
        `Spectral Born machines: classically trainable quantum generative models for discrete data <https://arxiv.org/pdf/2607.06675>`_.
    """
    base_estimator = _simulator_from_config(config).build_estimator("hw_expval")

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

        Raises:
            ValueError: If no observables are available.
        """
        obs = config.observables if observables is None else observables
        if obs is None:
            raise ValueError(
                "No observables specified. Provide them in QuditCircuitConfig "
                "or pass at call time via the observables argument."
            )

        elems = config.init_state_elems if init_state_elems is None else init_state_elems
        amps = config.init_state_amps if init_state_amps is None else init_state_amps

        return base_estimator(
            gates_params,
            obs,
            key=key,
            n_samples=n_samples,
            phase_params=phase_fn_params,
            init_state=None if elems is None or amps is None else (elems, amps),
        )

    return qudit_expval_batched
