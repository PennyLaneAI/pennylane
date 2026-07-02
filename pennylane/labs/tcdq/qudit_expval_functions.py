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
Implementations for the qudit expectation value functions.

See ``notes.md`` for the full theoretical derivation and a notation
cross-reference table mapping math symbols to code variables.
"""

import itertools
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike


@dataclass
class QuditCircuitConfig:  # pylint: disable=too-many-instance-attributes
    """
    Configuration data for a qudit IQP circuit simulation.

    Args:
        d (int): Qudit dimension (e.g. 2 for qubits, 3 for qutrits).
        n_qudits (int): Number of qudits.
        gates (dict[int, list[list[int]]]): Circuit structure mapping parameter indices to lists
            of generator vectors. Each generator vector has length ``n_qudits`` with integer
            entries in ``{0, ..., d-1}``, representing the Z-power on each qudit.
        observables (tuple[ArrayLike, ArrayLike] | None): Pair ``(l_vecs, m_vecs)`` specifying
            the displacement operators ``D(l, m)`` to measure. Each array has shape
            ``(n_obs, n_qudits)`` with integer entries in ``{0, ..., d-1}``.  If ``None``,
            observables must be supplied when constructing the expectation-value function
            (e.g. via ``build_qudit_mmd_loss``, which generates its own observables).
        n_samples (int): Number of Monte Carlo samples for the estimation of the expectation value.
        key (ArrayLike): Random key for JAX.
        init_state_elems (ArrayLike | None): Support elements of the initial state. Array of
            shape ``(N, n_qudits)`` with integer entries in ``{0, ..., d-1}``, where ``N`` is
            the number of non-zero amplitudes. Each row is a dit-string in ``Z_d^n``.
        init_state_amps (ArrayLike | None): Complex amplitudes of the initial state. Array of
            shape ``(N,)`` corresponding to the support elements.
    """

    d: int
    n_qudits: int
    gates: dict[int, list[list[int]]]
    observables: tuple[ArrayLike, ArrayLike] | None = None
    n_samples: int = 10000
    key: ArrayLike = field(default_factory=lambda: jax.random.PRNGKey(0))
    init_state_elems: ArrayLike | None = None
    init_state_amps: ArrayLike | None = None


def _parse_qudit_generator_dict(circuit_def: dict[int, list[list[int]]], n_qudits: int):
    """
    Converts a qudit circuit definition dict into generator matrix and parameter map.

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


def _compute_qudit_samples(key: ArrayLike, num_samples: int, n_qudits: int, d: int) -> jnp.ndarray:
    """Generates uniformly random dit-strings from Z_d^n."""
    return jax.random.randint(key, shape=(num_samples, n_qudits), minval=0, maxval=d)


class WeightGroupData(NamedTuple):
    """Precomputed factor matrices for gates that share the same weight (number of active qudits).

    Gates are grouped by weight ``ω`` (number of active qudits) so that
    the ``2^ω``-term angle-addition expansion of the phase difference
    ``Φ_g(z) − Φ_g(z ⊕ l)`` can be vectorised over gates.  See
    ``notes.md`` §2.3 for the full derivation.

    ``samples_matrices[σ]`` and ``obs_matrices[σ]`` store the ``B`` and
    ``C`` factor matrices for the ``σ``-th expansion term.  The first
    entry (``σ = 0…0``, all-cos) equals ``Φ_g(z)`` itself.

    Args:
        param_indices: Maps each gate in this group to its parameter index in the
            global ``gates_params`` array, shape ``(n_gates,)``.
        samples_matrices: ``2^ω`` matrices of shape ``(n_gates, n_samples)``
            giving the sample-side factor for each angle-addition term.
        obs_matrices: ``2^ω`` matrices of shape ``(n_gates, n_obs)`` giving the
            observable-side factor for each angle-addition term.
    """

    param_indices: jnp.ndarray
    samples_matrices: list[jnp.ndarray]
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
    gate_vals: np.ndarray, z_at_support: jnp.ndarray, l_at_support: jnp.ndarray, d: int
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Returns (state_cos, state_sin, obs_cos, obs_sin) trig factors over the gate support.

    Computes the ``c(v, u)`` and ``s̄(v, u)`` building blocks from
    ``notes.md`` §2.3 for each active qudit position.
    """
    g = jnp.array(gate_vals, dtype=jnp.float32)[:, :, jnp.newaxis]
    angle_z = 2 * jnp.pi * g * z_at_support.astype(jnp.float32) / d + jnp.pi / 4
    angle_l = 2 * jnp.pi * g * l_at_support.astype(jnp.float32) / d
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
    """Enumerate all ``2^ω`` angle-addition terms to build the ``B`` and ``C`` factor matrices.

    Each term corresponds to a choice ``σ ∈ {0,1}^ω`` of cos (0) or sin (1)
    at each active qudit position.  The all-cos entry (``σ = 0…0``) equals
    ``Φ_g(z)`` itself.  See ``notes.md`` §2.3 for the derivation.
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
    d: int,
) -> WeightGroupData:
    """Precompute the ``B`` and ``C`` factor matrices for a weight-``ω`` gate group.

    See ``notes.md`` §2.3 for the full pipeline and matrix definitions.
    """
    n_gates = len(generators_w)
    num_samples = samples.shape[0]
    n_obs = l_vecs.shape[0]
    omega = int(np.count_nonzero(generators_w[0]))
    supports = np.array([np.where(g != 0)[0] for g in generators_w])  # (n_gates, omega)
    gate_vals = np.array([g[s] for g, s in zip(generators_w, supports)])  # (n_gates, omega)

    z_at_support = _gather_support_values(samples, supports, num_samples, n_gates, omega)
    l_at_support = _gather_support_values(l_vecs, supports, n_obs, n_gates, omega)

    state_cos, state_sin, obs_cos, obs_sin = _compute_trigonometric_building_blocks(
        gate_vals, z_at_support, l_at_support, d
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
    samples: jnp.ndarray, m_f: jnp.ndarray, l_f: jnp.ndarray, d: int
) -> jnp.ndarray:
    """Compute the observable phase matrix ``J`` (see ``notes.md`` §2.1).

    ``J[i, j] = exp(iπ/d · m_i · (2z_j − l_i))``.
    """
    s_f = samples.astype(jnp.float32)
    return jnp.exp(
        (1j * jnp.pi / d) * (2 * m_f @ s_f.T - jnp.sum(m_f * l_f, axis=1, keepdims=True))
    )


# pylint: disable=too-many-arguments
def _build_all_weight_groups(
    gen_np: np.ndarray,
    pm_np: np.ndarray,
    gate_weights: np.ndarray,
    samples: jnp.ndarray,
    l_vecs: jnp.ndarray,
    d: int,
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
                d=d,
            )
        )
    return weight_data


def _accumulate_phase_diffs(
    gates_params: ArrayLike,
    weight_data: list[WeightGroupData],
    n_obs: int,
    n_samples: int,
) -> jnp.ndarray:
    """Assemble the accumulated phase-difference matrix ``E`` (see ``notes.md`` §2.3)."""
    accumulated = jnp.zeros((n_obs, n_samples))
    for group in weight_data:
        theta_w = jnp.asarray(gates_params)[group.param_indices]
        accumulated = accumulated + (theta_w @ group.samples_matrices[0])[jnp.newaxis, :]
        for B_sigma, C_sigma in zip(group.samples_matrices, group.obs_matrices):
            accumulated = accumulated - (C_sigma.T * theta_w) @ B_sigma
    return accumulated


def _compute_initial_state_correction(
    samples: jnp.ndarray,
    l_f: jnp.ndarray,
    state_elems: ArrayLike,
    state_amps: ArrayLike,
    d: int,
) -> jnp.ndarray:
    """Compute the initial-state correction factor ``H`` (see ``notes.md`` §3.2)."""
    s_f = samples.astype(jnp.float32)
    X_state = jnp.asarray(state_elems).astype(jnp.float32)  # (N, n)
    Psi = jnp.asarray(state_amps)  # (N,)

    # ω^{Z·X^T} where ω = exp(2πi/d) — shape (s, N)
    omega_ZX = jnp.exp(2j * jnp.pi * (s_f @ X_state.T) / d)

    # Ψ̃^(2) = ω^{Z·X^T} · Ψ — shape (s,)
    psi_tilde_2 = omega_ZX @ Psi

    # F = Ψ* · 1_{1×s} ⊙ ω^{-X·Z^T} — shape (N, s)
    F_mat = Psi.conj()[:, jnp.newaxis] * omega_ZX.conj().T

    # Ψ̃^(1) = ω^{L·X^T} · F — shape (l, s)
    omega_LX = jnp.exp(2j * jnp.pi * (l_f @ X_state.T) / d)  # (l, N)
    psi_tilde_1 = omega_LX @ F_mat

    # H = Ψ̃^(1) ⊙ (1_{l×1} · (Ψ̃^(2))^T) — shape (l, s)
    return psi_tilde_1 * psi_tilde_2[jnp.newaxis, :]


def _compute_mc_statistics(
    integrand: jnp.ndarray, n_samples: int
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute Monte Carlo statistics from the integrand (see ``notes.md`` §4).

    Returns ``(expvals, cov, mean_y_sq)`` where *cov* is the per-observable
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


def build_qudit_expval_func(
    config: QuditCircuitConfig,
) -> Callable:
    """Factory that returns an expectation-value function for a qudit circuit.

    The returned function estimates ``⟨O(l, m)⟩`` for each displacement-operator
    observable via a Monte Carlo method.  See ``notes.md`` §2 for the full
    derivation of the estimator.

    Args:
        config (QuditCircuitConfig): Circuit configuration.

    Returns:
        Callable: ``qudit_expval_batched(gates_params, key=None, n_samples=None)``
        returning ``(expvals, cov)`` by default (shapes ``(n_obs,)`` and
        ``(n_obs, 2, 2)``).  With ``return_mean_y_sq=True``, also returns
        ``mean_y_sq`` (shape ``(n_obs,)``), needed for the QQ U-statistic
        correction in the MMD loss (see ``notes.md`` §5.3).
    """
    generators, param_map = _parse_qudit_generator_dict(config.gates, config.n_qudits)

    d = config.d
    n = config.n_qudits

    default_samples = _compute_qudit_samples(config.key, config.n_samples, n, d)

    gen_np = np.array(generators)
    pm_np = np.array(param_map)
    gate_weights = np.sum(gen_np != 0, axis=1)

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
                gen_np, pm_np, gate_weights, default_samples, l_vecs, d
            ),
            obs_phase_matrix=_obs_phase_matrix(default_samples, m_f, l_f, d),
        )
    else:
        defaults = None

    def qudit_expval_batched(
        gates_params: ArrayLike,
        key: ArrayLike | None = None,
        n_samples: int | None = None,
        observables: tuple[ArrayLike, ArrayLike] | None = None,
        init_state_elems: ArrayLike | None = None,
        init_state_amps: ArrayLike | None = None,
        return_mean_y_sq: bool = False,
    ) -> (
        tuple[jnp.ndarray, jnp.ndarray] | tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
    ):  # pylint: disable=too-many-arguments
        """
        Compute batched Monte Carlo expectation values for the qudit IQP circuit.

        Args:
            gates_params (ArrayLike): 1-D array of gate parameters.
            key (ArrayLike | None, optional): Runtime override for the JAX PRNG key
                used for sampling. Defaults to None.
            n_samples (int | None, optional): Runtime override for the number of
                Monte Carlo samples. Defaults to None.
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
                per-observable Monte Carlo mean of ``|y_r|^2``. Defaults to ``False``.

        Returns:
            tuple[jnp.ndarray, jnp.ndarray] | tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            By default returns ``(expvals, cov)`` where ``expvals`` are the estimated
            complex expectation values, shape ``(n_obs,)``, and ``cov`` are the
            covariance matrices of the mean estimator, shape ``(n_obs, 2, 2)``.

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
            samples = _compute_qudit_samples(_key, _n, n, d)
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
            obs_pm = _obs_phase_matrix(samples, m_f, l_f, d)
            w_data = _build_all_weight_groups(gen_np, pm_np, gate_weights, samples, l_vecs, d)

        accumulated_phase_diffs = _accumulate_phase_diffs(gates_params, w_data, n_obs, _n)

        state_elems = config.init_state_elems if init_state_elems is None else init_state_elems
        state_amps = config.init_state_amps if init_state_amps is None else init_state_amps

        integrand = obs_pm * jnp.exp(1j * accumulated_phase_diffs)
        if state_elems is not None and state_amps is not None:
            H = _compute_initial_state_correction(samples, l_f, state_elems, state_amps, d)
            integrand = integrand * H

        expvals, cov, mean_y_sq = _compute_mc_statistics(integrand, _n)

        if return_mean_y_sq:
            return expvals, cov, mean_y_sq
        return expvals, cov

    return qudit_expval_batched
