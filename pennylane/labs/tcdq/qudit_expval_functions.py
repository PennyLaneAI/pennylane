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
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

_REAL_DTYPE = jnp.float32
_COMPLEX_DTYPE = jnp.complex64
_INDEX_DTYPE = jnp.int32


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
        max_memory (float | None): Upper bound, in gigabytes (1 GB = 1024**3 bytes), on the
            temporary memory used by the phase-difference computation. If ``None`` (default),
            memory is unbounded.

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
    #: Memory budget in GB for the blocked phase-difference computation, or ``None`` for unbounded.
    max_memory: float | None = None


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

    Raises:
        TypeError: If ``gate`` is not a ``dict`` (e.g. a dense length-``n_qudits`` vector).
        IndexError: If a qudit index is outside ``[-n_qudits, n_qudits)``.
    """
    if not isinstance(gate, dict):
        raise TypeError(
            "Each qudit gate must be a dict mapping qudit index to Z power, e.g. {0: 1, 2: 2}; "
        )

    support: list[int] = []
    powers: list[int] = []
    for qudit, power in gate.items():
        q = int(qudit)
        if q >= n_qudits or q < 0:
            raise IndexError(f"Qudit index {q} out of range for a {n_qudits}-qudit circuit.")
        # q = q % n_qudits
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

        if not isinstance(param_idx, int):
            raise TypeError(f"Qudit indices must be type int, got type {type(param_idx)} instead.")

        if param_idx < 0:
            raise ValueError(
                f"Qudit indices must be non-negative integers, got index {param_idx} instead."
            )

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
            param_indices=jnp.array(pidx_list, dtype=_INDEX_DTYPE),
        )
        for omega, (sup_list, pow_list, pidx_list) in sorted(buckets.items())
    ]


def _compute_qudit_samples(
    key: ArrayLike, num_samples: int, n_qudits: int, dims: ArrayLike
) -> jnp.ndarray:
    """Generates uniformly random dit-strings from the product Z_{d_1} x ... x Z_{d_n}."""

    maxval = jnp.asarray(dims, dtype=_INDEX_DTYPE)[jnp.newaxis, :]  # (1, n_qudits)
    samples = jax.random.randint(key, shape=(num_samples, n_qudits), minval=0, maxval=maxval)
    return samples.astype(_INDEX_DTYPE)


_BYTES_PER_GB = 1024**3
_FLOAT_BYTES = 4


def _memory_budget_bytes(max_memory: float | None) -> int | None:
    """Convert a ``max_memory`` budget in GB to an integer number of bytes."""
    if max_memory is None:
        return None

    if (
        isinstance(max_memory, bool)
        or not isinstance(max_memory, (int, float, np.integer, np.floating))
        or not math.isfinite(max_memory)
        or max_memory <= 0
    ):
        raise ValueError(
            "QuditCircuitConfig.max_memory must be None or a positive, finite number of "
            f"gigabytes, got {max_memory!r}."
        )
    return int(max_memory * _BYTES_PER_GB)


def _qudit_phase_bytes_per_gate(omega: int, n_samples: int, n_obs: int) -> int:
    """Approximate temporary bytes needed per gate of weight ``omega`` inside one block."""
    per_column = _FLOAT_BYTES * (3 * omega + 2**omega)
    return per_column * (n_samples + n_obs)


def _qudit_phase_block_size(max_memory_bytes: int, omega: int, n_samples: int, n_obs: int) -> int:
    """Number of weight-``omega`` gates per block that fits within ``max_memory_bytes``."""
    return max(1, max_memory_bytes // _qudit_phase_bytes_per_gate(omega, n_samples, n_obs))


def _obs_phase_matrix(
    samples: jnp.ndarray, m_f: jnp.ndarray, l_f: jnp.ndarray, dims: ArrayLike
) -> jnp.ndarray:
    """Compute the observable phase matrix.

    :math:`J[i, j] = \\exp(i\\pi \\sum_k m_{ik} (2 z_{jk} - l_{ik}) / d_k)`.
    """
    s_f = samples.astype(_REAL_DTYPE)
    inv_d = (1.0 / jnp.asarray(dims, dtype=_REAL_DTYPE))[jnp.newaxis, :]  # (1, n_qudits)
    m_scaled = m_f * inv_d  # (n_obs, n_qudits)
    return jnp.exp(
        1j * jnp.pi * (2 * m_scaled @ s_f.T - jnp.sum(m_scaled * l_f, axis=1, keepdims=True))
    ).astype(_COMPLEX_DTYPE)


class _PrecomputedObsData(NamedTuple):
    """Bundled precomputed observable data from the factory."""

    l_vecs: jnp.ndarray
    n_obs: int
    l_f: jnp.ndarray
    m_f: jnp.ndarray
    obs_phase_matrix: jnp.ndarray


# pylint: disable=too-many-arguments
def _group_block_contribution(
    theta_ext: jnp.ndarray,
    samples_t: jnp.ndarray,
    l_t: jnp.ndarray,
    dims_f: jnp.ndarray,
    supports: jnp.ndarray,
    powers: jnp.ndarray,
    param_indices: jnp.ndarray,
) -> jnp.ndarray:
    """Phase-difference contribution of one block of equal-weight gates.

    Args:
        theta_ext (jnp.ndarray): Gate parameters with a trailing zero appended, shape
            ``(n_params + 1,)``. Padded gates index the trailing zero and contribute nothing.
        samples_t (jnp.ndarray): Transposed samples, shape ``(n_qudits, n_samples)``, float.
        l_t (jnp.ndarray): Transposed observable ``l`` vectors, shape ``(n_qudits, n_obs)``, float.
        dims_f (jnp.ndarray): Per-qudit dimensions, shape ``(n_qudits,)``, float.
        supports (jnp.ndarray): Active qudit indices of the block, shape ``(block, omega)``.
        powers (jnp.ndarray): :math:`Z` powers of the block, shape ``(block, omega)``.
        param_indices (jnp.ndarray): Parameter index of each gate, shape ``(block,)``.

    Returns:
        jnp.ndarray: Accumulated phase differences, shape ``(n_obs, n_samples)``.
    """
    omega = supports.shape[1]
    theta = theta_ext[param_indices]  # (block,)

    g = powers.astype(_REAL_DTYPE)[:, :, jnp.newaxis]  # (block, omega, 1)
    d_s = dims_f[supports][:, :, jnp.newaxis]  # (block, omega, 1)
    z_at_support = jnp.take(samples_t, supports, axis=0)  # (block, omega, n_samples)
    l_at_support = jnp.take(l_t, supports, axis=0)  # (block, omega, n_obs)

    angle_z = 2 * jnp.pi * g * z_at_support / d_s + jnp.pi / 4
    angle_l = 2 * jnp.pi * g * l_at_support / d_s
    state_factors = (jnp.sqrt(2.0) * jnp.cos(angle_z), jnp.sqrt(2.0) * jnp.sin(angle_z))
    obs_factors = (jnp.cos(angle_l), jnp.sin(angle_l))

    accumulated = None
    for sigma in itertools.product([0, 1], repeat=omega):
        B_sigma = state_factors[sigma[0]][:, 0, :]
        C_sigma = obs_factors[sigma[0]][:, 0, :]
        for k in range(1, omega):
            B_sigma = B_sigma * state_factors[sigma[k]][:, k, :]
            C_sigma = C_sigma * obs_factors[sigma[k]][:, k, :]
        if accumulated is None:  # sigma == (0, ..., 0): B_sigma is the unshifted phi_g(z)
            accumulated = (theta @ B_sigma)[jnp.newaxis, :]
        accumulated = accumulated - (C_sigma.T * theta) @ B_sigma

    return accumulated


@partial(jax.jit, static_argnames=("block_size",))
def _group_phase_diffs(
    theta_ext: jnp.ndarray,
    samples_t: jnp.ndarray,
    l_t: jnp.ndarray,
    dims_f: jnp.ndarray,
    group: SparseGateGroup,
    block_size: int,
) -> jnp.ndarray:
    """Phase differences of one weight group, processed in memory-bounded blocks."""
    n_gates, omega = group.supports.shape
    supports = jnp.asarray(group.supports, dtype=_INDEX_DTYPE)
    powers = jnp.asarray(group.powers, dtype=_INDEX_DTYPE)
    param_indices = jnp.asarray(group.param_indices, dtype=_INDEX_DTYPE)

    block_fn = jax.checkpoint(
        lambda s, p, i: _group_block_contribution(theta_ext, samples_t, l_t, dims_f, s, p, i)
    )

    if n_gates <= block_size:
        return block_fn(supports, powers, param_indices)

    n_blocks = -(-n_gates // block_size)
    n_pad = n_blocks * block_size - n_gates
    if n_pad:
        pad_index = theta_ext.shape[0] - 1  # points at the appended zero parameter
        supports = jnp.concatenate([supports, jnp.zeros((n_pad, omega), supports.dtype)])
        powers = jnp.concatenate([powers, jnp.ones((n_pad, omega), powers.dtype)])
        param_indices = jnp.concatenate(
            [param_indices, jnp.full((n_pad,), pad_index, param_indices.dtype)]
        )

    def accumulate(total, block):
        return total + block_fn(*block), None

    zero = jnp.zeros((l_t.shape[1], samples_t.shape[1]), dtype=_REAL_DTYPE)
    total, _ = jax.lax.scan(
        accumulate,
        zero,
        (
            supports.reshape(n_blocks, block_size, omega),
            powers.reshape(n_blocks, block_size, omega),
            param_indices.reshape(n_blocks, block_size),
        ),
    )
    return total


# pylint: disable=too-many-arguments
def _accumulate_phase_diffs(
    gates_params: ArrayLike,
    gate_groups: list[SparseGateGroup],
    samples: jnp.ndarray,
    l_vecs: jnp.ndarray,
    dims: ArrayLike,
    max_memory_bytes: int | None,
    vmapped_phase_func: Callable | None,
    phase_fn_params: ArrayLike | None,
) -> jnp.ndarray:
    """Assemble the accumulated phase-difference matrix from all weight groups."""
    n_samples = samples.shape[0]
    n_obs = l_vecs.shape[0]
    theta = jnp.asarray(gates_params).astype(_REAL_DTYPE)
    theta_ext = jnp.concatenate([theta, jnp.zeros((1,), dtype=_REAL_DTYPE)])
    samples_t = samples.astype(_REAL_DTYPE).T  # (n_qudits, n_samples)
    l_t = jnp.asarray(l_vecs).astype(_REAL_DTYPE).T  # (n_qudits, n_obs)
    dims_f = jnp.asarray(dims, dtype=_REAL_DTYPE)

    accumulated = jnp.zeros((n_obs, n_samples), dtype=_REAL_DTYPE)
    for group in gate_groups:
        if max_memory_bytes is None:
            accumulated = accumulated + _group_block_contribution(
                theta_ext,
                samples_t,
                l_t,
                dims_f,
                jnp.asarray(group.supports, dtype=_INDEX_DTYPE),
                jnp.asarray(group.powers, dtype=_INDEX_DTYPE),
                jnp.asarray(group.param_indices, dtype=_INDEX_DTYPE),
            )
        else:
            block_size = _qudit_phase_block_size(max_memory_bytes, group.omega, n_samples, n_obs)
            accumulated = accumulated + _group_phase_diffs(
                theta_ext, samples_t, l_t, dims_f, group, block_size
            )

    if vmapped_phase_func is not None:
        accumulated += vmapped_phase_func(phase_fn_params, samples, l_vecs).astype(_REAL_DTYPE)

    return accumulated


def _compute_initial_state_correction(
    samples: jnp.ndarray,
    l_f: jnp.ndarray,
    state_elems: ArrayLike,
    state_amps: ArrayLike,
    dims: ArrayLike,
) -> jnp.ndarray:
    """Compute the correction factor for a non-standard initial state."""
    s_f = samples.astype(_REAL_DTYPE)
    X_state = jnp.asarray(state_elems).astype(_REAL_DTYPE)  # (N, n)
    Psi = jnp.asarray(state_amps).astype(_COMPLEX_DTYPE)  # (N,)
    inv_d = (1.0 / jnp.asarray(dims, dtype=_REAL_DTYPE))[jnp.newaxis, :]  # (1, n)

    # ω^{Z·X^T} where ω_j = exp(2πi/d_j) — shape (s, N)
    omega_ZX = jnp.exp(2j * jnp.pi * ((s_f * inv_d) @ X_state.T)).astype(_COMPLEX_DTYPE)

    # Ψ̃^(2) = ω^{Z·X^T} · Ψ — shape (s,)
    psi_tilde_2 = omega_ZX @ Psi

    # F = Ψ* · 1_{1×s} ⊙ ω^{-X·Z^T} — shape (N, s)
    F_mat = Psi.conj()[:, jnp.newaxis] * omega_ZX.conj().T

    # Ψ̃^(1) = ω^{L·X^T} · F — shape (l, s)
    omega_LX = jnp.exp(2j * jnp.pi * ((l_f * inv_d) @ X_state.T)).astype(_COMPLEX_DTYPE)  # (l, N)
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
    max_memory_bytes = _memory_budget_bytes(config.max_memory)
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
        l_vecs = jnp.array(config.observables[0], dtype=_INDEX_DTYPE)
        m_vecs = jnp.array(config.observables[1], dtype=_INDEX_DTYPE)
        l_f = l_vecs.astype(_REAL_DTYPE)
        m_f = m_vecs.astype(_REAL_DTYPE)
        n_obs = l_vecs.shape[0]
        defaults = _PrecomputedObsData(
            l_vecs=l_vecs,
            n_obs=n_obs,
            l_f=l_f,
            m_f=m_f,
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
            l_vecs = jnp.array(observables[0], dtype=_INDEX_DTYPE)
            l_f = l_vecs.astype(_REAL_DTYPE)
            m_f = jnp.array(observables[1], dtype=_INDEX_DTYPE).astype(_REAL_DTYPE)
        elif defaults is not None:
            l_vecs, l_f, m_f = defaults.l_vecs, defaults.l_f, defaults.m_f
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
        else:
            obs_pm = _obs_phase_matrix(samples, m_f, l_f, dims)

        accumulated_phase_diffs = _accumulate_phase_diffs(
            gates_params,
            gate_groups,
            samples,
            l_vecs,
            dims,
            max_memory_bytes,
            vmapped_phase_func,
            phase_fn_params,
        )

        state_elems = config.init_state_elems if init_state_elems is None else init_state_elems
        state_amps = config.init_state_amps if init_state_amps is None else init_state_amps

        integrand = obs_pm * jnp.exp(1j * accumulated_phase_diffs).astype(_COMPLEX_DTYPE)
        if state_elems is not None and state_amps is not None:
            H = _compute_initial_state_correction(samples, l_f, state_elems, state_amps, dims)
            integrand = integrand * H

        return _compute_mc_statistics(integrand, _n)

    return qudit_expval_batched
