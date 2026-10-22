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

Implementation notes
--------------------

Every gate :math:`g` of weight :math:`\\omega` contributes the eigenvalue factor
:math:`\\phi_g(x) = \\prod_j \\sqrt{2}\\cos(2\\pi p_j x_{q_j}/d_{q_j} + \\pi/4)`.
Each cosine has two Fourier modes, so :math:`\\phi_g` is a sum of
:math:`2^{\\omega-1}` terms :math:`\\mathrm{Re}[c_s\\, \\omega^{K_s\\cdot x}]`
("modes"). The phase difference between a sample :math:`z` and the shifted
sample :math:`z - l` is therefore

.. math::

    \\Phi(z) - \\Phi(z - l) = \\mathrm{Re}\\sum_{g,s} \\theta_g c_s\\, \\omega^{K_s\\cdot z}
    \\left(1 - \\omega^{-K_s\\cdot l}\\right).

Writing :math:`\\omega^{-K_s\\cdot l} = \\prod_j (1 + \\delta_j)` with
:math:`\\delta_j = \\omega^{-k_j l_{q_j}} - 1`, the terms linear in
:math:`\\delta` only depend on one qudit at a time. They are collected into a
per-qudit table whose size is independent of the number of gates, so all
weight-one gates and the single-qudit parts of heavier gates cost one matrix
product with inner dimension :math:`\\sum_q (d_q - 1)`. The remaining terms
(two or more active qudits of a mode hit by the observable) form a dense
matrix product over the modes of the heavier gates.

All quantities that do not depend on the trainable parameters (random
samples, observable-side factors, the initial-state correction) are computed
once per PRNG key and observable batch and reused between the loss and its
gradient; the gradient uses a hand-written vector-Jacobian product so that only
the small backward program has to be compiled.
"""

import itertools
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

_REAL_DTYPE = jnp.float32
_COMPLEX_DTYPE = jnp.complex64
_INDEX_DTYPE = jnp.int32

# sqrt(2) * exp(+- i pi / 4) = 1 +- 1j: coefficient of the single mode of a weight-one gate
_W1_COEF = 1.0 + 1.0j


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
            memory is unbounded: the observable- and sample-side factor matrices are built once
            per PRNG key and observable batch and reused by every evaluation and gradient. If a
            budget is given and the precomputed matrices fit within it, the same path is used;
            otherwise the gates are processed in blocks whose size is derived from the budget
            together with the gate weight, the number of samples and the number of observables,
            so peak memory does not grow with the number of gates.

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


# ---------------------------------------------------------------------------
# Random samples
# ---------------------------------------------------------------------------


def _is_tracer(x: Any) -> bool:
    """Return ``True`` if ``x`` is an abstract JAX tracer (i.e. we are inside a transformation)."""
    return isinstance(x, jax.core.Tracer)


def _dims_tuple(dims: ArrayLike) -> tuple[int, ...]:
    """Normalize per-qudit dimensions to a hashable tuple of Python ints."""
    return tuple(int(d) for d in np.asarray(dims).reshape(-1))


@partial(jax.jit, static_argnums=(1, 2, 3))
def _compute_qudit_samples_flat(
    key: ArrayLike, num_samples: int, n_qudits: int, dims: tuple[int, ...]
) -> jnp.ndarray:
    """Draw ``num_samples * n_qudits`` dits as one flat array and reshape.

    Drawing a one-dimensional array keeps the PRNG program cheap to compile: XLA's constant
    folding of the two-dimensional counter grid used by the partitionable threefry
    implementation takes seconds for :math:`10^6` elements, whereas the flat draw compiles in
    tens of milliseconds.
    """
    n_total = num_samples * n_qudits
    if len(set(dims)) == 1:
        maxval = dims[0]
    else:
        maxval = jnp.tile(jnp.asarray(dims, dtype=_INDEX_DTYPE), num_samples)
    flat = jax.random.randint(key, (n_total,), 0, maxval, dtype=_INDEX_DTYPE)
    return flat.reshape(num_samples, n_qudits)


def _compute_qudit_samples(
    key: ArrayLike, num_samples: int, n_qudits: int, dims: ArrayLike
) -> jnp.ndarray:
    """Generates uniformly random dit-strings from the product Z_{d_1} x ... x Z_{d_n}."""
    return _compute_qudit_samples_flat(key, int(num_samples), int(n_qudits), _dims_tuple(dims))


# ---------------------------------------------------------------------------
# Memory budget helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Exact phase fractions
# ---------------------------------------------------------------------------

# Largest integer such that every integer below it is exactly representable in float32.
_F32_EXACT_INT = 2**24


def _column_chunks(dims: tuple[int, ...], scale: int) -> list[tuple[int, np.ndarray]]:
    """Split the qudits into groups of equal dimension and bounded exact integer range.

    Within a group the integer inner products :math:`\\sum_k a_k b_k` are accumulated in
    float32, which is exact as long as they stay below :math:`2^{24}`; the groups are therefore
    additionally chunked so that ``len(chunk) * (d - 1)**2`` never exceeds that bound.
    """
    dims_np = np.asarray(dims, dtype=int)
    chunks: list[tuple[int, np.ndarray]] = []
    for d in sorted(set(dims_np.tolist())):
        cols = np.flatnonzero(dims_np == d)
        max_cols = max(1, (_F32_EXACT_INT // max(1, scale * (d - 1) ** 2)))
        for start in range(0, cols.size, max_cols):
            chunks.append((int(d), cols[start : start + max_cols]))
    return chunks


def _phase_fraction(
    a: jnp.ndarray, b: jnp.ndarray, dims: tuple[int, ...], scale: int = 1
) -> jnp.ndarray:
    """Return ``frac[i, j] = (sum_k a[i, k] b[j, k] / (scale * d_k)) mod 1`` exactly.

    Both inputs are integer arrays with ``len(dims)`` columns. The inner products are formed
    per qudit dimension with exact float32 integer arithmetic and reduced modulo
    ``scale * d`` before the division, so the result does not lose precision when the
    number of qudits is large (a naive float32 ``(a / d) @ b.T`` accumulates an absolute
    error of order ``1e-3`` for a thousand qudits).
    """
    n = len(dims)
    total = None
    chunks = _column_chunks(dims, scale)
    for d, cols in chunks:
        if cols.size == n:
            af, bf = a, b
        else:
            af, bf = a[:, cols], b[:, cols]
        prod = af.astype(_REAL_DTYPE) @ bf.astype(_REAL_DTYPE).T
        mod = float(scale * d)
        frac = jnp.mod(prod, mod) / mod
        total = frac if total is None else total + frac
    if total is None:
        return jnp.zeros((a.shape[0], b.shape[0]), dtype=_REAL_DTYPE)
    return total if len(chunks) == 1 else jnp.mod(total, 1.0)


def _phase_fraction_rowwise(
    a: jnp.ndarray, b: jnp.ndarray, dims: tuple[int, ...], scale: int = 1
) -> jnp.ndarray:
    """Return ``frac[i] = (sum_k a[i, k] b[i, k] / (scale * d_k)) mod 1`` exactly."""
    n = len(dims)
    total = None
    chunks = _column_chunks(dims, scale)
    for d, cols in chunks:
        if cols.size == n:
            af, bf = a, b
        else:
            af, bf = a[:, cols], b[:, cols]
        prod = jnp.sum(af.astype(_REAL_DTYPE) * bf.astype(_REAL_DTYPE), axis=1)
        mod = float(scale * d)
        frac = jnp.mod(prod, mod) / mod
        total = frac if total is None else total + frac
    if total is None:
        return jnp.zeros((a.shape[0],), dtype=_REAL_DTYPE)
    return total if len(chunks) == 1 else jnp.mod(total, 1.0)


def _obs_phase_matrix(
    samples: jnp.ndarray, m_vecs: jnp.ndarray, l_vecs: jnp.ndarray, dims: ArrayLike
) -> jnp.ndarray:
    """Compute the observable phase matrix.

    :math:`J[i, j] = \\exp(i\\pi \\sum_k m_{ik} (2 z_{jk} - l_{ik}) / d_k)`.
    """
    dims_t = _dims_tuple(dims)
    frac_mz = _phase_fraction(m_vecs, samples, dims_t)  # (n_obs, n_samples)
    frac_ml = _phase_fraction_rowwise(m_vecs, l_vecs, dims_t, scale=2)  # (n_obs,)
    return jnp.exp((2j * jnp.pi) * (frac_mz - frac_ml[:, jnp.newaxis])).astype(_COMPLEX_DTYPE)


def _compute_initial_state_correction(
    samples: jnp.ndarray,
    l_vecs: jnp.ndarray,
    state_elems: ArrayLike,
    state_amps: ArrayLike,
    dims: ArrayLike,
) -> jnp.ndarray:
    """Compute the correction factor for a non-standard initial state.

    With :math:`\\tilde\\psi(y) = \\sum_x \\omega^{y \\cdot x} \\Psi_x` the correction is
    :math:`H[l, z] = \\overline{\\tilde\\psi(z - l)}\\, \\tilde\\psi(z)`.
    """
    dims_t = _dims_tuple(dims)
    X_state = jnp.asarray(state_elems).astype(_INDEX_DTYPE)  # (N, n)
    Psi = jnp.asarray(state_amps).astype(_COMPLEX_DTYPE)  # (N,)

    # ω^{Z·X^T} where ω_j = exp(2πi/d_j) — shape (s, N)
    omega_ZX = jnp.exp((2j * jnp.pi) * _phase_fraction(samples, X_state, dims_t)).astype(
        _COMPLEX_DTYPE
    )

    # Ψ̃^(2) = ω^{Z·X^T} · Ψ — shape (s,)
    psi_tilde_2 = omega_ZX @ Psi

    # F = Ψ* · 1_{1×s} ⊙ ω^{-X·Z^T} — shape (N, s)
    F_mat = Psi.conj()[:, jnp.newaxis] * omega_ZX.conj().T

    # Ψ̃^(1) = ω^{L·X^T} · F — shape (l, s)
    omega_LX = jnp.exp((2j * jnp.pi) * _phase_fraction(l_vecs, X_state, dims_t)).astype(
        _COMPLEX_DTYPE
    )
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


# ---------------------------------------------------------------------------
# Fourier-mode expansion of the gates
# ---------------------------------------------------------------------------


class _LookupTables(NamedTuple):
    """Roots of unity for every distinct qudit dimension, addressed by a flat integer index.

    ``table[i * d_max**2 + k * d_max + v] = exp(2 pi i k v / d_values[i])``.
    """

    d_values: tuple[int, ...]
    d_max: int
    dsel: np.ndarray  # (n_qudits,) index of each qudit's dimension in ``d_values``
    table: jnp.ndarray  # flat complex64 array
    index_dtype: Any  # integer dtype used for sample / observable values in gathers


def _build_lookup_tables(dims: np.ndarray) -> _LookupTables:
    """Tabulate :math:`\\exp(2\\pi i k v / d)` for all dimensions present in ``dims``."""
    d_values = tuple(sorted({int(d) for d in dims}))
    d_max = max(d_values) if d_values else 1
    lookup = {d: i for i, d in enumerate(d_values)}
    dsel = np.asarray([lookup[int(d)] for d in dims], dtype=np.int32)
    table = np.zeros((len(d_values), d_max, d_max), dtype=np.complex64)
    for i, d in enumerate(d_values):
        k = np.arange(d)[:, None]
        v = np.arange(d)[None, :]
        table[i, :d, :d] = np.exp(2j * np.pi * ((k * v) % d) / d)
    index_dtype = jnp.int8 if d_max <= 127 else _INDEX_DTYPE
    return _LookupTables(d_values, d_max, dsel, jnp.asarray(table.reshape(-1)), index_dtype)


class _ModeGroup(NamedTuple):
    """Fourier-mode data of all gates of one weight ``omega`` (static NumPy arrays)."""

    omega: int
    n_gates: int
    supports: np.ndarray  # (n_gates, omega) int32
    base_idx: np.ndarray  # (n_gates, omega) int32: dsel * d_max**2 + power * d_max
    param_indices: np.ndarray  # (n_gates,) int32
    signs: np.ndarray  # (n_modes, omega) int8, first column is +1
    coef: np.ndarray  # (n_modes,) complex64


class _ModeStructure(NamedTuple):
    """All static data derived from the gate dictionary."""

    n_params: int
    n_slots: int
    slot_base_idx: np.ndarray  # (n_slots,) int32: dsel * d_max**2 + k * d_max of each slot
    slot_qudit: np.ndarray  # (n_slots,) int32
    w1: _ModeGroup | None  # weight-one gates
    w1_slot_ab: np.ndarray | None  # (n_w1,) slot index, offset by n_slots for the conjugate mode
    heavy: list[_ModeGroup]  # weight >= 2 groups


def _build_mode_structure(
    gate_groups: list[SparseGateGroup], dims: np.ndarray, n_params: int, tables: _LookupTables
) -> _ModeStructure:
    """Precompute the mode expansion and slot bookkeeping for all gates."""
    dims = np.asarray(dims, dtype=int)
    n_qudits = dims.size
    d_max = tables.d_max

    # Per-qudit slots: one for each k in 1..floor(d/2); k and d-k share a slot via conjugation.
    n_per_qudit = dims // 2
    slot_offset = np.zeros(n_qudits, dtype=np.int64)
    if n_qudits > 1:
        slot_offset[1:] = np.cumsum(n_per_qudit)[:-1]
    n_slots = int(np.sum(n_per_qudit))
    slot_qudit = np.repeat(np.arange(n_qudits), n_per_qudit).astype(np.int32)
    slot_k = np.concatenate([np.arange(1, m + 1) for m in n_per_qudit]).astype(np.int32)
    slot_base_idx = (tables.dsel[slot_qudit].astype(np.int64) * d_max**2 + slot_k * d_max).astype(
        np.int32
    )

    w1 = None
    w1_slot_ab = None
    heavy: list[_ModeGroup] = []
    for group in gate_groups:
        omega = group.omega
        sup = np.asarray(group.supports, dtype=np.int64)
        pw = np.asarray(group.powers, dtype=np.int64)
        d_sup = dims[sup]
        pidx = np.asarray(group.param_indices, dtype=np.int32)
        n_g = sup.shape[0]

        signs = np.asarray(
            [(1,) + rest for rest in itertools.product([1, -1], repeat=omega - 1)], dtype=np.int8
        )  # (n_modes, omega)
        coef = (2.0 ** (1.0 - omega / 2.0) * np.exp(1j * np.pi / 4.0 * signs.sum(axis=1))).astype(
            np.complex64
        )

        base_idx = (tables.dsel[sup].astype(np.int64) * d_max**2 + pw * d_max).astype(np.int32)
        mode_group = _ModeGroup(
            omega=omega,
            n_gates=n_g,
            supports=sup.astype(np.int32),
            base_idx=base_idx,
            param_indices=pidx,
            signs=signs,
            coef=coef,
        )
        if omega == 1:
            w1 = mode_group
            # canonical k = min(p, d - p); gates with p > d - p enter through the conjugate mode
            kc = np.minimum(pw[:, 0], d_sup[:, 0] - pw[:, 0])
            slot = slot_offset[sup[:, 0]] + kc - 1
            is_beta = (pw[:, 0] != kc).astype(np.int64)
            w1_slot_ab = (slot + n_slots * is_beta).astype(np.int32)
        else:
            heavy.append(mode_group)

    return _ModeStructure(
        n_params=int(n_params),
        n_slots=n_slots,
        slot_base_idx=slot_base_idx,
        slot_qudit=slot_qudit,
        w1=w1,
        w1_slot_ab=w1_slot_ab,
        heavy=heavy,
    )


def _support_factors(
    x_t: jnp.ndarray, supports: ArrayLike, base_idx: ArrayLike, table: jnp.ndarray
) -> jnp.ndarray:
    """Per-position roots of unity ``F[g, j, :] = exp(2 pi i p_gj x[q_gj, :] / d_q)``.

    Args:
        x_t: Integer array of shape ``(n_qudits, N)`` (transposed samples or observables).
        supports: ``(B, omega)`` qudit indices.
        base_idx: ``(B, omega)`` flat table offsets ``dsel * d_max**2 + power * d_max``.
        table: Flat lookup table from :func:`_build_lookup_tables`.

    Returns:
        Complex array of shape ``(B, omega, N)``.
    """
    vals = jnp.take(x_t, jnp.asarray(supports), axis=0).astype(_INDEX_DTYPE)  # (B, omega, N)
    idx = jnp.asarray(base_idx, dtype=_INDEX_DTYPE)[:, :, jnp.newaxis] + vals
    return jnp.take(table, idx)


def _mode_products(factors: jnp.ndarray, signs: np.ndarray) -> jnp.ndarray:
    """``prod_j F_j`` (``s_j = +1``) or ``prod_j conj(F_j)`` (``s_j = -1``) for every sign pattern.

    Returns an array of shape ``(B, n_modes, N)``.
    """
    omega = factors.shape[1]
    conj = jnp.conj(factors)
    prods = []
    for s in signs:
        acc = factors[:, 0, :]
        for j in range(1, omega):
            acc = acc * (factors[:, j, :] if s[j] > 0 else conj[:, j, :])
        prods.append(acc)
    return jnp.stack(prods, axis=1)


def _heavy_obs_coefficients(
    group: _ModeGroup, l_t: jnp.ndarray, table: jnp.ndarray, supports, base_idx
) -> jnp.ndarray:
    """Observable-side coefficients ``1 - conj(Omega_s(l))`` of a block of weight >= 2 gates.

    ``Omega_s(l) = prod_j F_j^{s_j}(l)`` so that ``conj(Omega_s(l)) = omega^{-K_s . l}``. The
    coefficient vanishes for every observable that does not touch the support of the mode.

    Returns a real array ``(n_obs, 2 * B * n_modes)`` holding ``Re`` followed by ``-Im``, so that
    its product with the sample-side ``[Re V; Im V]`` yields ``Re(coef * V)``.
    """
    n_obs = l_t.shape[1]
    factors = _support_factors(l_t, supports, base_idx, table)
    prods = _mode_products(factors, group.signs)  # (B, n_modes, n_obs)
    coefs = 1.0 - jnp.conj(prods)
    coefs_t = coefs.reshape(-1, n_obs).T
    return jnp.concatenate([jnp.real(coefs_t), -jnp.imag(coefs_t)], axis=1)


def _heavy_sample_modes(
    group: _ModeGroup, samples_t: jnp.ndarray, table: jnp.ndarray, supports, base_idx
) -> jnp.ndarray:
    """Sample-side mode vectors ``c_s Omega_s(z)`` of a block of gates, as ``[Re; Im]`` rows.

    Returns a real array ``(2 * B * n_modes, n_samples)``.
    """
    n_samples = samples_t.shape[1]
    factors = _support_factors(samples_t, supports, base_idx, table)
    modes_c = _mode_products(factors, group.signs) * jnp.asarray(group.coef)[None, :, None]
    return jnp.concatenate(
        [jnp.real(modes_c).reshape(-1, n_samples), jnp.imag(modes_c).reshape(-1, n_samples)],
        axis=0,
    )


# ---------------------------------------------------------------------------
# Phase differences from precomputed factor matrices (memory permitting)
# ---------------------------------------------------------------------------


class _FactorIndices(NamedTuple):
    """Gather indices linking parameters to the rows of the precomputed factor matrices."""

    modes_pidx: np.ndarray | None  # (n_modes_real,) parameter index of each mode row


def _build_factor_indices(structure: _ModeStructure) -> _FactorIndices:
    """Concatenate the per-group bookkeeping in the order used by the factor matrices."""
    if not structure.heavy:
        return _FactorIndices(None)
    modes_pidx = []
    for group in structure.heavy:
        pidx_modes = np.repeat(group.param_indices, group.signs.shape[0])
        modes_pidx.append(np.concatenate([pidx_modes, pidx_modes]))
    return _FactorIndices(modes_pidx=np.concatenate(modes_pidx).astype(np.int32))


def _full_context_bytes(structure: _ModeStructure, n_obs: int, n_samples: int) -> int:
    """Estimate the memory held by the precomputed factor matrices (plus transients)."""
    n_modes = sum(2 * g.n_gates * g.signs.shape[0] for g in structure.heavy)
    stored = _FLOAT_BYTES * (n_obs + n_samples) * n_modes
    if structure.w1 is not None:
        stored += _FLOAT_BYTES * structure.n_slots * (2 * n_obs + 2 * n_samples)
    stored += 2 * _FLOAT_BYTES * n_obs * n_samples
    return 2 * stored


def _linear_weights_w1(theta_ext: jnp.ndarray, structure: _ModeStructure) -> jnp.ndarray:
    """Complex per-slot weight of the weight-one gates: ``sqrt(2) (alpha e^{i pi/4} + beta e^{-i pi/4})``."""
    w = jax.ops.segment_sum(
        theta_ext[structure.w1.param_indices],
        structure.w1_slot_ab,
        num_segments=2 * structure.n_slots,
    )
    alpha, beta = w[: structure.n_slots], w[structure.n_slots :]
    return (alpha + beta) + 1j * (alpha - beta)


def _phase_diffs_from_factors(
    theta_ext: jnp.ndarray, ctx: dict, structure: _ModeStructure, indices: _FactorIndices
) -> jnp.ndarray | None:
    """Accumulated phase differences ``(n_obs, n_samples)`` from the precomputed factors.

    Weight-one gates enter through the per-qudit slot tables (inner dimension
    ``2 * n_slots``); all heavier gates through their modes (inner dimension
    ``2 * n_modes``).
    """
    delta = None
    if structure.w1 is not None:
        w_c = _linear_weights_w1(theta_ext, structure)
        ar = w_c[:, jnp.newaxis] * ctx["slot_samples"]  # (n_slots, n_samples) complex
        delta = ctx["slot_obs"] @ jnp.concatenate([jnp.real(ar), jnp.imag(ar)], axis=0)
    if indices.modes_pidx is not None:
        v = theta_ext[indices.modes_pidx][:, jnp.newaxis] * ctx["modes"]
        heavy = ctx["coef_obs"] @ v
        delta = heavy if delta is None else delta + heavy
    return delta


def _phase_diffs_vjp_from_factors(
    g_delta: jnp.ndarray, ctx: dict, structure: _ModeStructure, indices: _FactorIndices
) -> jnp.ndarray:
    """Hand-written transpose of :func:`_phase_diffs_from_factors` with respect to ``theta_ext``."""
    dtheta = jnp.zeros((structure.n_params + 1,), dtype=_REAL_DTYPE)
    if structure.w1 is not None:
        n_slots = structure.n_slots
        d_ar = ctx["slot_obs"].T @ g_delta  # (2 n_slots, n_samples)
        base = ctx["slot_samples"]
        # ar = w_c * base with w_c = (alpha + beta) + i (alpha - beta)
        d_re = jnp.sum(d_ar[:n_slots] * jnp.real(base) + d_ar[n_slots:] * jnp.imag(base), axis=1)
        d_im = jnp.sum(-d_ar[:n_slots] * jnp.imag(base) + d_ar[n_slots:] * jnp.real(base), axis=1)
        d_ab = jnp.concatenate([d_re + d_im, d_re - d_im])
        dtheta = dtheta.at[structure.w1.param_indices].add(d_ab[structure.w1_slot_ab])
    if indices.modes_pidx is not None:
        d_v = ctx["coef_obs"].T @ g_delta  # (n_modes_real, n_samples)
        dtheta = dtheta.at[indices.modes_pidx].add(jnp.sum(d_v * ctx["modes"], axis=1))
    return dtheta


# ---------------------------------------------------------------------------
# Memory-bounded blocked phase differences (fallback)
# ---------------------------------------------------------------------------


def _group_blocked_phase_diffs(
    theta_ext: jnp.ndarray,
    operands: tuple[jnp.ndarray, jnp.ndarray],
    group: _ModeGroup,
    tables: _LookupTables,
    block_size: int,
) -> jnp.ndarray:
    """Phase differences of one weight group, processed in memory-bounded blocks.

    Each block is wrapped in :func:`jax.checkpoint` so that reverse-mode differentiation
    recomputes the block's factors instead of storing them.

    Args:
        theta_ext: Gate parameters with a trailing zero appended, shape ``(n_params + 1,)``.
        operands: Transposed integer samples ``(n_qudits, n_samples)`` and transposed observable
            ``l`` vectors ``(n_qudits, n_obs)``.
        group: Mode data of the gates in this weight group.
        tables: Roots-of-unity lookup tables.
        block_size: Number of gates per block.
    """
    samples_t, l_t = operands
    n_samples, n_obs = samples_t.shape[1], l_t.shape[1]
    n_g, omega = group.supports.shape
    n_modes = group.signs.shape[0]

    def block_fn(sup, base, pidx):
        coefs = _heavy_obs_coefficients(group, l_t, tables.table, sup, base)
        modes = _heavy_sample_modes(group, samples_t, tables.table, sup, base)
        theta_modes = jnp.tile(jnp.repeat(theta_ext[pidx], n_modes), 2)
        return coefs @ (theta_modes[:, jnp.newaxis] * modes)

    block_fn = jax.checkpoint(block_fn)
    supports = jnp.asarray(group.supports, dtype=_INDEX_DTYPE)
    base_idx = jnp.asarray(group.base_idx, dtype=_INDEX_DTYPE)
    pidx = jnp.asarray(group.param_indices, dtype=_INDEX_DTYPE)

    if n_g <= block_size:
        return block_fn(supports, base_idx, pidx)

    n_blocks = -(-n_g // block_size)
    n_pad = n_blocks * block_size - n_g
    if n_pad:
        pad_index = theta_ext.shape[0] - 1  # points at the appended zero parameter
        pad_base = int(tables.dsel[0]) * tables.d_max**2 + 1 * tables.d_max
        supports = jnp.concatenate([supports, jnp.zeros((n_pad, omega), supports.dtype)])
        base_idx = jnp.concatenate([base_idx, jnp.full((n_pad, omega), pad_base, base_idx.dtype)])
        pidx = jnp.concatenate([pidx, jnp.full((n_pad,), pad_index, pidx.dtype)])

    def accumulate(acc, block):
        return acc + block_fn(*block), None

    zero = jnp.zeros((n_obs, n_samples), dtype=_REAL_DTYPE)
    total, _ = jax.lax.scan(
        accumulate,
        zero,
        (
            supports.reshape(n_blocks, block_size, omega),
            base_idx.reshape(n_blocks, block_size, omega),
            pidx.reshape(n_blocks, block_size),
        ),
    )
    return total


def _blocked_phase_diffs(
    theta_ext: jnp.ndarray,
    operands: tuple[jnp.ndarray, jnp.ndarray],
    structure: _ModeStructure,
    tables: _LookupTables,
    max_memory_bytes: int,
) -> jnp.ndarray | None:
    """Phase differences of all gates computed block by block without stored factor matrices."""
    samples_t, l_t = operands
    n_samples, n_obs = samples_t.shape[1], l_t.shape[1]
    groups = ([structure.w1] if structure.w1 is not None else []) + list(structure.heavy)
    total = None
    for group in groups:
        block_size = _qudit_phase_block_size(max_memory_bytes, group.omega, n_samples, n_obs)
        contribution = _group_blocked_phase_diffs(theta_ext, operands, group, tables, block_size)
        total = contribution if total is None else total + contribution
    return total


# ---------------------------------------------------------------------------
# Cache keys for parameter-independent precomputation
# ---------------------------------------------------------------------------


def _array_token(x: Any) -> Any:
    """Hashable identity of a concrete array for the precomputation cache.

    Immutable JAX arrays are identified by object identity (the cache keeps a reference so
    the identity cannot be recycled); NumPy arrays and other array-likes by content.
    """
    if x is None:
        return None
    if isinstance(x, jax.Array):
        return ("id", id(x))
    arr = np.asarray(x)
    return ("np", arr.shape, str(arr.dtype), hash(arr.tobytes()))


def _key_token(key: Any) -> Any:
    """Hashable content of a concrete PRNG key."""
    if jnp.issubdtype(jnp.asarray(key).dtype, jax.dtypes.prng_key):
        key = jax.random.key_data(key)
    return ("key", np.asarray(key).tobytes())


def build_qudit_expval_func(  # pylint: disable=too-many-statements,too-many-locals
    config: QuditCircuitConfig,
) -> Callable:
    """Build an estimator for expectation values of a qudit IQP circuit.

    Returns a pure function that estimates the complex expectation value
    :math:`\\langle O(\\mathbf{l}, \\mathbf{m}) \\rangle` for each
    observable by averaging over randomly sampled dit-strings.

    The returned function captures precomputed data from ``config`` (generator
    matrices, default samples, preprocessed observables) so that repeated
    evaluations with different parameters are fast. Everything that does not
    depend on the trainable parameters (random samples, observable-side
    factors, the initial-state correction) is computed once per PRNG key and
    observable batch and reused by subsequent evaluations and gradients with
    the same key and observables.

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
    n_params = (max(int(k) for k in config.gates.keys()) + 1) if config.gates else 0
    max_memory_bytes = _memory_budget_bytes(config.max_memory)
    tables = _build_lookup_tables(dims)
    structure = _build_mode_structure(gate_groups, dims, n_params, tables)
    indices = _build_factor_indices(structure)
    has_gates = structure.w1 is not None or bool(structure.heavy)

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

    # ---- static tokens of the configuration-level arrays (used in cache keys) ----
    config_state_token = (
        _array_token(config.init_state_elems),
        _array_token(config.init_state_amps),
    )
    config_obs_token = (
        None
        if config.observables is None
        else (_array_token(config.observables[0]), _array_token(config.observables[1]))
    )
    default_samples_cache: dict[str, jnp.ndarray] = {}

    def default_samples() -> jnp.ndarray:
        if "samples" not in default_samples_cache:
            default_samples_cache["samples"] = _compute_qudit_samples(
                config.key, config.n_samples, n, dims
            )
        return default_samples_cache["samples"]

    # ---- parameter-independent context -------------------------------------------------
    def _phase_layer(phase_params, samples, l_vecs):
        return vmapped_phase_func(phase_params, samples, l_vecs).astype(_REAL_DTYPE)

    def _obs_state_product(samples, l_vecs, m_vecs, state_elems, state_amps):
        """``J * H`` (or ``None`` if neither the m-phase nor an initial state is present)."""
        jh = None
        if m_vecs is not None:
            jh = _obs_phase_matrix(samples, m_vecs, l_vecs, dims)
        if state_elems is not None and state_amps is not None:
            h = _compute_initial_state_correction(samples, l_vecs, state_elems, state_amps, dims)
            jh = h if jh is None else jh * h
        return jh

    @jax.jit
    def prepare_full(samples, l_vecs, m_vecs, state_elems, state_amps):
        """Precompute every parameter-independent factor matrix."""
        samples_t = samples.T.astype(tables.index_dtype)
        l_t = l_vecs.T.astype(tables.index_dtype)
        ctx = {"samples": samples, "l_vecs": l_vecs}
        if structure.w1 is not None:
            base = jnp.asarray(structure.slot_base_idx, dtype=_INDEX_DTYPE)[:, jnp.newaxis]
            qudits = jnp.asarray(structure.slot_qudit, dtype=_INDEX_DTYPE)
            t_l = jnp.take(tables.table, base + jnp.take(l_t, qudits, axis=0).astype(_INDEX_DTYPE))
            # Dr = [1 - Re T_l, -Im T_l] so that Dr @ [Re A; Im A] = -Re(delta_slot(l) A(z))
            ctx["slot_obs"] = jnp.concatenate([1.0 - jnp.real(t_l), -jnp.imag(t_l)], axis=0).T
            ctx["slot_samples"] = jnp.take(
                tables.table, base + jnp.take(samples_t, qudits, axis=0).astype(_INDEX_DTYPE)
            )
        if structure.heavy:
            modes, coefs = [], []
            for group in structure.heavy:
                modes.append(
                    _heavy_sample_modes(
                        group, samples_t, tables.table, group.supports, group.base_idx
                    )
                )
                coefs.append(
                    _heavy_obs_coefficients(
                        group, l_t, tables.table, group.supports, group.base_idx
                    )
                )
            ctx["modes"] = jnp.concatenate(modes, axis=0)
            ctx["coef_obs"] = jnp.concatenate(coefs, axis=1)
        ctx["jh"] = _obs_state_product(samples, l_vecs, m_vecs, state_elems, state_amps)
        return ctx

    @jax.jit
    def prepare_lite(samples, l_vecs, m_vecs, state_elems, state_amps):
        """Context for the memory-bounded path: only the sample / observable products."""
        return {
            "samples": samples,
            "l_vecs": l_vecs,
            "jh": _obs_state_product(samples, l_vecs, m_vecs, state_elems, state_amps),
        }

    def _tail(delta, jh, n_samples, n_obs):
        """Integrand and Monte Carlo statistics from the accumulated phase differences."""
        if delta is None:
            integrand = jh if jh is not None else jnp.ones((n_obs, n_samples), dtype=_COMPLEX_DTYPE)
        else:
            integrand = jnp.exp(1j * delta).astype(_COMPLEX_DTYPE)
            if jh is not None:
                integrand = integrand * jh
        return _compute_mc_statistics(integrand, n_samples)

    def _theta_ext(gates_params):
        theta = jnp.asarray(gates_params).astype(_REAL_DTYPE).reshape(-1)
        return jnp.concatenate([theta, jnp.zeros((1,), dtype=_REAL_DTYPE)])

    # ---- full path: precomputed factors, hand-written VJP ----------------------------
    @jax.jit
    def core_full_fwd(gates_params, phase_params, ctx):
        theta_ext = _theta_ext(gates_params)
        n_samples, n_obs = ctx["samples"].shape[0], ctx["l_vecs"].shape[0]
        delta = _phase_diffs_from_factors(theta_ext, ctx, structure, indices)
        if vmapped_phase_func is not None:
            layer = _phase_layer(phase_params, ctx["samples"], ctx["l_vecs"])
            delta = layer if delta is None else delta + layer
        return _tail(delta, ctx["jh"], n_samples, n_obs), delta

    @jax.jit
    def core_full_bwd(gates_params, phase_params, ctx, delta, ct):
        n_samples, n_obs = ctx["samples"].shape[0], ctx["l_vecs"].shape[0]
        dtheta_ext = jnp.zeros((n_params + 1,), dtype=_REAL_DTYPE)
        dphase = None
        if delta is not None:
            _, tail_vjp = jax.vjp(lambda d: _tail(d, ctx["jh"], n_samples, n_obs), delta)
            (g_delta,) = tail_vjp(ct)
            dtheta_ext = _phase_diffs_vjp_from_factors(g_delta, ctx, structure, indices)
            if vmapped_phase_func is not None and phase_params is not None:
                _, layer_vjp = jax.vjp(
                    lambda p: _phase_layer(p, ctx["samples"], ctx["l_vecs"]), phase_params
                )
                (dphase,) = layer_vjp(g_delta)
        gp = jnp.asarray(gates_params)
        dtheta = dtheta_ext[:n_params].reshape(gp.shape).astype(gp.dtype)
        return dtheta, dphase

    # The backward program is compiled ahead of time during the first (forward-only) evaluation:
    # its input signature is fully determined by the forward outputs, and a training loss is
    # differentiated almost always, so this hides the compile latency of the first gradient.
    compiled_bwd: dict[Any, Any] = {}

    def _bwd_signature(*args):
        leaves, treedef = jax.tree.flatten(args)
        return treedef, tuple((tuple(leaf.shape), str(leaf.dtype)) for leaf in leaves)

    def _warm_backward(gates_params, phase_params, ctx, delta, out):
        if any(_is_tracer(x) for x in jax.tree.leaves((gates_params, phase_params, delta, out))):
            return
        ct_abstract = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), out)
        args = (gates_params, phase_params, ctx, delta, ct_abstract)
        key = _bwd_signature(*args)
        if key not in compiled_bwd:
            compiled_bwd[key] = core_full_bwd.lower(*args).compile()

    # The most recent forward evaluation is memoized on the identity of its (immutable) inputs,
    # so that a gradient evaluated right after the loss at the same parameters reuses the
    # forward pass instead of recomputing it.
    forward_memo: dict[str, Any] = {}

    def _forward(gates_params, phase_params, ctx):
        params_leaves = jax.tree.leaves((gates_params, phase_params))
        if any(_is_tracer(x) for x in params_leaves):
            return core_full_fwd(gates_params, phase_params, ctx)
        # ``jax.custom_vjp`` re-creates the ``ctx`` container on every call; its array leaves are
        # the cached objects, so they (not the dict) identify the context.
        token = (
            tuple(_array_token(x) for x in params_leaves),
            jax.tree.structure((gates_params, phase_params)),
            tuple(id(leaf) for leaf in jax.tree.leaves(ctx)),
        )
        entry = forward_memo.get("entry")
        if entry is not None and entry[0] == token:
            return entry[2]
        result = core_full_fwd(gates_params, phase_params, ctx)
        forward_memo["entry"] = (token, (gates_params, phase_params, ctx), result)
        return result

    @jax.custom_vjp
    def core_full(gates_params, phase_params, ctx):
        out, delta = _forward(gates_params, phase_params, ctx)
        _warm_backward(gates_params, phase_params, ctx, delta, out)
        return out

    def core_full_fwd_rule(gates_params, phase_params, ctx):
        out, delta = _forward(gates_params, phase_params, ctx)
        return out, (gates_params, phase_params, ctx, delta)

    def core_full_bwd_rule(res, ct):
        gates_params, phase_params, ctx, delta = res
        args = (gates_params, phase_params, ctx, delta, ct)
        compiled = None
        if not any(_is_tracer(x) for x in jax.tree.leaves(args)):
            compiled = compiled_bwd.get(_bwd_signature(*args))
        dtheta, dphase = (compiled or core_full_bwd)(*args)
        return dtheta, dphase, None

    core_full.defvjp(core_full_fwd_rule, core_full_bwd_rule)

    # ---- blocked path: memory-bounded, plain autodiff ----------------------------------
    @jax.jit
    def core_blocked(gates_params, phase_params, ctx):
        theta_ext = _theta_ext(gates_params)
        samples, l_vecs = ctx["samples"], ctx["l_vecs"]
        n_samples, n_obs = samples.shape[0], l_vecs.shape[0]
        delta = None
        if has_gates:
            operands = (samples.T.astype(tables.index_dtype), l_vecs.T.astype(tables.index_dtype))
            delta = _blocked_phase_diffs(theta_ext, operands, structure, tables, max_memory_bytes)
        if vmapped_phase_func is not None:
            layer = _phase_layer(phase_params, samples, l_vecs)
            delta = layer if delta is None else delta + layer
        return _tail(delta, ctx["jh"], n_samples, n_obs)

    context_cache: dict[str, Any] = {}

    def _resolve_observables(observables):
        """Raw ``(l, m)`` arrays and their cache token."""
        if observables is not None:
            l_raw, m_raw = observables
            return l_raw, m_raw, (_array_token(l_raw), _array_token(m_raw))
        if config.observables is not None:
            l_raw, m_raw = config.observables
            return l_raw, m_raw, config_obs_token
        raise ValueError(
            "No observables specified. Provide them in QuditCircuitConfig "
            "or pass at call time via the observables argument."
        )

    def _resolve_state(init_state_elems, init_state_amps):
        """Initial-state arrays (``None`` unless both are present) and their cache token."""
        if init_state_elems is None and init_state_amps is None:
            elems, amps, token = config.init_state_elems, config.init_state_amps, config_state_token
        else:
            elems = config.init_state_elems if init_state_elems is None else init_state_elems
            amps = config.init_state_amps if init_state_amps is None else init_state_amps
            token = (_array_token(elems), _array_token(amps))
        if elems is None or amps is None:
            return None, None, (None, None)
        return elems, amps, token

    def _cached_context(tokens, refs, build):
        """Return the cached parameter-independent context, rebuilding it if the inputs changed.

        The build runs eagerly even inside an outer ``jax.jit`` so that the result can be cached
        and reused across calls (and across loss and gradient evaluations). ``refs`` keeps the
        arrays behind the identity tokens alive so that an object id cannot be recycled by a
        different array. Returns ``None`` if the context could not be made concrete.
        """
        entry = context_cache.get("entry")
        if entry is not None and entry[0] == tokens:
            return entry[2]
        with jax.ensure_compile_time_eval():
            ctx = build()
        if any(_is_tracer(leaf) for leaf in jax.tree.leaves(ctx)):
            return None
        context_cache["entry"] = (tokens, refs, ctx)
        return ctx

    def _concrete_m_vecs(m_raw):
        """``None`` for a concrete all-zero ``m`` (no observable phase), else the int array."""
        if _is_tracer(m_raw):
            return jnp.asarray(m_raw, dtype=_INDEX_DTYPE)
        m_np = np.asarray(m_raw)
        return jnp.asarray(m_np, dtype=_INDEX_DTYPE) if m_np.any() else None

    def qudit_expval_batched(  # pylint: disable=too-many-arguments,too-many-locals
        gates_params: ArrayLike,
        phase_fn_params: ArrayLike | None = None,
        key: ArrayLike | None = None,
        n_samples: int | None = None,
        observables: tuple[ArrayLike, ArrayLike] | None = None,
        init_state_elems: ArrayLike | None = None,
        init_state_amps: ArrayLike | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
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
        l_raw, m_raw, obs_token = _resolve_observables(observables)
        state_elems, state_amps, state_token = _resolve_state(init_state_elems, init_state_amps)
        _n = int(n_samples) if n_samples is not None else config.n_samples
        explicit_key = key is not None or n_samples is not None
        _key = (key if key is not None else config.key) if explicit_key else None
        n_obs = int(np.shape(l_raw)[0])
        use_full = (
            max_memory_bytes is None
            or not has_gates
            or _full_context_bytes(structure, n_obs, _n) <= max_memory_bytes
        )
        prepare = prepare_full if use_full else prepare_lite

        def build_context():
            """Array conversions, sampling and the parameter-independent factor matrices."""
            l_vecs = jnp.asarray(l_raw, dtype=_INDEX_DTYPE)
            m_vecs = _concrete_m_vecs(m_raw)
            samples = (
                default_samples() if _key is None else _compute_qudit_samples(_key, _n, n, dims)
            )
            return prepare(samples, l_vecs, m_vecs, state_elems, state_amps)

        concrete = not any(_is_tracer(x) for x in (key, l_raw, m_raw, state_elems, state_amps))
        if concrete:
            sample_token = ("default", _n) if _key is None else (_key_token(_key), _n)
            tokens = (sample_token, obs_token, state_token, use_full)
            ctx = _cached_context(tokens, (l_raw, m_raw, state_elems, state_amps), build_context)
            if ctx is not None:
                core = core_full if use_full else core_blocked
                return core(gates_params, phase_fn_params, ctx)

        ctx = build_context()
        if use_full:
            return core_full_fwd(gates_params, phase_fn_params, ctx)[0]
        return core_blocked(gates_params, phase_fn_params, ctx)

    # Internal programs, exposed for diagnostics and benchmarking (not part of the public API).
    qudit_expval_batched.internals = {
        "prepare_full": prepare_full,
        "prepare_lite": prepare_lite,
        "core_full_fwd": core_full_fwd,
        "core_full_bwd": core_full_bwd,
        "core_blocked": core_blocked,
        "context_cache": context_cache,
        "forward_memo": forward_memo,
    }
    return qudit_expval_batched
