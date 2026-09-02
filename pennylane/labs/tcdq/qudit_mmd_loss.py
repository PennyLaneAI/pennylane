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
"""Maximum Mean Discrepancy (MMD) loss for qudit circuits.

This module extends :mod:`~pennylane.labs.tcdq.mmd_loss` from qubits to
qudits. It compares the circuit output to a dataset by sampling observables,
estimating their moments, and combining those estimates into an unbiased MMD
loss.

For qudits, the kernel is defined from a graph on one qudit level set. The
available choices are the cycle graph :math:`C_d` and the complete graph
:math:`K_d`.

The loss consumes an :class:`~pennylane.labs.tcdq.Estimator` rather than a
particular simulator, so it works with any
:class:`~pennylane.labs.tcdq.TCDQSimulator` that provides a Heisenberg-Weyl
estimator.

For the mathematical construction, see
`Section IV B of Spectral Born machines: classically trainable quantum generative models for discrete data <https://arxiv.org/abs/2607.06675>`_.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from .base import Estimator, ObservableAlgebra
from .mmd_loss import _resolve_bandwidths, _resolve_wires, _validate_target_data
from .qudit_iqp import QuditCircuitConfig, _simulator_from_config


@dataclass(frozen=True)
class QuditMMDConfig:
    r"""Hyperparameters for the qudit graph-kernel MMD loss.

    The MMD measures how well the circuit output matches a target dataset of
    dit-strings. In the qudit setting, the kernel comes from heat diffusion on
    a graph over the local levels of one qudit, applied independently to each
    visible wire.

    Args:
        bandwidth (float | Sequence[float]): The bandwidth :math:`\sigma^2` of the kernel. If a sequence is provided,
            the loss is evaluated for each value and then averaged, unless
            ``return_per_bandwidth=True``.
        n_ops (int): Number of sampled observables per bandwidth. Larger
            values reduce estimator variance.
        graph_type (str): Graph whose spectrum defines the kernel.
            ``"cycle"`` is usually the better default when neighbouring qudit
            levels have a natural notion of closeness. ``"complete"`` treats
            all distinct levels symmetrically. Defaults to ``"cycle"``.
        wires (Sequence[int] | None): Subset of qudit indices to include in
            the loss. If ``None`` (default), all qudits are used.
        sqrt_loss (bool): If ``True``, return ``sqrt(|MMD²|)`` instead of
            ``MMD²``. Defaults to ``False``.
        return_per_bandwidth (bool): If ``True``, return a list of
            per-bandwidth loss values instead of their scalar average.
            Defaults to ``False``.

    **Example**

    >>> from pennylane.labs.tcdq import QuditMMDConfig
    >>> config = QuditMMDConfig(bandwidth=[0.3, 1.0], n_ops=64, graph_type="cycle")
    """

    #: Width of the graph heat kernel (scalar or sequence for multi-bandwidth).
    bandwidth: float | Sequence[float] = None
    #: Number of sampled observables per bandwidth.
    n_ops: int = None
    #: Graph whose spectrum defines the kernel (``"cycle"`` or ``"complete"``).
    graph_type: str = "cycle"
    #: Subset of qudit indices to include, or ``None`` for all qudits.
    wires: Sequence[int] | None = None
    #: If ``True``, return ``sqrt(|MMD²|)`` instead of ``MMD²``.
    sqrt_loss: bool = False
    #: If ``True``, return per-bandwidth losses instead of their average.
    return_per_bandwidth: bool = False


def _cycle_marginal_probs(d: int, t: float) -> jnp.ndarray:
    """Return the per-site sampling distribution for the cycle-graph heat kernel.

    The probability of sampling index :math:`k` on a single qudit is
    proportional to :math:`\\exp(-4t \\sin^2(\\pi k / d))`, which are the
    eigenvalues of the heat kernel on the cycle graph :math:`C_d`.

    For the derivation, see
    `Section IV B of Spectral Born machines: classically trainable quantum generative models for discrete data <https://arxiv.org/abs/2607.06675>`_.
    """
    k = jnp.arange(d)
    log_p = -4.0 * t * jnp.sin(jnp.pi * k / d) ** 2
    p = jnp.exp(log_p)
    return p / jnp.sum(p)


def _complete_marginal_probs(d: int, t: float) -> jnp.ndarray:
    """Return the per-site sampling distribution for the complete-graph heat kernel.

    The complete graph :math:`K_d` has only two distinct eigenvalues,
    yielding a binary distribution: index 0 has elevated probability and all
    other indices share the remaining mass equally.

    For the derivation, see
    `Section IV B of Spectral Born machines: classically trainable quantum generative models for discrete data <https://arxiv.org/abs/2607.06675>`_.
    """
    log_unnorm = jnp.zeros(d).at[1:].set(-t * d)
    p = jnp.exp(log_unnorm)
    return p / jnp.sum(p)


def _marginal_probs(d: int, bandwidth: float, graph_type: str) -> jnp.ndarray:
    """Dispatch to the per-site heat-kernel marginal for a single qudit of dimension ``d``."""
    if graph_type == "cycle":
        return _cycle_marginal_probs(d, bandwidth)
    if graph_type == "complete":
        return _complete_marginal_probs(d, bandwidth)
    raise ValueError(f"Unknown graph_type {graph_type!r}; use 'cycle' or 'complete'.")


def _sample_fourier_indices(  # pylint: disable=too-many-arguments
    key: ArrayLike,
    n_ops: int,
    n_qudits: int,
    dims: tuple[int, ...],
    bandwidth: float,
    graph_type: str,
    wire_tuple: tuple[int, ...],
) -> jnp.ndarray:
    """Sample Fourier index vectors from the graph-kernel spectral distribution.

    Draws ``n_ops`` vectors from the product distribution
    :math:`P(\\mathbf{l}) = \\prod_i P_1(l_i)` where :math:`P_1` is the per-site
    heat kernel on a graph over that qudit's :math:`d_i` levels. Positions outside
    ``wire_tuple`` are zero.

    Args:
        dims (tuple[int, ...]): Per-qudit dimensions, length ``n_qudits``.

    Returns:
        Integer array of shape ``(n_ops, n_qudits)``; column ``i`` has entries
        in :math:`\\{0, \\ldots, d_i-1\\}`.
    """
    all_obs = jnp.zeros((n_ops, n_qudits), dtype=jnp.int32)
    keys = jax.random.split(key, len(wire_tuple)) if wire_tuple else []
    for col_key, wire in zip(keys, wire_tuple):
        d_i = int(dims[wire])
        marginal = _marginal_probs(d_i, bandwidth, graph_type)
        col = jax.random.choice(col_key, d_i, shape=(n_ops,), p=marginal)
        all_obs = all_obs.at[:, wire].set(col.astype(jnp.int32))
    return all_obs


def _empirical_fourier_moments(
    l_visible: jnp.ndarray,
    X_data: jnp.ndarray,
    dims_visible: jnp.ndarray,
) -> jnp.ndarray:
    """Compute the empirical Fourier moment for each sampled observable from the dataset.

    For each Fourier index vector :math:`\\mathbf{l}`, computes
    :math:`\\hat{\\mu}_p(\\mathbf{l}) = \\frac{1}{m} \\sum_i \\exp(2\\pi i \\sum_k l_k x_{ik} / d_k)`,
    i.e. the per-qudit root of unity :math:`\\omega_k = e^{2\\pi i / d_k}`. The
    per-visible-wire dimension is folded in by column-scaling ``l_visible`` with
    ``1 / dims_visible``.

    Args:
        l_visible: Integer array of shape ``(n_obs, n_visible)`` — the Fourier
            index vectors restricted to the visible wires.
        X_data: Integer array of shape ``(m, n_visible)`` — target dataset
            samples on the visible wires.
        dims_visible: Integer array of shape ``(n_visible,)`` — dimension of
            each visible qudit.

    Returns:
        Complex array of shape ``(n_obs,)``.
    """
    inv_d = 1.0 / jnp.asarray(dims_visible, dtype=jnp.float64)
    l_scaled = l_visible.astype(jnp.float64) * inv_d[jnp.newaxis, :]
    inner = l_scaled @ X_data.astype(jnp.float64).T
    return jnp.mean(jnp.exp(2j * jnp.pi * inner), axis=1)


def _pp_term(mu_p_hat: jnp.ndarray, m: int) -> jnp.ndarray:
    """Compute the unbiased data–data U-statistic contribution to the MMD.

    Removes the diagonal self-pairs from :math:`|\\hat{\\mu}_p|^2`:
    :math:`PP(l) = (m |\\hat{\\mu}_p(l)|^2 - 1) / (m - 1)`.

    Args:
        mu_p_hat: Complex array of shape ``(n_obs,)`` — empirical data moments.
        m: Number of samples in the dataset.

    Returns:
        Real array of shape ``(n_obs,)``.
    """
    return (m * jnp.abs(mu_p_hat) ** 2 - 1.0) / (m - 1)


def _qq_term(
    mu_q_hat: jnp.ndarray,
    cov: jnp.ndarray,
) -> jnp.ndarray:
    """Compute the unbiased model–model U-statistic contribution to the MMD.

    Removes the estimated variance of the complex sample mean from
    :math:`|\\hat{\\mu}_q|^2`.

    Args:
        mu_q_hat: Complex array of shape ``(n_obs,)`` — circuit-side Monte
            Carlo moment estimates.
        cov: Real array of shape ``(n_obs, 2, 2)`` — covariance matrices of
            the real and imaginary parts of the estimated moments.

    Returns:
        Real array of shape ``(n_obs,)``.
    """
    variances = jnp.trace(cov, axis1=-2, axis2=-1)
    return jnp.abs(mu_q_hat) ** 2 - variances


def _pq_cross_term(
    mu_p_hat: jnp.ndarray,
    mu_q_hat: jnp.ndarray,
) -> jnp.ndarray:
    """Compute the data–model cross term of the MMD.

    :math:`PQ(l) = 2 \\operatorname{Re}(\\hat{\\mu}_p(l)^* \\hat{\\mu}_q(l))`.
    No diagonal correction is needed because the data and circuit samples are
    independent.

    Args:
        mu_p_hat: Complex array of shape ``(n_obs,)`` — data-side moments.
        mu_q_hat: Complex array of shape ``(n_obs,)`` — model-side moments.

    Returns:
        Real array of shape ``(n_obs,)``.
    """
    return 2.0 * jnp.real(jnp.conj(mu_p_hat) * mu_q_hat)


@partial(jax.jit, static_argnames=["dims_visible", "sqrt_loss"])
def _unbiased_mmd_squared(  # pylint: disable=too-many-arguments
    mu_q_hat: jnp.ndarray,
    cov: jnp.ndarray,
    X_data: jnp.ndarray,
    l_visible: jnp.ndarray,
    dims_visible: tuple[int, ...],
    sqrt_loss: bool,
) -> jnp.ndarray:
    """Combine PP, PQ, and QQ terms into the unbiased MMD² estimator."""
    m = X_data.shape[0]

    mu_p_hat = _empirical_fourier_moments(l_visible, X_data, jnp.asarray(dims_visible))

    pp_term = _pp_term(mu_p_hat, m)
    pq_term = _pq_cross_term(mu_p_hat, mu_q_hat)
    qq_term = _qq_term(mu_q_hat, cov)

    mmd_sq = jnp.mean(qq_term - pq_term + pp_term)
    return jnp.sqrt(jnp.abs(mmd_sq)) if sqrt_loss else mmd_sq


@partial(
    jax.jit,
    static_argnames=[
        "n_ops",
        "n_qudits",
        "dims",
        "wire_tuple",
        "sqrt_loss",
        "estimator",
        "graph_type",
    ],
)
def _compute_qudit_loss_for_bandwidth(  # pylint: disable=too-many-arguments
    bandwidth: float,
    obs_key: jnp.ndarray,
    eval_key: jnp.ndarray,
    params: jnp.ndarray,
    target_data: jnp.ndarray,
    n_ops: int,
    n_qudits: int,
    dims: tuple[int, ...],
    wire_tuple: tuple[int, ...],
    sqrt_loss: bool,
    estimator: Estimator,
    graph_type: str,
) -> jnp.ndarray:
    """Estimate one unbiased MMD loss value for a single bandwidth setting."""
    l_obs = _sample_fourier_indices(
        obs_key, n_ops, n_qudits, dims, bandwidth, graph_type, wire_tuple
    )
    m_obs = jnp.zeros_like(l_obs)

    mu_q_hat, cov = estimator(params, (l_obs, m_obs), key=eval_key)

    l_visible = l_obs[:, list(wire_tuple)]
    dims_visible = tuple(int(dims[w]) for w in wire_tuple)

    return _unbiased_mmd_squared(mu_q_hat, cov, target_data, l_visible, dims_visible, sqrt_loss)


def _build_qudit_mmd_loss(estimator: Estimator, mmd_config: QuditMMDConfig) -> Callable:
    """Validate an estimator and build the qudit graph-kernel MMD loss.

    Raises:
        TypeError: If ``estimator`` does not measure Heisenberg-Weyl observables.
        ValueError: If the MMD hyperparameters are invalid.
    """
    if not isinstance(estimator, Estimator):
        raise TypeError(
            f"build_qudit_mmd_loss expects a tcdq Estimator, got {type(estimator).__name__}. "
            "Build one with TCDQSimulator.build_estimator(name)."
        )

    if estimator.algebra is not ObservableAlgebra.HEISENBERG_WEYL:
        raise TypeError(
            f"The graph-kernel qudit MMD loss samples Heisenberg-Weyl displacement "
            f"operators, but estimator {estimator.name!r} declares the "
            f"{estimator.algebra.value!r} observable algebra. It must declare "
            f"'heisenberg_weyl'."
        )

    if mmd_config.n_ops < 1:
        raise ValueError("n_ops must be at least 1")

    dims = estimator.local_dims
    n_qudits = estimator.n_wires
    wire_tuple = _resolve_wires(mmd_config.wires, n_qudits)
    bandwidths = _resolve_bandwidths(mmd_config.bandwidth)

    def loss_fn(
        params: ArrayLike,
        target_data: ArrayLike,
        key: ArrayLike,
    ) -> jnp.ndarray | list[jnp.ndarray]:
        """Estimate the empirical qudit MMD loss for one parameter setting.

        The input ``target_data`` is interpreted as samples from the empirical
        data distribution on the visible wires. For each requested bandwidth,
        this function samples a fresh batch of observables, estimates the
        corresponding circuit moments, computes the matching empirical moments
        from ``target_data``, and returns the resulting unbiased MMD estimate.

        If multiple bandwidths are configured, each bandwidth gets its own
        independent observable batch and circuit-evaluation randomness.

        Args:
            params (ArrayLike): Trainable circuit parameters passed to the
                underlying estimator.
            target_data (ArrayLike): Integer array of shape ``(m, n_visible)``
                whose rows are empirical samples on the visible wires.
            key (ArrayLike): JAX PRNG key for this call.

        Returns:
            jnp.ndarray | list[jnp.ndarray]: Either a scalar mean across
            bandwidths or a list of per-bandwidth loss values when
            ``return_per_bandwidth`` is enabled.
        """
        data = _validate_target_data(target_data, len(wire_tuple))

        losses: list[jnp.ndarray] = []
        for bandwidth in bandwidths:
            key, obs_key, eval_key = jax.random.split(key, 3)

            losses.append(
                _compute_qudit_loss_for_bandwidth(
                    bandwidth=bandwidth,
                    obs_key=obs_key,
                    eval_key=eval_key,
                    params=jnp.asarray(params),
                    target_data=data,
                    n_ops=mmd_config.n_ops,
                    n_qudits=n_qudits,
                    dims=dims,
                    wire_tuple=wire_tuple,
                    sqrt_loss=mmd_config.sqrt_loss,
                    estimator=estimator,
                    graph_type=mmd_config.graph_type,
                )
            )

        if mmd_config.return_per_bandwidth:
            return losses
        return jnp.mean(jnp.stack(losses))

    return loss_fn


def build_qudit_mmd_loss(
    estimator: Estimator | QuditCircuitConfig,
    mmd_config: QuditMMDConfig,
) -> Callable:
    """Build a graph-kernel MMD loss on top of any Heisenberg-Weyl estimator.

    The returned callable measures the distance between a qudit circuit's
    output distribution and an empirical target dataset of dit-strings, using
    the Maximum Mean Discrepancy (MMD) with a graph-based kernel.

    Any :class:`~pennylane.labs.tcdq.TCDQSimulator` exposing an estimator that
    declares the ``HEISENBERG_WEYL`` observable algebra can be used here, not
    only :class:`~pennylane.labs.tcdq.QuditIQPSimulator`.

    Args:
        estimator (Estimator): An estimator obtained from
            :meth:`~pennylane.labs.tcdq.TCDQSimulator.build_estimator`. Passing
            a :class:`~pennylane.labs.tcdq.QuditCircuitConfig` here is
            deprecated and supported only for backwards compatibility.
        mmd_config (QuditMMDConfig): MMD hyperparameters including the
            bandwidth, number of observables, and graph type. See
            :class:`QuditMMDConfig`.

    Returns:
        Callable: A function with signature ``loss_fn(params, target_data, key)``
        that returns either a scalar MMD² estimate (averaged across bandwidths)
        or a list of per-bandwidth values when
        ``mmd_config.return_per_bandwidth=True``.

    Raises:
        TypeError: If ``estimator`` does not measure Heisenberg-Weyl observables.
        ValueError: If ``mmd_config.n_ops < 1``, if ``mmd_config.bandwidth`` is
            empty, or if ``mmd_config.wires`` contains duplicates or indices
            outside ``[0, n_qudits)``.

    **Example**

    >>> import jax
    >>> import jax.numpy as jnp
    >>> from pennylane.labs.tcdq import (
    ...     QuditIQPSimulator, QuditMMDConfig, build_qudit_mmd_loss
    ... )
    >>> sim = QuditIQPSimulator(
    ...     dims=3,
    ...     n_qudits=2,
    ...     gates={0: [[1, 0]], 1: [[0, 1]]},
    ...     n_samples=512,
    ...     key=jax.random.PRNGKey(0),
    ... )
    >>> mmd_config = QuditMMDConfig(bandwidth=[0.3, 1.0], n_ops=32)
    >>> loss_fn = build_qudit_mmd_loss(sim.build_estimator("hw_expval"), mmd_config)
    >>> params = jnp.array([0.2, -0.1])
    >>> target_data = jnp.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=jnp.int32)
    >>> loss = loss_fn(params, target_data, jax.random.PRNGKey(123))
    >>> loss.shape
    ()

    .. seealso::

        :class:`~pennylane.labs.tcdq.QuditIQPSimulator`,
        `Section IV B of Spectral Born machines: classically trainable quantum generative models for discrete data <https://arxiv.org/abs/2607.06675>`_.
    """
    if not isinstance(estimator, QuditCircuitConfig):
        return _build_qudit_mmd_loss(estimator, mmd_config)

    circuit_config = estimator
    base_loss_fn = _build_qudit_mmd_loss(
        _simulator_from_config(circuit_config).build_estimator("hw_expval"), mmd_config
    )

    def legacy_loss_fn(
        params: ArrayLike,
        target_data: ArrayLike,
        key: ArrayLike | None = None,
    ) -> jnp.ndarray | list[jnp.ndarray]:
        """Estimate the qudit MMD loss, defaulting the key to the circuit config's.

        Args:
            params (ArrayLike): Trainable circuit parameters.
            target_data (ArrayLike): Integer array of shape ``(m, n_visible)``.
            key (ArrayLike | None): Optional PRNG key overriding
                ``circuit_config.key`` for this call.

        Returns:
            jnp.ndarray | list[jnp.ndarray]: The MMD estimate.
        """
        return base_loss_fn(params, target_data, circuit_config.key if key is None else key)

    return legacy_loss_fn
