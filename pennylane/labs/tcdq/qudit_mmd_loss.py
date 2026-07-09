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
"""Maximum Mean Discrepancy (MMD) loss for qudit IQP circuits.

This module extends :mod:`~pennylane.labs.tcdq.mmd_loss` from qubits to
qudits. It compares the circuit output to a dataset by sampling observables,
estimating their moments, and combining those estimates into an unbiased MMD
loss.

For qudits, the kernel is defined from a graph on one qudit level set. The
available choices are the cycle graph :math:`C_d` and the complete graph
:math:`K_d`.

For the mathematical construction, see
`Section 5, Graph-Kernel MMD Loss <https://github.com/PennyLaneAI/pennylane/blob/port_tcdq_docs_pr/pennylane/labs/tcdq/notes.md#5-graph-kernel-mmd-loss>`_
of the technical notes.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from .qudit_expval_functions import QuditCircuitConfig, build_qudit_expval_func


@dataclass(frozen=True)
class QuditMMDConfig:
    """Hyperparameters for the qudit graph-kernel MMD loss.

    The MMD measures how well the circuit output matches a target dataset of
    dit-strings. In the qudit setting, the kernel comes from heat diffusion on
    a graph over the local levels of one qudit, applied independently to each
    visible wire.

    Args:
        bandwidth (float | Sequence[float]): Width of the Radial Basis Function (RBF) kernel
            kernel. Small values make the loss sensitive to fine-grained
            differences between distributions; large values emphasize broad
            structure. If a sequence is provided, the loss is evaluated for
            each value and then averaged, unless
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

    .. seealso::

        `Section 5, Graph-Kernel MMD Loss <https://github.com/PennyLaneAI/pennylane/blob/port_tcdq_docs_pr/pennylane/labs/tcdq/notes.md#5-graph-kernel-mmd-loss>`_,
        `Section 5.2, Spectral Sampling Distributions <https://github.com/PennyLaneAI/pennylane/blob/port_tcdq_docs_pr/pennylane/labs/tcdq/notes.md#52-spectral-sampling-distributions>`_
    """

    #: Width of the graph heat kernel (scalar or sequence for multi-bandwidth).
    bandwidth: float | Sequence[float]
    #: Number of sampled observables per bandwidth.
    n_ops: int
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
    `Section 5.2, Spectral Sampling Distributions <https://github.com/PennyLaneAI/pennylane/blob/port_tcdq_docs_pr/pennylane/labs/tcdq/notes.md#52-spectral-sampling-distributions>`_.
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
    `Section 5.2, Spectral Sampling Distributions <https://github.com/PennyLaneAI/pennylane/blob/port_tcdq_docs_pr/pennylane/labs/tcdq/notes.md#52-spectral-sampling-distributions>`_.
    """
    log_unnorm = jnp.zeros(d).at[1:].set(-t * d)
    p = jnp.exp(log_unnorm)
    return p / jnp.sum(p)


def _sample_fourier_indices(  # pylint: disable=too-many-arguments
    key: ArrayLike,
    n_ops: int,
    n_qudits: int,
    d: int,
    bandwidth: float,
    graph_type: str,
    wire_tuple: tuple[int, ...],
) -> jnp.ndarray:
    """Sample Fourier index vectors from the graph-kernel spectral distribution.

    Draws ``n_ops`` vectors from the product distribution
    :math:`P(\\mathbf{l}) = \\prod_i P_1(l_i)` where :math:`P_1` is defined
    by the chosen heat kernel. Positions outside ``wire_tuple`` are zero.

    Returns:
        Integer array of shape ``(n_ops, n_qudits)`` with entries in
        :math:`\\{0, \\ldots, d-1\\}`.
    """
    if graph_type == "cycle":
        marginal = _cycle_marginal_probs(d, bandwidth)
    elif graph_type == "complete":
        marginal = _complete_marginal_probs(d, bandwidth)
    else:
        raise ValueError(f"Unknown graph_type {graph_type!r}; use 'cycle' or 'complete'.")

    n_visible = len(wire_tuple)
    visible_obs = jax.random.choice(key, d, shape=(n_ops, n_visible), p=marginal)

    all_obs = jnp.zeros((n_ops, n_qudits), dtype=jnp.int32)
    all_obs = all_obs.at[:, list(wire_tuple)].set(visible_obs.astype(jnp.int32))
    return all_obs


def _empirical_fourier_moments(
    L_visible: jnp.ndarray,
    X_data: jnp.ndarray,
    d: int,
) -> jnp.ndarray:
    """Compute the empirical Fourier moment for each sampled observable from the dataset.

    For each Fourier index vector :math:`\\mathbf{l}`, computes
    :math:`\\hat{\\mu}_p(\\mathbf{l}) = \\frac{1}{m} \\sum_i \\omega^{\\mathbf{l} \\cdot \\mathbf{x}_i}`
    where :math:`\\omega = e^{2\\pi i / d}`.

    Args:
        L_visible: Integer array of shape ``(n_obs, n_visible)`` — the Fourier
            index vectors restricted to the visible wires.
        X_data: Integer array of shape ``(m, n_visible)`` — target dataset
            samples on the visible wires.
        d: Qudit dimension.

    Returns:
        Complex array of shape ``(n_obs,)``.
    """
    inner = L_visible.astype(jnp.float64) @ X_data.astype(jnp.float64).T
    return jnp.mean(jnp.exp(2j * jnp.pi * inner / d), axis=1)


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
    mean_y_sq: jnp.ndarray,
    s: int,
) -> jnp.ndarray:
    """Compute the unbiased model–model U-statistic contribution to the MMD.

    Removes the diagonal self-pairs from :math:`|\\hat{\\mu}_q|^2`:
    :math:`QQ(l) = (s |\\hat{\\mu}_q(l)|^2 - \\overline{|y|^2}(l)) / (s - 1)`.

    Gradients are stopped through ``mean_y_sq`` (it is treated as a constant
    for differentiation purposes).

    Args:
        mu_q_hat: Complex array of shape ``(n_obs,)`` — circuit-side Monte
            Carlo moment estimates.
        mean_y_sq: Real array of shape ``(n_obs,)`` — mean squared magnitude
            of the per-sample integrand values.
        s: Number of circuit samples.

    Returns:
        Real array of shape ``(n_obs,)``.
    """
    mean_y_sq = jax.lax.stop_gradient(mean_y_sq)
    return (s * jnp.abs(mu_q_hat) ** 2 - mean_y_sq) / (s - 1)


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


@partial(jax.jit, static_argnames=["d", "n_samples", "sqrt_loss"])
def _unbiased_mmd_squared(  # pylint: disable=too-many-arguments
    mu_q_hat: jnp.ndarray,
    mean_y_sq: jnp.ndarray,
    X_data: jnp.ndarray,
    L_visible: jnp.ndarray,
    d: int,
    n_samples: int,
    sqrt_loss: bool,
) -> jnp.ndarray:
    """Combine PP, PQ, and QQ terms into the unbiased MMD² estimator."""
    m = X_data.shape[0]

    mu_p_hat = _empirical_fourier_moments(L_visible, X_data, d)

    pp_term = _pp_term(mu_p_hat, m)
    pq_term = _pq_cross_term(mu_p_hat, mu_q_hat)
    qq_term = _qq_term(mu_q_hat, mean_y_sq, n_samples)

    mmd_sq = jnp.mean(qq_term - pq_term + pp_term)
    return jnp.sqrt(jnp.abs(mmd_sq)) if sqrt_loss else mmd_sq


@partial(
    jax.jit,
    static_argnames=[
        "n_ops",
        "n_qudits",
        "d",
        "n_samples",
        "wire_tuple",
        "sqrt_loss",
        "expval_func",
        "graph_type",
    ],
)
def _compute_qudit_loss_for_bandwidth(  # pylint: disable=too-many-arguments
    bandwidth: float,
    obs_key: jnp.ndarray,
    eval_key: jnp.ndarray,
    params: jnp.ndarray,
    target_data: jnp.ndarray,
    init_state_elems: jnp.ndarray | None,
    init_state_amps: jnp.ndarray | None,
    n_ops: int,
    n_qudits: int,
    d: int,
    n_samples: int,
    wire_tuple: tuple[int, ...],
    sqrt_loss: bool,
    expval_func: Callable,
    graph_type: str,
) -> jnp.ndarray:
    """Estimate one unbiased MMD loss value for a single bandwidth setting."""
    l_obs = _sample_fourier_indices(obs_key, n_ops, n_qudits, d, bandwidth, graph_type, wire_tuple)
    m_obs = jnp.zeros_like(l_obs)

    mu_q_hat, _, mean_y_sq = expval_func(
        gates_params=params,
        observables=(l_obs, m_obs),
        key=eval_key,
        n_samples=n_samples,
        init_state_elems=init_state_elems,
        init_state_amps=init_state_amps,
        return_mean_y_sq=True,
    )

    L_visible = l_obs[:, list(wire_tuple)]
    return _unbiased_mmd_squared(
        mu_q_hat, mean_y_sq, target_data, L_visible, d, n_samples, sqrt_loss
    )


def build_qudit_mmd_loss(
    circuit_config: QuditCircuitConfig,
    mmd_config: QuditMMDConfig,
) -> Callable:
    """Build a reusable loss function that computes the qudit graph-kernel MMD.

    The returned callable measures the distance between the qudit circuit's
    output distribution and an empirical target dataset of dit-strings using
    the Maximum Mean Discrepancy (MMD) with a graph-based kernel.

    Args:
        circuit_config (QuditCircuitConfig): Qudit circuit description
            specifying gate structure, qudit dimension, and sample
            count. See :class:`~pennylane.labs.tcdq.QuditCircuitConfig`.
        mmd_config (QuditMMDConfig): MMD hyperparameters including the
            bandwidth, number of observables, and graph type. See
            :class:`QuditMMDConfig`.

    Returns:
        Callable: A function with signature
        ``loss_fn(params, target_data, key=None)`` that returns either a
        scalar MMD² estimate (averaged across bandwidths) or a list of
        per-bandwidth values when ``mmd_config.return_per_bandwidth=True``.

    Raises:
        ValueError: If ``circuit_config.n_samples <= 1``.
        ValueError: If ``mmd_config.n_ops < 1``.
        ValueError: If ``mmd_config.bandwidth`` is empty.
        ValueError: If ``mmd_config.wires`` contains duplicates or indices
            outside ``[0, n_qudits)``.

    **Example**

    >>> import jax
    >>> import jax.numpy as jnp
    >>> from pennylane.labs.tcdq import QuditCircuitConfig, QuditMMDConfig, build_qudit_mmd_loss
    >>> circuit_config = QuditCircuitConfig(
    ...     d=3,
    ...     n_qudits=2,
    ...     gates={0: [[1, 0]], 1: [[0, 1]]},
    ...     n_samples=512,
    ...     key=jax.random.PRNGKey(0),
    ... )
    >>> mmd_config = QuditMMDConfig(bandwidth=[0.3, 1.0], n_ops=32)
    >>> loss_fn = build_qudit_mmd_loss(circuit_config, mmd_config)
    >>> params = jnp.array([0.2, -0.1])
    >>> target_data = jnp.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=jnp.int32)
    >>> loss = loss_fn(params, target_data, key=jax.random.PRNGKey(123))
    >>> loss.shape
    ()

    .. seealso::

        :func:`~pennylane.labs.tcdq.build_qudit_expval_func`,
        `Section 5, Graph-Kernel MMD Loss <https://github.com/PennyLaneAI/pennylane/blob/port_tcdq_docs_pr/pennylane/labs/tcdq/notes.md#5-graph-kernel-mmd-loss>`_,
        `Section 5.3, Unbiased MMD Estimator <https://github.com/PennyLaneAI/pennylane/blob/port_tcdq_docs_pr/pennylane/labs/tcdq/notes.md#53-unbiased-mmd-estimator>`_
    """
    n_samples = circuit_config.n_samples
    if n_samples <= 1:
        raise ValueError("n_samples must be greater than 1")

    if mmd_config.n_ops < 1:
        raise ValueError("n_ops must be at least 1")

    d = circuit_config.d
    n_qudits = circuit_config.n_qudits

    wire_tuple = tuple(range(n_qudits)) if mmd_config.wires is None else tuple(mmd_config.wires)

    for w in wire_tuple:
        if w < 0 or w >= n_qudits:
            raise ValueError(f"Wire index {w} out of range for {n_qudits} qudits")

    if len(set(wire_tuple)) != len(wire_tuple):
        raise ValueError("wires must not contain duplicates")

    bandwidth_list = (
        [mmd_config.bandwidth]
        if isinstance(mmd_config.bandwidth, (int, float))
        else list(mmd_config.bandwidth)
    )

    if len(bandwidth_list) == 0:
        raise ValueError("bandwidth must not be empty")

    expval_config = QuditCircuitConfig(
        d=d,
        n_qudits=n_qudits,
        gates=circuit_config.gates,
        observables=None,
        n_samples=n_samples,
        key=circuit_config.key,
        init_state_elems=circuit_config.init_state_elems,
        init_state_amps=circuit_config.init_state_amps,
    )
    expval_func = build_qudit_expval_func(expval_config)

    def loss_fn(
        params: ArrayLike,
        target_data: ArrayLike,
        key: ArrayLike | None = None,
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
            params: Trainable circuit parameters passed to the underlying qudit
                expectation-value estimator.
            target_data: Integer array of shape ``(m, n_visible)`` whose rows
                are empirical samples on the visible wires.
            key: Optional PRNG key overriding ``circuit_config.key`` for this
                call.

        Returns:
            Either a scalar mean across bandwidths or a list of per-bandwidth
            loss values when ``return_per_bandwidth`` is enabled.
        """
        active_key = circuit_config.key if key is None else key
        X_data = jnp.asarray(target_data)

        if X_data.ndim != 2:
            raise ValueError(f"target_data must be 2-D, got shape {X_data.shape}")

        n_visible = len(wire_tuple)
        if X_data.shape[1] != n_visible:
            raise ValueError(
                f"target_data has {X_data.shape[1]} columns but expected "
                f"{n_visible} (number of visible wires)"
            )

        if X_data.shape[0] < 2:
            raise ValueError(f"target_data must have at least 2 samples, got {X_data.shape[0]}")

        losses: list[jnp.ndarray] = []
        for bandwidth in bandwidth_list:
            active_key, obs_key, eval_key = jax.random.split(active_key, 3)

            loss_val = _compute_qudit_loss_for_bandwidth(
                bandwidth=bandwidth,
                obs_key=obs_key,
                eval_key=eval_key,
                params=jnp.asarray(params),
                target_data=X_data,
                init_state_elems=circuit_config.init_state_elems,
                init_state_amps=circuit_config.init_state_amps,
                n_ops=mmd_config.n_ops,
                n_qudits=n_qudits,
                d=d,
                n_samples=n_samples,
                wire_tuple=wire_tuple,
                sqrt_loss=mmd_config.sqrt_loss,
                expval_func=expval_func,
                graph_type=mmd_config.graph_type,
            )
            losses.append(loss_val)

        if mmd_config.return_per_bandwidth:
            return losses
        return jnp.mean(jnp.stack(losses))

    return loss_fn
