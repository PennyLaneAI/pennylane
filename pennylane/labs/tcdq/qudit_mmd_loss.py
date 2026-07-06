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
"""Estimate graph-kernel MMD losses for qudit circuits on ``Z_d^n``.

If you set aside the quantum terminology, this module implements a
randomized feature-matching loss over discrete vectors:

1. Sample Fourier index vectors from a graph-kernel distribution.
2. Use the qudit expectation estimator to get model moments for those
    indices.
3. Compute the matching empirical moments directly from the dataset.
4. Combine the data-data, data-model, and model-model terms into an
    unbiased MMD estimate.

The public entry point is ``build_qudit_mmd_loss``. See ``notes.md`` §5 for
the derivation and Appendix A for the notation-to-code glossary.
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
    """Configuration for the graph-kernel MMD estimator.

    See ``notes.md`` §5.2 for the spectral sampling distributions.

    Args:
        bandwidth: Heat-kernel diffusion time ``t`` or a sequence of diffusion
            times.  Larger values emphasize lower graph frequencies.
        n_ops: Number of Fourier index vectors ``|L|`` sampled per bandwidth.
        graph_type: Single-site graph for the heat-kernel spectrum.
            ``"cycle"`` means ``C_d`` and ``"complete"`` means ``K_d``.
        wires: Optional subset of qudit indices to include in the loss. If
            ``None``, use all qudits.
        sqrt_loss: If ``True``, return ``sqrt(abs(MMD^2))`` instead of
            ``MMD^2``.
        return_per_bandwidth: If ``True``, return one loss value per bandwidth
            instead of averaging across bandwidths.
    """

    bandwidth: float | Sequence[float]
    n_ops: int
    graph_type: str = "cycle"
    wires: Sequence[int] | None = None
    sqrt_loss: bool = False
    return_per_bandwidth: bool = False


def _cycle_marginal_probs(d: int, t: float) -> jnp.ndarray:
    """Return the cycle-graph spectral sampling distribution (see ``notes.md`` §5.2).

    ``P_1(k) ∝ exp(-4 t sin²(π k / d))``.
    """
    k = jnp.arange(d)
    log_p = -4.0 * t * jnp.sin(jnp.pi * k / d) ** 2
    p = jnp.exp(log_p)
    return p / jnp.sum(p)


def _complete_marginal_probs(d: int, t: float) -> jnp.ndarray:
    """Return the complete-graph spectral sampling distribution (see ``notes.md`` §5.2).

    ``P_1(0) = π_0``, ``P_1(k≠0) = (1 − π_0)/(d − 1)`` with
    ``π_0 = 1 / (1 + (d − 1) exp(−td))``.
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
    """Sample a batch of Fourier index vectors from the graph-kernel spectrum.

    Draws ``n_ops`` vectors ``l ~ P(l) = ∏_i P_1(l_i)`` from the product
    distribution defined by the chosen heat kernel (see ``notes.md`` §5.2).
    Components outside ``wire_tuple`` are zero-filled.

    Returns:
        Integer array of shape ``(n_ops, n_qudits)`` with entries in
        ``{0, ..., d - 1}``. Coordinates outside ``wire_tuple`` are zero.
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
    """Return the data-side empirical Fourier moment for each sampled observable.

    Computes ``μ̂_p(l) = (1/m) Σ_i ω^{l·x_i}`` for each row ``l`` of
    ``L_visible``.  See ``notes.md`` §5.3 (PP term).

    Args:
        L_visible: Integer array of shape ``(n_obs, n_visible)`` whose rows are
            the sampled Fourier index vectors restricted to the visible wires.
        X_data: Integer array of shape ``(m, n_visible)`` containing dataset
            samples on the same visible wires.
        d: Qudit dimension.

    Returns:
        Complex array of shape ``(n_obs,)`` with one empirical moment per
        sampled observable.
    """
    inner = L_visible.astype(jnp.float64) @ X_data.astype(jnp.float64).T
    return jnp.mean(jnp.exp(2j * jnp.pi * inner / d), axis=1)


def _pp_term(mu_p_hat: jnp.ndarray, m: int) -> jnp.ndarray:
    """Return the unbiased data–data U-statistic ``PP(l)`` (see ``notes.md`` §5.3).

    Removes the diagonal self-pairs from ``|μ̂_p(l)|²``:

        ``PP(l) = (m |μ̂_p(l)|² − 1) / (m − 1)``.

    The subtraction is ``1`` because ``|f_p(l, x_i)|² = 1`` for all data
    samples.

    Args:
        mu_p_hat: Complex array of shape ``(n_obs,)`` containing the empirical
            mean for each sampled observable over the dataset.
        m: Number of samples in the dataset.

    Returns:
        Real array of shape ``(n_obs,)`` containing one unbiased data-data term
        per sampled observable.
    """
    return (m * jnp.abs(mu_p_hat) ** 2 - 1.0) / (m - 1)


def _qq_term(
    mu_q_hat: jnp.ndarray,
    mean_y_sq: jnp.ndarray,
    s: int,
) -> jnp.ndarray:
    """Return the unbiased model–model U-statistic ``QQ(l)`` (see ``notes.md`` §5.3).

    Removes the diagonal self-pairs from ``|μ̂_q(l)|²``:

        ``QQ(l) = (s |μ̂_q(l)|² − mean_y_sq(l)) / (s − 1)``.

    Unlike the data term, per-sample values may not have unit modulus
    (especially with non-trivial input states), so the diagonal correction
    uses ``mean_y_sq`` rather than the constant ``1``.

    Gradients are stopped through ``mean_y_sq``.

    Args:
        mu_q_hat: Complex array of shape ``(n_obs,)`` containing the estimated
            mean for each sampled observable over the circuit Monte Carlo batch.
        mean_y_sq: Real array of shape ``(n_obs,)`` containing the mean squared
            magnitude of the raw circuit sample values.
        s: Number of Monte Carlo circuit samples.

    Returns:
        Real array of shape ``(n_obs,)`` containing one unbiased model-model
        term per sampled observable.
    """
    mean_y_sq = jax.lax.stop_gradient(mean_y_sq)
    return (s * jnp.abs(mu_q_hat) ** 2 - mean_y_sq) / (s - 1)


def _pq_cross_term(
    mu_p_hat: jnp.ndarray,
    mu_q_hat: jnp.ndarray,
) -> jnp.ndarray:
    """Return the data–model cross term ``PQ(l)`` (see ``notes.md`` §5.3).

    ``PQ(l) = 2 Re(μ̂_p(l)* μ̂_q(l))``.  No diagonal correction is needed
    because the data and circuit batches are independent.

    Args:
        mu_p_hat: Complex array of shape ``(n_obs,)`` containing the empirical
            data-side moments.
        mu_q_hat: Complex array of shape ``(n_obs,)`` containing the model-side
            Monte Carlo moment estimates.

    Returns:
        Real array of shape ``(n_obs,)`` containing one data-model cross term
        per sampled observable.
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
    """Combine PP, PQ, and QQ into the unbiased MMD² estimator (see ``notes.md`` §5.3)."""
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
    """Estimate one unbiased MMD loss value for a fixed bandwidth (see ``notes.md`` §5)."""
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
    """Build a reusable qudit MMD loss function from circuit and loss configs.

    At a high level, the returned callable draws a random batch of
    Fourier-like observables, estimates those moments for the circuit,
    computes the same moments from ``target_data``, and averages the PP, PQ,
    and QQ contributions of the unbiased MMD estimator.

    See ``notes.md`` §5 for the theoretical background and Appendix A for
    notation.

    Args:
        circuit_config: Configuration for the underlying qudit expectation-value
            estimator.
        mmd_config: Configuration for the graph-kernel MMD estimator.

    Returns:
        Callable with signature ``loss_fn(params, target_data, key=None)``.
        The callable returns either a scalar average across bandwidths or a
        list of per-bandwidth values when
        ``mmd_config.return_per_bandwidth`` is true.

    Raises:
        ValueError: If ``n_samples <= 1``.
        ValueError: If ``n_ops < 1``.
        ValueError: If ``bandwidth`` is empty.
        ValueError: If ``wires`` contains duplicates or out-of-range indices.

    **Example**

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
        this function samples a fresh batch of Fourier observables, estimates
        the corresponding model moments from the circuit, computes the matching
        empirical data moments directly from ``target_data``, and returns the
        resulting unbiased MMD estimate.

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
