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

This module builds a stochastic estimate of the squared Maximum Mean
Discrepancy between

* an empirical dataset on ``Z_d^n``, and
* a model distribution defined by a qudit circuit whose Fourier moments are
  estimated with ``build_qudit_expval_func``.

The kernel is a heat kernel on a Cartesian-product graph whose single-site
graph is either the cycle ``C_d`` or the complete graph ``K_d``. For those
vertex-transitive graphs, the kernel spectrum defines a probability
distribution over Fourier index vectors ``l``, and the loss can be written as
an average of squared differences between data-side and model-side moments of
the ``Z``-type Heisenberg-Weyl observables ``O(l, 0)``.

For a fixed sampled observable ``l``, the key algebraic step is the expansion

    |mu_p(l) - mu_q(l)|^2
        = |mu_p(l)|^2 - 2 Re(mu_p(l)^* mu_q(l)) + |mu_q(l)|^2.

This file estimates those three pieces separately and unbiasedly:

* ``PP`` estimates ``|mu_p(l)|^2`` from the dataset,
* ``PQ`` estimates ``2 Re(mu_p(l)^* mu_q(l))`` from the data-model overlap,
  and
* ``QQ`` estimates ``|mu_q(l)|^2`` from circuit Monte Carlo samples.

Operationally, the helpers in this file do four things:

1. build the single-site spectral sampling distributions for the chosen graph,
2. sample a batch of Fourier indices ``l`` from that distribution,
3. compute data-side empirical moments and model-side Monte Carlo moments for
    those sampled observables, and
4. combine them into an unbiased estimate of ``MMD^2`` using U-statistic
    corrections for the data-data and model-model terms.

If you are reading this alongside ``iqpopt_notes.tex``, this file implements
the ``Loss functions via graph-Fourier kernels`` section, especially the
``Estimating from a dataset`` and ``Unbiased estimators`` subsections. The
code uses the same high-level objects as the notes: sampled Fourier indices
``l ~ P``, data moments ``mu_p(l)``, model moments ``mu_q(l)``, and the
per-observable ``PP``, ``PQ``, and ``QQ`` terms of the unbiased estimator.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from .expval_functions import QuditCircuitConfig, build_qudit_expval_func


@dataclass(frozen=True)
class QuditMMDConfig:
    """Configuration for the graph-kernel MMD estimator.

    These fields control how the Fourier observables are sampled and how the
    resulting Monte Carlo loss values are aggregated.

    If you are reading this alongside ``iqpopt_notes.tex``:

    * ``bandwidth`` is the heat-kernel diffusion time ``t``,
    * ``n_ops`` is the observable-batch size ``|L|``,
    * ``graph_type`` chooses the single-site graph whose spectrum defines the
      sampling distribution ``P(l)``, and
    * ``wires`` selects which qudits participate in the observed Fourier
      moments.

    Args:
        bandwidth: Heat-kernel diffusion time or a sequence of diffusion times.
            Larger values emphasize lower graph frequencies. If a sequence is
            supplied, the loss is estimated once per bandwidth and the results
            are either averaged or returned separately.
        n_ops: Number of Fourier index vectors sampled per bandwidth.
        graph_type: Single-site graph used to define the heat-kernel spectrum.
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
    """Return the single-site spectral sampling distribution for the cycle graph.

    For the heat kernel on ``C_d``, the single-site Laplacian eigenvalue at
    Fourier index ``k`` is ``4 sin^2(pi k / d)``. The resulting normalized
    sampling probability is therefore

        P_1(k) propto exp(-4 t sin^2(pi k / d)).

    This helper produces the categorical distribution used to sample each
    visible component of ``l`` independently when ``graph_type == "cycle"``.

    In ``iqpopt_notes.tex``, this is the per-site cycle-graph heat-kernel
    marginal

        P_1(k) = exp(-4 t sin^2(pi k / d)) / Z_1.
    """
    k = jnp.arange(d)
    log_p = -4.0 * t * jnp.sin(jnp.pi * k / d) ** 2
    p = jnp.exp(log_p)
    return p / jnp.sum(p)


def _complete_marginal_probs(d: int, t: float) -> jnp.ndarray:
    """Return the single-site spectral sampling distribution for the complete graph.

    For the heat kernel on ``K_d``, the zero Fourier mode has Laplacian
    eigenvalue ``0`` and every nonzero mode has eigenvalue ``d``. The
    resulting normalized probabilities are

        P_1(0) = 1 / (1 + (d - 1) exp(-t d))
        P_1(k) = exp(-t d) / (1 + (d - 1) exp(-t d))    for k != 0.

    All nonzero modes therefore share the same probability mass. This helper
    returns the categorical distribution used to sample each visible component
    of ``l`` independently when ``graph_type == "complete"``.

    In ``iqpopt_notes.tex``, this is the same complete-graph marginal written
    in piecewise form using

        pi_0 = 1 / (1 + (d - 1) exp(-t d)),

    so that ``P_1(0) = pi_0`` and ``P_1(k) = (1 - pi_0) / (d - 1)`` for
    ``k != 0``. The two forms are algebraically identical.
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

    The loss in this module averages over observables of the form ``O(l, 0)``,
    where ``l`` is drawn from a product distribution ``P(l) = prod_i P_1(l_i)``
    defined by the chosen single-site heat kernel. This helper draws ``n_ops``
    such index vectors.

    Only the wires listed in ``wire_tuple`` are active in the loss. Components
    outside that subset are filled with zeros so that the returned array still
    has shape ``(n_ops, n_qudits)`` and can be passed directly to the circuit
    expectation-value code.

    In ``iqpopt_notes.tex``, this is the code-level realization of sampling the
    observable batch ``L = {l_j ~ P}`` used in the MMD estimator.

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
    """Return the data-side empirical moment of ``O(l, 0)`` for each sampled observable.

    For each row ``l`` of ``L_visible``, this computes the sample mean

        mu_p_hat(l) = (1 / m) * sum_i omega ** (l . x_i),
        omega = exp(2 pi i / d),

    where the rows ``x_i`` of ``X_data`` are dataset samples on the visible
    wires and ``m = X_data.shape[0]``. For the ``Z``-type Heisenberg-Weyl
    observable family used by this loss, this sample mean is exactly the
    data-side expectation of ``O(l, 0)``. The result is a complex-valued batch
    of empirical moments, one per sampled Fourier index vector.

    In the MMD estimator, these are the data-side moments paired with the model
    estimates ``mu_q_hat``. If you are reading this alongside
    ``iqpopt_notes.tex``, this is the dataset-side estimator

        <hat{O}(l, 0)>_{p_X} = (1 / |X|) * sum_{x_i in X} omega ** (l . x_i),

    and it is the empirical version of the ``mu_p(l)`` term used in the
    unbiased decomposition.

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
    """Return the per-observable data-data term in the unbiased MMD estimator.

    For a fixed sampled observable ``l``, let

        mu_p_hat(l) = (1 / m) * sum_i f_p(l, x_i)

    with ``f_p(l, x_i) = omega ** (l . x_i)``. The naive plug-in estimate
    ``|mu_p_hat(l)|^2`` includes diagonal pairs ``i = i``. This helper removes
    those self-pairs and renormalizes the remainder, yielding the U-statistic

        PP(l) = (m * |mu_p_hat(l)|^2 - 1) / (m - 1).

    The subtraction is exactly ``1`` because every data sample has unit
    modulus: ``|f_p(l, x_i)|^2 = 1``. The returned vector therefore gives an
    unbiased estimate of the data-data term ``mu_p(l)^* mu_p(l)`` for each
    sampled observable before the outer average over Fourier indices.

    In ``iqpopt_notes.tex``, this is the per-observable data-data contribution
    to the decomposition

        |mu_p(l) - mu_q(l)|^2
            = |mu_p(l)|^2 - 2 Re(mu_p(l)^* mu_q(l)) + |mu_q(l)|^2,

    written as a U-statistic over distinct data pairs.

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
    """Return the per-observable model-model term in the unbiased MMD estimator.

    For a fixed sampled observable ``l``, the Monte Carlo circuit evaluator
    produces raw sample values ``y_r(l)`` over ``s`` random draws together with

        mu_q_hat(l) = (1 / s) * sum_r y_r(l)
        mean_y_sq(l) = (1 / s) * sum_r |y_r(l)|^2.

    This helper removes the diagonal ``r = r'`` contribution from the naive
    square ``|mu_q_hat(l)|^2`` and returns the corresponding U-statistic

        QQ(l) = (s * |mu_q_hat(l)|^2 - mean_y_sq(l)) / (s - 1).

    Unlike the data term, the raw circuit sample values need not have unit
    modulus, especially for arbitrary initial states. The diagonal correction
    is therefore the observed average ``mean_y_sq``, not the constant ``1``.
    The returned vector is an unbiased estimate of the model-model term
    ``mu_q(l)^* mu_q(l)`` for each sampled observable before the outer average
    over Fourier indices.

    Gradients are stopped through ``mean_y_sq`` so optimization differentiates
    through the estimated mean, not through the variance correction.

    In ``iqpopt_notes.tex``, this is the per-observable model-model term,
    written there as the U-statistic over distinct circuit-sample pairs.

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
    """Return the per-observable data-model cross term in the unbiased MMD estimator.

    For each sampled observable ``l``, this computes

        PQ(l) = 2 * Re(mu_p_hat(l)^* * mu_q_hat(l)),

    which is the middle term in the identity

        |mu_p(l) - mu_q(l)|^2
            = |mu_p(l)|^2 - 2 * Re(mu_p(l)^* mu_q(l)) + |mu_q(l)|^2.

    The multiplication is elementwise across the observable batch, so the
    returned array contains one cross-term value per sampled Fourier index.
    No diagonal correction is needed because the data batch and the circuit
    Monte Carlo batch are independent.

    In ``iqpopt_notes.tex``, this is the cross term from

        |mu_p(l) - mu_q(l)|^2
            = |mu_p(l)|^2 - 2 Re(mu_p(l)^* mu_q(l)) + |mu_q(l)|^2,

    estimated from independent data and circuit batches.

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
    """Combine per-observable PP, PQ, and QQ terms into unbiased MMD^2.

    This function builds the data-side empirical moments ``mu_p_hat`` from the
    dataset, forms the per-observable data-data, data-model, and model-model
    terms, and then averages

        PP(l) - PQ(l) + QQ(l)

    over the sampled Fourier indices in ``L_visible``.

    In ``iqpopt_notes.tex``, this is the code realization of the decomposition

        |mu_p(l) - mu_q(l)|^2
            = |mu_p(l)|^2 - 2 Re(mu_p(l)^* mu_q(l)) + |mu_q(l)|^2,

    together with the unbiased estimator obtained by replacing each of those
    three pieces with its U-statistic form and averaging over the sampled
    observables. If ``sqrt_loss`` is true, the function returns
    ``sqrt(abs(MMD^2_u))`` instead of ``MMD^2_u``.
    """
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
    """Estimate one unbiased MMD loss value for a fixed heat-kernel bandwidth.

    This helper is the per-bandwidth core of the loss estimator. It

    1. samples a batch of Fourier indices ``l`` from the graph-kernel spectrum,
    2. fixes ``m = 0`` so the observables are the ``Z``-type moments
       ``O(l, 0)`` used by the MMD,
    3. asks ``expval_func`` for the model-side moment estimates ``mu_q_hat``
       and the raw second-moment correction ``mean_y_sq``, and
    4. combines those model-side estimates with the data-side empirical moments
       into an unbiased estimate of ``MMD^2``.

    In ``iqpopt_notes.tex``, this corresponds to one Monte Carlo evaluation of
    the heat-kernel MMD at a fixed diffusion time ``t``.
    """
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

    This factory validates the MMD configuration, constructs the underlying
    expectation-value function once, and returns a closure that can be called
    repeatedly during training. Each call to the returned function will sample
    a fresh observable batch for each requested bandwidth, estimate the
    corresponding data-side and model-side Fourier moments, and combine them
    into one or more unbiased MMD loss values.

    If you are reading this alongside ``iqpopt_notes.tex``, the returned
    closure is a code implementation of the empirical estimator described in
    the ``Estimating from a dataset`` and ``Unbiased estimators`` subsections,
    specialized to the cycle-graph and complete-graph heat kernels.

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
