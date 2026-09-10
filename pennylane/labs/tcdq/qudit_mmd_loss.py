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
"""Maximum Mean Discrepancy (MMD) loss for Heisenberg-Weyl expectation value functions.

This module compares the output distribution of a qudit model to a dataset of
dit-strings. It samples Heisenberg-Weyl observables from a graph heat-kernel
distribution, estimates their moments with a user-supplied callable, and
combines the results into an unbiased MMD loss.

The kernel is defined from a graph on one qudit level set. The available
choices are the cycle graph :math:`C_d` and the complete graph :math:`K_d`.

For the mathematical construction, see
`Section IV B of Spectral Born machines: classically trainable quantum generative models for discrete data <https://arxiv.org/abs/2607.06675>`_.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from .qudit_expval_functions import _dims_to_numpy


@dataclass(frozen=True)
class QuditMMDConfig:
    r"""Hyperparameters for the qudit graph-kernel MMD loss.

    The MMD measures how well the model output matches a target dataset of
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
    cov: jnp.ndarray | None,
) -> jnp.ndarray:
    """Compute the unbiased model–model U-statistic contribution to the MMD.

    Removes the estimated variance of the complex sample mean from
    :math:`|\\hat{\\mu}_q|^2`. When ``cov`` is ``None`` the model moments are
    treated as exact and no variance correction is applied.

    Args:
        mu_q_hat: Complex array of shape ``(n_obs,)`` — model-side moment
            estimates.
        cov: Real array of shape ``(n_obs, 2, 2)`` — covariance matrices of
            the real and imaginary parts of the estimated moments, or ``None``
            for an exact model.

    Returns:
        Real array of shape ``(n_obs,)``.
    """
    if cov is None:
        return jnp.abs(mu_q_hat) ** 2
    variances = jnp.trace(cov, axis1=-2, axis2=-1)
    return jnp.abs(mu_q_hat) ** 2 - variances


def _pq_cross_term(
    mu_p_hat: jnp.ndarray,
    mu_q_hat: jnp.ndarray,
) -> jnp.ndarray:
    """Compute the data–model cross term of the MMD.

    :math:`PQ(l) = 2 \\operatorname{Re}(\\hat{\\mu}_p(l)^* \\hat{\\mu}_q(l))`.
    No diagonal correction is needed because the data and model samples are
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
    cov: jnp.ndarray | None,
    X_data: jnp.ndarray,
    l_visible: jnp.ndarray,
    dims_visible: tuple[int, ...],
    sqrt_loss: bool,
) -> jnp.ndarray:
    """Combine PP, PQ, and QQ terms into the unbiased MMD² estimator.

    ``cov`` may be ``None`` for an exact model.
    """
    m = X_data.shape[0]

    mu_p_hat = _empirical_fourier_moments(l_visible, X_data, jnp.asarray(dims_visible))

    pp_term = _pp_term(mu_p_hat, m)
    pq_term = _pq_cross_term(mu_p_hat, mu_q_hat)
    qq_term = _qq_term(mu_q_hat, cov)

    mmd_sq = jnp.mean(qq_term - pq_term + pp_term)
    return jnp.sqrt(jnp.abs(mmd_sq)) if sqrt_loss else mmd_sq


def _partition_expval_kwargs(
    expval_kwargs: dict,
) -> tuple[tuple[tuple[str, object], ...], dict]:
    """Split ``expval_kwargs`` into compile-time constants and traced values.

    Hashable values (for example ``n_samples=2000``) are returned as a sorted
    tuple of ``(name, value)`` pairs so they can be marked static under
    ``jax.jit``. Unhashable values, notably arrays, are returned as a dict and
    traced.
    """
    static: dict = {}
    traced: dict = {}
    for name, value in expval_kwargs.items():
        try:
            hash(value)
        except TypeError:
            traced[name] = value
        else:
            static[name] = value
    return tuple(sorted(static.items())), traced


# pylint: disable=too-many-arguments,too-many-locals
@partial(
    jax.jit,
    static_argnames=[
        "n_ops",
        "n_qudits",
        "dims",
        "wire_tuple",
        "sqrt_loss",
        "expval_fn",
        "graph_type",
        "static_expval_kwargs",
    ],
)
def _compute_qudit_loss_for_bandwidth(
    bandwidth: float,
    obs_key: jnp.ndarray,
    eval_key: jnp.ndarray,
    params: jnp.ndarray,
    target_data: jnp.ndarray,
    traced_expval_kwargs: dict,
    n_ops: int,
    n_qudits: int,
    dims: tuple[int, ...],
    wire_tuple: tuple[int, ...],
    sqrt_loss: bool,
    expval_fn: Callable,
    graph_type: str,
    static_expval_kwargs: tuple[tuple[str, object], ...],
) -> jnp.ndarray:
    """Estimate one unbiased MMD loss value for a single bandwidth setting."""
    l_obs = _sample_fourier_indices(
        obs_key, n_ops, n_qudits, dims, bandwidth, graph_type, wire_tuple
    )
    m_obs = jnp.zeros_like(l_obs)

    model_output = expval_fn(
        params,
        observables=(l_obs, m_obs),
        key=eval_key,
        **dict(static_expval_kwargs),
        **traced_expval_kwargs,
    )

    mu_q_hat, cov = model_output if isinstance(model_output, tuple) else (model_output, None)

    mu_q_hat = jnp.asarray(mu_q_hat)
    if cov is not None:
        cov = jnp.asarray(cov)

    if mu_q_hat.shape != (n_ops,):
        raise ValueError(
            f"expval_fn returned moments of shape {mu_q_hat.shape}, expected ({n_ops},)"
        )
    if cov is not None and cov.shape != (n_ops, 2, 2):
        raise ValueError(
            f"expval_fn returned covariances of shape {cov.shape}, expected ({n_ops}, 2, 2)"
        )

    l_visible = l_obs[:, list(wire_tuple)]
    dims_visible = tuple(int(dims[w]) for w in wire_tuple)

    return _unbiased_mmd_squared(mu_q_hat, cov, target_data, l_visible, dims_visible, sqrt_loss)


def build_qudit_mmd_loss(
    expval_fn: Callable,
    dims: int | Sequence[int],
    n_qudits: int,
    mmd_config: QuditMMDConfig,
) -> Callable:
    r"""Build a reusable loss function that computes the qudit graph-kernel MMD.

    The returned callable measures the distance between a qudit model's output
    distribution and an empirical target dataset of dit-strings using the
    Maximum Mean Discrepancy (MMD) with a graph-based kernel. The model is
    called as::

        expval_fn(params, observables=(l_vecs, m_vecs), key=..., **expval_kwargs)

    where ``(l_vecs, m_vecs)`` are integer arrays of shape ``(n_ops, n_qudits)``
    identifying the Heisenberg-Weyl operators :math:`O(\mathbf{l}, \mathbf{m})`
    to measure. Only :math:`\mathbf{m} = \mathbf{0}` is generated, so the
    requested moments are the graph-Fourier moments of the output distribution.
    It must return ``moments`` of shape ``(n_ops,)``, or ``(moments, cov)``
    where ``cov[i]`` is the ``(2, 2)`` real/imaginary covariance matrix of the
    estimator ``moments[i]``; returning ``moments`` alone declares the model
    exact.

    Args:
        expval_fn (Callable): Heisenberg-Weyl expectation value function, as
            above. Must be hashable and JAX-traceable.
        dims (int | Sequence[int]): Local qudit dimension(s). Either a single
            ``int`` broadcast to every qudit, or a sequence of length
            ``n_qudits`` giving a distinct dimension :math:`d_j` per qudit.
        n_qudits (int): Number of qudits the model acts on, i.e. the width of
            the observable arrays passed to ``expval_fn``.
        mmd_config (QuditMMDConfig): MMD hyperparameters including the
            bandwidth, number of observables, and graph type. See
            :class:`QuditMMDConfig`.

    Returns:
        Callable: A function with signature
        ``loss_fn(params, target_data, key=None, **expval_kwargs)`` that returns
        either a scalar MMD² estimate (averaged across bandwidths) or a list of
        per-bandwidth values when ``mmd_config.return_per_bandwidth=True``.

    Raises:
        ValueError: If ``mmd_config`` leaves ``bandwidth`` or ``n_ops`` unset,
            if ``mmd_config.bandwidth`` is empty, if ``mmd_config.n_ops < 1``,
            or if ``mmd_config.wires`` contains duplicates or indices outside
            ``[0, n_qudits)``.

    **Example**

    >>> import jax
    >>> import jax.numpy as jnp
    >>> from pennylane.labs.tcdq import (
    ...     QuditCircuitConfig, QuditMMDConfig, build_qudit_expval_func, build_qudit_mmd_loss,
    ... )
    >>> circuit_config = QuditCircuitConfig(
    ...     dims=3,
    ...     n_qudits=2,
    ...     gates={0: [[1, 0]], 1: [[0, 1]]},
    ...     n_samples=512,
    ...     key=jax.random.PRNGKey(0),
    ... )
    >>> mmd_config = QuditMMDConfig(bandwidth=[0.3, 1.0], n_ops=32)
    >>> loss_fn = build_qudit_mmd_loss(
    ...     build_qudit_expval_func(circuit_config), 3, 2, mmd_config
    ... )
    >>> params = jnp.array([0.2, -0.1])
    >>> target_data = jnp.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=jnp.int32)
    >>> loss = loss_fn(params, target_data, key=jax.random.PRNGKey(123))
    >>> loss.shape
    ()

    .. seealso::

        :func:`~pennylane.labs.tcdq.build_qudit_expval_func`,
        `Section IV B of Spectral Born machines: classically trainable quantum generative models for discrete data <https://arxiv.org/abs/2607.06675>`_.
    """
    if mmd_config.bandwidth is None or mmd_config.n_ops is None:
        raise ValueError("mmd_config must specify both bandwidth and n_ops")

    if mmd_config.n_ops < 1:
        raise ValueError("n_ops must be at least 1")

    dims_tuple = tuple(int(x) for x in _dims_to_numpy(dims, n_qudits))

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

    def loss_fn(
        params: ArrayLike,
        target_data: ArrayLike,
        key: ArrayLike | None = None,
        **expval_kwargs,
    ) -> jnp.ndarray | list[jnp.ndarray]:
        """Estimate the empirical qudit MMD loss for one parameter setting.

        The input ``target_data`` is interpreted as samples from the empirical
        data distribution on the visible wires. For each requested bandwidth,
        this function samples a fresh batch of Heisenberg-Weyl observables,
        estimates their moments with ``expval_fn``, computes the matching
        empirical moments from ``target_data``, and returns the resulting
        unbiased MMD estimate.

        If multiple bandwidths are configured, each bandwidth gets its own
        independent observable batch and model-evaluation randomness.

        Args:
            params: Trainable model parameters, passed to ``expval_fn`` as its
                first argument.
            target_data: Integer array of shape ``(m, n_visible)`` whose rows
                are empirical samples on the visible wires.
            key: Optional JAX PRNG key seeding this call. It is split once per
                bandwidth into one key for observable sampling and one that is
                forwarded to ``expval_fn``. If ``None``, uses
                ``jax.random.PRNGKey(0)``.
            **expval_kwargs: Extra keyword arguments forwarded to ``expval_fn``,
                for example ``n_samples=4000`` or ``phase_fn_params=xi``.
                Hashable values are forwarded as compile-time constants;
                unhashable ones, notably arrays, are traced. ``observables``
                is reserved.

        Returns:
            Either a scalar mean across bandwidths or a list of per-bandwidth
            loss values when ``return_per_bandwidth`` is enabled.

        Raises:
            ValueError: If ``target_data`` is not 2-D, has fewer than two rows
                or an unexpected number of columns, if ``expval_kwargs``
                contains ``"observables"``, or if ``expval_fn`` returns arrays
                of the wrong shape.
        """
        if "observables" in expval_kwargs:
            raise ValueError(
                "expval_kwargs must not contain 'observables': the loss samples the observables "
                "and passes them to expval_fn itself"
            )

        active_key = jax.random.PRNGKey(0) if key is None else key
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

        static_expval_kwargs, traced_expval_kwargs = _partition_expval_kwargs(expval_kwargs)

        losses: list[jnp.ndarray] = []
        for bandwidth in bandwidth_list:
            active_key, obs_key, eval_key = jax.random.split(active_key, 3)

            loss_val = _compute_qudit_loss_for_bandwidth(
                bandwidth=bandwidth,
                obs_key=obs_key,
                eval_key=eval_key,
                params=jnp.asarray(params),
                target_data=X_data,
                traced_expval_kwargs=traced_expval_kwargs,
                n_ops=mmd_config.n_ops,
                n_qudits=n_qudits,
                dims=dims_tuple,
                wire_tuple=wire_tuple,
                sqrt_loss=mmd_config.sqrt_loss,
                expval_fn=expval_fn,
                graph_type=mmd_config.graph_type,
                static_expval_kwargs=static_expval_kwargs,
            )
            losses.append(loss_val)

        if mmd_config.return_per_bandwidth:
            return losses
        return jnp.mean(jnp.stack(losses))

    return loss_fn
