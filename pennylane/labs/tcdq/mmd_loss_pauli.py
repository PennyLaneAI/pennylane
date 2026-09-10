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
"""Maximum Mean Discrepancy (MMD) loss for Pauli expectation value functions.

This module compares the output distribution of a model to a dataset of
bitstrings. It samples Pauli-Z observables from an RBF (Radial Basis Function) kernel distribution,
estimates their expectation values with a user-supplied callable, and combines the results into an
MMD loss.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike


@dataclass(frozen=True)
class MMDConfig:
    r"""Hyperparameters for the qubit Maximum Mean Discrepancy (MMD) loss.

    The MMD measures how well the model's output distribution matches a target
    dataset.

    Args:
        bandwidth (float | Sequence[float]): The bandwidth :math:`\sigma^2` of the kernel. If a sequence is provided,
            the loss is evaluated for each value and then averaged, unless
            ``return_per_bandwidth=True``.
        n_ops (int): Number of sampled observables per bandwidth. Larger
            values reduce estimator variance.
        wires (Sequence[int] | None): Subset of qubit indices to include in
            the loss. If ``None`` (default), all qubits are used.
        sqrt_loss (bool): If ``True``, return ``sqrt(|MMD²|)`` instead of
            ``MMD²``. Defaults to ``False``.
        return_per_bandwidth (bool): If ``True``, return a list of
            per-bandwidth loss values instead of their scalar average.
            Defaults to ``False``.
        bootstrap_target_data (bool): If ``True``, resample ``target_data`` with
            replacement before computing the loss. This makes the target-target
            U-statistic unbiased with respect to the empirical target
            distribution. Defaults to ``True``.

    **Example**

    >>> from pennylane.labs.tcdq import MMDConfig, median_heuristic
    >>> import numpy as np
    >>> target_data = np.random.binomial(1, 0.5, size=(200, 6))
    >>> bw = median_heuristic(target_data)
    >>> config = MMDConfig(bandwidth=bw, n_ops=64)
    """

    #: Width of the RBF kernel (scalar or sequence for multi-bandwidth).
    bandwidth: float | Sequence[float] = None
    #: Number of sampled observables per bandwidth.
    n_ops: int = None
    #: Subset of qubit indices to include, or ``None`` for all qubits.
    wires: Sequence[int] | None = None
    #: If ``True``, return ``sqrt(|MMD²|)`` instead of ``MMD²``.
    sqrt_loss: bool = False
    #: If ``True``, return per-bandwidth losses instead of their average.
    return_per_bandwidth: bool = False
    #: If ``True``, bootstrap the target data before computing the loss.
    bootstrap_target_data: bool = True


def median_heuristic(samples: ArrayLike) -> float:
    """Choose a reasonable radial basis function (RBF) kernel bandwidth from the target dataset.

    A good starting point for :class:`MMDConfig`'s ``bandwidth`` parameter.

    Args:
        samples (ArrayLike): Dataset array of shape ``(n_samples, n_features)``.
            For qubit circuits, this is typically a binary matrix of bitstrings.

    Returns:
        float: A scalar bandwidth derived from the target data. Returns ``1.0``
        if all samples are identical.

    Raises:
        ValueError: If fewer than two samples are provided.

    **Example**

    >>> import numpy as np
    >>> from pennylane.labs.tcdq import median_heuristic
    >>> data = np.array([[0, 1, 0], [1, 0, 1], [1, 1, 0], [0, 0, 1]])
    >>> median_heuristic(data)
    1.4142135623730951
    """
    arr = np.asarray(samples, dtype=float)
    if len(arr) < 2:
        raise ValueError("median_heuristic requires at least two samples")

    diffs = arr[:, None, :] - arr[None, :, :]
    dists = np.sqrt(np.sum(diffs * diffs, axis=-1))
    pairwise = dists[np.triu_indices(len(arr), k=1)]
    nonzero = pairwise[pairwise > 0]

    if len(nonzero) > 0:
        return float(np.median(nonzero))
    return 1.0


@jax.jit
def _binary_ops_to_pauli_int(binary_ops: ArrayLike) -> jnp.ndarray:
    """Map binary operator entries to Pauli integer codes (0 → I, 1 → Z=3)."""
    ops = jnp.asarray(binary_ops, dtype=jnp.int32)
    return jnp.where(ops == 1, 3, 0).astype(jnp.int32)


@partial(jax.jit, static_argnames=["sqrt_loss"])
def _compute_single_mmd(
    model_expvals: jnp.ndarray,
    model_expvals_variances: jnp.ndarray | None,
    target_data: jnp.ndarray,
    visible_ops: jnp.ndarray,
    sqrt_loss: bool,
) -> jnp.ndarray:
    """Core, heavily JIT-compiled math for MMD calculation.

    ``model_expvals_variances`` may be ``None`` for an exact model.
    """
    tr_train = jnp.mean(1 - 2 * ((target_data @ visible_ops.T) % 2), axis=0)
    m = target_data.shape[0]

    result = model_expvals**2
    if model_expvals_variances is not None:
        result = result - model_expvals_variances
    result = result - 2 * model_expvals * tr_train + (tr_train * tr_train * m - 1) / (m - 1)

    reduced = jnp.mean(result)
    return jnp.sqrt(jnp.abs(reduced)) if sqrt_loss else reduced


# pylint: disable=too-many-arguments,too-many-locals
@partial(
    jax.jit,
    static_argnames=[
        "n_ops",
        "n_qubits",
        "wire_tuple",
        "sqrt_loss",
        "expval_fn",
    ],
)
def _compute_loss_for_bandwidth(
    bandwidth: float,
    subkey: jnp.ndarray,
    eval_key: jnp.ndarray,
    params: jnp.ndarray,
    target_data: jnp.ndarray,
    expval_kwargs: dict,
    n_ops: int,
    n_qubits: int,
    wire_tuple: tuple[int, ...],
    sqrt_loss: bool,
    expval_fn: Callable,
):
    """JIT-compiled step that fuses observable generation and expectation value math."""
    wire_list = list(wire_tuple)

    p_mmd = (1 - jnp.exp(-1 / (2 * bandwidth**2))) / 2
    visible_ops = jnp.array(
        jax.random.binomial(subkey, 1, p_mmd, shape=(n_ops, len(wire_tuple))),
        dtype=jnp.float64,
    )

    all_ops = jnp.zeros((n_ops, n_qubits), dtype=jnp.float64)
    all_ops = all_ops.at[:, wire_list].set(visible_ops)

    pauli_obs = _binary_ops_to_pauli_int(all_ops)

    expval_kwargs["observables"] = pauli_obs
    expval_kwargs["key"] = eval_key

    model_output = expval_fn(params, **expval_kwargs)

    model_expvals, model_expvals_variances = (
        model_output if isinstance(model_output, tuple) else (model_output, None)
    )

    model_expvals = jnp.asarray(model_expvals)
    if model_expvals_variances is not None:
        model_expvals_variances = jnp.asarray(model_expvals_variances)

    if model_expvals.shape != (n_ops,):
        raise ValueError(
            f"expval_fn returned expectation values of shape {model_expvals.shape}, "
            f"expected ({n_ops},)"
        )
    if model_expvals_variances is not None and model_expvals_variances.shape != (n_ops,):
        raise ValueError(
            f"expval_fn returned variances of shape {model_expvals_variances.shape}, "
            f"expected ({n_ops},)"
        )

    return _compute_single_mmd(
        model_expvals,
        model_expvals_variances,
        target_data,
        visible_ops,
        sqrt_loss,
    )


def build_mmd_loss_pauli(
    expval_fn: Callable,
    n_qubits: int,
    mmd_config: MMDConfig,
) -> Callable:
    r"""Build a reusable loss function that computes the qubit Pauli-kernel MMD.

    The returned callable measures the distance between a model's output
    distribution and an empirical target dataset of bitstrings using the
    Maximum Mean Discrepancy (MMD) with an RBF kernel expanded in Pauli-Z
    strings. The model is called as

    .. code-block:: python

        expval_fn(params, observables=..., **expval_kwargs)

    where ``observables`` is an integer array of shape ``(n_ops, n_qubits)`` of
    Pauli codes (``0=I``, ``1=X``, ``2=Y``, ``3=Z``), of which only ``I`` and
    ``Z`` are generated. It must return ``expvals`` of shape ``(n_ops,)``, or
    ``(expvals, variances)`` where ``variances[i]`` is the variance of the
    estimator ``expvals[i]``; returning ``expvals`` alone declares the model exact.

    Args:
        expval_fn (Callable): Pauli expectation value function, as above. Must be
            hashable and JAX-traceable.
        n_qubits (int): Number of qubits the model acts on, i.e. the width of the
            observable array passed to ``expval_fn``.
        mmd_config (MMDConfig): Hyperparameters for the MMD computation,
            including the RBF bandwidth and number of observables. See
            :class:`MMDConfig`.

    Returns:
        Callable: A function with signature
        ``loss_fn(params, target_data, key=None, **expval_kwargs)`` that returns
        either a scalar MMD² estimate (averaged across bandwidths) or a list of
        per-bandwidth values when ``mmd_config.return_per_bandwidth=True``.

    Raises:
        ValueError: If ``mmd_config`` leaves ``bandwidth`` or ``n_ops`` unset, if
            ``mmd_config.bandwidth`` is empty, if ``mmd_config.n_ops < 1``, or if
            ``mmd_config.wires`` contains duplicates or indices outside
            ``[0, n_qubits)``.

    **Example**

    >>> import jax
    >>> import jax.numpy as jnp
    >>> import numpy as np
    >>> from pennylane.labs.tcdq import (
    ...     CircuitConfig, MMDConfig, build_expval_func, build_mmd_loss_pauli,
    ...     create_local_gates, median_heuristic,
    ... )
    >>> n_qubits = 4
    >>> gates = create_local_gates(n_qubits, max_weight=2)
    >>> config = CircuitConfig(
    ...     gates=gates, n_samples=1000, key=jax.random.PRNGKey(0), n_qubits=n_qubits
    ... )
    >>> target = np.random.binomial(1, 0.5, size=(100, n_qubits))
    >>> mmd_config = MMDConfig(bandwidth=median_heuristic(target), n_ops=50)
    >>> loss_fn = build_mmd_loss_pauli(build_expval_func(config), n_qubits, mmd_config)
    >>> params = jnp.zeros(len(gates))
    >>> loss = loss_fn(params, target, key=jax.random.PRNGKey(123))
    >>> loss.shape
    ()

    .. seealso::

        :func:`~pennylane.labs.tcdq.build_expval_func`,
        `Section 3.3 of IQPopt: Fast optimization of instantaneous quantum polynomial circuits in JAX <https://arxiv.org/pdf/2501.04776>`_
    """
    if mmd_config.bandwidth is None or mmd_config.n_ops is None:
        raise ValueError("mmd_config must specify both bandwidth and n_ops")

    if mmd_config.n_ops < 1:
        raise ValueError("n_ops must be at least 1")

    wire_tuple = tuple(range(n_qubits)) if mmd_config.wires is None else tuple(mmd_config.wires)

    for w in wire_tuple:
        if w < 0 or w >= n_qubits:
            raise ValueError(f"Wire index {w} out of range for {n_qubits} qubits")

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
        """Estimate the empirical qubit MMD loss for one parameter setting.

        The input ``target_data`` is interpreted as samples from the empirical
        data distribution on the visible wires. For each requested bandwidth,
        this function samples a fresh batch of Pauli-Z observables, estimates
        their expectation values with ``expval_fn``, computes the matching
        empirical moments from ``target_data``, and returns the resulting
        unbiased MMD estimate.

        If multiple bandwidths are configured, each bandwidth gets its own
        independent observable batch and model-evaluation randomness.

        Args:
            params: Trainable model parameters, passed to ``expval_fn`` as its
                first argument.
            target_data: Binary array of shape ``(m, n_qubits)``, or
                ``(m, len(mmd_config.wires))`` when a wire subset is selected,
                whose rows are bitstring samples from the target distribution.
            key: Optional JAX PRNG key seeding this call. It is split once per
                call to resample the target data when requested, then once per
                bandwidth into one key for observable sampling and one that is
                forwarded to ``expval_fn``. If ``None``, uses
                ``jax.random.PRNGKey(0)``.
            **expval_kwargs: Extra keyword arguments forwarded to ``expval_fn``,
                for example ``n_samples=4000``. Hashable values are forwarded as
                compile-time constants; unhashable ones, notably arrays, are
                traced. ``observables`` is reserved.

        Returns:
            Either a scalar mean across bandwidths or a list of per-bandwidth
            loss values when ``return_per_bandwidth`` is enabled.

        Raises:
            ValueError: If ``target_data`` is not 2-dimensional, has fewer than
                two rows or an unexpected number of columns, if
                ``expval_kwargs`` contains ``"observables"``, or if ``expval_fn``
                returns arrays of the wrong shape.
            TypeError: If ``expval_fn`` does not accept the ``observables`` or
                ``key`` keyword arguments.
        """
        if "observables" in expval_kwargs:
            raise ValueError(
                "expval_kwargs must not contain 'observables': the loss samples the observables "
                "and passes them to expval_fn itself"
            )

        active_key = jax.random.PRNGKey(0) if key is None else key

        target_data = jnp.asarray(target_data)
        if target_data.ndim != 2:
            raise ValueError(f"target_data must be 2-dimensional, got shape {target_data.shape}")
        if target_data.shape[0] <= 1:
            raise ValueError("target_data must contain more than one sample")
        if target_data.shape[1] == n_qubits:
            target_data = target_data[:, list(wire_tuple)]
        elif target_data.shape[1] != len(wire_tuple):
            expected = (
                f"{n_qubits} (one per qubit)"
                if len(wire_tuple) == n_qubits
                else f"{len(wire_tuple)} (one per selected wire) or {n_qubits} (one per qubit)"
            )
            raise ValueError(f"target_data has {target_data.shape[1]} columns, expected {expected}")

        if mmd_config.bootstrap_target_data:
            active_key, target_key = jax.random.split(active_key)
            target_indices = jax.random.choice(
                target_key, target_data.shape[0], shape=(target_data.shape[0],), replace=True
            )
            target_data = target_data[target_indices]

        losses = []
        for bandwidth in bandwidth_list:
            active_key, subkey, eval_key = jax.random.split(active_key, 3)

            loss_val = _compute_loss_for_bandwidth(
                bandwidth=bandwidth,
                subkey=subkey,
                eval_key=eval_key,
                params=jnp.asarray(params),
                target_data=target_data,
                n_ops=mmd_config.n_ops,
                n_qubits=n_qubits,
                wire_tuple=wire_tuple,
                sqrt_loss=mmd_config.sqrt_loss,
                expval_fn=expval_fn,
                expval_kwargs=expval_kwargs,
            )
            losses.append(loss_val)

        if mmd_config.return_per_bandwidth:
            return losses
        return jnp.mean(jnp.stack(losses))

    return loss_fn
