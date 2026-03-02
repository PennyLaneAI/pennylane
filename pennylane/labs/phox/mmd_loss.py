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
"""MMD loss utilities for Phox."""

from collections.abc import Sequence
from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from .expval_functions import CircuitConfig, build_expval_func


def median_heuristic(samples: ArrayLike) -> float:
    """Compute a robust median-distance heuristic for RBF bandwidth selection.

    Args:
        samples (ArrayLike): Dataset with shape ``(n_samples, n_features)``.

    Returns:
        float: Median non-zero pairwise Euclidean distance. Returns ``1.0`` when all
        pairwise distances are zero.

    Raises:
        ValueError: If fewer than two samples are provided.
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
    ops = jnp.asarray(binary_ops, dtype=jnp.int32)
    return jnp.where(ops == 1, 3, 0).astype(jnp.int32)


@partial(jax.jit, static_argnames=["n_samples", "sqrt_loss"])
def _compute_single_mmd(
    tr_iqp: jnp.ndarray,
    tr_iqp_std_err: jnp.ndarray,
    ground_truth: jnp.ndarray,
    visible_ops: jnp.ndarray,
    n_samples: int,
    sqrt_loss: bool,
) -> jnp.ndarray:
    """Core, heavily JIT-compiled math for MMD calculation."""
    tr_iqp_std_err = jax.lax.stop_gradient(tr_iqp_std_err)
    correction = (tr_iqp**2 + (n_samples - 1) * tr_iqp_std_err**2) / n_samples

    tr_train = jnp.mean(1 - 2 * ((ground_truth @ visible_ops.T) % 2), axis=0)
    m = ground_truth.shape[0]

    result = (tr_iqp * tr_iqp - correction) * n_samples / (n_samples - 1)
    result = result - 2 * tr_iqp * tr_train + (tr_train * tr_train * m - 1) / (m - 1)

    reduced = jnp.mean(result)
    return jnp.sqrt(jnp.abs(reduced)) if sqrt_loss else reduced


@partial(
    jax.jit,
    static_argnames=[
        "n_ops",
        "n_qubits",
        "wire_tuple",
        "effective_samples",
        "sqrt_loss",
        "expval_func",
    ],
)
def _compute_loss_for_sigma(
    sigma_val: float,
    subkey: jnp.ndarray,
    eval_key: jnp.ndarray,
    params: jnp.ndarray,
    ground_truth: jnp.ndarray,
    effective_init_state: tuple | None,
    n_ops: int,
    n_qubits: int,
    wire_tuple: tuple[int, ...],
    effective_samples: int,
    sqrt_loss: bool,
    expval_func: Callable,
):
    """JIT-compiled step that fuses observable generation and expectation value math."""
    wire_list = list(wire_tuple)

    p_mmd = (1 - jnp.exp(-1 / (2 * sigma_val**2))) / 2
    visible_ops = jnp.array(
        jax.random.binomial(subkey, 1, p_mmd, shape=(n_ops, len(wire_tuple))),
        dtype=jnp.float64,
    )

    all_ops = jnp.zeros((n_ops, n_qubits), dtype=jnp.float64)
    all_ops = all_ops.at[:, wire_list].set(visible_ops)

    pauli_obs = _binary_ops_to_pauli_int(all_ops)

    # Because this outer function is JITted, JAX will trace through expval_func
    # and compile it directly into the same optimized execution graph.
    tr_iqp, tr_iqp_std_err = expval_func(
        params=params,
        observables=pauli_obs,
        key=eval_key,
        n_samples=effective_samples,
        init_state=effective_init_state,
    )

    return _compute_single_mmd(
        tr_iqp,
        tr_iqp_std_err,
        ground_truth,
        visible_ops,
        effective_samples,
        sqrt_loss,
    )


def mmd_loss(
    params: ArrayLike,
    config: CircuitConfig,
    ground_truth: ArrayLike,
    sigma: float | Sequence[float],
    n_ops: int,
    key: ArrayLike | None = None,
    n_samples: int | None = None,
    init_state: tuple[ArrayLike, ArrayLike] | None = None,
    wires: Sequence[int] | None = None,
    sqrt_loss: bool = False,
    return_per_sigma: bool = False,
) -> jnp.ndarray | list[jnp.ndarray]:
    """Estimate MMD loss using a ``CircuitConfig`` and expval backend.

    Args:
        params (ArrayLike): Trainable circuit parameters.
        config (CircuitConfig): Circuit configuration used to build the expectation value function.
        ground_truth (ArrayLike): Binary training samples with shape ``(m, n_qubits)``.
        sigma (float | Sequence[float]): Kernel bandwidth(s).
        n_ops (int): Number of random operators used for estimation.
        key (ArrayLike | None): Optional runtime PRNG key override (useful for training loops).
        n_samples (int | None): Optional runtime sample-count override.
        init_state (tuple[ArrayLike, ArrayLike] | None): Optional runtime initial state override.
        wires (Sequence[int] | None): Measured wires. If ``None``, all wires are used.
        sqrt_loss (bool): Whether to return ``sqrt(abs(MMD^2))``.
        return_per_sigma (bool): If ``True``, return per-sigma estimates instead of the mean.

    Returns:
        jnp.ndarray | list[jnp.ndarray]: Scalar average across ``sigma`` values by default,
        or list of per-sigma estimates when ``return_per_sigma=True``.

    Raises:
        ValueError: If effective ``n_samples <= 1``.
    """
    effective_samples = config.n_samples if n_samples is None else n_samples
    effective_init_state = config.init_state if init_state is None else init_state
    if effective_samples <= 1:
        raise ValueError("n_samples must be greater than 1")

    active_key = config.key if key is None else key
    n_qubits = config.n_qubits

    wire_tuple = tuple(range(n_qubits)) if wires is None else tuple(wires)
    sigma_list = [sigma] if isinstance(sigma, (int, float)) else list(sigma)
    ground_truth = jnp.asarray(ground_truth)

    expval_func = build_expval_func(config)
    losses = []

    for sigma_val in sigma_list:
        active_key, subkey, eval_key = jax.random.split(active_key, 3)

        loss_val = _compute_loss_for_sigma(
            sigma_val=sigma_val,
            subkey=subkey,
            eval_key=eval_key,
            params=params,
            ground_truth=ground_truth,
            effective_init_state=effective_init_state,
            n_ops=n_ops,
            n_qubits=n_qubits,
            wire_tuple=wire_tuple,
            effective_samples=effective_samples,
            sqrt_loss=sqrt_loss,
            expval_func=expval_func,
        )
        losses.append(loss_val)

    if return_per_sigma:
        return losses
    return jnp.mean(jnp.stack(losses))
