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
"""Maximum Mean Discrepancy (MMD) loss for qubit IQP circuits.

This module compares the output of a qubit IQP circuit to a dataset of
bitstrings. It samples Pauli-Z observables from an RBF (Radial Basis Function) kernel distribution,
estimates their expectation values, and combines the results into an MMD loss.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from .expval_functions import CircuitConfig, build_expval_func


@dataclass(frozen=True)
class MMDConfig:
    """Hyperparameters for the qubit Maximum Mean Discrepancy (MMD) loss.

    The MMD measures how well the circuit's output distribution matches a
    target dataset.

    Args:
        bandwidth (float | Sequence[float]): Width of the Gaussian kernel
            used to compare distributions. Small values make the loss
            sensitive to fine-grained differences; large values emphasize
            broad structure. A good starting point is the median pairwise
            distance of the target data (see :func:`median_heuristic`). If a
            sequence is provided, the loss is computed independently for each
            bandwidth and the results are averaged (or returned individually
            when ``return_per_bandwidth=True``).
        n_ops (int): Number of sampled observables per bandwidth. Larger
            values reduce estimator variance.
        wires (Sequence[int] | None): Subset of qubit indices to include in
            the loss. If ``None`` (default), all qubits are used.
        sqrt_loss (bool): If ``True``, return ``sqrt(|MMD²|)`` instead of
            ``MMD²``. Defaults to ``False``.
        return_per_bandwidth (bool): If ``True``, return a list of
            per-bandwidth loss values instead of their scalar average.
            Defaults to ``False``.

    **Example**

    >>> from pennylane.labs.tcdq import MMDConfig, median_heuristic
    >>> import numpy as np
    >>> target_data = np.random.binomial(1, 0.5, size=(200, 6))
    >>> bw = median_heuristic(target_data)
    >>> config = MMDConfig(bandwidth=bw, n_ops=64)

    .. seealso::

        `Section 5, Graph-Kernel MMD Loss <https://github.com/PennyLaneAI/pennylane/blob/port_tcdq_docs_pr/pennylane/labs/tcdq/notes.md#5-graph-kernel-mmd-loss>`_
    """

    bandwidth: float | Sequence[float]
    n_ops: int
    wires: Sequence[int] | None = None
    sqrt_loss: bool = False
    return_per_bandwidth: bool = False


def median_heuristic(samples: ArrayLike) -> float:
    """Compute a bandwidth for the RBF kernel using the median pairwise distance heuristic.

    This is a widely used data-driven method for selecting the bandwidth
    (length-scale) of a radial basis function (RBF) kernel: set the bandwidth
    equal to the median of all non-zero pairwise Euclidean distances in the
    dataset. This ensures the kernel is sensitive at the typical inter-sample
    scale.

    Args:
        samples (ArrayLike): Dataset array of shape ``(n_samples, n_features)``.
            For qubit circuits, this is typically a binary matrix of bitstrings.

    Returns:
        float: The median non-zero pairwise Euclidean distance. Returns ``1.0``
        if all pairwise distances are zero (e.g., identical samples).

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


# pylint: disable=too-many-arguments
@partial(jax.jit, static_argnames=["n_samples", "sqrt_loss"])
def _compute_single_mmd(
    model_expvals: jnp.ndarray,
    model_expvals_std_err: jnp.ndarray,
    target_data: jnp.ndarray,
    visible_ops: jnp.ndarray,
    n_samples: int,
    sqrt_loss: bool,
) -> jnp.ndarray:
    """Core, heavily JIT-compiled math for MMD calculation."""
    model_expvals_std_err = jax.lax.stop_gradient(model_expvals_std_err)
    correction = (model_expvals**2 + (n_samples - 1) * model_expvals_std_err**2) / n_samples

    tr_train = jnp.mean(1 - 2 * ((target_data @ visible_ops.T) % 2), axis=0)
    m = target_data.shape[0]

    result = (model_expvals * model_expvals - correction) * n_samples / (n_samples - 1)
    result = result - 2 * model_expvals * tr_train + (tr_train * tr_train * m - 1) / (m - 1)

    reduced = jnp.mean(result)
    return jnp.sqrt(jnp.abs(reduced)) if sqrt_loss else reduced


# pylint: disable=too-many-arguments
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
def _compute_loss_for_bandwidth(
    bandwidth: float,
    subkey: jnp.ndarray,
    eval_key: jnp.ndarray,
    params: jnp.ndarray,
    target_data: jnp.ndarray,
    effective_init_state_elems: jnp.ndarray | None,
    effective_init_state_amps: jnp.ndarray | None,
    n_ops: int,
    n_qubits: int,
    wire_tuple: tuple[int, ...],
    effective_samples: int,
    sqrt_loss: bool,
    expval_func: Callable,
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

    model_expvals, model_expvals_std_err = expval_func(
        gates_params=params,
        observables=pauli_obs,
        key=eval_key,
        n_samples=effective_samples,
        init_state_elems=effective_init_state_elems,
        init_state_amps=effective_init_state_amps,
    )

    return _compute_single_mmd(
        model_expvals,
        model_expvals_std_err,
        target_data,
        visible_ops,
        effective_samples,
        sqrt_loss,
    )


def mmd_loss(
    params: ArrayLike,
    circuit_config: CircuitConfig,
    mmd_config: MMDConfig,
    target_data: ArrayLike,
    key: ArrayLike | None = None,
) -> jnp.ndarray | list[jnp.ndarray]:
    """Compute the MMD loss between a qubit IQP circuit and a target dataset.

    This function estimates how far the circuit's output distribution is from
    the empirical distribution defined by ``target_data``. At each call it:

    * Samples ``n_ops`` random Pauli-Z observables according to the RBF
      kernel defined by ``mmd_config.bandwidth``.
    * Estimates the circuit's expectation values for those observables
      (via :func:`~pennylane.labs.tcdq.build_expval_func`).
    * Computes the matching empirical averages directly from ``target_data``.
    * Combines these into an unbiased MMD² estimator.

    The result is differentiable with respect to ``params`` via JAX
    autodiff, making it suitable as a training objective.

    Args:
        params (ArrayLike): Trainable circuit parameters, shape ``(n_params,)``.
        circuit_config (CircuitConfig): Circuit description specifying the gate
            structure, number of qubits, and sample count. See
            :class:`~pennylane.labs.tcdq.CircuitConfig` for how to construct one.
        mmd_config (MMDConfig): Hyperparameters for the MMD computation,
            including the RBF bandwidth and number of observables. See
            :class:`MMDConfig`.
        target_data (ArrayLike): Binary dataset of shape ``(m, n_qubits)``
            where each row is a bitstring sample from the target distribution.
        key (ArrayLike | None): Optional JAX PRNG key. If ``None``, uses the
            key stored in ``circuit_config``.

    Returns:
        jnp.ndarray | list[jnp.ndarray]: A scalar MMD² estimate averaged over
        all bandwidths by default, or a list of per-bandwidth estimates when
        ``mmd_config.return_per_bandwidth=True``.

    Raises:
        ValueError: If ``circuit_config.n_samples <= 1``.

    **Example**

    >>> import jax
    >>> import numpy as np
    >>> from pennylane.labs.tcdq import (
    ...     CircuitConfig, MMDConfig, mmd_loss, create_local_gates, median_heuristic
    ... )
    >>> n_qubits = 4
    >>> gates = create_local_gates(n_qubits, max_weight=2)
    >>> config = CircuitConfig(
    ...     gates=gates, n_samples=1000, key=jax.random.PRNGKey(0), n_qubits=n_qubits
    ... )
    >>> target = np.random.binomial(1, 0.5, size=(100, n_qubits))
    >>> bw = median_heuristic(target)
    >>> mmd_cfg = MMDConfig(bandwidth=bw, n_ops=50)
    >>> import jax.numpy as jnp
    >>> params = jnp.zeros(len(gates))
    >>> loss_val = mmd_loss(params, config, mmd_cfg, target)
    >>> loss_val.shape
    ()

    .. seealso::

        :func:`~pennylane.labs.tcdq.build_expval_func`,
        `Section 5, Graph-Kernel MMD Loss <https://github.com/PennyLaneAI/pennylane/blob/port_tcdq_docs_pr/pennylane/labs/tcdq/notes.md#5-graph-kernel-mmd-loss>`_
    """
    effective_samples = circuit_config.n_samples
    if effective_samples <= 1:
        raise ValueError("n_samples must be greater than 1")

    active_key = circuit_config.key if key is None else key
    n_qubits = circuit_config.n_qubits

    wire_tuple = tuple(range(n_qubits)) if mmd_config.wires is None else tuple(mmd_config.wires)

    bandwidth_list = (
        [mmd_config.bandwidth]
        if isinstance(mmd_config.bandwidth, (int, float))
        else list(mmd_config.bandwidth)
    )
    target_data = jnp.asarray(target_data)

    expval_func = build_expval_func(circuit_config)
    losses = []

    for bandwidth in bandwidth_list:
        active_key, subkey, eval_key = jax.random.split(active_key, 3)

        loss_val = _compute_loss_for_bandwidth(
            bandwidth=bandwidth,
            subkey=subkey,
            eval_key=eval_key,
            params=params,
            target_data=target_data,
            effective_init_state_elems=circuit_config.init_state_elems,
            effective_init_state_amps=circuit_config.init_state_amps,
            n_ops=mmd_config.n_ops,
            n_qubits=n_qubits,
            wire_tuple=wire_tuple,
            effective_samples=effective_samples,
            sqrt_loss=mmd_config.sqrt_loss,
            expval_func=expval_func,
        )
        losses.append(loss_val)

    if mmd_config.return_per_bandwidth:
        return losses
    return jnp.mean(jnp.stack(losses))
