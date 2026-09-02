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
"""Maximum Mean Discrepancy (MMD) loss for qubit circuits.

This module compares the output of a qubit circuit to a dataset of
bitstrings. It samples Pauli-Z observables from an RBF (Radial Basis Function) kernel distribution,
estimates their expectation values, and combines the results into an MMD loss.

The loss consumes an :class:`~pennylane.labs.tcdq.Estimator` rather than a
particular simulator, so it works with any
:class:`~pennylane.labs.tcdq.TCDQSimulator` that provides a Pauli-Z capable
estimator over qubits.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from .base import Estimator, ObservableAlgebra
from .iqp import CircuitConfig, _simulator_from_config


@dataclass(frozen=True)
class MMDConfig:
    r"""Hyperparameters for the qubit Maximum Mean Discrepancy (MMD) loss.

    The MMD measures how well the circuit's output distribution matches a
    target dataset.

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


def _resolve_wires(wires: Sequence[int] | None, n_wires: int) -> tuple[int, ...]:
    """Normalize and validate the visible-wire selection.

    Args:
        wires (Sequence[int] | None): Requested visible wires, or ``None`` for all.
        n_wires (int): Total number of wires available.

    Returns:
        tuple[int, ...]: The validated visible wires.

    Raises:
        ValueError: If a wire index is out of range or repeated.
    """
    wire_tuple = tuple(range(n_wires)) if wires is None else tuple(wires)

    for wire in wire_tuple:
        if wire < 0 or wire >= n_wires:
            raise ValueError(f"Wire index {wire} out of range for {n_wires} wires")

    if len(set(wire_tuple)) != len(wire_tuple):
        raise ValueError("wires must not contain duplicates")

    return wire_tuple


def _resolve_bandwidths(bandwidth: float | Sequence[float]) -> list[float]:
    """Normalize the bandwidth field to a non-empty list.

    Raises:
        ValueError: If the resulting list is empty.
    """
    bandwidths = [bandwidth] if isinstance(bandwidth, (int, float)) else list(bandwidth)

    if len(bandwidths) == 0:
        raise ValueError("bandwidth must not be empty")

    return bandwidths


def _validate_target_data(target_data: ArrayLike, n_visible: int) -> jnp.ndarray:
    """Check that a target dataset matches the visible wires.

    Raises:
        ValueError: If the dataset is not 2-D, has the wrong number of columns,
            or has fewer than two samples.
    """
    data = jnp.asarray(target_data)

    if data.ndim != 2:
        raise ValueError(f"target_data must be 2-D, got shape {data.shape}")

    if data.shape[1] != n_visible:
        raise ValueError(
            f"target_data has {data.shape[1]} columns but expected "
            f"{n_visible} (number of visible wires)"
        )

    if data.shape[0] < 2:
        raise ValueError(f"target_data must have at least 2 samples, got {data.shape[0]}")

    return data


@jax.jit
def _binary_ops_to_pauli_int(binary_ops: ArrayLike) -> jnp.ndarray:
    """Map binary operator entries to Pauli integer codes (0 → I, 1 → Z=3)."""
    ops = jnp.asarray(binary_ops, dtype=jnp.int32)
    return jnp.where(ops == 1, 3, 0).astype(jnp.int32)


@partial(jax.jit, static_argnames=["sqrt_loss"])
def _compute_single_mmd(
    model_expvals: jnp.ndarray,
    model_expvals_variances: jnp.ndarray,
    target_data: jnp.ndarray,
    visible_ops: jnp.ndarray,
    sqrt_loss: bool,
) -> jnp.ndarray:
    """Core, heavily JIT-compiled math for MMD calculation."""
    tr_train = jnp.mean(1 - 2 * ((target_data @ visible_ops.T) % 2), axis=0)
    m = target_data.shape[0]

    result = model_expvals**2 - model_expvals_variances
    result = result - 2 * model_expvals * tr_train + (tr_train * tr_train * m - 1) / (m - 1)

    reduced = jnp.mean(result)
    return jnp.sqrt(jnp.abs(reduced)) if sqrt_loss else reduced


# pylint: disable=too-many-arguments
@partial(
    jax.jit,
    static_argnames=["n_ops", "n_wires", "wire_tuple", "sqrt_loss", "estimator"],
)
def _loss_for_bandwidth(
    bandwidth: float,
    obs_key: jnp.ndarray,
    eval_key: jnp.ndarray,
    params: jnp.ndarray,
    target_data: jnp.ndarray,
    n_ops: int,
    n_wires: int,
    wire_tuple: tuple[int, ...],
    sqrt_loss: bool,
    estimator: Estimator,
) -> jnp.ndarray:
    """JIT-compiled step that fuses observable generation and expectation value math."""
    p_mmd = (1 - jnp.exp(-1 / (2 * bandwidth**2))) / 2
    visible_ops = jnp.array(
        jax.random.binomial(obs_key, 1, p_mmd, shape=(n_ops, len(wire_tuple))),
        dtype=jnp.float64,
    )

    all_ops = jnp.zeros((n_ops, n_wires), dtype=jnp.float64)
    all_ops = all_ops.at[:, list(wire_tuple)].set(visible_ops)

    model_expvals, model_expvals_variances = estimator(
        params, _binary_ops_to_pauli_int(all_ops), key=eval_key
    )

    return _compute_single_mmd(
        model_expvals,
        model_expvals_variances,
        target_data,
        visible_ops,
        sqrt_loss,
    )


def build_mmd_loss(estimator: Estimator, mmd_config: MMDConfig) -> Callable:
    """Build an RBF-kernel MMD loss on top of any Pauli-Z capable estimator.

    The returned loss measures how far a circuit's output distribution over
    bitstrings is from an empirical target dataset. For each bandwidth it
    samples Pauli-Z observables from the RBF kernel's spectral distribution,
    asks ``estimator`` for their expectation values, and combines the results
    into an unbiased MMD estimate.

    Any :class:`~pennylane.labs.tcdq.TCDQSimulator` exposing an estimator that
    declares the ``PAULI_Z`` or ``PAULI`` observable algebra over qubits can be
    used here, not only :class:`~pennylane.labs.tcdq.IQPSimulator`.

    Args:
        estimator (Estimator): An estimator obtained from
            :meth:`~pennylane.labs.tcdq.TCDQSimulator.build_estimator`.
        mmd_config (MMDConfig): Hyperparameters for the MMD computation,
            including the RBF bandwidth and number of observables.

    Returns:
        Callable: A function with signature ``loss_fn(params, target_data, key)``
        returning a scalar MMD² estimate averaged over all bandwidths, or a list
        of per-bandwidth estimates when ``mmd_config.return_per_bandwidth=True``.

    Raises:
        TypeError: If ``estimator`` is not an
            :class:`~pennylane.labs.tcdq.Estimator`, or does not measure
            Pauli-Z observables.
        ValueError: If the estimator is not defined over qubits, if
            ``mmd_config.n_ops < 1``, if ``mmd_config.bandwidth`` is empty, or
            if ``mmd_config.wires`` contains duplicates or out-of-range indices.

    **Example**

    >>> import jax
    >>> import jax.numpy as jnp
    >>> import numpy as np
    >>> from pennylane.labs.tcdq import (
    ...     IQPSimulator, MMDConfig, build_mmd_loss, create_local_gates, median_heuristic
    ... )
    >>> gates = create_local_gates(4, max_weight=2)
    >>> sim = IQPSimulator(
    ...     gates=gates, n_qubits=4, n_samples=1000, key=jax.random.PRNGKey(0)
    ... )
    >>> target = np.random.binomial(1, 0.5, size=(100, 4))
    >>> loss_fn = build_mmd_loss(
    ...     sim.build_estimator("pauli_expval"),
    ...     MMDConfig(bandwidth=median_heuristic(target), n_ops=50),
    ... )
    >>> loss_fn(jnp.zeros(len(gates)), target, jax.random.PRNGKey(1)).shape
    ()

    .. seealso::

        :class:`~pennylane.labs.tcdq.IQPSimulator`,
        `Section 3.3 of IQPopt: Fast optimization of instantaneous quantum polynomial circuits in JAX <https://arxiv.org/pdf/2501.04776>`_
    """
    if not isinstance(estimator, Estimator):
        raise TypeError(
            f"build_mmd_loss expects a tcdq Estimator, got {type(estimator).__name__}. "
            "Build one with TCDQSimulator.build_estimator(name)."
        )

    if estimator.algebra not in (ObservableAlgebra.PAULI_Z, ObservableAlgebra.PAULI):
        raise TypeError(
            f"The RBF-kernel MMD loss samples Pauli-Z observables, but estimator "
            f"{estimator.name!r} declares the {estimator.algebra.value!r} "
            f"observable algebra. It must declare 'pauli_z' or 'pauli'."
        )

    local_dims = estimator.local_dims
    if set(local_dims) != {2}:
        raise ValueError(
            f"The RBF-kernel MMD loss is defined over qubits, but estimator "
            f"{estimator.name!r} has local dimensions {local_dims}."
        )

    if mmd_config.n_ops < 1:
        raise ValueError("n_ops must be at least 1")

    n_wires = estimator.n_wires
    wire_tuple = _resolve_wires(mmd_config.wires, n_wires)
    bandwidths = _resolve_bandwidths(mmd_config.bandwidth)

    def loss_fn(
        params: ArrayLike,
        target_data: ArrayLike,
        key: ArrayLike,
    ) -> jnp.ndarray | list[jnp.ndarray]:
        """Estimate the empirical MMD loss for one parameter setting.

        Args:
            params (ArrayLike): Trainable circuit parameters.
            target_data (ArrayLike): Binary dataset of shape ``(m, n_visible)``
                whose rows are bitstring samples on the visible wires.
            key (ArrayLike): JAX PRNG key. Each bandwidth consumes independent
                randomness for observable sampling and circuit evaluation.

        Returns:
            jnp.ndarray | list[jnp.ndarray]: A scalar mean across bandwidths, or
            a list of per-bandwidth values when ``return_per_bandwidth`` is set.

        Raises:
            ValueError: If ``target_data`` does not match the visible wires.
        """
        data = _validate_target_data(target_data, len(wire_tuple))
        losses = []

        for bandwidth in bandwidths:
            key, obs_key, eval_key = jax.random.split(key, 3)
            losses.append(
                _loss_for_bandwidth(
                    bandwidth=bandwidth,
                    obs_key=obs_key,
                    eval_key=eval_key,
                    params=params,
                    target_data=data,
                    n_ops=mmd_config.n_ops,
                    n_wires=n_wires,
                    wire_tuple=wire_tuple,
                    sqrt_loss=mmd_config.sqrt_loss,
                    estimator=estimator,
                )
            )

        if mmd_config.return_per_bandwidth:
            return losses
        return jnp.mean(jnp.stack(losses))

    return loss_fn


def mmd_loss(
    params: ArrayLike,
    circuit_config: CircuitConfig,
    mmd_config: MMDConfig,
    target_data: ArrayLike,
    key: ArrayLike | None = None,
) -> jnp.ndarray | list[jnp.ndarray]:
    """Compute the MMD loss between a qubit IQP circuit and a target dataset.

    .. warning::

        This function is superseded by :func:`build_mmd_loss`, which accepts an
        estimator from any :class:`~pennylane.labs.tcdq.TCDQSimulator` instead
        of a fixed :class:`~pennylane.labs.tcdq.CircuitConfig`. It is kept for
        backwards compatibility and will be removed.

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
        ValueError: If ``circuit_config.n_samples <= 1``, or if the MMD
            hyperparameters or ``target_data`` are invalid.

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

        :func:`build_mmd_loss`,
        `Section 3.3 of IQPopt: Fast optimization of instantaneous quantum polynomial circuits in JAX <https://arxiv.org/pdf/2501.04776>`_
    """
    estimator = _simulator_from_config(circuit_config).build_estimator("pauli_expval")
    loss_fn = build_mmd_loss(estimator, mmd_config)
    return loss_fn(params, target_data, circuit_config.key if key is None else key)
