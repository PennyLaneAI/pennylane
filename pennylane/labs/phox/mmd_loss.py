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

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from .expval_functions import CircuitConfig, build_expval_func


@dataclass(frozen=True)
class MMDConfig:
    """Hyperparameters for Maximum Mean Discrepancy (MMD) loss calculation.

    Args:
        bandwidth (float | Sequence[float]): RBF kernel bandwidth(s) for the MMD calculation.
            If a sequence is provided, the loss will be computed for each bandwidth and either
            averaged or returned as a list depending on ``return_per_bandwidth``.
        n_ops (int): The number of binary operators (observables) to sample when approximating
            the MMD loss.
        wires (Sequence[int] | None, optional): The specific wires (qubits) to evaluate the
            MMD over. If ``None``, the calculation defaults to using all available qubits.
            Defaults to ``None``.
        sqrt_loss (bool, optional): If ``True``, computes the square root of the absolute
            reduced MMD loss. Defaults to ``False``.
        return_per_bandwidth (bool, optional): If ``True``, returns a list containing the
            individual loss estimates for each bandwidth. If ``False``, returns the scalar
            average across all specified bandwidths. Defaults to ``False``.
    """

    bandwidth: float | Sequence[float]
    n_ops: int
    wires: Sequence[int] | None = None
    sqrt_loss: bool = False
    return_per_bandwidth: bool = False


@jax.jit
def _binary_ops_to_pauli_int(binary_ops: ArrayLike) -> jnp.ndarray:
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
        init_state=effective_init_state,
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
    """Estimate MMD loss using configuration dataclasses.

    Args:
        params (ArrayLike): Trainable circuit parameters.
        circuit_config (CircuitConfig): Circuit configuration used to build the expval function.
        mmd_config (MMDConfig): Hyperparameters for the MMD computation.
        target_data (ArrayLike): Binary target samples with shape ``(m, n_qubits)``.
        key (ArrayLike | None): Optional runtime PRNG key override for the training loop.

    Returns:
        jnp.ndarray | list[jnp.ndarray]: Scalar average across ``sigma`` values by default,
        or list of per-sigma estimates when ``return_per_sigma=True``.

    Raises:
        ValueError: If effective ``n_samples <= 1``.
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
            effective_init_state=circuit_config.init_state,
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
