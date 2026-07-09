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
"""Training helpers for the TCDQ workflow.

This module provides a high-level training loop and a lower-level iterator for
users who want more control over logging, stopping criteria, or validation.

Supported optimizers (via `jaxopt <https://jaxopt.github.io/>`_):

- ``"Adam"``
- ``"GradientDescent"``
- ``"BFGS"``
"""

import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import partial
from inspect import signature
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

try:
    import jaxopt
    import optax
    from tqdm import tqdm
except (ModuleNotFoundError, ImportError) as import_error:
    pass


@dataclass
class TrainingOptions:
    """Options controlling the behaviour of :func:`train` and :func:`training_iterator`.

    Args:
        unroll_steps (int): Number of optimization steps fused into a single
            ``jax.lax.scan`` call. Higher values reduce Python overhead and
            speed up training, but produce less frequent progress updates.
            Defaults to ``1``.
        val_kwargs (dict[str, Any] | None): Keyword arguments passed to the loss
            function when evaluating validation loss. If ``None`` (the default),
            no validation loss is computed.
        convergence_interval (int): The training loop compares the mean loss over
            the most recent ``convergence_interval`` steps against the preceding
            interval. If the improvement is smaller than half the standard error,
            training stops early. Defaults to ``100``.
        random_state (int): Integer seed used to create the JAX PRNG key for
            stochastic loss functions. Defaults to ``666``.
        opt_jit (bool): Whether to JIT-compile the optimizer's internal update
            step. Usually ``False`` is sufficient because the outer scan is
            already JIT-compiled. Defaults to ``False``.

    **Example**

    >>> options = TrainingOptions(unroll_steps=20, convergence_interval=50)
    >>> result = train(
    ...     optimizer="Adam", loss=my_loss, stepsize=0.01,
    ...     n_iters=200, loss_kwargs={"params": params}, options=options,
    ... )
    """

    unroll_steps: int = 1
    val_kwargs: dict[str, Any] | None = None
    convergence_interval: int = 100
    random_state: int = 666
    opt_jit: bool = False


class TrainingResult(NamedTuple):
    """Results returned by :func:`train` after the optimization loop completes.

    Args:
        final_params (jnp.ndarray): Optimized circuit parameters after training.
        losses (jnp.ndarray): Training loss recorded at every optimization step,
            shape ``(n_steps,)``.
        val_losses (jnp.ndarray): Validation loss at every step (all zeros if no
            ``val_kwargs`` was provided), shape ``(n_steps,)``.
        run_time (float): Wall-clock time of the training loop in seconds.

    **Example**

    >>> result = train(optimizer="Adam", loss=my_loss, stepsize=0.01,
    ...               n_iters=100, loss_kwargs={"params": params})
    >>> result.final_params.shape
    (10,)
    >>> len(result.losses)
    100
    """

    final_params: jnp.ndarray
    losses: jnp.ndarray
    val_losses: jnp.ndarray
    run_time: float


class BatchResult(NamedTuple):
    """Intermediate result yielded by :func:`training_iterator` after each unrolled batch.

    Each ``BatchResult`` covers consecutive optimization steps defined by ``unroll_steps``.

    Args:
        params (jnp.ndarray): Circuit parameters at the end of this batch.
        state (jnp.ndarray): Internal optimizer state (e.g., Adam moments).
        key (jax.Array): Updated PRNG key for the next training batch.
        key_val (jax.Array): Updated PRNG key for the next validation evaluation.
        losses (jnp.ndarray): Training loss values for each step in this batch,
            shape ``(unroll_steps,)``.
        val_losses (jnp.ndarray): Validation loss values for each step in this
            batch (zeros if validation is disabled), shape ``(unroll_steps,)``.
    """

    params: jnp.ndarray
    state: jnp.ndarray
    key: jax.Array
    key_val: jax.Array
    losses: jnp.ndarray
    val_losses: jnp.ndarray


def _prepare_loss_function(loss: Callable) -> Callable:
    """Wrap a loss function so it always accepts a ``key`` argument.

    Args:
        loss (Callable): The original loss function.

    Returns:
        Callable: Wrapped loss function with a uniform call signature.
    """
    if "key" in signature(loss).parameters:
        return loss

    return lambda params, key, **kwargs: loss(params, **kwargs)


def _create_optimizer(name: str, loss_fn: Callable, stepsize: float, opt_jit: bool):
    """Create the requested JAX optimizer.

    Args:
        name (str): The name of the optimizer to create ('GradientDescent', 'Adam', 'BFGS').
        loss_fn (Callable): The loss function to minimize.
        stepsize (float): The step size (learning rate) for the optimizer.
        opt_jit (bool): Whether to JIT compile the optimizer update.

    Returns:
        jaxopt.Optimizer: An instance of the requested optimizer.

    Raises:
        ValueError: If the optimizer name is not recognized.
    """
    if name == "GradientDescent":
        return jaxopt.GradientDescent(loss_fn, stepsize=stepsize, verbose=False, jit=opt_jit)
    if name == "Adam":
        return jaxopt.OptaxSolver(loss_fn, optax.adam(stepsize), verbose=False, jit=opt_jit)
    if name == "BFGS":
        return jaxopt.BFGS(loss_fn, verbose=False, jit=opt_jit)

    raise ValueError(
        f"Optimizer {name} not recognized. Choose from 'Adam', 'BFGS', 'GradientDescent'."
    )


def _check_convergence(losses: jnp.ndarray, convergence_interval: int) -> bool:
    """Check convergence by comparing two recent windows of loss values.

    Args:
        losses (jnp.ndarray): Array of recorded loss values.
        convergence_interval (int): number of steps to look back when comparing means.

    Returns:
        bool: True if converged, False otherwise.
    """
    recent = losses[-convergence_interval:]
    previous = losses[-2 * convergence_interval : -convergence_interval]

    avg1 = jnp.mean(recent)
    avg2 = jnp.mean(previous)
    std1 = jnp.std(recent)

    # Stop if improvement is statistically insignificant or loss increases
    cond1 = jnp.abs(avg2 - avg1) <= std1 / jnp.sqrt(convergence_interval) / 2
    cond2 = avg1 > avg2

    return cond1 or cond2


def _update_step_scan(carry, _, opt, loss_fn, loss_kwargs, val_kwargs, validation, optimizer_name):
    # pylint: disable=too-many-arguments
    """Run one optimizer step inside the scanned training loop.

    Args:
        carry (list): List of carried state [params, state, key, key_val].
        _ (Any): Unused variable to accommodate `jax.lax.scan`
        opt (jaxopt.Optimizer): The optimizer instance.
        loss_fn (Callable): The loss function.
        loss_kwargs (dict[str, Any]): Arguments for the loss function.
        val_kwargs (dict[str, Any]): Arguments for the validation function.
        validation (bool): Whether validation is enabled.
        optimizer_name (str): Name of the optimizer.

    Returns:
        tuple[list, list]: Tuple containing:
            - The new carry state [params, state, key2, key2_val].
            - The stacked list [training_loss, validation_loss].
    """
    params, state, key, key_val = carry
    key1, key2 = jax.random.split(key, 2)
    key1_val, key2_val = jax.random.split(key_val, 2)

    params, state = opt.update(params, state, **loss_kwargs, key=key1)

    v_loss = loss_fn(params, **val_kwargs, key=key1_val) if validation else 0.0

    if optimizer_name == "GradientDescent":
        t_loss = loss_fn(params, **loss_kwargs, key=key1)
    else:
        t_loss = state.value

    return [params, state, key2, key2_val], [t_loss, v_loss]


def training_iterator(
    optimizer: str,
    loss: Callable,
    stepsize: float,
    loss_kwargs: dict[str, Any],
    options: TrainingOptions | None = None,
) -> Iterator[BatchResult]:
    """Create an infinite iterator that yields optimization results one batch at a time.

    This function is for users who need custom convergence
    logic, logging, or early-stopping criteria. Each iteration advances the
    optimizer by ``unroll_steps`` steps and returns a :class:`BatchResult`
    containing the updated parameters and per-step losses.

    Args:
        optimizer (str): Name of the optimizer. One of ``"Adam"``,
            ``"GradientDescent"``, or ``"BFGS"``.
        loss (Callable): Loss function with signature
            ``loss(params, **loss_kwargs)`` or
            ``loss(params, key=..., **loss_kwargs)`` for stochastic losses.
        stepsize (float): Learning rate (step size) for the optimizer.
        loss_kwargs (dict[str, Any]): Keyword arguments forwarded to ``loss``
            at every step. Must include ``"params"`` as the initial parameter
            array.
        options (TrainingOptions | None): Training configuration. Uses
            defaults if ``None``.

    Yields:
        BatchResult: One result per ``unroll_steps`` optimization steps.

    **Example**

    >>> import jax.numpy as jnp
    >>> from pennylane.labs.tcdq import training_iterator, TrainingOptions
    >>> def quadratic(params):
    ...     return jnp.sum(params ** 2)
    >>> iterator = training_iterator(
    ...     optimizer="Adam", loss=quadratic, stepsize=0.01,
    ...     loss_kwargs={"params": jnp.ones(3)},
    ...     options=TrainingOptions(unroll_steps=10),
    ... )
    >>> batch = next(iterator)
    >>> batch.losses.shape
    (10,)
    """
    options = options or TrainingOptions()
    unroll_steps = max(1, options.unroll_steps)

    wrapped_loss = _prepare_loss_function(loss)
    opt = _create_optimizer(optimizer, wrapped_loss, stepsize, options.opt_jit)

    fixed_loss_kwargs = loss_kwargs.copy()
    params_init = fixed_loss_kwargs.pop("params")

    validation = options.val_kwargs is not None
    fixed_val_kwargs = options.val_kwargs.copy() if validation else {}

    key = jax.random.PRNGKey(options.random_state)
    key1, key2 = jax.random.split(key, 2)
    key = fixed_loss_kwargs.pop("key", key1)
    key_val = fixed_val_kwargs.pop("key", key2)

    state = opt.init_state(params_init, **fixed_loss_kwargs, key=key)
    params = params_init

    scan_fn = partial(
        _update_step_scan,
        opt=opt,
        loss_fn=wrapped_loss,
        loss_kwargs=fixed_loss_kwargs,
        val_kwargs=fixed_val_kwargs,
        validation=validation,
        optimizer_name=optimizer,
    )

    @jax.jit
    def step_batch(params, state, key, key_val):
        carry = [params, state, key, key_val]
        carry, [chunk_losses, chunk_vals] = jax.lax.scan(scan_fn, carry, jnp.arange(unroll_steps))
        return BatchResult(
            params=carry[0],
            state=carry[1],
            key=carry[2],
            key_val=carry[3],
            losses=chunk_losses,
            val_losses=chunk_vals,
        )

    while True:
        result = step_batch(params, state, key, key_val)

        params = result.params
        state = result.state
        key = result.key
        key_val = result.key_val

        yield result


def train(
    optimizer: str,
    loss: Callable,
    stepsize: float,
    n_iters: int,
    loss_kwargs: dict[str, Any],
    options: TrainingOptions | None = None,
) -> TrainingResult:
    # pylint: disable=too-many-arguments
    """Run a complete optimization loop with automatic convergence detection.

    This is the high-level training entry point. It provides progress-bar display,
    loss history accumulation, and an automatic early-stopping check based on
    the ``convergence_interval`` setting in :class:`TrainingOptions`.

    Args:
        optimizer (str): Name of the optimizer. One of ``"Adam"``,
            ``"GradientDescent"``, or ``"BFGS"``.
        loss (Callable): Loss function with signature
            ``loss(params, **loss_kwargs)`` or
            ``loss(params, key=..., **loss_kwargs)`` for stochastic losses.
        stepsize (float): Learning rate (step size) for the optimizer.
        n_iters (int): Maximum number of optimization steps. Training may
            stop earlier if convergence is detected.
        loss_kwargs (dict[str, Any]): Keyword arguments forwarded to ``loss``.
            Must include ``"params"`` as the initial parameter array.
        options (TrainingOptions | None): Training configuration. Uses
            defaults if ``None``.

    Returns:
        TrainingResult: A named tuple containing the optimized parameters,
        full loss history, validation loss history, and wall-clock runtime.

    **Example**

    >>> import jax, jax.numpy as jnp
    >>> from pennylane.labs.tcdq import train, TrainingOptions
    >>> def quadratic(params):
    ...     return jnp.sum(params ** 2)
    >>> result = train(
    ...     optimizer="Adam",
    ...     loss=quadratic,
    ...     stepsize=0.1,
    ...     n_iters=50,
    ...     loss_kwargs={"params": jnp.ones(3)},
    ...     options=TrainingOptions(unroll_steps=10),
    ... )
    >>> result.final_params.shape
    (3,)
    """

    options = options or TrainingOptions()

    unroll_steps = max(1, options.unroll_steps)
    total_batches = (n_iters + unroll_steps - 1) // unroll_steps

    start_time = time.time()

    loss_acc = []
    val_loss_acc = []

    converged = False
    final_params = loss_kwargs["params"]

    iterator = training_iterator(
        optimizer=optimizer, loss=loss, stepsize=stepsize, loss_kwargs=loss_kwargs, options=options
    )

    with tqdm(total=n_iters, desc="Training Progress") as pbar:

        for i, batch_result in enumerate(iterator):
            if i >= total_batches:
                break

            final_params = batch_result.params
            loss_acc.append(batch_result.losses)
            val_loss_acc.append(batch_result.val_losses)

            curr_loss = batch_result.losses[-1]
            pbar.set_postfix(
                {"loss": f"{curr_loss:.6f}", "elapsed": f"{time.time() - start_time:.1f}s"}
            )
            pbar.update(unroll_steps)

            current_step = (i + 1) * unroll_steps

            # Check based on validation loss if available, else training loss
            metric_acc = val_loss_acc if options.val_kwargs else loss_acc

            history_needed = 2 * options.convergence_interval

            if current_step > history_needed:
                recent_history = jnp.concatenate(
                    metric_acc[-10:]
                )  # Grab last 10 chunks (heuristic)

                if len(recent_history) >= history_needed:
                    if _check_convergence(recent_history, options.convergence_interval):
                        print(f"Training converged after {current_step} steps")
                        converged = True
                        break

    if not converged:
        print(f"Training has not converged after {n_iters} steps")

    all_losses = jnp.concatenate(loss_acc) if loss_acc else jnp.array([])
    all_val_losses = jnp.concatenate(val_loss_acc) if val_loss_acc else jnp.array([])

    if len(all_losses) > n_iters:
        all_losses = all_losses[:n_iters]
        all_val_losses = all_val_losses[:n_iters]

    return TrainingResult(
        final_params=final_params,
        losses=all_losses,
        val_losses=all_val_losses,
        run_time=time.time() - start_time,
    )
