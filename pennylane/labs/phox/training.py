import time
from typing import Callable, Optional, Dict, Any, NamedTuple, Iterator
from inspect import signature
from functools import partial
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jaxopt
import optax
from tqdm import tqdm


@dataclass
class TrainingOptions:
    """
    Configuration options for training.

    Args:
        unroll_steps (int): How many optimization steps to run on the GPU before yielding
            control back to Python. Higher = Faster. Lower = More interactive/granular logging.
            Defaults to 1 (slow, good for debugging).
        val_kwargs (dict[str, Any] | None): Arguments for the loss function to be used during validation.
        convergence_interval (int | None): Number of steps over which to check for convergence.
        random_state (int): Seed for PRNGKey.
        opt_jit (bool): Whether to JIT the optimizer creation. Default is False.
    """

    unroll_steps: int = 1
    val_kwargs: dict[str, Any] | None = None
    convergence_interval: int | None = None
    random_state: int = 666
    opt_jit: bool = False


class TrainingResult(NamedTuple):
    """Container for final training results."""

    final_params: Any
    losses: jnp.ndarray
    val_losses: jnp.ndarray
    run_time: float


class BatchResult(NamedTuple):
    """Result from a single batch (unrolled chunk) of training steps."""

    params: Any
    state: Any
    key: Any
    key_val: Any
    losses: jnp.ndarray
    val_losses: jnp.ndarray


def _prepare_loss_function(loss: Callable) -> Callable:
    """
    Wraps the loss function to ensure it accepts a 'key' argument.
    If the original function doesn't accept 'key', we consume and ignore it.

    Args:
        loss (Callable): The original loss function.

    Returns:
        Callable: A wrapped loss function that accepts a ``key`` argument, regardless of whether the original did.
    """
    if "key" in signature(loss).parameters:
        return loss

    return lambda params, key, **kwargs: loss(params, **kwargs)


def _create_optimizer(name: str, loss_fn: Callable, stepsize: float, opt_jit: bool):
    """
    Create the JAX optimizer instance.

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
    elif name == "Adam":
        return jaxopt.OptaxSolver(loss_fn, optax.adam(stepsize), verbose=False, jit=opt_jit)
    elif name == "BFGS":
        return jaxopt.BFGS(loss_fn, verbose=False, jit=opt_jit)
    else:
        raise ValueError(
            f"Optimizer {name} not recognized. Choose from 'Adam', 'BFGS', 'GradientDescent'."
        )


def _check_convergence(losses: jnp.ndarray, convergence_interval: int) -> bool:
    """
    Check for convergence based on loss history.

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
    """
    Single step update logic to be scanned.

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
    """
    Generator that yields training results in batches of size 'unroll_steps'.

    Args:
        optimizer (str): Name of the optimizer to use.
        loss (Callable): The loss function.
        stepsize (float): The learning rate.
        loss_kwargs (dict[str, Any]): Arguments to pass to the loss function.
        options (TrainingOptions | None): Configuration options for training.

    Yields:
        Iterator[BatchResult]: An iterator over batch results.
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
    """
    Main training function.
    Manages the loop, accumulation of history, and convergence checks.

    Args:
        optimizer (str): Name of the optimizer to use.
        loss (Callable): The loss function.
        stepsize (float): The learning rate.
        n_iters (int): Total number of training iterations.
        loss_kwargs (dict[str, Any]): Arguments to pass to the loss function.
        options (TrainingOptions | None): Configuration options for training.

    Returns:
        TrainingResult: The results of the training process, including final parameters and loss history.
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

            # Check Convergence
            if options.convergence_interval is not None:
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
