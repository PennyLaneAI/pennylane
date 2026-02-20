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
"""
Tests for the Phox training module.
"""
import jax
import jax.numpy as jnp
import pytest

from pennylane.labs.phox.expval_functions import CircuitConfig, build_expval_func
from pennylane.labs.phox.training import TrainingOptions, train


@pytest.fixture
def quadratic_problem():
    """
    Defines a simple convex problem: L(theta) = sum((theta - target)^2).
    Target is 0, start is 10. Global minimum is 0.
    """

    def loss_fn(params, target, key=None):  # pylint: disable=unused-argument
        diff = params - target
        return jnp.sum(diff**2)

    params_init = jnp.array([10.0, 10.0])
    target = jnp.array([0.0, 0.0])

    loss_kwargs = {"params": params_init, "target": target}

    return loss_fn, params_init, target, loss_kwargs


def test_optimization_success(quadratic_problem):
    # pylint: disable=redefined-outer-name
    """
    Test 1: Functional Correctness.
    Does the optimizer actually minimize the loss?
    """
    loss_fn, _, target, loss_kwargs = quadratic_problem

    result = train(
        optimizer="Adam",
        loss=loss_fn,
        stepsize=0.1,
        n_iters=3000,
        loss_kwargs=loss_kwargs,
        options=TrainingOptions(unroll_steps=10, random_state=42),
    )

    final_loss = result.losses[-1]
    assert final_loss < 1e-3, f"Optimizer failed to converge, final loss: {final_loss}"
    assert jnp.allclose(result.final_params, target, atol=1e-1)


def test_determinism(quadratic_problem):
    # pylint: disable=redefined-outer-name
    """
    Test 2: Determinism.
    Running the function twice with the same seed must produce bit-exact results.
    """
    loss_fn, _, _, loss_kwargs = quadratic_problem
    opts = TrainingOptions(unroll_steps=5, random_state=123)

    res1 = train("GradientDescent", loss_fn, 0.01, 50, loss_kwargs.copy(), opts)
    res2 = train("GradientDescent", loss_fn, 0.01, 50, loss_kwargs.copy(), opts)

    assert jnp.array_equal(res1.final_params, res2.final_params)
    assert jnp.array_equal(res1.losses, res2.losses)


def test_unroll_consistency(quadratic_problem):
    # pylint: disable=redefined-outer-name
    """
    Test 3: Unrolling Equivalence.
    Running step-by-step (unroll=1) must yield the same math as
    running in batches (unroll=10).
    """
    loss_fn, _, _, loss_kwargs = quadratic_problem
    seed = 999

    opts_slow = TrainingOptions(unroll_steps=1, random_state=seed)
    res_slow = train("GradientDescent", loss_fn, 0.01, 20, loss_kwargs.copy(), opts_slow)

    opts_fast = TrainingOptions(unroll_steps=5, random_state=seed)
    res_fast = train("GradientDescent", loss_fn, 0.01, 20, loss_kwargs.copy(), opts_fast)

    assert jnp.allclose(res_slow.final_params, res_fast.final_params, atol=1e-6)
    assert jnp.allclose(res_slow.losses, res_fast.losses, atol=1e-6)


def test_convergence_early_stopping(quadratic_problem):
    # pylint: disable=redefined-outer-name
    """
    Test 4: Convergence Logic.
    If the loss stabilizes, training should stop before n_iters.
    """
    loss_fn, _, target, _ = quadratic_problem

    params_near = jnp.array([0.01, 0.01])
    loss_kwargs = {"params": params_near, "target": target}

    opts = TrainingOptions(convergence_interval=10, unroll_steps=10)

    result = train("GradientDescent", loss_fn, 0.001, 1000, loss_kwargs, opts)

    # It should have stopped significantly earlier than 1000
    assert len(result.losses) < 1000
    print(f"Converged in {len(result.losses)} steps.")


def test_validation_handling(quadratic_problem):
    # pylint: disable=redefined-outer-name
    """
    Test 5: Validation Data.
    Ensures validation loss is calculated correctly and separately.
    """
    loss_fn, params_init, target, _ = quadratic_problem

    loss_kwargs = {"params": params_init, "target": target}

    val_kwargs = {"target": jnp.array([10.0, 10.0])}

    opts = TrainingOptions(val_kwargs=val_kwargs, unroll_steps=1)
    result = train("Adam", loss_fn, 0.1, 10, loss_kwargs, opts)

    assert len(result.val_losses) == 10
    assert result.losses[-1] < result.losses[0]
    assert not jnp.allclose(result.losses, result.val_losses)


def test_loss_signature_variations():
    """
    Test 6: Signature Flexibility.
    Verifies that the trainer works for loss functions WITH and WITHOUT 'key'.
    """
    params = jnp.array([1.0])

    def loss_with_key(params, key):  # pylint: disable=unused-argument
        return jnp.sum(params**2)

    def loss_no_key(params):
        return jnp.sum(params**2)

    opts = TrainingOptions(unroll_steps=1)

    train("GradientDescent", loss_with_key, 0.1, 5, {"params": params}, opts)
    train("GradientDescent", loss_no_key, 0.1, 5, {"params": params}, opts)


def test_history_logging_manual(quadratic_problem):
    # pylint: disable=redefined-outer-name
    """
    Test 7: Manual Logging via Iterator.
    Since we removed 'monitor_interval', we verify the user can manually
    log params using the lower-level iterator if they wish.
    """
    from pennylane.labs.phox.training import training_iterator

    loss_fn, _, _, loss_kwargs = quadratic_problem
    opts = TrainingOptions(unroll_steps=10)

    iterator = training_iterator("Adam", loss_fn, 0.1, loss_kwargs, opts)

    history = []
    max_steps = 5

    for i, batch_result in enumerate(iterator):
        if i >= max_steps:
            break
        history.append(batch_result.params)

    assert len(history) == 5
    assert history[0].shape == (2,)


def test_iqp_optimization():
    """
    Test 6: Integration Test.
    Test full loop with actual IQP simulation as loss function.
    Objective: Minimize sum of Z expectation values.
    """
    n_qubits = 2
    n_samples = 100

    gates = {0: [[0]], 1: [[1]]}

    params_init = jnp.array([0.1, 0.1])
    ops_ints = [[3, 0], [0, 3]]

    key = jax.random.PRNGKey(42)

    config = CircuitConfig(
        gates=gates,
        observables=ops_ints,
        n_samples=n_samples,
        key=key,
        n_qubits=n_qubits,
    )
    expval_func = build_expval_func(config)

    def loss_fn(params):
        expvals, _ = expval_func(params)
        return jnp.sum(expvals)

    loss_kwargs = {"params": params_init}

    options = TrainingOptions(unroll_steps=10, random_state=42)

    result = train(
        optimizer="Adam",
        loss=loss_fn,
        stepsize=0.1,
        n_iters=50,
        loss_kwargs=loss_kwargs,
        options=options,
    )

    init_loss = result.losses[0]
    final_loss = result.losses[-1]

    # Optimizer should have moved parameters to reduce loss
    assert final_loss < init_loss
