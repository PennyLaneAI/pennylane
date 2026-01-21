import pytest
import jax
import jax.numpy as jnp
from pennylane.labs.phox import trainer
from pennylane.labs.phox.trainer import Trainer  # Assuming your class is in trainer.py


@pytest.fixture
def quadratic_problem():
    """
    Defines a simple convex problem: L(theta) = sum((theta - target)^2).
    Target is 0, start is 10. Global minimum is 0.
    """

    def loss_fn(params, x, key=None):
        # We include 'key' to test the signature detection logic
        return jnp.sum((params - x) ** 2)

    params_init = jnp.array([10.0, 10.0])
    target = jnp.array([0.0, 0.0])
    return loss_fn, params_init, target


def test_optimization_correctness(quadratic_problem):
    """
    Test 1: Does the optimizer actually optimize?
    Verifies that loss decreases and parameters get closer to the target.
    """
    loss_fn, params_init, target = quadratic_problem

    trainer = Trainer(optimizer="Adam", loss=loss_fn, stepsize=0.1)

    # Run training
    loss_kwargs = {"params": params_init, "x": target}
    trainer.train(n_iters=500, loss_kwargs=loss_kwargs, random_state=42)

    # Check if final loss is near zero
    final_loss = trainer.losses[-1]
    assert final_loss < 1e-3, f"Optimizer failed to converge, final loss: {final_loss}"

    # Check if params are near target (0.0)
    assert jnp.allclose(trainer.final_params, target, atol=1e-1)


def test_determinism_and_seed(quadratic_problem):
    """
    Test 2: Determinism (Crucial for Refactoring).
    Running the trainer twice with the same seed must produce bit-exact results.
    """
    loss_fn, params_init, target = quadratic_problem
    loss_kwargs = {"params": params_init, "x": target}

    # Run 1
    t1 = Trainer(optimizer="GradientDescent", loss=loss_fn, stepsize=0.01)
    t1.train(n_iters=50, loss_kwargs=loss_kwargs.copy(), random_state=123)

    # Run 2
    t2 = Trainer(optimizer="GradientDescent", loss=loss_fn, stepsize=0.01)
    t2.train(n_iters=50, loss_kwargs=loss_kwargs.copy(), random_state=123)

    assert jnp.array_equal(t1.final_params, t2.final_params)
    assert jnp.array_equal(t1.losses, t2.losses)


def test_turbo_equivalence(quadratic_problem):
    """
    Test 3: Logic Equivalence.
    The 'turbo' mode (jax.lax.scan) and the standard python loop mode
    must produce mathematically identical results given the same seed.
    """
    loss_fn, params_init, target = quadratic_problem
    loss_kwargs = {"params": params_init, "x": target}
    seed = 555

    # Standard Loop Mode (turbo=None)
    t_loop = Trainer(optimizer="Adam", loss=loss_fn, stepsize=0.01)
    t_loop.train(n_iters=100, loss_kwargs=loss_kwargs.copy(), turbo=None, random_state=seed)

    # Turbo Mode (turbo=10)
    t_turbo = Trainer(optimizer="Adam", loss=loss_fn, stepsize=0.01)
    t_turbo.train(n_iters=100, loss_kwargs=loss_kwargs.copy(), turbo=10, random_state=seed)

    # Tolerances might need to be slightly loose due to float associativity
    # differences in JIT compilation, but usually they are exact in JAX.
    assert jnp.allclose(t_loop.final_params, t_turbo.final_params, atol=1e-6)
    assert jnp.allclose(t_loop.losses, t_turbo.losses, atol=1e-6)


def test_convergence_early_stopping(quadratic_problem):
    """
    Test 4: Convergence Logic.
    If the loss stops improving, the trainer should stop before n_iters.
    """
    loss_fn, params_init, target = quadratic_problem

    # Start very close to solution so it converges quickly
    params_near = jnp.array([0.01, 0.01])
    loss_kwargs = {"params": params_near, "x": target}

    trainer = Trainer(optimizer="GradientDescent", loss=loss_fn, stepsize=0.001)

    # Set a huge n_iters but a strict convergence interval
    n_iters = 1000
    trainer.train(
        n_iters=n_iters, loss_kwargs=loss_kwargs, convergence_interval=10, random_state=42
    )

    # It should have stopped early
    assert len(trainer.losses) < n_iters
    print(f"Converged in {len(trainer.losses)} steps.")


def test_parameter_history_shape(quadratic_problem):
    """
    Test 5: Monitoring.
    Verifies that params_hist is populated with the correct shape and interval.
    """
    loss_fn, params_init, target = quadratic_problem
    loss_kwargs = {"params": params_init, "x": target}

    trainer = Trainer(optimizer="Adam", loss=loss_fn, stepsize=0.1)

    n_iters = 100
    monitor_interval = 10

    trainer.train(
        n_iters=n_iters,
        loss_kwargs=loss_kwargs,
        monitor_interval=monitor_interval,
        turbo=None,  # Test loop mode specifically for array concatenation logic
    )

    history = jnp.array(trainer.params_hist)
    expected_snapshots = n_iters // monitor_interval

    assert (
        history.shape[0] == expected_snapshots
    ), f"Expected {expected_snapshots} snapshots, got {history.shape[0]}"

    assert (
        history.shape[1] == params_init.shape[0]
    ), f"Expected param dimension {params_init.shape[0]}, got {history.shape[1]}"


def test_validation_loss_logic():
    """
    Test 6: Validation Args.
    Ensures validation loss is calculated and separate from training loss.
    """

    def simple_loss(params, x):
        return (params - x) ** 2

    trainer = Trainer("GradientDescent", simple_loss, stepsize=0.1)

    # Train on target 0, Validate on target 10
    loss_kwargs = {"params": jnp.array(5.0), "x": 0.0}
    val_kwargs = {"x": 10.0}

    trainer.train(n_iters=10, loss_kwargs=loss_kwargs, val_kwargs=val_kwargs, turbo=None)

    # Training loss should go down (getting closer to 0)
    # Validation loss should go UP (as we move towards 0, we move away from 10)
    assert trainer.losses[-1] < trainer.losses[0]
    assert trainer.val_losses[-1] > trainer.val_losses[0]

def test_bug_1_resetting_index(quadratic_problem):
    """
    BUG 1: The 'Resetting Index' Bug.
    
    Scenario: 
        We use turbo=1 (standard loop) but ask for a monitor_interval=10.
        
    The Setup:
        - Total iterations: 20
        - Monitor interval: 10
        - Expected snapshots: 2 (Steps 0 and 10) or 3 (0, 10, 20).
    
    The Bug:
        Inside the loop, the scanner index 'x' resets to 0 every single time.
        Therefore `x % 10 == 0` evaluates to True for EVERY step.
        The Trainer logs 20 times instead of 2.
    """
    loss_fn, params_init, target = quadratic_problem
    
    # Note: The fixture's loss_fn expects argument 'x', not 'target'
    loss_kwargs = {"params": params_init, "x": target}
    
    trainer = Trainer(optimizer="GradientDescent", loss=loss_fn, stepsize=0.1)
    
    # Trigger the bug by using turbo=1
    trainer.train(
        n_iters=20, 
        loss_kwargs=loss_kwargs, 
        monitor_interval=10, 
        turbo=1 
    )
    
    actual_snapshots = len(trainer.params_hist)
    expected_snapshots = 2 # approximately (step 0 and step 10)
    
    print(f"\n[Bug 1 Detection] Steps: 20, Interval: 10, Turbo: 1")
    print(f"Expected History Length: ~{expected_snapshots}")
    print(f"Actual History Length:   {actual_snapshots}")

    # This assertion will FAIL on the original code
    assert actual_snapshots <= 3, \
        f"BUG CONFIRMED: Logged {actual_snapshots} times! Index resets to 0 every step."


def test_bug_2_garbage_data(quadratic_problem):
    """
    BUG 2: The 'Garbage Data' Bug.
    
    Scenario:
        We use a high turbo value (batch mode).
        
    The Setup:
        - Total iterations: 10
        - Turbo: 10 (Runs all steps in one compiled scan)
        - Monitor interval: 10 (Should log only step 0)
        
    The Bug:
        jax.lax.scan outputs an array of the same length as the input (10).
        For steps where (x % interval != 0), the original code puts 0.
        The user receives a history of length 10 where 9 rows are garbage zeros.
    """
    loss_fn, params_init, target = quadratic_problem
    loss_kwargs = {"params": params_init, "x": target}
    
    trainer = Trainer(optimizer="GradientDescent", loss=loss_fn, stepsize=0.1)
    
    # Trigger the bug by using high turbo
    trainer.train(
        n_iters=10, 
        loss_kwargs=loss_kwargs, 
        monitor_interval=10, 
        turbo=10
    )
    
    hist = trainer.params_hist
    
    # 1. Check History Length
    # We expect 1 snapshot (the 0-th step). The bug returns 10.
    print(f"\n[Bug 2 Detection] Turbo=10, Interval=10")
    print(f"History Shape: {hist.shape}")
    
    assert len(hist) == 1, \
        f"BUG CONFIRMED: History has length {len(hist)}, expected 1."

    # 2. Check for Zero-Filling
    # In the fixture, params_init is [10.0, 10.0]. 
    # If we see a row of [0.0, 0.0], it is definitely an artifact/bug.
    # Step 1 (index 1) should be skipped, so in the bug it becomes 0.
    bug_artifact = hist[1]
    is_zero_filled = jnp.all(bug_artifact == 0.0)
    
    assert not is_zero_filled, \
        "BUG CONFIRMED: History contains zero-filled rows (artifacts from lax.scan)."
