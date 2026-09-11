import json
import multiprocessing
import sys
import time
from collections import defaultdict

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pennylane.labs.tcdq import (
    CircuitConfig,
    MMDConfig,
    build_expval_func,
    build_mmd_loss_pauli,
    create_local_gates,
    create_random_gates,
    median_heuristic,
)

OOM_EXIT_CODE = 42


@pytest.fixture(scope="session")
def _benchmark_store():
    """Session storage mapping test_function_name -> list of results."""
    store = defaultdict(list)
    yield store

    # Session teardown: Write a separate JSON file for each test function
    for func_name, data in store.items():
        if data:  # Only write files if measurements were logged
            with open(f"{func_name}_results.json", "w") as f:
                json.dump(data, f, indent=2)


@pytest.fixture
def benchmark_results(request, _benchmark_store):
    """
    Injects the results list specific to the current test function.
    Groups all parameterized variations into the same list.
    """
    func_name = request.node.originalname or request.node.name
    return _benchmark_store[func_name]


def _sparse_state(n_qubits, weight):

    if weight == 0:
        return None, None

    rng = np.random.default_rng()

    state = rng.integers(0, 2, size=(weight, n_qubits), dtype=np.int8)
    u = rng.standard_normal(weight) + 1j * rng.standard_normal(weight)
    amps = u / np.linalg.norm(u)

    return state, amps


def _mixed_observables(n_obs, n_qubits, weight):
    assert weight <= n_qubits, "Weight cannot exceed number of qubits"

    permute_columns = np.random.rand(n_obs, n_qubits).argsort(axis=1)
    mask = permute_columns < weight
    values = np.random.choice([1, 2, 3], size=(n_obs, n_qubits))

    return values * mask


def _pauli_Z_observables(n_obs, n_qubits, weight):
    assert weight <= n_qubits, "Weight cannot exceed number of qubits"

    permute_columns = np.random.rand(n_obs, n_qubits).argsort(axis=1)
    mask = permute_columns < weight
    values = np.ones(shape=(n_obs, n_qubits)) * 3

    return values * mask


def mlp_forward(params: jnp.ndarray, bitstring: jnp.ndarray):
    x = bitstring.astype(params.dtype)[..., None]

    W1 = params[0:100].reshape((1, 100))
    b1 = params[100:200]
    W2 = params[200:300].reshape((100, 1))
    b2 = params[300:301]

    hidden = jax.nn.relu(jnp.dot(x, W1) + b1)
    out = jnp.dot(hidden, W2) + b2

    return jnp.sum(out)


def common_cases(func):
    lo_gates, hi_gates = 1000, 100000
    lo_qubits, hi_qubits = 50, 10000
    lo_obs, hi_obs = 100, 10000
    lo_elems, hi_elems = 100, 10000
    lo_samples, hi_samples = 100, 10000

    cases = [
        (hi_gates, lo_qubits, lo_obs, lo_samples),
        (lo_gates, hi_qubits, lo_obs, lo_samples),
        (lo_gates, lo_qubits, hi_obs, lo_samples),
        (lo_gates, lo_qubits, lo_obs, hi_samples),
    ]

    decorators = [
        pytest.mark.parametrize(
            "n_gates, n_qubits, n_obs, n_samples", cases, ids=["gates", "qubits", "obs", "samples"]
        ),
        pytest.mark.parametrize("gate_type", ["local", "random"]),
        pytest.mark.parametrize(
            "n_elems",
            [0, lo_elems, hi_elems],
            ids=["default state", "low sparse state", "high sparse state"],
        ),
    ]

    for dec in decorators:
        func = dec(func)

    return func


def _expval_worker(
    queue, n_gates, n_qubits, n_obs, n_samples, obs_type, gate_type, n_elems, phase_fn
):

    try:
        init_state_elems, init_state_amps = _sparse_state(n_qubits, n_elems)
        observables = (
            _mixed_observables(n_obs, n_qubits, n_qubits)
            if obs_type == "mixed"
            else _pauli_Z_observables(n_obs, n_qubits, n_qubits)
        )
        gates = (
            create_local_gates(n_qubits, max_weight=3)
            if gate_type == "local"
            else create_random_gates(n_qubits, n_gates, min_weight=3, max_weight=3)
        )

        config = CircuitConfig(
            key=jax.random.PRNGKey(0),
            gates=gates,
            n_samples=n_samples,
            n_qubits=n_qubits,
            observables=observables,
            init_state_elems=init_state_elems,
            init_state_amps=init_state_amps,
            phase_fn=phase_fn,
        )

        key = jax.random.PRNGKey(42)
        kwargs = {"phase_fn_params": jax.random.normal(key, shape=301)} if phase_fn else {}
        params = jax.random.normal(key, shape=len(gates))

        expval = jax.jit(build_expval_func(config))

        # JIT Warmup
        _ = jax.block_until_ready(expval(params, **kwargs))

        t0 = time.perf_counter()
        _, variances = jax.block_until_ready(expval(params, **kwargs))
        t1 = time.perf_counter()

        queue.put((t1 - t0, float(np.mean(variances))))

    except MemoryError:
        sys.exit(OOM_EXIT_CODE)


@common_cases
@pytest.mark.parametrize("obs_type", ["mixed", "pauli_z"])
@pytest.mark.parametrize("phase_fn", [None, mlp_forward], ids=["phaseless", "phased"])
def test_expval_execution(
    benchmark_results, n_gates, n_qubits, n_obs, n_samples, obs_type, gate_type, n_elems, phase_fn
):
    queue = multiprocessing.Queue()
    ctx = multiprocessing.get_context("fork")

    p = ctx.Process(
        target=_expval_worker,
        args=(queue, n_gates, n_qubits, n_obs, n_samples, obs_type, gate_type, n_elems, phase_fn),
    )
    p.start()
    p.join()

    if p.exitcode != 0:
        pytest.fail(f"Test crashed unexpectedly with exit code {p.exitcode}")

    duration, mean_variance = queue.get()

    benchmark_results.append(
        {
            "n_qubits": n_qubits,
            "n_gates": n_gates,
            "n_obs": n_obs,
            "n_samples": n_samples,
            "obs_type": obs_type,
            "gate_type": gate_type,
            "n_elems": n_elems,
            "has_phase_fn": phase_fn is not None,
            "duration_seconds": duration,
            "mean_variance": mean_variance,
        }
    )


def _mmd_execution_worker(queue, n_gates, n_qubits, n_obs, n_samples, gate_type, n_elems):

    try:
        init_state_elems, init_state_amps = _sparse_state(n_qubits, n_elems)
        gates = (
            create_local_gates(n_qubits, max_weight=3)
            if gate_type == "local"
            else create_random_gates(n_qubits, n_gates, min_weight=3, max_weight=3)
        )

        circuit_config = CircuitConfig(
            key=jax.random.PRNGKey(0),
            gates=gates,
            n_samples=n_samples,
            n_qubits=n_qubits,
            init_state_elems=init_state_elems,
            init_state_amps=init_state_amps,
        )

        target = np.random.binomial(1, 0.5, size=(100, n_qubits))
        bw = median_heuristic(target)
        mmd_config = MMDConfig(bandwidth=bw, n_ops=n_obs)
        expval_fn = build_expval_func(circuit_config)
        mmd_loss = build_mmd_loss_pauli(expval_fn, n_qubits, mmd_config)

        key = jax.random.PRNGKey(42)
        params = jax.random.normal(key, shape=len(gates))

        # JIT Warmup
        jax.block_until_ready(mmd_loss(params, circuit_config, mmd_config, target))

        t0 = time.perf_counter()
        jax.block_until_ready(mmd_loss(params, circuit_config, mmd_config, target))
        t1 = time.perf_counter()

        queue.put(t1 - t0)

    except MemoryError:
        sys.exit(OOM_EXIT_CODE)


@common_cases
def test_mmd_execution(benchmark_results, n_gates, n_qubits, n_obs, n_samples, gate_type, n_elems):
    queue = multiprocessing.Queue()
    ctx = multiprocessing.get_context("fork")

    p = ctx.Process(
        target=_mmd_execution_worker,
        args=(queue, n_gates, n_qubits, n_obs, n_samples, gate_type, n_elems),
    )
    p.start()
    p.join()

    if p.exitcode != 0:
        pytest.fail(f"Test crashed unexpectedly with exit code {p.exitcode}")

    duration = queue.get()

    benchmark_results.append(
        {
            "n_qubits": n_qubits,
            "n_gates": n_gates,
            "n_obs": n_obs,
            "n_samples": n_samples,
            "gate_type": gate_type,
            "n_elems": n_elems,
            "duration_seconds": duration,
        }
    )


def _mmd_grad_worker(queue, n_gates, n_qubits, n_obs, n_samples, gate_type, n_elems):

    try:
        init_state_elems, init_state_amps = _sparse_state(n_qubits, n_elems)
        gates = (
            create_local_gates(n_qubits, max_weight=3)
            if gate_type == "local"
            else create_random_gates(n_qubits, n_gates, min_weight=3, max_weight=3)
        )

        circuit_config = CircuitConfig(
            key=jax.random.PRNGKey(0),
            gates=gates,
            n_samples=n_samples,
            n_qubits=n_qubits,
            init_state_elems=init_state_elems,
            init_state_amps=init_state_amps,
        )

        target = np.random.binomial(1, 0.5, size=(100, n_qubits))
        bw = median_heuristic(target)
        mmd_config = MMDConfig(bandwidth=bw, n_ops=n_obs)
        expval_fn = build_expval_func(circuit_config)
        mmd_loss = build_mmd_loss_pauli(expval_fn, n_qubits, mmd_config)

        key = jax.random.PRNGKey(42)
        params = jax.random.normal(key, shape=len(gates))
        mmd_grad = jax.grad(mmd_loss)

        # JIT Warmup
        jax.block_until_ready(mmd_grad(params, circuit_config, mmd_config, target))

        t0 = time.perf_counter()
        jax.block_until_ready(mmd_grad(params, circuit_config, mmd_config, target))
        t1 = time.perf_counter()

        queue.put(t1 - t0)

    except MemoryError:
        sys.exit(OOM_EXIT_CODE)


@common_cases
def test_mmd_grad(benchmark_results, n_gates, n_qubits, n_obs, n_samples, gate_type, n_elems):
    queue = multiprocessing.Queue()
    ctx = multiprocessing.get_context("fork")

    p = ctx.Process(
        target=_mmd_grad_worker,
        args=(queue, n_gates, n_qubits, n_obs, n_samples, gate_type, n_elems),
    )
    p.start()
    p.join()

    if p.exitcode != 0:
        pytest.fail(f"Test crashed unexpectedly with exit code {p.exitcode}")

    duration = queue.get()

    benchmark_results.append(
        {
            "n_qubits": n_qubits,
            "n_gates": n_gates,
            "n_obs": n_obs,
            "n_samples": n_samples,
            "gate_type": gate_type,
            "n_elems": n_elems,
            "duration_seconds": duration,
        }
    )
