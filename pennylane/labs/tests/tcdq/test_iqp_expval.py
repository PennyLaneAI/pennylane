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
"""Regression tests for the qubit IQP expectation-value estimator."""

import numpy as np
import pytest

import pennylane as qp
from pennylane.labs.tcdq.expval_functions import (
    CircuitConfig,
    _memory_budget_bytes,
    _parse_generator_dict,
    _phase_block_size,
    build_expval_func,
)

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")


def _prepare_obs_batch(obs_strings):
    """Normalize observable labels into integer-coded batches."""
    base_map = {"I": 0, "X": 1, "Y": 2, "Z": 3}

    if isinstance(obs_strings[0], str) and len(obs_strings[0]) == 1 and obs_strings[0] in base_map:
        mapped = [[base_map[s] for s in obs_strings]]
        return mapped, len(obs_strings)

    mapped = [[base_map[s] for s in row] for row in obs_strings]
    return mapped, len(obs_strings[0])


def _prepare_pennylane_state(n_qubits, init_state_spec):
    """Build a dense statevector for the PennyLane reference circuit."""
    state = np.zeros(2**n_qubits, dtype=complex)

    if init_state_spec is None:
        state[0] = 1.0
        return state

    is_single_bitstring = isinstance(init_state_spec, list) and (
        not init_state_spec or not isinstance(init_state_spec[0], (list, tuple))
    )

    if is_single_bitstring:
        idx = int("".join(str(b) for b in init_state_spec), 2)
        state[idx] = 1.0
        return state

    X, P = init_state_spec
    X = np.array(X)
    P = np.array(P)
    for x, p in zip(X, P):
        idx = int("".join(str(b) for b in x), 2)
        state[idx] = p

    return state


def _prepare_jax_state(init_state_spec):
    """Convert the optional initial-state specification into JAX arrays."""
    if init_state_spec is None:
        return None, None

    is_single_bitstring = isinstance(init_state_spec, list) and (
        not init_state_spec or not isinstance(init_state_spec[0], (list, tuple))
    )

    if is_single_bitstring:
        return jnp.array([init_state_spec]), jnp.array([1.0])

    return jnp.array(init_state_spec[0]), jnp.array(init_state_spec[1])


def _run_pennylane_ground_truth(generators_pl, params_pl, obs_batch_ints, init_state):
    """Evaluate the PennyLane reference circuit for each observable in a batch."""
    exact_vals = []
    for obs in obs_batch_ints:
        circuit = iqp_circuit_pl(generators_pl, params_pl, obs, init_state)
        exact_vals.append(circuit())
    return np.array(exact_vals).flatten()


def iqp_circuit_pl(generators, params, obs_ints, init_state):
    """Build a PennyLane reference circuit for one integer-encoded observable."""
    n_qubits = len(obs_ints)

    expval_ops = []
    for i, op in enumerate(obs_ints):
        if op == 1:
            expval_ops.append(qp.X(i))
        elif op == 2:
            expval_ops.append(qp.Y(i))
        elif op == 3:
            expval_ops.append(qp.Z(i))
        elif op == 0:
            expval_ops.append(qp.Identity(i))

    expval_op = qp.prod(*expval_ops)

    dev = qp.device("default.qubit", wires=n_qubits)

    @qp.qnode(dev)
    def circuit():
        qp.StatePrep(np.array(init_state), wires=range(n_qubits))

        for i in range(n_qubits):
            qp.Hadamard(i)

        for param, gen in zip(params, generators):
            qp.MultiRZ(2 * -param, wires=gen)

        for i in range(n_qubits):
            qp.Hadamard(i)

        return qp.expval(expval_op)

    return circuit


class TestIQPExpval:
    """Tests for IQP expectation value calculation."""

    @pytest.mark.parametrize("n_samples", [1000, 10000])
    @pytest.mark.parametrize(
        "obs_strings, generators_pl, params, init_state_spec",
        [
            (["X", "Z", "Y"], [[0], [1], [0, 1, 2]], [0.37, 0.95, 0.73], None),
            (["X"], [[0]], [0.1], None),
            (["Y", "Y"], [[0], [1], [0, 1]], [0.2, 0.3, 0.4], None),
            (["Z", "Z", "Z"], [[0, 1], [1, 2]], [0.1, 0.2], None),
            (
                ["X", "Y", "Z", "I"],
                [[0, 1], [2, 3], [0, 2, 3]],
                [0.1, 0.2, 0.3],
                None,
            ),
            (["I", "I", "I", "I"], [[0, 1], [2, 3]], [0.5, 0.6], None),
            ([["Z", "Z"], ["X", "X"]], [[0], [1]], [0.1, 0.2], None),
            (["Z", "Z"], [[0, 1]], [0.1], [1, 0]),
            (["X", "Z", "Y"], [[0], [1], [0, 1, 2]], [0.2, 0.8, 0.4], [1, 0, 1]),
            (["Z", "Z", "Z"], [[0, 1], [1, 2]], [0.1, 0.2], [1, 1, 1]),
            (["X", "X", "X", "X"], [[0, 1], [2, 3], [0, 3]], [0.1, 0.2, 0.3], [1, 0, 0, 1]),
            (
                ["Z", "Z"],
                [[0, 1]],
                [0.1],
                ([[0, 0], [1, 1]], [1 / np.sqrt(2), 1 / np.sqrt(2)]),
            ),
            (["Y"], [], [], ([[0], [1]], [1 / np.sqrt(2), 1j / np.sqrt(2)])),
        ],
    )
    def test_build_expval_func_core_vs_pennylane(
        self, n_samples, obs_strings, generators_pl, params, init_state_spec
    ):
        """Test core expval function against PennyLane ground truth."""
        # pylint: disable=too-many-arguments
        obs_batch, n_qubits = _prepare_obs_batch(obs_strings)
        pl_state = _prepare_pennylane_state(n_qubits, init_state_spec)
        jax_state_elems, jax_state_amps = _prepare_jax_state(init_state_spec)

        exact_vals = _run_pennylane_ground_truth(generators_pl, params, obs_batch, pl_state)

        gates = {i: [wires] for i, wires in enumerate(generators_pl)}

        params_jax = jnp.array(params)
        key = jax.random.PRNGKey(42)
        atol = 3.5 / np.sqrt(n_samples)

        config = CircuitConfig(
            gates=gates,
            observables=obs_batch,
            n_samples=n_samples,
            key=key,
            n_qubits=n_qubits,
            init_state_elems=jax_state_elems,
            init_state_amps=jax_state_amps,
        )
        expval_func = build_expval_func(config)
        approx_val, _ = expval_func(params_jax)

        assert np.allclose(exact_vals, approx_val, atol=atol)

    @pytest.mark.parametrize(
        "n_qubits, gates, params, obs_strings, init_state_spec",
        [
            (3, {0: [[0], [1]], 1: [[0, 1], [1, 2]]}, [0.1, 0.2], ["X", "Z", "Y"], None),
            (2, {}, [], ["Z", "Z"], None),
            (3, {0: [[0, 1]], 1: [[1, 2]]}, [0.1, 0.2], ["X", "I", "Z"], None),
            (2, {0: [[0, 1]]}, [0.5], ["I", "I"], None),
            (2, {0: [[0, 1]]}, [0.5], [["Z", "Z"], ["X", "X"]], None),
            (2, {0: [[0, 1]]}, [0.5], ["Z", "Z"], [1, 0]),
            (3, {0: [[0, 1]], 1: [[1, 2]]}, [0.1, 0.2], ["X", "Z", "Y"], [1, 0, 1]),
            (3, {0: [[0], [1], [2]]}, [0.1, 0.2, 0.3], ["Z", "Z", "Z"], [1, 1, 1]),
            (
                2,
                {0: [[0, 1]]},
                [0.1],
                ["Z", "Z"],
                ([[0, 0], [1, 1]], [1 / np.sqrt(2), 1 / np.sqrt(2)]),
            ),
        ],
    )
    def test_build_expval_func_vs_pennylane(
        self, n_qubits, gates, params, obs_strings, init_state_spec
    ):
        """Test built expval function versus full PennyLane simulation."""
        # pylint: disable=too-many-arguments
        generators, param_map = _parse_generator_dict(gates, n_qubits)
        generators_pl = [[int(q) for q in row if q != n_qubits] for row in generators]
        params_pl = np.array(params)[param_map]

        obs_batch, _ = _prepare_obs_batch(obs_strings)
        pl_state = _prepare_pennylane_state(n_qubits, init_state_spec)
        jax_state_elems, jax_state_amps = _prepare_jax_state(init_state_spec)

        exact_vals = _run_pennylane_ground_truth(generators_pl, params_pl, obs_batch, pl_state)

        key = jax.random.PRNGKey(42)
        n_samples = 10000
        atol = 3.5 / np.sqrt(n_samples)

        config = CircuitConfig(
            gates=gates,
            observables=obs_batch,
            n_samples=n_samples,
            key=key,
            n_qubits=n_qubits,
            init_state_elems=jax_state_elems,
            init_state_amps=jax_state_amps,
        )
        expval_func = build_expval_func(config)
        approx_val, _ = expval_func(np.array(params))

        assert np.allclose(exact_vals, approx_val, atol=atol)

    def test_iqp_parameter_broadcasting(self):
        """Test that single parameter is broadcast to multiple generators."""
        n_qubits = 3
        gates = {0: [[0, 1], [1, 2]]}
        params = [0.8]

        obs_strings = ["X", "X", "X"]
        obs_batch, _ = _prepare_obs_batch(obs_strings)

        generators_pl = [[0, 1], [1, 2]]
        params_pl = [0.8, 0.8]

        pl_state = _prepare_pennylane_state(n_qubits, None)
        exact_vals = _run_pennylane_ground_truth(generators_pl, params_pl, obs_batch, pl_state)

        key = jax.random.PRNGKey(99)
        n_samples = 20000
        atol = 0.05

        config = CircuitConfig(
            gates=gates,
            observables=obs_batch,
            n_samples=n_samples,
            key=key,
            n_qubits=n_qubits,
        )
        expval_func = build_expval_func(config)
        approx_val, _ = expval_func(np.array(params))

        assert np.allclose(exact_vals, approx_val, atol=atol)

    def test_build_expval_func_with_phase_layer(self):
        """Test expectation values when a phase layer is supplied."""

        def compute_phase(params, z):
            hamming = jnp.mean(jnp.abs(z))
            hamming_powers = jnp.array([hamming**t for t in range(4)])
            return jnp.sum(params * hamming_powers)

        bitstrings = jnp.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        phase_params = jnp.array([0.11, 0.7, 3.0, 1.0])

        phases = jax.vmap(compute_phase, in_axes=(None, 0))(phase_params, bitstrings)
        diagonal = jnp.exp(1j * phases).flatten()

        generators_pl = [[0], [1], [0, 1]]
        params = [0.37, 0.95, 0.73]
        pl_state = [1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)]

        jax_state_elems = jnp.array([[0, 0], [1, 1]])
        jax_state_amps = jnp.array([1 / jnp.sqrt(2), 1 / jnp.sqrt(2)])

        n_qubits = 2
        dev = qp.device("default.qubit", wires=n_qubits)

        expval_ops = [qp.Z(0), qp.Y(1)]
        expval_op = qp.prod(*expval_ops)

        @qp.qnode(dev)
        def circuit():
            qp.StatePrep(np.array(pl_state), wires=range(n_qubits))

            for i in range(n_qubits):
                qp.Hadamard(i)

            for param, gen in zip(params, generators_pl):
                qp.MultiRZ(2 * -param, wires=gen)

            qp.DiagonalQubitUnitary(diagonal, wires=[0, 1])

            for i in range(n_qubits):
                qp.Hadamard(i)

            return qp.expval(expval_op)

        exact_val = circuit()

        gates = {0: [[0]], 1: [[1]], 2: [[0, 1]]}
        obs_batch = [[3, 2]]  # Using integer mapped observables

        config = CircuitConfig(
            n_qubits=n_qubits,
            gates=gates,
            observables=obs_batch,
            init_state_elems=jax_state_elems,
            init_state_amps=jax_state_amps,
            phase_fn=compute_phase,
            n_samples=50000,
            key=jax.random.PRNGKey(42),
        )

        f = build_expval_func(config)
        approx_val, _ = f(jnp.array(params), phase_params)

        atol = 3.5 / np.sqrt(50000)
        assert np.allclose(exact_val, approx_val, atol=atol)


@pytest.mark.parametrize(
    "circuit_def,n_qubits,expected_generators,expected_param_map",
    [
        ({0: [[0, 1]]}, 3, [[0, 1]], [0]),
        ({0: [[0]], 1: [[1, 2], [0, 2]]}, 3, [[0, 3], [1, 2], [0, 2]], [0, 1, 1]),
        ({}, 2, np.empty((0, 1)), []),
        ({10: [[0]], 2: [[1]]}, 2, [[1], [0]], [2, 10]),
    ],
)
def test_parse_generator_dict(circuit_def, n_qubits, expected_generators, expected_param_map):
    """Test generator parsing produces expected matrices and parameter maps."""
    generators, param_map = _parse_generator_dict(circuit_def, n_qubits)

    assert isinstance(generators, jnp.ndarray)
    assert isinstance(param_map, jnp.ndarray)

    expected_generators = np.array(expected_generators)
    expected_param_map = np.array(expected_param_map)

    assert generators.shape == expected_generators.shape
    assert param_map.shape == expected_param_map.shape

    assert np.allclose(generators, expected_generators)
    assert np.allclose(param_map, expected_param_map)


def test_parse_generator_dict_index_error():
    """Test generator parsing raises IndexError for invalid qubit indices."""
    circuit_def = {0: [[5]]}
    n_qubits = 2

    with pytest.raises(IndexError):
        _parse_generator_dict(circuit_def, n_qubits)


class TestMemoryBudget:
    """Tests for sizing the phase-difference blocks from ``max_memory``."""

    @pytest.mark.parametrize(
        "gb, expected", [(1, 1 << 30), (1.0, 1 << 30), (0.5, 1 << 29), (2.5, 5 << 29)]
    )
    def test_memory_budget_bytes(self, gb, expected):
        """Gigabytes are binary (1024**3 bytes) and accept ints and floats."""
        assert _memory_budget_bytes(gb) == expected
        assert isinstance(_memory_budget_bytes(gb), int)

    def test_phase_block_size_scaling(self):
        """The block size scales linearly with the budget and inversely with the shapes."""
        base = _phase_block_size(1 << 30, 2000, 100)
        assert _phase_block_size(1 << 31, 2000, 100) == 2 * base
        assert _phase_block_size(1 << 30, 4000, 200) == base // 2
        assert base * (16 * 2000 + 4 * 100) <= 1 << 30

    def test_phase_block_size_floor(self):
        """A budget too small for a single gate still yields one gate per block."""
        assert _phase_block_size(1, 2000, 100) == 1

    @pytest.mark.parametrize("bad", [0, -1, 0.0, -0.5, True, None, "1", float("nan"), float("inf")])
    def test_invalid_max_memory(self, bad):
        """Non-positive, non-finite or non-numeric budgets raise a clear error at build time."""
        config = CircuitConfig(
            gates={0: [[0]]},
            n_samples=10,
            key=jax.random.PRNGKey(0),
            n_qubits=1,
            observables=[[3]],
            max_memory=bad,
        )
        with pytest.raises(ValueError, match="max_memory must be a positive, finite number"):
            build_expval_func(config)

    @pytest.mark.parametrize("n_samples", [7, 64])
    def test_results_independent_of_budget(self, n_samples):
        """Blocked (multi-block, padded) and single-block evaluations agree, in value and gradient."""
        n_qubits = 5
        gates = {
            0: [[0, 1], [1, 2]],
            1: [[2, 3], [3, 4], [0, 4]],
            2: [[0], [1], [2], [3], [4]],
            3: [[0, 1, 2], [2, 3, 4]],
        }  # 12 gates, not a power of two, so small budgets exercise the padding path
        observables = [[3, 3, 0, 0, 0], [1, 1, 0, 0, 0], [2, 3, 1, 0, 0], [0, 0, 0, 3, 3]]
        params = jnp.array([0.3, -1.1, 0.7, 0.2])

        def make(max_memory):
            config = CircuitConfig(
                gates=gates,
                n_samples=n_samples,
                key=jax.random.PRNGKey(3),
                n_qubits=n_qubits,
                observables=observables,
                max_memory=max_memory,
            )
            return build_expval_func(config)

        f_single = make(1.0)
        assert _phase_block_size(_memory_budget_bytes(1.0), n_samples, len(observables)) >= 12

        # budget for exactly 5 gates -> 3 blocks, 3 padded rows
        five_gates_gb = 5 * (16 * n_samples + 4 * len(observables)) / 1024**3
        assert (
            _phase_block_size(_memory_budget_bytes(five_gates_gb), n_samples, len(observables)) == 5
        )
        f_blocked = make(five_gates_gb)

        tiny_gb = 1 / 1024**3  # one byte: one gate per block
        assert _phase_block_size(_memory_budget_bytes(tiny_gb), n_samples, len(observables)) == 1
        f_one = make(tiny_gb)

        ref_ev, ref_var = f_single(params)
        ref_grad = jax.grad(lambda p: jnp.sum(f_single(p)[0]))(params)

        for f in (f_blocked, f_one):
            ev, var = f(params)
            grad = jax.grad(lambda p, f=f: jnp.sum(f(p)[0]))(params)
            assert np.allclose(ev, ref_ev, atol=1e-5)
            assert np.allclose(var, ref_var, atol=1e-5)
            assert np.allclose(grad, ref_grad, atol=1e-5)

    def test_runtime_override_respects_budget(self):
        """Overriding ``n_samples`` at call time still runs within the configured budget."""
        gates = {0: [[0, 1]], 1: [[1, 2]], 2: [[0]], 3: [[2]]}
        config = CircuitConfig(
            gates=gates,
            n_samples=8,
            key=jax.random.PRNGKey(0),
            n_qubits=3,
            observables=[[3, 3, 0], [0, 3, 3]],
            max_memory=2 * (16 * 64 + 4 * 2) / 1024**3,  # two gates per block at 64 samples
        )
        f = build_expval_func(config)
        ev_default, _ = f(jnp.array([0.1, 0.2, 0.3, 0.4]))
        ev_override, _ = f(jnp.array([0.1, 0.2, 0.3, 0.4]), n_samples=64, key=jax.random.PRNGKey(1))
        assert ev_default.shape == ev_override.shape == (2,)
        assert np.all(np.abs(ev_override) <= 1.0 + 1e-6)
