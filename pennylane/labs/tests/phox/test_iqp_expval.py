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
Tests for the IQP expectation value calculator.
"""

import numpy as np
import pytest

import pennylane as qp

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

try:
    from pennylane.labs.phox.expval_functions import (
        CircuitConfig,
        _compute_control_variate_mean,
        _cv_mean_z_only,
        _parse_generator_dict,
        _prep_observables,
        build_expval_func,
    )
except ImportError:
    pytest.skip("pennylane.labs.phox not found", allow_module_level=True)


def _prepare_obs_batch(obs_strings):
    """Refactor helper: Normalize obs_strings into a batch integer array and count qubits."""
    base_map = {"I": 0, "X": 1, "Y": 2, "Z": 3}

    if isinstance(obs_strings[0], str) and len(obs_strings[0]) == 1 and obs_strings[0] in base_map:
        mapped = [[base_map[s] for s in obs_strings]]
        return mapped, len(obs_strings)

    mapped = [[base_map[s] for s in row] for row in obs_strings]
    return mapped, len(obs_strings[0])


def _prepare_pennylane_state(n_qubits, init_state_spec):
    """Check init_state_spec and build dense complex state vector."""
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
    """Convert spec into JAX state elements (X) and amplitudes (P)."""
    if init_state_spec is None:
        return None, None

    is_single_bitstring = isinstance(init_state_spec, list) and (
        not init_state_spec or not isinstance(init_state_spec[0], (list, tuple))
    )

    if is_single_bitstring:
        return jnp.array([init_state_spec]), jnp.array([1.0])

    return jnp.array(init_state_spec[0]), jnp.array(init_state_spec[1])


def _run_pennylane_ground_truth(generators_pl, params_pl, obs_batch_ints, init_state):
    """Run the PennyLane default.qubit simulation for the batch of observables."""
    exact_vals = []
    for obs in obs_batch_ints:
        circuit = iqp_circuit_pl(generators_pl, params_pl, obs, init_state)
        exact_vals.append(circuit())
    return np.array(exact_vals).flatten()


def iqp_circuit_pl(generators, params, obs_ints, init_state):
    """Creates a PennyLane QNode for the IQP circuit using integer observables."""
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
        generators_binary, param_map = _parse_generator_dict(gates, n_qubits)
        generators_pl = [list(np.where(row)[0]) for row in generators_binary]
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
        ({0: [[0, 1]]}, 3, [[1, 1, 0]], [0]),
        ({0: [[0]], 1: [[1, 2], [0, 2]]}, 3, [[1, 0, 0], [0, 1, 1], [1, 0, 1]], [0, 1, 1]),
        ({}, 2, np.zeros((0, 2), dtype=int), []),
        ({10: [[0]], 2: [[1]]}, 2, [[0, 1], [1, 0]], [2, 10]),
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


def _input_state_pauli_expval(n_qubits, init_state_spec, obs_ints):
    """Compute <psi_in | P_a | psi_in> exactly using PennyLane (ground truth)."""
    state = _prepare_pennylane_state(n_qubits, init_state_spec)

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
        qp.StatePrep(np.array(state), wires=range(n_qubits))
        return qp.expval(expval_op)

    return float(np.real(circuit()))


class TestControlVariate:
    """Tests for the closed-form control-variate variance-reduction option."""

    @pytest.mark.parametrize(
        "obs_strings, init_state_spec",
        [
            (["Z", "Z"], None),
            (["Z", "Z", "Z"], None),
            (
                ["Z", "Z"],
                ([[0, 0], [1, 1]], [1 / np.sqrt(2), 1 / np.sqrt(2)]),
            ),
            (
                [["Z", "Z", "Z"], ["Z", "I", "Z"], ["I", "Z", "I"]],
                ([[0, 0, 1], [0, 1, 1], [1, 1, 0]], [0.6, 0.6, np.sqrt(1 - 2 * 0.36)]),
            ),
            (["Z", "Z", "Z"], [1, 0, 1]),
        ],
    )
    def test_control_variate_mean_matches_pennylane(self, obs_strings, init_state_spec):
        """The closed-form CV mean must equal <psi_in | P_a | psi_in> exactly."""
        obs_batch, n_qubits = _prepare_obs_batch(obs_strings)

        bitflips, _mask_XY, _y_phase = _prep_observables(obs_batch)
        state_elems, state_amps = _prepare_jax_state(init_state_spec)

        cv_mean = _compute_control_variate_mean(bitflips, state_elems, state_amps)

        expected = np.array(
            [_input_state_pauli_expval(n_qubits, init_state_spec, obs) for obs in obs_batch]
        )

        assert np.allclose(cv_mean, expected, atol=1e-10)

    @pytest.mark.parametrize(
        "obs_strings, generators_pl, params, init_state_spec",
        [
            (["Z", "Z", "Z"], [[0, 1], [1, 2]], [0.1, 0.2], None),
            (["Z", "Z", "Z"], [[0], [1], [0, 1, 2]], [0.2, 0.8, 0.4], [1, 0, 1]),
            (
                ["Z", "Z"],
                [[0, 1]],
                [0.1],
                ([[0, 0], [1, 1]], [1 / np.sqrt(2), 1 / np.sqrt(2)]),
            ),
            (
                [["Z", "Z"], ["I", "Z"], ["Z", "I"]],
                [[0], [1], [0, 1]],
                [0.3, 0.4, 0.5],
                ([[0, 0], [0, 1], [1, 1]], [0.5, np.sqrt(0.5), 0.5]),
            ),
        ],
    )
    def test_control_variate_unbiased_vs_pennylane(
        self, obs_strings, generators_pl, params, init_state_spec
    ):
        """The CV estimator must remain unbiased: it agrees with PennyLane up to MC error."""
        obs_batch, n_qubits = _prepare_obs_batch(obs_strings)
        pl_state = _prepare_pennylane_state(n_qubits, init_state_spec)
        jax_state_elems, jax_state_amps = _prepare_jax_state(init_state_spec)

        exact_vals = _run_pennylane_ground_truth(generators_pl, params, obs_batch, pl_state)

        gates = {i: [wires] for i, wires in enumerate(generators_pl)}

        n_samples = 10000
        atol = 4.0 / np.sqrt(n_samples)
        key = jax.random.PRNGKey(7)

        config_cv = CircuitConfig(
            gates=gates,
            observables=obs_batch,
            n_samples=n_samples,
            key=key,
            n_qubits=n_qubits,
            init_state_elems=jax_state_elems,
            init_state_amps=jax_state_amps,
            control_variate=True,
        )
        approx_val, _ = build_expval_func(config_cv)(jnp.array(params))

        assert np.allclose(exact_vals, approx_val, atol=atol)

    def test_control_variate_zero_params_is_exact(self):
        """At gates_params=0 the CV estimator returns the input-state moment with zero MC variance."""
        n_qubits = 3
        init_state_elems = jnp.array([[0, 0, 1], [1, 1, 0], [0, 1, 1]])
        init_state_amps = jnp.array([0.6, np.sqrt(0.5), np.sqrt(1 - 0.36 - 0.5)])
        obs_batch = [[3, 3, 3], [3, 3, 0], [3, 0, 0]]
        n_samples = 64

        config_cv = CircuitConfig(
            gates={0: [[0, 1]], 1: [[1, 2]]},
            observables=obs_batch,
            n_samples=n_samples,
            key=jax.random.PRNGKey(0),
            n_qubits=n_qubits,
            init_state_elems=init_state_elems,
            init_state_amps=init_state_amps,
            control_variate=True,
        )
        zero_params = jnp.zeros(2)
        approx_val, std_err = build_expval_func(config_cv)(zero_params)

        init_state_spec = (np.array(init_state_elems), np.array(init_state_amps))
        expected = np.array(
            [_input_state_pauli_expval(n_qubits, init_state_spec, obs) for obs in obs_batch]
        )

        assert np.allclose(approx_val, expected, atol=1e-10)
        assert np.allclose(std_err, 0.0, atol=1e-10)

    def test_control_variate_reduces_variance_for_small_angles(self):
        """For sparse data states and small theta, the CV estimator must have much smaller std_err."""
        n_qubits = 4
        rng = np.random.default_rng(0)
        # asymmetric sparse data state to avoid symmetry-induced exact cancellations
        init_state_elems = jnp.array(rng.binomial(1, 0.5, size=(6, n_qubits)))
        amps = rng.uniform(0.5, 1.5, size=6)
        init_state_amps = jnp.array(amps / np.linalg.norm(amps))
        obs_batch = [[3, 3, 0, 0], [3, 0, 0, 3], [0, 3, 3, 0], [3, 3, 3, 3]]
        gates = {0: [[0, 1]], 1: [[1, 2]], 2: [[2, 3]], 3: [[0, 3]]}
        n_samples = 4000
        key = jax.random.PRNGKey(123)

        small_theta = 0.01
        small_params = jnp.full(4, small_theta)

        kwargs = dict(
            gates=gates,
            observables=obs_batch,
            n_samples=n_samples,
            key=key,
            n_qubits=n_qubits,
            init_state_elems=init_state_elems,
            init_state_amps=init_state_amps,
        )
        _, std_err_plain = build_expval_func(CircuitConfig(**kwargs, control_variate=False))(
            small_params
        )
        _, std_err_cv = build_expval_func(CircuitConfig(**kwargs, control_variate=True))(
            small_params
        )

        # For small theta the CV variance is O(theta^2) per sample versus O(1)
        # for the plain estimator. With theta=0.01 the ratio should be at least
        # two orders of magnitude. Skip degenerate observables for which the
        # plain estimator already has zero variance.
        meaningful = std_err_plain > 1e-4
        assert meaningful.any(), "test setup degenerate: plain std_err uniformly zero"
        ratio = std_err_cv[meaningful] / std_err_plain[meaningful]
        assert np.all(ratio < 0.05)

    def test_control_variate_reduces_variance_across_seeds(self):
        """Independent estimator variance across PRNG seeds must drop significantly with CV."""
        n_qubits = 3
        rng = np.random.default_rng(42)
        init_state_elems = jnp.array(rng.binomial(1, 0.5, size=(5, n_qubits)))
        amps = rng.uniform(0.5, 1.5, size=5)
        init_state_amps = jnp.array(amps / np.linalg.norm(amps))
        obs_batch = [[3, 0, 3], [0, 3, 3]]
        gates = {0: [[0, 1]], 1: [[1, 2]]}
        params = jnp.array([0.05, 0.05])
        n_samples = 2000

        n_seeds = 30
        seeds = jax.random.split(jax.random.PRNGKey(0), n_seeds)

        def _estimates(use_cv):
            config = CircuitConfig(
                gates=gates,
                observables=obs_batch,
                n_samples=n_samples,
                key=seeds[0],
                n_qubits=n_qubits,
                init_state_elems=init_state_elems,
                init_state_amps=init_state_amps,
                control_variate=use_cv,
            )
            f = build_expval_func(config)
            vals = np.stack([np.asarray(f(params, key=seeds[k])[0]) for k in range(n_seeds)])
            return vals

        vals_plain = _estimates(False)
        vals_cv = _estimates(True)

        var_plain = vals_plain.var(axis=0, ddof=1)
        var_cv = vals_cv.var(axis=0, ddof=1)

        meaningful = var_plain > 1e-10
        assert meaningful.any()
        assert np.all(var_cv[meaningful] < 0.1 * var_plain[meaningful])

    def test_control_variate_no_init_state_z_observable(self):
        r"""For pure-Z observables on |0...0>, CV adds the constant mu_Y=1 and is consistent."""
        n_qubits = 3
        gates = {0: [[0, 1]], 1: [[1, 2]]}
        params = jnp.array([0.3, 0.6])
        obs_batch = [[3, 3, 3], [3, 0, 3]]
        n_samples = 4000
        key = jax.random.PRNGKey(2)

        kwargs = dict(
            gates=gates,
            observables=obs_batch,
            n_samples=n_samples,
            key=key,
            n_qubits=n_qubits,
        )
        plain_val, _ = build_expval_func(CircuitConfig(**kwargs, control_variate=False))(params)
        cv_val, _ = build_expval_func(CircuitConfig(**kwargs, control_variate=True))(params)

        assert np.allclose(plain_val, cv_val, atol=1e-10)

    def test_control_variate_gradient_matches_plain(self):
        """Gradients of the CV estimator match plain gradients (up to MC error from shared samples)."""
        n_qubits = 3
        init_state_elems = jnp.array([[0, 0, 0], [1, 1, 0], [0, 1, 1]])
        init_state_amps = jnp.array([0.7, 0.5, np.sqrt(1 - 0.49 - 0.25)])
        obs_batch = [[3, 3, 3], [3, 0, 0], [0, 3, 3]]
        gates = {0: [[0, 1]], 1: [[1, 2]], 2: [[0, 2]]}
        params = jnp.array([0.21, 0.13, 0.45])
        n_samples = 50000
        key = jax.random.PRNGKey(11)

        kwargs = dict(
            gates=gates,
            observables=obs_batch,
            n_samples=n_samples,
            key=key,
            n_qubits=n_qubits,
            init_state_elems=init_state_elems,
            init_state_amps=init_state_amps,
        )

        def make_loss(use_cv):
            f = build_expval_func(CircuitConfig(**kwargs, control_variate=use_cv))
            return lambda p: jnp.sum(f(p)[0])

        grad_plain = jax.grad(make_loss(False))(params)
        grad_cv = jax.grad(make_loss(True))(params)

        atol = 4.0 * len(obs_batch) / np.sqrt(n_samples)
        assert np.allclose(grad_plain, grad_cv, atol=atol)

    def test_cv_fast_path_unique_support(self):
        """Fast path on unique support reproduces the basis-state probability moment.

        With distinct rows in ``init_state_elems`` the closed form reduces to
        ``sum_i |P_i|^2 * (-1)^{b_a . x_i}``. The identity observable then
        sums to ``||P||^2 = 1`` and a single-Z observable to a signed
        marginal of the basis-state probabilities.
        """
        X = jnp.array(
            [[0, 0, 0, 0], [0, 1, 0, 0], [1, 0, 1, 0], [1, 1, 1, 1]],
            dtype=jnp.int32,
        )
        amps = np.array([0.4, 0.3, 0.5, np.sqrt(1 - 0.16 - 0.09 - 0.25)])
        P = jnp.array(amps)

        obs_batch = [[0, 0, 0, 0], [3, 0, 0, 0]]
        bitflips, _, _ = _prep_observables(obs_batch)
        vals = _cv_mean_z_only(bitflips, X, P)

        expected_identity = float(np.sum(amps**2))
        signs_q0 = np.array([1, 1, -1, -1])
        expected_z0 = float(np.sum(amps**2 * signs_q0))

        assert np.isclose(float(vals[0]), expected_identity, atol=1e-12)
        assert np.isclose(float(vals[1]), expected_z0, atol=1e-12)

    def test_control_variate_jit_compatible(self):
        """The CV-enabled expval function must be jit-compilable."""
        n_qubits = 2
        init_state_elems = jnp.array([[0, 0], [1, 1]])
        init_state_amps = jnp.array([1 / np.sqrt(2), 1 / np.sqrt(2)])
        obs_batch = [[3, 3], [3, 0]]

        config = CircuitConfig(
            gates={0: [[0, 1]]},
            observables=obs_batch,
            n_samples=512,
            key=jax.random.PRNGKey(0),
            n_qubits=n_qubits,
            init_state_elems=init_state_elems,
            init_state_amps=init_state_amps,
            control_variate=True,
        )
        f = jax.jit(build_expval_func(config))
        params = jnp.array([0.1])

        val_a, _ = f(params)
        val_b, _ = f(params)
        assert np.allclose(val_a, val_b)
