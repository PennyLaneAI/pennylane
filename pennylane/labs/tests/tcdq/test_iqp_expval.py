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
    _control_variate_expected_value,
    _control_variate_expval_execution,
    _parse_generator_dict,
    _prep_observables,
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


class TestControlVariate:
    """Tests for the theta=0 control-variate helpers and the CV branch of build_expval_func."""

    @staticmethod
    def _obs_data(obs_batch):
        """Preprocess an integer-coded observable batch into (bitflips, mask_XY, y_phase)."""
        return _prep_observables(jnp.array(obs_batch))

    # ------------------------------------------------------------------
    # 1. The key requested test: the analytic control expectation must equal
    #    the *core* estimator evaluated at params = 0, on the SAME samples.
    #    At theta=0 the core integrand collapses to the control integrand, so the
    #    match is exact (up to float tolerance), not merely statistical.
    # ------------------------------------------------------------------
    @pytest.mark.parametrize(
        "n_qubits, gates, obs_strings, init_state_spec",
        [
            (3, {0: [[0], [1]], 1: [[0, 1], [1, 2]]}, ["X", "Z", "Y"], None),
            (2, {0: [[0, 1]]}, ["Z", "Z"], None),
            (2, {0: [[0, 1]]}, ["I", "I"], None),
            (2, {0: [[0, 1]]}, [["Z", "Z"], ["X", "X"]], None),
            (3, {0: [[0, 1]], 1: [[1, 2]]}, ["X", "Z", "Y"], [1, 0, 1]),
            (3, {0: [[0], [1], [2]]}, ["Z", "Z", "Z"], [1, 1, 1]),
            (
                2,
                {0: [[0, 1]]},
                [["Z", "Z"], ["X", "X"], ["Y", "Y"]],
                ([[0, 0], [1, 1]], [1 / np.sqrt(2), 1 / np.sqrt(2)]),
            ),
        ],
    )
    def test_control_variate_expected_value_matches_core_at_zero_params(
        self, n_qubits, gates, obs_strings, init_state_spec
    ):
        """_control_variate_expected_value equals the core estimator at params=0."""
        obs_batch, _ = _prepare_obs_batch(obs_strings)
        jax_state_elems, jax_state_amps = _prepare_jax_state(init_state_spec)

        n_params = len(gates)
        zero_params = jnp.zeros(n_params)
        key = jax.random.PRNGKey(7)
        # Large sample count: the core mean at params=0 is itself a Monte Carlo
        # estimate of tau, so compare within Monte Carlo tolerance.
        n_samples = 200000

        config = CircuitConfig(
            gates=gates,
            observables=obs_batch,
            n_samples=n_samples,
            key=key,
            n_qubits=n_qubits,
            init_state_elems=jax_state_elems,
            init_state_amps=jax_state_amps,
        )
        core_func = build_expval_func(config)
        core_mean_at_zero, _ = core_func(zero_params)

        obs_data = self._obs_data(obs_batch)
        tau = _control_variate_expected_value(obs_data, jax_state_elems, jax_state_amps)

        atol = 3.5 / np.sqrt(n_samples)
        assert np.allclose(np.array(core_mean_at_zero), np.array(tau), atol=atol)

    def test_control_variate_expected_value_equals_persample_mean_exactly(self):
        """Analytic tau equals the exact mean of the per-sample control over all bitstrings."""
        n_qubits = 3
        obs_batch = [[3, 3, 0], [1, 0, 2], [2, 2, 0], [0, 0, 0]]
        # Custom complex initial state.
        rng = np.random.default_rng(0)
        elems = np.array([[0, 0, 0], [1, 0, 1], [0, 1, 1], [1, 1, 0]])
        amps = rng.normal(size=4) + 1j * rng.normal(size=4)
        amps = amps / np.linalg.norm(amps)
        elems_j = jnp.array(elems.astype(float))
        amps_j = jnp.array(amps)

        # Enumerate ALL 2**n bitstrings -> the per-sample mean is the exact expectation.
        all_bits = jnp.array(
            [[int(b) for b in format(k, f"0{n_qubits}b")] for k in range(2**n_qubits)]
        )
        obs_data = self._obs_data(obs_batch)

        cv_samples = _control_variate_expval_execution(all_bits, obs_data, elems_j, amps_j)
        tau_from_samples = np.array(jnp.mean(cv_samples, axis=1))
        tau_analytic = np.array(_control_variate_expected_value(obs_data, elems_j, amps_j))

        assert np.allclose(tau_from_samples, tau_analytic, atol=1e-6)

    # ------------------------------------------------------------------
    # 2. Pure I/Z observables from |0...0>: tau must be exactly 1, and the
    #    per-sample control must be the constant 1 (the degenerate/no-op case).
    # ------------------------------------------------------------------
    @pytest.mark.parametrize(
        "obs_batch, expected_tau",
        [
            ([[3, 0, 0]], [1.0]),  # Z I I
            ([[3, 3, 0]], [1.0]),  # Z Z I
            ([[3, 3, 3]], [1.0]),  # Z Z Z
            ([[0, 0, 0]], [1.0]),  # I I I
            ([[1, 0, 0]], [0.0]),  # X -> 0
            ([[2, 0, 0]], [0.0]),  # Y -> 0
            ([[3, 0, 0], [1, 0, 0], [0, 2, 0]], [1.0, 0.0, 0.0]),  # mixed batch
        ],
    )
    def test_default_state_tau_is_one_for_pure_iz(self, obs_batch, expected_tau):
        """For |0...0>, tau = 1 for pure I/Z observables and 0 when any X/Y is present."""
        obs_data = self._obs_data(obs_batch)
        tau = _control_variate_expected_value(obs_data, None, None)
        assert np.allclose(np.array(tau), np.array(expected_tau), atol=1e-12)

    def test_default_state_pure_z_control_is_constant_one(self):
        """The per-sample control for a pure Z observable from |0...0> is identically 1."""
        n_qubits = 3
        obs_batch = [[3, 3, 3]]
        samples = jnp.array(
            [[int(b) for b in format(k, f"0{n_qubits}b")] for k in range(2**n_qubits)]
        )
        obs_data = self._obs_data(obs_batch)
        cv = np.array(_control_variate_expval_execution(samples, obs_data, None, None))
        assert np.allclose(cv, 1.0, atol=1e-12)
        # Zero variance => degenerate control (the c = -cov/var guard must handle it).
        assert np.isclose(np.var(cv), 0.0, atol=1e-12)

    def test_control_variate_branch_no_op_for_pure_z_default_state(self):
        """The CV branch must not produce NaNs when the control is degenerate (var=0)."""
        n_qubits = 2
        gates = {0: [[0, 1]]}
        obs_batch = [[3, 3]]  # pure Z from |0...0> -> control is constant
        params = jnp.array([0.4])
        key = jax.random.PRNGKey(3)
        n_samples = 5000

        base_kwargs = dict(
            gates=gates,
            observables=obs_batch,
            n_samples=n_samples,
            key=key,
            n_qubits=n_qubits,
        )
        plain_mean, _ = build_expval_func(CircuitConfig(**base_kwargs))(params)
        cv_mean, _ = build_expval_func(CircuitConfig(control_variate=True, **base_kwargs))(params)

        assert np.all(np.isfinite(np.array(cv_mean)))
        # With a degenerate control the CV estimator falls back to the plain mean.
        assert np.allclose(np.array(plain_mean), np.array(cv_mean), atol=1e-6)

    # ------------------------------------------------------------------
    # 3. Shapes and unbiasedness of the CV branch.
    # ------------------------------------------------------------------
    def test_control_variate_helper_shapes(self):
        """The two helpers return the documented shapes."""
        n_qubits = 3
        obs_batch = [[3, 3, 0], [1, 0, 2]]
        n_samples = 128
        samples = (
            jax.random.bits(  # any (n_samples, n_qubits) binary array
                jax.random.PRNGKey(0), shape=(n_samples, n_qubits), dtype=jnp.uint8
            )
            % 2
        )
        obs_data = self._obs_data(obs_batch)

        elems = jnp.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
        amps = jnp.array([1 / np.sqrt(2), 1 / np.sqrt(2)])

        cv = _control_variate_expval_execution(samples, obs_data, elems, amps)
        tau = _control_variate_expected_value(obs_data, elems, amps)

        assert cv.shape == (len(obs_batch), n_samples)
        assert tau.shape == (len(obs_batch),)

    @pytest.mark.parametrize(
        "n_qubits, gates, obs_strings, init_state_spec",
        [
            (
                3,
                {0: [[0], [1]], 1: [[0, 1], [1, 2]]},
                ["X", "Z", "Y"],
                ([[0, 0, 0], [1, 0, 1], [0, 1, 1]], [0.6, 0.6, 0.52915026]),
            ),
            (
                2,
                {0: [[0, 1]], 1: [[0]]},
                [["Z", "Z"], ["X", "X"]],
                ([[0, 0], [1, 1]], [1 / np.sqrt(2), 1 / np.sqrt(2)]),
            ),
        ],
    )
    def test_control_variate_branch_is_unbiased_vs_pennylane(
        self, n_qubits, gates, obs_strings, init_state_spec
    ):
        """The CV estimator agrees with the PennyLane ground truth (unbiasedness)."""
        generators_binary, param_map = _parse_generator_dict(gates, n_qubits)
        generators_pl = [list(np.where(row)[0]) for row in generators_binary]

        rng = np.random.default_rng(1)
        n_params = len(gates)
        params = rng.uniform(-0.6, 0.6, size=n_params)
        params_pl = np.array(params)[param_map]

        obs_batch, _ = _prepare_obs_batch(obs_strings)
        pl_state = _prepare_pennylane_state(n_qubits, init_state_spec)
        jax_state_elems, jax_state_amps = _prepare_jax_state(init_state_spec)

        exact_vals = _run_pennylane_ground_truth(generators_pl, params_pl, obs_batch, pl_state)

        n_samples = 40000
        atol = 3.5 / np.sqrt(n_samples)

        config = CircuitConfig(
            gates=gates,
            observables=obs_batch,
            n_samples=n_samples,
            key=jax.random.PRNGKey(42),
            n_qubits=n_qubits,
            init_state_elems=jax_state_elems,
            init_state_amps=jax_state_amps,
            control_variate=True,
        )
        cv_mean, _ = build_expval_func(config)(jnp.array(params))

        assert np.allclose(exact_vals, cv_mean, atol=atol)

    def test_control_variate_reduces_variance_for_custom_state(self):
        """With a non-trivial initial state the CV branch lowers the *actual* dispersion
        of the estimator across seeds.

        We measure the empirical standard deviation of the returned mean over many PRNG
        keys rather than the reported ``std_err``: the two branches are compared on the
        same footing, independent of any per-observable normalization convention in the
        returned standard error. Small rotation angles are used so that the theta=0
        control is strongly correlated with the estimator and the reduction is large and
        non-flaky.
        """
        n_qubits = 3
        gates = {0: [[0], [1], [2]], 1: [[0, 1], [1, 2]]}
        obs_batch = [[3, 3, 0]]  # ZZ: control is constant for |0>, but not for a custom state
        params = jnp.array([0.05, 0.03])  # small angles -> high correlation -> large reduction

        rng = np.random.default_rng(2)
        elems = np.array([[0, 0, 0], [1, 0, 1], [0, 1, 1], [1, 1, 0]])
        amps = rng.normal(size=4) + 1j * rng.normal(size=4)
        amps = amps / np.linalg.norm(amps)
        elems_j = jnp.array(elems.astype(float))
        amps_j = jnp.array(amps)

        def empirical_std_of_mean(control_variate):
            means = []
            for seed in range(40):
                config = CircuitConfig(
                    gates=gates,
                    observables=obs_batch,
                    n_samples=4000,
                    key=jax.random.PRNGKey(seed),
                    n_qubits=n_qubits,
                    init_state_elems=elems_j,
                    init_state_amps=amps_j,
                    control_variate=control_variate,
                )
                mean, _ = build_expval_func(config)(params)
                means.append(float(np.array(mean)[0]))
            return np.std(means, ddof=1)

        plain_std = empirical_std_of_mean(False)
        cv_std = empirical_std_of_mean(True)

        assert cv_std < plain_std
