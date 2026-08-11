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
    _core_expval_execution,
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


def _all_bitstrings(n_qubits):
    """Every bitstring of length ``n_qubits``, so a sample mean becomes an exact average."""
    return jnp.array([[int(b) for b in format(k, f"0{n_qubits}b")] for k in range(2**n_qubits)])


def _taylor_control_on_samples(
    params, samples, obs_data, generators, param_map, elems=None, amps=None
):
    """Evaluate the order-2 Taylor control variate on ``samples``.

    The control consumes ``(phases, E, H)`` produced by the core integrand, so it is
    built here exactly the way ``build_expval_func`` builds it internally.
    """
    _, phases, E, H = _core_expval_execution(
        params, None, samples, obs_data, elems, amps, generators, param_map, None
    )
    return _control_variate_expval_execution(phases, E, H)


def _taylor_coefficients(params, obs_data, generators, param_map):
    """The per-observable coefficients ``a_g = 2[(b.g) mod 2] * theta_g``."""
    bitflips = np.array(obs_data[0])
    gen = np.array(generators)
    expanded = np.array(params)[np.array(param_map)]
    return (2 * ((bitflips @ gen.T) % 2)) * expanded[np.newaxis, :]


class TestControlVariate:
    """Tests for the order-2 Taylor control variate and the CV branch of build_expval_func."""

    @staticmethod
    def _obs_data(obs_batch):
        """Preprocess an integer-coded observable batch into (bitflips, mask_XY, y_phase)."""
        return _prep_observables(jnp.array(obs_batch))

    @pytest.mark.parametrize(
        "n_qubits, gates, params, obs_batch, init_state_spec",
        [
            # Default |0...0> state.
            (3, {0: [[0], [1]], 1: [[0, 1], [1, 2]]}, [0.35, 0.22], [[1, 3, 2]], None),
            (2, {0: [[0, 1]]}, [0.4], [[3, 3]], None),
            (2, {0: [[0, 1]]}, [0.4], [[0, 0]], None),
            (3, {0: [[0], [1]], 1: [[0, 1], [1, 2]]}, [0.3, -0.45], [[3, 3, 0], [1, 0, 2]], None),
            # Mixed batch spanning every Pauli type.
            (
                3,
                {0: [[0], [1], [2]], 1: [[0, 1], [1, 2]]},
                [0.25, 0.17],
                [[3, 3, 0], [1, 0, 2], [2, 2, 0], [0, 0, 0], [1, 2, 3]],
                None,
            ),
            # Single non-zero basis element (a bitstring initial state).
            (3, {0: [[0, 1]], 1: [[1, 2]]}, [0.3, 0.2], [[1, 3, 2]], [1, 0, 1]),
            # Superposition initial state. Real amplitudes: see the note on
            # test_control_variate_branch_is_unbiased_vs_pennylane.
            (
                2,
                {0: [[0, 1]]},
                [0.31],
                [[3, 3], [1, 1], [2, 2]],
                ([[0, 0], [1, 1]], [1 / np.sqrt(2), 1 / np.sqrt(2)]),
            ),
        ],
    )
    def test_analytic_mean_equals_exact_average_of_control(
        self, n_qubits, gates, params, obs_batch, init_state_spec
    ):
        """The analytic expectation value equals the exact uniform average of the Taylor control."""
        # pylint: disable=too-many-arguments
        generators, param_map = _parse_generator_dict(gates, n_qubits)
        elems, amps = _prepare_jax_state(init_state_spec)
        obs_data = self._obs_data(obs_batch)
        params_jax = jnp.array(params)

        control = _taylor_control_on_samples(
            params_jax, _all_bitstrings(n_qubits), obs_data, generators, param_map, elems, amps
        )
        exact_average = np.array(jnp.mean(control, axis=1))
        tau = np.array(
            _control_variate_expected_value(
                params_jax, obs_data, generators, param_map, elems, amps
            )
        )

        assert np.allclose(exact_average, tau, atol=1e-6)

    def test_analytic_mean_exact_for_complex_amplitudes(self):
        """Test the closed form is correct for a complex-amplitude initial state."""
        n_qubits = 3
        gates = {0: [[0], [1]], 1: [[0, 1], [1, 2]]}
        generators, param_map = _parse_generator_dict(gates, n_qubits)
        obs_batch = [[3, 3, 0], [1, 0, 2], [2, 2, 0], [0, 0, 0]]
        params = jnp.array([0.3, 0.2])

        rng = np.random.default_rng(0)
        elems = jnp.array(np.array([[0, 0, 0], [1, 0, 1], [0, 1, 1], [1, 1, 0]], dtype=float))
        amps = rng.normal(size=4) + 1j * rng.normal(size=4)
        amps = jnp.array(amps / np.linalg.norm(amps))

        obs_data = self._obs_data(obs_batch)
        control = _taylor_control_on_samples(
            params, _all_bitstrings(n_qubits), obs_data, generators, param_map, elems, amps
        )
        tau = _control_variate_expected_value(
            params, obs_data, generators, param_map, elems, amps
        )

        assert np.allclose(np.array(jnp.mean(control, axis=1)), np.array(tau), atol=1e-6)

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
    def test_zero_params_matches_identity_control(self, obs_batch, expected_tau):
        """At theta=0 the Taylor control collapses to the identity control.

        With every angle zero we have E = 0, so the truncation ``1 + iE - E^2/2``
        reduces to 1 and tau must equal the theta=0 values: 1 for pure I/Z
        observables on ``|0...0>`` and 0 as soon as any X or Y is present.
        """
        n_qubits = 3
        gates = {0: [[0], [1]], 1: [[0, 1], [1, 2]]}
        generators, param_map = _parse_generator_dict(gates, n_qubits)
        obs_data = self._obs_data(obs_batch)

        tau = _control_variate_expected_value(
            jnp.zeros(len(gates)), obs_data, generators, param_map, None, None
        )

        assert np.allclose(np.array(tau), np.array(expected_tau), atol=1e-6)

    @pytest.mark.parametrize("obs_batch", [[[3, 3, 0]], [[3, 3, 3]], [[0, 0, 0]], [[3, 0, 3]]])
    @pytest.mark.parametrize("scale", [0.15, 0.5])
    def test_tau_closed_form_for_pure_iz_default_state(self, obs_batch, scale):
        """For pure I/Z observables on |0...0>, tau = 1 - (1/2) sum_g a_g^2."""
        n_qubits = 3
        gates = {0: [[0], [1]], 1: [[0, 1], [1, 2]]}
        generators, param_map = _parse_generator_dict(gates, n_qubits)
        rng = np.random.default_rng(4)
        params = jnp.array(rng.normal(size=len(gates)) * scale)
        obs_data = self._obs_data(obs_batch)

        tau = np.array(
            _control_variate_expected_value(params, obs_data, generators, param_map, None, None)
        )
        a = _taylor_coefficients(params, obs_data, generators, param_map)
        expected = 1.0 - 0.5 * np.sum(a**2, axis=1)

        assert np.allclose(tau, expected, atol=1e-6)

    def test_taylor_control_fluctuates_for_pure_z_default_state(self):
        """For pure Z on |0...0> the Taylor control varies over bitstrings."""
        n_qubits = 3
        gates = {0: [[0], [1]], 1: [[0, 1], [1, 2]]}
        generators, param_map = _parse_generator_dict(gates, n_qubits)
        obs_data = self._obs_data([[3, 3, 3]])
        samples = _all_bitstrings(n_qubits)

        control = np.array(
            _taylor_control_on_samples(
                jnp.array([0.4, 0.3]), samples, obs_data, generators, param_map
            )
        )
        # Non-degenerate: this is what the identity control fails to provide here.
        assert np.var(control) > 1e-6

        # At theta=0 it does collapse back to the constant identity control.
        control_at_zero = np.array(
            _taylor_control_on_samples(
                jnp.zeros(len(gates)), samples, obs_data, generators, param_map
            )
        )
        assert np.allclose(control_at_zero, 1.0, atol=1e-6)

    def test_taylor_control_equals_integrand_when_phase_vanishes(self):
        """For observables with no Z or Y the control reproduces the integrand exactly."""
        n_qubits = 3
        gates = {0: [[0], [1]], 1: [[0, 1], [1, 2]]}
        generators, param_map = _parse_generator_dict(gates, n_qubits)
        obs_data = self._obs_data([[1, 1, 1]])
        params = jnp.array([0.4, 0.3])
        samples = _all_bitstrings(n_qubits)

        integrand, phases, E, H = _core_expval_execution(
            params, None, samples, obs_data, None, None, generators, param_map, None
        )
        control = _control_variate_expval_execution(phases, E, H)

        assert np.allclose(np.array(E), 0.0, atol=1e-12)
        assert np.allclose(np.array(control), np.array(integrand), atol=1e-12)

    def test_control_variate_eliminates_all_error_when_control_is_exact(self):
        """When the control equals the integrand the CV branch becomes exact."""
        n_qubits = 2
        gates = {0: [[0, 1]]}
        obs_batch = [[1, 1]]
        params = jnp.array([0.4])
        generators, param_map = _parse_generator_dict(gates, n_qubits)
        obs_data = self._obs_data(obs_batch)

        base_kwargs = {
           "gates": gates,
           "observables": obs_batch,
           "n_samples": 5000,
           "key": jax.random.PRNGKey(3),
           "n_qubits": n_qubits,
        }

        plain_mean, plain_err = build_expval_func(CircuitConfig(**base_kwargs))(params)
        cv_mean, cv_err = build_expval_func(CircuitConfig(control_variate=True, **base_kwargs))(
            params
        )
        tau = _control_variate_expected_value(
            params, obs_data, generators, param_map, None, None
        )

        assert np.all(np.isfinite(np.array(cv_mean)))
        assert np.all(np.isfinite(np.array(cv_err)))
        # The control removes the sampling error outright.
        assert np.allclose(np.array(cv_mean), np.array(tau), atol=1e-6)
        assert np.allclose(np.array(cv_err), 0.0, atol=1e-6)
        # The plain estimator does not have that benefit on the same samples.
        assert np.array(plain_err)[0] > np.array(cv_err)[0]
        assert not np.allclose(np.array(plain_mean), np.array(tau), atol=1e-6)

    def test_control_variate_helper_shapes(self):
        """The two helpers return the documented shapes."""
        n_qubits = 3
        gates = {0: [[0], [1]], 1: [[0, 1], [1, 2]]}
        generators, param_map = _parse_generator_dict(gates, n_qubits)
        obs_batch = [[3, 3, 0], [1, 0, 2]]
        n_samples = 128
        samples = (
            jax.random.bits(  # any (n_samples, n_qubits) binary array
                jax.random.PRNGKey(0), shape=(n_samples, n_qubits), dtype=jnp.uint8
            )
            % 2
        )
        obs_data = self._obs_data(obs_batch)
        params = jnp.array([0.3, 0.2])

        elems = jnp.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
        amps = jnp.array([1 / np.sqrt(2), 1 / np.sqrt(2)])

        cv = _taylor_control_on_samples(
            params, samples, obs_data, generators, param_map, elems, amps
        )
        tau = _control_variate_expected_value(
            params, obs_data, generators, param_map, elems, amps
        )

        assert cv.shape == (len(obs_batch), n_samples)
        assert tau.shape == (len(obs_batch),)

    def test_control_variate_rejects_phase_layer(self):
        """Phase layers are not compatible with the control variate."""

        def phase_fn(params, z):
            return jnp.sum(params * z)

        config = CircuitConfig(
            gates={0: [[0, 1]]},
            observables=[[3, 3]],
            n_samples=100,
            key=jax.random.PRNGKey(0),
            n_qubits=2,
            phase_fn=phase_fn,
            control_variate=True,
        )
        with pytest.raises(ValueError, match="Phase layers are not compatible"):
            build_expval_func(config)

    @pytest.mark.parametrize(
        "n_qubits, gates, obs_strings, init_state_spec",
        [
            (3, {0: [[0], [1]], 1: [[0, 1], [1, 2]]}, ["X", "Z", "Y"], None),
            (3, {0: [[0], [1], [2]]}, ["Z", "Z", "Z"], None),
            (2, {0: [[0, 1]]}, [["Z", "Z"], ["X", "X"], ["Y", "Y"]], None),
            (3, {0: [[0, 1]], 1: [[1, 2]]}, ["X", "Z", "Y"], [1, 0, 1]),
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
        params = rng.uniform(-0.6, 0.6, size=len(gates))
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

    def test_control_variate_is_differentiable(self):
        """The CV branch supports jax.grad, since it is meant for optimization."""
        n_qubits = 3
        gates = {0: [[0], [1]], 1: [[0, 1], [1, 2]]}
        config = CircuitConfig(
            gates=gates,
            observables=[[3, 3, 0], [1, 0, 2]],
            n_samples=2000,
            key=jax.random.PRNGKey(5),
            n_qubits=n_qubits,
            control_variate=True,
        )
        expval_func = build_expval_func(config)

        def cost(params):
            return jnp.sum(expval_func(params)[0])

        grad = jax.grad(cost)(jnp.array([0.2, 0.1]))

        assert grad.shape == (len(gates),)
        assert np.all(np.isfinite(np.array(grad)))

    def test_control_variate_reduces_variance_for_pure_z_default_state(self):
        """The Taylor control reduces dispersion where the identity control cannot."""
        n_qubits = 3
        gates = {0: [[0], [1], [2]], 1: [[0, 1], [1, 2]]}
        obs_batch = [[3, 3, 3]]
        params = jnp.array([0.08, 0.05])

        def empirical_std_of_mean(control_variate):
            means = []
            for seed in range(30):
                config = CircuitConfig(
                    gates=gates,
                    observables=obs_batch,
                    n_samples=4000,
                    key=jax.random.PRNGKey(seed),
                    n_qubits=n_qubits,
                    control_variate=control_variate,
                )
                mean, _ = build_expval_func(config)(params)
                means.append(float(np.array(mean)[0]))
            return np.std(means, ddof=1)

        assert empirical_std_of_mean(True) < empirical_std_of_mean(False)

    def test_control_variate_reduces_variance_for_custom_state(self):
        """With a non-trivial initial state the CV branch lowers the variance."""
        n_qubits = 3
        gates = {0: [[0], [1], [2]], 1: [[0, 1], [1, 2]]}
        obs_batch = [[3, 3, 0]]
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
