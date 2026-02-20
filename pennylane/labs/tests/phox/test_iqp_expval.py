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

import pennylane as qml

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

try:
    from pennylane.labs.phox.expval_functions import (
        CircuitConfig,
        _parse_generator_dict,
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
    """Convert spec into JAX (X, P) tuple format."""
    if init_state_spec is None:
        return None

    is_single_bitstring = isinstance(init_state_spec, list) and (
        not init_state_spec or not isinstance(init_state_spec[0], (list, tuple))
    )

    if is_single_bitstring:
        return (jnp.array([init_state_spec]), jnp.array([1.0]))

    return (jnp.array(init_state_spec[0]), jnp.array(init_state_spec[1]))


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
            expval_ops.append(qml.X(i))
        elif op == 2:
            expval_ops.append(qml.Y(i))
        elif op == 3:
            expval_ops.append(qml.Z(i))
        elif op == 0:
            expval_ops.append(qml.Identity(i))

    expval_op = qml.prod(*expval_ops)

    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit():
        qml.StatePrep(np.array(init_state), wires=range(n_qubits))

        for i in range(n_qubits):
            qml.Hadamard(i)

        for param, gen in zip(params, generators):
            qml.MultiRZ(2 * -param, wires=gen)

        for i in range(n_qubits):
            qml.Hadamard(i)

        return qml.expval(expval_op)

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
        # pylint: disable=too-many-arguments
        obs_batch, n_qubits = _prepare_obs_batch(obs_strings)
        pl_state = _prepare_pennylane_state(n_qubits, init_state_spec)
        jax_state = _prepare_jax_state(init_state_spec)

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
            init_state=jax_state,
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
    def test_build_expval_func_vs_pennylane(self, n_qubits, gates, params, obs_strings, init_state_spec):
        # pylint: disable=too-many-arguments
        generators_binary, param_map = _parse_generator_dict(gates, n_qubits)
        generators_pl = [
            list(np.where(row)[0]) for row in generators_binary
        ]
        params_pl = np.array(params)[param_map]

        obs_batch, _ = _prepare_obs_batch(obs_strings)
        pl_state = _prepare_pennylane_state(n_qubits, init_state_spec)
        jax_state = _prepare_jax_state(init_state_spec)

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
            init_state=jax_state,
        )
        expval_func = build_expval_func(config)
        approx_val, _ = expval_func(np.array(params))

        assert np.allclose(exact_vals, approx_val, atol=atol)

    def test_iqp_parameter_broadcasting(self):
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
        jax_state = (jnp.array([[0, 0], [1, 1]]), jnp.array([1 / jnp.sqrt(2), 1 / jnp.sqrt(2)]))

        n_qubits = 2
        dev = qml.device("default.qubit", wires=n_qubits)

        expval_ops = [qml.Z(0), qml.Y(1)]
        expval_op = qml.prod(*expval_ops)

        @qml.qnode(dev)
        def circuit():
            qml.StatePrep(np.array(pl_state), wires=range(n_qubits))

            for i in range(n_qubits):
                qml.Hadamard(i)

            for param, gen in zip(params, generators_pl):
                qml.MultiRZ(2 * -param, wires=gen)

            qml.DiagonalQubitUnitary(diagonal, wires=[0, 1])

            for i in range(n_qubits):
                qml.Hadamard(i)

            return qml.expval(expval_op)

        exact_val = circuit()

        gates = {0: [[0]], 1: [[1]], 2: [[0, 1]]}
        obs_batch = [[3, 2]]  # Using integer mapped observables

        config = CircuitConfig(
            n_qubits=n_qubits,
            gates=gates,
            observables=obs_batch,
            init_state=jax_state,
            phase_layer=compute_phase,
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
    circuit_def = {0: [[5]]}
    n_qubits = 2

    with pytest.raises(IndexError):
        _parse_generator_dict(circuit_def, n_qubits)
