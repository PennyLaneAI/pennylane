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

# Attempt to import JAX and the lab module
jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

try:
    from pennylane.labs.phox.simulator_pure_functions import (
        _iqp_expval_core,
        _parse_iqp_dict,
        iqp_expval,
    )
except ImportError:
    pytest.skip("pennylane.labs.phox not found", allow_module_level=True)


def _prepare_obs_batch(obs_strings):
    """Refactor helper: Normalize obs_strings into a batch array and count qubits."""
    obs_arr = np.array(obs_strings)
    if obs_arr.ndim == 1:
        return [obs_strings], len(obs_strings)
    return obs_strings, len(obs_strings[0])


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

    # Otherwise assume (X, P) tuple or similar structure for superposition
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


def _run_pennylane_ground_truth(generators_pl, params_pl, obs_batch, init_state):
    """Run the PennyLane default.qubit simulation for the batch of observables."""
    exact_vals = []
    for obs in obs_batch:
        circuit = iqp_circuit_pl(generators_pl, params_pl, obs, init_state)
        exact_vals.append(circuit())
    return np.array(exact_vals).flatten()


def iqp_circuit_pl(generators, params, obs, init_state):
    """Creates a PennyLane QNode for the IQP circuit."""
    n_qubits = len(obs)

    expval_ops = []
    for i, op in enumerate(obs):
        if op == "X":
            expval_ops.append(qml.X(i))
        elif op == "Y":
            expval_ops.append(qml.Y(i))
        elif op == "Z":
            expval_ops.append(qml.Z(i))
        elif op == "I":
            expval_ops.append(qml.Identity(i))

    expval_op = qml.prod(*expval_ops)

    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit():
        # Start with specified initial state
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
    def test_iqp_expval_core_vs_pennylane(
        self, n_samples, obs_strings, generators_pl, params, init_state_spec
    ):
        """Test that _iqp_expval_core matches PennyLane default.qubit with parametrization."""
        obs_batch, n_qubits = _prepare_obs_batch(obs_strings)
        pl_state = _prepare_pennylane_state(n_qubits, init_state_spec)
        jax_state = _prepare_jax_state(init_state_spec)

        exact_vals = _run_pennylane_ground_truth(generators_pl, params, obs_batch, pl_state)

        generators_matrix = np.zeros((len(generators_pl), n_qubits), dtype=int)
        for i, wires in enumerate(generators_pl):
            generators_matrix[i, wires] = 1
        generators = jnp.array(generators_matrix)

        params_jax = jnp.array(params)
        obs_jax = np.array(obs_batch)
        key = jax.random.PRNGKey(42)
        atol = 3.5 / np.sqrt(n_samples)

        expval_func = _iqp_expval_core(generators, obs_jax, n_samples, key, init_state=jax_state)
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
    def test_iqp_expval_vs_pennylane(self, n_qubits, gates, params, obs_strings, init_state_spec):
        """Test that iqp_expval matches PennyLane default.qubit."""
        generators_binary, param_map = _parse_iqp_dict(gates, n_qubits)
        generators_pl = [
            list(np.where(row)[0]) for row in generators_binary
        ]  # generators in list form for PL
        params_pl = np.array(params)[param_map]  # one entry per gate

        obs_batch, _ = _prepare_obs_batch(obs_strings)
        pl_state = _prepare_pennylane_state(n_qubits, init_state_spec)
        jax_state = _prepare_jax_state(init_state_spec)

        exact_vals = _run_pennylane_ground_truth(generators_pl, params_pl, obs_batch, pl_state)

        obs_jax = np.array(obs_batch)
        key = jax.random.PRNGKey(42)
        n_samples = 10000
        atol = 3.5 / np.sqrt(n_samples)

        approx_val, _ = iqp_expval(
            gates, params, obs_jax, n_samples, n_qubits, key, init_state=jax_state
        )

        assert np.allclose(exact_vals, approx_val, atol=atol)


@pytest.mark.parametrize(
    "circuit_def,n_qubits,expected_generators,expected_param_map",
    [
        ({0: [[0, 1]]}, 3, [[1, 1, 0]], [0]),
        ({0: [[0]], 1: [[1, 2], [0, 2]]}, 3, [[1, 0, 0], [0, 1, 1], [1, 0, 1]], [0, 1, 1]),
        ({}, 2, np.zeros((0, 2), dtype=int), []),
        ({10: [[0]], 2: [[1]]}, 2, [[0, 1], [1, 0]], [2, 10]),
    ],
)
def test_parse_iqp_dict(circuit_def, n_qubits, expected_generators, expected_param_map):
    """Test that _parse_iqp_dict correctly converts dictionary circuit definition into JAX arrays."""
    generators, param_map = _parse_iqp_dict(circuit_def, n_qubits)

    assert isinstance(generators, jnp.ndarray)
    assert isinstance(param_map, jnp.ndarray)

    expected_generators = np.array(expected_generators)
    expected_param_map = np.array(expected_param_map)

    assert generators.shape == expected_generators.shape
    assert param_map.shape == expected_param_map.shape

    assert np.allclose(generators, expected_generators)
    assert np.allclose(param_map, expected_param_map)


def test_parse_iqp_dict_index_error():
    """Test that _parse_iqp_dict raises IndexError if qubits indices are out of bounds."""
    circuit_def = {0: [[5]]}
    n_qubits = 2

    with pytest.raises(IndexError):
        _parse_iqp_dict(circuit_def, n_qubits)
