# Copyright 2024 Xanadu Quantum Technologies Inc.

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
import pytest
import numpy as np
import pennylane as qml

# Attempt to import JAX and the lab module
jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

try:
    from pennylane.labs.phox.simulator_pure_functions import _iqp_expval_core, _parse_iqp_dict, iqp_expval
except ImportError:
    pytest.skip("pennylane.labs.phox not found", allow_module_level=True)


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
        "obs_strings, generators_pl, params",
        [
            (["X", "Z", "Y"], [[0], [1], [0, 1, 2]], [0.37, 0.95, 0.73]),
            (["X"], [[0]], [0.1]),
            (["Y", "Y"], [[0], [1], [0, 1]], [0.2, 0.3, 0.4]),
            (["Z", "Z", "Z"], [[0, 1], [1, 2]], [0.1, 0.2]),
            (
                ["X", "Y", "Z", "I"],
                [[0, 1], [2, 3], [0, 2, 3]],
                [0.1, 0.2, 0.3],
            ),
            (["I", "I", "I", "I"], [[0, 1], [2, 3]], [0.5, 0.6]),
        ],
    )
    def test_iqp_expval_core_vs_pennylane(
        self, n_samples, obs_strings, generators_pl, params
    ):
        """Test that _iqp_expval_core matches PennyLane default.qubit with parametrization."""
        n_qubits = len(obs_strings)

        state = np.zeros(2**n_qubits)
        state[0] = 1.0

        circuit = iqp_circuit_pl(generators_pl, params, obs_strings, state)
        exact_val = circuit()

        generators_matrix = np.zeros((len(generators_pl), n_qubits), dtype=int)
        for i, wires in enumerate(generators_pl):
            generators_matrix[i, wires] = 1
        generators = jnp.array(generators_matrix)

        params_jax = jnp.array(params)
        obs_jax = np.array([obs_strings])
        key = jax.random.PRNGKey(42)  # Fixed key for reproducibility
        atol = 3.5 / np.sqrt(n_samples)

        approx_val, _ = _iqp_expval_core(
            generators, params_jax, obs_jax, n_samples, key
        )

        assert np.allclose(exact_val, approx_val, atol=atol)

    @pytest.mark.parametrize(
        "n_qubits, gates, params, obs_strings",
        [
            (3, {0: [[0], [1]], 1: [[0, 1], [1, 2]]}, [0.1, 0.2], ["X", "Z", "Y"]),
            (2, {}, [], ["Z", "Z"]),
            (3, {0: [[0, 1]], 1: [[1, 2]]}, [0.1, 0.2], ["X", "I", "Z"]),
            (2, {0: [[0, 1]]}, [0.5], ["I", "I"]),
        ],
    )
    def test_iqp_expval_vs_pennylane(self, n_qubits, gates, params, obs_strings):
        """Test that iqp_expval matches PennyLane default.qubit."""
        generators_binary, param_map = _parse_iqp_dict(gates, n_qubits)
        generators_pl = [list(np.where(row)[0]) for row in generators_binary] # generators in list form for PL
        params_pl = np.array(params)[param_map] # one entry per gate

        state = np.zeros(2**n_qubits)
        state[0] = 1.0

        circuit = iqp_circuit_pl(generators_pl, params_pl, obs_strings, state)
        exact_val = circuit()

        obs_jax = np.array([obs_strings])
        key = jax.random.PRNGKey(42)
        n_samples = 10000
        atol = 3 * 1/np.sqrt(n_samples)

        approx_val, _ = iqp_expval(gates, params, obs_jax, n_samples, n_qubits, key)

        assert np.allclose(exact_val, approx_val, atol=atol)


@pytest.mark.parametrize(
    "circuit_def,n_qubits,expected_generators,expected_param_map",
    [
        (
            {0: [[0, 1]]},
            3,
            [[1, 1, 0]],
            [0]
        ),
        (
            {0: [[0]], 1: [[1, 2], [0, 2]]},
            3,
            [[1, 0, 0], [0, 1, 1], [1, 0, 1]],
            [0, 1, 1]
        ),
        (
            {},
            2,
            np.zeros((0, 2), dtype=int),
            []
        ),
        (
            {10: [[0]], 2: [[1]]},
            2,
            [[0, 1], [1, 0]],
            [2, 10]
        ),
    ]
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
