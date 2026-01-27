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
    from pennylane.labs.phox.simulator_pure_functions import _iqp_expval_core, _parse_iqp_dict
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

    def test_iqp_expval_vs_pennylane(self):
        """Test that _iqp_expval_core matches PennyLane default.qubit."""
        generators_pl = [[0], [1], [0, 1, 2]]
        params = [0.37454012, 0.95071431, 0.73199394]
        obs_strings = ["X", "Z", "Y"]

        state = [1] + [0] * 7

        circuit = iqp_circuit_pl(generators_pl, params, obs_strings, state)
        exact_val = circuit()

        generators = jnp.array([[1, 0, 0], [0, 1, 0], [1, 1, 1]])
        params_jax = jnp.array(params)
        obs_jax = [obs_strings]
        key = jax.random.PRNGKey(42)
        n_samples = 100000

        approx_val, _ = _iqp_expval_core(generators, params_jax, obs_jax, n_samples, key)

        assert np.allclose(exact_val, approx_val, atol=0.02)

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
    n_qubits = 2  # 5 is out of bounds for 2 qubits

    with pytest.raises(IndexError):
        _parse_iqp_dict(circuit_def, n_qubits)
