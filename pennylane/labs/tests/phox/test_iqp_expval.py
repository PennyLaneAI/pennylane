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
    from pennylane.labs.phox.simulator_pure_functions import _iqp_expval_core
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
            qml.MultiRZ(2 * param, wires=gen)

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
        n_qubits = len(obs_strings)

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
