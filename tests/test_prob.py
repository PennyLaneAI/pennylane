# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

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
Unit tests for the PennyLane qml.probs() measurement function.
"""
import pytest

import numpy as np
import pennylane as qml


# make the test deterministic
np.random.seed(42)


@pytest.fixture
def init_state(scope="session"):
    """Fixture that creates an initial state"""

    def _init_state(n):
        """An initial state over n wires"""
        state = np.random.random([2 ** n]) + np.random.random([2 ** n]) * 1j
        state /= np.linalg.norm(state)
        return state

    return _init_state


def test_full_prob(init_state, tol):
    """Test that the correct probability is returned."""
    dev = qml.device("default.qubit", wires=4)
    state = init_state(4)

    @qml.qnode(dev)
    def circuit():
        qml.QubitStateVector(state, wires=list(range(4)))
        return qml.probs(wires=range(4))

    res = circuit()
    expected = np.abs(state) ** 2
    assert np.allclose(res, expected, atol=tol, rtol=0)


def test_marginal_prob(init_state, tol):
    """Test that the correct marginal probability is returned."""
    dev = qml.device("default.qubit", wires=4)
    state = init_state(4)

    @qml.qnode(dev)
    def circuit():
        qml.QubitStateVector(state, wires=list(range(4)))
        return qml.probs(wires=[1, 3])

    res = circuit()
    expected = np.reshape(np.abs(state) ** 2, [2] * 4)
    expected = np.einsum("ijkl->jl", expected).flatten()
    assert np.allclose(res, expected, atol=tol, rtol=0)


def test_marginal_prob_more_wires(init_state, mocker, tol):
    """Test that the correct marginal probability is returned, when the
    states_to_binary method is used for probability computations."""
    dev = qml.device("default.qubit", wires=4)
    state = init_state(4)

    spy = mocker.spy(qml.QubitDevice, "states_to_binary")

    @qml.qnode(dev)
    def circuit():
        qml.QubitStateVector(state, wires=list(range(4)))
        return qml.probs(wires=[1, 0, 3])  # <--- more than 2 wires: states_to_binary used

    res = circuit()

    expected = np.reshape(np.abs(state) ** 2, [2] * 4)
    expected = np.einsum("ijkl->jil", expected).flatten()
    assert np.allclose(res, expected, atol=tol, rtol=0)

    spy.assert_called_once()


def test_integration(tol):
    """Test the probability is correct for a known state preparation."""
    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def circuit():
        qml.Hadamard(wires=1)
        qml.CNOT(wires=[0, 1])
        return qml.probs(wires=[0, 1])

    # expected probability, using [00, 01, 10, 11]
    # ordering, is [0.5, 0.5, 0, 0]

    res = circuit()
    expected = np.array([0.5, 0.5, 0, 0])
    assert np.allclose(res, expected, atol=tol, rtol=0)


def test_integration_analytic_false(tol):
    """Test the probability is correct for a known state preparation when the
    analytic attribute is set to False."""
    dev = qml.device("default.qubit", wires=3, shots=1000)

    @qml.qnode(dev)
    def circuit():
        qml.PauliX(0)
        return qml.probs(wires=[0])

    res = circuit()
    expected = np.array([0, 1])
    assert np.allclose(res, expected, atol=tol, rtol=0)


def test_numerical_analytic_diff_agree(init_state, tol):
    """Test that the finite difference and parameter shift rule
    provide the same Jacobian."""
    w = 4
    dev = qml.device("default.qubit", wires=w)
    state = init_state(w)

    def circuit(x, y, z):
        for i in range(w):
            qml.RX(x, wires=i)
            qml.PhaseShift(z, wires=i)
            qml.RY(y, wires=i)

        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 3])

        return qml.probs(wires=[1, 3])

    params = [0.543, -0.765, -0.3]

    circuit_F = qml.QNode(circuit, dev, diff_method="finite-diff")
    circuit_A = qml.QNode(circuit, dev, diff_method="parameter-shift")
    res_F = qml.jacobian(circuit_F)(*params)
    res_A = qml.jacobian(circuit_A)(*params)

    # Both jacobians should be of shape (2**prob.wires, num_params)
    assert res_F.shape == (2 ** 2, 3)
    assert res_F.shape == (2 ** 2, 3)

    # Check that they agree up to numeric tolerance
    assert np.allclose(res_F, res_A, atol=tol, rtol=0)
