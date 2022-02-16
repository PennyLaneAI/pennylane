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
from pennylane import numpy as pnp


# make the test deterministic
np.random.seed(42)


@pytest.fixture
def init_state(scope="session"):
    """Fixture that creates an initial state"""

    def _init_state(n):
        """An initial state over n wires"""
        state = np.random.random([2**n]) + np.random.random([2**n]) * 1j
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

    params = pnp.array([0.543, -0.765, -0.3], requires_grad=True)

    circuit_F = qml.QNode(circuit, dev, diff_method="finite-diff")
    circuit_A = qml.QNode(circuit, dev, diff_method="parameter-shift")
    res_F = qml.jacobian(circuit_F)(*params)
    res_A = qml.jacobian(circuit_A)(*params)

    # Both jacobians should be of shape (2**prob.wires, num_params)
    assert isinstance(res_F, tuple) and len(res_F) == 3
    assert all(_r.shape == (2**2,) for _r in res_F)
    assert isinstance(res_A, tuple) and len(res_A) == 3
    assert all(_r.shape == (2**2,) for _r in res_A)

    # Check that they agree up to numeric tolerance
    assert all(np.allclose(_rF, _rA, atol=tol, rtol=0) for _rF, _rA in zip(res_F, res_A))


@pytest.mark.parametrize("hermitian", [1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])])
def test_prob_generalize_param_one_qubit(hermitian, init_state, tol):
    """Test that the correct probability is returned."""
    dev = qml.device("default.qubit", wires=1)

    @qml.qnode(dev)
    def circuit(x):
        qml.RZ(x, wires=0)
        return qml.probs(op=qml.Hermitian(hermitian, wires=0))

    res = circuit(0.56)

    def circuit_rotated(x):
        qml.RZ(x, wires=0)
        qml.Hermitian(hermitian, wires=0).diagonalizing_gates()

    state = np.array([1, 0])
    matrix = qml.transforms.get_unitary_matrix(circuit_rotated)(0.56)
    state = np.dot(matrix, state)
    expected = np.reshape(np.abs(state) ** 2, [2] * 1)
    expected = expected.flatten()

    assert np.allclose(res, expected, atol=tol, rtol=0)


@pytest.mark.parametrize("hermitian", [1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])])
def test_prob_generalize_param(hermitian, init_state, tol):
    """Test that the correct probability is returned."""
    dev = qml.device("default.qubit", wires=3)

    @qml.qnode(dev)
    def circuit(x, y):
        qml.RZ(x, wires=0)
        qml.CNOT(wires=[0, 1])
        qml.RY(y, wires=1)
        qml.CNOT(wires=[0, 2])
        return qml.probs(op=qml.Hermitian(hermitian, wires=0))

    res = circuit(0.56, 0.1)

    def circuit_rotated(x, y):
        qml.RZ(x, wires=0)
        qml.CNOT(wires=[0, 1])
        qml.RY(y, wires=1)
        qml.CNOT(wires=[0, 2])
        qml.Hermitian(hermitian, wires=0).diagonalizing_gates()

    state = np.array([1, 0, 0, 0, 0, 0, 0, 0])
    matrix = qml.transforms.get_unitary_matrix(circuit_rotated)(0.56, 0.1)
    state = np.dot(matrix, state)
    expected = np.reshape(np.abs(state) ** 2, [2] * 3)
    expected = np.einsum("ijk->i", expected).flatten()
    assert np.allclose(res, expected, atol=tol, rtol=0)


@pytest.mark.parametrize("hermitian", [1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])])
def test_prob_generalize_param_multiple(hermitian, init_state, tol):
    """Test that the correct probability is returned."""
    dev = qml.device("default.qubit", wires=3)

    @qml.qnode(dev)
    def circuit(x, y):
        qml.RZ(x, wires=0)
        qml.CNOT(wires=[0, 1])
        qml.RY(y, wires=1)
        qml.CNOT(wires=[0, 2])
        return (
            qml.probs(op=qml.Hermitian(hermitian, wires=0)),
            qml.probs(wires=[1]),
            qml.probs(wires=[2]),
        )

    res = circuit(0.56, 0.1)
    res = np.reshape(res, (3, 2))

    def circuit_rotated(x, y):
        qml.RZ(x, wires=0)
        qml.CNOT(wires=[0, 1])
        qml.RY(y, wires=1)
        qml.CNOT(wires=[0, 2])
        qml.Hermitian(hermitian, wires=0).diagonalizing_gates()

    state = np.array([1, 0, 0, 0, 0, 0, 0, 0])
    matrix = qml.transforms.get_unitary_matrix(circuit_rotated)(0.56, 0.1)
    state = np.dot(matrix, state)

    expected = np.reshape(np.abs(state) ** 2, [2] * 3)
    expected_0 = np.einsum("ijk->i", expected).flatten()
    expected_1 = np.einsum("ijk->j", expected).flatten()
    expected_2 = np.einsum("ijk->k", expected).flatten()

    assert np.allclose(res[0], expected_0, atol=tol, rtol=0)
    assert np.allclose(res[1], expected_1, atol=tol, rtol=0)
    assert np.allclose(res[2], expected_2, atol=tol, rtol=0)


@pytest.mark.parametrize("hermitian", [1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])])
@pytest.mark.parametrize("wire", [0, 1, 2, 3])
def test_prob_generalize_initial_state(hermitian, wire, init_state, tol):
    """Test that the correct probability is returned."""
    dev = qml.device("default.qubit", wires=4)
    state = init_state(4)

    @qml.qnode(dev)
    def circuit():
        qml.QubitStateVector(state, wires=list(range(4)))
        qml.PauliX(wires=0)
        qml.PauliX(wires=1)
        qml.PauliX(wires=2)
        qml.PauliX(wires=3)
        return qml.probs(op=qml.Hermitian(hermitian, wires=wire))

    res = circuit()

    def circuit_rotated():
        qml.PauliX(wires=0)
        qml.PauliX(wires=1)
        qml.PauliX(wires=2)
        qml.PauliX(wires=3)
        qml.Hermitian(hermitian, wires=wire).diagonalizing_gates()

    matrix = qml.transforms.get_unitary_matrix(circuit_rotated)()
    state = np.dot(matrix, state)
    expected = np.reshape(np.abs(state) ** 2, [2] * 4)

    if wire == 0:
        expected = np.einsum("ijkl->i", expected).flatten()
    elif wire == 1:
        expected = np.einsum("ijkl->j", expected).flatten()
    elif wire == 2:
        expected = np.einsum("ijkl->k", expected).flatten()
    elif wire == 3:
        expected = np.einsum("ijkl->l", expected).flatten()

    assert np.allclose(res, expected, atol=tol, rtol=0)


@pytest.mark.parametrize("operation", [qml.PauliX, qml.PauliY, qml.Hadamard])
@pytest.mark.parametrize("wire", [0, 1, 2, 3])
def test_operation_prob(operation, wire, init_state, tol):
    "Test the rotated probability with different wires and rotating operations."
    dev = qml.device("default.qubit", wires=4)
    state = init_state(4)

    @qml.qnode(dev)
    def circuit():
        qml.QubitStateVector(state, wires=list(range(4)))
        qml.PauliX(wires=0)
        qml.PauliZ(wires=1)
        qml.PauliY(wires=2)
        qml.PauliZ(wires=3)
        return qml.probs(op=operation(wires=wire))

    res = circuit()

    def circuit_rotated():
        qml.PauliX(wires=0)
        qml.PauliZ(wires=1)
        qml.PauliY(wires=2)
        qml.PauliZ(wires=3)
        operation(wires=wire).diagonalizing_gates()

    matrix = qml.transforms.get_unitary_matrix(circuit_rotated)()
    state = np.dot(matrix, state)
    expected = np.reshape(np.abs(state) ** 2, [2] * 4)

    if wire == 0:
        expected = np.einsum("ijkl->i", expected).flatten()
    elif wire == 1:
        expected = np.einsum("ijkl->j", expected).flatten()
    elif wire == 2:
        expected = np.einsum("ijkl->k", expected).flatten()
    elif wire == 3:
        expected = np.einsum("ijkl->l", expected).flatten()

    assert np.allclose(res, expected, atol=tol, rtol=0)


@pytest.mark.parametrize("operation", [qml.PauliX, qml.PauliY, qml.Hadamard])
@pytest.mark.parametrize("wire", [0, 1, 2, 3])
def test_operation_prob(operation, wire, init_state, tol):
    "Test the rotated probability with different wires and rotating operations."
    dev = qml.device("default.qubit", wires=4)
    state = init_state(4)

    @qml.qnode(dev)
    def circuit():
        qml.QubitStateVector(state, wires=list(range(4)))
        qml.PauliX(wires=0)
        qml.PauliZ(wires=1)
        qml.PauliY(wires=2)
        qml.PauliZ(wires=3)
        return qml.probs(op=operation(wires=wire))

    res = circuit()

    def circuit_rotated():
        qml.PauliX(wires=0)
        qml.PauliZ(wires=1)
        qml.PauliY(wires=2)
        qml.PauliZ(wires=3)
        operation(wires=wire).diagonalizing_gates()

    matrix = qml.transforms.get_unitary_matrix(circuit_rotated)()
    state = np.dot(matrix, state)
    expected = np.reshape(np.abs(state) ** 2, [2] * 4)

    if wire == 0:
        expected = np.einsum("ijkl->i", expected).flatten()
    elif wire == 1:
        expected = np.einsum("ijkl->j", expected).flatten()
    elif wire == 2:
        expected = np.einsum("ijkl->k", expected).flatten()
    elif wire == 3:
        expected = np.einsum("ijkl->l", expected).flatten()

    assert np.allclose(res, expected, atol=tol, rtol=0)


@pytest.mark.parametrize("observable", [(qml.PauliX, qml.PauliY)])
def test_observable_tensor_prob(observable, init_state, tol):
    "Test the rotated probability with a tensor observable."
    dev = qml.device("default.qubit", wires=4)
    state = init_state(4)

    @qml.qnode(dev)
    def circuit():
        qml.QubitStateVector(state, wires=list(range(4)))
        qml.PauliX(wires=0)
        qml.PauliZ(wires=1)
        qml.PauliY(wires=2)
        qml.PauliZ(wires=3)
        return qml.probs(op=observable[0](wires=0) @ observable[1](wires=1))

    res = circuit()

    def circuit_rotated():
        qml.PauliX(wires=0)
        qml.PauliZ(wires=1)
        qml.PauliY(wires=2)
        qml.PauliZ(wires=3)
        observable[0](wires=0).diagonalizing_gates()
        observable[1](wires=1).diagonalizing_gates()

    matrix = qml.transforms.get_unitary_matrix(circuit_rotated)()
    state = np.dot(matrix, state)
    expected = np.reshape(np.abs(state) ** 2, [2] * 4)

    expected = np.einsum("ijkl->ij", expected).flatten()

    assert np.allclose(res, expected, atol=tol, rtol=0)


@pytest.mark.parametrize("coeffs, obs", [([1, 1], [qml.PauliX(wires=0), qml.PauliX(wires=1)])])
def test_hamiltonian_error(coeffs, obs, init_state, tol):
    "Test that an error is returned for hamiltonians."
    H = qml.Hamiltonian(coeffs, obs)

    dev = qml.device("default.qubit", wires=4)
    state = init_state(4)

    @qml.qnode(dev)
    def circuit():
        qml.QubitStateVector(state, wires=list(range(4)))
        qml.PauliX(wires=0)
        qml.PauliZ(wires=1)
        qml.PauliY(wires=2)
        qml.PauliZ(wires=3)
        return qml.probs(op=H)

    with pytest.raises(
        qml.QuantumFunctionError,
        match="Hamiltonians are not supported for rotating probabilities.",
    ):
        circuit()


@pytest.mark.parametrize(
    "operation", [qml.SingleExcitation, qml.SingleExcitationPlus, qml.SingleExcitationMinus]
)
def test_generalize_prob_not_hermitian(operation):
    """Test that Operators that do not have a diagonalizing_gates representation cannot
    be used in probability measurements."""

    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def circuit():
        qml.PauliX(wires=0)
        qml.PauliZ(wires=1)
        return qml.probs(op=operation(0.56, wires=[0, 1]))

    with pytest.raises(
        qml.QuantumFunctionError,
        match="does not define diagonalizing gates : cannot be used to rotate the probability",
    ):
        circuit()


@pytest.mark.parametrize("hermitian", [1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])])
def test_prob_wires_and_hermitian(hermitian):
    """Test that we can cannot give simultaneously wires and a hermitian."""

    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def circuit():
        qml.PauliX(wires=0)
        return qml.probs(op=qml.Hermitian(hermitian, wires=0), wires=1)

    with pytest.raises(
        qml.QuantumFunctionError,
        match="Cannot specify the wires to probs if an observable is "
        "provided. The wires for probs will be determined directly from the observable.",
    ):
        circuit()
