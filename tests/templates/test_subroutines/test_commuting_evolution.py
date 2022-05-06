# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

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
Tests for the CommutingEvolution template.
"""
import pytest
from pennylane import numpy as np
import pennylane as qml

from scipy.linalg import expm


def test_adjoint():
    """Tests the CommutingEvolution.adjoint method provides the correct adjoint operation."""

    n_wires = 2
    dev1 = qml.device("default.qubit", wires=n_wires)
    dev2 = qml.device("default.qubit", wires=n_wires)

    obs = [qml.PauliX(0) @ qml.PauliY(1), qml.PauliY(0) @ qml.PauliX(1)]
    coeffs = [1, -1]
    hamiltonian = qml.Hamiltonian(coeffs, obs)
    frequencies = (2,)

    @qml.qnode(dev1)
    def adjoint_evolution_circuit(time):
        for i in range(n_wires):
            qml.Hadamard(i)
        qml.adjoint(qml.CommutingEvolution)(hamiltonian, time, frequencies)
        return qml.expval(qml.PauliZ(1))

    @qml.qnode(dev2)
    def evolution_circuit(time):
        for i in range(n_wires):
            qml.Hadamard(i)
        qml.CommutingEvolution(hamiltonian, time, frequencies)
        return qml.expval(qml.PauliZ(1))

    evolution_circuit(0.13)
    adjoint_evolution_circuit(-0.13)

    assert all(np.isclose(dev1.state, dev2.state))


def test_decomposition_expand():
    """Test that the decomposition of CommutingEvolution is an ApproxTimeEvolution with one step."""

    hamiltonian = 0.5 * qml.PauliX(0) @ qml.PauliY(1)
    time = 2.345

    op = qml.CommutingEvolution(hamiltonian, time)

    decomp = op.decomposition()

    assert isinstance(decomp, qml.ApproxTimeEvolution)
    assert all(decomp.hyperparameters["hamiltonian"].coeffs == hamiltonian.coeffs)
    assert decomp.hyperparameters["n"] == 1

    tape = op.expand()
    assert len(tape) == 1
    assert isinstance(tape[0], qml.ApproxTimeEvolution)


def test_matrix():
    """Test that the matrix of commuting evolution is the same as exponentiating -1j * t the hamiltonian."""

    h = 2.34 * qml.PauliX(0)
    time = 0.234
    op = qml.CommutingEvolution(h, time)

    mat = qml.matrix(op)

    expected = expm(-1j * time * qml.matrix(h))

    assert qml.math.allclose(mat, expected)


def test_forward_execution():
    """Compare the foward execution to an exactly known result."""
    dev = qml.device("default.qubit", wires=2)

    H = qml.PauliX(0) @ qml.PauliY(1) - 1.0 * qml.PauliY(0) @ qml.PauliX(1)
    freq = (2, 4)

    @qml.qnode(dev, diff_method=None)
    def circuit(time):
        qml.PauliX(0)
        qml.CommutingEvolution(H, time, freq)
        return qml.expval(qml.PauliZ(0))

    t = 1.0
    res = circuit(t)
    expected = -np.cos(4)
    assert np.allclose(res, expected)


class TestInputs:
    """Tests for input validation of `CommutingEvolution`."""

    def test_invalid_hamiltonian(self):
        """Tests TypeError is raised if `hamiltonian` is not type `qml.Hamiltonian`."""

        invalid_operator = qml.PauliX(0)

        assert pytest.raises(TypeError, qml.CommutingEvolution, invalid_operator, 1)


class TestGradients:
    """Tests that correct gradients are obtained for `CommutingEvolution` when frequencies
    are specified."""

    def test_two_term_case(self):
        """Tests the parameter shift rules for `CommutingEvolution` equal the
        finite difference result for a two term shift rule case."""

        n_wires = 1
        dev = qml.device("default.qubit", wires=n_wires)

        hamiltonian = qml.Hamiltonian([1], [qml.PauliX(0)])
        frequencies = (2,)

        @qml.qnode(dev)
        def circuit(time):
            qml.PauliX(0)
            qml.CommutingEvolution(hamiltonian, time, frequencies)
            return qml.expval(qml.PauliZ(0))

        x_vals = np.linspace(-np.pi, np.pi, num=10)

        grads_finite_diff = [qml.gradients.finite_diff(circuit)(x) for x in x_vals]
        grads_param_shift = [qml.gradients.param_shift(circuit)(x) for x in x_vals]

        assert all(np.isclose(grads_finite_diff, grads_param_shift, atol=1e-4))

    def test_four_term_case(self):
        """Tests the parameter shift rules for `CommutingEvolution` equal the
        finite difference result for a four term shift rule case."""

        n_wires = 2
        dev = qml.device("default.qubit", wires=n_wires)

        coeffs = [1, -1]
        obs = [qml.PauliX(0) @ qml.PauliY(1), qml.PauliY(0) @ qml.PauliX(1)]
        hamiltonian = qml.Hamiltonian(coeffs, obs)
        frequencies = (2, 4)

        @qml.qnode(dev)
        def circuit(time):
            qml.PauliX(0)
            qml.CommutingEvolution(hamiltonian, time, frequencies)
            return qml.expval(qml.PauliZ(0))

        x_vals = [np.array(x, requires_grad=True) for x in np.linspace(-np.pi, np.pi, num=10)]

        grads_finite_diff = [qml.gradients.finite_diff(circuit)(x) for x in x_vals]
        grads_param_shift = [qml.gradients.param_shift(circuit)(x) for x in x_vals]

        assert all(np.isclose(grads_finite_diff, grads_param_shift, atol=1e-4))

    def test_differentiable_hamiltonian(self):
        """Tests correct gradients are produced when the Hamiltonian is differentiable."""

        n_wires = 2
        dev = qml.device("default.qubit", wires=n_wires)
        obs = [qml.PauliX(0) @ qml.PauliY(1), qml.PauliY(0) @ qml.PauliX(1)]
        diff_coeffs = np.array([1.0, -1.0], requires_grad=True)
        frequencies = (2, 4)

        def parameterized_hamiltonian(coeffs):
            return qml.Hamiltonian(coeffs, obs)

        @qml.qnode(dev)
        def circuit(time, coeffs):
            qml.PauliX(0)
            qml.CommutingEvolution(parameterized_hamiltonian(coeffs), time, frequencies)
            return qml.expval(qml.PauliZ(0))

        x_vals = [np.array(x, requires_grad=True) for x in np.linspace(-np.pi, np.pi, num=10)]

        grads_finite_diff = [
            np.hstack(qml.gradients.finite_diff(circuit)(x, diff_coeffs)) for x in x_vals
        ]
        grads_param_shift = [
            np.hstack(qml.gradients.param_shift(circuit)(x, diff_coeffs)) for x in x_vals
        ]

        assert np.isclose(grads_finite_diff, grads_param_shift, atol=1e-6).all()
