# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for measuring states in devices/qubit_mixed."""
# pylint: disable=too-few-public-methods

from functools import reduce

import numpy as np
import pytest

import pennylane as qml
from pennylane import math
from pennylane.devices.qubit_mixed import apply_operation, create_initial_state, measure
from pennylane.devices.qubit_mixed.measure import (
    csr_dot_products_density_matrix,
    full_dot_products_density_matrix,
    get_measurement_function,
    state_diagonalizing_gates,
    sum_of_terms_method,
)

ml_frameworks_list = [
    "numpy",
    pytest.param("autograd", marks=pytest.mark.autograd),
    pytest.param("jax", marks=pytest.mark.jax),
    pytest.param("torch", marks=pytest.mark.torch),
    pytest.param("tensorflow", marks=pytest.mark.tf),
]
BATCH_SIZE = 2


def get_expanded_op_mult_state(op, state):
    """Finds the expanded matrix to multiply state by and multiplies it by a flattened state"""
    num_qubits = int(len(state.shape) / 2)
    pre_wires_identity = np.eye(2 ** min(op.wires))
    post_wires_identity = np.eye(2 ** ((num_qubits - 1) - op.wires[-1]))

    expanded_op = reduce(np.kron, (pre_wires_identity, op.matrix(), post_wires_identity))
    flattened_state = state.reshape((2**num_qubits,) * 2)
    return expanded_op @ flattened_state


def get_expval(op, state):
    """Finds op@state and traces to find the expectation value of observable on the state"""
    op_mult_state = get_expanded_op_mult_state(op, state)
    return np.trace(op_mult_state)


@pytest.mark.parametrize(
    "mp", [qml.sample(), qml.counts(), qml.sample(wires=0), qml.counts(wires=0)]
)
class TestCurrentlyUnsupportedCases:
    # pylint: disable=too-few-public-methods
    def test_sample_based_observable(self, mp, two_qubit_state):
        """Test sample-only measurements raise a NotImplementedError."""
        with pytest.raises(NotImplementedError):
            _ = measure(mp, two_qubit_state)


@pytest.mark.unit
class TestMeasurementDispatch:
    """Test that get_measurement_function dispatchs to the correct place."""

    def test_state_no_obs(self):
        """Test that the correct internal function is used for a measurement process with no observables."""
        # Test a case where state_measurement_process is used
        mp1 = qml.state()
        assert get_measurement_function(mp1, state=1) == state_diagonalizing_gates

    @pytest.mark.parametrize(
        "m",
        (
            qml.var(qml.PauliZ(0)),
            qml.expval(qml.sum(qml.PauliZ(0), qml.PauliX(0))),
            qml.expval(qml.sum(*(qml.PauliX(i) for i in range(15)))),
            qml.expval(qml.prod(qml.PauliX(0), qml.PauliY(1), qml.PauliZ(10))),
        ),
    )
    def test_diagonalizing_gates(self, m):
        """Test that the state_diagonalizing gates are used when there's an observable has diagonalizing
        gates and allows the measurement to be efficiently computed with them."""
        assert get_measurement_function(m, state=1) is state_diagonalizing_gates

    def test_hermitian_full_dot_product(self):
        """Test that the expectation value of a hermitian uses the full dot products method."""
        mp = qml.expval(qml.Hermitian(np.eye(2), wires=0))
        assert get_measurement_function(mp, state=1) is full_dot_products_density_matrix

    def test_hamiltonian_sparse_method(self):
        """Check that the sum_of_terms_method method is used if the state is numpy."""
        H = qml.Hamiltonian([2], [qml.PauliX(0)])
        state = np.zeros(2)
        assert get_measurement_function(qml.expval(H), state) is csr_dot_products_density_matrix

    def test_hamiltonian_sum_of_terms_when_backprop(self):
        """Check that the sum of terms method is used when the state is trainable."""
        H = qml.Hamiltonian([2], [qml.PauliX(0)])
        state = qml.numpy.zeros(2)
        assert get_measurement_function(qml.expval(H), state) is sum_of_terms_method

    def test_sum_sparse_method_when_large_and_nonoverlapping(self):
        """Check that the sum_of_terms_method is used if the state is numpy and
        the Sum is large with overlapping wires."""
        S = qml.prod(*(qml.PauliX(i) for i in range(8))) + qml.prod(
            *(qml.PauliY(i) for i in range(8))
        )
        state = np.zeros(2)
        assert get_measurement_function(qml.expval(S), state) is csr_dot_products_density_matrix

    def test_sum_sum_of_terms_when_backprop(self):
        """Check that the sum of terms method is used when"""
        S = qml.prod(*(qml.PauliX(i) for i in range(8))) + qml.prod(
            *(qml.PauliY(i) for i in range(8))
        )
        state = qml.numpy.zeros(2)
        assert get_measurement_function(qml.expval(S), state) is sum_of_terms_method

    def test_sparse_method_for_density_matrix(self):
        """Check that csr_dot_products_density_matrix is used for sparse measurements on density matrices"""
        # Create a sparse observable
        H = qml.SparseHamiltonian(
            qml.Hamiltonian([1.0], [qml.PauliZ(0)]).sparse_matrix(), wires=[0]
        )
        state = np.zeros((2, 2))  # 2x2 density matrix

        # Verify the correct measurement function is selected
        assert get_measurement_function(qml.expval(H), state) is csr_dot_products_density_matrix

        # Also test with a larger system
        H_large = qml.SparseHamiltonian(
            qml.Hamiltonian([1.0], [qml.PauliZ(0) @ qml.PauliX(1)]).sparse_matrix(), wires=[0, 1]
        )
        state_large = np.zeros((4, 4))  # 4x4 density matrix for 2 qubits

        assert (
            get_measurement_function(qml.expval(H_large), state_large)
            is csr_dot_products_density_matrix
        )

    def test_no_sparse_matrix(self):
        """Tests Hamiltonians/Sums containing observables that do not have a sparse matrix."""

        class DummyOp(qml.operation.Operator):  # pylint: disable=too-few-public-methods
            num_wires = 1

        S1 = qml.Hamiltonian([0.5, 0.5], [qml.X(0), DummyOp(wires=1)])
        state = np.zeros(2)
        assert get_measurement_function(qml.expval(S1), state) is sum_of_terms_method

        S2 = qml.X(0) + DummyOp(wires=1)
        assert get_measurement_function(qml.expval(S2), state) is sum_of_terms_method

        S3 = 0.5 * qml.X(0) + 0.5 * DummyOp(wires=1)
        assert get_measurement_function(qml.expval(S3), state) is sum_of_terms_method

        S4 = qml.Y(0) + qml.X(0) @ DummyOp(wires=1)
        assert get_measurement_function(qml.expval(S4), state) is sum_of_terms_method

    def test_hamiltonian_no_sparse_matrix_in_second_term(self):
        """Tests when not all terms of a Hamiltonian have sparse matrices, excluding the first term."""

        class DummyOp(qml.operation.Operator):  # Custom observable with no sparse matrix
            num_wires = 1

        H = qml.Hamiltonian([0.5, 0.5, 0.5], [qml.PauliX(0), DummyOp(wires=1), qml.PauliZ(2)])
        state = np.zeros(2)
        assert get_measurement_function(qml.expval(H), state) is sum_of_terms_method

    def test_sum_no_sparse_matrix(self):
        """Tests when not all terms in a Sum observable have sparse matrices."""

        class DummyOp(qml.operation.Operator):  # Custom observable with no sparse matrix
            num_wires = 1

        S = qml.sum(qml.PauliX(0), DummyOp(wires=1))
        state = np.zeros(2)
        assert get_measurement_function(qml.expval(S), state) is sum_of_terms_method

    def test_has_overlapping_wires(self):
        """Test that the has_overlapping_wires property correctly detects overlapping wires."""

        # Define some operators with overlapping and non-overlapping wires
        op1 = qml.PauliX(wires=0)
        op2 = qml.PauliZ(wires=1)
        op3 = qml.PauliY(wires=0)  # Overlaps with op1
        op4 = qml.PauliX(wires=2)  # No overlap
        op5 = qml.MultiControlledX(wires=range(8))

        # Create Prod operators with and without overlapping wires
        prod_with_overlap = op1 @ op3
        prod_without_overlap = op1 @ op2 @ op4

        # Assert that overlapping wires are correctly detected
        assert (
            prod_with_overlap.has_overlapping_wires is True
        ), "Expected overlapping wires to be detected."
        assert (
            prod_without_overlap.has_overlapping_wires is False
        ), "Expected no overlapping wires to be detected."
        # Create a Sum observable that involves the operators
        sum_obs = qml.sum(op1, op2, op3, op4, op5)  # 5 terms
        assert sum_obs.has_overlapping_wires is True, "Expected overlapping wires to be detected."

        # Create the measurement process
        measurementprocess = qml.expval(op=sum_obs)

        # Create a mock state (you would normally use a real state here)
        dim = 2**8
        state = np.diag([1 / dim] * dim)  # Example state, length of 16 for the test

        # Check if we hit the tensor contraction branch
        result = get_measurement_function(measurementprocess, state)

        # Verify the correct function is returned (csr_dot_products_density_matrix)
        assert (
            result == csr_dot_products_density_matrix
        ), "Expected csr_dot_products_density_matrix method"


class TestMeasurements:
    """Test that measurements on unbatched states work as expected."""

    @pytest.mark.parametrize(
        "measurement, get_expected",
        [
            (qml.density_matrix(wires=0), lambda x: math.trace(x, axis1=1, axis2=3)),
            (
                qml.probs(wires=[0]),
                lambda x: math.real(math.diag(math.trace(x, axis1=1, axis2=3))),
            ),
            (
                qml.probs(),
                lambda x: math.real(math.diag(x.reshape((4, 4)))),
            ),
        ],
    )
    def test_state_measurement_no_obs(self, measurement, get_expected, two_qubit_state):
        """Test that state measurements with no observable work as expected."""
        res = measure(measurement, two_qubit_state)
        expected = get_expected(two_qubit_state)

        assert np.allclose(res, expected)

    @pytest.mark.parametrize(
        "coeffs, observables",
        [
            ([-0.5, 2], [qml.PauliX(0), qml.PauliX(1)]),
            ([-0.3, 1], [qml.PauliY(0), qml.PauliX(1)]),
            ([-0.45, 2.6], [qml.PauliZ(1), qml.PauliX(0)]),
        ],
    )
    def test_hamiltonian_expval(self, coeffs, observables, two_qubit_state):
        """Test that measurements of hamiltonian work correctly."""

        obs = qml.Hamiltonian(coeffs, observables)
        res = measure(qml.expval(obs), two_qubit_state)

        expected = 0
        for i, coeff in enumerate(coeffs):
            expected += coeff * get_expval(observables[i], two_qubit_state)

        assert np.isclose(res, expected)

    @pytest.mark.parametrize(
        "observable",
        [
            qml.Hermitian(-0.5 * qml.PauliX(7).matrix() + 2 * qml.PauliX(8).matrix(), wires=0),
            qml.Hermitian(-0.55 * qml.PauliX(4).matrix() + 2.4 * qml.PauliX(5).matrix(), wires=1),
        ],
    )
    def test_hermitian_expval(self, observable, two_qubit_state):
        """Test that measurements of qubit hermitian work correctly."""
        res = measure(qml.expval(observable), two_qubit_state)
        expected = get_expval(observable, two_qubit_state)

        assert np.isclose(res, expected)

    @pytest.mark.parametrize(
        "observable",
        [
            qml.PauliX(0),
            qml.PauliX(1),
            (qml.PauliX(0) @ qml.PauliX(1)),
        ],
    )
    def test_variance_measurement(self, observable, two_qubit_state):
        """Test that variance measurements work as expected."""
        res = measure(qml.var(observable), two_qubit_state)

        expval_obs = get_expval(observable, two_qubit_state)

        obs_squared = qml.prod(observable, observable)
        expval_of_squared_obs = get_expval(obs_squared, two_qubit_state)

        expected = expval_of_squared_obs - expval_obs**2
        assert np.allclose(res, expected)


@pytest.mark.parametrize(
    "obs, coeffs, matrices",
    [
        (
            qml.Hamiltonian([-0.5, 2], [qml.PauliX(0), qml.PauliX(1)]),
            [-0.5, 2],
            [
                np.kron(qml.PauliX(0).matrix(), np.eye(2)),
                np.kron(np.eye(2), qml.PauliX(1).matrix()),
            ],
        ),
        (
            qml.Hermitian(
                -0.5 * np.kron(qml.PauliX(0).matrix(), np.eye(2))
                + 2 * np.kron(np.eye(2), qml.PauliX(1).matrix()),
                wires=[0, 1],
            ),
            [-0.5, 2],
            [
                np.kron(qml.PauliX(0).matrix(), np.eye(2)),
                np.kron(np.eye(2), qml.PauliX(1).matrix()),
            ],
        ),
    ],
)
class TestExpValAnalytical:
    @staticmethod
    def prepare_pure_state(theta):
        """Helper function to prepare a two-qubit pure state density matrix."""
        qubit0 = np.array([np.cos(theta), -1j * np.sin(theta)])
        qubit1 = np.array([1, 0])  # Second qubit in |0⟩ state
        state_vector = np.kron(qubit0, qubit1)  # Shape: (4,)
        state = np.outer(state_vector, np.conj(state_vector))  # Shape: (4, 4)
        return state

    @staticmethod
    def prepare_mixed_state(theta, weights):
        """Helper function to prepare a two-qubit mixed state density matrix."""
        qubit1 = np.array([1, 0])  # Second qubit in |0⟩ state

        qubit0_one = np.array([np.cos(theta), -1j * np.sin(theta)])
        qubit0_two = np.array([np.cos(theta), 1j * np.sin(theta)])

        state_vector_one = np.kron(qubit0_one, qubit1)
        state_vector_two = np.kron(qubit0_two, qubit1)

        state_one = np.outer(state_vector_one, np.conj(state_vector_one))
        state_two = np.outer(state_vector_two, np.conj(state_vector_two))

        return weights[0] * state_one + weights[1] * state_two

    @staticmethod
    def compute_expected_value(obs, state):
        """Helper function to compute the analytical expectation value."""
        if isinstance(obs, qml.Hamiltonian):
            matrices = [
                np.kron(qml.PauliX(0).matrix(), np.eye(2)),
                np.kron(np.eye(2), qml.PauliX(1).matrix()),
            ]
            hamiltonian_matrix = sum(c * m for c, m in zip(obs.coeffs, matrices))
            obs_matrix = hamiltonian_matrix
        else:
            obs_matrix = obs.matrix()
        return np.trace(state @ obs_matrix)

    def test_expval_pure_state(self, obs, coeffs, matrices):
        theta = 0.123
        state = self.prepare_pure_state(theta)

        if isinstance(obs, qml.Hamiltonian):
            hamiltonian_matrix = sum(c * m for c, m in zip(coeffs, matrices))
            res = np.trace(state @ hamiltonian_matrix)
        else:
            res = np.trace(state @ obs.matrix())

        expected = self.compute_expected_value(obs, state)
        assert np.allclose(res, expected.real)

    def test_expval_mixed_state(self, obs, coeffs, matrices):
        theta = 0.123
        weights = [0.33, 0.67]
        state = self.prepare_mixed_state(theta, weights)

        if isinstance(obs, qml.Hamiltonian):
            hamiltonian_matrix = sum(c * m for c, m in zip(coeffs, matrices))
            res = np.trace(state @ hamiltonian_matrix)
        else:
            res = np.trace(state @ obs.matrix())

        expected = self.compute_expected_value(obs, state)
        assert np.allclose(res, expected.real)


@pytest.mark.parametrize("ml_framework", ml_frameworks_list)
class TestBroadcasting:
    """Test that measurements work when the state has a batch dim"""

    num_qubits = 2

    @pytest.mark.parametrize(
        "measurement, get_expected",
        [
            (
                qml.density_matrix(wires=[0, 1]),
                lambda x: math.reshape(x, newshape=(BATCH_SIZE, 4, 4)),
            ),
            (qml.density_matrix(wires=[1]), lambda x: math.trace(x, axis1=1, axis2=3)),
        ],
    )
    def test_state_measurement(
        self, measurement, get_expected, ml_framework, two_qubit_batched_state
    ):
        """Test that state measurements work on broadcasted state"""
        initial_state = math.asarray(two_qubit_batched_state, like=ml_framework)
        res = measure(measurement, initial_state, is_state_batched=True)
        expected = get_expected(two_qubit_batched_state)

        assert qml.math.get_interface(res) == ml_framework
        assert np.allclose(res, expected)

    @pytest.mark.parametrize(
        "measurement, matrix_transform",
        [
            (qml.probs(wires=[0, 1]), lambda x: math.reshape(x, (2, 4, 4))),
            (qml.probs(wires=[0]), lambda x: math.trace(x, axis1=2, axis2=4)),
        ],
    )
    def test_probs_measurement(
        self, measurement, matrix_transform, ml_framework, two_qubit_batched_state
    ):
        """Test that probability measurements work on broadcasted state"""
        initial_state = math.asarray(two_qubit_batched_state, like=ml_framework)
        res = measure(measurement, initial_state, is_state_batched=True)

        transformed_state = matrix_transform(two_qubit_batched_state)

        expected = []
        for i in range(BATCH_SIZE):
            expected.append(math.diag(transformed_state[i]))

        assert qml.math.get_interface(res) == ml_framework
        assert np.allclose(res, expected)

    @pytest.mark.parametrize(
        "observable",
        [
            qml.PauliX(0),
            qml.PauliX(1),
            qml.prod(qml.PauliX(0), qml.PauliX(1)),
            qml.Hermitian(
                np.array(
                    [
                        [1.37770247 + 0.0j, 0.60335894 - 0.10889947j],
                        [0.60335894 + 0.10889947j, 0.90178212 + 0.0j],
                    ]
                ),
                wires=1,
            ),
        ],
    )
    def test_expval_measurement(self, observable, ml_framework, two_qubit_batched_state):
        """Test that expval measurements work on broadcasted state"""
        initial_state = math.asarray(two_qubit_batched_state, like=ml_framework)
        res = measure(qml.expval(observable), initial_state, is_state_batched=True)

        expected = [get_expval(observable, two_qubit_batched_state[i]) for i in range(BATCH_SIZE)]

        assert qml.math.get_interface(res) == ml_framework
        assert qml.math.allclose(res, expected)

    @pytest.mark.parametrize(
        "observable",
        [
            qml.sum((2 * qml.PauliX(0)), (0.4 * qml.PauliX(1))),
            qml.sum((2.4 * qml.PauliZ(0)), (0.2 * qml.PauliX(1))),
            qml.sum((0.9 * qml.PauliY(1)), (1.2 * qml.PauliX(0))),
        ],
    )
    def test_expval_sum_measurement(self, observable, ml_framework, two_qubit_batched_state):
        """Test that expval Sum measurements work on broadcasted state"""
        initial_state = math.asarray(two_qubit_batched_state, like=ml_framework)
        res = measure(qml.expval(observable), initial_state, is_state_batched=True)

        expanded_mat = np.zeros(((4, 4)), dtype=complex)
        for summand in observable:
            mat = summand.matrix()
            expanded_mat += (
                np.kron(np.eye(2), mat) if summand.wires[0] == 1 else np.kron(mat, np.eye(2))
            )

        expected = []
        for i in range(BATCH_SIZE):
            expval_sum = 0.0
            for summand in observable:
                expval_sum += get_expval(summand, two_qubit_batched_state[i])
            expected.append(expval_sum)

        assert qml.math.get_interface(res) == ml_framework
        assert qml.math.allclose(res, expected)

    def test_expval_hamiltonian_measurement(self, ml_framework, two_qubit_batched_state):
        """Test that expval Hamiltonian measurements work on broadcasted state"""
        initial_state = math.asarray(two_qubit_batched_state, like=ml_framework)
        observables = [qml.PauliX(1), qml.PauliX(0)]
        coeffs = math.convert_like([2, 0.4], initial_state)
        observable = qml.Hamiltonian(coeffs, observables)
        res = measure(qml.expval(observable), initial_state, is_state_batched=True)

        expanded_mat = np.zeros(((4, 4)), dtype=complex)
        for coeff, summand in zip(coeffs, observables):
            mat = summand.matrix()
            expanded_mat = np.add(
                expanded_mat,
                coeff
                * (np.kron(np.eye(2), mat) if summand.wires[0] == 1 else np.kron(mat, np.eye(2))),
            )

        expected = []
        for i in range(BATCH_SIZE):
            expval_sum = 0.0
            for coeff, summand in zip(coeffs, observables):
                expval_sum += coeff * get_expval(summand, two_qubit_batched_state[i])
            expected.append(expval_sum)

        assert qml.math.get_interface(res) == ml_framework
        assert qml.math.allclose(res, expected)

    @pytest.mark.parametrize(
        "observable",
        [
            qml.PauliX(0),
            qml.PauliX(1),
            (qml.PauliX(0) @ qml.PauliX(1)),
        ],
    )
    def test_variance_measurement(self, observable, ml_framework, two_qubit_batched_state):
        """Test that variance measurements work on broadcasted state."""
        initial_state = math.asarray(two_qubit_batched_state, like=ml_framework)
        res = measure(qml.var(observable), initial_state, is_state_batched=True)

        obs_squared = qml.prod(observable, observable)

        expected = []
        for state in two_qubit_batched_state:
            expval_obs = get_expval(observable, state)
            expval_of_squared_obs = get_expval(obs_squared, state)
            expected.append(expval_of_squared_obs - expval_obs**2)

        assert qml.math.get_interface(res) == ml_framework
        assert np.allclose(res, expected)


class TestSumOfTermsDifferentiability:
    x = 0.52

    @staticmethod
    def f(scale, coeffs, n_wires=5, offset=0.1):
        """Function to differentiate that implements a circuit with a SumOfTerms operator"""
        ops = [qml.RX(offset + scale * i, wires=i) for i in range(n_wires)]
        H = qml.Hamiltonian(
            coeffs,
            [
                reduce(lambda x, y: x @ y, (qml.PauliX(i) for i in range(n_wires))),
                reduce(lambda x, y: x @ y, (qml.PauliY(i) for i in range(n_wires))),
            ],
        )
        state = create_initial_state(range(n_wires), like=math.get_interface(scale))
        for op in ops:
            state = apply_operation(op, state)
        return measure(qml.expval(H), state)

    @staticmethod
    def expected(scale, coeffs, n_wires=5, offset=0.1, like="numpy"):
        """Get the expected expval of the class' circuit."""
        phase = offset + scale * qml.math.arange(n_wires, like=like)
        sines = qml.math.sin(phase)
        sign = (-1) ** n_wires  # For n_wires=5, sign = -1
        return coeffs[1] * sign * qml.math.prod(sines)

    @pytest.mark.autograd
    @pytest.mark.parametrize(
        "coeffs",
        [
            (qml.numpy.array(2.5), qml.numpy.array(6.2)),
            (qml.numpy.array(2.5, requires_grad=False), qml.numpy.array(6.2, requires_grad=False)),
        ],
    )
    def test_autograd_backprop(self, coeffs):
        """Test that backpropagation derivatives work in autograd with
        Hamiltonians using new and old math."""

        x = qml.numpy.array(self.x)
        out = self.f(x, coeffs)
        expected_out = self.expected(x, coeffs)
        assert qml.math.allclose(out, expected_out)

        gradient = qml.grad(self.f)(x, coeffs)
        expected_gradient = qml.grad(self.expected)(x, coeffs)
        assert qml.math.allclose(expected_gradient, gradient)

    @pytest.mark.autograd
    def test_autograd_backprop_coeffs(self):
        """Test that backpropagation derivatives work in autograd with
        the coefficients of Hamiltonians using new and old math."""

        coeffs = qml.numpy.array((2.5, 6.2), requires_grad=True)
        gradient = qml.grad(self.f, argnum=1)(self.x, coeffs)
        expected_gradient = qml.grad(self.expected)(self.x, coeffs)

        assert len(gradient) == 2
        assert qml.math.allclose(expected_gradient, gradient)

    @pytest.mark.jax
    @pytest.mark.parametrize("use_jit", (True, False))
    def test_jax_backprop(self, use_jit):
        """Test that backpropagation derivatives work with jax with
        Hamiltonians using new and old math."""
        import jax

        jax.config.update("jax_enable_x64", True)

        x = jax.numpy.array(self.x, dtype=jax.numpy.float64)
        coeffs = (5.2, 6.7)
        f = jax.jit(self.f, static_argnums=(1, 2, 3)) if use_jit else self.f

        out = f(x, coeffs)
        expected_out = self.expected(x, coeffs)
        assert qml.math.allclose(out, expected_out)

        gradient = jax.grad(f)(x, coeffs)
        expected_gradient = jax.grad(self.expected)(x, coeffs)
        assert qml.math.allclose(expected_gradient, gradient)

    @pytest.mark.jax
    def test_jax_backprop_coeffs(self):
        """Test that backpropagation derivatives work with jax with
        the coefficients of Hamiltonians using new and old math."""
        import jax

        jax.config.update("jax_enable_x64", True)
        coeffs = jax.numpy.array((5.2, 6.7), dtype=jax.numpy.float64)

        gradient = jax.grad(self.f, argnums=1)(self.x, coeffs)
        expected_gradient = jax.grad(self.expected, argnums=1)(self.x, coeffs)
        assert len(gradient) == 2
        assert qml.math.allclose(expected_gradient, gradient)

    @pytest.mark.torch
    def test_torch_backprop(self):
        """Test that backpropagation derivatives work with torch with
        Hamiltonians using new and old math."""
        import torch

        coeffs = [
            torch.tensor(9.2, requires_grad=False, dtype=torch.float64),
            torch.tensor(6.2, requires_grad=False, dtype=torch.float64),
        ]

        x = torch.tensor(-0.289, requires_grad=True, dtype=torch.float64)
        x2 = torch.tensor(-0.289, requires_grad=True, dtype=torch.float64)
        out = self.f(x, coeffs)
        expected_out = self.expected(x2, coeffs, like="torch")
        assert qml.math.allclose(out, expected_out)

        out.backward()
        expected_out.backward()
        assert qml.math.allclose(x.grad, x2.grad)

    @pytest.mark.torch
    def test_torch_backprop_coeffs(self):
        """Test that backpropagation derivatives work with torch with
        the coefficients of Hamiltonians using new and old math."""
        import torch

        coeffs = torch.tensor((9.2, 6.2), requires_grad=True, dtype=torch.float64)
        coeffs_expected = torch.tensor((9.2, 6.2), requires_grad=True, dtype=torch.float64)

        x = torch.tensor(-0.289, requires_grad=False, dtype=torch.float64)
        out = self.f(x, coeffs)
        expected_out = self.expected(x, coeffs_expected, like="torch")
        assert qml.math.allclose(out, expected_out)

        out.backward()
        expected_out.backward()
        assert len(coeffs.grad) == 2
        assert qml.math.allclose(coeffs.grad, coeffs_expected.grad)

    @pytest.mark.tf
    def test_tf_backprop(self):
        """Test that backpropagation derivatives work with tensorflow with
        Hamiltonians using new and old math."""
        import tensorflow as tf

        x = tf.Variable(self.x, dtype="float64")
        coeffs = [8.3, 5.7]

        with tf.GradientTape() as tape1:
            out = self.f(x, coeffs)

        with tf.GradientTape() as tape2:
            expected_out = self.expected(x, coeffs)

        assert qml.math.allclose(out, expected_out)
        gradient = tape1.gradient(out, x)
        expected_gradient = tape2.gradient(expected_out, x)
        assert qml.math.allclose(expected_gradient, gradient)

    @pytest.mark.tf
    def test_tf_backprop_coeffs(self):
        """Test that backpropagation derivatives work with tensorflow with
        the coefficients of Hamiltonians using new and old math."""
        import tensorflow as tf

        coeffs = tf.Variable([8.3, 5.7], dtype="float64")

        with tf.GradientTape() as tape1:
            out = self.f(self.x, coeffs)

        with tf.GradientTape() as tape2:
            expected_out = self.expected(self.x, coeffs)

        gradient = tape1.gradient(out, coeffs)
        expected_gradient = tape2.gradient(expected_out, coeffs)
        assert len(gradient) == 2
        assert qml.math.allclose(expected_gradient, gradient)


# pylint: disable=too-few-public-methods
class TestReadoutErrors:
    """Test that readout errors are correctly applied to measurements."""

    def test_readout_error(self):
        """Test that readout errors are correctly applied to measurements."""
        # Define the readout error probability
        p = 0.1  # Probability of bit-flip error during readout

        # Define the readout error operation using qml.BitFlip
        def readout_error(wires):
            return qml.BitFlip(p, wires=wires)

        # Define the state: let's use the |+⟩ state
        state_vector = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)])
        state = np.outer(state_vector, np.conj(state_vector))

        # Define the observable
        obs = qml.PauliX(0)

        # Calculate the expected value with readout errors
        expected = 1 - 2 * p  # Since p = 0.1, expected = 0.8

        # Measure the observable using the measure function with readout errors
        res = measure(qml.expval(obs), state, readout_errors=[readout_error])

        assert np.allclose(res, expected), f"Expected {expected}, got {res}"
