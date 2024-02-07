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
"""Unit tests for measuring states in devices/qutrit_mixed."""

from functools import reduce
import pytest

import pennylane as qml
from pennylane import numpy as np
from pennylane import math

from pennylane.devices.qutrit_mixed import measure
from pennylane.devices.qutrit_mixed.measure import (
    apply_observable_einsum,
    get_measurement_function,
    calculate_expval,
    calculate_expval_sum_of_terms,
    calculate_reduced_density_matrix,
    calculate_probability,
    calculate_variance,
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
    num_qutrits = int(len(state.shape) / 2)
    pre_wires_identity = np.eye(3 ** min(op.wires))
    post_wires_identity = np.eye(3 ** ((num_qutrits - 1) - op.wires[-1]))

    expanded_op = reduce(np.kron, (pre_wires_identity, op.matrix(), post_wires_identity))
    flattened_state = state.reshape((3**num_qutrits,) * 2)
    return expanded_op @ flattened_state


def get_expval(op, state):
    """Finds op@state and traces to find the expectation value of observable on the state"""
    op_mult_state = get_expanded_op_mult_state(op, state)
    return np.trace(op_mult_state)


def test_probs_with_negative_on_diagonal():
    """Test that if there is a negative on diagonal it is clipped to 0"""
    state = np.array([[1 - 1e-4 + 0j, 0, 0], [0, -1e-4, 0], [0, 0, 1e-4]])
    probs = measure(qml.probs(), state)
    expected_probs = np.array([1 - 1e-4, 0, 1e-4])
    assert np.allclose(probs, expected_probs)


@pytest.mark.parametrize(
    "mp", [qml.sample(), qml.counts(), qml.sample(wires=0), qml.counts(wires=0)]
)
class TestCurrentlyUnsupportedCases:
    # pylint: disable=too-few-public-methods
    def test_sample_based_observable(self, mp, two_qutrit_state):
        """Test sample-only measurements raise a NotImplementedError."""
        with pytest.raises(NotImplementedError):
            _ = measure(mp, two_qutrit_state)


class TestMeasurementDispatch:
    """Test that get_measurement_function dispatchs to the correct place."""

    def test_state_no_obs(self):
        """Test that the correct internal function is used for a measurement process with no observables."""
        # Test a case where state_measurement_process is used
        mp1 = qml.state()
        assert get_measurement_function(mp1) is calculate_reduced_density_matrix

    def test_prod_calculate_expval_method(self):
        """Test that the expectation value of a product uses the calculate expval method."""
        prod = qml.prod(*(qml.GellMann(i, 1) for i in range(8)))
        assert get_measurement_function(qml.expval(prod)) is calculate_expval

    def test_hermitian_calculate_expval_method(self):
        """Test that the expectation value of a hermitian uses the calculate expval method."""
        mp = qml.expval(qml.THermitian(np.eye(3), wires=0))
        assert get_measurement_function(mp) is calculate_expval

    def test_hamiltonian_sum_of_terms(self):
        """Check that the sum of terms method is used when Hamiltonian."""
        H = qml.Hamiltonian([2], [qml.GellMann(0, 1)])
        assert get_measurement_function(qml.expval(H)) is calculate_expval_sum_of_terms

    def test_sum_sum_of_terms(self):
        """Check that the sum of terms method is used when sum of terms"""
        S = qml.prod(*(qml.GellMann(i, 1) for i in range(8))) + qml.prod(
            *(qml.GellMann(i, 2) for i in range(8))
        )
        assert get_measurement_function(qml.expval(S)) is calculate_expval_sum_of_terms

    def test_probs_compute_probabilities(self):
        """Check that compute probabilities method is used when probs"""
        assert get_measurement_function(qml.probs()) is calculate_probability

    def test_var_compute_variance(self):
        """Check that the compute variance method is used when variance"""
        obs = qml.GellMann(0, 1)
        assert get_measurement_function(qml.var(obs)) is calculate_variance


@pytest.mark.parametrize(
    "obs",
    [
        qml.GellMann(2, 2),
        qml.GellMann(1, 8),
        qml.GellMann(0, 5),
        (qml.GellMann(0, 5) @ qml.GellMann(1, 8)),
    ],
)
@pytest.mark.parametrize("ml_framework", ml_frameworks_list)
class TestApplyObservableEinsum:
    """Tests that observables are applied correctly for calculate expval method."""

    num_qutrits = 3
    dims = (3**num_qutrits, 3**num_qutrits)

    def test_apply_observable_einsum(self, obs, three_qutrit_state, ml_framework):
        """Tests that unbatched observables are applied correctly to a unbatched state."""
        res = apply_observable_einsum(
            obs, qml.math.asarray(three_qutrit_state, like=ml_framework), is_state_batched=False
        )
        expected = get_expanded_op_mult_state(obs, three_qutrit_state)
        expected = expected.reshape([3] * (3 * 2))

        assert qml.math.get_interface(res) == ml_framework
        assert qml.math.allclose(res, expected)

    def test_apply_observable_einsum_batched(self, obs, three_qutrit_batched_state, ml_framework):
        """Tests that unbatched observables are applied correctly to a batched state."""
        res = apply_observable_einsum(
            obs,
            qml.math.asarray(three_qutrit_batched_state, like=ml_framework),
            is_state_batched=True,
        )
        expected = [
            get_expanded_op_mult_state(obs, state).reshape([3] * (3 * 2))
            for state in three_qutrit_batched_state
        ]

        assert qml.math.get_interface(res) == ml_framework
        assert qml.math.allclose(res, expected)


class TestMeasurements:
    """Test that measurements on unbatched states work as expected."""

    @pytest.mark.parametrize(
        "measurement, get_expected",
        [
            (qml.state(), lambda x: math.reshape(x, newshape=(9, 9))),
            (qml.density_matrix(wires=0), lambda x: math.trace(x, axis1=1, axis2=3)),
            (
                qml.probs(wires=[0]),
                lambda x: math.real(math.diag(math.trace(x, axis1=1, axis2=3))),
            ),
            (
                qml.probs(),
                lambda x: math.real(math.diag(x.reshape(9, 9))),
            ),
        ],
    )
    def test_state_measurement_no_obs(self, measurement, get_expected, two_qutrit_state):
        """Test that state measurements with no observable work as expected."""
        res = measure(measurement, two_qutrit_state)
        expected = get_expected(two_qutrit_state)

        assert np.allclose(res, expected)

    @pytest.mark.parametrize(
        "coeffs, observables",
        [
            ([-0.5, 2], [qml.GellMann(0, 7), qml.GellMann(1, 8)]),
            ([-0.3, 1], [qml.GellMann(0, 2), qml.GellMann(0, 4)]),
            ([-0.45, 2.6], [qml.GellMann(1, 6), qml.GellMann(1, 3)]),
        ],
    )
    def test_hamiltonian_expval(self, coeffs, observables, two_qutrit_state):
        """Test that measurements of hamiltonian work correctly."""

        obs = qml.Hamiltonian(coeffs, observables)
        res = measure(qml.expval(obs), two_qutrit_state)

        expected = 0
        for i, coeff in enumerate(coeffs):
            expected += coeff * get_expval(observables[i], two_qutrit_state)

        assert np.isclose(res, expected)

    @pytest.mark.parametrize(
        "observable",
        [
            qml.THermitian(
                -0.5 * qml.GellMann(0, 7).matrix() + 2 * qml.GellMann(0, 8).matrix(), wires=0
            ),
            qml.THermitian(
                -0.55 * qml.GellMann(1, 4).matrix() + 2.4 * qml.GellMann(1, 5).matrix(), wires=1
            ),
        ],
    )
    def test_hermitian_expval(self, observable, two_qutrit_state):
        """Test that measurements of qutrit hermitian work correctly."""
        res = measure(qml.expval(observable), two_qutrit_state)
        expected = get_expval(observable, two_qutrit_state)

        assert np.isclose(res, expected)

    def test_sum_expval_tensor_contraction(self):
        """Test that `Sum` expectation values are correct when tensor contraction
        is used for computation."""
        summands = (qml.prod(qml.GellMann(i, 2), qml.GellMann(i + 1, 3)) for i in range(4))
        obs = qml.sum(*summands)

        @qml.qnode(qml.device("default.qutrit", wires=5))
        def find_state(x):
            for i in range(5):
                qml.TRX(x, wires=i)
            return qml.state()

        rots = [0.123, 0.321]
        schmidts = [0.7, 0.3]
        state = np.zeros([3] * (2 * 5), dtype=complex)

        for i in range(2):
            vec = find_state(rots[i])
            state += schmidts[i] * np.outer(vec, np.conj(vec)).reshape([3] * (2 * 5))

        res = measure(qml.expval(obs), state)
        expected = 0
        for schmidt, theta in zip(schmidts, rots):
            expected += schmidt * (4 * (-np.sin(theta) * np.cos(theta)))

        assert np.allclose(res, expected)

    @pytest.mark.parametrize(
        "observable",
        [
            qml.GellMann(0, 1),
            qml.GellMann(1, 6),
            (qml.GellMann(0, 6) @ qml.GellMann(1, 2)),
        ],
    )
    def test_variance_measurement(self, observable, two_qutrit_state):
        """Test that variance measurements work as expected."""
        res = measure(qml.var(observable), two_qutrit_state)

        expval_obs = get_expval(observable, two_qutrit_state)

        obs_squared = qml.prod(observable, observable)
        expval_of_squared_obs = get_expval(obs_squared, two_qutrit_state)

        expected = expval_of_squared_obs - expval_obs**2
        assert np.allclose(res, expected)


@pytest.mark.parametrize(
    "obs",
    [
        qml.Hamiltonian([-0.5, 2], [qml.GellMann(0, 5), qml.GellMann(0, 3)]),
        qml.THermitian(
            -0.5 * qml.GellMann(0, 5).matrix() + 2 * qml.GellMann(0, 3).matrix(), wires=0
        ),
    ],
)
class TestExpValAnalytical:
    def test_expval_pure_state(self, obs):
        """Test that measurements work on pure states as expected from analytical calculation."""
        # Create TRX[0,2](0.246)TRX[0,1](0.246)|0> state
        state_vector = np.array([np.cos(0.123) ** 2, -1j * np.sin(0.123), -1j * np.sin(0.246) / 2])
        state = np.outer(state_vector, np.conj(state_vector))
        res = measure(qml.expval(obs), state)

        expected = 0.5 * (np.sin(0.246) * np.cos(0.123) ** 2) + 2 * (
            np.cos(0.123) ** 4 - np.sin(0.123) ** 2
        )
        assert np.allclose(res, expected)

    def test_expval_mixed_state(self, obs):
        """Test that measurements work on mixed states as expected from analytical calculation."""
        # Create TRX[0,1](0.246)|0> state mixed with TRX[0,2](0.246)|0>
        state_vector_one = np.array([np.cos(0.123), -1j * np.sin(0.123), 0])
        state_one = np.outer(state_vector_one, np.conj(state_vector_one))

        state_vector_two = np.array([np.cos(0.123), 0, -1j * np.sin(0.123)])
        state_two = np.outer(state_vector_two, np.conj(state_vector_two))

        state = (0.33 * state_one) + (0.67 * state_two)

        res = measure(qml.expval(obs), state)
        expected_pure_state_one = 2 * np.cos(0.246)
        expected_pure_state_two = 0.5 * np.sin(0.246) + 2 * np.cos(0.123) ** 2
        expected = (0.33 * expected_pure_state_one) + (0.67 * expected_pure_state_two)
        assert np.allclose(res, expected)


@pytest.mark.parametrize("ml_framework", ml_frameworks_list)
class TestBroadcasting:
    """Test that measurements work when the state has a batch dim"""

    num_qutrits = 2

    @pytest.mark.parametrize(
        "measurement, get_expected",
        [
            (qml.state(), lambda x: math.reshape(x, newshape=(BATCH_SIZE, 9, 9))),
            (
                qml.density_matrix(wires=[0, 1]),
                lambda x: math.reshape(x, newshape=(BATCH_SIZE, 9, 9)),
            ),
            (qml.density_matrix(wires=[1]), lambda x: math.trace(x, axis1=1, axis2=3)),
        ],
    )
    def test_state_measurement(
        self, measurement, get_expected, ml_framework, two_qutrit_batched_state
    ):
        """Test that state measurements work on broadcasted state"""
        initial_state = math.asarray(two_qutrit_batched_state, like=ml_framework)
        res = measure(measurement, initial_state, is_state_batched=True)
        expected = get_expected(two_qutrit_batched_state)

        assert qml.math.get_interface(res) == ml_framework
        assert np.allclose(res, expected)

    @pytest.mark.parametrize(
        "measurement, matrix_transform",
        [
            (qml.probs(wires=[0, 1]), lambda x: math.reshape(x, (2, 9, 9))),
            (qml.probs(wires=[0]), lambda x: math.trace(x, axis1=2, axis2=4)),
        ],
    )
    def test_probs_measurement(
        self, measurement, matrix_transform, ml_framework, two_qutrit_batched_state
    ):
        """Test that probability measurements work on broadcasted state"""
        initial_state = math.asarray(two_qutrit_batched_state, like=ml_framework)
        res = measure(measurement, initial_state, is_state_batched=True)

        transformed_state = matrix_transform(two_qutrit_batched_state)

        expected = []
        for i in range(BATCH_SIZE):
            expected.append(math.diag(transformed_state[i]))

        assert qml.math.get_interface(res) == ml_framework
        assert np.allclose(res, expected)

    @pytest.mark.parametrize(
        "observable",
        [
            qml.GellMann(1, 3),
            qml.GellMann(0, 6),
            qml.prod(qml.GellMann(0, 2), qml.GellMann(1, 3)),
            qml.THermitian(
                np.array(
                    [
                        [1.37770247 + 0.0j, 0.60335894 - 0.10889947j, 0.98223403 - 0.94429544j],
                        [0.60335894 + 0.10889947j, 0.90178212 + 0.0j, 0.45529663 - 0.03054001j],
                        [0.98223403 + 0.94429544j, 0.45529663 + 0.03054001j, 0.37721683 + 0.0j],
                    ]
                ),
                wires=1,
            ),
        ],
    )
    def test_expval_measurement(self, observable, ml_framework, two_qutrit_batched_state):
        """Test that expval measurements work on broadcasted state"""
        initial_state = math.asarray(two_qutrit_batched_state, like=ml_framework)
        res = measure(qml.expval(observable), initial_state, is_state_batched=True)

        expected = [get_expval(observable, two_qutrit_batched_state[i]) for i in range(BATCH_SIZE)]

        assert qml.math.get_interface(res) == ml_framework
        assert qml.math.allclose(res, expected)

    @pytest.mark.parametrize(
        "observable",
        [
            qml.sum((2 * qml.GellMann(1, 1)), (0.4 * qml.GellMann(0, 6))),
            qml.sum((2.4 * qml.GellMann(0, 3)), (0.2 * qml.GellMann(0, 7))),
            qml.sum((0.9 * qml.GellMann(1, 5)), (1.2 * qml.GellMann(1, 8))),
        ],
    )
    def test_expval_sum_measurement(self, observable, ml_framework, two_qutrit_batched_state):
        """Test that expval Sum measurements work on broadcasted state"""
        initial_state = math.asarray(two_qutrit_batched_state, like=ml_framework)
        res = measure(qml.expval(observable), initial_state, is_state_batched=True)

        expanded_mat = np.zeros((9, 9), dtype=complex)
        for summand in observable:
            mat = summand.matrix()
            expanded_mat += (
                np.kron(np.eye(3), mat) if summand.wires[0] == 1 else np.kron(mat, np.eye(3))
            )

        expected = []
        for i in range(BATCH_SIZE):
            expval_sum = 0.0
            for summand in observable:
                expval_sum += get_expval(summand, two_qutrit_batched_state[i])
            expected.append(expval_sum)

        assert qml.math.get_interface(res) == ml_framework
        assert qml.math.allclose(res, expected)

    def test_expval_hamiltonian_measurement(self, ml_framework, two_qutrit_batched_state):
        """Test that expval Hamiltonian measurements work on broadcasted state"""
        initial_state = math.asarray(two_qutrit_batched_state, like=ml_framework)
        observables = [qml.GellMann(1, 1), qml.GellMann(0, 6)]
        coeffs = [2, 0.4]
        observable = qml.Hamiltonian(coeffs, observables)
        res = measure(qml.expval(observable), initial_state, is_state_batched=True)

        expanded_mat = np.zeros((9, 9), dtype=complex)
        for coeff, summand in zip(coeffs, observables):
            mat = summand.matrix()
            expanded_mat += coeff * (
                np.kron(np.eye(3), mat) if summand.wires[0] == 1 else np.kron(mat, np.eye(3))
            )

        expected = []
        for i in range(BATCH_SIZE):
            expval_sum = 0.0
            for coeff, summand in zip(coeffs, observables):
                expval_sum += coeff * get_expval(summand, two_qutrit_batched_state[i])
            expected.append(expval_sum)

        assert qml.math.get_interface(res) == ml_framework
        assert qml.math.allclose(res, expected)

    @pytest.mark.parametrize(
        "observable",
        [
            qml.GellMann(0, 1),
            qml.GellMann(1, 6),
            (qml.GellMann(0, 6) @ qml.GellMann(1, 2)),
        ],
    )
    def test_variance_measurement(self, observable, ml_framework, two_qutrit_batched_state):
        """Test that variance measurements work on broadcasted state."""
        initial_state = math.asarray(two_qutrit_batched_state, like=ml_framework)
        res = measure(qml.var(observable), initial_state, is_state_batched=True)

        obs_squared = qml.prod(observable, observable)

        expected = []
        for state in two_qutrit_batched_state:
            expval_obs = get_expval(observable, state)
            expval_of_squared_obs = get_expval(obs_squared, state)
            expected.append(expval_of_squared_obs - expval_obs**2)

        assert qml.math.get_interface(res) == ml_framework
        assert np.allclose(res, expected)


# TODO TestSumOfTermsDifferentiability in future PR (with other differentiabilty tests)
