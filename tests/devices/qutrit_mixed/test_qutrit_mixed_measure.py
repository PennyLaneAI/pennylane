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

import pennylane as qp
from pennylane import math
from pennylane import numpy as np
from pennylane.devices.qutrit_mixed import apply_operation, create_initial_state, measure
from pennylane.devices.qutrit_mixed.measure import (
    calculate_expval,
    calculate_expval_sum_of_terms,
    calculate_probability,
    calculate_reduced_density_matrix,
    calculate_variance,
    get_measurement_function,
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
    probs = measure(qp.probs(), state)
    expected_probs = np.array([1 - 1e-4, 0, 1e-4])
    assert np.allclose(probs, expected_probs)


@pytest.mark.parametrize(
    "mp", [qp.sample(), qp.counts(), qp.sample(wires=0), qp.counts(wires=0)]
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
        mp1 = qp.state()
        assert get_measurement_function(mp1) is calculate_reduced_density_matrix

    def test_prod_calculate_expval_method(self):
        """Test that the expectation value of a product uses the calculate expval method."""
        prod = qp.prod(*(qp.GellMann(i, 1) for i in range(8)))
        assert get_measurement_function(qp.expval(prod)) is calculate_expval

    def test_hermitian_calculate_expval_method(self):
        """Test that the expectation value of a hermitian uses the calculate expval method."""
        mp = qp.expval(qp.THermitian(np.eye(3), wires=0))
        assert get_measurement_function(mp) is calculate_expval

    def test_hamiltonian_sum_of_terms(self):
        """Check that the sum of terms method is used when Hamiltonian."""
        H = qp.Hamiltonian([2], [qp.GellMann(0, 1)])
        assert get_measurement_function(qp.expval(H)) is calculate_expval_sum_of_terms

    def test_sum_sum_of_terms(self):
        """Check that the sum of terms method is used when sum of terms"""
        S = qp.prod(*(qp.GellMann(i, 1) for i in range(8))) + qp.prod(
            *(qp.GellMann(i, 2) for i in range(8))
        )
        assert get_measurement_function(qp.expval(S)) is calculate_expval_sum_of_terms

    def test_probs_compute_probabilities(self):
        """Check that compute probabilities method is used when probs"""
        assert get_measurement_function(qp.probs()) is calculate_probability

    def test_var_compute_variance(self):
        """Check that the compute variance method is used when variance"""
        obs = qp.GellMann(0, 1)
        assert get_measurement_function(qp.var(obs)) is calculate_variance


class TestMeasurements:
    """Test that measurements on unbatched states work as expected."""

    @pytest.mark.parametrize(
        "measurement, get_expected",
        [
            (qp.state(), lambda x: math.reshape(x, (9, 9))),
            (qp.density_matrix(wires=0), lambda x: math.trace(x, axis1=1, axis2=3)),
            (
                qp.probs(wires=[0]),
                lambda x: math.real(math.diag(math.trace(x, axis1=1, axis2=3))),
            ),
            (
                qp.probs(),
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
            ([-0.5, 2], [qp.GellMann(0, 7), qp.GellMann(1, 8)]),
            ([-0.3, 1], [qp.GellMann(0, 2), qp.GellMann(0, 4)]),
            ([-0.45, 2.6], [qp.GellMann(1, 6), qp.GellMann(1, 3)]),
        ],
    )
    def test_hamiltonian_expval(self, coeffs, observables, two_qutrit_state):
        """Test that measurements of hamiltonian work correctly."""

        obs = qp.Hamiltonian(coeffs, observables)
        res = measure(qp.expval(obs), two_qutrit_state)

        expected = 0
        for i, coeff in enumerate(coeffs):
            expected += coeff * get_expval(observables[i], two_qutrit_state)

        assert np.isclose(res, expected)

    @pytest.mark.parametrize(
        "observable",
        [
            qp.THermitian(
                -0.5 * qp.GellMann(0, 7).matrix() + 2 * qp.GellMann(0, 8).matrix(), wires=0
            ),
            qp.THermitian(
                -0.55 * qp.GellMann(1, 4).matrix() + 2.4 * qp.GellMann(1, 5).matrix(), wires=1
            ),
        ],
    )
    def test_hermitian_expval(self, observable, two_qutrit_state):
        """Test that measurements of qutrit hermitian work correctly."""
        res = measure(qp.expval(observable), two_qutrit_state)
        expected = get_expval(observable, two_qutrit_state)

        assert np.isclose(res, expected)

    def test_sum_expval_tensor_contraction(self):
        """Test that `Sum` expectation values are correct when tensor contraction
        is used for computation."""
        summands = (qp.prod(qp.GellMann(i, 2), qp.GellMann(i + 1, 3)) for i in range(4))
        obs = qp.sum(*summands)

        @qp.qnode(qp.device("default.qutrit", wires=5))
        def find_state(x):
            for i in range(5):
                qp.TRX(x, wires=i)
            return qp.state()

        rots = [0.123, 0.321]
        schmidts = [0.7, 0.3]
        state = np.zeros([3] * (2 * 5), dtype=complex)

        for i in range(2):
            vec = find_state(rots[i])
            state += schmidts[i] * np.outer(vec, np.conj(vec)).reshape([3] * (2 * 5))

        res = measure(qp.expval(obs), state)
        expected = 0
        for schmidt, theta in zip(schmidts, rots):
            expected += schmidt * (4 * (-np.sin(theta) * np.cos(theta)))

        assert np.allclose(res, expected)

    @pytest.mark.parametrize(
        "observable",
        [
            qp.GellMann(0, 1),
            qp.GellMann(1, 6),
            (qp.GellMann(0, 6) @ qp.GellMann(1, 2)),
        ],
    )
    def test_variance_measurement(self, observable, two_qutrit_state):
        """Test that variance measurements work as expected."""
        res = measure(qp.var(observable), two_qutrit_state)

        expval_obs = get_expval(observable, two_qutrit_state)

        obs_squared = qp.prod(observable, observable)
        expval_of_squared_obs = get_expval(obs_squared, two_qutrit_state)

        expected = expval_of_squared_obs - expval_obs**2
        assert np.allclose(res, expected)


@pytest.mark.parametrize(
    "obs",
    [
        qp.Hamiltonian([-0.5, 2], [qp.GellMann(0, 5), qp.GellMann(0, 3)]),
        qp.THermitian(
            -0.5 * qp.GellMann(0, 5).matrix() + 2 * qp.GellMann(0, 3).matrix(), wires=0
        ),
    ],
)
class TestExpValAnalytical:
    def test_expval_pure_state(self, obs):
        """Test that measurements work on pure states as expected from analytical calculation."""
        # Create TRX[0,2](0.246)TRX[0,1](0.246)|0> state
        state_vector = np.array([np.cos(0.123) ** 2, -1j * np.sin(0.123), -1j * np.sin(0.246) / 2])
        state = np.outer(state_vector, np.conj(state_vector))
        res = measure(qp.expval(obs), state)

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

        res = measure(qp.expval(obs), state)
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
            (qp.state(), lambda x: math.reshape(x, (BATCH_SIZE, 9, 9))),
            (
                qp.density_matrix(wires=[0, 1]),
                lambda x: math.reshape(x, (BATCH_SIZE, 9, 9)),
            ),
            (qp.density_matrix(wires=[1]), lambda x: math.trace(x, axis1=1, axis2=3)),
        ],
    )
    def test_state_measurement(
        self, measurement, get_expected, ml_framework, two_qutrit_batched_state
    ):
        """Test that state measurements work on broadcasted state"""
        initial_state = math.asarray(two_qutrit_batched_state, like=ml_framework)
        res = measure(measurement, initial_state, is_state_batched=True)
        expected = get_expected(two_qutrit_batched_state)

        assert qp.math.get_interface(res) == ml_framework
        assert np.allclose(res, expected)

    @pytest.mark.parametrize(
        "measurement, matrix_transform",
        [
            (qp.probs(wires=[0, 1]), lambda x: math.reshape(x, (2, 9, 9))),
            (qp.probs(wires=[0]), lambda x: math.trace(x, axis1=2, axis2=4)),
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

        assert qp.math.get_interface(res) == ml_framework
        assert np.allclose(res, expected)

    @pytest.mark.parametrize(
        "observable",
        [
            qp.GellMann(1, 3),
            qp.GellMann(0, 6),
            qp.prod(qp.GellMann(0, 2), qp.GellMann(1, 3)),
            qp.THermitian(
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
        res = measure(qp.expval(observable), initial_state, is_state_batched=True)

        expected = [get_expval(observable, two_qutrit_batched_state[i]) for i in range(BATCH_SIZE)]

        assert qp.math.get_interface(res) == ml_framework
        assert qp.math.allclose(res, expected)

    @pytest.mark.parametrize(
        "observable",
        [
            qp.sum((2 * qp.GellMann(1, 1)), (0.4 * qp.GellMann(0, 6))),
            qp.sum((2.4 * qp.GellMann(0, 3)), (0.2 * qp.GellMann(0, 7))),
            qp.sum((0.9 * qp.GellMann(1, 5)), (1.2 * qp.GellMann(1, 8))),
        ],
    )
    def test_expval_sum_measurement(self, observable, ml_framework, two_qutrit_batched_state):
        """Test that expval Sum measurements work on broadcasted state"""
        initial_state = math.asarray(two_qutrit_batched_state, like=ml_framework)
        res = measure(qp.expval(observable), initial_state, is_state_batched=True)

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

        assert qp.math.get_interface(res) == ml_framework
        assert qp.math.allclose(res, expected)

    def test_expval_hamiltonian_measurement(self, ml_framework, two_qutrit_batched_state):
        """Test that expval Hamiltonian measurements work on broadcasted state"""
        initial_state = math.asarray(two_qutrit_batched_state, like=ml_framework)
        observables = [qp.GellMann(1, 1), qp.GellMann(0, 6)]
        coeffs = [2, 0.4]
        observable = qp.Hamiltonian(coeffs, observables)
        res = measure(qp.expval(observable), initial_state, is_state_batched=True)

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

        assert qp.math.get_interface(res) == ml_framework
        assert qp.math.allclose(res, expected)

    @pytest.mark.parametrize(
        "observable",
        [
            qp.GellMann(0, 1),
            qp.GellMann(1, 6),
            (qp.GellMann(0, 6) @ qp.GellMann(1, 2)),
        ],
    )
    def test_variance_measurement(self, observable, ml_framework, two_qutrit_batched_state):
        """Test that variance measurements work on broadcasted state."""
        initial_state = math.asarray(two_qutrit_batched_state, like=ml_framework)
        res = measure(qp.var(observable), initial_state, is_state_batched=True)

        obs_squared = qp.prod(observable, observable)

        expected = []
        for state in two_qutrit_batched_state:
            expval_obs = get_expval(observable, state)
            expval_of_squared_obs = get_expval(obs_squared, state)
            expected.append(expval_of_squared_obs - expval_obs**2)

        assert qp.math.get_interface(res) == ml_framework
        assert np.allclose(res, expected)


class TestSumOfTermsDifferentiability:
    x = 0.52

    @staticmethod
    def f(scale, coeffs, n_wires=5, offset=0.1):
        """Function to differentiate that implements a circuit with a SumOfTerms operator"""
        ops = [qp.TRX(offset + scale * i, wires=i, subspace=(0, 2)) for i in range(n_wires)]
        H = qp.Hamiltonian(
            coeffs,
            [
                reduce(lambda x, y: x @ y, (qp.GellMann(i, 3) for i in range(n_wires))),
                reduce(lambda x, y: x @ y, (qp.GellMann(i, 5) for i in range(n_wires))),
            ],
        )
        state = create_initial_state(range(n_wires), like=math.get_interface(scale))
        for op in ops:
            state = apply_operation(op, state)
        return measure(qp.expval(H), state)

    @staticmethod
    def expected(scale, coeffs, n_wires=5, offset=0.1, like="numpy"):
        """Get the expected expval of the class' circuit."""
        phase = offset + scale * qp.math.asarray(range(n_wires), like=like)
        cosines = math.cos(phase / 2) ** 2
        sines = -math.sin(phase)
        return coeffs[0] * qp.math.prod(cosines) + coeffs[1] * qp.math.prod(sines)

    @pytest.mark.autograd
    @pytest.mark.parametrize(
        "coeffs",
        [
            (qp.numpy.array(2.5), qp.numpy.array(6.2)),
            (qp.numpy.array(2.5, requires_grad=False), qp.numpy.array(6.2, requires_grad=False)),
        ],
    )
    def test_autograd_backprop(self, coeffs):
        """Test that backpropagation derivatives work in autograd with
        Hamiltonians using new and old math."""

        x = qp.numpy.array(self.x)
        out = self.f(x, coeffs)
        expected_out = self.expected(x, coeffs)
        assert qp.math.allclose(out, expected_out)

        gradient = qp.grad(self.f)(x, coeffs)
        expected_gradient = qp.grad(self.expected)(x, coeffs)
        assert qp.math.allclose(expected_gradient, gradient)

    @pytest.mark.autograd
    def test_autograd_backprop_coeffs(self):
        """Test that backpropagation derivatives work in autograd with
        the coefficients of Hamiltonians using new and old math."""

        coeffs = qp.numpy.array((2.5, 6.2), requires_grad=True)
        gradient = qp.grad(self.f, argnums=1)(self.x, coeffs)
        expected_gradient = qp.grad(self.expected)(self.x, coeffs)

        assert len(gradient) == 2
        assert qp.math.allclose(expected_gradient, gradient)

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
        assert qp.math.allclose(out, expected_out)

        gradient = jax.grad(f)(x, coeffs)
        expected_gradient = jax.grad(self.expected)(x, coeffs)
        assert qp.math.allclose(expected_gradient, gradient)

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
        assert qp.math.allclose(expected_gradient, gradient)

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
        assert qp.math.allclose(out, expected_out)

        out.backward()
        expected_out.backward()
        assert qp.math.allclose(x.grad, x2.grad)

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
        assert qp.math.allclose(out, expected_out)

        out.backward()
        expected_out.backward()
        assert len(coeffs.grad) == 2
        assert qp.math.allclose(coeffs.grad, coeffs_expected.grad)

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

        assert qp.math.allclose(out, expected_out)
        gradient = tape1.gradient(out, x)
        expected_gradient = tape2.gradient(expected_out, x)
        assert qp.math.allclose(expected_gradient, gradient)

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
        assert qp.math.allclose(expected_gradient, gradient)
