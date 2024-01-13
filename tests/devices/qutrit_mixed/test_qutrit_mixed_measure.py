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

import pytest

import pennylane as qml
from pennylane import numpy as np

# from pennylane.devices.qutrit_mixed. import measure
from pennylane.devices.qutrit_mixed.measure import (
    measure,
    apply_observable_einsum,
    get_measurement_function,
    state_diagonalizing_gates,
    trace_method,
    sum_of_terms_method,
)
import os

ml_frameworks_list = [
    "numpy",
    pytest.param("autograd", marks=pytest.mark.autograd),
    pytest.param("jax", marks=pytest.mark.jax),
    pytest.param("torch", marks=pytest.mark.torch),
    pytest.param("tensorflow", marks=pytest.mark.tf),
]
BATCH_SIZE = 2


class TestCurrentlyUnsupportedCases:
    # pylint: disable=too-few-public-methods
    def test_sample_based_observable(self, two_qutrit_state):
        """Test sample-only measurements raise a notimplementedError."""
        with pytest.raises(NotImplementedError):
            _ = measure(qml.sample(wires=0), two_qutrit_state)


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
            qml.var(qml.GellMann(0, 3)),
            qml.expval(qml.sum(qml.GellMann(0, 3), qml.GellMann(0, 1))),
            qml.expval(qml.sum(*(qml.GellMann(i, 1) for i in range(15)))),
            qml.expval(qml.prod(qml.GellMann(0, 4), qml.GellMann(1, 5), qml.GellMann(10, 6))),
        ),
    )
    def test_diagonalizing_gates(self, m):
        """Test that the state_diagonalizing gates are used when there's an observable has diagonalizing
        gates and allows the measurement to be efficiently computed with them."""
        assert get_measurement_function(m, state=1) is state_diagonalizing_gates

    def test_sum_of_terms(self):
        """Check that the sum of terms method is used when TODO"""
        pass

    def test_hermitian_trace(self):
        """Test that the expectation value of a hermitian uses the trace method."""
        mp = qml.expval(qml.THermitian(np.eye(3), wires=0))
        assert get_measurement_function(mp, state=1) is trace_method

    def test_hamiltonian_sum_of_terms_when_backprop(self):
        """Check that the sum of terms method is used when the state is trainable."""
        H = qml.Hamiltonian([2], [qml.GellMann(0, 1)])
        state = qml.numpy.zeros((3, 3))
        assert get_measurement_function(qml.expval(H), state) is sum_of_terms_method

    def test_sum_sum_of_terms_when_backprop(self):
        """Check that the sum of terms method is used when"""
        S = qml.prod(*(qml.GellMann(i, 1) for i in range(8))) + qml.prod(
            *(qml.GellMann(i, 2) for i in range(8))
        )
        state = qml.numpy.zeros((3, 3))
        assert get_measurement_function(qml.expval(S), state) is sum_of_terms_method


observable_list = [qml.GellMann(2, 3)]  # TODO add more observables


@pytest.mark.parametrize("obs", observable_list)
@pytest.mark.parametrize("ml_framework", ml_frameworks_list)
class TestApplyObservableEinsum:
    num_qutrits = 3
    dims = (3**num_qutrits, 3**num_qutrits)

    def test_apply_observable_einsum(self, obs, three_qutrit_state, ml_framework):
        res = apply_observable_einsum(
            obs, qml.math.asarray(three_qutrit_state, like=ml_framework), is_state_batched=False
        )
        missing_wires = self.num_qutrits - len(obs.wires)
        mat = obs.matrix()
        expanded_mat = np.kron(np.eye(3**missing_wires), mat) if missing_wires else mat

        flattened_state = three_qutrit_state.reshape(self.dims)
        expected = (expanded_mat @ flattened_state).reshape([3] * (self.num_qutrits * 2))

        assert qml.math.get_interface(res) == ml_framework
        assert qml.math.allclose(res, expected)

    def test_apply_observable_einsum_batched(self, obs, three_qutrit_batched_state, ml_framework):
        """Tests that unbatched operations are applied correctly to a batched state. TODO fix docstring"""
        res = apply_observable_einsum(
            obs,
            qml.math.asarray(three_qutrit_batched_state, like=ml_framework),
            is_state_batched=True,
        )
        missing_wires = self.num_qutrits - len(obs.wires)
        mat = obs.matrix()
        expanded_mat = np.kron(np.eye(3**missing_wires), mat) if missing_wires else mat
        expected = []

        for i in range(BATCH_SIZE):
            flattened_state = three_qutrit_batched_state[i].reshape(self.dims)
            expected.append((expanded_mat @ flattened_state).reshape([3] * (self.num_qutrits * 2)))

        assert qml.math.get_interface(res) == ml_framework
        assert qml.math.allclose(res, expected)


class TestMeasurements:
    @pytest.mark.parametrize(
        "measurement, expected",
        [
            # (qml.density_matrix(wires=0), 0.5 * np.ones((2, 2))),
            # (qml.probs(wires=[0]), np.array([0.5, 0.5])),
        ],
    )
    def test_state_measurement_no_obs(self, measurement, expected):
        """Test that state measurements with no observable work as expected."""
        pass

    @pytest.mark.parametrize(
        "obs, expected",
        [
            (
                qml.Hamiltonian(
                    [-0.5, 2], [qml.GellMann(0, 7), qml.GellMann(0, 8)]
                ),  # TODO fix error
                0.5 * np.sin(0.123) + 2 * np.cos(0.123),  # TODO: find value
            ),
            (
                qml.THermitian(
                    -0.5 * qml.GellMann(0, 7).matrix() + 2 * qml.GellMann(0, 8).matrix(), wires=0
                ),
                0.5 * np.sin(0.123) + 2 * np.cos(0.123),  # TODO: find value
            ),
        ],
    )
    def test_hamiltonian_expval(self, obs, expected, two_qutrit_state):
        """Test that measurements of hamiltonian/ Thermitians work correctly."""
        res = measure(qml.expval(obs), two_qutrit_state)
        assert np.allclose(res, expected)

    def test_sum_expval_tensor_contraction(self):
        """Test that `Sum` expectation values are correct when tensor contraction
        is used for computation."""
        summands = (qml.prod(qml.GellMann(i, 1), qml.GellMann(i + 1, 3)) for i in range(4))
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
        expect_from_rot = lambda x: 4 * (-np.sin(x) * np.cos(x))
        expected = shmidts[0] * expect_from_rot(rots[0]) + schmidts[1] * expect_from_rot(rots[1])
        assert np.allclose(res, expected)


class TestBroadcasting:
    """Test that measurements work when the state has a batch dim"""

    @pytest.mark.parametrize(
        "measurement, expected",
        [
            (
                qml.state(),
                np.array(
                    [
                        # TODO Add states
                    ]
                ),
            ),
            (
                qml.density_matrix(wires=[0, 1]),
                np.array(
                    [
                        # TODO Add states
                    ]
                ),
            ),
            (qml.probs(wires=[0, 1]), []),  # TODO Add expected
            (qml.expval(qml.GellMann(1)), None),  # TODO Add expvals
            (qml.var(qml.GellMann(1)), None),  # TODO Add expvals
        ],
    )
    def test_state_measurement(self, measurement, expected):
        """Test that broadcasting works for regular state measurements"""
        state = None  # TODO add states
        state = np.stack(state)

        res = measure(measurement, state, is_state_batched=True)
        assert np.allclose(res, expected)


# TODO TestNaNMeasurements,
# TODO TestSumOfTermsDifferentiability in future PR
