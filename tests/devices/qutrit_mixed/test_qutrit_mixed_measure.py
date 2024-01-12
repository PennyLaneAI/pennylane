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
from pennylane.devices.qutrit_mixed import measure, apply_observable_einsum

class TestCurrentlyUnsupportedCases:
    # pylint: disable=too-few-public-methods
    def test_sample_based_observable(self):
        """Test sample-only measurements raise a notimplementedError."""
        pass
        # state = 0.5 * np.ones((2, 2))
        # with pytest.raises(NotImplementedError):
        #     _ = measure(qml.sample(wires=0), state)

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
                # qml.var(qml.PauliZ(0)),
                # qml.expval(qml.sum(qml.PauliZ(0), qml.PauliX(0))),
                # qml.expval(qml.sum(*(qml.PauliX(i) for i in range(15)))),
                # qml.expval(qml.prod(qml.PauliX(0), qml.PauliY(1), qml.PauliZ(10))),
        ),
    )
    def test_diagonalizing_gates(self, m):
        """Test that the state_diagonalizing gates are used when there's an observable has diagonalizing
        gates and allows the measurement to be efficiently computed with them."""
        assert get_measurement_function(m, state=1) is state_diagonalizing_gates

    def test_sum_of_terms(self):
        """Check that the sum of terms method is used when TODO"""
        pass

    def test_trace_method(self):
        """Check that the trace method is used when TODO"""
        pass


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
            # (
            #         qml.Hamiltonian([-0.5, 2], [qml.PauliY(0), qml.PauliZ(0)]),
            #         0.5 * np.sin(0.123) + 2 * np.cos(0.123),
            # ),
            # (
            #         qml.SparseHamiltonian(
            #             csr_matrix(-0.5 * qml.PauliY(0).matrix() + 2 * qml.PauliZ(0).matrix()),
            #             wires=[0],
            #         ),
            #         0.5 * np.sin(0.123) + 2 * np.cos(0.123),
            # ),
            # (
            #         qml.Hermitian(-0.5 * qml.PauliY(0).matrix() + 2 * qml.PauliZ(0).matrix(), wires=0),
            #         0.5 * np.sin(0.123) + 2 * np.cos(0.123),
            # ),
        ],
    )
    def test_hamiltonian_expval(self, obs, expected):
        """Test that measurements of hamiltonian/ sparse hamiltonian/ hermitians work correctly."""
        pass

    def test_sum_expval_tensor_contraction(self):
        """Test that `Sum` expectation values are correct when tensor contraction
        is used for computation."""


class TestBroadcasting:
    """Test that measurements work when the state has a batch dim"""

    @pytest.mark.parametrize(
        "measurement, expected",
        [
            (
                    qml.state(),
                    np.array(
                        [
                            [0, 0, 0, 1],
                            [1 / np.sqrt(2), 0, 1 / np.sqrt(2), 0],
                            [1 / 2, 1 / 2, 1 / 2, 1 / 2],
                        ]
                    ),
            ),
            (
                    qml.density_matrix(wires=[0, 1]),
                    np.array(
                        [
                            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]],
                            [[1 / 2, 0, 1 / 2, 0], [0, 0, 0, 0], [1 / 2, 0, 1 / 2, 0], [0, 0, 0, 0]],
                            [
                                [1 / 4, 1 / 4, 1 / 4, 1 / 4],
                                [1 / 4, 1 / 4, 1 / 4, 1 / 4],
                                [1 / 4, 1 / 4, 1 / 4, 1 / 4],
                                [1 / 4, 1 / 4, 1 / 4, 1 / 4],
                            ],
                        ]
                    ),
            ),
            (
                    qml.probs(wires=[0, 1]),
                    np.array([[0, 0, 0, 1], [1 / 2, 0, 1 / 2, 0], [1 / 4, 1 / 4, 1 / 4, 1 / 4]]),
            ),
            (qml.expval(qml.PauliZ(1)), np.array([-1, 1, 0])),
            (qml.var(qml.PauliZ(1)), np.array([0, 0, 1])),
        ],
    )
    def test_state_measurement(self, measurement, expected):
        """Test that broadcasting works for regular state measurements"""
        pass

    def test_sparse_hamiltonian(self):
        """Test that broadcasting works for expectation values of SparseHamiltonians"""
        pass


#TODO TestNaNMeasurements, TestSumOfTermsDifferentiability