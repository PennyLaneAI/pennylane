# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for measure in devices/qubit."""

import pytest

import numpy as np
from scipy.sparse import csr_matrix

import pennylane as qml
from pennylane.devices.qubit import simulate
from pennylane.devices.qubit.measure import (
    measure,
    state_diagonalizing_gates,
    state_hamiltonian_expval,
    state_measurement_process,
    get_measurement_function,
)


class TestCurrentlyUnsupportedCases:
    # pylint: disable=too-few-public-methods
    def test_sample_based_observable(self):
        """Test sample-only measurements raise a notimplementedError."""

        state = 0.5 * np.ones((2, 2))
        with pytest.raises(NotImplementedError):
            _ = measure(qml.sample(wires=0), state)


class TestMeasurements:
    @pytest.mark.parametrize(
        "measurement, expected",
        [
            (qml.state(), -0.5j * np.ones(4)),
            (qml.density_matrix(wires=0), 0.5 * np.ones((2, 2))),
            (qml.probs(wires=[0]), np.array([0.5, 0.5])),
        ],
    )
    def test_state_measurement_no_obs(self, measurement, expected):
        """Test that state measurements with no observable work as expected."""
        state = -0.5j * np.ones((2, 2))
        res = measure(measurement, state)

        assert np.allclose(res, expected)

    @pytest.mark.parametrize(
        "obs, expected",
        [
            (
                qml.Hamiltonian([-0.5, 2], [qml.PauliY(0), qml.PauliZ(0)]),
                0.5 * np.sin(0.123) + 2 * np.cos(0.123),
            ),
            (
                qml.SparseHamiltonian(
                    csr_matrix(-0.5 * qml.PauliY(0).matrix() + 2 * qml.PauliZ(0).matrix()),
                    wires=[0],
                ),
                0.5 * np.sin(0.123) + 2 * np.cos(0.123),
            ),
        ],
    )
    def test_hamiltonian_expval(self, obs, expected):
        """Test that the `measure_hamiltonian_expval` function works correctly."""
        # Create RX(0.123)|0> state
        state = np.array([np.cos(0.123 / 2), -1j * np.sin(0.123 / 2)])
        res = measure(qml.expval(obs), state)
        assert np.allclose(res, expected)

    def test_sum_expval_tensor_contraction(self):
        """Test that `Sum` expectation values are correct when tensor contraction
        is used for computation."""
        summands = (qml.prod(qml.PauliY(i), qml.PauliZ(i + 1)) for i in range(7))
        obs = qml.sum(*summands)
        ops = [qml.RX(0.123, wires=i) for i in range(8)]
        meas = [qml.expval(obs)]
        qs = qml.tape.QuantumScript(ops, meas)

        res = simulate(qs)
        expected = 7 * (-np.sin(0.123) * np.cos(0.123))
        assert np.allclose(res, expected)

    @pytest.mark.parametrize(
        "obs, expected",
        [
            (qml.sum(qml.PauliY(0), qml.PauliZ(0)), -np.sin(0.123) + np.cos(0.123)),
            (
                qml.sum(*(qml.PauliZ(i) for i in range(8))),
                sum(np.sin(i * np.pi / 2 + 0.123) for i in range(8)),
            ),
        ],
    )
    def test_sum_expval_eigs(self, obs, expected):
        """Test that `Sum` expectation values are correct when eigenvalues are used
        for computation."""
        ops = [qml.RX(i * np.pi / 2 + 0.123, wires=i) for i in range(8)]
        meas = [qml.expval(obs)]
        qs = qml.tape.QuantumScript(ops, meas)

        res = simulate(qs)
        assert np.allclose(res, expected)

    def test_correct_measurement_used(self):
        """Test that the correct internal function is used for a given measurement process."""
        # Test a case where state_measurement_process is used
        mp1 = qml.state()
        assert get_measurement_function(mp1) == state_measurement_process

        # Test cases where state_diagonalizing_gates is used
        mp2 = qml.var(qml.PauliZ(0))
        assert get_measurement_function(mp2) == state_diagonalizing_gates
        mp3 = qml.expval(qml.sum(qml.PauliZ(0), qml.PauliX(0)))
        assert get_measurement_function(mp3) == state_diagonalizing_gates
        mp4 = qml.expval(qml.sum(*(qml.PauliZ(i) for i in range(8))))
        assert get_measurement_function(mp4) == state_diagonalizing_gates

        # Test cases where state_hamiltonian_expval is used
        mp5 = qml.expval(qml.Hamiltonian([], []))
        assert get_measurement_function(mp5) is state_hamiltonian_expval
        mp6 = qml.expval(qml.sum(*(qml.prod(qml.PauliY(i), qml.PauliZ(i + 1)) for i in range(7))))
        assert get_measurement_function(mp6) is state_hamiltonian_expval
