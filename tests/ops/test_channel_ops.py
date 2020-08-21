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
Unit tests for the available built-in quantum channels.
"""
import pytest
import functools
import numpy as np
from numpy.linalg import multi_dot

import pennylane as qml
from pennylane.ops import channel

ch_list = [channel.AmplitudeDamping,
           channel.GeneralizedAmplitudeDamping,
           channel.PhaseDamping,
           channel.DepolarizingChannel,
           ]

X = np.array([[0, 1], [1, 0]])

class TestChannels:
    """Tests for the quantum channels"""

    @pytest.mark.parametrize("ops", ch_list)
    def test_kraus_matrices_sum_identity(self, ops, tol):
        """Test channels are trace-preserving"""
        if ops.__name__ == "GeneralizedAmplitudeDamping":
            op = ops(0.1, 0.1, wires=0)
        else:
            op = ops(0.1, wires=0)
        km = op.kraus_matrices
        kd = [k.conj().T for k in km]
        assert np.allclose(np.sum(np.array([a @ b for a, b in zip(kd, km)]), axis=0), np.eye(2), atol=tol, rtol=0)

    @pytest.mark.parametrize("ops", ch_list)
    def test_probabilities_valid(self, ops):
        if ops.__name__ == "GeneralizedAmplitudeDamping":
            with pytest.raises(ValueError, match="Probabilities of noisy processes should not exceed 1."):
                ops(0.1, 1.1, wires=0)

        with pytest.raises(ValueError, match="Probabilities of noisy processes should not exceed 1."):
            ops(1.1, wires=0)

    def test_amplitude_damping(self, tol):
        """Tests for amplitude damping channel"""

        # test gamma=0 gives correct Kraus matrices
        op = channel.AmplitudeDamping
        assert np.allclose(op(0, wires=0).kraus_matrices[0], np.zeros((2, 2)), atol=tol, rtol=0)
        assert np.allclose(op(0, wires=0).kraus_matrices[1], np.eye(2), atol=tol, rtol=0)

        # test gamma=0.1 gives correct Kraus matrices
        expected = [np.array([[0., 0.31622777],
                              [0., 0.]]),
                    np.array([[1., 0.],
                              [0., 0.9486833]])]
        assert np.allclose(op(0.1, wires=0).kraus_matrices, expected, atol=tol, rtol=0)

    def test_generalized_amplitude_damping(self, tol):
        """Tests for generalized amplitude damping channel"""

        # test p=0, gamma=0 gives correct Kraus matrices
        op = channel.GeneralizedAmplitudeDamping
        assert np.allclose(op(0, 0, wires=0).kraus_matrices[0], np.zeros((2, 2)), atol=tol, rtol=0)
        assert np.allclose(op(0, 0, wires=0).kraus_matrices[2], np.eye(2), atol=tol, rtol=0)

        # test p=0.1, gamma=0.1 gives correct first Kraus matrix
        expected = np.array([[0.31622777, 0.],
                             [0., 0.3]])
        assert np.allclose(op(0.1, 0.1, wires=0).kraus_matrices[0], expected, atol=tol, rtol=0)

    def test_phase_damping(self, tol):
        """Tests for phase damping channel"""

        # test gamma=0 gives correct Kraus matrices
        op = channel.PhaseDamping
        assert np.allclose(op(0, wires=0).kraus_matrices[0], np.zeros((2, 2)), atol=tol, rtol=0)
        assert np.allclose(op(0, wires=0).kraus_matrices[1], np.eye(2), atol=tol, rtol=0)

        # test gamma=0.1 gives correct Kraus matrices
        expected = [np.array([[0., 0.],
                              [0., 0.31622777]]),
                    np.array([[1., 0.],
                              [0., 0.9486833]])]
        assert np.allclose(op(0.1, wires=0).kraus_matrices, expected, atol=tol, rtol=0)

    def test_depolarization(self, tol):
        """Tests for symmetrically depolarizing channel"""

        # test p=0 gives correct Kraus matrices
        op = channel.DepolarizingChannel
        assert np.allclose(op(0, wires=0).kraus_matrices[1], np.zeros((2, 2)), atol=tol, rtol=0)
        assert np.allclose(op(0, wires=0).kraus_matrices[0], np.eye(2), atol=tol, rtol=0)

        # test p=0.1 gives correct Kraus matrices
        p = 0.1
        expected = np.sqrt(p/3) * X
        assert np.allclose(op(0.1, wires=0).kraus_matrices[1], expected, atol=tol, rtol=0)
