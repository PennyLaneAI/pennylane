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
    @pytest.mark.parametrize("p", [0, 0.1, 1])
    def test_kraus_matrices_sum_identity(self, ops, p, tol):
        """Test channels are trace-preserving"""
        if ops.__name__ == "GeneralizedAmplitudeDamping":
            op = ops(p, p, wires=0)
        else:
            op = ops(p, wires=0)
        kmat = op.kraus_matrices
        kdag = [k.conj().T for k in kmat]
        kraus_sum = np.sum(np.array([a @ b for a, b in zip(kdag, kmat)]), axis=0)
        assert np.allclose(kraus_sum, np.eye(2), atol=tol, rtol=0)

    @pytest.mark.parametrize("ops", ch_list)
    @pytest.mark.parametrize("p", [1.5])
    def test_valid_input(self, ops, p):
        """Test input parameters are valid probabilities"""
        if ops.__name__ == "GeneralizedAmplitudeDamping":
            with pytest.raises(ValueError, match="Channel probability parameters should be"):
                ops(0.1, p, wires=0)
        else:
            with pytest.raises(ValueError, match="Channel probability parameters should be"):
                ops(p, wires=0)

class TestAmplitudeDamping:
    """Tests for the quantum channel AmplitudeDamping"""

    def test_gamma_zero(self, tol):
        """Test gamma=0 gives correct Kraus matrices"""
        op = channel.AmplitudeDamping
        assert np.allclose(op(0, wires=0).kraus_matrices[0], np.zeros((2, 2)), atol=tol, rtol=0)
        assert np.allclose(op(0, wires=0).kraus_matrices[1], np.eye(2), atol=tol, rtol=0)

    def test_gamma_arbitrary(self, tol):
        """Test gamma=0.1 gives correct Kraus matrice"""
        op = channel.AmplitudeDamping
        expected = [np.array([[0., 0.31622777],
                              [0., 0.]]),
                    np.array([[1., 0.],
                              [0., 0.9486833]])]
        assert np.allclose(op(0.1, wires=0).kraus_matrices, expected, atol=tol, rtol=0)

class TestGeneralizedAmplitudeDamping:
    """Tests for the quantum channel GeneralizedAmplitudeDamping"""

    def test_gamma_p_zero(self, tol):
        """Test p=0, gamma=0 gives correct Kraus matrices"""
        op = channel.GeneralizedAmplitudeDamping
        assert np.allclose(op(0, 0, wires=0).kraus_matrices[0], np.zeros((2, 2)), atol=tol, rtol=0)
        assert np.allclose(op(0, 0, wires=0).kraus_matrices[2], np.eye(2), atol=tol, rtol=0)

    def test_gamma_p_arbitrary(self, tol):
        """Test p=0.1, gamma=0.1 gives correct first Kraus matrix"""
        op = channel.GeneralizedAmplitudeDamping
        expected = np.array([[0.31622777, 0.],
                            [0., 0.3]])
        assert np.allclose(op(0.1, 0.1, wires=0).kraus_matrices[0], expected, atol=tol, rtol=0)

class TestPhaseDamping:
    """Tests for the quantum channel PhaseDamping"""

    def test_gamma_zero(self, tol):
        """Test gamma=0 gives correct Kraus matrices"""
        op = channel.PhaseDamping
        assert np.allclose(op(0, wires=0).kraus_matrices[0], np.zeros((2, 2)), atol=tol, rtol=0)
        assert np.allclose(op(0, wires=0).kraus_matrices[1], np.eye(2), atol=tol, rtol=0)

    def test_gamma_arbitrary(self, tol):
        """Test gamma=0.1 gives correct Kraus matrice"""
        op = channel.PhaseDamping
        expected = [np.array([[0., 0.],
                              [0., 0.31622777]]),
                    np.array([[1., 0.],
                              [0., 0.9486833]])]
        assert np.allclose(op(0.1, wires=0).kraus_matrices, expected, atol=tol, rtol=0)

class TestDepolarizingChannel:
    """Tests for the quantum channel DepolarizingChannel"""

    def test_p_zero(self, tol):
        """Test p=0 gives correct Kraus matrices"""
        op = channel.DepolarizingChannel
        assert np.allclose(op(0, wires=0).kraus_matrices[0], np.eye(2), atol=tol, rtol=0)
        assert np.allclose(op(0, wires=0).kraus_matrices[1], np.zeros((2, 2)), atol=tol, rtol=0)

    def test_p_arbitrary(self, tol):
        """Test p=0.1 gives correct Kraus matrices"""
        p = 0.1
        op = channel.DepolarizingChannel
        expected = np.sqrt(p/3) * X
        assert np.allclose(op(0.1, wires=0).kraus_matrices[1], expected, atol=tol, rtol=0)
