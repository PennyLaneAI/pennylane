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
from pennylane.wires import Wires

Y = np.array([[0, -1j], [1j, 0]])


class TestQubitChannel:
    """Tests for the quantum channel QubitChannel"""

    def test_input_correctly_handled(self, tol):
        """Test that Kraus matrices are correctly processed"""
        K_list1 = [
            np.array([[1.0, 0.0], [0.0, 0.9486833]]),
            np.array([[0.0, 0.31622777], [0.0, 0.0]]),
        ]
        out = channel.QubitChannel(K_list1, wires=0).kraus_matrices

        # verify equivalent to input state
        assert np.allclose(out, K_list1, atol=tol, rtol=0)

    def test_kraus_matrices_valid(self):
        """Tests that the given Kraus matrices are valid"""

        # check all Kraus matrices are square matrices
        K_list1 = [np.zeros((2, 2)), np.zeros((2, 3))]
        with pytest.raises(
            ValueError, match="Only channels with similar input and output Hilbert space"
        ):
            channel.QubitChannel(K_list1, wires=0).kraus_matrices

        # check all Kraus matrices have the same shape
        K_list2 = [np.eye(2), np.eye(4)]
        with pytest.raises(ValueError, match="All Kraus matrices must have the same shape."):
            channel.QubitChannel(K_list2, wires=0).kraus_matrices

        # check the dimension of all Kraus matrices are valid
        K_list3 = [np.array([np.eye(2), np.eye(2)]), np.array([np.eye(2), np.eye(2)])]
        with pytest.raises(ValueError, match="Dimension of all Kraus matrices must be "):
            channel.QubitChannel(K_list3, wires=0).kraus_matrices

    def test_channel_trace_preserving(self):
        """Tests that the channel represents a trace-preserving map"""

        # real Kraus matrices
        K_list1 = [
            np.array([[1.0, 0.0], [0.0, 0.9486833]]),
            np.array([[0.0, 0.31622777], [0.0, 0.0]]),
        ]
        with pytest.raises(ValueError, match="Only trace preserving channels can be applied."):
            channel.QubitChannel(K_list1 * 2, wires=0).kraus_matrices

        # complex Kraus matrices
        p = 0.1
        K_list2 = [np.sqrt(p) * Y, np.sqrt(1 - p) * np.eye(2)]
        with pytest.raises(ValueError, match="Only trace preserving channels can be applied."):
            channel.QubitChannel(K_list2 * 2, wires=0).kraus_matrices
