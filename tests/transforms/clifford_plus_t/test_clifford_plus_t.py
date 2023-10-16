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
"""Test the Clifford+T transform."""

import numpy as np
import pytest

import pennylane as qml
from pennylane.transforms.decompositions.clifford_plus_t.synthesis import (
    theta_to_gates,
    gridsynth,
    synthesis_nqubit,
    to_gates,
)

OP_MAP = {"H": qml.Hadamard, "S": qml.S, "T": qml.T, "X": qml.PauliX, "Z": qml.PauliZ}


def convert_to_operators(gate_string, wire):
    """Helper function to convert a list of strings to a list of operators."""
    return [OP_MAP[i](wire) for i in reversed(gate_string)]


class TestRZDecomposition:
    """Test that RZ rotations are correctly decomposed."""

    @staticmethod
    def assert_decomposition_close(theta, epsilon):
        """Compute the actual RZ matrix, compare with the decomposition matrix."""
        expected = qml.RZ.compute_matrix(theta)
        actual_str = theta_to_gates(theta, epsilon)
        actual_gates = convert_to_operators(actual_str, 0)
        actual = qml.matrix(qml.tape.QuantumScript(actual_gates))
        assert qml.math.allclose(actual, expected, atol=epsilon)

    @pytest.mark.parametrize("theta", [0.25, np.pi / 4, 1.2345])
    @pytest.mark.parametrize("epsilon", [1e-4, 1e-5, 1e-6])
    def test_basic_values(self, theta, epsilon):
        """Test some reasonable angles and error values."""
        self.assert_decomposition_close(theta, epsilon)

    @pytest.mark.skip(reason="gridsynth hangs forever on specific values")
    @pytest.mark.parametrize("theta,epsilon", [(0, 1e-6), (np.pi / 4, 1e-7)])
    def test_values_that_never_return(self, theta, epsilon):
        """Test some values that hang without tweaking."""
        self.assert_decomposition_close(theta, epsilon)

    @pytest.mark.parametrize(
        "theta,epsilon",
        [
            (0, 1e-7),
            (1.2345, 1e-7),
            (np.pi / 150, 1e-8),
        ],
    )
    def test_harder_values(self, theta, epsilon):
        """Test some small values with larger epsilon values."""
        self.assert_decomposition_close(theta, epsilon)

    @pytest.mark.xfail(reason="ellipse operators have negative determinant, sqrt returns Nan")
    def test_faulty_values_for_sqrt(self):
        """Xfail a test for certain angles and bounds that result in invalid ellipses."""
        self.assert_decomposition_close(np.pi / 128, 1e-10)


@pytest.mark.parametrize("theta", [0.25, np.pi / 4, 1.2345])
@pytest.mark.parametrize("epsilon", [1e-4, 1e-5, 1e-6])
def test_compare_gridsynth_to_decomp(theta, epsilon):
    """Test that the decomposition is identical to gridsynth's matrix."""
    grid_mat, _, _ = gridsynth(epsilon, theta)
    synth = synthesis_nqubit(grid_mat)
    synth_str = to_gates(synth)
    synth_gates = convert_to_operators(synth_str, 0)
    synth_mat = qml.matrix(qml.tape.QuantumScript(synth_gates))
    assert qml.math.allclose(grid_mat.astype(complex), synth_mat, atol=1e-13)
