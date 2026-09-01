# Copyright 2026 Xanadu Quantum Technologies Inc.

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
Tests for the HammingFour template.
"""

import numpy as np
import pytest

import pennylane as qp
from pennylane.labs.templates import HammingFour
from pennylane.ops.functions.assert_valid import assert_valid

# Interleaved labels, so that a decomposition acting on the wrong wires does not go unnoticed.
INPUT_WIRES = [0, 2, 4, 6]
OUTPUT_WIRES = [1, 3, 5, 7, 8]
ALL_WIRES = INPUT_WIRES + OUTPUT_WIRES

# Work wires for a QROM loading the same table as HammingFour.
QROM_WORK_WIRES = [9, 10, 11]


def _expected_output(x):
    """The five output bits ``(w0, w1, w2, t, k)`` for the four-bit input ``x``."""
    i0, i1, i2, i3 = (int(b) for b in format(x, "04b"))
    weight = i0 + i1 + i2 + i3
    w0, w1, w2 = ((weight >> i) & 1 for i in range(3))
    return w0, w1, w2, w0 & w1, (i0 ^ i1 ^ i2) & i3


def _expected_basis_index(x):
    """Index of the expected output basis state, in the ``ALL_WIRES`` order."""
    index = x
    for bit in _expected_output(x):
        index = (index << 1) | bit
    return index


# The table that HammingFour computes, in the index order expected by QROM.
BITSTRINGS = [_expected_output(x) for x in range(16)]


class TestBasicValidity:
    """Test basic validity of the HammingFour template."""

    def test_standard_validity(self):
        """Check the operation using the assert_valid function."""
        op = HammingFour(INPUT_WIRES, OUTPUT_WIRES)
        assert_valid(op)

        assert op.hyperparameters["input_wires"] == qp.wires.Wires(INPUT_WIRES)
        assert op.hyperparameters["output_wires"] == qp.wires.Wires(OUTPUT_WIRES)
        assert op.wires == qp.wires.Wires(ALL_WIRES)
        assert not op.resource_params

    @pytest.mark.parametrize("num_input_wires", [3, 5])
    def test_wrong_number_of_input_wires(self, num_input_wires):
        """Test that exactly four input wires are required."""
        with pytest.raises(ValueError, match="Expected four input"):
            HammingFour(list(range(num_input_wires)), OUTPUT_WIRES)

    @pytest.mark.parametrize("num_output_wires", [4, 6])
    def test_wrong_number_of_output_wires(self, num_output_wires):
        """Test that exactly five output wires are required."""
        with pytest.raises(ValueError, match="Expected five output"):
            HammingFour(INPUT_WIRES, list(range(10, 10 + num_output_wires)))


class TestHammingFour:
    """Test the HammingFour template."""

    @pytest.mark.parametrize("x", range(16))
    def test_operation_result(self, x):
        """Test that the Hamming weight, the product bit and the cache bit are computed."""

        @qp.qnode(qp.device("default.qubit", wires=9))
        def circuit():
            qp.BasisState(qp.math.int_to_binary(x, 4), wires=INPUT_WIRES)
            HammingFour(INPUT_WIRES, OUTPUT_WIRES)
            return qp.probs(wires=ALL_WIRES)

        probs = np.asarray(circuit())
        expected = np.zeros(2 ** len(ALL_WIRES))
        expected[_expected_basis_index(x)] = 1.0
        assert np.allclose(probs, expected)

    def test_no_phase_errors(self):
        """Test that the template acts as a real permutation on zeroed output wires.

        ``TemporaryAND`` only reduces to a ``Toffoli`` if its target wire is zeroed, so a
        misplaced elbow would show up as a complex phase on the computed basis state."""
        matrix = np.asarray(qp.matrix(HammingFour(INPUT_WIRES, OUTPUT_WIRES), wire_order=ALL_WIRES))

        # The output wires are the last five in ``ALL_WIRES``, so zeroing them means x << 5.
        columns = [x << len(OUTPUT_WIRES) for x in range(16)]
        rows = [_expected_basis_index(x) for x in range(16)]
        assert np.allclose(matrix[rows, columns], 1.0)

    def test_uncomputation(self):
        """Test that the adjoint uncomputes a QROM that loads the same table."""

        # The device wire order fixes the axis order of the returned state.
        @qp.qnode(qp.device("default.qubit", wires=ALL_WIRES + QROM_WORK_WIRES))
        def circuit(input_state):
            qp.StatePrep(input_state, wires=INPUT_WIRES)
            qp.QROM(BITSTRINGS, INPUT_WIRES, OUTPUT_WIRES, QROM_WORK_WIRES)
            qp.adjoint(HammingFour(INPUT_WIRES, OUTPUT_WIRES))
            return qp.state()

        rng = np.random.default_rng(42)
        input_state = rng.normal(size=16) + 1j * rng.normal(size=16)
        input_state /= np.linalg.norm(input_state)

        state = np.asarray(circuit(input_state)).reshape((16, -1))
        # The output and QROM work registers are returned to zero, and the input is unchanged.
        assert np.allclose(state[:, 0], input_state)
        assert np.allclose(state[:, 1:], 0.0)
