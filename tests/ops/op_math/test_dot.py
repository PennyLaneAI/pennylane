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
"""
Unit tests for the dot function
"""
import pytest

import pennylane as qml
from pennylane.ops import SProd, Sum, dot
from pennylane.pauli.pauli_arithmetic import PauliSentence


class TestDot:
    """Unittests for the dot function."""

    def test_dot_returns_pauli_sentence(self):
        """Test that the dot function returns a PauliSentence class when ``pauli=True``."""
        coeffs = [1.0, 2.0, 3.0]
        ops = [qml.PauliX(0), qml.PauliY(1), qml.PauliZ(2)]
        ps = dot(coeffs, ops, pauli=True)
        assert isinstance(ps, PauliSentence)

    def test_dot_returns_sum(self):
        """Test that the dot function returns a Sum operator when ``pauli=False``."""
        coeffs = [1.0, 2.0, 3.0]
        ops = [qml.PauliX(0), qml.PauliY(1), qml.PauliZ(2)]
        S = dot(coeffs, ops)
        assert isinstance(S, Sum)
        for summand, coeff, op in zip(S.operands, coeffs, ops):
            if coeff != 1:
                assert isinstance(summand, SProd)
                assert summand.scalar == coeff
            else:
                assert isinstance(summand, type(op))

    def test_dot_returns_sprod(self):
        """Test that the dot function returns a SProd operator when only one operator is input."""
        coeffs = [2.0]
        ops = [qml.PauliX(0)]
        O = dot(coeffs, ops)
        assert isinstance(O, SProd)
        assert O.scalar == 2

    def test_dot_different_number_of_coeffs_and_ops(self):
        """Test that a ValueError is raised when the number of coefficients and operators does
        not match."""
        with pytest.raises(
            ValueError,
            match="Number of coefficients and operators does not match",
        ):
            dot([1.0], [qml.PauliX(0), qml.PauliY(1)])

    def test_dot_empty_coeffs_or_ops(self):
        """Test that a ValueError is raised when the number of coefficients and operators does
        not match."""
        with pytest.raises(
            ValueError,
            match="Cannot compute the dot product of an empty sequence",
        ):
            dot([], [])
