# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for pennylane/labs/dla/lie_closure_dense.py functionality"""
# pylint: disable=too-few-public-methods, protected-access, no-self-use
import pytest

import pennylane as qml
from pennylane import X, Y, Z
from pennylane.labs.dla import AI, AII


class TestMatrixConstructors:
    """Tests for the matrix constructing methods used in Cartan involutions."""


class TestInvolutions:
    """Test the involutions themselves."""

    @pytest.mark.parametrize(
        "op, expected",
        [
            (X(0) @ Y(1) @ Z(2), True),
            (X(0) @ Y(1) @ Z(2) - Z(0) @ X(1) @ Y(2), True),
            (X(0) @ Y(1) @ Z(2) - Y(0) @ Y(1) @ Y(2), True),
            (Y(0) @ Y(1) @ Z(2), False),
            (X(0) @ X(1) @ Z(2) + X(0) @ Y(1) @ Y(2), False),
        ],
    )
    def test_AI(self, op, expected):
        """Test singledispatch for AI involution"""
        inputs = [op, op.pauli_rep, qml.matrix(op, wire_order=[0, 1, 2])]
        outputs = [AI(_input) for _input in inputs]
        if expected:
            assert all(outputs)
        else:
            assert not any(outputs)

    @pytest.mark.parametrize(
        "op, expected",
        [  # (#_Y is odd, I/Y on first wire)
            (X(0) @ Y(1) @ Z(2), False),  # (True, False) -> -1
            (X(0) @ Y(1) @ Z(2) - Z(0) @ X(1) @ Y(2), False),  # (True, False) -> -1
            (Y(0) @ X(1) @ Z(2) - Y(0) @ Y(1) @ Y(2), True),  # (True, True) -> 1
            (Y(0) @ Y(1) @ Z(2), False),  # (False, True) -> -1
            (X(0) @ X(1) @ Z(2) + X(0) @ Y(1) @ Y(2), True),  # (False, False) -> 1
        ],
    )
    def test_AII(self, op, expected):
        """Test singledispatch for AI involution"""
        inputs = [op, op.pauli_rep, qml.matrix(op, wire_order=[0, 1, 2])]
        outputs = [AII(_input) for _input in inputs]
        if expected:
            assert all(outputs)
        else:
            assert not any(outputs)
