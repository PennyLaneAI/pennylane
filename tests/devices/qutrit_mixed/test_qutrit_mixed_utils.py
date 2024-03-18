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
"""Unit tests for util functions in devices/qutrit_mixed."""
import pytest

import pennylane as qml
from pennylane import numpy as np
from pennylane.devices.qutrit_mixed.utils import get_eigvals, expand_qutrit_vector


@pytest.mark.parametrize(
    "observable, expected_eigvals",
    [
        (qml.GellMann(0, 8), (1 / 3, -np.sqrt(2) / 3)),
        (qml.s_prod(3, qml.GellMann(0, 8)), (1, -np.sqrt(2))),
        (
            qml.prod(qml.GellMann(0, 8), qml.GellMann(1, 1)),
            (0, 1 / 3, -np.sqrt(2) / 3, np.sqrt(2) / 3, 2 / 9),
        ),
        (
            qml.prod(qml.GellMann(0, 8), qml.GellMann(0, 1), qml.THadamard(1), qml.GellMann(0, 3)),
            (0, 1 / 3, -np.sqrt(2) / 3, np.sqrt(2) / 3, 2 / 9),
        ),
        (
            qml.s_prod(3, qml.prod(qml.GellMann(0, 8), qml.GellMann(1, 1))),
            (0, 1, -np.sqrt(2), np.sqrt(2), 2 / 3),
        ),
    ],
)
def get_obs_eigvals(observable, expected_eigvals):
    get_eigvals(observable)
    assert np.allclose(observable, expected_eigvals)


class TestExpandQutritVector:
    """Tests vector expansion to more wires, for qutrit case"""

    w = np.exp(2j * np.pi / 3)
    VECTOR1 = np.array([1, w, w**2])
    ONES = np.array([1, 1, 1])

    @pytest.mark.parametrize(
        "original_wires,expanded_wires,expected",
        [
            ([0], 3, np.kron(np.kron(VECTOR1, ONES), ONES)),
            ([1], 3, np.kron(np.kron(ONES, VECTOR1), ONES)),
            ([2], 3, np.kron(np.kron(ONES, ONES), VECTOR1)),
            ([0], [0, 4, 7], np.kron(np.kron(VECTOR1, ONES), ONES)),
            ([4], [0, 4, 7], np.kron(np.kron(ONES, VECTOR1), ONES)),
            ([7], [0, 4, 7], np.kron(np.kron(ONES, ONES), VECTOR1)),
            ([0], [0, 4, 7], np.kron(np.kron(VECTOR1, ONES), ONES)),
            ([4], [4, 0, 7], np.kron(np.kron(VECTOR1, ONES), ONES)),
            ([7], [7, 4, 0], np.kron(np.kron(VECTOR1, ONES), ONES)),
        ],
    )
    def test_expand_vector_single_wire(self, original_wires, expanded_wires, expected, tol):
        """Test that expand_vector works with a single-wire vector."""

        res = expand_qutrit_vector(TestExpandQutritVector.VECTOR1, original_wires, expanded_wires)

        assert np.allclose(res, expected, atol=tol, rtol=0)

    VECTOR2 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

    @pytest.mark.parametrize(
        "original_wires,expanded_wires,expected",
        [
            ([0, 1], 3, np.kron(VECTOR2, ONES)),
            ([1, 2], 3, np.kron(ONES, VECTOR2)),
            ([0, 2], 3, np.array(([1, 2, 3] * 3) + ([4, 5, 6] * 3) + ([7, 8, 9] * 3))),
            ([0, 5], [0, 5, 9], np.kron(VECTOR2, ONES)),
            ([5, 9], [0, 5, 9], np.kron(ONES, VECTOR2)),
            ([0, 9], [0, 5, 9], np.array(([1, 2, 3] * 3) + ([4, 5, 6] * 3) + ([7, 8, 9] * 3))),
            ([9, 0], [0, 5, 9], np.array(([1, 4, 7] * 3) + ([2, 5, 8] * 3) + ([3, 6, 9] * 3))),
            ([0, 1], [0, 1], VECTOR2),
        ],
    )
    def test_expand_vector_two_wires(self, original_wires, expanded_wires, expected, tol):
        """Test that expand_vector works with a single-wire vector."""

        res = expand_qutrit_vector(TestExpandQutritVector.VECTOR2, original_wires, expanded_wires)

        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_expand_vector_invalid_wires(self):
        """Test exception raised if unphysical subsystems provided."""
        with pytest.raises(
            ValueError,
            match="Invalid target subsystems provided in 'original_wires' argument",
        ):
            expand_qutrit_vector(TestExpandQutritVector.VECTOR2, [-1, 5], 4)

    def test_expand_vector_invalid_vector(self):
        """Test exception raised if incorrect sized vector provided."""
        with pytest.raises(ValueError, match="Vector parameter must be of length"):
            expand_qutrit_vector(TestExpandQutritVector.VECTOR1, [0, 1], 4)
