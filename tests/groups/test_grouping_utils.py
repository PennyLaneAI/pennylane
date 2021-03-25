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
Unit tests for the :mod:`grouping` utility functions in ``groups/grouping_utils.py``.
"""
import pytest
import numpy as np
from pennylane import Identity, PauliX, PauliY, PauliZ
from pennylane.groups.pauli_utils import pauli_to_binary
from pennylane.groups.grouping_utils import (
    is_qwc,
    observables_to_binary_matrix,
    qwc_complement_adj_matrix,
)


class TestGroupingUtils:
    """Basic usage and edge-case tests for the measurement optimization utility functions."""

    def test_is_qwc(self):
        """Determining if two Pauli words are qubit-wise commuting."""

        wire_map = {0: 0, "a": 1, "b": 2}
        p1_vec = pauli_to_binary(PauliX(0) @ PauliY("a"), wire_map=wire_map)
        p2_vec = pauli_to_binary(PauliX(0) @ Identity("a") @ PauliX("b"), wire_map=wire_map)
        p3_vec = pauli_to_binary(PauliX(0) @ PauliZ("a") @ Identity("b"), wire_map=wire_map)
        identity = pauli_to_binary(Identity("a") @ Identity(0), wire_map=wire_map)

        assert is_qwc(p1_vec, p2_vec)
        assert not is_qwc(p1_vec, p3_vec)
        assert is_qwc(p2_vec, p3_vec)
        assert (
            is_qwc(p1_vec, identity)
            == is_qwc(p2_vec, identity)
            == is_qwc(p3_vec, identity)
            == is_qwc(identity, identity)
            == True
        )

    def test_is_qwc_not_equal_lengths(self):
        """Tests ValueError is raised when input Pauli vectors are not of equal length."""

        pauli_vec_1 = [0, 1, 0, 1]
        pauli_vec_2 = [1, 1, 0, 1, 0, 1]

        assert pytest.raises(ValueError, is_qwc, pauli_vec_1, pauli_vec_2)

    def test_is_qwc_not_even_lengths(self):
        """Tests ValueError is raised when input Pauli vectors are not of even length."""

        pauli_vec_1 = [1, 0, 1]
        pauli_vec_2 = [1, 1, 1]

        assert pytest.raises(ValueError, is_qwc, pauli_vec_1, pauli_vec_2)

    def test_is_qwc_not_binary_vectors(self):
        """Tests ValueError is raised when input Pauli vectors do not have binary
        components."""

        pauli_vec_1 = [1, 3.2, 1, 1 + 2j]
        pauli_vec_2 = [1, 0, 0, 0]

        assert pytest.raises(ValueError, is_qwc, pauli_vec_1, pauli_vec_2)

    def test_qwc_complement_adj_matrix(self):
        """Tests that the ``qwc_complement_adj_matrix`` function returns the correct
        adjacency matrix."""
        binary_observables = np.array(
            [
                [1.0, 0.0, 1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 1.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            ]
        )
        adj = qwc_complement_adj_matrix(binary_observables)

        expected = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

        assert np.all(adj == expected)

        binary_obs_list = list(binary_observables)
        adj = qwc_complement_adj_matrix(binary_obs_list)
        assert np.all(adj == expected)

        binary_obs_tuple = tuple(binary_observables)
        adj = qwc_complement_adj_matrix(binary_obs_tuple)
        assert np.all(adj == expected)

    def test_qwc_complement_adj_matrix_exception(self):
        """Tests that the ``qwc_complement_adj_matrix`` function raises an exception if
        the matrix is not binary."""
        not_binary_observables = np.array(
            [
                [1.1, 0.5, 1.0, 0.0, 0.0, 1.0],
                [0.0, 1.3, 1.0, 1.0, 0.0, 1.0],
                [2.2, 0.0, 0.0, 1.0, 0.0, 0.0],
            ]
        )

        with pytest.raises(ValueError, match="Expected a binary array, instead got"):
            qwc_complement_adj_matrix(not_binary_observables)
