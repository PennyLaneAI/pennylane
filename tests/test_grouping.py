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
Unit tests for the :mod:`grouping` measurement optimization and utility functions.
"""
import pytest
import numpy as np
import pennylane as qml
from pennylane import Identity, PauliX, PauliY, PauliZ, Hadamard
from pennylane.operation import Tensor
from grouping.group_observables import PauliGroupingStrategy, group_observables
from grouping.graph_colouring import largest_first, recursive_largest_first
from grouping.optimize_measurements import optimize_measurements
from grouping.utils import (get_n_qubits,
                            is_pauli_word,
                            are_identical_pauli_words,
                            binary_symplectic_inner_prod,
                            pauli_to_binary,
                            binary_to_pauli,
                            is_qwc,
                            convert_observables_to_binary)

class TestGroupingUtils:
    """Basic usage and edge-case tests for the measurement optimization utility functions."""

    def test_pauli_to_binary(self):

        n_qubits = 3

        P1 = PauliX(0) @ PauliY(1) @ PauliZ(2)
        P2 = PauliZ(0) @ PauliY(2)
        P3 = PauliY(1) @ PauliX(2)
        identity = Tensor(Identity(0))

        P1_vec = np.array([1, 1, 0, 0, 1, 1])
        P2_vec = np.array([0, 0, 1, 1, 0, 1])
        P3_vec = np.array([0, 1, 1, 0 ,1, 0])
        zero_vec = np.zeros(2*n_qubits)

        assert all(pauli_to_binary(P1, n_qubits) == P1_vec)
        assert all(pauli_to_binary(P2, n_qubits) == P2_vec)
        assert all(pauli_to_binary(P3, n_qubits) == P3_vec)
        assert all(pauli_to_binary(identity, n_qubits) == zero_vec)

    def test_binary_to_pauli(self):

        P1_vec = np.array([1, 0, 1, 0, 0, 1])
        P2_vec = np.array([1, 1, 1, 1, 1, 1])
        P3_vec = np.array([1, 0, 1, 0, 1, 1])
        zero_vec = np.zeros(6)

        P1 = PauliX(0) @ PauliY(2)
        P2 = PauliY(0) @ PauliY(1) @ PauliY(2)
        P3 = PauliX(0) @ PauliZ(1) @ PauliY(2)
        identity = Tensor(Identity(0))

        assert are_identical_pauli_words(binary_to_pauli(P1_vec), P1)
        assert are_identical_pauli_words(binary_to_pauli(P2_vec), P2)
        assert are_identical_pauli_words(binary_to_pauli(P3_vec), P3)
        assert are_identical_pauli_words(binary_to_pauli(zero_vec), identity)

    def test_convert_observables_to_binary(self):

        observables = [Identity(1) , PauliX(1), PauliZ(0) @ PauliZ(1)]

        binary_observables = np.array([[0, 0, 0],
                                       [0, 1, 0],
                                       [0, 0, 1],
                                       [0, 0, 1]])

        assert (convert_observables_to_binary(observables) == binary_observables).all()

    def test_is_qwc(self):

        n_qubits = 2
        P1_vec = pauli_to_binary(PauliX(0) @ PauliY(1), n_qubits)
        P2_vec = pauli_to_binary(PauliX(0), n_qubits)
        P3_vec = pauli_to_binary(PauliX(0) @ PauliZ(1), n_qubits)

        assert is_qwc(P1_vec, P2_vec) == True
        assert is_qwc(P1_vec, P3_vec) == False
        assert is_qwc(P2_vec, P3_vec) == True

    def test_binary_symplectic_inner_prod(self):

        P1_vec = np.array([1, 0, 1, 0, 1, 1, 0, 0])
        P2_vec = np.array([0, 0, 1, 1, 1, 0, 1, 0])
        P3_vec = np.array([1, 1, 0, 1, 1, 1, 0, 1])

        assert binary_symplectic_inner_prod(P1_vec, P2_vec) == 0
        assert binary_symplectic_inner_prod(P1_vec, P3_vec) == 1
        assert binary_symplectic_inner_prod(P2_vec, P3_vec) == 0

    def test_get_n_qubits(self):

        observables_1 = [Identity(10)]
        observables_2 = [Identity(0) @ Identity(10), PauliX(2) @ Identity(10)]

        assert get_n_qubits(observables_1) == 1
        assert get_n_qubits(observables_2) == 3

    def test_is_pauli_word(self):

        observable_1 = PauliX(0)
        observable_2 = PauliZ(1) @ PauliX(2) @ PauliZ(4)
        observable_3 = PauliX(1) @ Hadamard(4)
        observable_4 = Hadamard(0)

        assert is_pauli_word(observable_1)
        assert is_pauli_word(observable_2)
        assert not is_pauli_word(observable_3)
        assert not is_pauli_word(observable_4)

    def test_are_identical_pauli_words(self):
        1

class TestPauliGroupingStrategy:
    """Tests for the PauliGroupingStrategy class"""
    def test_construct_complement_adj_matrix_for_operator(self):

        observables = [PauliY(0), PauliZ(0) @ PauliZ(1), PauliY(0) @ PauliX(1)]

        qwc_complement_adjacency_matrix = np.array([[0, 1, 0],
                                                    [1, 0, 1],
                                                    [0, 1, 0]])

        commuting_complement_adjacency_matrix = np.array([[0, 1, 0],
                                                          [1, 0, 0],
                                                          [0, 0, 0]])

        anticommuting_complement_adjacency_matrix = np.array([[0, 0, 1],
                                                              [0, 0, 1],
                                                              [1, 1, 0]])

        grouping_instance = PauliGroupingStrategy(observables, 'qwc')
        assert (grouping_instance.construct_complement_adj_matrix_for_operator() == qwc_complement_adjacency_matrix).all()
        grouping_instance = PauliGroupingStrategy(observables, 'commuting')
        assert (grouping_instance.construct_complement_adj_matrix_for_operator() == commuting_complement_adjacency_matrix).all()
        grouping_instance = PauliGroupingStrategy(observables, 'anticommuting')
        assert (grouping_instance.construct_complement_adj_matrix_for_operator() == anticommuting_complement_adjacency_matrix).all()

    def test_construct_complement_adj_matrix_for_trivial_operators(self):

        observables = [Identity(0), Identity(0), Identity(7)]

        qwc_complement_adjacency_matrix = np.array([[0, 0, 0],
                                                    [0, 0, 0],
                                                    [0, 0, 0]])

        commuting_complement_adjacency_matrix = np.array([[0, 0, 0],
                                                          [0, 0, 0],
                                                          [0, 0, 0]])

        anticommuting_complement_adjacency_matrix = np.array([[0, 1, 1],
                                                              [1, 0, 1],
                                                              [1, 1, 0]])

        grouping_instance = PauliGroupingStrategy(observables, 'qwc')
        assert (grouping_instance.construct_complement_adj_matrix_for_operator() == qwc_complement_adjacency_matrix).all()
        grouping_instance = PauliGroupingStrategy(observables, 'commuting')
        assert (grouping_instance.construct_complement_adj_matrix_for_operator() == commuting_complement_adjacency_matrix).all()
        grouping_instance = PauliGroupingStrategy(observables, 'anticommuting')
        assert (grouping_instance.construct_complement_adj_matrix_for_operator() == anticommuting_complement_adjacency_matrix).all()

class TestGraphcolouringFunctions:
    '''Tests for graph colouring functions.'''

    def verify_graph_colour_solution(self, adjacency_matrix, colouring):
        '''Verifies if all vertices of the same colour are not connected.'''

        for colour in colouring.keys():

            grouping = colouring[colour]
            size_grouping = len(grouping)

            for i in range(size_grouping):
                for j in range(i+1, size_grouping):
                    vert_i = grouping[i][0]
                    vert_j = grouping[j][0]

                    if adjacency_matrix[vert_i][vert_j] == 1:
                        return False

        return True

    def test_graph_colouring(self):

        n = 8
        adjacency_matrix = np.random.randint(2, size=(n, n))
        np.fill_diagonal(adjacency_matrix, 0)

        for i in range(n):
            for j in range(i+1, n):
                adjacency_matrix[j, i] = adjacency_matrix[i, j]

        lf_colouring = largest_first(np.asarray([list(range(n))]), adjacency_matrix)
        rlf_colouring = recursive_largest_first(np.asarray([list(range(n))]), adjacency_matrix)

        assert self.verify_graph_colour_solution(adjacency_matrix, lf_colouring)
        assert self.verify_graph_colour_solution(adjacency_matrix, rlf_colouring)

    def test_trivial_graph_colouring(self):

        n = 8
        adjacency_matrix = np.zeros((n,n))

        lf_colouring = largest_first(np.asarray([list(range(n))]), adjacency_matrix)
        rlf_colouring = recursive_largest_first(np.asarray([list(range(n))]), adjacency_matrix)

        assert self.verify_graph_colour_solution(adjacency_matrix, lf_colouring)
        assert self.verify_graph_colour_solution(adjacency_matrix, rlf_colouring)

class TestOptimizeMeasurements:
    """Tests for the `optimize_measurements` function."""

    def test_optimize_measurements_qwc_generic_case(self):

        observables = [PauliY(0), PauliX(0) @ PauliX(1), PauliZ(1)]

        diagonalized_groupings_sol =[[PauliZ(wires=[0]) @ PauliZ(wires=[1])],
                                     [PauliZ(wires=[0]), PauliZ(wires=[1])]]

        post_rotations, diagonalized_groupings = optimize_measurements(observables, grouping='qwc', colouring_method='rlf')

        assert len(diagonalized_groupings) == len(diagonalized_groupings_sol)

        for i in range(len(diagonalized_groupings_sol)):
            assert len(diagonalized_groupings[i]) == len(diagonalized_groupings_sol[i])
            for j in range(len(diagonalized_groupings_sol[i])):
                assert are_identical_pauli_words(diagonalized_groupings[i][j], diagonalized_groupings_sol[i][j])

    def test_optimize_measurements_qwc_generic_case_with_coefficients(self):

        observables = [PauliY(0), PauliX(0) @ PauliX(1), PauliZ(1)]
        coefficients = [1.43, 4.21, 0.97]

        diagonalized_groupings_sol =[[PauliZ(wires=[0]) @ PauliZ(wires=[1])],
                                     [PauliZ(wires=[0]), PauliZ(wires=[1])]]

        grouped_coeffs_sol = [[4.21], [1.43, 0.97]]

        post_rotations, diagonalized_groupings, grouped_coeffs = optimize_measurements(observables, coefficients, grouping='qwc', colouring_method='rlf')

        assert len(grouped_coeffs) == len(grouped_coeffs)

        assert all(grouped_coeffs[i] == grouped_coeffs_sol[i] for i in range(len(grouped_coeffs_sol)))
