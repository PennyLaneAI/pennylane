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
from pennylane import Identity, PauliX, PauliY, PauliZ, Hadamard
from pennylane.operation import Tensor
from pennylane.wires import Wires
from pennylane.grouping.group_observables import PauliGroupingStrategy, group_observables
from pennylane.grouping.graph_colouring import largest_first, recursive_largest_first
from pennylane.grouping.optimize_measurements import optimize_measurements
from pennylane.grouping.utils import (
    get_n_qubits,
    is_pauli_word,
    are_identical_pauli_words,
    binary_symplectic_inner_prod,
    pauli_to_binary,
    binary_to_pauli,
    is_qwc,
    convert_observables_to_binary,
)


class TestGroupingUtils:
    """Basic usage and edge-case tests for the measurement optimization utility functions."""

    def test_pauli_to_binary_no_wire_map(self):
        """Test conversion of Pauli word from operator to binary vector representation when no
        `wire_map` is specified."""

        p1_op = PauliX(0) @ PauliY(1) @ PauliZ(2)
        p2_op = PauliZ(0) @ PauliY(2)
        p3_op = PauliY(1) @ PauliX(2)
        identity = Tensor(Identity(0))

        p1_vec = np.array([1, 1, 0, 0, 1, 1])
        p2_vec = np.array([0, 1, 1, 1])
        p3_vec = np.array([1, 1, 1, 0])

        assert (pauli_to_binary(p1_op) == p1_vec).all()
        assert (pauli_to_binary(p2_op) == p2_vec).all()
        assert (pauli_to_binary(p3_op) == p3_vec).all()
        assert (pauli_to_binary(identity) == np.zeros(2)).all()

    def test_pauli_to_binary_with_wire_map(self):
        """Test conversion of Pauli word from operator to binary vector representation if a
        `wire_map` is specified."""

        p1_op = PauliX("a") @ PauliZ("b") @ Identity("c")
        p2_op = PauliY(6) @ PauliZ("a") @ PauliZ("b")
        p3_op = PauliX("b") @ PauliY("c")
        identity = Identity("a") @ Identity(6)

        wire_map = {Wires("a"): 0, Wires("b"): 1, Wires("c"): 2, Wires(6): 3}

        p1_vec = np.array([1, 0, 0, 0, 0, 1, 0, 0])
        p2_vec = np.array([0, 0, 0, 1, 1, 1, 0, 1])
        p3_vec = np.array([0, 1, 1, 0, 0, 0, 1, 0])

        assert (pauli_to_binary(p1_op, wire_map=wire_map) == p1_vec).all()
        assert (pauli_to_binary(p2_op, wire_map=wire_map) == p2_vec).all()
        assert (pauli_to_binary(p3_op, wire_map=wire_map) == p3_vec).all()
        assert (pauli_to_binary(identity, wire_map=wire_map) == np.zeros(8)).all()

    def test_binary_to_pauli_no_wire_map(self):
        """Test conversion of Pauli in binary vector representation to operator form when no
        `wire_map` is specified."""

        p1_vec = np.array([1, 0, 1, 0, 0, 1])
        p2_vec = np.array([1, 1, 1, 1, 1, 1])
        p3_vec = np.array([1, 0, 1, 0, 1, 1])
        zero_vec = np.zeros(6)

        p1_op = PauliX(0) @ PauliY(2)
        p2_op = PauliY(0) @ PauliY(1) @ PauliY(2)
        p3_op = PauliX(0) @ PauliZ(1) @ PauliY(2)
        identity = Tensor(Identity(0))

        assert are_identical_pauli_words(binary_to_pauli(p1_vec), p1_op)
        assert are_identical_pauli_words(binary_to_pauli(p2_vec), p2_op)
        assert are_identical_pauli_words(binary_to_pauli(p3_vec), p3_op)
        assert are_identical_pauli_words(binary_to_pauli(zero_vec), identity)

    def test_binary_to_pauli_with_wire_map(self):
        """Test conversion of Pauli in binary vector representation to operator form when
        `wire_map` is specified."""

        p1_vec = np.array([1, 0, 1, 0, 0, 1])
        p2_vec = np.array([1, 1, 1, 1, 1, 1])
        p3_vec = np.array([1, 0, 1, 0, 1, 0])
        zero_vec = np.zeros(6)

        wire_map = {Wires("alice"): 0, Wires("bob"): 1, Wires("ancilla"): 2}

        p1_op = PauliX("alice") @ PauliY("ancilla")
        p2_op = PauliY("alice") @ PauliY("bob") @ PauliY("ancilla")
        p3_op = PauliX("alice") @ PauliZ("bob") @ PauliX("ancilla")
        identity = Identity("alice")

        assert are_identical_pauli_words(binary_to_pauli(p1_vec, wire_map=wire_map), p1_op)
        assert are_identical_pauli_words(binary_to_pauli(p2_vec, wire_map=wire_map), p2_op)
        assert are_identical_pauli_words(binary_to_pauli(p3_vec, wire_map=wire_map), p3_op)
        assert are_identical_pauli_words(binary_to_pauli(zero_vec, wire_map=wire_map), identity)

    def test_convert_observables_to_binary(self):
        """Test conversion of list of Pauli word operators to representation as a binary matrix."""

        observables = [Identity(1), PauliX(1), PauliZ(0) @ PauliZ(1)]

        binary_observables = np.array(
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]
        )

        assert (convert_observables_to_binary(observables) == binary_observables).all()

    def test_is_qwc(self):
        """Determining if two Pauli words are qubit-wise commuting."""

        n_qubits = 2
        p1_vec = pauli_to_binary(PauliX(0) @ PauliY(1), n_qubits)
        p2_vec = pauli_to_binary(PauliX(0), n_qubits)
        p3_vec = pauli_to_binary(PauliX(0) @ PauliZ(1), n_qubits)

        assert is_qwc(p1_vec, p2_vec)
        assert not is_qwc(p1_vec, p3_vec)
        assert is_qwc(p2_vec, p3_vec)

    def test_binary_symplectic_inner_prod(self):
        """Test taking the binary symplectic inner product between two elements of binary
        symplectic vector space."""

        p1_vec = np.array([1, 0, 1, 0, 1, 1, 0, 0])
        p2_vec = np.array([0, 0, 1, 1, 1, 0, 1, 0])
        p3_vec = np.array([1, 1, 0, 1, 1, 1, 0, 1])

        assert binary_symplectic_inner_prod(p1_vec, p2_vec) == 0
        assert binary_symplectic_inner_prod(p1_vec, p3_vec) == 1
        assert binary_symplectic_inner_prod(p2_vec, p3_vec) == 0

    def test_get_n_qubits(self):
        """Test for obtaining minimum number of qubits required for a set of observables."""

        observables_1 = [Identity(10)]
        observables_2 = [Identity(0) @ Identity(10), PauliX(2) @ Identity(10)]
        observables_3 = [PauliX("a") @ PauliZ("0"), PauliX("a") @ PauliY("b")]

        assert get_n_qubits(observables_1) == 0
        assert get_n_qubits(observables_2) == 1
        assert get_n_qubits(observables_3) == 3

    def test_is_pauli_word(self):
        """Test for determining whether input `Observable` instance is a Pauli word."""

        observable_1 = PauliX(0)
        observable_2 = PauliZ(1) @ PauliX(2) @ PauliZ(4)
        observable_3 = PauliX(1) @ Hadamard(4)
        observable_4 = Hadamard(0)

        assert is_pauli_word(observable_1)
        assert is_pauli_word(observable_2)
        assert not is_pauli_word(observable_3)
        assert not is_pauli_word(observable_4)

    def test_are_identical_pauli_words(self):
        """Tests for determining if two Pauli words have the same `wires` and `name` attributes."""

        pauli_word_1 = PauliX(0) @ PauliY(1)
        pauli_word_2 = PauliY(1) @ PauliX(0)
        pauli_word_3 = Tensor(PauliX(0), PauliY(1))
        pauli_word_4 = PauliX(1) @ PauliZ(2)

        assert are_identical_pauli_words(pauli_word_1, pauli_word_2)
        assert are_identical_pauli_words(pauli_word_1, pauli_word_3)
        assert not are_identical_pauli_words(pauli_word_1, pauli_word_4)
        assert not are_identical_pauli_words(pauli_word_3, pauli_word_4)


class TestPauliGroupingStrategy:
    """Tests for the PauliGroupingStrategy class"""

    def test_construct_complement_adj_matrix_for_operators(self):
        """Constructing the complement graph adjacency matrix for a list of Pauli words and a
        given symmetric binary relation."""

        observables = [PauliY(0), PauliZ(0) @ PauliZ(1), PauliY(0) @ PauliX(1)]

        qwc_complement_adjacency_matrix = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])

        commuting_complement_adjacency_matrix = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])

        anticommuting_complement_adjacency_matrix = np.array([[0, 0, 1], [0, 0, 1], [1, 1, 0]])

        grouping_instance = PauliGroupingStrategy(observables, "qwc")
        assert (
            grouping_instance.construct_complement_adj_matrix_for_operator()
            == qwc_complement_adjacency_matrix
        ).all()
        grouping_instance = PauliGroupingStrategy(observables, "commuting")
        assert (
            grouping_instance.construct_complement_adj_matrix_for_operator()
            == commuting_complement_adjacency_matrix
        ).all()
        grouping_instance = PauliGroupingStrategy(observables, "anticommuting")
        assert (
            grouping_instance.construct_complement_adj_matrix_for_operator()
            == anticommuting_complement_adjacency_matrix
        ).all()

    def test_construct_complement_adj_matrix_for_trivial_operators(self):
        """Constructing the adjacency matrix for a list of identity operations and various
        symmetric binary relations"""

        observables = [Identity(0), Identity(0), Identity(7)]

        qwc_complement_adjacency_matrix = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

        commuting_complement_adjacency_matrix = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

        anticommuting_complement_adjacency_matrix = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])

        grouping_instance = PauliGroupingStrategy(observables, "qwc")
        assert (
            grouping_instance.construct_complement_adj_matrix_for_operator()
            == qwc_complement_adjacency_matrix
        ).all()
        grouping_instance = PauliGroupingStrategy(observables, "commuting")
        assert (
            grouping_instance.construct_complement_adj_matrix_for_operator()
            == commuting_complement_adjacency_matrix
        ).all()
        grouping_instance = PauliGroupingStrategy(observables, "anticommuting")
        assert (
            grouping_instance.construct_complement_adj_matrix_for_operator()
            == anticommuting_complement_adjacency_matrix
        ).all()


class TestGraphcolouringFunctions:
    """Tests for graph colouring functions."""

    def verify_graph_colour_solution(self, adjacency_matrix, colouring):
        """Verifies if all vertices of the same colour are not connected."""

        for colour in colouring.keys():

            grouping = colouring[colour]
            size_grouping = len(grouping)

            for i in range(size_grouping):
                for j in range(i + 1, size_grouping):
                    vert_i = grouping[i][0]
                    vert_j = grouping[j][0]

                    if adjacency_matrix[vert_i][vert_j] == 1:
                        return False

        return True

    def test_graph_colouring(self):
        """Verify that random unweighted, undirected graph's colour is a valid solution."""

        n_qubits = 8
        adjacency_matrix = np.random.randint(2, size=(n_qubits, n_qubits))
        np.fill_diagonal(adjacency_matrix, 0)

        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                adjacency_matrix[j, i] = adjacency_matrix[i, j]

        lf_colouring = largest_first(np.asarray([list(range(n_qubits))]), adjacency_matrix)
        rlf_colouring = recursive_largest_first(
            np.asarray([list(range(n_qubits))]), adjacency_matrix
        )

        assert self.verify_graph_colour_solution(adjacency_matrix, lf_colouring)
        assert self.verify_graph_colour_solution(adjacency_matrix, rlf_colouring)

    def test_trivial_graph_colouring(self):
        """Tests validity of graph colouring solution for a graph with no edges."""

        n_qubits = 8
        adjacency_matrix = np.zeros((n_qubits, n_qubits))

        lf_colouring = largest_first(np.asarray([list(range(n_qubits))]), adjacency_matrix)
        rlf_colouring = recursive_largest_first(
            np.asarray([list(range(n_qubits))]), adjacency_matrix
        )

        assert self.verify_graph_colour_solution(adjacency_matrix, lf_colouring)
        assert self.verify_graph_colour_solution(adjacency_matrix, rlf_colouring)


class TestOptimizeMeasurements:
    """Tests for the `optimize_measurements` function."""

    def test_optimize_measurements_qwc_generic_case(self):
        """Generic test case without coefficients."""

        observables = [PauliY(0), PauliX(0) @ PauliX(1), PauliZ(1)]

        diagonalized_groupings_sol = [
            [PauliZ(wires=[0]) @ PauliZ(wires=[1])],
            [PauliZ(wires=[0]), PauliZ(wires=[1])],
        ]

        diagonalized_groupings = optimize_measurements(
            observables, grouping="qwc", colouring_method="rlf"
        )[1]

        assert len(diagonalized_groupings) == len(diagonalized_groupings_sol)

        for i in range(len(diagonalized_groupings_sol)):
            assert len(diagonalized_groupings[i]) == len(diagonalized_groupings_sol[i])
            for j in range(len(diagonalized_groupings_sol[i])):
                assert are_identical_pauli_words(
                    diagonalized_groupings[i][j], diagonalized_groupings_sol[i][j]
                )

    def test_optimize_measurements_qwc_generic_case_with_coefficients(self):
        """Tests if coefficients are properly re-structured."""

        observables = [PauliY(0), PauliX(0) @ PauliX(1), PauliZ(1)]
        coefficients = [1.43, 4.21, 0.97]

        diagonalized_groupings_sol = [
            [PauliZ(wires=[0]) @ PauliZ(wires=[1])],
            [PauliZ(wires=[0]), PauliZ(wires=[1])],
        ]

        grouped_coeffs_sol = [[4.21], [1.43, 0.97]]

        grouped_coeffs = optimize_measurements(
            observables, coefficients, grouping="qwc", colouring_method="rlf"
        )[2]

        assert len(grouped_coeffs) == len(grouped_coeffs)

        assert all(
            grouped_coeffs[i] == grouped_coeffs_sol[i] for i in range(len(grouped_coeffs_sol))
        )
