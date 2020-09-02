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
Unit tests for `PauliGroupingStrategy` and `group_observables` in `grouping/group_observables.py`.
"""
import pytest
import numpy as np
from pennylane import Identity, PauliX, PauliY, PauliZ
from pennylane.grouping.utils import are_identical_pauli_words
from pennylane.grouping.group_observables import PauliGroupingStrategy, group_observables


class TestPauliGroupingStrategy:
    """Tests for the PauliGroupingStrategy class"""

    def test_construct_qwc_complement_adj_matrix_for_operators(self):
        """Constructing the complement graph adjacency matrix for a list of Pauli words and a
        given symmetric binary relation."""

        observables = [PauliY(0), PauliZ(0) @ PauliZ(1), PauliY(0) @ PauliX(1)]
        qwc_complement_adjacency_matrix = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])

        grouping_instance = PauliGroupingStrategy(observables, "qwc")
        assert (
            grouping_instance.construct_complement_adj_matrix_for_operator()
            == qwc_complement_adjacency_matrix
        ).all()

    def test_construct_commuting_complement_adj_matrix_for_operators(self):
        """Constructing the complement graph adjacency matrix for a list of Pauli words and a
        given symmetric binary relation."""

        observables = [PauliY(0), PauliZ(0) @ PauliZ(1), PauliY(0) @ PauliX(1)]
        commuting_complement_adjacency_matrix = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])

        grouping_instance = PauliGroupingStrategy(observables, "commuting")
        assert (
            grouping_instance.construct_complement_adj_matrix_for_operator()
            == commuting_complement_adjacency_matrix
        ).all()

    def test_construct_anticommuting_complement_adj_matrix_for_operators(self):
        """Constructing the complement graph adjacency matrix for a list of Pauli words and a
        given symmetric binary relation."""

        observables = [PauliY(0), PauliZ(0) @ PauliZ(1), PauliY(0) @ PauliX(1)]
        anticommuting_complement_adjacency_matrix = np.array([[0, 0, 1], [0, 0, 1], [1, 1, 0]])

        grouping_instance = PauliGroupingStrategy(observables, "anticommuting")
        assert (
            grouping_instance.construct_complement_adj_matrix_for_operator()
            == anticommuting_complement_adjacency_matrix
        ).all()

    trivial_ops = [
        [Identity(0), Identity(0), Identity(7)],
        [Identity("a") @ Identity(1), Identity("b"), Identity("b") @ Identity("c")],
    ]

    @pytest.mark.parametrize("observables", trivial_ops)
    def test_construct_complement_qwc_adj_matrix_for_trivial_operators(self, observables):
        """Constructing the compliment of QWC graph's adjacency matrix for a list of identity
        operations and various symmetric binary relations"""

        qwc_complement_adjacency_matrix = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

        grouping_instance = PauliGroupingStrategy(observables, "qwc")
        assert (
            grouping_instance.construct_complement_adj_matrix_for_operator()
            == qwc_complement_adjacency_matrix
        ).all()

    @pytest.mark.parametrize("observables", trivial_ops)
    def test_construct_complement_commuting_adj_matrix_for_trivial_operators(self, observables):
        """Constructing the compliment of commutativity graph's adjacency matrix for a list of
        identity operations and various symmetric binary relations"""

        commuting_complement_adjacency_matrix = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

        grouping_instance = PauliGroupingStrategy(observables, "commuting")
        assert (
            grouping_instance.construct_complement_adj_matrix_for_operator()
            == commuting_complement_adjacency_matrix
        ).all()

    @pytest.mark.parametrize("observables", trivial_ops)
    def test_construct_complement_anticommuting_adj_matrix_for_trivial_operators(self, observables):
        """Constructing the compliment of anticommutativity graph's adjacency matrix for a list of
        identity operations and various symmetric binary relations"""

        observables = [Identity(0), Identity(0), Identity(7)]

        anticommuting_complement_adjacency_matrix = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])

        grouping_instance = PauliGroupingStrategy(observables, "anticommuting")
        assert (
            grouping_instance.construct_complement_adj_matrix_for_operator()
            == anticommuting_complement_adjacency_matrix
        ).all()


observables_list = [
    [PauliX(0) @ PauliZ(1), PauliY(2) @ PauliZ(1), PauliX(1), PauliY(0), PauliZ(1) @ PauliZ(2)],
    [
        Identity(1) @ Identity(0),
        PauliX(1) @ PauliY(0) @ Identity(2),
        PauliZ(2),
        Identity(0),
        PauliZ(2) @ Identity(0),
        PauliX(0) @ PauliX(1),
    ],
    [
        PauliX("a") @ Identity("b"),
        PauliX("a") @ PauliZ("b"),
        PauliX("b") @ PauliZ("a"),
        PauliZ("a") @ PauliZ("b") @ PauliZ("c"),
    ],
]

qwc_sols = [
    [
        [PauliX(wires=[1]), PauliY(wires=[0])],
        [PauliX(wires=[0]) @ PauliZ(wires=[1]), PauliZ(wires=[1]) @ PauliY(wires=[2])],
        [PauliZ(wires=[1]) @ PauliZ(wires=[2])],
    ],
    [
        [
            Identity(wires=[1]),
            PauliX(wires=[1]) @ PauliY(wires=[0]),
            PauliZ(wires=[2]),
            Identity(wires=[1]),
            PauliZ(wires=[2]),
        ],
        [PauliX(wires=[1]) @ PauliX(wires=[0])],
    ],
    [
        [PauliZ(wires=["a"]) @ PauliX(wires=["b"])],
        [PauliZ(wires=["a"]) @ PauliZ(wires=["b"]) @ PauliZ(wires=["c"])],
        [PauliX(wires=["a"]), PauliX(wires=["a"]) @ PauliZ(wires=["b"])],
    ],
]

commuting_sols = [
    [
        [PauliX(wires=[1]), PauliY(wires=[0])],
        [PauliX(wires=[0]) @ PauliZ(wires=[1]), PauliZ(wires=[1]) @ PauliY(wires=[2])],
        [PauliZ(wires=[1]) @ PauliZ(wires=[2])],
    ],
    [
        [
            Identity(wires=[1]),
            PauliX(wires=[1]) @ PauliY(wires=[0]),
            PauliZ(wires=[2]),
            Identity(wires=[1]),
            PauliZ(wires=[2]),
        ],
        [PauliX(wires=[1]) @ PauliX(wires=[0])],
    ],
    [
        [PauliZ(wires=["a"]) @ PauliZ(wires=["b"]) @ PauliZ(wires=["c"])],
        [PauliX(wires=["a"]), PauliX(wires=["a"]) @ PauliZ(wires=["b"])],
        [PauliZ(wires=["a"]) @ PauliX(wires=["b"])],
    ],
]

anticommuting_sols = [
    [
        [PauliX(wires=[0]) @ PauliZ(wires=[1]), PauliY(wires=[0])],
        [
            PauliZ(wires=[1]) @ PauliY(wires=[2]),
            PauliX(wires=[1]),
            PauliZ(wires=[1]) @ PauliZ(wires=[2]),
        ],
    ],
    [
        [Identity(wires=[1])],
        [PauliZ(wires=[2])],
        [Identity(wires=[1])],
        [PauliZ(wires=[2])],
        [PauliX(wires=[1]) @ PauliY(wires=[0]), PauliX(wires=[1]) @ PauliX(wires=[0])],
    ],
    [
        [
            PauliX(wires=["a"]) @ PauliZ(wires=["b"]),
            PauliZ(wires=["a"]) @ PauliZ(wires=["b"]) @ PauliZ(wires=["c"]),
        ],
        [PauliX(wires=["a"]), PauliZ(wires=["a"]) @ PauliX(wires=["b"])],
    ],
]


class TestGroupObservables:
    """
    Tests for `group_observables` function using QWC, commuting, and anticommuting partitioning.
    """

    qwc_tuples = [(obs, qwc_sols[i]) for i, obs in enumerate(observables_list)]

    com_tuples = [(obs, commuting_sols[i]) for i, obs in enumerate(observables_list)]

    anticom_tuples = [(obs, anticommuting_sols[i]) for i, obs in enumerate(observables_list)]

    @pytest.mark.parametrize("observables,qwc_partitions_sol", qwc_tuples)
    def test_qwc_partitioning(self, observables, qwc_partitions_sol):

        qwc_partitions = group_observables(observables, grouping_type="qwc")

        # assert the correct number of partitions:
        n_partitions = len(qwc_partitions_sol)
        assert len(qwc_partitions) == n_partitions
        # assert each partition is of the correct length:
        assert all(
            [len(qwc_partitions[i]) == len(qwc_partitions_sol[i]) for i in range(n_partitions)]
        )
        # assert each partition contains the same Pauli terms as the solution partition:
        for i, partition in enumerate(qwc_partitions):
            for j, pauli in enumerate(partition):
                assert are_identical_pauli_words(pauli, qwc_partitions_sol[i][j])

    @pytest.mark.parametrize("observables,com_partitions_sol", com_tuples)
    def test_commuting_partitioning(self, observables, com_partitions_sol):

        com_partitions = group_observables(observables, grouping_type="commuting")

        # assert the correct number of partitions:
        n_partitions = len(com_partitions_sol)
        assert len(com_partitions) == n_partitions
        # assert each partition is of the correct length:
        assert all(
            [len(com_partitions[i]) == len(com_partitions_sol[i]) for i in range(n_partitions)]
        )
        # assert each partition contains the same Pauli terms as the solution partition:
        for i, partition in enumerate(com_partitions):
            for j, pauli in enumerate(partition):
                assert are_identical_pauli_words(pauli, com_partitions_sol[i][j])

    @pytest.mark.parametrize("observables,anticom_partitions_sol", anticom_tuples)
    def test_anticommuting_partitioning(self, observables, anticom_partitions_sol):

        anticom_partitions = group_observables(observables, grouping_type="anticommuting")

        # assert the correct number of partitions:
        n_partitions = len(anticom_partitions_sol)
        assert len(anticom_partitions) == n_partitions
        # assert each partition is of the correct length:
        assert all(
            [
                len(anticom_partitions[i]) == len(anticom_partitions_sol[i])
                for i in range(n_partitions)
            ]
        )
        # assert each partition contains the same Pauli terms as the solution partition:
        for i, partition in enumerate(anticom_partitions):
            for j, pauli in enumerate(partition):
                assert are_identical_pauli_words(pauli, anticom_partitions_sol[i][j])
