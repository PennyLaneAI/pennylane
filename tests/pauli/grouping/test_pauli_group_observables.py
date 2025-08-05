# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

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
Unit tests for ``PauliGroupingStrategy`` and ``group_observables`` in ``/pauli/grouping/group_observables.py``.
"""
import sys

import numpy as np
import pytest

import pennylane as qml
from pennylane import Identity, PauliX, PauliY, PauliZ
from pennylane import numpy as pnp
from pennylane.pauli import are_identical_pauli_words, are_pauli_words_qwc
from pennylane.pauli.grouping.group_observables import (
    PauliGroupingStrategy,
    compute_partition_indices,
    group_observables,
    items_partitions_from_idx_partitions,
)


class TestOldRX:
    """Test PauliGroupingStrategy behaves correctly when versions of rx older than 0.15 are used"""

    @pytest.mark.parametrize("new_colourer", ["dsatur", "gis"])
    def test_new_strategies_with_old_rx_raise_error(self, monkeypatch, new_colourer):
        """Test that an error is raised if a new strategy is used with old rx"""
        # Monkey patch the new_rx variable to False
        grouping = sys.modules["pennylane.pauli.grouping.group_observables"]
        monkeypatch.setattr(grouping, "new_rx", False)

        observables = [qml.X(0)]
        with pytest.raises(ValueError, match="not supported in this version of Rustworkx"):
            PauliGroupingStrategy(observables, graph_colourer=new_colourer)

    def test_old_rx_produces_right_results(self, monkeypatch):
        """Test that the results produced with an older rx version is the same as with lf"""
        observables = [qml.X(0) @ qml.Z(1), qml.Z(0), qml.X(1)]

        new_groupper = PauliGroupingStrategy(observables, graph_colourer="lf")
        new_partitions = new_groupper.partition_observables()

        grouping = sys.modules["pennylane.pauli.grouping.group_observables"]
        monkeypatch.setattr(grouping, "new_rx", False)

        old_groupper = PauliGroupingStrategy(observables, graph_colourer="lf")
        old_partitions = old_groupper.partition_observables()

        assert new_partitions == old_partitions


class TestPauliGroupingStrategy:
    """Tests for the PauliGroupingStrategy class"""

    def test_initialize_with_invalid_grouping(self):
        """Tests ValueError is raised if specified grouping_type is not recognized."""

        observables = [PauliX(0) @ PauliY(2), PauliZ(2)]

        assert pytest.raises(
            ValueError, PauliGroupingStrategy, observables, grouping_type="invalid"
        )

    def test_initialize_with_invalid_colourer(self):
        """Tests ValueError is raised if specified graph_colourer is not recognized."""

        observables = [PauliX(0) @ PauliY(2), PauliZ(2)]

        assert pytest.raises(
            ValueError, PauliGroupingStrategy, observables, graph_colourer="invalid"
        )

    def test_construct_qwc_adj_matrix(self):
        """Constructing the complement graph adjacency matrix for a list of Pauli words according
        to qubit-wise commutativity."""

        observables = [PauliY(0), PauliZ(0) @ PauliZ(1), PauliY(0) @ PauliX(1)]
        qwc_complement_adjacency_matrix = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])

        grouping_instance = PauliGroupingStrategy(observables, "qwc")
        assert (grouping_instance.adj_matrix == qwc_complement_adjacency_matrix).all()

    def test_construct_commuting_adj_matrix(self):
        """Constructing the complement graph adjacency matrix for a list of Pauli words according
        to general commutativity."""

        observables = [PauliY(0), PauliZ(0) @ PauliZ(1), PauliY(0) @ PauliX(1)]
        commuting_complement_adjacency_matrix = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])

        grouping_instance = PauliGroupingStrategy(observables, "commuting")
        assert (grouping_instance.adj_matrix == commuting_complement_adjacency_matrix).all()

    def test_construct_anticommuting_adj_matrix(self):
        """Constructing the complement graph adjacency matrix for a list of Pauli words according
        to anticommutativity."""

        observables = [PauliY(0), PauliZ(0) @ PauliZ(1), PauliY(0) @ PauliX(1)]
        anticommuting_complement_adjacency_matrix = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 1]])

        grouping_instance = PauliGroupingStrategy(observables, "anticommuting")
        assert (grouping_instance.adj_matrix == anticommuting_complement_adjacency_matrix).all()

    trivial_ops = [
        [Identity(0), Identity(0), Identity(7)],
        [Identity("a") @ Identity(1), Identity("b"), Identity("b") @ Identity("c")],
    ]

    @pytest.mark.parametrize("observables", trivial_ops)
    def test_construct_complement_qwc_adj_matrix_for_trivial_operators(self, observables):
        """Constructing the complement of QWC graph's adjacency matrix for a list of identity
        operations and various symmetric binary relations"""

        qwc_complement_adjacency_matrix = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

        grouping_instance = PauliGroupingStrategy(observables, "qwc")
        assert (grouping_instance.adj_matrix == qwc_complement_adjacency_matrix).all()

    @pytest.mark.parametrize("observables", trivial_ops)
    def test_construct_complement_commuting_adj_matrix_for_trivial_operators(self, observables):
        """Constructing the complement of commutativity graph's adjacency matrix for a list of
        identity operations and various symmetric binary relations"""

        commuting_complement_adjacency_matrix = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

        grouping_instance = PauliGroupingStrategy(observables, "commuting")
        assert (grouping_instance.adj_matrix == commuting_complement_adjacency_matrix).all()

    @pytest.mark.parametrize("observables", trivial_ops)
    def test_construct_complement_anticommuting_adj_matrix_for_trivial_operators(self, observables):
        """Constructing the complement of anticommutativity graph's adjacency matrix for a list of
        identity operations and various symmetric binary relations"""

        anticommuting_complement_adjacency_matrix = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])

        grouping_instance = PauliGroupingStrategy(observables, "anticommuting")
        assert (grouping_instance.adj_matrix == anticommuting_complement_adjacency_matrix).all()

    def test_wrong_length_of_custom_indices(self):
        """Test that an error is raised if the length of indices does not match the length of observables"""
        observables = [qml.X(0) @ qml.Z(1), qml.Z(0), qml.X(1)]
        groupper = PauliGroupingStrategy(observables=observables)

        custom_indices = [1, 3, 5, 7]

        with pytest.raises(ValueError, match="The length of the list of indices:"):
            groupper.idx_partitions_from_graph(observables_indices=custom_indices)

    def test_custom_indices_partition(self):
        """Test that a custom list indices is partitioned according to the observables they correspond to."""
        observables = [qml.X(0) @ qml.Z(1), qml.Z(0), qml.X(1)]
        groupper = PauliGroupingStrategy(observables=observables)

        custom_indices = [1, 3, 5]
        # the indices rustworkx assigns to each observable
        standard_indices = list(range(len(observables)))
        # map between custom and standard indices
        map_indices = dict(zip(custom_indices, standard_indices))

        # compute observable and custom indices partitions
        observables_partitioned = groupper.partition_observables()
        # pylint: disable=protected-access
        indices_partitioned = groupper.idx_partitions_from_graph(observables_indices=custom_indices)
        for group_obs, group_custom_indices in zip(observables_partitioned, indices_partitioned):
            for i, custom_idx in enumerate(group_custom_indices):
                standard_idx = map_indices[custom_idx]
                # observable corresponding to the custom index
                observable_from_idx_partition = observables[standard_idx]
                # observable in partition in the position of the custom idx
                observable_from_partition = group_obs[i]
                assert observable_from_idx_partition == observable_from_partition


observables_list = [
    [PauliX(0) @ PauliZ(1), PauliY(2) @ PauliZ(1), PauliX(1), PauliY(0), PauliZ(1) @ PauliZ(2)],
    [
        qml.s_prod(1.5, qml.prod(PauliX(0), PauliZ(1))),
        qml.prod(PauliY(2), PauliZ(1)),
        PauliX(1),
        PauliY(0),
        qml.prod(PauliZ(1), PauliZ(2)),
    ],
    [
        Identity(1) @ Identity(0),
        PauliX(1) @ PauliY(0) @ Identity(2),
        PauliZ(2),
        Identity(0),
        PauliZ(2) @ Identity(0),
        PauliX(0) @ PauliX(1),
    ],
    [
        qml.prod(Identity(1), Identity(0)),
        qml.prod(PauliX(1), PauliY(0), Identity(2)),
        PauliZ(2),
        qml.s_prod(-2.0, Identity(0)),
        qml.prod(PauliZ(2), Identity(0)),
        qml.prod(PauliX(0), PauliX(1)),
    ],
    [
        PauliX("a") @ Identity("b"),
        PauliX("a") @ PauliZ("b"),
        PauliX("b") @ PauliZ("a"),
        PauliZ("a") @ PauliZ("b") @ PauliZ("c"),
    ],
    [
        qml.prod(PauliX("a"), Identity("b")),
        qml.s_prod(0.5, qml.prod(PauliX("a"), PauliZ("b"))),
        qml.s_prod(0.5, qml.prod(PauliX("b"), PauliZ("a"))),
        qml.prod(PauliZ("a"), PauliZ("b") @ PauliZ("c")),
    ],
    [PauliX([(0, 0)]), PauliZ([(0, 0)])],
]

qwc_sols = [
    [
        [PauliX(wires=[0]) @ PauliZ(wires=[1]), PauliZ(wires=[1]) @ PauliY(wires=[2])],
        [PauliX(wires=[1]), PauliY(wires=[0])],
        [PauliZ(wires=[1]) @ PauliZ(wires=[2])],
    ],
    [
        [PauliX(wires=[0]) @ PauliZ(wires=[1]), PauliZ(wires=[1]) @ PauliY(wires=[2])],
        [PauliX(wires=[1]), PauliY(wires=[0])],
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
        [PauliX(wires=["a"]), PauliX(wires=["a"]) @ PauliZ(wires=["b"])],
        [PauliZ(wires=["a"]) @ PauliX(wires=["b"])],
        [PauliZ(wires=["a"]) @ PauliZ(wires=["b"]) @ PauliZ(wires=["c"])],
    ],
    [
        [PauliX(wires=["a"]), PauliX(wires=["a"]) @ PauliZ(wires=["b"])],
        [PauliZ(wires=["a"]) @ PauliX(wires=["b"])],
        [PauliZ(wires=["a"]) @ PauliZ(wires=["b"]) @ PauliZ(wires=["c"])],
    ],
    [[PauliX([(0, 0)])], [PauliZ([(0, 0)])]],
]

commuting_sols = [
    [
        [PauliX(wires=[1]), PauliY(wires=[0])],
        [PauliX(wires=[0]) @ PauliZ(wires=[1]), PauliZ(wires=[1]) @ PauliY(wires=[2])],
        [PauliZ(wires=[1]) @ PauliZ(wires=[2])],
    ],
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
    [
        [PauliZ(wires=["a"]) @ PauliZ(wires=["b"]) @ PauliZ(wires=["c"])],
        [PauliX(wires=["a"]), PauliX(wires=["a"]) @ PauliZ(wires=["b"])],
        [PauliZ(wires=["a"]) @ PauliX(wires=["b"])],
    ],
    [[PauliX([(0, 0)])], [PauliZ([(0, 0)])]],
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
    [
        [
            PauliX(wires=["a"]) @ PauliZ(wires=["b"]),
            PauliZ(wires=["a"]) @ PauliZ(wires=["b"]) @ PauliZ(wires=["c"]),
        ],
        [PauliX(wires=["a"]), PauliZ(wires=["a"]) @ PauliX(wires=["b"])],
    ],
    [[PauliX([(0, 0)]), PauliZ([(0, 0)])]],
]

com_tuples = list(zip(observables_list, commuting_sols))

anticom_tuples = list(zip(observables_list, anticommuting_sols))


def are_partitions_equal(partition_1: list, partition_2: list) -> bool:
    """Checks whether two partitions are the same, i.e. contain the same Pauli terms.

    We check this way since the partitions might vary in the order of the elements

    Args:
        partition_1 (list[Operator]): list of Pauli word ``Operator`` instances corresponding to a partition.
        partition_2 (list[Operator]): list of Pauli word ``Operator`` instances corresponding to a partition.

    """
    partition_3 = set(
        partition_2
    )  # to improve the lookup time for similar obs in the second partition
    for pauli in partition_1:
        if not any(are_identical_pauli_words(pauli, other) for other in partition_3):
            return False
    return True


class TestGroupObservables:
    """
    Tests for ``group_observables`` function using QWC, commuting, and anticommuting partitioning.
    """

    qwc_tuples = list(zip(observables_list, qwc_sols))

    @pytest.mark.parametrize("observables,qwc_partitions_sol", qwc_tuples)
    def test_qwc_partitioning(self, observables, qwc_partitions_sol):
        qwc_partitions = group_observables(observables, grouping_type="qwc")

        # assert the correct number of partitions:
        assert len(qwc_partitions) == len(qwc_partitions_sol)

        # assert each computed partition contains appears in the computed solution.
        for comp_partition in qwc_partitions:
            assert any(
                are_partitions_equal(exp_partition, comp_partition)
                for exp_partition in qwc_partitions_sol
            )

    @pytest.mark.parametrize("observables,com_partitions_sol", com_tuples)
    def test_commuting_partitioning(self, observables, com_partitions_sol):
        com_partitions = group_observables(observables, grouping_type="commuting")

        assert len(com_partitions) == len(com_partitions_sol)
        # assert each computed partition contains appears in the computed solution.
        for comp_partition in com_partitions:
            assert any(
                are_partitions_equal(exp_partition, comp_partition)
                for exp_partition in com_partitions_sol
            )

    @pytest.mark.parametrize("observables,anticom_partitions_sol", anticom_tuples)
    def test_anticommuting_partitioning(self, observables, anticom_partitions_sol):
        anticom_partitions = group_observables(observables, grouping_type="anticommuting")

        # assert the correct number of partitions:
        assert len(anticom_partitions) == len(anticom_partitions_sol)
        # assert each computed partition contains appears in the computed solution.
        for comp_partition in anticom_partitions:
            assert any(
                are_partitions_equal(exp_partition, comp_partition)
                for exp_partition in anticom_partitions_sol
            )

    def test_group_observables_exception(self):
        """Tests that the ``group_observables`` function raises an exception if
        the lengths of coefficients and observables do not agree."""
        observables = [Identity(0), PauliX(1)]
        coefficients = [0.5]
        with pytest.raises(IndexError, match="must be the same length"):
            group_observables(observables, coefficients)

    def test_binary_repr_custom_wire_map(self):
        """Tests that the ``binary_repr`` method sets a custom
        wire map correctly."""
        # pylint: disable=protected-access

        observables = [Identity("alice"), Identity("bob"), Identity("charlie")]
        grouping_instance = PauliGroupingStrategy(observables, "anticommuting")

        n_qubits = 3
        wire_map = {"alice": 1, "bob": 0, "charlie": 2}
        _ = grouping_instance.binary_repr(n_qubits, wire_map)

        assert grouping_instance._wire_map == wire_map

    def test_return_list_coefficients(self):
        """Tests that if the coefficients are given as a list, the groups
        are likewise lists."""
        obs = [qml.PauliX(0), qml.PauliX(1)]
        coeffs = [1.0, 2.0]
        _, grouped_coeffs = group_observables(obs, coeffs)
        assert isinstance(grouped_coeffs[0], list)

    def test_observables_on_no_wires(self):
        """Test that observables on no wires are stuck in the first group."""

        observables = [
            qml.I(),
            qml.X(0) @ qml.Y(1),
            qml.Z(0),
            2 * qml.I(),
        ]

        groups = group_observables(observables)
        assert groups == [[qml.X(0) @ qml.Y(1), qml.I(), 2 * qml.I()], [qml.Z(0)]]

    def test_no_observables_with_wires(self):
        """Test when only observables with no wires are present."""

        observables = [qml.I(), 2 * qml.I()]
        groups = group_observables(observables)
        assert groups == [observables]

        groups, coeffs = group_observables(observables, [1, 2])
        assert groups == [observables]
        assert coeffs == [[1, 2]]

    def test_observables_on_no_wires_coeffs(self):
        """Test that observables on no wires are stuck in the first group and
        coefficients are tracked when provided."""

        observables = [
            qml.X(0),
            qml.Z(0),
            2 * qml.I(),
            qml.I() @ qml.I(),
        ]
        coeffs = [1, 2, 3, 4]
        groups, out_coeffs = group_observables(observables, coeffs)
        assert groups == [[qml.X(0), 2 * qml.I(), qml.I() @ qml.I()], [qml.Z(0)]]
        assert out_coeffs == [[1, 3, 4], [2]]


class TestComputePartitionIndices:
    """Tests for ``compute_partition_indices``"""

    OBS_IDX_PARTITIONS = [
        (
            [qml.I(), qml.X(0) @ qml.X(1), qml.Z(0) @ qml.Z(1), 2 * qml.I(), 2 * qml.Z(0)],
            ((0, 1, 3), (2, 4)),
            [[qml.I(), qml.X(0) @ qml.X(1), 2 * qml.I()], [qml.Z(0) @ qml.Z(1), 2 * qml.Z(0)]],
        ),
    ]

    def test_invalid_colouring_method(self):
        """Test that passing an invalid colouring method raises an error"""
        observables = [qml.X(0) @ qml.Z(1), qml.Z(0), qml.X(1)]
        with pytest.raises(ValueError, match="Graph colouring method must be one of"):
            compute_partition_indices(observables=observables, method="recursive")

    def test_only_observables_without_wires(self):
        """Test that if none of the observables has wires, they are all in one single partition."""

        observables = [qml.I(), 2 * qml.I()]
        partition_indices = compute_partition_indices(observables=observables)
        assert partition_indices == (tuple(range(len(observables))),)

    @pytest.mark.parametrize("observables, indices, obs_partitions", OBS_IDX_PARTITIONS)
    def test_obs_from_indices_partitions(self, observables, indices, obs_partitions):
        """Test that obs_partition_from_idx_partitions returns the correct observables"""

        partition_obs = items_partitions_from_idx_partitions(observables, indices)
        assert partition_obs == obs_partitions

    def test_mixed_observables_qwc(self):
        """Test that if both observables with wires and without wires are present,
        the latter are appended on the first element of the former and the partitions are qwc."""
        observables = [qml.I(), qml.X(0), qml.Z(0), 2 * qml.I(), 2 * qml.Z(0)]
        partition_indices = compute_partition_indices(observables=observables, grouping_type="qwc")
        indices_no_wires = (0, 3)
        assert set(indices_no_wires) < set(partition_indices[0])

        partition_obs = items_partitions_from_idx_partitions(observables, partition_indices)
        for partition in partition_obs:
            assert are_pauli_words_qwc(partition)

    @pytest.mark.parametrize("observables,com_partitions_sol", com_tuples)
    def test_commuting_partitioning(self, observables, com_partitions_sol):
        """Test that using the commuting grouping type returns the correct solutions."""

        partition_indices = compute_partition_indices(
            observables=observables, grouping_type="commuting"
        )

        com_partitions = items_partitions_from_idx_partitions(observables, partition_indices)

        assert len(com_partitions) == len(com_partitions_sol)
        # assert each computed partition contains appears in the computed solution.
        for comp_partition in com_partitions:
            assert any(
                are_partitions_equal(exp_partition, comp_partition)
                for exp_partition in com_partitions_sol
            )

    @pytest.mark.parametrize("observables,anticom_partitions_sol", anticom_tuples)
    def test_anticommuting_partitioning(self, observables, anticom_partitions_sol):
        """Test that using the anticommuting grouping type returns the correct solutions."""

        partition_indices = compute_partition_indices(
            observables=observables, grouping_type="anticommuting"
        )

        anticom_partitions = items_partitions_from_idx_partitions(observables, partition_indices)

        # assert the correct number of partitions:
        assert len(anticom_partitions) == len(anticom_partitions_sol)
        # assert each computed partition contains appears in the computed solution.
        for comp_partition in anticom_partitions:
            assert any(
                are_partitions_equal(exp_partition, comp_partition)
                for exp_partition in anticom_partitions_sol
            )

    @pytest.mark.parametrize("method", ("rlf", "lf", "dsatur", "gis"))
    def test_colouring_methods(self, method):
        """Test that all colouring methods return the correct results."""
        observables = [qml.X(0) @ qml.Z(1), qml.Z(0), qml.X(1)]
        partition_indices = compute_partition_indices(
            observables, grouping_type="qwc", method=method
        )
        assert set(partition_indices) == {(0,), (1, 2)}


class TestDifferentiable:
    """Tests that grouping observables is differentiable with respect to the coefficients."""

    def test_differentiation_autograd(self, tol):
        """Test that grouping is differentiable with autograd tensors as coefficient"""
        coeffs = pnp.array([1.0, 2.0, 3.0], requires_grad=True)
        obs = [PauliX(wires=0), PauliX(wires=1), PauliZ(wires=1)]

        def group(coeffs, select=None):
            _, grouped_coeffs = group_observables(obs, coeffs)
            return grouped_coeffs[select]

        jac_fn = qml.jacobian(group)
        assert pnp.allclose(
            jac_fn(coeffs, select=0), pnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), atol=tol
        )
        assert pnp.allclose(jac_fn(coeffs, select=1), pnp.array([[0.0, 0.0, 1.0]]), atol=tol)

    @pytest.mark.jax
    def test_differentiation_jax(self, tol):
        """Test that grouping is differentiable with jax tensors as coefficient"""
        import jax
        import jax.numpy as jnp

        coeffs = jnp.array([1.0, 2.0, 3.0])
        obs = [PauliX(wires=0), PauliX(wires=1), PauliZ(wires=1)]

        def group(coeffs, select=None):
            _, grouped_coeffs = group_observables(obs, coeffs)
            return grouped_coeffs[select]

        jac_fn = jax.jacobian(group)
        assert np.allclose(
            jac_fn(coeffs, select=0), pnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), atol=tol
        )
        assert np.allclose(jac_fn(coeffs, select=1), pnp.array([[0.0, 0.0, 1.0]]), atol=tol)

    @pytest.mark.torch
    def test_differentiation_torch(self, tol):
        """Test that grouping is differentiable with torch tensors as coefficient"""
        import torch

        obs = [PauliX(wires=0), PauliX(wires=1), PauliZ(wires=1)]

        def group(coeffs, select_group=None, select_index=None):
            # we return a scalar, since torch is best at computing gradients
            _, grouped_coeffs = group_observables(obs, coeffs)
            return grouped_coeffs[select_group][select_index]

        coeffs = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        res = group(coeffs, select_group=0, select_index=0)
        res.backward()
        assert np.allclose(coeffs.grad, [1.0, 0.0, 0.0], atol=tol)

        coeffs = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        res = group(coeffs, select_group=0, select_index=1)
        res.backward()
        assert np.allclose(coeffs.grad, [0.0, 1.0, 0.0], atol=tol)

        coeffs = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        res = group(coeffs, select_group=1, select_index=0)
        res.backward()
        assert np.allclose(coeffs.grad, [0.0, 0.0, 1.0], atol=tol)

    @pytest.mark.tf
    def test_differentiation_tf(self, tol):
        """Test that grouping is differentiable with tf tensors as coefficient"""
        import tensorflow as tf

        obs = [PauliX(wires=0), PauliX(wires=1), PauliZ(wires=1)]

        def group(coeffs, select=None):
            _, grouped_coeffs = group_observables(obs, coeffs)
            return grouped_coeffs[select]

        coeffs = tf.Variable([1.0, 2.0, 3.0], dtype=tf.double)

        with tf.GradientTape() as tape:
            res = group(coeffs, select=0)
        grad = tape.jacobian(res, [coeffs])
        assert np.allclose(grad, pnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), atol=tol)

        with tf.GradientTape() as tape:
            res = group(coeffs, select=1)
        grad = tape.jacobian(res, [coeffs])
        assert np.allclose(grad, pnp.array([[0.0, 0.0, 1.0]]), atol=tol)
