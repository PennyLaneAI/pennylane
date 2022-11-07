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
import pytest
import numpy as np
import pennylane as qml
from pennylane import Identity, PauliX, PauliY, PauliZ
from pennylane.pauli import are_identical_pauli_words
from pennylane.pauli.grouping.group_observables import PauliGroupingStrategy, group_observables
from pennylane import numpy as pnp


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

    def test_construct_qwc_complement_adj_matrix_for_operators(self):
        """Constructing the complement graph adjacency matrix for a list of Pauli words according
        to qubit-wise commutativity."""

        observables = [PauliY(0), PauliZ(0) @ PauliZ(1), PauliY(0) @ PauliX(1)]
        qwc_complement_adjacency_matrix = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])

        grouping_instance = PauliGroupingStrategy(observables, "qwc")
        assert (
            grouping_instance.complement_adj_matrix_for_operator()
            == qwc_complement_adjacency_matrix
        ).all()

    def test_construct_commuting_complement_adj_matrix_for_operators(self):
        """Constructing the complement graph adjacency matrix for a list of Pauli words according
        to general commutativity."""

        observables = [PauliY(0), PauliZ(0) @ PauliZ(1), PauliY(0) @ PauliX(1)]
        commuting_complement_adjacency_matrix = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])

        grouping_instance = PauliGroupingStrategy(observables, "commuting")
        assert (
            grouping_instance.complement_adj_matrix_for_operator()
            == commuting_complement_adjacency_matrix
        ).all()

    def test_construct_anticommuting_complement_adj_matrix_for_operators(self):
        """Constructing the complement graph adjacency matrix for a list of Pauli words according
        to anticommutativity."""

        observables = [PauliY(0), PauliZ(0) @ PauliZ(1), PauliY(0) @ PauliX(1)]
        anticommuting_complement_adjacency_matrix = np.array([[0, 0, 1], [0, 0, 1], [1, 1, 0]])

        grouping_instance = PauliGroupingStrategy(observables, "anticommuting")
        assert (
            grouping_instance.complement_adj_matrix_for_operator()
            == anticommuting_complement_adjacency_matrix
        ).all()

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
        assert (
            grouping_instance.complement_adj_matrix_for_operator()
            == qwc_complement_adjacency_matrix
        ).all()

    @pytest.mark.parametrize("observables", trivial_ops)
    def test_construct_complement_commuting_adj_matrix_for_trivial_operators(self, observables):
        """Constructing the complement of commutativity graph's adjacency matrix for a list of
        identity operations and various symmetric binary relations"""

        commuting_complement_adjacency_matrix = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

        grouping_instance = PauliGroupingStrategy(observables, "commuting")
        assert (
            grouping_instance.complement_adj_matrix_for_operator()
            == commuting_complement_adjacency_matrix
        ).all()

    @pytest.mark.parametrize("observables", trivial_ops)
    def test_construct_complement_anticommuting_adj_matrix_for_trivial_operators(self, observables):
        """Constructing the complement of anticommutativity graph's adjacency matrix for a list of
        identity operations and various symmetric binary relations"""

        anticommuting_complement_adjacency_matrix = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])

        grouping_instance = PauliGroupingStrategy(observables, "anticommuting")
        assert (
            grouping_instance.complement_adj_matrix_for_operator()
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
    [PauliX([(0, 0)]), PauliZ([(0, 0)])],
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
    [[PauliX([(0, 0)])], [PauliZ([(0, 0)])]],
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
    [[PauliX([(0, 0)]), PauliZ([(0, 0)])]],
]


class TestGroupObservables:
    """
    Tests for ``group_observables`` function using QWC, commuting, and anticommuting partitioning.
    """

    qwc_tuples = [(obs, sol) for obs, sol in zip(observables_list, qwc_sols)]

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

    com_tuples = [(obs, sol) for obs, sol in zip(observables_list, commuting_sols)]

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

    anticom_tuples = [(obs, sols) for obs, sols in zip(observables_list, anticommuting_sols)]

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
