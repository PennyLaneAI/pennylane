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
Unit tests for the :mod:`pauli` utility functions in ``pauli/utils.py``.
"""
# pylint: disable=too-few-public-methods,too-many-public-methods
import numpy as np
import pytest

import pennylane as qml
from pennylane import (
    RX,
    RY,
    U3,
    Hadamard,
    Hamiltonian,
    Hermitian,
    Identity,
    PauliX,
    PauliY,
    PauliZ,
    is_commuting,
)
from pennylane.operation import Tensor
from pennylane.pauli import (
    are_identical_pauli_words,
    are_pauli_words_qwc,
    binary_to_pauli,
    diagonalize_pauli_word,
    diagonalize_qwc_pauli_words,
    is_pauli_word,
    is_qwc,
    observables_to_binary_matrix,
    partition_pauli_group,
    pauli_group,
    pauli_to_binary,
    pauli_word_to_matrix,
    pauli_word_to_string,
    qwc_complement_adj_matrix,
    qwc_rotation,
    simplify,
    string_to_pauli_word,
)

non_pauli_words = [
    PauliX(0) @ Hadamard(1) @ Identity(2),
    Hadamard("a"),
    U3(0.1, 1, 1, wires="a"),
    Hermitian(np.array([[3.2, 1.1 + 0.6j], [1.1 - 0.6j, 3.2]]), wires="a") @ PauliX("b"),
]


def _make_pauli_word_strings():
    return [
        (PauliX(0), {0: 0}, "X"),
        (Identity(0), {0: 0}, "I"),
        (PauliZ(0) @ PauliY(1), {0: 0, 1: 1}, "ZY"),
        (PauliX(1), {0: 0, 1: 1}, "IX"),
        (PauliX(1), None, "X"),
        (PauliX(1), {1: 0, 0: 1}, "XI"),
        (PauliZ("a") @ PauliY("b") @ PauliZ("d"), {"a": 0, "b": 1, "c": 2, "d": 3}, "ZYIZ"),
        (PauliZ("a") @ PauliY("b") @ PauliZ("d"), None, "ZYZ"),
        (PauliX("a") @ PauliY("b") @ PauliZ("d"), {"d": 0, "c": 1, "b": 2, "a": 3}, "ZIYX"),
        (4.5 * PauliX(0), {0: 0}, "X"),
        (qml.prod(PauliX(0), PauliY(1)), {0: 0, 1: 1}, "XY"),
        (PauliX(0) @ PauliZ(0), {0: 0}, "Y"),
        (3 * PauliZ(0) @ PauliY(3), {0: 0, 3: 1}, "ZY"),
        (qml.s_prod(8, qml.PauliX(0) @ qml.PauliZ(1)), {0: 0, 1: 1}, "XZ"),
        (qml.Hamiltonian([1], [qml.X(0) @ qml.Y(1)]), None, "XY"),
    ]


class TestGroupingUtils:
    """Basic usage and edge-case tests for the measurement optimization utility functions."""

    ops_to_vecs_explicit_wires = [
        (PauliX(0) @ PauliY(1) @ PauliZ(2), np.array([1, 1, 0, 0, 1, 1])),
        (PauliZ(0) @ PauliY(2), np.array([0, 1, 1, 1])),
        (PauliY(1) @ PauliX(2), np.array([1, 1, 1, 0])),
        (Identity(0), np.zeros(2)),
        (qml.prod(qml.PauliX(0), qml.PauliZ(1), qml.PauliY(2)), np.array([1, 0, 1, 0, 1, 1])),
        (qml.s_prod(1.5, qml.PauliZ(0)), np.array([0, 1])),
        (qml.sum(qml.PauliX(0), qml.PauliX(0), qml.PauliX(0)), np.array([1, 0])),
    ]

    @pytest.mark.parametrize("op,vec", ops_to_vecs_explicit_wires)
    def test_pauli_to_binary_no_wire_map(self, op, vec):
        """Test conversion of Pauli word from operator to binary vector representation when no
        ``wire_map`` is specified."""

        assert (pauli_to_binary(op) == vec).all()

    ops_to_vecs_abstract_wires = [
        (PauliX("a") @ PauliZ("b") @ Identity("c"), np.array([1, 0, 0, 0, 0, 1, 0, 0])),
        (PauliY(6) @ PauliZ("a") @ PauliZ("b"), np.array([0, 0, 0, 1, 1, 1, 0, 1])),
        (PauliX("b") @ PauliY("c"), np.array([0, 1, 1, 0, 0, 0, 1, 0])),
        (Identity("a") @ Identity(6), np.zeros(8)),
        (
            qml.prod(qml.PauliX("a"), qml.PauliZ("b"), qml.PauliY(6)),
            np.array([1, 0, 0, 1, 0, 1, 0, 1]),
        ),
        (qml.s_prod(1.5, qml.PauliZ(6)), np.array([0, 0, 0, 0, 0, 0, 0, 1])),
        (
            qml.sum(qml.PauliX("b"), qml.PauliX("b"), qml.PauliX("b")),
            np.array([0, 1, 0, 0, 0, 0, 0, 0]),
        ),
    ]

    @pytest.mark.parametrize("op,vec", ops_to_vecs_abstract_wires)
    def test_pauli_to_binary_with_wire_map(self, op, vec):
        """Test conversion of Pauli word from operator to binary vector representation if a
        ``wire_map`` is specified."""

        wire_map = {"a": 0, "b": 1, "c": 2, 6: 3}

        assert (pauli_to_binary(op, wire_map=wire_map) == vec).all()

    vecs_to_ops_explicit_wires = [
        (np.array([1, 0, 1, 0, 0, 1]), PauliX(0) @ PauliY(2)),
        (np.array([1, 1, 1, 1, 1, 1]), PauliY(0) @ PauliY(1) @ PauliY(2)),
        (np.array([1, 0, 1, 0, 1, 1]), PauliX(0) @ PauliZ(1) @ PauliY(2)),
        (np.zeros(6), Identity(0)),
    ]

    @pytest.mark.parametrize("non_pauli_word", non_pauli_words)
    def test_pauli_to_binary_non_pauli_word_catch(self, non_pauli_word):
        """Tests TypeError raise for when non Pauli-word Pennylane operations/operators are given
        as input to pauli_to_binary."""

        assert pytest.raises(TypeError, pauli_to_binary, non_pauli_word)

    def test_pauli_to_binary_incompatable_wire_map_n_qubits(self):
        """Tests ValueError raise when n_qubits is not high enough to support the highest wire_map
        value."""

        pauli_word = PauliX("a") @ PauliY("b") @ PauliZ("c")
        wire_map = {"a": 0, "b": 1, "c": 3}
        n_qubits = 3
        assert pytest.raises(ValueError, pauli_to_binary, pauli_word, n_qubits, wire_map)

    @pytest.mark.parametrize("pauli_word,binary_pauli", ops_to_vecs_explicit_wires)
    def test_pauli_to_binary_no_check(self, pauli_word, binary_pauli):
        """Tests that pauli_to_binary runs well when pauli words are provided and
        check_is_pauli_word is False."""

        assert (pauli_to_binary(pauli_word, check_is_pauli_word=False) == binary_pauli).all()

    @pytest.mark.parametrize("vec,op", vecs_to_ops_explicit_wires)
    def test_binary_to_pauli_no_wire_map(self, vec, op):
        """Test conversion of Pauli in binary vector representation to operator form when no
        ``wire_map`` is specified."""

        assert are_identical_pauli_words(binary_to_pauli(vec), op)

    vecs_to_ops_abstract_wires = [
        (np.array([1, 0, 1, 0, 0, 1]), PauliX("alice") @ PauliY("ancilla")),
        (np.array([1, 1, 1, 1, 1, 1]), PauliY("alice") @ PauliY("bob") @ PauliY("ancilla")),
        (np.array([1, 0, 1, 0, 1, 0]), PauliX("alice") @ PauliZ("bob") @ PauliX("ancilla")),
        (np.zeros(6), Identity("alice")),
    ]

    @pytest.mark.parametrize("vec,op", vecs_to_ops_abstract_wires)
    def test_binary_to_pauli_with_wire_map(self, vec, op):
        """Test conversion of Pauli in binary vector representation to operator form when
        ``wire_map`` is specified."""

        wire_map = {"alice": 0, "bob": 1, "ancilla": 2}

        assert are_identical_pauli_words(binary_to_pauli(vec, wire_map=wire_map), op)

    binary_vecs_with_invalid_wire_maps = [
        ([1, 0], {"a": 1}),
        ([1, 1, 1, 0], {"a": 0}),
        ([1, 0, 1, 0, 1, 1], {"a": 0, "b": 2, "c": 3}),
        ([1, 0, 1, 0], {"a": 0, "b": 2}),
    ]

    @pytest.mark.parametrize("binary_vec,wire_map", binary_vecs_with_invalid_wire_maps)
    def test_binary_to_pauli_invalid_wire_map(self, binary_vec, wire_map):
        """Tests ValueError raise when wire_map values are not integers 0 to N, for input 2N
        dimensional binary vector."""

        assert pytest.raises(ValueError, binary_to_pauli, binary_vec, wire_map)

    not_binary_symplectic_vecs = [[1, 0, 1, 1, 0], [1], [2, 0, 0, 1], [0.1, 4.3, 2.0, 1.3]]

    @pytest.mark.parametrize("not_binary_symplectic", not_binary_symplectic_vecs)
    def test_binary_to_pauli_with_illegal_vectors(self, not_binary_symplectic):
        """Test ValueError raise for when non even-dimensional binary vectors are given to
        binary_to_pauli."""

        assert pytest.raises(ValueError, binary_to_pauli, not_binary_symplectic)

    def test_observables_to_binary_matrix(self):
        """Test conversion of list of Pauli word operators to representation as a binary matrix."""

        observables = [Identity(1), PauliX(1), PauliZ(0) @ PauliZ(1)]

        binary_observables = np.array(
            [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]
        ).T

        assert (observables_to_binary_matrix(observables) == binary_observables).all()

    def test_observables_to_binary_matrix_n_qubits_arg(self):
        """Tests if ValueError is raised when specified n_qubits is not large enough to support
        the number of distinct wire labels in input observables."""

        observables = [Identity(1) @ PauliZ("a"), PauliX(1), PauliZ(0) @ PauliZ(2)]
        n_qubits_invalid = 3

        assert pytest.raises(
            ValueError, observables_to_binary_matrix, observables, n_qubits_invalid
        )

    @pytest.mark.usefixtures("use_legacy_opmath")
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

    obs_lsts = [
        ([qml.PauliZ(0) @ qml.PauliX(1), qml.PauliY(2), qml.PauliX(1) @ qml.PauliY(2)], True),
        ([qml.PauliZ(0) @ qml.Identity(1), qml.PauliY(2), qml.PauliX(2) @ qml.PauliY(1)], False),
        (
            [
                qml.PauliZ(0) @ qml.PauliX(1),
                qml.PauliY(2),
                qml.Identity(1) @ qml.PauliY(2),
                qml.Identity(0),
            ],
            True,
        ),  # multi I
        ([qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliZ(2), qml.PauliX(1) @ qml.PauliY(2)], False),
        # The following are use cases for `expand_tape`, so should be tested, even though Hadamard is not a Pauli op
        ([qml.Hadamard(0) @ qml.PauliX(1), qml.Identity(0)], True),
        ([qml.Hadamard(0) @ qml.PauliX(1), qml.PauliZ(0)], False),
    ]

    @pytest.mark.parametrize("obs_lst, expected_qwc", obs_lsts)
    def test_are_qwc_pauli_words(self, obs_lst, expected_qwc):
        """Given a list of Pauli words test that this function accurately
        determines if they are pairwise qubit-wise commuting."""
        qwc = are_pauli_words_qwc(obs_lst)
        assert qwc == expected_qwc

    @pytest.mark.parametrize(
        "ops",
        (
            [qml.PauliX(0), qml.sum(qml.PauliX(2), qml.PauliZ(1))],
            [qml.sum(qml.prod(qml.PauliX(0), qml.PauliY(1)), 2 * qml.PauliZ(0))],
            [qml.sum(qml.PauliY(0), qml.PauliX(1), qml.PauliZ(2))],
            [qml.prod(qml.sum(qml.PauliX(0), qml.PauliY(1)), 2 * qml.PauliZ(0)), qml.PauliZ(0)],
        ),
    )
    def test_are_pauli_words_qwc_sum_false(self, ops):
        """Test that using operators with pauli rep containing more than one term with
        are_pauli_words_qwc always returns False."""

        assert are_pauli_words_qwc(ops) is False

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

    obs_pw_status = (
        (PauliX(0), True),
        (PauliZ(1) @ PauliX(2) @ PauliZ(4), True),
        (PauliX(1) @ Hadamard(4), False),
        (Hadamard(0), False),
        (Hamiltonian([], []), False),
        (Hamiltonian([0.5], [PauliZ(1) @ PauliX(2)]), True),
        (Hamiltonian([0.5], [PauliZ(1) @ PauliX(1)]), True),
        (Hamiltonian([1.0], [Hadamard(0)]), False),
        (Hamiltonian([1.0, 0.5], [PauliX(0), PauliZ(1) @ PauliX(2)]), False),
        (qml.prod(qml.PauliX(0), qml.PauliY(0)), True),
        (qml.prod(qml.PauliX(0), qml.PauliY(1)), True),
        (qml.prod(qml.PauliX(0), qml.Hadamard(1)), False),
        (qml.s_prod(5, qml.PauliX(0) @ qml.PauliZ(1)), True),
        (qml.s_prod(5, qml.Hadamard(0)), False),
        (qml.sum(qml.PauliX(0), qml.PauliY(0)), False),
        (qml.sum(qml.s_prod(0.5, qml.PauliX(1)), qml.PauliX(1)), True),
        (qml.sum(qml.s_prod(0.5, qml.Hadamard(1)), qml.PauliX(1)), False),
        (qml.sum(qml.Hadamard(0), qml.Hadamard(0)), False),
    )

    @pytest.mark.parametrize("ob, is_pw", obs_pw_status)
    def test_is_pauli_word(self, ob, is_pw):
        """Test for determining whether input ``Observable`` instance is a Pauli word."""
        assert is_pauli_word(ob) == is_pw

    def test_is_pauli_word_non_observable(self):
        """Test that non-observables are not Pauli Words."""

        class DummyOp(qml.operation.Operator):
            num_wires = 1

        assert not is_pauli_word(DummyOp(1))

    def test_are_identical_pauli_words(self):
        """Tests for determining if two Pauli words have the same ``wires`` and ``name`` attributes."""

        pauli_word_1 = Tensor(PauliX(0))
        pauli_word_2 = PauliX(0)

        assert are_identical_pauli_words(pauli_word_1, pauli_word_2)
        assert are_identical_pauli_words(pauli_word_2, pauli_word_1)

        pauli_word_1 = PauliX(0) @ PauliY(1)
        pauli_word_2 = PauliY(1) @ PauliX(0)
        pauli_word_3 = Tensor(PauliX(0), PauliY(1))
        pauli_word_4 = PauliX(1) @ PauliZ(2)
        pauli_word_5 = qml.s_prod(1.5, qml.PauliX(0))
        pauli_word_6 = qml.sum(qml.s_prod(0.5, qml.PauliX(0)), qml.s_prod(1.0, qml.PauliX(0)))
        pauli_word_7 = qml.s_prod(2.2, qml.prod(qml.PauliX(0), qml.PauliY(1)))

        assert are_identical_pauli_words(pauli_word_1, pauli_word_2)
        assert are_identical_pauli_words(pauli_word_1, pauli_word_3)
        assert not are_identical_pauli_words(pauli_word_1, pauli_word_4)
        assert not are_identical_pauli_words(pauli_word_3, pauli_word_4)
        assert are_identical_pauli_words(pauli_word_5, pauli_word_6)
        assert are_identical_pauli_words(pauli_word_7, pauli_word_1)
        assert not are_identical_pauli_words(pauli_word_7, pauli_word_4)
        assert not are_identical_pauli_words(pauli_word_6, pauli_word_4)

    @pytest.mark.usefixtures("use_legacy_opmath")
    def test_are_identical_pauli_words_hamiltonian_unsupported(self):
        """Test that using Hamiltonians that are valid Pauli words with are_identical_pauli_words
        always returns False"""
        pauli_word_1 = qml.Hamiltonian([1.0], [qml.PauliX(0)])
        pauli_word_2 = qml.PauliX(0)
        assert not are_identical_pauli_words(pauli_word_1, pauli_word_2)

    def test_identities_always_pauli_words(self):
        """Tests that identity terms are always identical."""
        assert are_identical_pauli_words(qml.Identity(0), qml.Identity("a"))
        assert are_identical_pauli_words(qml.Identity((-2, -3)), qml.Identity("wire"))

    @pytest.mark.parametrize("non_pauli_word", non_pauli_words)
    def test_are_identical_pauli_words_non_pauli_word_catch(self, non_pauli_word):
        """Tests TypeError raise for when non-Pauli word Pennylane operators/operations are given
        as input to are_identical_pauli_words."""

        with pytest.raises(TypeError):
            are_identical_pauli_words(non_pauli_word, PauliZ(0) @ PauliZ(1))

        with pytest.raises(TypeError):
            are_identical_pauli_words(non_pauli_word, PauliZ(0) @ PauliZ(1))

        with pytest.raises(TypeError):
            are_identical_pauli_words(PauliX("a") @ Identity("b"), non_pauli_word)

        with pytest.raises(TypeError):
            are_identical_pauli_words(non_pauli_word, non_pauli_word)

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

    PAULI_WORD_STRINGS = _make_pauli_word_strings()

    @pytest.mark.usefixtures("use_legacy_and_new_opmath")
    @pytest.mark.parametrize("pauli_word,wire_map,expected_string", PAULI_WORD_STRINGS)
    def test_pauli_word_to_string(self, pauli_word, wire_map, expected_string):
        """Test that Pauli words are correctly converted into strings."""
        obtained_string = pauli_word_to_string(pauli_word, wire_map)
        assert obtained_string == expected_string

    def test_pauli_word_to_string_tensor(self):
        """Test pauli_word_to_string with tensor instances."""
        op = qml.operation.Tensor(qml.X(0), qml.Y(1))
        assert pauli_word_to_string(op) == "XY"

        op = qml.operation.Tensor(qml.Z(0), qml.Y(1), qml.X(2))
        assert pauli_word_to_string(op) == "ZYX"

    with qml.operation.disable_new_opmath_cm():
        PAULI_WORD_STRINGS_LEGACY = _make_pauli_word_strings()

    @pytest.mark.usefixtures("use_legacy_opmath")
    @pytest.mark.parametrize("pauli_word,wire_map,expected_string", PAULI_WORD_STRINGS_LEGACY)
    def test_pauli_word_to_string_legacy_opmath(self, pauli_word, wire_map, expected_string):
        """Test that Pauli words are correctly converted into strings."""
        obtained_string = pauli_word_to_string(pauli_word, wire_map)
        assert obtained_string == expected_string

    @pytest.mark.parametrize("non_pauli_word", non_pauli_words)
    def test_pauli_word_to_string_invalid_input(self, non_pauli_word):
        """Ensure invalid inputs are handled properly when converting Pauli words to strings."""
        with pytest.raises(TypeError):
            pauli_word_to_string(non_pauli_word)

    @pytest.mark.parametrize(
        "pauli_string,wire_map,expected_pauli",
        [
            ("I", {"a": 0}, Identity("a")),
            ("X", {0: 0}, PauliX(0)),
            ("XI", {1: 0, 0: 1}, PauliX(1)),
            ("II", {0: 0, 1: 1}, Identity(0)),
            ("ZYIZ", {"a": 0, "b": 1, "c": 2, "d": 3}, PauliZ("a") @ PauliY("b") @ PauliZ("d")),
            ("ZYZ", None, PauliZ(0) @ PauliY(1) @ PauliZ(2)),
            ("ZIYX", {"d": 0, "c": 1, "b": 2, "a": 3}, PauliZ("d") @ PauliY("b") @ PauliX("a")),
        ],
    )
    def test_string_to_pauli_word(self, pauli_string, wire_map, expected_pauli):
        """Test that valid strings are correctly converted into Pauli words."""
        obtained_pauli = string_to_pauli_word(pauli_string, wire_map)
        assert qml.equal(obtained_pauli, expected_pauli)

    @pytest.mark.parametrize(
        "non_pauli_string,wire_map,error_type,error_message",
        [
            (Identity("a"), None, TypeError, "must be string"),
            ("XAYZ", None, ValueError, "Invalid characters encountered"),
            ("XYYZ", {0: 0, 1: 1, 2: 2}, ValueError, "must have the same length"),
        ],
    )
    def test_string_to_pauli_word_invalid_input(
        self, non_pauli_string, wire_map, error_type, error_message
    ):
        """Ensure invalid inputs are handled properly when converting strings to Pauli words."""
        with pytest.raises(error_type, match=error_message):
            string_to_pauli_word(non_pauli_string, wire_map)

    @pytest.mark.parametrize(
        "pauli_word,wire_map,expected_matrix",
        [
            (PauliX(0), {0: 0}, PauliX(0).matrix()),
            # (
            #     Identity(0),
            #     {0: 0},
            #     np.eye(2),
            # ),  # TODO update PauliSentence.to_mat to handle Identities better https://github.com/PennyLaneAI/pennylane/issues/5354
            (
                PauliZ(0) @ PauliY(1),
                {0: 0, 1: 1},
                np.array([[0, -1j, 0, 0], [1j, 0, 0, 0], [0, 0, 0, 1j], [0, 0, -1j, 0]]),
            ),
            (
                PauliY(1) @ PauliZ(0),
                {0: 0, 1: 1},
                np.array([[0, -1j, 0, 0], [1j, 0, 0, 0], [0, 0, 0, 1j], [0, 0, -1j, 0]]),
            ),
            (
                PauliY(1) @ PauliZ(0),
                {1: 0, 0: 1},
                np.array([[0, 0, -1j, 0], [0, 0, 0, 1j], [1j, 0, 0, 0], [0, -1j, 0, 0]]),
            ),
            # (Identity(0), {0: 0, 1: 1}, np.eye(4)), # TODO update PauliSentence.to_mat to handle Identities better https://github.com/PennyLaneAI/pennylane/issues/5354
            (PauliX(2), None, PauliX(2).matrix()),
            (
                PauliX(2),
                {0: 0, 1: 1, 2: 2},
                np.array(
                    [
                        [0, 1, 0, 0, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0, 0, 1, 0],
                    ]
                ),
            ),
            (
                PauliZ("a") @ PauliX(2),
                {"a": 0, 1: 1, 2: 2},
                np.array(
                    [
                        [0, 1, 0, 0, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, -1, 0, 0],
                        [0, 0, 0, 0, -1, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, -1],
                        [0, 0, 0, 0, 0, 0, -1, 0],
                    ]
                ),
            ),
            (
                PauliX(2) @ PauliZ("a"),
                {"a": 0, 1: 1, 2: 2},
                np.array(
                    [
                        [0, 1, 0, 0, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, -1, 0, 0],
                        [0, 0, 0, 0, -1, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, -1],
                        [0, 0, 0, 0, 0, 0, -1, 0],
                    ]
                ),
            ),
        ],
    )
    def test_pauli_word_to_matrix(self, pauli_word, wire_map, expected_matrix):
        """Test that Pauli words are correctly converted into matrices."""

        obtained_matrix = pauli_word_to_matrix(pauli_word, wire_map)
        assert np.allclose(obtained_matrix, expected_matrix)

    @pytest.mark.parametrize("non_pauli_word", non_pauli_words)
    def test_pauli_word_to_matrix_invalid_input(self, non_pauli_word):
        """Ensure invalid inputs are handled properly when converting Pauli words to matrices."""
        with pytest.raises(TypeError):
            pauli_word_to_matrix(non_pauli_word)


class TestPauliGroup:
    """Testing for Pauli group construction and manipulation functions."""

    def test_pauli_group_size(self):
        """Test that the size of the returned Pauli group is correct."""
        for n_qubits in range(1, 5):
            pg = list(pauli_group(n_qubits))
            assert len(pg) == 4**n_qubits

    def test_pauli_group_invalid_input(self):
        """Test that invalid inputs to the Pauli group are handled correctly."""
        with pytest.raises(TypeError, match="Must specify an integer number"):
            pauli_group("3")

        with pytest.raises(ValueError, match="Number of qubits must be at least 1"):
            pauli_group(-1)

    def test_one_qubit_pauli_group_integer_wire_map(self):
        """Test that the single-qubit Pauli group is constructed correctly with a wire
        labeled by an integer."""

        expected_pg_1 = [Identity(0), PauliZ(0), PauliX(0), PauliY(0)]
        pg_1 = list(pauli_group(1))
        assert all(expected.compare(obtained) for expected, obtained in zip(expected_pg_1, pg_1))

    def test_one_qubit_pauli_group_valid_float_input(self):
        """Test that the single-qubit Pauli group is constructed correctly when a float
        that represents an integer is passed."""

        expected_pg_1 = [Identity(0), PauliZ(0), PauliX(0), PauliY(0)]
        pg_1 = list(pauli_group(1.0))
        assert all(expected.compare(obtained) for expected, obtained in zip(expected_pg_1, pg_1))

    def test_one_qubit_pauli_group_string_wire_map(self):
        """Test that the single-qubit Pauli group is constructed correctly with a wire
        labeled by a string."""

        wire_map = {"qubit": 0}
        expected_pg_1_wires = [
            Identity("qubit"),
            PauliZ("qubit"),
            PauliX("qubit"),
            PauliY("qubit"),
        ]
        pg_1_wires = list(pauli_group(1, wire_map=wire_map))
        assert all(exp.compare(ob) for exp, ob in zip(expected_pg_1_wires, pg_1_wires))

    def test_two_qubit_pauli_group(self):
        """Test that the two-qubit Pauli group is constructed correctly."""
        # With no wire map; ordering is based on construction from binary representation
        wire_map = {"a": 0, "b": 1}

        expected_pg_2 = [
            Identity("a"),
            PauliZ("b"),
            PauliZ("a"),
            PauliZ("a") @ PauliZ("b"),
            PauliX("b"),
            PauliY("b"),
            PauliZ("a") @ PauliX("b"),
            PauliZ("a") @ PauliY("b"),
            PauliX("a"),
            PauliX("a") @ PauliZ("b"),
            PauliY("a"),
            PauliY("a") @ PauliZ("b"),
            PauliX("a") @ PauliX("b"),
            PauliX("a") @ PauliY("b"),
            PauliY("a") @ PauliX("b"),
            PauliY("a") @ PauliY("b"),
        ]

        pg_2 = list(pauli_group(2, wire_map=wire_map))
        assert all(qml.equal(expected, obtained) for expected, obtained in zip(expected_pg_2, pg_2))

    @pytest.mark.parametrize(
        "pauli_word_1,pauli_word_2,expected_product",
        [
            (PauliX(0), Identity(0), PauliX(0)),
            (PauliZ(0), PauliY(0), PauliX(0)),
            (PauliZ(0), PauliZ(0), Identity(0)),
            (Identity("a"), Identity("b"), Identity(["a", "b"])),
            (PauliZ("b") @ PauliY("a"), PauliZ("b") @ PauliY("a"), Identity(["b", "a"])),
            (
                PauliZ("b") @ PauliY("a"),
                PauliZ("b") @ PauliY("a"),
                Identity(["b", "a"]),
            ),
            (
                PauliZ(0) @ PauliY(1),
                PauliX(0) @ PauliZ(1),
                PauliY(0) @ PauliX(1),
            ),
            (PauliZ(0) @ PauliY(1), PauliX(1) @ PauliY(0), PauliX(0) @ PauliZ(1)),
            (
                PauliZ(0) @ PauliY(3) @ PauliZ(1),
                PauliX(1) @ PauliX(2) @ PauliY(0),
                PauliX(0) @ PauliY(1) @ PauliX(2) @ PauliY(3),
            ),
            (
                PauliX(0) @ PauliX(2),
                PauliX(0) @ PauliZ(2),
                PauliY(2),
            ),
            (PauliZ("a"), PauliX("b"), PauliZ("a") @ PauliX("b")),
            (
                PauliZ("a"),
                PauliX("e"),
                PauliZ("a") @ PauliX("e"),
            ),
            (PauliZ("a"), PauliY("e"), PauliZ("a") @ PauliY("e")),
            (
                PauliZ(0) @ PauliZ(2) @ PauliZ(4),
                PauliZ(0) @ PauliY(1) @ PauliX(3),
                PauliZ(2) @ PauliY(1) @ PauliZ(4) @ PauliX(3),
            ),
            (
                PauliZ(0) @ PauliZ(4) @ PauliZ(2),
                PauliZ(0) @ PauliY(1) @ PauliX(3),
                PauliZ(2) @ PauliY(1) @ PauliZ(4) @ PauliX(3),
            ),
            (
                PauliZ(0) @ PauliY(3) @ PauliZ(1),
                PauliX(1) @ PauliX(2) @ PauliY(0),
                PauliX(0) @ PauliY(3) @ PauliY(1) @ PauliX(2),
            ),
        ],
    )
    def test_pauli_mult_using_prod(self, pauli_word_1, pauli_word_2, expected_product):
        """Test that Pauli words are multiplied together correctly."""
        obtained_product = qml.prod(pauli_word_1, pauli_word_2).simplify()
        if isinstance(obtained_product, qml.ops.SProd):  # don't care about phase here
            obtained_product = obtained_product.base
        assert obtained_product == qml.operation.convert_to_opmath(expected_product)

    @pytest.mark.parametrize(
        "pauli_word_1,pauli_word_2,expected_phase",
        [
            (PauliX(0), Identity(0), 1),
            (PauliZ(0), PauliY(0), -1j),
            (PauliZ(0), PauliZ(0), 1),
            (Identity("a"), Identity("b"), 1),
            (PauliZ("b") @ PauliY("a"), PauliZ("b") @ PauliY("a"), 1),
            (PauliZ(0), PauliY("b"), 1),
            (PauliZ("a") @ PauliY("b"), PauliX("a") @ PauliZ("b"), -1),
            (PauliX(0) @ PauliX(2), PauliX(0) @ PauliZ(2), -1j),
            (PauliX(0) @ PauliY(1) @ PauliZ(2), PauliY(0) @ PauliY(1), 1j),
            (PauliZ(0) @ PauliY(1), PauliX(1) @ PauliY(0), -1),
            (PauliZ(0) @ PauliY(1), PauliX(1) @ PauliY(0), -1),
            (PauliZ(0) @ PauliY(3) @ PauliZ(1), PauliX(1) @ PauliX(2) @ PauliY(0), 1),
        ],
    )
    def test_pauli_mult_with_phase_using_prod(self, pauli_word_1, pauli_word_2, expected_phase):
        """Test that multiplication including phases works as expected."""
        prod = qml.prod(pauli_word_1, pauli_word_2).simplify()
        obtained_phase = prod.scalar if isinstance(prod, qml.ops.SProd) else 1
        assert obtained_phase == expected_phase


class TestPartitionPauliGroup:
    """Tests for the partition_pauli_group function"""

    def test_expected_answer(self):
        """Test if we get the expected answer for 3 qubits"""
        expected = [
            ["III", "IIZ", "IZI", "IZZ", "ZII", "ZIZ", "ZZI", "ZZZ"],
            ["IIX", "IZX", "ZIX", "ZZX"],
            ["IIY", "IZY", "ZIY", "ZZY"],
            ["IXI", "IXZ", "ZXI", "ZXZ"],
            ["IXX", "ZXX"],
            ["IXY", "ZXY"],
            ["IYI", "IYZ", "ZYI", "ZYZ"],
            ["IYX", "ZYX"],
            ["IYY", "ZYY"],
            ["XII", "XIZ", "XZI", "XZZ"],
            ["XIX", "XZX"],
            ["XIY", "XZY"],
            ["XXI", "XXZ"],
            ["XXX"],
            ["XXY"],
            ["XYI", "XYZ"],
            ["XYX"],
            ["XYY"],
            ["YII", "YIZ", "YZI", "YZZ"],
            ["YIX", "YZX"],
            ["YIY", "YZY"],
            ["YXI", "YXZ"],
            ["YXX"],
            ["YXY"],
            ["YYI", "YYZ"],
            ["YYX"],
            ["YYY"],
        ]
        assert expected == partition_pauli_group(3.0)

    @pytest.mark.parametrize("n", range(1, 9))
    def test_scaling(self, n):
        """Test if the number of groups is equal to 3**n"""
        assert len(partition_pauli_group(n)) == 3**n

    @pytest.mark.usefixtures("use_legacy_opmath")
    @pytest.mark.parametrize("n", range(1, 6))
    def test_is_qwc_legacy_opmath(self, n):
        """Test if each group contains only qubit-wise commuting terms"""
        for group in partition_pauli_group(n):
            size = len(group)
            for i in range(size):
                for j in range(i, size):
                    s1 = group[i]
                    s2 = group[j]
                    w1 = string_to_pauli_word(s1)
                    w2 = string_to_pauli_word(s2)
                    assert is_commuting(w1, w2)

    @pytest.mark.parametrize("n", range(2, 6))
    def test_is_qwc(self, n):
        """Test if each group contains only qubit-wise commuting terms"""
        for group in partition_pauli_group(n):
            size = len(group)
            for i in range(size):
                for j in range(i, size):
                    s1 = group[i]
                    s2 = group[j]
                    w1 = string_to_pauli_word(s1)
                    w2 = string_to_pauli_word(s2)
                    assert is_commuting(w1, w2)

    def test_invalid_input(self):
        """Test that invalid inputs are handled correctly."""
        with pytest.raises(TypeError, match="Must specify an integer number"):
            partition_pauli_group("3")

        with pytest.raises(ValueError, match="Number of qubits must be at least 0"):
            partition_pauli_group(-1)

    def test_zero(self):
        """Test if [[""]] is returned with zero qubits"""
        assert partition_pauli_group(0) == [[""]]


class TestMeasurementTransformations:
    """Tests for the functions involved in obtaining post-rotations necessary in the measurement
    optimization schemes implemented in :mod:`grouping`."""

    def are_identical_rotation_gates(self, gate_1, gate_2, param_tol=1e-6):
        """Checks whether the two input gates are identical up to a certain threshold in their
        parameters.

        Arguments:
            gate_1 (Union[RX, RY, RZ]): the first single-qubit rotation gate
            gate_2 (Union[RX, RY, RZ]): the second single-qubit rotation gate

        Keyword arguments:
            param_tol (float): the absolute tolerance for considering whether two gates parameter
                values are the same

        Returns:
            bool: whether the input rotation gates are identical up to the parameter tolerance

        """

        return (
            gate_1.wires == gate_2.wires
            and np.allclose(gate_1.parameters, gate_2.parameters, atol=param_tol, rtol=0)
            and gate_1.name == gate_2.name
        )

    qwc_rotation_io = [
        ([PauliX(0), PauliZ(1), PauliZ(3)], [RY(-np.pi / 2, wires=[0])]),
        ([Identity(0), PauliZ(1)], []),
        (
            [PauliX(2), PauliY(0), Identity(1)],
            [RY(-np.pi / 2, wires=[2]), RX(np.pi / 2, wires=[0])],
        ),
        (
            [PauliZ("a"), PauliX("b"), PauliY(0)],
            [RY(-np.pi / 2, wires=["b"]), RX(np.pi / 2, wires=[0])],
        ),
    ]

    @pytest.mark.parametrize("pauli_ops,qwc_rot_sol", qwc_rotation_io)
    def test_qwc_rotation(self, pauli_ops, qwc_rot_sol):
        """Tests that the correct single-qubit post-rotation gates are obtained for the input list
        of Pauli operators."""

        qwc_rot = qwc_rotation(pauli_ops)

        assert all(
            self.are_identical_rotation_gates(qwc_rot[i], qwc_rot_sol[i])
            for i in range(len(qwc_rot))
        )

    invalid_qwc_rotation_inputs = [
        [PauliX(0), PauliY(1), Hadamard(2)],
        [PauliX(0) @ PauliY(1), PauliZ(1), Identity(2)],
        [RX(1, wires="a"), PauliX("b")],
    ]

    @pytest.mark.parametrize("bad_input", invalid_qwc_rotation_inputs)
    def test_invalid_qwc_rotation_input_catch(self, bad_input):
        """Verifies that a TypeError is raised when the input to qwc_rotations is not a list of
        single Pauli operators."""

        assert pytest.raises(TypeError, qwc_rotation, bad_input)

    diagonalized_paulis = [
        (Identity("a"), Identity("a")),
        (PauliX(1) @ Identity(2) @ PauliZ("b"), PauliZ(1) @ PauliZ("b")),
        (PauliZ(1) @ PauliZ(2), PauliZ(1) @ PauliZ(2)),
        (PauliX("a") @ PauliY("b") @ PauliY("c"), PauliZ("a") @ PauliZ("b") @ PauliZ("c")),
    ]

    @pytest.mark.parametrize("pauli_word, diag_pauli_word", diagonalized_paulis)
    def test_diagonalize_pauli_word(self, pauli_word, diag_pauli_word):
        """Tests `diagonalize_pauli_word` returns the correct diagonal Pauli word in computational
        basis for a given Pauli word."""

        assert are_identical_pauli_words(diagonalize_pauli_word(pauli_word), diag_pauli_word)

    non_pauli_words = [
        PauliX(0) @ Hadamard(1) @ Identity(2),
        Hadamard("a"),
        U3(0.1, 1, 1, wires="a"),
        Hermitian(np.array([[3.2, 1.1 + 0.6j], [1.1 - 0.6j, 3.2]]), wires="a") @ PauliX("b"),
    ]

    @pytest.mark.parametrize("non_pauli_word", non_pauli_words)
    def test_diagonalize_pauli_word_catch_non_pauli_word(self, non_pauli_word):
        assert pytest.raises(TypeError, diagonalize_pauli_word, non_pauli_word)

    qwc_diagonalization_io = [
        (
            [PauliX(0) @ PauliY(1), PauliX(0) @ PauliZ(2)],
            (
                [
                    RX(np.pi / 2, wires=[1]),
                    RY(-np.pi / 2, wires=[0]),
                ],
                [PauliZ(wires=[0]) @ PauliZ(wires=[1]), PauliZ(wires=[0]) @ PauliZ(wires=[2])],
            ),
        ),
        (
            [PauliX(2) @ Identity(0), PauliY(1), PauliZ(0) @ PauliY(1), PauliX(2) @ PauliY(1)],
            (
                [RY(-np.pi / 2, wires=[2]), RX(np.pi / 2, wires=[1])],
                [
                    PauliZ(wires=[2]),
                    PauliZ(wires=[1]),
                    PauliZ(wires=[0]) @ PauliZ(wires=[1]),
                    PauliZ(wires=[2]) @ PauliZ(wires=[1]),
                ],
            ),
        ),
        (
            [PauliZ("a") @ PauliY("b") @ PauliZ("c"), PauliY("b") @ PauliZ("d")],
            (
                [RX(np.pi / 2, wires=["b"])],
                [
                    PauliZ(wires=["a"]) @ PauliZ(wires=["b"]) @ PauliZ(wires=["c"]),
                    PauliZ(wires=["b"]) @ PauliZ(wires=["d"]),
                ],
            ),
        ),
        ([PauliX("a")], ([RY(-np.pi / 2, wires=["a"])], [PauliZ(wires=["a"])])),
        (
            [PauliX(0), PauliX(1) @ PauliX(0)],
            (
                [RY(-1.5707963267948966, wires=[0]), RY(-1.5707963267948966, wires=[1])],
                [PauliZ(wires=[0]), PauliZ(wires=[1]) @ PauliZ(wires=[0])],
            ),
        ),
    ]

    @pytest.mark.parametrize("convert_to_opmath", (True, False))
    @pytest.mark.parametrize("qwc_grouping,qwc_sol_tuple", qwc_diagonalization_io)
    def test_diagonalize_qwc_pauli_words(self, qwc_grouping, qwc_sol_tuple, convert_to_opmath):
        """Tests for validating diagonalize_qwc_pauli_words solutions."""

        if convert_to_opmath:
            qwc_grouping = [qml.operation.convert_to_opmath(o) for o in qwc_grouping]
            diag_terms = [qml.operation.convert_to_opmath(o) for o in qwc_sol_tuple[1]]
            qwc_sol_tuple = (qwc_sol_tuple[0], diag_terms)

        qwc_rot, diag_qwc_grouping = diagonalize_qwc_pauli_words(qwc_grouping)
        qwc_rot_sol, diag_qwc_grouping_sol = qwc_sol_tuple

        assert all(
            self.are_identical_rotation_gates(qwc_rot[i], qwc_rot_sol[i])
            for i in range(len(qwc_rot))
        )
        for diag_op, expected in zip(diag_qwc_grouping, diag_qwc_grouping_sol):
            assert qml.equal(diag_op, expected)

    not_qwc_groupings = [
        [PauliX("a"), PauliY("a")],
        [PauliZ(0) @ Identity(1), PauliZ(0) @ PauliZ(1), PauliX(0) @ Identity(1)],
        [PauliX("a") @ PauliX(0), PauliZ(0) @ PauliZ("a")],
        [PauliZ("a") @ PauliY(1), PauliZ(1) @ PauliY("a")],
        [qml.prod(qml.PauliZ(0), qml.PauliX(1)), qml.prod(qml.PauliZ(1), qml.PauliX(0))],
    ]

    @pytest.mark.parametrize("not_qwc_grouping", not_qwc_groupings)
    def test_diagonalize_qwc_pauli_words_catch_when_not_qwc(self, not_qwc_grouping):
        """Test for ValueError raise when diagonalize_qwc_pauli_words is not given a list of
        qubit-wise commuting Pauli words."""

        assert pytest.raises(ValueError, diagonalize_qwc_pauli_words, not_qwc_grouping)

    @pytest.mark.usefixtures(
        "use_legacy_opmath"
    )  # Handling a LinearCombination is not a problem under new opmath anymore
    def test_diagonalize_qwc_pauli_words_catch_invalid_type(self):
        """Test for ValueError raise when diagonalize_qwc_pauli_words is given a list
        containing invalid operator types."""
        invalid_ops = [qml.PauliX(0), qml.Hamiltonian([1.0], [qml.PauliZ(1)])]

        with pytest.raises(ValueError, match="This function only supports pauli words."):
            _ = diagonalize_qwc_pauli_words(invalid_ops)


class TestObservableHF:

    with qml.operation.disable_new_opmath_cm():
        HAMILTONIAN_SIMPLIFY = [
            (
                qml.Hamiltonian(
                    np.array([0.5, 0.5]),
                    [qml.PauliX(0) @ qml.PauliY(1), qml.PauliX(0) @ qml.PauliY(1)],
                ),
                qml.Hamiltonian(np.array([1.0]), [qml.PauliX(0) @ qml.PauliY(1)]),
            ),
            (
                qml.Hamiltonian(
                    np.array([0.5, -0.5]),
                    [qml.PauliX(0) @ qml.PauliY(1), qml.PauliX(0) @ qml.PauliY(1)],
                ),
                qml.Hamiltonian([], []),
            ),
            (
                qml.Hamiltonian(
                    np.array([0.0, -0.5]),
                    [qml.PauliX(0) @ qml.PauliY(1), qml.PauliX(0) @ qml.PauliZ(1)],
                ),
                qml.Hamiltonian(np.array([-0.5]), [qml.PauliX(0) @ qml.PauliZ(1)]),
            ),
            (
                qml.Hamiltonian(
                    np.array([0.25, 0.25, 0.25, -0.25]),
                    [
                        qml.PauliX(0) @ qml.PauliY(1),
                        qml.PauliX(0) @ qml.PauliZ(1),
                        qml.PauliX(0) @ qml.PauliY(1),
                        qml.PauliX(0) @ qml.PauliY(1),
                    ],
                ),
                qml.Hamiltonian(
                    np.array([0.25, 0.25]),
                    [qml.PauliX(0) @ qml.PauliY(1), qml.PauliX(0) @ qml.PauliZ(1)],
                ),
            ),
        ]

    @pytest.mark.usefixtures("use_legacy_opmath")
    @pytest.mark.parametrize(("hamiltonian", "result"), HAMILTONIAN_SIMPLIFY)
    def test_simplify(self, hamiltonian, result):
        r"""Test that simplify returns the correct hamiltonian."""
        h = simplify(hamiltonian)
        assert h.compare(result)


class TestTapering:
    with qml.operation.disable_new_opmath_cm():
        terms_bin_mat_data = [
            (
                [
                    qml.Identity(wires=[0]),
                    qml.PauliZ(wires=[0]),
                    qml.PauliZ(wires=[1]),
                    qml.PauliZ(wires=[2]),
                    qml.PauliZ(wires=[3]),
                    qml.PauliZ(wires=[0]) @ qml.PauliZ(wires=[1]),
                    qml.PauliY(wires=[0])
                    @ qml.PauliX(wires=[1])
                    @ qml.PauliX(wires=[2])
                    @ qml.PauliY(wires=[3]),
                    qml.PauliY(wires=[0])
                    @ qml.PauliY(wires=[1])
                    @ qml.PauliX(wires=[2])
                    @ qml.PauliX(wires=[3]),
                    qml.PauliX(wires=[0])
                    @ qml.PauliX(wires=[1])
                    @ qml.PauliY(wires=[2])
                    @ qml.PauliY(wires=[3]),
                    qml.PauliX(wires=[0])
                    @ qml.PauliY(wires=[1])
                    @ qml.PauliY(wires=[2])
                    @ qml.PauliX(wires=[3]),
                    qml.PauliZ(wires=[0]) @ qml.PauliZ(wires=[2]),
                    qml.PauliZ(wires=[0]) @ qml.PauliZ(wires=[3]),
                    qml.PauliZ(wires=[1]) @ qml.PauliZ(wires=[2]),
                    qml.PauliZ(wires=[1]) @ qml.PauliZ(wires=[3]),
                    qml.PauliZ(wires=[2]) @ qml.PauliZ(wires=[3]),
                ],
                4,
                np.array(
                    [
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 0, 0],
                        [1, 1, 0, 0, 0, 0, 0, 0],
                        [1, 0, 0, 1, 1, 1, 1, 1],
                        [1, 1, 0, 0, 1, 1, 1, 1],
                        [0, 0, 1, 1, 1, 1, 1, 1],
                        [0, 1, 1, 0, 1, 1, 1, 1],
                        [1, 0, 1, 0, 0, 0, 0, 0],
                        [1, 0, 0, 1, 0, 0, 0, 0],
                        [0, 1, 1, 0, 0, 0, 0, 0],
                        [0, 1, 0, 1, 0, 0, 0, 0],
                        [0, 0, 1, 1, 0, 0, 0, 0],
                    ]
                ),
            ),
            (
                [
                    qml.PauliZ(wires=["a"]) @ qml.PauliX(wires=["b"]),
                    qml.PauliZ(wires=["a"]) @ qml.PauliY(wires=["c"]),
                    qml.PauliX(wires=["a"]) @ qml.PauliY(wires=["d"]),
                ],
                4,
                np.array(
                    [[1, 0, 0, 0, 0, 1, 0, 0], [1, 0, 1, 0, 0, 0, 1, 0], [0, 0, 0, 1, 1, 0, 0, 1]]
                ),
            ),
        ]

    @pytest.mark.usefixtures("use_legacy_opmath")
    @pytest.mark.parametrize(("terms", "num_qubits", "result"), terms_bin_mat_data)
    def test_binary_matrix_from_pws(self, terms, num_qubits, result):
        r"""Test that _binary_matrix_from_pws returns the correct result."""
        # pylint: disable=protected-access
        pws_lst = [list(qml.pauli.pauli_sentence(t))[0] for t in terms]
        binary_matrix = qml.pauli.utils._binary_matrix_from_pws(pws_lst, num_qubits)
        assert (binary_matrix == result).all()
