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
Unit tests for the :mod:`grouping` utility functions in `grouping/utils.py`.
"""
import pytest
import numpy as np
from pennylane import Identity, PauliX, PauliY, PauliZ, Hadamard, Hermitian, U3
from pennylane.operation import Tensor
from pennylane.wires import Wires
from pennylane.grouping.utils import (
    is_pauli_word,
    are_identical_pauli_words,
    pauli_to_binary,
    binary_to_pauli,
    is_qwc,
    convert_observables_to_binary_matrix,
)


non_pauli_words = [
    PauliX(0) @ Hadamard(1) @ Identity(2),
    Hadamard("a"),
    U3(0.1, 1, 1, wires="a"),
    Hermitian(np.array([[3.2, 1.1 + 0.6j], [1.1 - 0.6j, 3.2]]), wires="a") @ PauliX("b"),
]

class TestGroupingUtils:
    """Basic usage and edge-case tests for the measurement optimization utility functions."""

    ops_to_vecs_explicit_wires = [
        (PauliX(0) @ PauliY(1) @ PauliZ(2), np.array([1, 1, 0, 0, 1, 1])),
        (PauliZ(0) @ PauliY(2), np.array([0, 1, 1, 1])),
        (PauliY(1) @ PauliX(2), np.array([1, 1, 1, 0])),
        (Identity(0), np.zeros(2)),
    ]

    @pytest.mark.parametrize("op,vec", ops_to_vecs_explicit_wires)
    def test_pauli_to_binary_no_wire_map(self, op, vec):
        """Test conversion of Pauli word from operator to binary vector representation when no
        `wire_map` is specified."""

        assert (pauli_to_binary(op) == vec).all()

    ops_to_vecs_abstract_wires = [
        (PauliX("a") @ PauliZ("b") @ Identity("c"), np.array([1, 0, 0, 0, 0, 1, 0, 0])),
        (PauliY(6) @ PauliZ("a") @ PauliZ("b"), np.array([0, 0, 0, 1, 1, 1, 0, 1])),
        (PauliX("b") @ PauliY("c"), np.array([0, 1, 1, 0, 0, 0, 1, 0])),
        (Identity("a") @ Identity(6), np.zeros(8)),
    ]

    @pytest.mark.parametrize("op,vec", ops_to_vecs_abstract_wires)
    def test_pauli_to_binary_with_wire_map(self, op, vec):
        """Test conversion of Pauli word from operator to binary vector representation if a
        `wire_map` is specified."""

        wire_map = {Wires("a"): 0, Wires("b"): 1, Wires("c"): 2, Wires(6): 3}

        assert (pauli_to_binary(op, wire_map=wire_map) == vec).all()

    vecs_to_ops_explicit_wires = [
        (np.array([1, 0, 1, 0, 0, 1]), PauliX(0) @ PauliY(2)),
        (np.array([1, 1, 1, 1, 1, 1]), PauliY(0) @ PauliY(1) @ PauliY(2)),
        (np.array([1, 0, 1, 0, 1, 1]), PauliX(0) @ PauliZ(1) @ PauliY(2)),
        (np.zeros(6), Identity(0)),
    ]

    @pytest.mark.parametrize("non_pauli_word", non_pauli_words)
    def test_pauli_to_binary_non_pauli_word_catch(self, non_pauli_word):
        assert pytest.raises(TypeError, pauli_to_binary, non_pauli_word)

    @pytest.mark.parametrize("vec,op", vecs_to_ops_explicit_wires)
    def test_binary_to_pauli_no_wire_map(self, vec, op):
        """Test conversion of Pauli in binary vector representation to operator form when no
        `wire_map` is specified."""

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
        `wire_map` is specified."""

        wire_map = {Wires("alice"): 0, Wires("bob"): 1, Wires("ancilla"): 2}

        assert are_identical_pauli_words(binary_to_pauli(vec, wire_map=wire_map), op)

    not_binary_symplectic_vecs = [[1,0,1,1,0],
                                  [1],
                                  [2,0,0,1],
                                  [0.1,4.3,2.0,1.3]]

    @pytest.mark.parametrize("not_binary_symplectic", not_binary_symplectic_vecs)
    def test_binary_to_pauli_with_illegal_vectors(self, not_binary_symplectic):
        assert pytest.raises(ValueError, binary_to_pauli, not_binary_symplectic)

    def test_convert_observables_to_binary_matrix(self):
        """Test conversion of list of Pauli word operators to representation as a binary matrix."""

        observables = [Identity(1), PauliX(1), PauliZ(0) @ PauliZ(1)]

        binary_observables = np.array(
            [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]
        )

        assert (convert_observables_to_binary_matrix(observables) == binary_observables).all()

    def test_is_qwc(self):
        """Determining if two Pauli words are qubit-wise commuting."""

        n_qubits = 3
        wire_map = {Wires(0): 0, Wires("a"): 1, Wires("b"): 2}
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

    @pytest.mark.parametrize("non_pauli_word", non_pauli_words)
    def test_are_identical_pauli_words_non_pauli_word_catch(self, non_pauli_word):
        assert pytest.raises(
            TypeError, are_identical_pauli_words, (non_pauli_word, PauliZ(0) @ PauliZ(1))
        )
        assert pytest.raises(TypeError, are_identical_pauli_words, (non_pauli_word, non_pauli_word))
