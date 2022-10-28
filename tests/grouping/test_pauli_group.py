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
Unit tests for the :mod:`pauli_group`  functions in ``grouping/pauli.py``.
"""
import pytest

from pennylane import Identity, PauliX, PauliY, PauliZ, is_commuting
from pennylane.grouping import string_to_pauli_word
from pennylane.grouping.pauli import (
    partition_pauli_group,
    pauli_group,
    pauli_mult,
    pauli_mult_with_phase,
)


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
        assert all([expected.compare(obtained) for expected, obtained in zip(expected_pg_1, pg_1)])

    def test_one_qubit_pauli_group_valid_float_input(self):
        """Test that the single-qubit Pauli group is constructed correctly when a float
        that represents an integer is passed."""

        expected_pg_1 = [Identity(0), PauliZ(0), PauliX(0), PauliY(0)]
        pg_1 = list(pauli_group(1.0))
        assert all([expected.compare(obtained) for expected, obtained in zip(expected_pg_1, pg_1)])

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
        assert all(
            [
                expected.compare(obtained)
                for expected, obtained in zip(expected_pg_1_wires, pg_1_wires)
            ]
        )

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
        assert all([expected.compare(obtained) for expected, obtained in zip(expected_pg_2, pg_2)])

    @pytest.mark.parametrize(
        "pauli_word_1,pauli_word_2,wire_map,expected_product",
        [
            (PauliX(0), Identity(0), {0: 0}, PauliX(0)),
            (PauliZ(0), PauliY(0), {0: 0}, PauliX(0)),
            (PauliZ(0), PauliZ(0), {0: 0}, Identity(0)),
            (Identity("a"), Identity("b"), None, Identity("a")),
            (PauliZ("b") @ PauliY("a"), PauliZ("b") @ PauliY("a"), None, Identity("b")),
            (
                PauliZ("b") @ PauliY("a"),
                PauliZ("b") @ PauliY("a"),
                {"b": 0, "a": 1},
                Identity("b"),
            ),
            (
                PauliZ(0) @ PauliY(1),
                PauliX(0) @ PauliZ(1),
                {0: 0, 1: 1},
                PauliY(0) @ PauliX(1),
            ),
            (PauliZ(0) @ PauliY(1), PauliX(1) @ PauliY(0), {0: 0, 1: 1}, PauliX(0) @ PauliZ(1)),
            (
                PauliZ(0) @ PauliY(3) @ PauliZ(1),
                PauliX(1) @ PauliX(2) @ PauliY(0),
                None,
                PauliX(0) @ PauliY(1) @ PauliX(2) @ PauliY(3),
            ),
            (
                PauliX(0) @ PauliX(2),
                PauliX(0) @ PauliZ(2),
                None,
                PauliY(2),
            ),
            (PauliZ("a"), PauliX("b"), {"a": 0, "b": 1}, PauliZ("a") @ PauliX("b")),
            (
                PauliZ("a"),
                PauliX("e"),
                {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4},
                PauliZ("a") @ PauliX("e"),
            ),
            (PauliZ("a"), PauliY("e"), None, PauliZ("a") @ PauliY("e")),
            (
                PauliZ(0) @ PauliZ(2) @ PauliZ(4),
                PauliZ(0) @ PauliY(1) @ PauliX(3),
                None,
                PauliZ(2) @ PauliY(1) @ PauliZ(4) @ PauliX(3),
            ),
            (
                PauliZ(0) @ PauliZ(4) @ PauliZ(2),
                PauliZ(0) @ PauliY(1) @ PauliX(3),
                {0: 0, 2: 1, 4: 2, 1: 3, 3: 4},
                PauliZ(2) @ PauliY(1) @ PauliZ(4) @ PauliX(3),
            ),
            (
                PauliZ(0) @ PauliY(3) @ PauliZ(1),
                PauliX(1) @ PauliX(2) @ PauliY(0),
                None,
                PauliX(0) @ PauliY(3) @ PauliY(1) @ PauliX(2),
            ),
        ],
    )
    def test_pauli_mult(self, pauli_word_1, pauli_word_2, wire_map, expected_product):
        """Test that Pauli words are multiplied together correctly."""
        obtained_product = pauli_mult(pauli_word_1, pauli_word_2, wire_map=wire_map)
        assert obtained_product.compare(expected_product)

    @pytest.mark.parametrize(
        "pauli_word_1,pauli_word_2,wire_map,expected_phase",
        [
            (PauliX(0), Identity(0), {0: 0}, 1),
            (PauliZ(0), PauliY(0), {0: 0}, -1j),
            (PauliZ(0), PauliZ(0), {0: 0}, 1),
            (Identity("a"), Identity("b"), None, 1),
            (PauliZ("b") @ PauliY("a"), PauliZ("b") @ PauliY("a"), None, 1),
            (PauliZ(0), PauliY("b"), None, 1),
            (
                PauliZ("a") @ PauliY("b"),
                PauliX("a") @ PauliZ("b"),
                {"a": 0, "b": 1},
                -1,
            ),
            (PauliX(0) @ PauliX(2), PauliX(0) @ PauliZ(2), None, -1j),
            (
                PauliX(0) @ PauliY(1) @ PauliZ(2),
                PauliY(0) @ PauliY(1),
                {0: 0, 1: 1, 2: 2},
                1j,
            ),
            (PauliZ(0) @ PauliY(1), PauliX(1) @ PauliY(0), {0: 0, 1: 1}, -1),
            (PauliZ(0) @ PauliY(1), PauliX(1) @ PauliY(0), {1: 0, 0: 1}, -1),
            (PauliZ(0) @ PauliY(3) @ PauliZ(1), PauliX(1) @ PauliX(2) @ PauliY(0), None, 1),
        ],
    )
    def test_pauli_mult_with_phase(self, pauli_word_1, pauli_word_2, wire_map, expected_phase):
        """Test that multiplication including phases works as expected."""
        _, obtained_phase = pauli_mult_with_phase(pauli_word_1, pauli_word_2, wire_map=wire_map)
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

    @pytest.mark.parametrize("n", range(1, 6))
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
