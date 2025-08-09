# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit testing of christiansen mapping for Bose operators."""
import pytest

import pennylane as qml
from pennylane import I, X, Y, Z
from pennylane.bose import BoseSentence, BoseWord, christiansen_mapping
from pennylane.ops import SProd
from pennylane.pauli import PauliSentence, PauliWord
from pennylane.pauli.conversion import pauli_sentence

BOSE_WORDS_AND_OPS = [
    (
        BoseWord({(0, 0): "+"}),
        # trivial case of a creation operator, 0^ -> (X_0 - iY_0) / 2
        ([0.5, -0.5j], [X(0), Y(0)]),
    ),
    (
        BoseWord({(0, 0): "-"}),
        # trivial case of an annihilation operator, 0 -> (X_0 + iY_0) / 2
        ([(0.5 + 0j), (0.0 + 0.5j)], [X(0), Y(0)]),
    ),
    (
        BoseWord({(0, 0): "+", (1, 0): "-"}),
        ([(0.5 + 0j), (-0.5 + 0j)], [I(0), Z(0)]),
    ),
    (
        BoseWord({(0, 0): "-", (1, 0): "+"}),
        ([(0.5 + 0j), (0.5 + 0j)], [I(0), Z(0)]),
    ),
    (
        BoseWord({(0, 0): "-", (1, 1): "+"}),
        (
            [(0.25 + 0j), -0.25j, 0.25j, (0.25 + 0j)],
            [
                X(0) @ X(1),
                X(0) @ Y(1),
                Y(0) @ X(1),
                Y(0) @ Y(1),
            ],
        ),
    ),
    (
        BoseWord({(0, 1): "-", (1, 0): "+"}),
        (
            [(0.25 + 0j), -0.25j, 0.25j, (0.25 + 0j)],
            [
                X(1) @ X(0),
                X(1) @ Y(0),
                Y(1) @ X(0),
                Y(1) @ Y(0),
            ],
        ),
    ),
    (
        BoseWord({(0, 3): "+", (1, 0): "-"}),
        (
            [(0.25 + 0j), -0.25j, 0.25j, (0.25 + 0j)],
            [
                X(0) @ X(3),
                X(0) @ Y(3),
                Y(0) @ X(3),
                Y(0) @ Y(3),
            ],
        ),
    ),
    (
        BoseWord({(0, 1): "+", (1, 4): "-"}),
        (
            [(0.25 + 0j), 0.25j, -0.25j, (0.25 + 0j)],
            [
                X(1) @ X(4),
                X(1) @ Y(4),
                Y(1) @ X(4),
                Y(1) @ Y(4),
            ],
        ),
    ),
    (
        BoseWord({(0, 3): "+", (1, 1): "+", (2, 3): "-", (3, 1): "-"}),
        (
            [(0.25 + 0j), (-0.25 + 0j), (0.25 + 0j), (-0.25 + 0j)],
            [I(0), Z(1), Z(3) @ Z(1), Z(3)],
        ),
    ),
]

WIRE_MAP_FOR_BOSE_WORDS = [
    (
        {0: 3, 1: 2},
        [
            qml.s_prod(-0.25j, qml.prod(Y(3), X(2))),
            qml.s_prod(-0.25 + 0j, qml.prod(Y(3), Y(2))),
            qml.s_prod(0.25 + 0j, qml.prod(X(3), X(2))),
            qml.s_prod(-0.25j, qml.prod(X(3), Y(2))),
        ],
    ),
    (
        {0: "b", 1: "a"},
        [
            qml.s_prod(-0.25j, qml.prod(Y("b"), X("a"))),
            qml.s_prod(-0.25 + 0j, qml.prod(Y("b"), Y("a"))),
            qml.s_prod(0.25 + 0j, qml.prod(X("b"), X("a"))),
            qml.s_prod(-0.25j, qml.prod(X("b"), Y("a"))),
        ],
    ),
]


class TestBoseWordMapping:
    """Tests for mapping BoseWords"""

    @pytest.mark.parametrize("bosonic_op, result", BOSE_WORDS_AND_OPS)
    def test_christiansen_mapping_bose_word_ps(self, bosonic_op, result):
        """Test that the christiansen_mapping function returns the correct qubit operator."""

        qubit_op = christiansen_mapping(bosonic_op, ps=True)
        qubit_op.simplify()

        expected_op = pauli_sentence(qml.Hamiltonian(result[0], result[1]))
        expected_op.simplify()

        assert qubit_op == expected_op

    @pytest.mark.parametrize("bosonic_op, result", BOSE_WORDS_AND_OPS)
    def test_christiansen_mapping_bose_word_operation(self, bosonic_op, result):
        r"""Test that the christiansen_mapping function returns the correct operator for
        return type ps=False."""
        wires = bosonic_op.wires or [0]

        qubit_op = christiansen_mapping(bosonic_op)

        expected_op = pauli_sentence(qml.Hamiltonian(result[0], result[1]))
        expected_op = expected_op.operation(wires)

        qml.assert_equal(qubit_op.simplify(), expected_op.simplify())

    def test_christiansen_mapping_for_identity(self):
        """Test that the christiansen_mapping function returns the correct qubit operator for Identity."""
        qml.assert_equal(christiansen_mapping(BoseWord({})), I(0))

    def test_christiansen_mapping_for_identity_ps(self):
        """Test that the christiansen_mapping function returns the correct PauliSentence for Identity when ps=True."""
        assert christiansen_mapping(BoseWord({}), ps=True) == PauliSentence(
            {PauliWord({0: "I"}): 1.0 + 0.0j}
        )

    @pytest.mark.parametrize("wire_map, ops", WIRE_MAP_FOR_BOSE_WORDS)
    def test_providing_wire_map_bose_word_to_operation(self, wire_map, ops):
        r"""Test that the christiansen_mapping function returns the correct operator
        for a given wiremap."""

        w = BoseWord({(0, 0): "+", (1, 1): "+"})

        op = christiansen_mapping(w, wire_map=wire_map)
        result = qml.sum(*ops)

        op.simplify()

        assert pauli_sentence(op) == pauli_sentence(result)

    @pytest.mark.parametrize("wire_map, ops", WIRE_MAP_FOR_BOSE_WORDS)
    def test_providing_wire_map_bose_word_to_ps(self, wire_map, ops):
        r"""Test that the christiansen_mapping function returns the correct PauliSentence
        for a given wiremap."""
        w = BoseWord({(0, 0): "+", (1, 1): "+"})

        op = christiansen_mapping(w, wire_map=wire_map, ps=True)
        result_op = qml.sum(*ops)
        ps = pauli_sentence(result_op)

        ps.simplify()
        op.simplify()

        assert ps == op


bw1 = BoseWord({(0, 0): "+", (1, 1): "-"})
bw2 = BoseWord({(0, 0): "+", (1, 0): "-"})
bw3 = BoseWord({(0, 0): "+", (1, 3): "-", (2, 0): "+", (3, 4): "-"})
bw4 = BoseWord({})
bw5 = BoseWord({(0, 3): "+", (1, 2): "-"})
bw6 = BoseWord({(0, 1): "+", (1, 4): "-"})


BOSE_AND_PAULI_SENTENCES = [
    (BoseSentence({bw4: 0, bw2: 0}), PauliSentence({})),
    (
        BoseSentence({bw2: 2}),
        PauliSentence({PauliWord({}): (1 + 0j), PauliWord({0: "Z"}): (-1 + 0j)}),
    ),
    (
        BoseSentence({bw1: 1, bw2: 1}),
        PauliSentence(
            {
                PauliWord({0: "Y", 1: "X"}): -0.25j,
                PauliWord({0: "Y", 1: "Y"}): (0.25 + 0j),
                PauliWord({0: "X", 1: "X"}): (0.25 + 0j),
                PauliWord({0: "X", 1: "Y"}): 0.25j,
                PauliWord({}): (0.5 + 0j),
                PauliWord({0: "Z"}): (-0.5 + 0j),
            }
        ),
    ),
    (
        BoseSentence({bw1: 1j, bw2: -2}),
        PauliSentence(
            {
                PauliWord({0: "Y", 1: "X"}): (0.25 + 0j),
                PauliWord({0: "Y", 1: "Y"}): 0.25j,
                PauliWord({0: "X", 1: "X"}): 0.25j,
                PauliWord({0: "X", 1: "Y"}): (-0.25 + 0j),
                PauliWord({}): (-1 + 0j),
                PauliWord({0: "Z"}): (1 + 0j),
            }
        ),
    ),
    (
        BoseSentence({bw1: -2, bw5: 1j}),
        PauliSentence(
            {
                PauliWord({0: "X", 1: "X"}): -0.5,
                PauliWord({0: "X", 1: "Y"}): -0.5j,
                PauliWord({0: "Y", 1: "X"}): 0.5j,
                PauliWord({0: "Y", 1: "Y"}): -0.5,
                PauliWord({2: "X", 3: "X"}): 0.25j,
                PauliWord({2: "X", 3: "Y"}): 0.25,
                PauliWord({2: "Y", 3: "X"}): -0.25,
                PauliWord({2: "Y", 3: "Y"}): 0.25j,
            }
        ),
    ),
    (
        BoseSentence({bw6: 1, bw2: 2}),
        PauliSentence(
            {
                PauliWord({0: "I"}): 1.0,
                PauliWord({0: "Z"}): -1.0,
                PauliWord({1: "X", 4: "X"}): 0.25,
                PauliWord({1: "X", 4: "Y"}): 0.25j,
                PauliWord({1: "Y", 4: "X"}): -0.25j,
                PauliWord({1: "Y", 4: "Y"}): 0.25,
            }
        ),
    ),
]

WIRE_MAP_FOR_BOSE_SENTENCE = [
    (
        {0: 3, 1: 2},
        [
            qml.s_prod(-0.25j, qml.prod(Y(3), X(2))),
            qml.s_prod((0.25 + 0j), qml.prod(Y(3), Y(2))),
            qml.s_prod((0.25 + 0j), qml.prod(X(3), X(2))),
            qml.s_prod(0.25j, qml.prod(X(3), Y(2))),
            qml.s_prod((0.5 + 0j), I(3)),
            qml.s_prod((-0.5 + 0j), Z(3)),
        ],
    ),
    (
        {0: "b", 1: "a"},
        [
            qml.s_prod(-0.25j, qml.prod(Y("b"), X("a"))),
            qml.s_prod((0.25 + 0j), qml.prod(Y("b"), Y("a"))),
            qml.s_prod((0.25 + 0j), qml.prod(X("b"), X("a"))),
            qml.s_prod(0.25j, qml.prod(X("b"), Y("a"))),
            qml.s_prod((0.5 + 0j), I("b")),
            qml.s_prod((-0.5 + 0j), Z("b")),
        ],
    ),
]


class TestBoseSentencesMapping:
    """Tests for mapping BoseSentences"""

    def test_empty_bose_sentence(self):
        """Test that an empty BoseSentence (bose null operator) is
        converted to an empty PauliSentence or the null operator"""
        op = BoseSentence({})

        ps_op = christiansen_mapping(op, ps=True)
        ps_op.simplify()
        assert ps_op == PauliSentence({})

        op = christiansen_mapping(op).simplify()
        assert isinstance(op, SProd)
        assert isinstance(op.base, I)
        assert op.scalar == 0

    def test_bose_sentence_identity(self):
        """Test that a BoseSentence composed of a single Identity operator
        converts to PauliSentence and operation as expected"""
        op = BoseSentence({bw4: 1})
        ps = PauliSentence({PauliWord({}): 1})

        ps_op = christiansen_mapping(op, ps=True)
        qubit_op = christiansen_mapping(op)

        assert ps_op == ps

        result = ps.operation(wire_order=[0])
        qml.assert_equal(qubit_op.simplify(), result.simplify())

    @pytest.mark.parametrize("bosonic_op, result", BOSE_AND_PAULI_SENTENCES)
    def test_christiansen_mapping_for_bose_sentence_ps(self, bosonic_op, result):
        r"""Test that christiansen_mapping function returns the correct PauliSentence."""
        qubit_op = christiansen_mapping(bosonic_op, ps=True)
        qubit_op.simplify()

        assert qubit_op == result

    @pytest.mark.parametrize("bosonic_op, result", BOSE_AND_PAULI_SENTENCES)
    def test_christiansen_mapping_for_bose_sentence_operation(self, bosonic_op, result):
        r"""Test that christiansen_mapping function returns the correct qubit operator."""
        wires = bosonic_op.wires or [0]

        qubit_op = christiansen_mapping(bosonic_op)
        result = result.operation(wires)

        qml.assert_equal(qubit_op.simplify(), result.simplify())

    @pytest.mark.parametrize("wire_map, ops", WIRE_MAP_FOR_BOSE_SENTENCE)
    def test_providing_wire_map_bose_sentence_to_operation(self, wire_map, ops):
        r"""Test that the christiansen_mapping function returns the correct operator
        for a given wiremap."""
        bs = BoseSentence(
            {BoseWord({(0, 0): "+", (1, 1): "-"}): 1, BoseWord({(0, 0): "+", (1, 0): "-"}): 1}
        )

        op = christiansen_mapping(bs, wire_map=wire_map)
        result = qml.sum(*ops)

        assert op.wires == result.wires

        assert pauli_sentence(op) == pauli_sentence(result)

    @pytest.mark.parametrize("wire_map, ops", WIRE_MAP_FOR_BOSE_SENTENCE)
    def test_providing_wire_map_bose_sentence_to_ps(self, wire_map, ops):
        r"""Test that the christiansen_mapping function returns the correct PauliSentence
        for a given wiremap."""
        bs = BoseSentence(
            {BoseWord({(0, 0): "+", (1, 1): "-"}): 1, BoseWord({(0, 0): "+", (1, 0): "-"}): 1}
        )

        op = christiansen_mapping(bs, wire_map=wire_map, ps=True)
        result_op = qml.sum(*ops)
        ps = pauli_sentence(result_op)

        ps.simplify()
        op.simplify()

        assert ps == op


bs1 = BoseSentence({bw1: 1})


@pytest.mark.parametrize(
    "bose_op, qubit_op_data, tol",
    (
        (bw1, (0.25, 0.25j, -0.25j, (0.25 + 0j)), None),
        (bw1, (0.25, 0.25j, -0.25j, 0.25), 0.0),
        (bw1, (0.25, 0.25j, -0.25j, 0.25), 1.0e-12),
        (bs1, (0.25, 0.25j, -0.25j, (0.25 + 0j)), None),
        (bs1, (0.25, 0.25j, -0.25j, 0.25), 0.0),
        (bs1, (0.25, 0.25j, -0.25j, 0.25), 1.0e-12),
    ),
)
def test_christiansen_mapping_tolerance(bose_op, qubit_op_data, tol):
    """Test that christiansen_mapping properly removes negligible imaginary components"""
    op = christiansen_mapping(bose_op, tol=tol)
    assert isinstance(op.data[1], type(qubit_op_data[1]))


def test_error_is_raised_for_incompatible_type():
    """Test that an error is raised if the input is not a BoseWord or BoseSentence"""

    with pytest.raises(TypeError, match="bose_operator must be a BoseWord or BoseSentence"):
        christiansen_mapping(X(0))
