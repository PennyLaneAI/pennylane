# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This module contains tests for unary_mapping of bosonic operators."""
import pytest

import pennylane as qml
from pennylane import I, X, Y, Z
from pennylane.bose import BoseSentence, BoseWord, unary_mapping
from pennylane.pauli import PauliSentence, PauliWord
from pennylane.pauli.conversion import pauli_sentence

# Expected results were generated manually
BOSE_WORDS_AND_OPS = [
    (
        BoseWord({(0, 0): "+"}),
        # trivial case of a creation operator with 2 allowed bosonic states, 0^ -> (X_0 - iY_0) / 2
        2,
        ([0.25, -0.25j, 0.25j, (0.25 + 0j)], [X(0) @ X(1), X(0) @ Y(1), Y(0) @ X(1), Y(0) @ Y(1)]),
    ),
    (
        BoseWord({(0, 0): "-"}),
        # trivial case of an annihilation operator with 2 allowed bosonic states, 0 -> (X_0 + iY_0) / 2
        2,
        ([0.25, 0.25j, -0.25j, (0.25 + 0j)], [X(0) @ X(1), X(0) @ Y(1), Y(0) @ X(1), Y(0) @ Y(1)]),
    ),
    (
        BoseWord({(0, 0): "+"}),
        # creation operator with 4 allowed bosonic states
        4,
        (
            [
                0.25,
                -0.25j,
                0.25j,
                (0.25 + 0j),
                0.3535533905932738,
                -0.3535533905932738j,
                0.3535533905932738j,
                (0.3535533905932738 + 0j),
                0.4330127018922193,
                -0.4330127018922193j,
                0.4330127018922193j,
                (0.4330127018922193 + 0j),
            ],
            [
                X(0) @ X(1),
                X(0) @ Y(1),
                Y(0) @ X(1),
                Y(0) @ Y(1),
                X(1) @ X(2),
                X(1) @ Y(2),
                Y(1) @ X(2),
                Y(1) @ Y(2),
                X(2) @ X(3),
                X(2) @ Y(3),
                Y(2) @ X(3),
                Y(2) @ Y(3),
            ],
        ),
    ),
    (
        BoseWord({(0, 0): "-"}),
        # annihilation operator with 4 allowed bosonic states
        4,
        (
            [
                0.25,
                0.25j,
                -0.25j,
                (0.25 + 0j),
                0.3535533905932738,
                0.3535533905932738j,
                -0.3535533905932738j,
                (0.3535533905932738 + 0j),
                0.4330127018922193,
                0.4330127018922193j,
                -0.4330127018922193j,
                (0.4330127018922193 + 0j),
            ],
            [
                X(0) @ X(1),
                X(0) @ Y(1),
                Y(0) @ X(1),
                Y(0) @ Y(1),
                X(1) @ X(2),
                X(1) @ Y(2),
                Y(1) @ X(2),
                Y(1) @ Y(2),
                X(2) @ X(3),
                X(2) @ Y(3),
                Y(2) @ X(3),
                Y(2) @ Y(3),
            ],
        ),
    ),
    (
        BoseWord({(0, 0): "+", (1, 0): "-"}),
        4,
        ([3.0, -0.5, -1.0000000000000002, -1.4999999999999998], [I(), Z(1), Z(2), Z(3)]),
    ),
    (
        BoseWord({(0, 0): "-", (1, 0): "+", (2, 0): "-"}),
        4,
        (
            [
                0.25,
                0.25j,
                -0.25j,
                (0.25 + 0j),
                0.7071067811865477,
                0.7071067811865477j,
                -0.7071067811865477j,
                (0.7071067811865477 + 0j),
                1.2990381056766578,
                1.2990381056766578j,
                -1.2990381056766578j,
                (1.2990381056766578 + 0j),
            ],
            [
                X(0) @ X(1),
                X(0) @ Y(1),
                Y(0) @ X(1),
                Y(0) @ Y(1),
                X(1) @ X(2),
                X(1) @ Y(2),
                Y(1) @ X(2),
                Y(1) @ Y(2),
                X(2) @ X(3),
                X(2) @ Y(3),
                Y(2) @ X(3),
                Y(2) @ Y(3),
            ],
        ),
    ),
    (
        BoseWord({(0, 0): "-", (1, 1): "-", (2, 0): "+"}),
        2,
        (
            [0.125, 0.125j, -0.125j, (0.125 + 0j), -0.125, -0.125j, 0.125j, (-0.125 + 0j)],
            [
                X(2) @ X(3),
                X(2) @ Y(3),
                Y(2) @ X(3),
                Y(2) @ Y(3),
                Z(0) @ X(2) @ X(3),
                Z(0) @ X(2) @ Y(3),
                Z(0) @ Y(2) @ X(3),
                Z(0) @ Y(2) @ Y(3),
            ],
        ),
    ),
    (
        BoseWord({(0, 1): "+", (1, 1): "-", (2, 0): "+", (3, 0): "-"}),
        4,
        (
            [
                9.0,
                -1.5,
                -3.000000000000001,
                -4.499999999999999,
                -1.5,
                0.25,
                0.5000000000000001,
                0.7499999999999999,
                -3.000000000000001,
                0.5000000000000001,
                1.0000000000000004,
                1.5,
                -4.499999999999999,
                0.7499999999999999,
                1.5,
                2.2499999999999996,
            ],
            [
                I(),
                Z(5),
                Z(6),
                Z(7),
                Z(1),
                Z(1) @ Z(5),
                Z(1) @ Z(6),
                Z(1) @ Z(7),
                Z(2),
                Z(2) @ Z(5),
                Z(2) @ Z(6),
                Z(2) @ Z(7),
                Z(3),
                Z(3) @ Z(5),
                Z(3) @ Z(6),
                Z(3) @ Z(7),
            ],
        ),
    ),
]


class TestBoseWordMapping:
    """Tests for mapping BoseWords."""

    @pytest.mark.parametrize("bose_op, n_states, result", BOSE_WORDS_AND_OPS)
    def test_unary_mapping_boseword(self, bose_op, n_states, result):
        """Test that the unary_mapping function returns the correct qubit operator."""
        qubit_op = unary_mapping(bose_op, n_states=n_states, ps=True)
        qubit_op.simplify(tol=1e-8)

        expected_op = pauli_sentence(qml.Hamiltonian(result[0], result[1]))
        expected_op.simplify(tol=1e-8)
        assert qubit_op == expected_op

    @pytest.mark.parametrize("bosonic_op, n_states, result", BOSE_WORDS_AND_OPS)
    def test_unary_mapping_bose_word_operation(self, bosonic_op, n_states, result):
        r"""Test that the unary_mapping function returns the correct operator for
        return type ps=False."""
        wires = bosonic_op.wires or [0]

        qubit_op = unary_mapping(bosonic_op, n_states=n_states, ps=False)

        expected_op = pauli_sentence(qml.Hamiltonian(result[0], result[1]))
        expected_op = expected_op.operation(wires)

        qml.assert_equal(qubit_op.simplify(), expected_op.simplify())

    def test_unary_mapping_for_identity(self):
        """Test that the unary_mapping function returns the correct qubit operator for Identity."""
        qml.assert_equal(unary_mapping(BoseWord({})), I(0))

    def test_unary_mapping_for_identity_ps(self):
        """Test that the unary_mapping function returns the correct PauliSentence for Identity when ps=True."""
        assert unary_mapping(BoseWord({}), ps=True) == PauliSentence(
            {PauliWord({0: "I"}): 1.0 + 0.0j}
        )


bw1 = BoseWord({(0, 0): "+"})
bw2 = BoseWord({(0, 0): "+", (1, 1): "-", (2, 0): "-"})
bw3 = BoseWord({(0, 0): "+", (1, 0): "-"})

BOSE_SEN_AND_OPS = [
    (
        BoseSentence({bw1: 1.0, bw3: -1.0}),
        4,
        (
            [
                0.25,
                -0.25j,
                0.25j,
                (0.25 + 0j),
                0.3535533905932738,
                -0.3535533905932738j,
                0.3535533905932738j,
                (0.3535533905932738 + 0j),
                0.4330127018922193,
                -0.4330127018922193j,
                0.4330127018922193j,
                (0.4330127018922193 + 0j),
                -3.0,
                0.5,
                1.0000000000000002,
                1.4999999999999998,
            ],
            [
                X(0) @ X(1),
                X(0) @ Y(1),
                Y(0) @ X(1),
                Y(0) @ Y(1),
                X(1) @ X(2),
                X(1) @ Y(2),
                Y(1) @ X(2),
                Y(1) @ Y(2),
                X(2) @ X(3),
                X(2) @ Y(3),
                Y(2) @ X(3),
                Y(2) @ Y(3),
                I(),
                Z(1),
                Z(2),
                Z(3),
            ],
        ),
    ),
    (
        BoseSentence({bw2: 1.0, bw3: -0.5}),
        4,
        (
            [
                0.75,
                0.75j,
                -0.75j,
                (0.75 + 0j),
                1.0606601717798214,
                1.0606601717798214j,
                -1.0606601717798214j,
                (1.0606601717798214 + 0j),
                1.299038105676658,
                1.299038105676658j,
                -1.299038105676658j,
                (1.299038105676658 + 0j),
                -0.125,
                -0.125j,
                0.125j,
                (-0.125 + 0j),
                -0.1767766952966369,
                -0.1767766952966369j,
                0.1767766952966369j,
                (-0.1767766952966369 + 0j),
                -0.21650635094610965,
                -0.21650635094610965j,
                0.21650635094610965j,
                (-0.21650635094610965 + 0j),
                -0.25000000000000006,
                -0.25000000000000006j,
                0.25000000000000006j,
                (-0.25000000000000006 + 0j),
                -0.35355339059327384,
                -0.35355339059327384j,
                0.35355339059327384j,
                (-0.35355339059327384 + 0j),
                -0.4330127018922194,
                -0.4330127018922194j,
                0.4330127018922194j,
                (-0.4330127018922194 + 0j),
                -0.37499999999999994,
                -0.37499999999999994j,
                0.37499999999999994j,
                (-0.37499999999999994 + 0j),
                -0.5303300858899106,
                -0.5303300858899106j,
                0.5303300858899106j,
                (-0.5303300858899106 + 0j),
                -0.6495190528383289,
                -0.6495190528383289j,
                0.6495190528383289j,
                (-0.6495190528383289 + 0j),
                -1.5,
                0.25,
                0.5000000000000001,
                0.7499999999999999,
            ],
            [
                X(4) @ X(5),
                X(4) @ Y(5),
                Y(4) @ X(5),
                Y(4) @ Y(5),
                X(5) @ X(6),
                X(5) @ Y(6),
                Y(5) @ X(6),
                Y(5) @ Y(6),
                X(6) @ X(7),
                X(6) @ Y(7),
                Y(6) @ X(7),
                Y(6) @ Y(7),
                Z(1) @ X(4) @ X(5),
                Z(1) @ X(4) @ Y(5),
                Z(1) @ Y(4) @ X(5),
                Z(1) @ Y(4) @ Y(5),
                Z(1) @ X(5) @ X(6),
                Z(1) @ X(5) @ Y(6),
                Z(1) @ Y(5) @ X(6),
                Z(1) @ Y(5) @ Y(6),
                Z(1) @ X(6) @ X(7),
                Z(1) @ X(6) @ Y(7),
                Z(1) @ Y(6) @ X(7),
                Z(1) @ Y(6) @ Y(7),
                Z(2) @ X(4) @ X(5),
                Z(2) @ X(4) @ Y(5),
                Z(2) @ Y(4) @ X(5),
                Z(2) @ Y(4) @ Y(5),
                Z(2) @ X(5) @ X(6),
                Z(2) @ X(5) @ Y(6),
                Z(2) @ Y(5) @ X(6),
                Z(2) @ Y(5) @ Y(6),
                Z(2) @ X(6) @ X(7),
                Z(2) @ X(6) @ Y(7),
                Z(2) @ Y(6) @ X(7),
                Z(2) @ Y(6) @ Y(7),
                Z(3) @ X(4) @ X(5),
                Z(3) @ X(4) @ Y(5),
                Z(3) @ Y(4) @ X(5),
                Z(3) @ Y(4) @ Y(5),
                Z(3) @ X(5) @ X(6),
                Z(3) @ X(5) @ Y(6),
                Z(3) @ Y(5) @ X(6),
                Z(3) @ Y(5) @ Y(6),
                Z(3) @ X(6) @ X(7),
                Z(3) @ X(6) @ Y(7),
                Z(3) @ Y(6) @ X(7),
                Z(3) @ Y(6) @ Y(7),
                I(),
                Z(1),
                Z(2),
                Z(3),
            ],
        ),
    ),
]


class TestBoseSentenceMapping:
    """Tests for mapping BoseSentences"""

    def test_empty_bose_sentence(self):
        """Test that an empty BoseSentence (bose null operator) is
        converted to an empty PauliSentence or the null operator"""
        op = BoseSentence({})

        ps_op = unary_mapping(op, ps=True)
        ps_op.simplify()
        assert ps_op == PauliSentence({})

        op = unary_mapping(op).simplify()
        assert isinstance(op, qml.ops.SProd)
        assert isinstance(op.base, I)
        assert op.scalar == 0

    @pytest.mark.parametrize("bose_op, n_states, result", BOSE_SEN_AND_OPS)
    def test_unary_mapping_bosesentence(self, bose_op, n_states, result):
        """Test that the unary_mapping function returns the correct qubit operator."""

        qubit_op = unary_mapping(bose_op, n_states=n_states, ps=True)
        qubit_op.simplify(tol=1e-8)

        expected_op = pauli_sentence(qml.Hamiltonian(result[0], result[1]))
        expected_op.simplify(tol=1e-8)
        assert qubit_op == expected_op


@pytest.mark.parametrize(
    "bose_op",
    [
        BoseWord({(0, 0): "-"}),
        BoseSentence({BoseWord({(0, 0): "-"}): 1.0, BoseWord({(0, 1): "-"}): 1.0}),
    ],
)
def test_return_unary_mapping_sum(bose_op):
    """Test that the correct type is returned for unary mapping
    when ps is set to False."""

    qubit_op = unary_mapping(bose_op, ps=False)
    assert isinstance(qubit_op, qml.ops.Sum)


@pytest.mark.parametrize(
    "bose_op",
    [
        BoseWord({(0, 0): "-"}),
        BoseSentence({BoseWord({(0, 0): "-"}): 1.0, BoseWord({(0, 1): "-"}): 1.0}),
    ],
)
def test_return_unary_mapping_ps(bose_op):
    """Test that the correct type is returned for unary mapping
    when ps is set to False."""

    qubit_op = unary_mapping(bose_op, ps=True)
    assert isinstance(qubit_op, qml.pauli.PauliSentence)


@pytest.mark.parametrize(
    ("bose_op, wire_map, result"),
    [
        (
            BoseWord({(0, 0): "+"}),
            {0: 1, 1: 2},
            (
                [0.25, -0.25j, 0.25j, (0.25 + 0j)],
                [X(1) @ X(2), X(1) @ Y(2), Y(1) @ X(2), Y(1) @ Y(2)],
            ),
        ),
        (
            BoseSentence({BoseWord({(0, 0): "-"}): 1.0, BoseWord({(0, 1): "+"}): 1.0}),
            {0: "a", 1: "b", 2: "c", 3: "d"},
            (
                [0.25, 0.25j, -0.25j, (0.25 + 0j), 0.25, -0.25j, 0.25j, (0.25 + 0j)],
                [
                    X("a") @ X("b"),
                    X("a") @ Y("b"),
                    Y("a") @ X("b"),
                    Y("a") @ Y("b"),
                    X("c") @ X("d"),
                    X("c") @ Y("d"),
                    Y("c") @ X("d"),
                    Y("c") @ Y("d"),
                ],
            ),
        ),
    ],
)
def test_unary_mapping_wiremap(bose_op, wire_map, result):
    """Test that the unary_mapping function returns the correct qubit operator."""
    qubit_op = unary_mapping(bose_op, n_states=2, wire_map=wire_map, ps=True)
    qubit_op.simplify(tol=1e-8)

    expected_op = pauli_sentence(qml.Hamiltonian(result[0], result[1]))
    expected_op.simplify(tol=1e-8)
    assert qubit_op == expected_op


def test_n_states_error_unary():
    """Test that an error is raised if invalid number of states is provided."""
    bw = BoseWord({(0, 0): "-"})
    with pytest.raises(
        ValueError, match="Number of allowed bosonic states cannot be less than 2, provided 0."
    ):
        unary_mapping(bw, n_states=0)


def test_error_is_raised_for_incompatible_type():
    """Test that an error is raised if the input is not a BoseWord or BoseSentence"""

    with pytest.raises(TypeError, match="bose_operator must be a BoseWord or BoseSentence"):
        unary_mapping(X(0))


bs1 = BoseSentence({bw1: 1})


@pytest.mark.parametrize(
    "bose_op, qubit_op_data, tol",
    (
        (
            bw1,
            (
                0.25,
                -0.25j,
                0.25j,
                (0.25 + 0j),
                0.3535533905932738,
                -0.3535533905932738j,
                0.3535533905932738j,
                (0.3535533905932738 + 0j),
                0.4330127018922193,
                -0.4330127018922193j,
                0.4330127018922193j,
                (0.4330127018922193 + 0j),
            ),
            None,
        ),
        (
            bw1,
            (
                0.25,
                -0.25j,
                0.25j,
                0.25,
                0.3535533905932738,
                -0.3535533905932738j,
                0.3535533905932738j,
                0.3535533905932738,
                0.4330127018922193,
                -0.4330127018922193j,
                0.4330127018922193j,
                0.4330127018922193,
            ),
            0.0,
        ),
        (
            bw1,
            (
                0.25,
                0.25,
                0.3535533905932738,
                -0.3535533905932738j,
                0.3535533905932738j,
                0.3535533905932738,
                0.4330127018922193,
                -0.4330127018922193j,
                0.4330127018922193j,
                0.4330127018922193,
            ),
            0.3,
        ),
        (
            bs1,
            (
                0.25,
                -0.25j,
                0.25j,
                (0.25 + 0j),
                0.3535533905932738,
                -0.3535533905932738j,
                0.3535533905932738j,
                (0.3535533905932738 + 0j),
                0.4330127018922193,
                -0.4330127018922193j,
                0.4330127018922193j,
                (0.4330127018922193 + 0j),
            ),
            None,
        ),
        (
            bs1,
            (
                0.25,
                -0.25j,
                0.25j,
                0.25,
                0.3535533905932738,
                -0.3535533905932738j,
                0.3535533905932738j,
                0.3535533905932738,
                0.4330127018922193,
                -0.4330127018922193j,
                0.4330127018922193j,
                0.4330127018922193,
            ),
            0.0,
        ),
        (
            bs1,
            (
                0.25,
                0.25,
                0.3535533905932738,
                0.3535533905932738,
                0.4330127018922193,
                -0.4330127018922193j,
                0.4330127018922193j,
                0.4330127018922193,
            ),
            0.4,
        ),
    ),
)
def test_unary_mapping_tolerance(bose_op, qubit_op_data, tol):
    """Test that unary_mapping properly removes negligible imaginary components"""
    op = unary_mapping(bose_op, n_states=4, tol=tol)
    assert isinstance(op.data[1], type(qubit_op_data[1]))
