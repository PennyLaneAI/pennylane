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
"""This module contains tests for binary_mapping of bosonic operators."""
import pytest

import pennylane as qml
from pennylane.qchem import BoseWord, BoseSentence, binary_mapping
from pennylane import I, X, Y, Z
from pennylane.pauli.conversion import pauli_sentence

# Expected results were generated manually
BOSE_WORDS_AND_OPS = [
    (
        BoseWord({(0, 0): "+"}),
        # trivial case of a creation operator with 2 states in a boson, 0^ -> (X_0 - iY_0) / 2
        2,
        ([0.5, -0.5j], [X(0), Y(0)]),
    ),
    (
        BoseWord({(0, 0): "-"}),
        # trivial case of an annihilation operator with 2 states in a boson, 0 -> (X_0 + iY_0) / 2
        2,
        ([(0.5 + 0j), (0.0 + 0.5j)], [X(0), Y(0)]),
    ),
    (
        BoseWord({(0, 0): "+"}),
        # creation operator with 4 states in a boson
        4,
        (
            [
                0.6830127018922193,
                0.3535533905932738,
                -0.3535533905932738j,
                -0.1830127018922193,
                -0.6830127018922193j,
                0.3535533905932738j,
                (0.3535533905932738 + 0j),
                0.1830127018922193j,
            ],
            [
                X(0),
                X(0) @ X(1),
                X(0) @ Y(1),
                X(0) @ Z(1),
                Y(0),
                Y(0) @ X(1),
                Y(0) @ Y(1),
                Y(0) @ Z(1),
            ],
        ),
    ),
    (
        BoseWord({(0, 0): "-"}),
        # annihilation operator with 4 states in a boson
        4,
        (
            [
                0.6830127018922193,
                0.3535533905932738,
                0.3535533905932738j,
                -0.1830127018922193,
                0.6830127018922193j,
                -0.3535533905932738j,
                (0.3535533905932738 + 0j),
                -0.1830127018922193j,
            ],
            [
                X(0),
                X(0) @ X(1),
                X(0) @ Y(1),
                X(0) @ Z(1),
                Y(0),
                Y(0) @ X(1),
                Y(0) @ Y(1),
                Y(0) @ Z(1),
            ],
        ),
    ),
    (
        BoseWord({(0, 0): "+", (1, 0): "-"}),
        4,
        ([(1.5 + 0j), (-0.4999999999999999 + 0j), (-0.9999999999999999 + 0j)], [I(0), Z(0), Z(1)]),
    ),
    (
        BoseWord({(0, 0): "-", (1, 0): "+"}),
        4,
        (
            [(1.5 + 0j), (0.4999999999999999 + 0j), (-0.9999999999999999 + 0j)],
            [I(0), Z(0), Z(0) @ Z(1)],
        ),
    ),
    (
        BoseWord({(0, 0): "-", (1, 1): "+"}),
        2,
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
        4,
        (
            [
                0.46650635094610965,
                -0.12499999999999997,
                0.46650635094610965j,
                -0.12499999999999997j,
                0.24148145657226708,
                0.24148145657226708j,
                -0.24148145657226708j,
                (0.24148145657226708 + 0j),
                -0.12499999999999997,
                0.03349364905389033,
                -0.12499999999999997j,
                0.03349364905389033j,
                -0.06470476127563018,
                -0.06470476127563018j,
                0.06470476127563018j,
                (-0.06470476127563018 + 0j),
                -0.46650635094610965j,
                0.12499999999999997j,
                (0.46650635094610965 + 0j),
                (-0.12499999999999997 + 0j),
                -0.24148145657226708j,
                (0.24148145657226708 + 0j),
                (-0.24148145657226708 + 0j),
                -0.24148145657226708j,
                0.12499999999999997j,
                -0.03349364905389033j,
                (-0.12499999999999997 + 0j),
                (0.03349364905389033 + 0j),
                0.06470476127563018j,
                (-0.06470476127563018 + 0j),
                (0.06470476127563018 + 0j),
                0.06470476127563018j,
                0.24148145657226708,
                -0.06470476127563018,
                0.24148145657226708j,
                -0.06470476127563018j,
                0.12500000000000003,
                0.12500000000000003j,
                -0.12500000000000003j,
                (0.12500000000000003 + 0j),
                -0.24148145657226708j,
                0.06470476127563018j,
                (0.24148145657226708 + 0j),
                (-0.06470476127563018 + 0j),
                -0.12500000000000003j,
                (0.12500000000000003 + 0j),
                (-0.12500000000000003 + 0j),
                -0.12500000000000003j,
                0.24148145657226708j,
                -0.06470476127563018j,
                (-0.24148145657226708 + 0j),
                (0.06470476127563018 + 0j),
                0.12500000000000003j,
                (-0.12500000000000003 + 0j),
                (0.12500000000000003 + 0j),
                0.12500000000000003j,
                (0.24148145657226708 + 0j),
                (-0.06470476127563018 + 0j),
                0.24148145657226708j,
                -0.06470476127563018j,
                (0.12500000000000003 + 0j),
                0.12500000000000003j,
                -0.12500000000000003j,
                (0.12500000000000003 + 0j),
            ],
            [
                X(0) @ X(2),
                X(0) @ X(2) @ Z(3),
                X(0) @ Y(2),
                X(0) @ Y(2) @ Z(3),
                X(0) @ X(2) @ X(3),
                X(0) @ X(2) @ Y(3),
                X(0) @ Y(2) @ X(3),
                X(0) @ Y(2) @ Y(3),
                X(0) @ Z(1) @ X(2),
                X(0) @ Z(1) @ X(2) @ Z(3),
                X(0) @ Z(1) @ Y(2),
                X(0) @ Z(1) @ Y(2) @ Z(3),
                X(0) @ Z(1) @ X(2) @ X(3),
                X(0) @ Z(1) @ X(2) @ Y(3),
                X(0) @ Z(1) @ Y(2) @ X(3),
                X(0) @ Z(1) @ Y(2) @ Y(3),
                Y(0) @ X(2),
                Y(0) @ X(2) @ Z(3),
                Y(0) @ Y(2),
                Y(0) @ Y(2) @ Z(3),
                Y(0) @ X(2) @ X(3),
                Y(0) @ X(2) @ Y(3),
                Y(0) @ Y(2) @ X(3),
                Y(0) @ Y(2) @ Y(3),
                Y(0) @ Z(1) @ X(2),
                Y(0) @ Z(1) @ X(2) @ Z(3),
                Y(0) @ Z(1) @ Y(2),
                Y(0) @ Z(1) @ Y(2) @ Z(3),
                Y(0) @ Z(1) @ X(2) @ X(3),
                Y(0) @ Z(1) @ X(2) @ Y(3),
                Y(0) @ Z(1) @ Y(2) @ X(3),
                Y(0) @ Z(1) @ Y(2) @ Y(3),
                X(0) @ X(1) @ X(2),
                X(0) @ X(1) @ X(2) @ Z(3),
                X(0) @ X(1) @ Y(2),
                X(0) @ X(1) @ Y(2) @ Z(3),
                X(0) @ X(1) @ X(2) @ X(3),
                X(0) @ X(1) @ X(2) @ Y(3),
                X(0) @ X(1) @ Y(2) @ X(3),
                X(0) @ X(1) @ Y(2) @ Y(3),
                X(0) @ Y(1) @ X(2),
                X(0) @ Y(1) @ X(2) @ Z(3),
                X(0) @ Y(1) @ Y(2),
                X(0) @ Y(1) @ Y(2) @ Z(3),
                X(0) @ Y(1) @ X(2) @ X(3),
                X(0) @ Y(1) @ X(2) @ Y(3),
                X(0) @ Y(1) @ Y(2) @ X(3),
                X(0) @ Y(1) @ Y(2) @ Y(3),
                Y(0) @ X(1) @ X(2),
                Y(0) @ X(1) @ X(2) @ Z(3),
                Y(0) @ X(1) @ Y(2),
                Y(0) @ X(1) @ Y(2) @ Z(3),
                Y(0) @ X(1) @ X(2) @ X(3),
                Y(0) @ X(1) @ X(2) @ Y(3),
                Y(0) @ X(1) @ Y(2) @ X(3),
                Y(0) @ X(1) @ Y(2) @ Y(3),
                Y(0) @ Y(1) @ X(2),
                Y(0) @ Y(1) @ X(2) @ Z(3),
                Y(0) @ Y(1) @ Y(2),
                Y(0) @ Y(1) @ Y(2) @ Z(3),
                Y(0) @ Y(1) @ X(2) @ X(3),
                Y(0) @ Y(1) @ X(2) @ Y(3),
                Y(0) @ Y(1) @ Y(2) @ X(3),
                Y(0) @ Y(1) @ Y(2) @ Y(3),
            ],
        ),
    ),
]


@pytest.mark.parametrize("bose_op, nstates, result", BOSE_WORDS_AND_OPS)
def test_binary_mapping_boseword(bose_op, nstates, result):
    """Test that the binary_mapping function returns the correct qubit operator."""
    qubit_op = binary_mapping(bose_op, nstates_boson=nstates)
    qubit_op.simplify(tol=1e-8)

    expected_op = pauli_sentence(qml.Hamiltonian(result[0], result[1]))
    expected_op.simplify(tol=1e-8)
    assert qubit_op == expected_op


bw1 = BoseWord({(0, 0): "+"})
bw2 = BoseWord({(0, 0): "+", (1, 1): "-", (2, 0): "-"})
bw3 = BoseWord({(0, 0): "+", (1, 0): "-"})

BOSE_SEN_AND_OPS = [
    (
        BoseSentence({bw1: 1.0, bw3: -1.0}),
        4,
        (
            [
                0.6830127018922193,
                -0.1830127018922193,
                -0.6830127018922193j,
                0.1830127018922193j,
                0.3535533905932738,
                -0.3535533905932738j,
                0.3535533905932738j,
                (0.3535533905932738 + 0j),
                -1.5,
                0.9999999999999999,
                0.4999999999999999,
            ],
            [
                X(0),
                X(0) @ Z(1),
                Y(0),
                Y(0) @ Z(1),
                X(0) @ X(1),
                X(0) @ Y(1),
                Y(0) @ X(1),
                Y(0) @ Y(1),
                I(),
                Z(1),
                Z(0),
            ],
        ),
    ),
    (
        BoseSentence({bw2: 1.0, bw3: -1.0}),
        4,
        (
            [
                1.024519052838329,
                -0.27451905283832895,
                1.024519052838329j,
                -0.27451905283832895j,
                0.5303300858899106,
                0.5303300858899106j,
                -0.5303300858899106j,
                (0.5303300858899106 + 0j),
                -0.6830127018922194,
                0.18301270189221933,
                -0.6830127018922194j,
                0.18301270189221933j,
                -0.35355339059327384,
                -0.35355339059327384j,
                0.35355339059327384j,
                (-0.35355339059327384 + 0j),
                -0.3415063509461095,
                0.09150635094610962,
                -0.3415063509461095j,
                0.09150635094610962j,
                -0.1767766952966367,
                -0.1767766952966367j,
                0.1767766952966367j,
                (-0.1767766952966367 + 0j),
                -1.5,
                0.9999999999999999,
                0.4999999999999999,
            ],
            [
                X(2),
                X(2) @ Z(3),
                Y(2),
                Y(2) @ Z(3),
                X(2) @ X(3),
                X(2) @ Y(3),
                Y(2) @ X(3),
                Y(2) @ Y(3),
                Z(1) @ X(2),
                Z(1) @ X(2) @ Z(3),
                Z(1) @ Y(2),
                Z(1) @ Y(2) @ Z(3),
                Z(1) @ X(2) @ X(3),
                Z(1) @ X(2) @ Y(3),
                Z(1) @ Y(2) @ X(3),
                Z(1) @ Y(2) @ Y(3),
                Z(0) @ X(2),
                Z(0) @ X(2) @ Z(3),
                Z(0) @ Y(2),
                Z(0) @ Y(2) @ Z(3),
                Z(0) @ X(2) @ X(3),
                Z(0) @ X(2) @ Y(3),
                Z(0) @ Y(2) @ X(3),
                Z(0) @ Y(2) @ Y(3),
                I(),
                Z(1),
                Z(0),
            ],
        ),
    ),
]


@pytest.mark.parametrize("bose_op, nstates, result", BOSE_SEN_AND_OPS)
def test_binary_mapping_bosesentence(bose_op, nstates, result):
    """Test that the binary_mapping function returns the correct qubit operator."""
    qubit_op = binary_mapping(bose_op, nstates_boson=nstates)
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
def test_return_binary_mapping_sum(bose_op):
    """Test that the correct type is returned for binary mapping
    when ps is set to False."""

    qubit_op = binary_mapping(bose_op, ps=False)
    assert isinstance(qubit_op, qml.ops.Sum)


@pytest.mark.parametrize(
    "bose_op",
    [
        BoseWord({(0, 0): "-"}),
        BoseSentence({BoseWord({(0, 0): "-"}): 1.0, BoseWord({(0, 1): "-"}): 1.0}),
    ],
)
def test_return_binary_mapping_ps(bose_op):
    """Test that the correct type is returned for binary mapping
    when ps is set to True."""

    qubit_op = binary_mapping(bose_op, ps=True)
    assert isinstance(qubit_op, qml.pauli.PauliSentence)


@pytest.mark.parametrize(
    ("bose_op, wire_map, result"),
    [
        (
            BoseWord({(0, 0): "+"}),
            {0: 1, 1: 2},
            (
                [
                    0.6830127018922193,
                    0.3535533905932738,
                    -0.3535533905932738j,
                    -0.1830127018922193,
                    -0.6830127018922193j,
                    0.3535533905932738j,
                    (0.3535533905932738 + 0j),
                    0.1830127018922193j,
                ],
                [
                    X(1),
                    X(1) @ X(2),
                    X(1) @ Y(2),
                    X(1) @ Z(2),
                    Y(1),
                    Y(1) @ X(2),
                    Y(1) @ Y(2),
                    Y(1) @ Z(2),
                ],
            ),
        ),
        (
            BoseSentence({BoseWord({(0, 0): "-"}): 1.0, BoseWord({(0, 1): "-"}): -1.0}),
            {0: "a", 1: "b", 2: "c", 3: "d"},
            (
                [
                    0.6830127018922193,
                    -0.1830127018922193,
                    0.6830127018922193j,
                    -0.1830127018922193j,
                    0.3535533905932738,
                    0.3535533905932738j,
                    -0.3535533905932738j,
                    (0.3535533905932738 + 0j),
                    -0.6830127018922193,
                    0.1830127018922193,
                    -0.6830127018922193j,
                    0.1830127018922193j,
                    -0.3535533905932738,
                    -0.3535533905932738j,
                    0.3535533905932738j,
                    (-0.3535533905932738 + 0j),
                ],
                [
                    X("a"),
                    X("a") @ Z("b"),
                    Y("a"),
                    Y("a") @ Z("b"),
                    X("a") @ X("b"),
                    X("a") @ Y("b"),
                    Y("a") @ X("b"),
                    Y("a") @ Y("b"),
                    X("c"),
                    X("c") @ Z("d"),
                    Y("c"),
                    Y("c") @ Z("d"),
                    X("c") @ X("d"),
                    X("c") @ Y("d"),
                    Y("c") @ X("d"),
                    Y("c") @ Y("d"),
                ],
            ),
        ),
    ],
)
def test_binary_mapping_wiremap(bose_op, wire_map, result):
    """Test that the binary_mapping function returns the correct qubit operator."""
    qubit_op = binary_mapping(bose_op, nstates_boson=4, wire_map=wire_map)
    qubit_op.simplify(tol=1e-8)

    expected_op = pauli_sentence(qml.Hamiltonian(result[0], result[1]))
    expected_op.simplify(tol=1e-8)
    assert qubit_op == expected_op


def test_d_error_binary():
    """Test that an error is raised if invalid number of states is provided."""
    bw = BoseWord({(0, 0): "-"})
    with pytest.raises(
        ValueError, match="Number of bosonic states cannot be less than 2, provided 1."
    ):
        binary_mapping(bw, nstates_boson=1)
