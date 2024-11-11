import pytest

import pennylane as qml
from pennylane.labs.vibrational_ham import BoseWord, BoseSentence, binary_mapping
from pennylane.pauli import PauliSentence, PauliWord
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
]


@pytest.mark.parametrize("bose_op, d, result", BOSE_WORDS_AND_OPS)
def test_binary_mapping_boseword(bose_op, d, result):
    """Test that the binary_mapping function returns the correct qubit operator."""
    # convert BoseWord to PauliSentence and simplify
    qubit_op = binary_mapping(bose_op, d=d)
    qubit_op.simplify(tol=1e-8)

    # get expected op as PauliSentence and simplify
    expected_op = pauli_sentence(qml.Hamiltonian(result[0], result[1]))
    expected_op.simplify(tol=1e-8)
    print(qubit_op)
    assert qubit_op == expected_op
