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
"""Unit testing of conversion functions for Fermi operators."""
import pytest

import pennylane as qml
from pennylane.fermi.conversion import jordan_wigner
from pennylane.fermi.fermionic import FermiSentence, FermiWord
from pennylane.ops import Identity, SProd
from pennylane.pauli import PauliSentence, PauliWord
from pennylane.pauli.conversion import pauli_sentence

FERMI_WORDS_AND_OPS = [
    (
        FermiWord({(0, 0): "+"}),
        # trivial case of a creation operator, 0^ -> (X_0 - iY_0) / 2
        ([0.5, -0.5j], [qml.PauliX(0), qml.PauliY(0)]),
    ),
    (
        FermiWord({(0, 0): "-"}),
        # trivial case of an annihilation operator, 0 -> (X_0 + iY_0) / 2
        ([(0.5 + 0j), (0.0 + 0.5j)], [qml.PauliX(0), qml.PauliY(0)]),
    ),
    (
        FermiWord({(0, 0): "+", (1, 0): "-"}),
        # obtained with openfermion using: jordan_wigner(FermionOperator('0^ 0', 1))
        # reformatted the original openfermion output: (0.5+0j) [] + (-0.5+0j) [Z0]
        ([(0.5 + 0j), (-0.5 + 0j)], [qml.Identity(0), qml.PauliZ(0)]),
    ),
    (
        FermiWord({(0, 0): "-", (1, 0): "+"}),
        # obtained with openfermion using: jordan_wigner(FermionOperator('0 0^'))
        # reformatted the original openfermion output: (0.5+0j) [] + (0.5+0j) [Z0]
        ([(0.5 + 0j), (0.5 + 0j)], [qml.Identity(0), qml.PauliZ(0)]),
    ),
    (
        FermiWord({(0, 0): "-", (1, 1): "+"}),
        # obtained with openfermion using: jordan_wigner(FermionOperator('0 1^'))
        # reformatted the original openfermion output:
        # (-0.25+0j) [X0 X1] +
        # 0.25j [X0 Y1] +
        # -0.25j [Y0 X1] +
        # (-0.25+0j) [Y0 Y1]
        (
            [(-0.25 + 0j), 0.25j, -0.25j, (-0.25 + 0j)],
            [
                qml.PauliX(0) @ qml.PauliX(1),
                qml.PauliX(0) @ qml.PauliY(1),
                qml.PauliY(0) @ qml.PauliX(1),
                qml.PauliY(0) @ qml.PauliY(1),
            ],
        ),
    ),
    (
        FermiWord({(0, 1): "-", (1, 0): "+"}),
        # obtained with openfermion using: jordan_wigner(FermionOperator('1 0^'))
        # reformatted the original openfermion output:
        # (-0.25+0j) [X0 X1] +
        # -0.25j [X0 Y1] +
        # 0.25j [Y0 X1] +
        # (-0.25+0j) [Y0 Y1]
        (
            [(-0.25 + 0j), -0.25j, 0.25j, (-0.25 + 0j)],
            [
                qml.PauliX(0) @ qml.PauliX(1),
                qml.PauliX(0) @ qml.PauliY(1),
                qml.PauliY(0) @ qml.PauliX(1),
                qml.PauliY(0) @ qml.PauliY(1),
            ],
        ),
    ),
    (
        FermiWord({(0, 3): "+", (1, 0): "-"}),
        # obtained with openfermion using: jordan_wigner(FermionOperator('3^ 0', 1))
        # reformatted the original openfermion output
        (
            [(0.25 + 0j), -0.25j, 0.25j, (0.25 + 0j)],
            [
                qml.PauliX(0) @ qml.PauliZ(1) @ qml.PauliZ(2) @ qml.PauliX(3),
                qml.PauliX(0) @ qml.PauliZ(1) @ qml.PauliZ(2) @ qml.PauliY(3),
                qml.PauliY(0) @ qml.PauliZ(1) @ qml.PauliZ(2) @ qml.PauliX(3),
                qml.PauliY(0) @ qml.PauliZ(1) @ qml.PauliZ(2) @ qml.PauliY(3),
            ],
        ),
    ),
    (
        FermiWord({(0, 0): "-", (1, 3): "+"}),
        # obtained with openfermion using: jordan_wigner(FermionOperator('0 3^'))
        # reformatted the original openfermion output
        (
            [(-0.25 + 0j), 0.25j, -0.25j, (-0.25 + 0j)],
            [
                qml.PauliX(0) @ qml.PauliZ(1) @ qml.PauliZ(2) @ qml.PauliX(3),
                qml.PauliX(0) @ qml.PauliZ(1) @ qml.PauliZ(2) @ qml.PauliY(3),
                qml.PauliY(0) @ qml.PauliZ(1) @ qml.PauliZ(2) @ qml.PauliX(3),
                qml.PauliY(0) @ qml.PauliZ(1) @ qml.PauliZ(2) @ qml.PauliY(3),
            ],
        ),
    ),
    (
        FermiWord({(0, 3): "-", (1, 0): "+"}),
        # obtained with openfermion using: jordan_wigner(FermionOperator('3 0^'))
        # reformatted the original openfermion output
        (
            [(-0.25 + 0j), -0.25j, 0.25j, (-0.25 + 0j)],
            [
                qml.PauliX(0) @ qml.PauliZ(1) @ qml.PauliZ(2) @ qml.PauliX(3),
                qml.PauliX(0) @ qml.PauliZ(1) @ qml.PauliZ(2) @ qml.PauliY(3),
                qml.PauliY(0) @ qml.PauliZ(1) @ qml.PauliZ(2) @ qml.PauliX(3),
                qml.PauliY(0) @ qml.PauliZ(1) @ qml.PauliZ(2) @ qml.PauliY(3),
            ],
        ),
    ),
    (
        FermiWord({(0, 1): "+", (1, 4): "-"}),
        # obtained with openfermion using: jordan_wigner(FermionOperator('1^ 4', 1))
        # reformatted the original openfermion output
        (
            [(0.25 + 0j), 0.25j, -0.25j, (0.25 + 0j)],
            [
                qml.PauliX(1) @ qml.PauliZ(2) @ qml.PauliZ(3) @ qml.PauliX(4),
                qml.PauliX(1) @ qml.PauliZ(2) @ qml.PauliZ(3) @ qml.PauliY(4),
                qml.PauliY(1) @ qml.PauliZ(2) @ qml.PauliZ(3) @ qml.PauliX(4),
                qml.PauliY(1) @ qml.PauliZ(2) @ qml.PauliZ(3) @ qml.PauliY(4),
            ],
        ),
    ),
    (
        FermiWord({(0, 1): "+", (1, 1): "+", (2, 1): "-", (3, 1): "-"}),  # [1, 1, 1, 1],
        # obtained with openfermion using: jordan_wigner(FermionOperator('1^ 1^ 1 1', 1))
        ([0], [qml.Identity(1)]),
    ),
    (
        FermiWord({(0, 3): "+", (1, 1): "+", (2, 3): "-", (3, 1): "-"}),  # [3, 1, 3, 1],
        # obtained with openfermion using: jordan_wigner(FermionOperator('3^ 1^ 3 1', 1))
        # reformatted the original openfermion output
        (
            [(-0.25 + 0j), (0.25 + 0j), (-0.25 + 0j), (0.25 + 0j)],
            [qml.Identity(0), qml.PauliZ(1), qml.PauliZ(1) @ qml.PauliZ(3), qml.PauliZ(3)],
        ),
    ),
    (
        FermiWord({(0, 3): "+", (1, 1): "-", (2, 3): "+", (3, 1): "-"}),  # [3, 1, 3, 1],
        # obtained with openfermion using: jordan_wigner(FermionOperator('3^ 1 3^ 1', 1))
        ([0], [qml.Identity(1)]),
    ),
    (
        FermiWord({(0, 1): "+", (1, 0): "-", (2, 1): "+", (3, 1): "-"}),  # [1, 0, 1, 1],
        # obtained with openfermion using: jordan_wigner(FermionOperator('1^ 0 1^ 1', 1))
        ([0], [qml.Identity(0)]),
    ),
    (
        FermiWord({(0, 1): "+", (1, 1): "-", (2, 0): "+", (3, 0): "-"}),  # [1, 1, 0, 0],
        # obtained with openfermion using: jordan_wigner(FermionOperator('1^ 1 0^ 0', 1))
        (
            [(0.25 + 0j), (-0.25 + 0j), (0.25 + 0j), (-0.25 + 0j)],
            [qml.Identity(0), qml.PauliZ(0), qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliZ(1)],
        ),
    ),
    (
        FermiWord({(0, 5): "+", (1, 5): "-", (2, 5): "+", (3, 5): "-"}),  # [5, 5, 5, 5],
        # obtained with openfermion using: jordan_wigner(FermionOperator('5^ 5 5^ 5', 1))
        (
            [(0.5 + 0j), (-0.5 + 0j)],
            [qml.Identity(0), qml.PauliZ(5)],
        ),
    ),
    (
        FermiWord({(0, 3): "+", (1, 3): "-", (2, 3): "+", (3, 1): "-"}),
        # obtained with openfermion using: jordan_wigner(FermionOperator('3^ 3 3^ 1', 1))
        (
            [(0.25 + 0j), (-0.25j), (0.25j), (0.25 + 0j)],
            [
                qml.PauliX(1) @ qml.PauliZ(2) @ qml.PauliX(3),
                qml.PauliX(1) @ qml.PauliZ(2) @ qml.PauliY(3),
                qml.PauliY(1) @ qml.PauliZ(2) @ qml.PauliX(3),
                qml.PauliY(1) @ qml.PauliZ(2) @ qml.PauliY(3),
            ],
        ),
    ),
]

# can't be tested with conversion to operators yet, because the resulting operators
# are too complicated for qml.equal to successfully compare
FERMI_WORDS_AND_OPS_EXTENDED = [
    (
        FermiWord({(0, 3): "+", (1, 0): "-", (2, 2): "+", (3, 1): "-"}),
        # obtained with openfermion using: jordan_wigner(FermionOperator('3^ 0 2^ 1', 1))
        (
            [
                (-0.0625 + 0j),
                0.0625j,
                0.0625j,
                (0.0625 + 0j),
                -0.0625j,
                (-0.0625 + 0j),
                (-0.0625 + 0j),
                0.0625j,
                -0.0625j,
                (-0.0625 + 0j),
                (-0.0625 + 0j),
                0.0625j,
                (0.0625 + 0j),
                -0.0625j,
                -0.0625j,
                (-0.0625 + 0j),
            ],
            [
                qml.PauliX(0) @ qml.PauliX(1) @ qml.PauliX(2) @ qml.PauliX(3),
                qml.PauliX(0) @ qml.PauliX(1) @ qml.PauliX(2) @ qml.PauliY(3),
                qml.PauliX(0) @ qml.PauliX(1) @ qml.PauliY(2) @ qml.PauliX(3),
                qml.PauliX(0) @ qml.PauliX(1) @ qml.PauliY(2) @ qml.PauliY(3),
                qml.PauliX(0) @ qml.PauliY(1) @ qml.PauliX(2) @ qml.PauliX(3),
                qml.PauliX(0) @ qml.PauliY(1) @ qml.PauliX(2) @ qml.PauliY(3),
                qml.PauliX(0) @ qml.PauliY(1) @ qml.PauliY(2) @ qml.PauliX(3),
                qml.PauliX(0) @ qml.PauliY(1) @ qml.PauliY(2) @ qml.PauliY(3),
                qml.PauliY(0) @ qml.PauliX(1) @ qml.PauliX(2) @ qml.PauliX(3),
                qml.PauliY(0) @ qml.PauliX(1) @ qml.PauliX(2) @ qml.PauliY(3),
                qml.PauliY(0) @ qml.PauliX(1) @ qml.PauliY(2) @ qml.PauliX(3),
                qml.PauliY(0) @ qml.PauliX(1) @ qml.PauliY(2) @ qml.PauliY(3),
                qml.PauliY(0) @ qml.PauliY(1) @ qml.PauliX(2) @ qml.PauliX(3),
                qml.PauliY(0) @ qml.PauliY(1) @ qml.PauliX(2) @ qml.PauliY(3),
                qml.PauliY(0) @ qml.PauliY(1) @ qml.PauliY(2) @ qml.PauliX(3),
                qml.PauliY(0) @ qml.PauliY(1) @ qml.PauliY(2) @ qml.PauliY(3),
            ],
        ),
    ),
    (
        FermiWord({(0, 3): "-", (1, 0): "+", (2, 2): "-", (3, 1): "+"}),
        # obtained with openfermion using: jordan_wigner(FermionOperator('3 0^ 2 1^'))
        (
            [
                (-0.0625 + 0j),
                -0.0625j,
                -0.0625j,
                (0.0625 + 0j),
                0.0625j,
                (-0.0625 + 0j),
                (-0.0625 + 0j),
                -0.0625j,
                0.0625j,
                (-0.0625 + 0j),
                (-0.0625 + 0j),
                -0.0625j,
                (0.0625 + 0j),
                0.0625j,
                0.0625j,
                (-0.0625 + 0j),
            ],
            [
                qml.PauliX(0) @ qml.PauliX(1) @ qml.PauliX(2) @ qml.PauliX(3),
                qml.PauliX(0) @ qml.PauliX(1) @ qml.PauliX(2) @ qml.PauliY(3),
                qml.PauliX(0) @ qml.PauliX(1) @ qml.PauliY(2) @ qml.PauliX(3),
                qml.PauliX(0) @ qml.PauliX(1) @ qml.PauliY(2) @ qml.PauliY(3),
                qml.PauliX(0) @ qml.PauliY(1) @ qml.PauliX(2) @ qml.PauliX(3),
                qml.PauliX(0) @ qml.PauliY(1) @ qml.PauliX(2) @ qml.PauliY(3),
                qml.PauliX(0) @ qml.PauliY(1) @ qml.PauliY(2) @ qml.PauliX(3),
                qml.PauliX(0) @ qml.PauliY(1) @ qml.PauliY(2) @ qml.PauliY(3),
                qml.PauliY(0) @ qml.PauliX(1) @ qml.PauliX(2) @ qml.PauliX(3),
                qml.PauliY(0) @ qml.PauliX(1) @ qml.PauliX(2) @ qml.PauliY(3),
                qml.PauliY(0) @ qml.PauliX(1) @ qml.PauliY(2) @ qml.PauliX(3),
                qml.PauliY(0) @ qml.PauliX(1) @ qml.PauliY(2) @ qml.PauliY(3),
                qml.PauliY(0) @ qml.PauliY(1) @ qml.PauliX(2) @ qml.PauliX(3),
                qml.PauliY(0) @ qml.PauliY(1) @ qml.PauliX(2) @ qml.PauliY(3),
                qml.PauliY(0) @ qml.PauliY(1) @ qml.PauliY(2) @ qml.PauliX(3),
                qml.PauliY(0) @ qml.PauliY(1) @ qml.PauliY(2) @ qml.PauliY(3),
            ],
        ),
    ),
    (
        FermiWord({(0, 3): "-", (1, 0): "+", (2, 2): "-", (3, 1): "+"}),
        # obtained with openfermion using: jordan_wigner(FermionOperator('3 0^ 2 1^'))
        (
            [
                (-0.0625 + 0j),
                -0.0625j,
                -0.0625j,
                (0.0625 + 0j),
                0.0625j,
                (-0.0625 + 0j),
                (-0.0625 + 0j),
                -0.0625j,
                0.0625j,
                (-0.0625 + 0j),
                (-0.0625 + 0j),
                -0.0625j,
                (0.0625 + 0j),
                0.0625j,
                0.0625j,
                (-0.0625 + 0j),
            ],
            [
                qml.PauliX(0) @ qml.PauliX(1) @ qml.PauliX(2) @ qml.PauliX(3),
                qml.PauliX(0) @ qml.PauliX(1) @ qml.PauliX(2) @ qml.PauliY(3),
                qml.PauliX(0) @ qml.PauliX(1) @ qml.PauliY(2) @ qml.PauliX(3),
                qml.PauliX(0) @ qml.PauliX(1) @ qml.PauliY(2) @ qml.PauliY(3),
                qml.PauliX(0) @ qml.PauliY(1) @ qml.PauliX(2) @ qml.PauliX(3),
                qml.PauliX(0) @ qml.PauliY(1) @ qml.PauliX(2) @ qml.PauliY(3),
                qml.PauliX(0) @ qml.PauliY(1) @ qml.PauliY(2) @ qml.PauliX(3),
                qml.PauliX(0) @ qml.PauliY(1) @ qml.PauliY(2) @ qml.PauliY(3),
                qml.PauliY(0) @ qml.PauliX(1) @ qml.PauliX(2) @ qml.PauliX(3),
                qml.PauliY(0) @ qml.PauliX(1) @ qml.PauliX(2) @ qml.PauliY(3),
                qml.PauliY(0) @ qml.PauliX(1) @ qml.PauliY(2) @ qml.PauliX(3),
                qml.PauliY(0) @ qml.PauliX(1) @ qml.PauliY(2) @ qml.PauliY(3),
                qml.PauliY(0) @ qml.PauliY(1) @ qml.PauliX(2) @ qml.PauliX(3),
                qml.PauliY(0) @ qml.PauliY(1) @ qml.PauliX(2) @ qml.PauliY(3),
                qml.PauliY(0) @ qml.PauliY(1) @ qml.PauliY(2) @ qml.PauliX(3),
                qml.PauliY(0) @ qml.PauliY(1) @ qml.PauliY(2) @ qml.PauliY(3),
            ],
        ),
    ),
    (
        FermiWord({(0, 0): "-", (1, 0): "+", (2, 2): "+", (3, 1): "-"}),
        # obtained with openfermion using: jordan_wigner(FermionOperator('0 0^ 2^ 1'))
        (
            [
                (0.125 + 0j),
                -0.125j,
                0.125j,
                (0.125 + 0j),
                (0.125 + 0j),
                -0.125j,
                0.125j,
                (0.125 + 0j),
            ],
            [
                qml.PauliZ(0) @ qml.PauliX(1) @ qml.PauliX(2),
                qml.PauliZ(0) @ qml.PauliX(1) @ qml.PauliY(2),
                qml.PauliZ(0) @ qml.PauliY(1) @ qml.PauliX(2),
                qml.PauliZ(0) @ qml.PauliY(1) @ qml.PauliY(2),
                qml.PauliX(1) @ qml.PauliX(2),
                qml.PauliX(1) @ qml.PauliY(2),
                qml.PauliY(1) @ qml.PauliX(2),
                qml.PauliY(1) @ qml.PauliY(2),
            ],
        ),
    ),
]


@pytest.mark.parametrize("fermionic_op, result", FERMI_WORDS_AND_OPS + FERMI_WORDS_AND_OPS_EXTENDED)
def test_jordan_wigner_fermi_word_ps(fermionic_op, result):
    """Test that the jordan_wigner function returns the correct qubit operator."""
    # convert FermiWord to PauliSentence and simplify
    qubit_op = jordan_wigner(fermionic_op, ps=True)
    qubit_op.simplify()

    # get expected op as PauliSentence and simplify
    expected_op = pauli_sentence(qml.Hamiltonian(result[0], result[1]))
    expected_op.simplify()

    assert qubit_op == expected_op


@pytest.mark.parametrize("fermionic_op, result", FERMI_WORDS_AND_OPS + FERMI_WORDS_AND_OPS_EXTENDED)
def test_jordan_wigner_fermi_word_operation(fermionic_op, result):
    wires = fermionic_op.wires or [0]

    qubit_op = jordan_wigner(fermionic_op)

    expected_op = pauli_sentence(qml.Hamiltonian(result[0], result[1]))
    expected_op = expected_op.operation(wires)

    qml.assert_equal(qubit_op.simplify(), expected_op.simplify())


def test_jordan_wigner_for_identity():
    """Test that the jordan_wigner function returns the correct qubit operator for Identity."""
    qml.assert_equal(jordan_wigner(FermiWord({})), qml.Identity(0))


def test_jordan_wigner_for_identity_ps():
    """Test that the jordan_wigner function returns the correct PauliSentence for Identity when ps=True."""
    assert jordan_wigner(FermiWord({}), ps=True) == PauliSentence({PauliWord({0: "I"}): 1.0 + 0.0j})


@pytest.mark.parametrize(
    "operator",
    (
        FermiWord({(0, 1): "-", (1, 0): "+", (2, 2): "-", (3, 1): "-"}),  # ('1 0^ 2 1')
        FermiWord({(0, 1): "-", (1, 0): "+", (2, 2): "+", (3, 1): "-"}),  # ('1 0^ 2^ 1')
        FermiWord({(0, 3): "-", (1, 0): "+", (2, 2): "+", (3, 3): "-"}),  # ('3 0^ 2^ 3')
        FermiWord({(0, 3): "-", (1, 2): "+", (2, 2): "+", (3, 3): "-"}),  # ('3 2^ 2^ 3')
    ),
)
def test_jordan_wigner_for_null_operator_fermi_word_ps(operator):
    """Test that the jordan_wigner function works when the result is 0"""
    # in PauliSentence return format, returns None
    assert jordan_wigner(operator, ps=True).simplify() is None

    # in operation return format, '0 * I'
    op = jordan_wigner(operator).simplify()

    assert isinstance(op, SProd)
    assert isinstance(op.base, Identity)
    assert op.scalar == 0


fw1 = FermiWord({(0, 0): "+", (1, 1): "-"})
fw2 = FermiWord({(0, 0): "+", (1, 0): "-"})
fw3 = FermiWord({(0, 0): "+", (1, 3): "-", (2, 0): "+", (3, 4): "-"})
fw4 = FermiWord({})
fw5 = FermiWord({(0, 3): "+", (1, 2): "-"})
fw6 = FermiWord({(0, 1): "+", (1, 4): "-"})


def test_empty_fermi_sentence():
    """Test that an empty FermiSentence (fermi null operator) is
    converted to an empty PauliSentence or the null operator"""
    op = FermiSentence({})

    ps_op = jordan_wigner(op, ps=True)
    ps_op.simplify()
    assert ps_op == PauliSentence({})

    op = jordan_wigner(op).simplify()
    assert isinstance(op, SProd)
    assert isinstance(op.base, Identity)
    assert op.scalar == 0


def test_fermi_sentence_identity():
    """Test that a FermiSentence composed of a single Identity operator
    converts to PauliSentence and operation as expected"""
    op = FermiSentence({fw4: 1})
    ps = PauliSentence({PauliWord({}): 1})

    ps_op = jordan_wigner(op, ps=True)
    qubit_op = jordan_wigner(op)

    assert ps_op == ps

    result = ps.operation(wire_order=[0])
    qml.assert_equal(qubit_op.simplify(), result.simplify())


# used above results translating fermiword --> paulisentence, to calculate expected output by hand
FERMI_AND_PAULI_SENTENCES = [
    (FermiSentence({fw4: 0, fw2: 0}), PauliSentence({})),  # all 0 coeffs FermiSentence to null
    (
        FermiSentence({fw2: 2}),
        PauliSentence({PauliWord({}): (1 + 0j), PauliWord({0: "Z"}): (-1 + 0j)}),
    ),
    (
        FermiSentence({fw1: 1, fw2: 1}),
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
        FermiSentence({fw1: 1j, fw2: -2}),
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
        FermiSentence({fw1: -2, fw5: 1j}),
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
        FermiSentence({fw6: 1, fw2: 2}),
        PauliSentence(
            {
                PauliWord({0: "I"}): 1.0,
                PauliWord({0: "Z"}): -1.0,
                PauliWord({1: "X", 2: "Z", 3: "Z", 4: "X"}): 0.25,
                PauliWord({1: "X", 2: "Z", 3: "Z", 4: "Y"}): 0.25j,
                PauliWord({1: "Y", 2: "Z", 3: "Z", 4: "X"}): -0.25j,
                PauliWord({1: "Y", 2: "Z", 3: "Z", 4: "Y"}): 0.25,
            }
        ),
    ),
    (
        FermiSentence({fw5: 1, fw6: 1}),
        PauliSentence(
            {
                PauliWord({1: "X", 2: "Z", 3: "Z", 4: "X"}): 0.25,
                PauliWord({1: "X", 2: "Z", 3: "Z", 4: "Y"}): 0.25j,
                PauliWord({1: "Y", 2: "Z", 3: "Z", 4: "X"}): -0.25j,
                PauliWord({1: "Y", 2: "Z", 3: "Z", 4: "Y"}): 0.25,
                PauliWord({2: "X", 3: "X"}): 0.25,
                PauliWord({2: "X", 3: "Y"}): -0.25j,
                PauliWord({2: "Y", 3: "X"}): 0.25j,
                PauliWord({2: "Y", 3: "Y"}): 0.25,
            }
        ),
    ),
]


@pytest.mark.parametrize("fermionic_op, result", FERMI_AND_PAULI_SENTENCES)
def test_jordan_wigner_for_fermi_sentence_ps(fermionic_op, result):
    qubit_op = jordan_wigner(fermionic_op, ps=True)
    qubit_op.simplify()

    assert qubit_op == result


@pytest.mark.parametrize("fermionic_op, result", FERMI_AND_PAULI_SENTENCES)
def test_jordan_wigner_for_fermi_sentence_operation(fermionic_op, result):
    wires = fermionic_op.wires or [0]

    qubit_op = jordan_wigner(fermionic_op)
    result = result.operation(wires)

    qml.assert_equal(qubit_op.simplify(), result.simplify())


def test_error_is_raised_for_incompatible_type():
    """Test that an error is raised in the input is not a FermiWord or FermiSentence"""

    with pytest.raises(ValueError, match="fermi_operator must be a FermiWord or FermiSentence"):
        jordan_wigner(qml.PauliX(0))


WIRE_MAP_FOR_FERMI_SENTENCE = [
    (
        None,
        [
            qml.s_prod(-0.25j, qml.prod(qml.PauliY(0), qml.PauliX(1))),
            qml.s_prod((0.25 + 0j), qml.prod(qml.PauliY(0), qml.PauliY(1))),
            qml.s_prod((0.25 + 0j), qml.prod(qml.PauliX(0), qml.PauliX(1))),
            qml.s_prod(0.25j, qml.prod(qml.PauliX(0), qml.PauliY(1))),
            qml.s_prod((0.5 + 0j), qml.Identity(0)),
            qml.s_prod((-0.5 + 0j), qml.PauliZ(0)),
        ],
    ),
    (
        {0: 0, 1: 1},
        [
            qml.s_prod(-0.25j, qml.prod(qml.PauliY(0), qml.PauliX(1))),
            qml.s_prod((0.25 + 0j), qml.prod(qml.PauliY(0), qml.PauliY(1))),
            qml.s_prod((0.25 + 0j), qml.prod(qml.PauliX(0), qml.PauliX(1))),
            qml.s_prod(0.25j, qml.prod(qml.PauliX(0), qml.PauliY(1))),
            qml.s_prod((0.5 + 0j), qml.Identity(0)),
            qml.s_prod((-0.5 + 0j), qml.PauliZ(0)),
        ],
    ),
    (
        {0: 1, 1: 0},
        [
            qml.s_prod(-0.25j, qml.prod(qml.PauliY(1), qml.PauliX(0))),
            qml.s_prod((0.25 + 0j), qml.prod(qml.PauliY(1), qml.PauliY(0))),
            qml.s_prod((0.25 + 0j), qml.prod(qml.PauliX(1), qml.PauliX(0))),
            qml.s_prod(0.25j, qml.prod(qml.PauliX(1), qml.PauliY(0))),
            qml.s_prod((0.5 + 0j), qml.Identity(1)),
            qml.s_prod((-0.5 + 0j), qml.PauliZ(1)),
        ],
    ),
    (
        {0: 3, 1: 2},
        [
            qml.s_prod(-0.25j, qml.prod(qml.PauliY(3), qml.PauliX(2))),
            qml.s_prod((0.25 + 0j), qml.prod(qml.PauliY(3), qml.PauliY(2))),
            qml.s_prod((0.25 + 0j), qml.prod(qml.PauliX(3), qml.PauliX(2))),
            qml.s_prod(0.25j, qml.prod(qml.PauliX(3), qml.PauliY(2))),
            qml.s_prod((0.5 + 0j), qml.Identity(3)),
            qml.s_prod((-0.5 + 0j), qml.PauliZ(3)),
        ],
    ),
    (
        {0: "b", 1: "a"},
        [
            qml.s_prod(-0.25j, qml.prod(qml.PauliY("b"), qml.PauliX("a"))),
            qml.s_prod((0.25 + 0j), qml.prod(qml.PauliY("b"), qml.PauliY("a"))),
            qml.s_prod((0.25 + 0j), qml.prod(qml.PauliX("b"), qml.PauliX("a"))),
            qml.s_prod(0.25j, qml.prod(qml.PauliX("b"), qml.PauliY("a"))),
            qml.s_prod((0.5 + 0j), qml.Identity("b")),
            qml.s_prod((-0.5 + 0j), qml.PauliZ("b")),
        ],
    ),
]


@pytest.mark.parametrize("wire_map, ops", WIRE_MAP_FOR_FERMI_SENTENCE)
def test_providing_wire_map_fermi_sentence_to_operation(wire_map, ops):
    fs = FermiSentence(
        {FermiWord({(0, 0): "+", (1, 1): "-"}): 1, FermiWord({(0, 0): "+", (1, 0): "-"}): 1}
    )

    op = jordan_wigner(fs, wire_map=wire_map)
    result = qml.sum(*ops)

    assert op.wires == result.wires

    # converting to Pauli representation for comparison because
    # qml.equal isn't playing nicely with term ordering
    assert pauli_sentence(op) == pauli_sentence(result)


@pytest.mark.parametrize("wire_map, ops", WIRE_MAP_FOR_FERMI_SENTENCE)
def test_providing_wire_map_fermi_sentence_to_ps(wire_map, ops):
    fs = FermiSentence(
        {FermiWord({(0, 0): "+", (1, 1): "-"}): 1, FermiWord({(0, 0): "+", (1, 0): "-"}): 1}
    )

    op = jordan_wigner(fs, wire_map=wire_map, ps=True)
    result_op = qml.sum(*ops)
    ps = pauli_sentence(result_op)

    ps.simplify()
    op.simplify()

    assert ps == op


WIRE_MAP_FOR_FERMI_WORDS = [
    (
        None,
        [
            qml.s_prod(-0.25j, qml.prod(qml.PauliY(0), qml.PauliX(1))),
            qml.s_prod(-0.25 + 0j, qml.prod(qml.PauliY(0), qml.PauliY(1))),
            qml.s_prod(0.25 + 0j, qml.prod(qml.PauliX(0), qml.PauliX(1))),
            qml.s_prod(-0.25j, qml.prod(qml.PauliX(0), qml.PauliY(1))),
        ],
    ),
    (
        {0: 3, 1: 2},
        [
            qml.s_prod(-0.25j, qml.prod(qml.PauliY(3), qml.PauliX(2))),
            qml.s_prod(-0.25 + 0j, qml.prod(qml.PauliY(3), qml.PauliY(2))),
            qml.s_prod(0.25 + 0j, qml.prod(qml.PauliX(3), qml.PauliX(2))),
            qml.s_prod(-0.25j, qml.prod(qml.PauliX(3), qml.PauliY(2))),
        ],
    ),
    (
        {0: "b", 1: "a"},
        [
            qml.s_prod(-0.25j, qml.prod(qml.PauliY("b"), qml.PauliX("a"))),
            qml.s_prod(-0.25 + 0j, qml.prod(qml.PauliY("b"), qml.PauliY("a"))),
            qml.s_prod(0.25 + 0j, qml.prod(qml.PauliX("b"), qml.PauliX("a"))),
            qml.s_prod(-0.25j, qml.prod(qml.PauliX("b"), qml.PauliY("a"))),
        ],
    ),
]


@pytest.mark.parametrize("wire_map, ops", WIRE_MAP_FOR_FERMI_WORDS)
def test_providing_wire_map_fermi_word_to_operation(wire_map, ops):
    w = FermiWord({(0, 0): "+", (1, 1): "+"})

    op = jordan_wigner(w, wire_map=wire_map)
    result = qml.sum(*ops)

    op.simplify()

    # converting to Pauli representation for comparison because
    # qml.equal isn't playing nicely with term ordering
    assert pauli_sentence(op) == pauli_sentence(result)


@pytest.mark.parametrize("wire_map, ops", WIRE_MAP_FOR_FERMI_WORDS)
def test_providing_wire_map_fermi_word_to_ps(wire_map, ops):
    w = FermiWord({(0, 0): "+", (1, 1): "+"})

    op = jordan_wigner(w, wire_map=wire_map, ps=True)
    result_op = qml.sum(*ops)
    ps = pauli_sentence(result_op)

    ps.simplify()
    op.simplify()

    assert ps == op


fs1 = FermiSentence({fw1: 1})


@pytest.mark.parametrize(
    "fermi_op, qubit_op_data, tol",
    (
        (fw1, (-0.25j, (0.25 + 0j), (0.25 + 0j), 0.25j), None),
        (fw1, (-0.25j, 0.25, 0.25, 0.25j), 0.0),
        (fw1, (-0.25j, 0.25, 0.25, 0.25j), 1.0e-12),
        (fs1, (-0.25j, (0.25 + 0j), (0.25 + 0j), 0.25j), None),
        (fs1, (-0.25j, 0.25, 0.25, 0.25j), 0.0),
        (fs1, (-0.25j, 0.25, 0.25, 0.25j), 1.0e-12),
    ),
)
def test_jordan_wigner_tolerance(fermi_op, qubit_op_data, tol):
    """Test that jordan_wigner properly removes negligible imaginary components"""
    op = jordan_wigner(fermi_op, tol=tol)
    assert isinstance(op.data[1], type(qubit_op_data[1]))
