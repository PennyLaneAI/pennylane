"""Unit testing of conversion functions for parity transform"""

import pytest

import pennylane as qml
from pennylane.fermi.conversion import parity_transform
from pennylane.fermi.fermionic import FermiSentence, FermiWord
from pennylane.ops import Identity, SProd
from pennylane.pauli import PauliSentence, PauliWord
from pennylane.pauli.conversion import pauli_sentence


def test_error_is_raised_for_incompatible_type():
    """Test that an error is raised in the input is not a FermiWord or FermiSentence"""
    with pytest.raises(ValueError, match="fermi_operator must be a FermiWord or FermiSentence"):
        parity_transform(qml.PauliX(0), 10)


def test_error_is_raised_for_dimension_mismatch():
    """Test that an error is raised if the number of qubits are not compatible with the FermiWord or FermiSentence"""

    with pytest.raises(
        ValueError,
        match="Can't create or annihilate a particle on qubit number 6 for a system with only 6 qubits",
    ):
        parity_transform(FermiWord({(0, 1): "-", (1, 0): "+", (2, 6): "-"}), 6)


FERMI_WORDS_AND_OPS = [
    (
        FermiWord({(0, 0): "+"}),
        1,
        # trivial case of a creation operator with one qubit, 0^ -> (X_0 - iY_0) / 2 : Same as Jordan-Wigner
        ([0.5, -0.5j], [qml.PauliX(0), qml.PauliY(0)]),
    ),
    (
        FermiWord({(0, 0): "-"}),
        1,
        # trivial case of an annihilation operator with one qubit , 0 -> (X_0 + iY_0) / 2 : Same as Jordan-Wigner
        ([(0.5 + 0j), (0.0 + 0.5j)], [qml.PauliX(0), qml.PauliY(0)]),
    ),
    (
        FermiWord({(0, 0): "+"}),
        2,
        # trivial case of a creation operator with two qubits, 0^ -> (X_0 @ X_1 - iY_0 @ X_1) / 2
        ([0.5, -0.5j], [qml.PauliX(0) @ qml.PauliX(1), qml.PauliY(0) @ qml.PauliX(1)]),
    ),
    (
        FermiWord({(0, 0): "-"}),
        2,
        # trivial case of an annihilation operator with two qubits , 0 -> (X_0 @ X_1 + iY_0 @ X_1) / 2
        (
            [(0.5 + 0j), (0.0 + 0.5j)],
            [qml.PauliX(0) @ qml.PauliX(1), qml.PauliY(0) @ qml.PauliX(1)],
        ),
    ),
    (
        FermiWord({(0, 0): "+", (1, 0): "-"}),
        4,
        # obtained with openfermion using: binary_code_transform(FermionOperator('0^ 0'), parity_code(n_qubits)) for n_qubits = 4
        # reformatted the original openfermion output: (0.5+0j) [] + (-0.5+0j) [Z0]
        ([(0.5 + 0j), (-0.5 + 0j)], [qml.Identity(0), qml.PauliZ(0)]),
    ),
    (
        FermiWord({(0, 0): "-", (1, 0): "+"}),
        4,
        # obtained with openfermion using: binary_code_transform(FermionOperator('0 0^'), parity_code(n_qubits)) for n_qubits = 4
        # reformatted the original openfermion output: (0.5+0j) [] + (0.5+0j) [Z0]
        ([(0.5 + 0j), (0.5 + 0j)], [qml.Identity(0), qml.PauliZ(0)]),
    ),
    (
        FermiWord({(0, 0): "-", (1, 1): "+"}),
        4,
        # obtained with openfermion using: binary_code_transform(FermionOperator('0 1^'), parity_code(n_qubits)) for n_qubits = 4
        # reformatted the original openfermion output:
        # (-0.25+0j) [X0] +
        # 0.25 [X0 Z1] +
        # (-0-0.25j) [Y0] +
        # 0.25j [Y0 Z1]
        (
            [(-0.25 + 0j), 0.25, (-0 - 0.25j), (0.25j)],
            [
                qml.PauliX(0),
                qml.PauliX(0) @ qml.PauliZ(1),
                qml.PauliY(0),
                qml.PauliY(0) @ qml.PauliZ(1),
            ],
        ),
    ),
    (
        FermiWord({(0, 3): "+", (1, 0): "-"}),
        4,
        # obtained with openfermion using: binary_code_transform(FermionOperator('3^ 0'), parity_code(n_qubits)) for n_qubits = 4
        # reformatted the original openfermion output
        #  -0.25 [X0 X1 X2 Z3] +
        #  0.25j [X0 X1 Y2] +
        #  (-0-0.25j) [Y0 X1 X2 Z3] +
        #   -0.25 [Y0 X1 Y2]
        (
            [(-0.25 + 0j), 0.25j, (-0 - 0.25j), -0.25],
            [
                qml.PauliX(0) @ qml.PauliX(1) @ qml.PauliX(2) @ qml.PauliZ(3),
                qml.PauliX(0) @ qml.PauliX(1) @ qml.PauliY(2),
                qml.PauliY(0) @ qml.PauliX(1) @ qml.PauliX(2) @ qml.PauliZ(3),
                qml.PauliY(0) @ qml.PauliX(1) @ qml.PauliY(2),
            ],
        ),
    ),
    (
        FermiWord({(0, 5): "+", (1, 5): "-", (2, 5): "+", (3, 5): "-"}),
        6,
        # obtained with openfermion using: binary_code_transform(FermionOperator('5^ 5 5^ 5'), parity_code(n_qubits)) with 6 qubits
        (
            [(0.5 + 0j), (-0.5 + 0j)],
            [qml.Identity(0), qml.PauliZ(4) @ qml.PauliZ(5)],
        ),
    ),
    (
        FermiWord({(0, 3): "+", (1, 3): "-", (2, 3): "+", (3, 1): "-"}),
        6,
        # obtained with openfermion using: binary_code_transform(FermionOperator('3^ 3 3^ 1'), parity_code(n_qubits)) with 6 qubits
        # -0.25 [Z0 X1 X2 Z3] +
        # 0.25j [Z0 X1 Y2] +
        # (-0-0.25j) [Y1 X2 Z3] +
        # -0.25 [Y1 Y2]
        (
            [-0.25, 0.25j, -0.25j, -0.25],
            [
                qml.PauliZ(0) @ qml.PauliX(1) @ qml.PauliX(2) @ qml.PauliZ(3),
                qml.PauliZ(0) @ qml.PauliX(1) @ qml.PauliY(2),
                qml.PauliY(1) @ qml.PauliX(2) @ qml.PauliZ(3),
                qml.PauliY(1) @ qml.PauliY(2),
            ],
        ),
    ),
    (
        FermiWord({(0, 1): "+", (1, 0): "-", (2, 1): "+", (3, 1): "-"}),
        6,
        # obtained with openfermion using: binary_code_transform(FermionOperator('1^ 0 1^ 1'), parity_code(n_qubits)) with 6 qubits
        ([0], [qml.Identity(0)]),
    ),
]

FERMI_OPS_COMPLEX = [
    (
        FermiWord({(0, 2): "-", (1, 0): "+", (2, 3): "+"}),
        4,
        # obtained with openfermion using: binary_code_transform(FermionOperator('2 0^ 3^'), parity_code(n_qubits)) for n_qubits = 4
        # reformatted the original openfermion output
        # (-0-0.125j) [X0 X1 Z2 Y3] +
        # 0.125 [X0 X1 X3] +
        # 0.125j [X0 Y1 Z2 X3] +
        # 0.125 [X0 Y1 Y3] +
        # -0.125 [Y0 X1 Z2 Y3] +
        # (-0-0.125j) [Y0 X1 X3] +
        # 0.125 [Y0 Y1 Z2 X3] +
        # (-0-0.125j) [Y0 Y1 Y3]
        (
            [(-0 - 0.125j), 0.125, 0.125j, 0.125, -0.125, -0.125j, 0.125, -0.125j],
            [
                qml.PauliX(0) @ qml.PauliX(1) @ qml.PauliZ(2) @ qml.PauliY(3),
                qml.PauliX(0) @ qml.PauliX(1) @ qml.PauliX(3),
                qml.PauliX(0) @ qml.PauliY(1) @ qml.PauliZ(2) @ qml.PauliX(3),
                qml.PauliX(0) @ qml.PauliY(1) @ qml.PauliY(3),
                qml.PauliY(0) @ qml.PauliX(1) @ qml.PauliZ(2) @ qml.PauliY(3),
                qml.PauliY(0) @ qml.PauliX(1) @ qml.PauliX(3),
                qml.PauliY(0) @ qml.PauliY(1) @ qml.PauliZ(2) @ qml.PauliX(3),
                qml.PauliY(0) @ qml.PauliY(1) @ qml.PauliY(3),
            ],
        ),
    ),
    (
        FermiWord({(0, 0): "-", (1, 3): "+"}),
        4,
        # obtained with openfermion using: binary_code_transform(FermionOperator('0 3^'), parity_code(n_qubits)) for n_qubits = 4
        # reformatted the original openfermion output
        # 0.25 [X0 X1 X2 Z3] +
        # (-0-0.25j) [X0 X1 Y2] +
        # 0.25j [Y0 X1 X2 Z3] +
        # 0.25 [Y0 X1 Y2]
        (
            [0.25, -0.25j, 0.25j, 0.25],
            [
                qml.PauliX(0) @ qml.PauliX(1) @ qml.PauliX(2) @ qml.PauliZ(3),
                qml.PauliX(0) @ qml.PauliX(1) @ qml.PauliY(2),
                qml.PauliY(0) @ qml.PauliX(1) @ qml.PauliX(2) @ qml.PauliZ(3),
                qml.PauliY(0) @ qml.PauliX(1) @ qml.PauliY(2),
            ],
        ),
    ),
    (
        FermiWord({(0, 3): "+", (1, 1): "+", (2, 3): "-", (3, 1): "-"}),
        6,
        # obtained with openfermion using: binary_code_transform(FermionOperator('3^ 1^ 3 1'), parity_code(n_qubits)) with 6 qubits
        # -0.25 [] +
        # 0.25 [Z0 Z1] +
        # -0.25 [Z0 Z1 Z2 Z3] +
        # 0.25 [Z2 Z3]
        # reformatted the original openfermion output
        (
            [(-0.25 + 0j), (0.25 + 0j), (-0.25 + 0j), (0.25 + 0j)],
            [
                qml.Identity(0),
                qml.PauliZ(0) @ qml.PauliZ(1),
                qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliZ(2) @ qml.PauliZ(3),
                qml.PauliZ(2) @ qml.PauliZ(3),
            ],
        ),
    ),
    (
        FermiWord({(0, 1): "+", (1, 4): "-", (2, 3): "-", (3, 4): "+"}),
        6,
        # obtained with openfermion using: binary_code_transform(FermionOperator('1^ 4 3 4^'), parity_code(n_qubits)) with 6 qubits
        # reformatted the original openfermion output
        # 0.125 [Z0 X1 X2 Z3] +
        # 0.125 [Z0 X1 X2 Z4] +
        # 0.125j [Z0 X1 Y2] +
        # 0.125j [Z0 X1 Y2 Z3 Z4] +
        # (-0-0.125j) [Y1 X2 Z3] +
        # (-0-0.125j) [Y1 X2 Z4] +
        # 0.125 [Y1 Y2] +
        # 0.125 [Y1 Y2 Z3 Z4]
        (
            [0.125, 0.125, 0.125j, 0.125j, -0.125j, -0.125j, 0.125, 0.125],
            [
                qml.PauliZ(0) @ qml.PauliX(1) @ qml.PauliX(2) @ qml.PauliZ(3),
                qml.PauliZ(0) @ qml.PauliX(1) @ qml.PauliX(2) @ qml.PauliZ(4),
                qml.PauliZ(0) @ qml.PauliX(1) @ qml.PauliY(2),
                qml.PauliZ(0) @ qml.PauliX(1) @ qml.PauliY(2) @ qml.PauliZ(3) @ qml.PauliZ(4),
                qml.PauliY(1) @ qml.PauliX(2) @ qml.PauliZ(3),
                qml.PauliY(1) @ qml.PauliX(2) @ qml.PauliZ(4),
                qml.PauliY(1) @ qml.PauliY(2),
                qml.PauliY(1) @ qml.PauliY(2) @ qml.PauliZ(3) @ qml.PauliZ(4),
            ],
        ),
    ),
    (
        FermiWord({(0, 3): "+", (1, 1): "-", (2, 3): "+", (3, 1): "-"}),
        6,
        # obtained with openfermion using: binary_code_transform(FermionOperator('3^ 1 3^ 1'), parity_code(n_qubits)) with 6 qubits
        ([0], [qml.Identity(3)]),
    ),
    (
        FermiWord({(0, 1): "+", (1, 0): "+", (2, 4): "-", (3, 5): "-"}),
        6,
        # obtained with openfermion using: binary_code_transform(FermionOperator('1^ 0^ 4 5^'), parity_code(n_qubits)) with 6 qubits
        # 0.0625 [X0 Z1 Z3 X4 Z5] +
        # 0.0625j [X0 Z1 Z3 Y4] +
        # 0.0625 [X0 Z1 X4] +
        # 0.0625j [X0 Z1 Y4 Z5] +
        # 0.0625 [X0 Z3 X4 Z5] +
        # 0.0625j [X0 Z3 Y4] +
        # 0.0625 [X0 X4] +
        # 0.0625j [X0 Y4 Z5] +
        # (-0-0.0625j) [Y0 Z1 Z3 X4 Z5] +
        # 0.0625 [Y0 Z1 Z3 Y4] +
        # (-0-0.0625j) [Y0 Z1 X4] +
        # 0.0625 [Y0 Z1 Y4 Z5] +
        # (-0-0.0625j) [Y0 Z3 X4 Z5] +
        # 0.0625 [Y0 Z3 Y4] +
        # (-0-0.0625j) [Y0 X4] +
        # 0.0625 [Y0 Y4 Z5])
        (
            [
                0.0625,
                0.0625j,
                0.0625,
                0.0625j,
                0.0625,
                0.0625j,
                0.0625,
                0.0625j,
                -0.0625j,
                0.0625,
                -0.0625j,
                0.0625,
                -0.0625j,
                0.0625,
                -0.0625j,
                0.0625,
            ],
            [
                qml.PauliX(0) @ qml.PauliZ(1) @ qml.PauliZ(3) @ qml.PauliX(4) @ qml.PauliZ(5),
                qml.PauliX(0) @ qml.PauliZ(1) @ qml.PauliZ(3) @ qml.PauliY(4),
                qml.PauliX(0) @ qml.PauliZ(1) @ qml.PauliX(4),
                qml.PauliX(0) @ qml.PauliZ(1) @ qml.PauliY(4) @ qml.PauliZ(5),
                qml.PauliX(0) @ qml.PauliZ(3) @ qml.PauliX(4) @ qml.PauliZ(5),
                qml.PauliX(0) @ qml.PauliZ(3) @ qml.PauliY(4),
                qml.PauliX(0) @ qml.PauliX(4),
                qml.PauliX(0) @ qml.PauliY(4) @ qml.PauliZ(5),
                qml.PauliY(0) @ qml.PauliZ(1) @ qml.PauliZ(3) @ qml.PauliX(4) @ qml.PauliZ(5),
                qml.PauliY(0) @ qml.PauliZ(1) @ qml.PauliZ(3) @ qml.PauliY(4),
                qml.PauliY(0) @ qml.PauliZ(1) @ qml.PauliX(4),
                qml.PauliY(0) @ qml.PauliZ(1) @ qml.PauliY(4) @ qml.PauliZ(5),
                qml.PauliY(0) @ qml.PauliZ(3) @ qml.PauliX(4) @ qml.PauliZ(5),
                qml.PauliY(0) @ qml.PauliZ(3) @ qml.PauliY(4),
                qml.PauliY(0) @ qml.PauliX(4),
                qml.PauliY(0) @ qml.PauliY(4) @ qml.PauliZ(5),
            ],
        ),
    ),
    (
        FermiWord({(0, 1): "-", (1, 0): "+"}),
        4,
        # obtained with openfermion using: binary_code_transform(FermionOperator('1 0^'), parity_code(n_qubits)) for n_qubits = 4
        # reformatted the original openfermion output:
        # -0.25 [X0] +
        # 0.25 [X0 Z1] +
        # 0.25j [Y0] +
        # (-0-0.25j) [Y0 Z1]
        (
            [(-0.25 + 0j), 0.25, 0.25j, (-0 - 0.25j)],
            [
                qml.PauliX(0),
                qml.PauliX(0) @ qml.PauliZ(1),
                qml.PauliY(0),
                qml.PauliY(0) @ qml.PauliZ(1),
            ],
        ),
    ),
    (
        FermiWord({(0, 1): "+", (1, 1): "-", (2, 3): "+", (3, 4): "-", (4, 2): "+", (5, 5): "-"}),
        6,
        # obtained with openfermion using: binary_code_transform(FermionOperator('1^ 1 3^ 4 2^ 5'), parity_code(n_qubits)) with 6 qubits
        # 0.03125 [Z0 Z1 X2 Z3 X4 Z5] +
        # 0.03125j [Z0 Z1 X2 Z3 Y4] +
        # 0.03125 [Z0 Z1 X2 X4] +
        # 0.03125j [Z0 Z1 X2 Y4 Z5] +
        # (-0-0.03125j) [Z0 Z1 Y2 Z3 X4] +
        # 0.03125 [Z0 Z1 Y2 Z3 Y4 Z5] +
        # (-0-0.03125j) [Z0 Z1 Y2 X4 Z5] +
        # 0.03125 [Z0 Z1 Y2 Y4] +
        # 0.03125 [Z0 X2 Z3 X4] +
        # 0.03125j [Z0 X2 Z3 Y4 Z5] +
        # 0.03125 [Z0 X2 X4 Z5] +
        # 0.03125j [Z0 X2 Y4] +
        # (-0-0.03125j) [Z0 Y2 Z3 X4 Z5] +
        # 0.03125 [Z0 Y2 Z3 Y4] +
        # (-0-0.03125j) [Z0 Y2 X4] +
        # 0.03125 [Z0 Y2 Y4 Z5] +
        # -0.03125 [Z1 X2 Z3 X4] +
        # (-0-0.03125j) [Z1 X2 Z3 Y4 Z5] +
        # -0.03125 [Z1 X2 X4 Z5] +
        # (-0-0.03125j) [Z1 X2 Y4] +
        # 0.03125j [Z1 Y2 Z3 X4 Z5] +
        # -0.03125 [Z1 Y2 Z3 Y4] +
        # 0.03125j [Z1 Y2 X4] +
        # -0.03125 [Z1 Y2 Y4 Z5] +
        # -0.03125 [X2 Z3 X4 Z5] +
        # (-0-0.03125j) [X2 Z3 Y4] +
        # -0.03125 [X2 X4] +
        # (-0-0.03125j) [X2 Y4 Z5] +
        # 0.03125j [Y2 Z3 X4] +
        # -0.03125 [Y2 Z3 Y4 Z5] +
        # 0.03125j [Y2 X4 Z5] +
        # -0.03125 [Y2 Y4]
        (
            [
                0.03125,
                0.03125j,
                0.03125,
                0.03125j,
                -0.03125j,
                0.03125,
                -0.03125j,
                0.03125,
                0.03125,
                0.03125j,
                0.03125,
                0.03125j,
                -0.03125j,
                0.03125,
                -0.03125j,
                0.03125,
                -0.03125,
                -0.03125j,
                -0.03125,
                -0.03125j,
                0.03125j,
                -0.03125,
                0.03125j,
                -0.03125,
                -0.03125,
                -0.03125j,
                -0.03125,
                -0.03125j,
                0.03125j,
                -0.03125,
                0.03125j,
                -0.03125,
            ],
            [
                qml.PauliZ(0)
                @ qml.PauliZ(1)
                @ qml.PauliX(2)
                @ qml.PauliZ(3)
                @ qml.PauliX(4)
                @ qml.PauliZ(5),
                qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliX(2) @ qml.PauliZ(3) @ qml.PauliY(4),
                qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliX(2) @ qml.PauliX(4),
                qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliX(2) @ qml.PauliY(4) @ qml.PauliZ(5),
                qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliY(2) @ qml.PauliZ(3) @ qml.PauliX(4),
                qml.PauliZ(0)
                @ qml.PauliZ(1)
                @ qml.PauliY(2)
                @ qml.PauliZ(3)
                @ qml.PauliY(4)
                @ qml.PauliZ(5),
                qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliY(2) @ qml.PauliX(4) @ qml.PauliZ(5),
                qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliY(2) @ qml.PauliY(4),
                qml.PauliZ(0) @ qml.PauliX(2) @ qml.PauliZ(3) @ qml.PauliX(4),
                qml.PauliZ(0) @ qml.PauliX(2) @ qml.PauliZ(3) @ qml.PauliY(4) @ qml.PauliZ(5),
                qml.PauliZ(0) @ qml.PauliX(2) @ qml.PauliX(4) @ qml.PauliZ(5),
                qml.PauliZ(0) @ qml.PauliX(2) @ qml.PauliY(4),
                qml.PauliZ(0) @ qml.PauliY(2) @ qml.PauliZ(3) @ qml.PauliX(4) @ qml.PauliZ(5),
                qml.PauliZ(0) @ qml.PauliY(2) @ qml.PauliZ(3) @ qml.PauliY(4),
                qml.PauliZ(0) @ qml.PauliY(2) @ qml.PauliX(4),
                qml.PauliZ(0) @ qml.PauliY(2) @ qml.PauliY(4) @ qml.PauliZ(5),
                qml.PauliZ(1) @ qml.PauliX(2) @ qml.PauliZ(3) @ qml.PauliX(4),
                qml.PauliZ(1) @ qml.PauliX(2) @ qml.PauliZ(3) @ qml.PauliY(4) @ qml.PauliZ(5),
                qml.PauliZ(1) @ qml.PauliX(2) @ qml.PauliX(4) @ qml.PauliZ(5),
                qml.PauliZ(1) @ qml.PauliX(2) @ qml.PauliY(4),
                qml.PauliZ(1) @ qml.PauliY(2) @ qml.PauliZ(3) @ qml.PauliX(4) @ qml.PauliZ(5),
                qml.PauliZ(1) @ qml.PauliY(2) @ qml.PauliZ(3) @ qml.PauliY(4),
                qml.PauliZ(1) @ qml.PauliY(2) @ qml.PauliX(4),
                qml.PauliZ(1) @ qml.PauliY(2) @ qml.PauliY(4) @ qml.PauliZ(5),
                qml.PauliX(2) @ qml.PauliZ(3) @ qml.PauliX(4) @ qml.PauliZ(5),
                qml.PauliX(2) @ qml.PauliZ(3) @ qml.PauliY(4),
                qml.PauliX(2) @ qml.PauliX(4),
                qml.PauliX(2) @ qml.PauliY(4) @ qml.PauliZ(5),
                qml.PauliY(2) @ qml.PauliZ(3) @ qml.PauliX(4),
                qml.PauliY(2) @ qml.PauliZ(3) @ qml.PauliY(4) @ qml.PauliZ(5),
                qml.PauliY(2) @ qml.PauliX(4) @ qml.PauliZ(5),
                qml.PauliY(2) @ qml.PauliY(4),
            ],
        ),
    ),
]


@pytest.mark.parametrize("fermionic_op, n_qubits, result", FERMI_WORDS_AND_OPS + FERMI_OPS_COMPLEX)
def test_parity_transform_fermi_word_ps(fermionic_op, n_qubits, result):
    """Test that the parity_transform function returns the correct qubit operator."""
    # convert FermiWord to PauliSentence and simplify
    qubit_op = parity_transform(fermionic_op, n_qubits, ps=True)
    qubit_op.simplify()

    # get expected op as PauliSentence and simplify
    expected_op = pauli_sentence(qml.Hamiltonian(result[0], result[1]))
    expected_op.simplify()

    assert qubit_op == expected_op


@pytest.mark.parametrize("fermionic_op, n_qubits, result", FERMI_WORDS_AND_OPS)
def test_parity_transform_fermi_word_operation(fermionic_op, n_qubits, result):
    wires = fermionic_op.wires or [0]

    qubit_op = parity_transform(fermionic_op, n_qubits)

    expected_op = pauli_sentence(qml.Hamiltonian(result[0], result[1]))
    expected_op = expected_op.operation(wires)

    qml.assert_equal(qubit_op.simplify(), expected_op.simplify())


def test_parity_transform_for_identity():
    """Test that the parity_transform function returns the correct qubit operator for Identity."""
    qml.assert_equal(parity_transform(FermiWord({}), 2), qml.Identity(0))


def test_parity_transform_for_identity_ps():
    """Test that the parity_transform function returns the correct PauliSentence for Identity when ps=True."""
    assert parity_transform(FermiWord({}), 2, ps=True) == PauliSentence(
        {PauliWord({0: "I"}): 1.0 + 0.0j}
    )


@pytest.mark.parametrize(
    "operator",
    (
        FermiWord({(0, 1): "-", (1, 0): "+", (2, 2): "-", (3, 1): "-"}),  # ('1 0^ 2 1')
        FermiWord({(0, 1): "-", (1, 0): "+", (2, 2): "+", (3, 1): "-"}),  # ('1 0^ 2^ 1')
        FermiWord({(0, 3): "-", (1, 0): "+", (2, 2): "+", (3, 3): "-"}),  # ('3 0^ 2^ 3')
        FermiWord({(0, 3): "-", (1, 2): "+", (2, 2): "+", (3, 3): "-"}),  # ('3 2^ 2^ 3')
    ),
)
def test_parity_transform_for_null_operator_fermi_word_ps(operator):
    """Test that the parity_transform function works when the result is 0"""
    # in PauliSentence return format, returns None
    assert parity_transform(operator, 4, ps=True).simplify() is None

    # in operation return format, '0 * I'
    op = parity_transform(operator, 4).simplify()

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

    ps_op = parity_transform(op, 6, ps=True)
    ps_op.simplify()
    assert ps_op == PauliSentence({})

    op = parity_transform(op, 6).simplify()
    assert isinstance(op, SProd)
    assert isinstance(op.base, Identity)
    assert op.scalar == 0


def test_fermi_sentence_identity():
    """Test that a FermiSentence composed of a single Identity operator
    converts to PauliSentence and operation as expected"""
    op = FermiSentence({fw4: 1})
    ps = PauliSentence({PauliWord({}): 1})

    ps_op = parity_transform(op, 6, ps=True)
    qubit_op = parity_transform(op, 6)

    assert ps_op == ps

    result = ps.operation(wire_order=[0])
    qml.assert_equal(qubit_op.simplify(), result.simplify())


FERMI_AND_PAULI_SENTENCES = [
    (FermiSentence({fw4: 0, fw2: 0}), 4, PauliSentence({})),  # all 0 coeffs FermiSentence to null
    (
        FermiSentence({fw2: 2}),
        4,
        PauliSentence({PauliWord({}): (1 + 0j), PauliWord({0: "Z"}): (-1 + 0j)}),
    ),
    (
        FermiSentence({fw1: 1, fw2: 1}),
        4,
        PauliSentence(
            {
                PauliWord({0: "Y"}): -0.25j,
                PauliWord({0: "X", 1: "Z"}): (-0.25 + 0j),
                PauliWord({0: "X"}): (0.25 + 0j),
                PauliWord({0: "Y", 1: "Z"}): 0.25j,
                PauliWord({}): (0.5 + 0j),
                PauliWord({0: "Z"}): (-0.5 + 0j),
            }
        ),
    ),
    (
        FermiSentence({fw1: 1j, fw2: -2}),
        4,
        PauliSentence(
            {
                PauliWord({0: "Y"}): (0.25 + 0j),
                PauliWord({0: "X", 1: "Z"}): -0.25j,
                PauliWord({0: "X"}): 0.25j,
                PauliWord({0: "Y", 1: "Z"}): (-0.25 + 0j),
                PauliWord({}): (-1 + 0j),
                PauliWord({0: "Z"}): (1 + 0j),
            }
        ),
    ),
    (
        FermiSentence({fw1: -2, fw5: 1j}),
        4,
        PauliSentence(
            {
                PauliWord({0: "X"}): -0.5,
                PauliWord({0: "X", 1: "Z"}): 0.5,
                PauliWord({0: "Y"}): 0.5j,
                PauliWord({0: "Y", 1: "Z"}): -0.5j,
                PauliWord({1: "Z", 2: "X", 3: "Z"}): -0.25j,
                PauliWord({1: "Z", 2: "Y"}): -0.25,
                PauliWord({2: "X"}): 0.25j,
                PauliWord({2: "Y", 3: "Z"}): 0.25,
            }
        ),
    ),
    (
        FermiSentence({fw6: 1, fw2: 2}),
        5,
        PauliSentence(
            {
                PauliWord({0: "I"}): 1.0,
                PauliWord({0: "Z"}): -1.0,
                PauliWord({0: "Z", 1: "X", 2: "X", 3: "X", 4: "Z"}): -0.25,
                PauliWord({0: "Z", 1: "X", 2: "X", 3: "Y"}): -0.25j,
                PauliWord({1: "Y", 2: "X", 3: "X", 4: "Z"}): 0.25j,
                PauliWord({1: "Y", 2: "X", 3: "Y"}): -0.25,
            }
        ),
    ),
    (
        FermiSentence({fw5: 1, fw6: 1}),
        5,
        PauliSentence(
            {
                PauliWord({0: "Z", 1: "X", 2: "X", 3: "X", 4: "Z"}): -0.25,
                PauliWord({0: "Z", 1: "X", 2: "X", 3: "Y"}): -0.25j,
                PauliWord({1: "Y", 2: "X", 3: "X", 4: "Z"}): 0.25j,
                PauliWord({1: "Y", 2: "X", 3: "Y"}): -0.25,
                PauliWord({1: "Z", 2: "X", 3: "Z"}): -0.25,
                PauliWord({1: "Z", 2: "Y"}): 0.25j,
                PauliWord({2: "X"}): 0.25,
                PauliWord({2: "Y", 3: "Z"}): -0.25j,
            }
        ),
    ),
]


@pytest.mark.parametrize("fermionic_op, n_qubits, result", FERMI_AND_PAULI_SENTENCES)
def test_parity_transform_for_fermi_sentence_ps(fermionic_op, n_qubits, result):
    qubit_op = parity_transform(fermionic_op, n_qubits, ps=True)
    qubit_op.simplify()

    assert qubit_op == result


@pytest.mark.parametrize("fermionic_op, n_qubits, result", FERMI_AND_PAULI_SENTENCES)
def test_parity_transform_for_fermi_sentence_operation(fermionic_op, n_qubits, result):
    wires = fermionic_op.wires or [0]

    qubit_op = parity_transform(fermionic_op, n_qubits)
    result = result.operation(wires)

    qml.assert_equal(qubit_op.simplify(), result.simplify())


WIRE_MAP_FOR_FERMI_SENTENCE = [
    (
        None,
        [
            qml.s_prod(-0.25j, qml.PauliY(0)),
            qml.s_prod((0.25j), qml.prod(qml.PauliY(0), qml.PauliZ(1))),
            qml.s_prod((-0.25 + 0j), qml.prod(qml.PauliX(0), qml.PauliZ(1))),
            qml.s_prod(0.25, qml.PauliX(0)),
            qml.s_prod((0.5 + 0j), qml.Identity(0)),
            qml.s_prod((-0.5 + 0j), qml.PauliZ(0)),
        ],
    ),
    (
        {0: 0, 1: 1},
        [
            qml.s_prod(-0.25j, qml.PauliY(0)),
            qml.s_prod((0.25j), qml.prod(qml.PauliY(0), qml.PauliZ(1))),
            qml.s_prod((-0.25 + 0j), qml.prod(qml.PauliX(0), qml.PauliZ(1))),
            qml.s_prod(0.25, qml.PauliX(0)),
            qml.s_prod((0.5 + 0j), qml.Identity(0)),
            qml.s_prod((-0.5 + 0j), qml.PauliZ(0)),
        ],
    ),
    (
        {0: 1, 1: 0},
        [
            qml.s_prod(-0.25j, qml.PauliY(1)),
            qml.s_prod((0.25j), qml.prod(qml.PauliY(1), qml.PauliZ(0))),
            qml.s_prod((-0.25 + 0j), qml.prod(qml.PauliX(1), qml.PauliZ(0))),
            qml.s_prod(0.25, qml.PauliX(1)),
            qml.s_prod((0.5 + 0j), qml.Identity(0)),
            qml.s_prod((-0.5 + 0j), qml.PauliZ(1)),
        ],
    ),
    (
        {0: 3, 1: 2},
        [
            qml.s_prod(-0.25j, qml.PauliY(3)),
            qml.s_prod((0.25j), qml.prod(qml.PauliY(3), qml.PauliZ(2))),
            qml.s_prod((-0.25 + 0j), qml.prod(qml.PauliX(3), qml.PauliZ(2))),
            qml.s_prod(0.25, qml.PauliX(3)),
            qml.s_prod((0.5 + 0j), qml.Identity(3)),
            qml.s_prod((-0.5 + 0j), qml.PauliZ(3)),
        ],
    ),
    (
        {0: "b", 1: "a"},
        [
            qml.s_prod(-0.25j, qml.PauliY("b")),
            qml.s_prod((0.25j), qml.prod(qml.PauliY("b"), qml.PauliZ("a"))),
            qml.s_prod((-0.25 + 0j), qml.prod(qml.PauliX("b"), qml.PauliZ("a"))),
            qml.s_prod(0.25, qml.PauliX("b")),
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
    n_qubits = 4
    op = parity_transform(fs, n_qubits, wire_map=wire_map)
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
    n_qubits = 4
    op = parity_transform(fs, n_qubits, wire_map=wire_map, ps=True)
    result_op = qml.sum(*ops)
    ps = pauli_sentence(result_op)

    ps.simplify()
    op.simplify()

    assert ps == op


WIRE_MAP_FOR_FERMI_WORDS = [
    (
        None,
        [
            qml.s_prod(-0.25j, qml.PauliY(0)),
            qml.s_prod(0.25 + 0j, qml.prod(qml.PauliX(0), qml.PauliZ(1))),
            qml.s_prod(0.25 + 0j, qml.PauliX(0)),
            qml.s_prod(-0.25j, qml.prod(qml.PauliY(0), qml.PauliZ(1))),
        ],
    ),
    (
        {0: 3, 1: 2},
        [
            qml.s_prod(-0.25j, qml.PauliY(3)),
            qml.s_prod(0.25 + 0j, qml.prod(qml.PauliX(3), qml.PauliZ(2))),
            qml.s_prod(0.25 + 0j, qml.PauliX(3)),
            qml.s_prod(-0.25j, qml.prod(qml.PauliY(3), qml.PauliZ(2))),
        ],
    ),
    (
        {0: "b", 1: "a"},
        [
            qml.s_prod(-0.25j, qml.PauliY("b")),
            qml.s_prod(0.25 + 0j, qml.prod(qml.PauliX("b"), qml.PauliZ("a"))),
            qml.s_prod(0.25 + 0j, qml.PauliX("b")),
            qml.s_prod(-0.25j, qml.prod(qml.PauliY("b"), qml.PauliZ("a"))),
        ],
    ),
]


@pytest.mark.parametrize("wire_map, ops", WIRE_MAP_FOR_FERMI_WORDS)
def test_providing_wire_map_fermi_word_to_operation(wire_map, ops):
    w = FermiWord({(0, 0): "+", (1, 1): "+"})
    n_qubits = 4
    op = parity_transform(w, n_qubits, wire_map=wire_map)
    result = qml.sum(*ops)

    op.simplify()

    # converting to Pauli representation for comparison because
    # qml.equal isn't playing nicely with term ordering
    assert pauli_sentence(op) == pauli_sentence(result)


@pytest.mark.parametrize("wire_map, ops", WIRE_MAP_FOR_FERMI_WORDS)
def test_providing_wire_map_fermi_word_to_ps(wire_map, ops):
    w = FermiWord({(0, 0): "+", (1, 1): "+"})
    n_qubits = 4
    op = parity_transform(w, n_qubits, wire_map=wire_map, ps=True)
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
        (fw1, (0, -0.25, 0.25, 0), 0.3),
        (fs1, (-0.25j, (0.25 + 0j), (0.25 + 0j), 0.25j), None),
        (fs1, (-0.25j, 0.25, 0.25, 0.25j), 0.0),
        (fs1, (-0.25j, 0.25, 0.25, 0.25j), 1.0e-12),
        (fs1, (0, -0.25, 0.25, 0), 0.3),
    ),
)
def test_parity_transform_tolerance(fermi_op, qubit_op_data, tol):
    """Test that parity_transform properly removes negligible imaginary components"""
    n_qubits = 4
    op = parity_transform(fermi_op, n_qubits, tol=tol)
    assert isinstance(op.data[1], type(qubit_op_data[1]))
