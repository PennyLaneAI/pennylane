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
from pennylane.pauli.conversion import pauli_sentence
from pennylane.fermi.conversion import jw_mapping
from pennylane.fermi.fermionic import FermiWord


@pytest.mark.parametrize(
    ("fermionic_op", "result"),
    [
        (
            FermiWord({(0, 0): "+"}),
            # trivial case of a creation operator, 0^ -> (X_0 - iY_0) / 2
            ([(0.5 + 0j), (0.0 - 0.5j)], [qml.PauliX(0), qml.PauliY(0)]),
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
    ],
)
def test_jw_mapping(fermionic_op, result):
    r"""Test that the jw_mapping function returns the correct qubit operator."""
    # convert FermiWord to PauliSentence and simplify
    qubit_op = jw_mapping(fermionic_op)
    qubit_op.simplify()

    # get expected op as PauliSentence and simplify
    expected_op = pauli_sentence(qml.Hamiltonian(result[0], result[1]))
    expected_op.simplify()

    assert qubit_op == expected_op
