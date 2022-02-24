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
Unit tests for functions needed for computing the Hamiltonian.
"""
import pennylane as qml
import pytest
from pennylane import numpy as np
from pennylane.hf.transform import fermionic_operator, jordan_wigner, qubit_operator


@pytest.mark.parametrize(
    ("core_constant", "integral", "f_ref"),
    [
        (
            np.array([2.869]),
            np.array(
                [
                    [0.95622463, 0.7827277, -0.53222294],
                    [0.7827277, 1.42895581, 0.23469918],
                    [-0.53222294, 0.23469918, 0.48381955],
                ]
            ),
            # computed with PL-QChem dipole (format is modified)
            (
                np.array(
                    [
                        2.869,
                        0.956224634652776,
                        0.782727697897828,
                        -0.532222940905614,
                        0.956224634652776,
                        0.782727697897828,
                        -0.532222940905614,
                        0.782727697897828,
                        1.42895581236226,
                        0.234699175620383,
                        0.782727697897828,
                        1.42895581236226,
                        0.234699175620383,
                        -0.532222940905614,
                        0.234699175620383,
                        0.483819552892797,
                        -0.532222940905614,
                        0.234699175620383,
                        0.483819552892797,
                    ]
                ),
                [
                    [],
                    [0, 0],
                    [0, 2],
                    [0, 4],
                    [1, 1],
                    [1, 3],
                    [1, 5],
                    [2, 0],
                    [2, 2],
                    [2, 4],
                    [3, 1],
                    [3, 3],
                    [3, 5],
                    [4, 0],
                    [4, 2],
                    [4, 4],
                    [5, 1],
                    [5, 3],
                    [5, 5],
                ],
            ),
        ),
    ],
)
def test_fermionic_operator(core_constant, integral, f_ref):
    r"""Test that fermionic_operator returns the correct fermionic observable."""
    f = fermionic_operator(core_constant, integral)

    assert np.allclose(f[0], f_ref[0])  # fermionic coefficients
    assert f[1] == f_ref[1]  # fermionic operators


@pytest.mark.parametrize(
    ("f_operator", "q_operator"),
    [
        (
            (np.array([1.0]), [[0, 0]]),
            # obtained with openfermion: jordan_wigner(FermionOperator('0^ 0', 1)) and reformatted
            [[0.5 + 0j, -0.5 + 0j], [qml.Identity(0), qml.PauliZ(0)]],
        ),
        (
            (np.array([1.0, 1.0]), [[0, 0], [0, 0]]),
            # obtained with openfermion: jordan_wigner(FermionOperator('0^ 0', 1)) and reformatted
            [[1.0 + 0j, -1.0 + 0j], [qml.Identity(0), qml.PauliZ(0)]],
        ),
        (
            (np.array([1.0]), [[2, 0, 2, 0]]),
            # obtained with openfermion: jordan_wigner(FermionOperator('0^ 0', 1)) and reformatted
            [
                [-0.25 + 0j, 0.25 + 0j, -0.25 + 0j, 0.25 + 0j],
                [qml.Identity(0), qml.PauliZ(0), qml.PauliZ(0) @ qml.PauliZ(2), qml.PauliZ(2)],
            ],
        ),
        (
            (np.array([1.0, 1.0]), [[2, 0, 2, 0], [2, 0]]),
            # obtained with openfermion: jordan_wigner(FermionOperator('0^ 0', 1)) and reformatted
            [
                [-0.25 + 0j, 0.25 + 0j, -0.25j, 0.25j, 0.25 + 0j, 0.25 + 0j, -0.25 + 0j, 0.25 + 0j],
                [
                    qml.Identity(0),
                    qml.PauliX(0) @ qml.PauliZ(1) @ qml.PauliX(2),
                    qml.PauliX(0) @ qml.PauliZ(1) @ qml.PauliY(2),
                    qml.PauliY(0) @ qml.PauliZ(1) @ qml.PauliX(2),
                    qml.PauliY(0) @ qml.PauliZ(1) @ qml.PauliY(2),
                    qml.PauliZ(0),
                    qml.PauliZ(0) @ qml.PauliZ(2),
                    qml.PauliZ(2),
                ],
            ],
        ),
    ],
)
def test_qubit_operator(f_operator, q_operator):
    r"""Test that qubit_operator returns the correct operator."""
    h = qubit_operator(f_operator)
    h_ref = qml.Hamiltonian(q_operator[0], q_operator[1])

    assert h.compare(h_ref)


@pytest.mark.parametrize(
    ("f_operator", "q_operator"),
    [
        (
            [0, 0],
            # obtained with openfermion using: jordan_wigner(FermionOperator('0^ 0', 1))
            # reformatted the original openfermion output: (0.5+0j) [] + (-0.5+0j) [Z0]
            ([(0.5 + 0j), (-0.5 + 0j)], [[], [(0, "Z")]]),
        ),
    ],
)
def test_jordan_wigner(f_operator, q_operator):
    r"""Test that jordan_wigner returns the correct operator."""
    result = jordan_wigner(f_operator)

    assert result == q_operator
