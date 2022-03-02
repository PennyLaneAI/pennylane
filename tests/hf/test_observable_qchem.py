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
from pennylane.hf.observable import (
    _pauli_mult,
    fermionic_observable,
    jordan_wigner,
    qubit_observable,
    simplify,
)


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
def test_fermionic_observable(core_constant, integral, f_ref):
    r"""Test that fermionic_observable returns the correct fermionic observable."""
    f = fermionic_observable(core_constant, integral)

    assert np.allclose(f[0], f_ref[0])  # fermionic coefficients
    assert f[1] == f_ref[1]  # fermionic operators


@pytest.mark.parametrize(
    ("f_observable", "q_observable"),
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
def test_qubit_observable(f_observable, q_observable):
    r"""Test that qubit_observable returns the correct operator."""
    print(f_observable)
    h = qubit_observable(f_observable)
    h_ref = qml.Hamiltonian(q_observable[0], q_observable[1])

    assert h.compare(h_ref)


@pytest.mark.parametrize(
    ("f_obs", "q_obs"),
    [
        (
            [0, 0],
            # obtained with openfermion using: jordan_wigner(FermionOperator('0^ 0', 1))
            # reformatted the original openfermion output: (0.5+0j) [] + (-0.5+0j) [Z0]
            ([(0.5 + 0j), (-0.5 + 0j)], [qml.Identity(0), qml.PauliZ(0)]),
        ),
        (
            [3, 0],
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
            [1, 4],
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
            [3, 1, 3, 1],
            # obtained with openfermion using: jordan_wigner(FermionOperator('3^ 1^ 3 1', 1))
            # reformatted the original openfermion output
            (
                [(-0.25 + 0j), (0.25 + 0j), (-0.25 + 0j), (0.25 + 0j)],
                [qml.Identity(0), qml.PauliZ(1), qml.PauliZ(1) @ qml.PauliZ(3), qml.PauliZ(3)],
            ),
        ),
    ],
)
def test_jordan_wigner(f_obs, q_obs):
    r"""Test that jordan_wigner returns the correct operator."""
    res = jordan_wigner(f_obs)
    assert qml.Hamiltonian(res[0], res[1]).compare(qml.Hamiltonian(q_obs[0], q_obs[1]))


@pytest.mark.parametrize(
    ("f_obs", "q_obs"),
    [
        (
            [1, 1, 1, 1],
            # obtained with openfermion using: jordan_wigner(FermionOperator('1^ 1^ 1 1', 1))
            0.0,
        ),
    ],
)
def test_jordan_wigner_zero_output(f_obs, q_obs):
    r"""Test that jordan_wigner returns the correct operator."""
    res = jordan_wigner(f_obs)
    assert res == q_obs


@pytest.mark.parametrize(
    ("p1", "p2", "p_ref"),
    [
        (
            [(0, "X"), (1, "Y")],  # X_0 @ Y_1
            [(0, "X"), (2, "Y")],  # X_0 @ Y_2
            ([(2, "Y"), (1, "Y")], 1.0),  # X_0 @ Y_1 @ X_0 @ Y_2
        ),
    ],
)
def test_pauli_mult(p1, p2, p_ref):
    r"""Test that _pauli_mult returns the correct operator."""
    result = _pauli_mult(p1, p2)

    assert result == p_ref


@pytest.mark.parametrize(
    ("hamiltonian", "result"),
    [
        (
            qml.Hamiltonian(
                np.array([0.5, 0.5]), [qml.PauliX(0) @ qml.PauliY(1), qml.PauliX(0) @ qml.PauliY(1)]
            ),
            qml.Hamiltonian(np.array([1.0]), [qml.PauliX(0) @ qml.PauliY(1)]),
        ),
        (
            qml.Hamiltonian(
                np.array([0.5, -0.5]),
                [qml.PauliX(0) @ qml.PauliY(1), qml.PauliX(0) @ qml.PauliY(1)],
            ),
            qml.Hamiltonian([], []),
        ),
        (
            qml.Hamiltonian(
                np.array([0.0, -0.5]),
                [qml.PauliX(0) @ qml.PauliY(1), qml.PauliX(0) @ qml.PauliZ(1)],
            ),
            qml.Hamiltonian(np.array([-0.5]), [qml.PauliX(0) @ qml.PauliZ(1)]),
        ),
        (
            qml.Hamiltonian(
                np.array([0.25, 0.25, 0.25, -0.25]),
                [
                    qml.PauliX(0) @ qml.PauliY(1),
                    qml.PauliX(0) @ qml.PauliZ(1),
                    qml.PauliX(0) @ qml.PauliY(1),
                    qml.PauliX(0) @ qml.PauliY(1),
                ],
            ),
            qml.Hamiltonian(
                np.array([0.25, 0.25]),
                [qml.PauliX(0) @ qml.PauliY(1), qml.PauliX(0) @ qml.PauliZ(1)],
            ),
        ),
    ],
)
def test_simplify(hamiltonian, result):
    r"""Test that simplify returns the correct hamiltonian."""
    h = simplify(hamiltonian)
    assert h.compare(result)
