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
import pytest

import pennylane as qml
from pennylane import numpy as np
from pennylane import qchem
from pennylane.fermi import from_string


@pytest.mark.parametrize(
    ("core_constant", "integral_one", "integral_two", "f_ref"),
    [
        (  # computed with openfermion for H2 (format is modified):
            # H2 bond length: 1 Angstrom, basis = 'sto-3g', multiplicity = 1, charge = 0
            # molecule = openfermion.MolecularData(geometry, basis, multiplicity, charge)
            # run_pyscf(molecule).get_integrals()
            np.array([0.529177210903]),  # nuclear repulsion 1 / 1.88973 Bohr
            np.array([[-1.11084418e00, 1.01781501e-16], [7.32122533e-17, -5.89121004e-01]]),
            np.array(
                [
                    [
                        [[6.26402500e-01, -1.84129592e-16], [-2.14279171e-16, 1.96790583e-01]],
                        [[-2.14279171e-16, 1.96790583e-01], [6.21706763e-01, -1.84062159e-17]],
                    ],
                    [
                        [[-1.84129592e-16, 6.21706763e-01], [1.96790583e-01, -5.28427412e-17]],
                        [[1.96790583e-01, -5.28427412e-17], [-1.84062159e-17, 6.53070747e-01]],
                    ],
                ]
            ),
            # computed with openfermion for H2 (format is modified):
            # get_fermion_operator(run_pyscf(molecule).get_molecular_hamiltonian())
            0.52917721092 * from_string("")
            - 1.1108441798837276 * from_string("0+ 0-")
            + 0.31320124976475916 * from_string("0+ 0+ 0- 0-")
            + 0.09839529174273519 * from_string("0+ 0+ 2- 2-")
            + 0.31320124976475916 * from_string("0+ 1+ 1- 0-")
            + 0.09839529174273519 * from_string("0+ 1+ 3- 2-")
            + 0.09839529174273519 * from_string("0+ 2+ 0- 2-")
            + 0.3108533815598568 * from_string("0+ 2+ 2- 0-")
            + 0.09839529174273519 * from_string("0+ 3+ 1- 2-")
            + 0.3108533815598568 * from_string("0+ 3+ 3- 0-")
            + 0.31320124976475916 * from_string("1+ 0+ 0- 1-")
            + 0.09839529174273519 * from_string("1+ 0+ 2- 3-")
            - 1.1108441798837276 * from_string("1+ 1-")
            + 0.31320124976475916 * from_string("1+ 1+ 1- 1-")
            + 0.09839529174273519 * from_string("1+ 1+ 3- 3-")
            + 0.09839529174273519 * from_string("1+ 2+ 0- 3-")
            + 0.3108533815598568 * from_string("1+ 2+ 2- 1-")
            + 0.09839529174273519 * from_string("1+ 3+ 1- 3-")
            + 0.3108533815598568 * from_string("1+ 3+ 3- 1-")
            + 0.3108533815598569 * from_string("2+ 0+ 0- 2-")
            + 0.09839529174273519 * from_string("2+ 0+ 2- 0-")
            + 0.3108533815598569 * from_string("2+ 1+ 1- 2-")
            + 0.09839529174273519 * from_string("2+ 1+ 3- 0-")
            - 0.5891210037060831 * from_string("2+ 2-")
            + 0.09839529174273519 * from_string("2+ 2+ 0- 0-")
            + 0.32653537347128725 * from_string("2+ 2+ 2- 2-")
            + 0.09839529174273519 * from_string("2+ 3+ 1- 0-")
            + 0.32653537347128725 * from_string("2+ 3+ 3- 2-")
            + 0.3108533815598569 * from_string("3+ 0+ 0- 3-")
            + 0.09839529174273519 * from_string("3+ 0+ 2- 1-")
            + 0.3108533815598569 * from_string("3+ 1+ 1- 3-")
            + 0.09839529174273519 * from_string("3+ 1+ 3- 1-")
            + 0.09839529174273519 * from_string("3+ 2+ 0- 1-")
            + 0.32653537347128725 * from_string("3+ 2+ 2- 3-")
            - 0.5891210037060831 * from_string("3+ 3-")
            + 0.09839529174273519 * from_string("3+ 3+ 1- 1-")
            + 0.32653537347128725 * from_string("3+ 3+ 3- 3-"),
        ),
        (
            np.array([2.869]),
            np.array(
                [
                    [0.95622463, 0.7827277, -0.53222294],
                    [0.7827277, 1.42895581, 0.23469918],
                    [-0.53222294, 0.23469918, 0.48381955],
                ]
            ),
            None,
            # computed with PL-QChem dipole (format is modified)
            2.869 * from_string("")
            + 0.956224634652776 * from_string("0+ 0-")
            + 0.782727697897828 * from_string("0+ 2-")
            - 0.532222940905614 * from_string("0+ 4-")
            + 0.956224634652776 * from_string("1+ 1-")
            + 0.782727697897828 * from_string("1+ 3-")
            - 0.532222940905614 * from_string("1+ 5-")
            + 0.782727697897828 * from_string("2+ 0-")
            + 1.42895581236226 * from_string("2+ 2-")
            + 0.234699175620383 * from_string("2+ 4-")
            + 0.782727697897828 * from_string("3+ 1-")
            + 1.42895581236226 * from_string("3+ 3-")
            + 0.234699175620383 * from_string("3+ 5-")
            - 0.532222940905614 * from_string("4+ 0-")
            + 0.234699175620383 * from_string("4+ 2-")
            + 0.483819552892797 * from_string("4+ 4-")
            - 0.532222940905614 * from_string("5+ 1-")
            + 0.234699175620383 * from_string("5+ 3-")
            + 0.483819552892797 * from_string("5+ 5-"),
        ),
    ],
)
def test_fermionic_observable(core_constant, integral_one, integral_two, f_ref):
    r"""Test that fermionic_observable returns the correct fermionic observable."""
    f = qchem.fermionic_observable(core_constant, integral_one, integral_two)

    assert np.allclose(list(f.values()), list(f_ref.values()))
    assert f.keys() == f_ref.keys()


@pytest.mark.parametrize(
    ("f_observable", "q_observable"),
    [
        (
            from_string("0+ 0-"),
            # obtained with openfermion: jordan_wigner(FermionOperator('0^ 0', 1)) and reformatted
            [[0.5 + 0j, -0.5 + 0j], [qml.Identity(0), qml.PauliZ(0)]],
        ),
        (
            from_string("0+ 0-") + from_string("0+ 0-"),
            # obtained with openfermion: jordan_wigner(FermionOperator('0^ 0', 1)) and reformatted
            [[1.0 + 0j, -1.0 + 0j], [qml.Identity(0), qml.PauliZ(0)]],
        ),
        (
            from_string("2+ 0+ 2- 0-"),
            # obtained with openfermion: jordan_wigner(FermionOperator('2^ 0^ 2 0', 1)) and reformatted
            [
                [-0.25 + 0j, 0.25 + 0j, -0.25 + 0j, 0.25 + 0j],
                [qml.Identity(0), qml.PauliZ(0), qml.PauliZ(0) @ qml.PauliZ(2), qml.PauliZ(2)],
            ],
        ),
        (
            from_string("2+ 0+ 2- 0-") + from_string("2+ 0-"),
            # obtained with openfermion: jordan_wigner(FermionOperator('2^ 0^ 2 0', 1)) and
            # jordan_wigner(FermionOperator('2^ 0', 1)) and reformatted
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
        (1.23 * from_string(""), [[1.23], [qml.Identity(0)]]),
    ],
)
def test_qubit_observable(f_observable, q_observable):
    r"""Test that qubit_observable returns the correct operator."""
    h_as_op = qchem.qubit_observable(f_observable)
    ops = list(map(qml.simplify, q_observable[1]))
    h_ref = qml.dot(q_observable[0], ops)
    qml.assert_equal(h_ref, h_as_op)
    assert np.allclose(
        qml.matrix(h_as_op, wire_order=[0, 1, 2]), qml.matrix(h_ref, wire_order=[0, 1, 2])
    )


@pytest.mark.parametrize(
    ("f_observable", "cut_off"),
    [
        (
            0.01 * from_string("0+ 0-"),
            0.1,
        ),
        (
            from_string("0+ 0+ 1- 1-"),  # should produce the 0 operator
            0.1,
        ),
    ],
)
def test_qubit_observable_cutoff(f_observable, cut_off):
    """Test that qubit_observable returns the correct operator when a cutoff is provided."""
    h_ref, h_ref_op = 0 * qml.I(0), qml.s_prod(0, qml.Identity(0))
    h_as_op = qchem.qubit_observable(f_observable, cutoff=cut_off)

    qml.assert_equal(h_ref, h_as_op)
    assert np.allclose(
        qml.matrix(h_ref_op, wire_order=[0, 1, 2]), qml.matrix(h_as_op, wire_order=[0, 1, 2])
    )


def test_qubit_observable_error():
    """Test that qubit_observable raises an error for unsupported mapping."""

    f_observable = 0.01 * from_string("0+ 0-")

    with pytest.raises(ValueError, match="transformation is not available"):
        qchem.qubit_observable(f_observable, mapping="random")
