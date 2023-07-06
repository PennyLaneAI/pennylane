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
"""
Unit tests for molecular Hamiltonians.
"""
# pylint: disable=too-many-arguments
import pytest

from pennylane import Identity, PauliX, PauliY, PauliZ
from pennylane import numpy as np
from pennylane import qchem
from pennylane.ops import Hamiltonian
from pennylane.ops.functions import dot
from pennylane.pauli import pauli_sentence
from pennylane.operation import enable_new_opmath, disable_new_opmath

test_symbols = ["C", "C", "N", "H", "H", "H", "H", "H"]
test_coordinates = np.array(
    [
        0.68219113,
        -0.85415621,
        -1.04123909,
        -1.34926445,
        0.23621577,
        0.61794044,
        1.29068294,
        0.25133357,
        1.40784596,
        0.83525895,
        -2.88939124,
        -1.16974047,
        1.26989596,
        0.19275206,
        -2.69852891,
        -2.57758643,
        -1.05824663,
        1.61949529,
        -2.17129532,
        2.04090421,
        0.11338357,
        2.06547065,
        2.00877887,
        1.20186581,
    ]
)


@pytest.mark.parametrize("op_arithmetic", [False, True])
@pytest.mark.parametrize(
    (
        "charge",
        "mult",
        "package",
        "nact_els",
        "nact_orbs",
        "mapping",
    ),
    [
        (0, 1, "pyscf", 2, 2, "jordan_WIGNER"),
        (1, 2, "pyscf", 3, 4, "BRAVYI_kitaev"),
        (-1, 2, "pyscf", 1, 2, "jordan_WIGNER"),
        (2, 1, "pyscf", 2, 2, "BRAVYI_kitaev"),
    ],
)
@pytest.mark.usefixtures("skip_if_no_openfermion_support")
def test_building_hamiltonian(
    charge,
    mult,
    package,
    nact_els,
    nact_orbs,
    mapping,
    tmpdir,
    op_arithmetic,
):
    r"""Test that the generated Hamiltonian `built_hamiltonian` is an instance of the PennyLane
    Hamiltonian class and the correctness of the total number of qubits required to run the
    quantum simulation. The latter is tested for different values of the molecule's charge and
    for active spaces with different size"""

    args = (test_symbols, test_coordinates)
    kwargs = {
        "charge": charge,
        "mult": mult,
        "method": package,
        "active_electrons": nact_els,
        "active_orbitals": nact_orbs,
        "mapping": mapping,
        "outpath": tmpdir.strpath,
    }
    if op_arithmetic:
        enable_new_opmath()

    built_hamiltonian, qubits = qchem.molecular_hamiltonian(*args, **kwargs)

    if op_arithmetic:
        disable_new_opmath()
        assert not isinstance(built_hamiltonian, Hamiltonian)
    else:
        assert isinstance(built_hamiltonian, Hamiltonian)
    assert qubits == 2 * nact_orbs


@pytest.mark.parametrize("op_arithmetic", [False, True])
@pytest.mark.parametrize(
    ("symbols", "geometry", "h_ref_data"),
    [
        (
            ["H", "H"],
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            # computed with OpenFermion; data reordered
            # h_mol = molecule.get_molecular_hamiltonian()
            # h_f = openfermion.transforms.get_fermion_operator(h_mol)
            # h_q = openfermion.transforms.jordan_wigner(h_f)
            # h_pl = qchem.convert_observable(h_q, wires=[0, 1, 2, 3], tol=(5e-5))
            (
                np.array(
                    [
                        0.2981788017,
                        0.2081336485,
                        0.2081336485,
                        0.1786097698,
                        0.042560361,
                        -0.042560361,
                        -0.042560361,
                        0.042560361,
                        -0.3472487379,
                        0.1329029281,
                        -0.3472487379,
                        0.175463289,
                        0.175463289,
                        0.1329029281,
                        0.1847091733,
                    ]
                ),
                [
                    Identity(wires=[0]),
                    PauliZ(wires=[0]),
                    PauliZ(wires=[1]),
                    PauliZ(wires=[0]) @ PauliZ(wires=[1]),
                    PauliY(wires=[0]) @ PauliX(wires=[1]) @ PauliX(wires=[2]) @ PauliY(wires=[3]),
                    PauliY(wires=[0]) @ PauliY(wires=[1]) @ PauliX(wires=[2]) @ PauliX(wires=[3]),
                    PauliX(wires=[0]) @ PauliX(wires=[1]) @ PauliY(wires=[2]) @ PauliY(wires=[3]),
                    PauliX(wires=[0]) @ PauliY(wires=[1]) @ PauliY(wires=[2]) @ PauliX(wires=[3]),
                    PauliZ(wires=[2]),
                    PauliZ(wires=[0]) @ PauliZ(wires=[2]),
                    PauliZ(wires=[3]),
                    PauliZ(wires=[0]) @ PauliZ(wires=[3]),
                    PauliZ(wires=[1]) @ PauliZ(wires=[2]),
                    PauliZ(wires=[1]) @ PauliZ(wires=[3]),
                    PauliZ(wires=[2]) @ PauliZ(wires=[3]),
                ],
            ),
        ),
        (
            ["H", "H"],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
            # computed with OpenFermion; data reordered
            # h_mol = molecule.get_molecular_hamiltonian()
            # h_f = openfermion.transforms.get_fermion_operator(h_mol)
            # h_q = openfermion.transforms.jordan_wigner(h_f)
            # h_pl = qchem.convert_observable(h_q, wires=[0, 1, 2, 3], tol=(5e-5))
            (
                np.array(
                    [
                        0.2981788017,
                        0.2081336485,
                        0.2081336485,
                        0.1786097698,
                        0.042560361,
                        -0.042560361,
                        -0.042560361,
                        0.042560361,
                        -0.3472487379,
                        0.1329029281,
                        -0.3472487379,
                        0.175463289,
                        0.175463289,
                        0.1329029281,
                        0.1847091733,
                    ]
                ),
                [
                    Identity(wires=[0]),
                    PauliZ(wires=[0]),
                    PauliZ(wires=[1]),
                    PauliZ(wires=[0]) @ PauliZ(wires=[1]),
                    PauliY(wires=[0]) @ PauliX(wires=[1]) @ PauliX(wires=[2]) @ PauliY(wires=[3]),
                    PauliY(wires=[0]) @ PauliY(wires=[1]) @ PauliX(wires=[2]) @ PauliX(wires=[3]),
                    PauliX(wires=[0]) @ PauliX(wires=[1]) @ PauliY(wires=[2]) @ PauliY(wires=[3]),
                    PauliX(wires=[0]) @ PauliY(wires=[1]) @ PauliY(wires=[2]) @ PauliX(wires=[3]),
                    PauliZ(wires=[2]),
                    PauliZ(wires=[0]) @ PauliZ(wires=[2]),
                    PauliZ(wires=[3]),
                    PauliZ(wires=[0]) @ PauliZ(wires=[3]),
                    PauliZ(wires=[1]) @ PauliZ(wires=[2]),
                    PauliZ(wires=[1]) @ PauliZ(wires=[3]),
                    PauliZ(wires=[2]) @ PauliZ(wires=[3]),
                ],
            ),
        ),
    ],
)
def test_differentiable_hamiltonian(symbols, geometry, h_ref_data, op_arithmetic):
    r"""Test that molecular_hamiltonian returns the correct Hamiltonian with the differentiable
    backend."""
    if op_arithmetic:
        enable_new_opmath()

    geometry.requires_grad = True
    args = [geometry.reshape(2, 3)]
    h_args = qchem.molecular_hamiltonian(symbols, geometry, method="dhf", args=args)[0]

    geometry.requires_grad = False
    h_noargs = qchem.molecular_hamiltonian(symbols, geometry, method="dhf")[0]

    h_ref = (
        dot(h_ref_data[0], h_ref_data[1], pauli=True)
        if op_arithmetic
        else Hamiltonian(h_ref_data[0], h_ref_data[1])
    )

    if op_arithmetic:
        disable_new_opmath()
        h_args_ps = pauli_sentence(h_args)
        h_noargs_ps = pauli_sentence(h_noargs)
        h_ref_sorted_coeffs = np.sort(list(h_ref.values()))

        assert set(h_args_ps) == set(h_ref)
        assert set(h_noargs_ps) == set(h_ref)

        assert np.allclose(np.sort(list(h_args_ps.values())), h_ref_sorted_coeffs)
        assert np.allclose(np.sort(list(h_noargs_ps.values())), h_ref_sorted_coeffs)

        assert all(val.requires_grad is True for val in h_args_ps.values())
        assert all(val.requires_grad is False for val in h_noargs_ps.values())

    else:
        assert np.allclose(np.sort(h_args.coeffs), np.sort(h_ref.coeffs))
        assert Hamiltonian(np.ones(len(h_args.coeffs)), h_args.ops).compare(
            Hamiltonian(np.ones(len(h_ref.coeffs)), h_ref.ops)
        )

        assert np.allclose(np.sort(h_noargs.coeffs), np.sort(h_ref.coeffs))
        assert Hamiltonian(np.ones(len(h_noargs.coeffs)), h_noargs.ops).compare(
            Hamiltonian(np.ones(len(h_ref.coeffs)), h_ref.ops)
        )

        assert h_args.coeffs.requires_grad is True
        assert h_noargs.coeffs.requires_grad is False


@pytest.mark.parametrize("op_arithmetic", [False, True])
@pytest.mark.parametrize(
    ("symbols", "geometry", "method", "wiremap"),
    [
        (
            ["H", "H"],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]),
            "pyscf",
            ["a", "b", "c", "d"],
        ),
        (
            ["H", "H"],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]),
            "pyscf",
            [0, "z", 3, "ancilla"],
        ),
    ],
)
@pytest.mark.usefixtures("skip_if_no_openfermion_support")
def test_custom_wiremap_hamiltonian_pyscf(
    symbols, geometry, method, wiremap, tmpdir, op_arithmetic
):
    r"""Test that the generated Hamiltonian has the correct wire labels given by a custom wiremap."""
    if op_arithmetic:
        enable_new_opmath()

    hamiltonian, _ = qchem.molecular_hamiltonian(
        symbols=symbols,
        coordinates=geometry,
        method=method,
        wires=wiremap,
        outpath=tmpdir.strpath,
    )

    assert set(hamiltonian.wires) == set(wiremap)

    if op_arithmetic:
        disable_new_opmath()


@pytest.mark.parametrize("op_arithmetic", [False, True])
@pytest.mark.parametrize(
    ("symbols", "geometry", "wiremap", "args"),
    [
        (
            ["H", "H"],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]),
            [0, "z", 3, "ancilla"],
            None,
        ),
        (
            ["H", "H"],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]),
            [0, "z", 3, "ancilla"],
            [np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]])],
        ),
    ],
)
def test_custom_wiremap_hamiltonian_dhf(symbols, geometry, wiremap, args, tmpdir, op_arithmetic):
    r"""Test that the generated Hamiltonian has the correct wire labels given by a custom wiremap."""
    if op_arithmetic:
        enable_new_opmath()

    hamiltonian, _ = qchem.molecular_hamiltonian(
        symbols=symbols,
        coordinates=geometry,
        wires=wiremap,
        args=args,
        outpath=tmpdir.strpath,
    )

    assert list(hamiltonian.wires) == list(wiremap)

    if op_arithmetic:
        disable_new_opmath()


file_content = """\
2
in Angstrom
H          0.00000        0.00000       -0.35000
H          0.00000        0.00000        0.35000
"""


def test_mol_hamiltonian_with_read_structure(tmpdir):
    """Test that the pipeline of using molecular_hamiltonian with
    read_structure executes without errors."""
    f_name = "h2.xyz"
    filename = tmpdir.join(f_name)

    with open(filename, "w") as f:
        f.write(file_content)

    symbols, coordinates = qchem.read_structure(str(filename), outpath=tmpdir)
    H, num_qubits = qchem.molecular_hamiltonian(symbols, coordinates)
    assert len(H.terms()) == 2
    assert num_qubits == 4


@pytest.mark.parametrize(
    ("symbols", "geometry"),
    [
        (
            ["H", "H"],
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        ),
    ],
)
def test_diff_hamiltonian_error(symbols, geometry):
    r"""Test that molecular_hamiltonian raises an error with unsupported mapping."""

    with pytest.raises(ValueError, match="Only 'jordan_wigner' mapping is supported"):
        qchem.molecular_hamiltonian(symbols, geometry, method="dhf", mapping="bravyi_kitaev")

    with pytest.raises(ValueError, match="Only 'dhf' and 'pyscf' backends are supported"):
        qchem.molecular_hamiltonian(symbols, geometry, method="psi4")

    with pytest.raises(ValueError, match="Openshell systems are not supported"):
        qchem.molecular_hamiltonian(symbols, geometry, mult=3)


@pytest.mark.parametrize("op_arithmetic", [False, True])
@pytest.mark.parametrize(
    ("symbols", "geometry", "method", "args"),
    [
        (
            ["H", "H"],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]),
            "pyscf",
            None,
        ),
        (
            ["H", "H"],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]),
            "dhf",
            None,
        ),
        (
            ["H", "H"],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]),
            "dhf",
            [np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]])],
        ),
    ],
)
@pytest.mark.usefixtures("skip_if_no_openfermion_support")
def test_real_hamiltonian(symbols, geometry, method, args, tmpdir, op_arithmetic):
    r"""Test that the generated Hamiltonian has real coefficients."""
    if op_arithmetic:
        enable_new_opmath()

    hamiltonian, _ = qchem.molecular_hamiltonian(
        symbols=symbols,
        coordinates=geometry,
        method=method,
        args=args,
        outpath=tmpdir.strpath,
    )

    if op_arithmetic:
        disable_new_opmath()
        h_as_ps = pauli_sentence(hamiltonian)
        assert np.isrealobj(np.array(h_as_ps.values()))

    else:
        assert np.isrealobj(hamiltonian.coeffs)
