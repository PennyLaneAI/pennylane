import sys

import pytest

from pennylane import Identity, PauliX, PauliY, PauliZ
from pennylane import numpy as np
from pennylane import qchem
from pennylane.ops import Hamiltonian

# TODO: Bring pytest skip to relevant tests.
openfermion = pytest.importorskip("openfermion")
openfermionpyscf = pytest.importorskip("openfermionpyscf")

symbols = ["C", "C", "N", "H", "H", "H", "H", "H"]
coordinates = np.array(
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


@pytest.mark.parametrize(
    (
        "charge",
        "mult",
        "package",
        "nact_els",
        "nact_orbs",
        "mapping",
        "grouping_type",
    ),
    [
        (0, 1, "pyscf", 2, 2, "jordan_WIGNER", None),
        (1, 2, "pyscf", 3, 4, "BRAVYI_kitaev", None),
        (-1, 2, "pyscf", 1, 2, "jordan_WIGNER", "qwc"),
        (2, 1, "pyscf", 2, 2, "BRAVYI_kitaev", "commuting"),
    ],
)
def test_building_hamiltonian(
    charge,
    mult,
    package,
    nact_els,
    nact_orbs,
    mapping,
    grouping_type,
    tmpdir,
):
    r"""Test that the generated Hamiltonian `built_hamiltonian` is an instance of the PennyLane
    Hamiltonian class and the correctness of the total number of qubits required to run the
    quantum simulation. The latter is tested for different values of the molecule's charge and
    for active spaces with different size"""
    built_hamiltonian, qubits = qchem.molecular_hamiltonian(
        symbols,
        coordinates,
        charge=charge,
        mult=mult,
        method=package,
        active_electrons=nact_els,
        active_orbitals=nact_orbs,
        mapping=mapping,
        grouping_type=grouping_type,
        outpath=tmpdir.strpath,
    )

    assert isinstance(built_hamiltonian, Hamiltonian)
    assert qubits == 2 * nact_orbs


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
def test_differentiable_hamiltonian(symbols, geometry, h_ref_data):
    r"""Test that molecular_hamiltonian returns the correct Hamiltonian with the differentiable
    backend."""
    geometry.requires_grad = True
    args = [geometry.reshape(2, 3)]
    h_args = qchem.molecular_hamiltonian(symbols, geometry, method="dhf", args=args)[0]

    geometry.requires_grad = False
    h_noargs = qchem.molecular_hamiltonian(symbols, geometry, method="dhf", grouping_type="qwc")[0]

    h_ref = Hamiltonian(h_ref_data[0], h_ref_data[1])

    assert np.allclose(np.sort(h_args.coeffs), np.sort(h_ref.coeffs))
    assert Hamiltonian(np.ones(len(h_args.coeffs)), h_args.ops).compare(
        Hamiltonian(np.ones(len(h_ref.coeffs)), h_ref.ops)
    )

    assert np.allclose(np.sort(h_noargs.coeffs), np.sort(h_ref.coeffs))
    assert Hamiltonian(np.ones(len(h_noargs.coeffs)), h_noargs.ops).compare(
        Hamiltonian(np.ones(len(h_ref.coeffs)), h_ref.ops)
    )


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
        qchem.molecular_hamiltonian(symbols, geometry, method="dhf", mapping="bravyi_kitaev")[0]

    with pytest.raises(ValueError, match="Only 'dhf' and 'pyscf' backends are supported"):
        qchem.molecular_hamiltonian(symbols, geometry, method="psi4")[0]
