import os

import pytest

from pennylane import qchem

from pennylane.vqe import Hamiltonian

ref_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_ref_files")


@pytest.mark.parametrize(
    ("charge", "mult", "package", "nact_els", "nact_orbs", "mapping",),
    [
        (0, 1, "psi4", 2, 2, "jordan_WIGNER"),
        (1, 2, "psi4", 3, 4, "BRAVYI_kitaev"),
        (-1, 2, "psi4", 1, 2, "jordan_WIGNER"),
        (2, 1, "psi4", 2, 2, "BRAVYI_kitaev"),
        (0, 1, "pyscf", 2, 2, "BRAVYI_kitaev"),
        (1, 2, "pyscf", 3, 4, "jordan_WIGNER"),
        (-1, 2, "pyscf", 1, 2, "BRAVYI_kitaev"),
        (2, 1, "pyscf", 2, 2, "jordan_WIGNER"),
    ],
)
def test_building_hamiltonian(
    charge, mult, package, nact_els, nact_orbs, mapping, psi4_support, requires_babel, tmpdir,
):
    r"""Test that the generated Hamiltonian `built_hamiltonian` is an instance of the PennyLane
    Hamiltonian class and the correctness of the total number of qubits required to run the
    quantum simulation. The latter is tested for different values of the molecule's charge and
    for active spaces with different size"""

    name = "gdb3"
    geo_file = "gdb3.mol5.PDB"

    if package == "psi4" and not psi4_support:
        pytest.skip("Skipped, no Psi4 support")

    geo_file = os.path.join(ref_dir, geo_file)

    built_hamiltonian, qubits = qchem.molecular_hamiltonian(
        name,
        geo_file,
        charge=charge,
        mult=mult,
        package=package,
        active_electrons=nact_els,
        active_orbitals=nact_orbs,
        mapping=mapping,
        outpath=tmpdir.strpath,
    )

    assert isinstance(built_hamiltonian, Hamiltonian)
    assert qubits == 2 * nact_orbs
