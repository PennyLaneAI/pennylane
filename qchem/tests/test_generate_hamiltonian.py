import os

import pytest

from pennylane import qchem

from pennylane.vqe import Hamiltonian

ref_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_ref_files")


@pytest.mark.parametrize(
    (
        "mol_name",
        "geo_file",
        "charge",
        "multiplicity",
        "qc_prog",
        "n_act_electrons",
        "n_act_orbitals",
        "transformation",
    ),
    [
        ("gdb3", "gdb3.mol5.PDB", 0, 1, "psi4", 2, 2, "jordan_WIGNER"),
        ("gdb3", "gdb3.mol5.PDB", 0, 1, "psi4", 2, 2, "BRAVYI_kitaev"),
        ("gdb3", "gdb3.mol5.PDB", 1, 2, "psi4", 3, 4, "jordan_WIGNER"),
        ("gdb3", "gdb3.mol5.PDB", 1, 2, "psi4", 3, 4, "BRAVYI_kitaev"),
        ("gdb3", "gdb3.mol5.PDB", -1, 2, "psi4", 1, 2, "jordan_WIGNER"),
        ("gdb3", "gdb3.mol5.PDB", -1, 2, "psi4", 1, 2, "BRAVYI_kitaev"),
        ("gdb3", "gdb3.mol5.PDB", 2, 1, "psi4", 2, 2, "jordan_WIGNER"),
        ("gdb3", "gdb3.mol5.PDB", 2, 1, "psi4", 2, 2, "BRAVYI_kitaev"),
        ("gdb3", "gdb3.mol5.PDB", 0, 1, "pyscf", 2, 2, "jordan_WIGNER"),
        ("gdb3", "gdb3.mol5.PDB", 0, 1, "pyscf", 2, 2, "BRAVYI_kitaev"),
        ("gdb3", "gdb3.mol5.PDB", 1, 2, "pyscf", 3, 4, "jordan_WIGNER"),
        ("gdb3", "gdb3.mol5.PDB", 1, 2, "pyscf", 3, 4, "BRAVYI_kitaev"),
        ("gdb3", "gdb3.mol5.PDB", -1, 2, "pyscf", 1, 2, "jordan_WIGNER"),
        ("gdb3", "gdb3.mol5.PDB", -1, 2, "pyscf", 1, 2, "BRAVYI_kitaev"),
        ("gdb3", "gdb3.mol5.PDB", 2, 1, "pyscf", 2, 2, "jordan_WIGNER"),
        ("gdb3", "gdb3.mol5.PDB", 2, 1, "pyscf", 2, 2, "BRAVYI_kitaev"),
    ],
)
def test_building_hamiltonian(
    mol_name,
    geo_file,
    charge,
    multiplicity,
    qc_prog,
    n_act_electrons,
    n_act_orbitals,
    transformation,
    psi4_support,
    requires_babel,
    tmpdir,
):
    r"""Test that the generated Hamiltonian `built_hamiltonian` is an instance of the PennyLane
    Hamiltonian class and the correctness of the total number of qubits required to run the
    quantum simulation. The latter is tested for different values of the molecule's charge and
    for active spaces with different size"""

    if qc_prog == "psi4" and not psi4_support:
        pytest.skip("Skipped, no Psi4 support")

    geo_file = os.path.join(ref_dir, geo_file)

    built_hamiltonian, n_qubits = qchem.generate_hamiltonian(
        mol_name,
        geo_file,
        charge,
        multiplicity,
        "sto-3g",
        qc_package=qc_prog,
        n_active_electrons=n_act_electrons,
        n_active_orbitals=n_act_orbitals,
        mapping=transformation,
        outpath=tmpdir.strpath,
    )

    assert isinstance(built_hamiltonian, Hamiltonian)
    assert n_qubits == 2 * n_act_orbitals
