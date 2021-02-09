import os

import pytest

from pennylane import qchem

from pennylane.vqe import Hamiltonian

import numpy as np


symbols = ["C", "C", "N", "H", "H", "H", "H", "H"]
coordinates = np.array(
    [
        0.361,
        -0.452,
        -0.551,
        -0.714,
        0.125,
        0.327,
        0.683,
        0.133,
        0.745,
        0.442,
        -1.529,
        -0.619,
        0.672,
        0.102,
        -1.428,
        -1.364,
        -0.56,
        0.857,
        -1.149,
        1.08,
        0.06,
        1.093,
        1.063,
        0.636,
    ]
)


@pytest.mark.parametrize(
    ("charge", "mult", "package", "nact_els", "nact_orbs", "mapping",),
    [
        (0, 1, "psi4", 2, 2, "jordan_WIGNER"),
        (1, 2, "pyscf", 3, 4, "BRAVYI_kitaev"),
        (-1, 2, "pyscf", 1, 2, "jordan_WIGNER"),
        (2, 1, "psi4", 2, 2, "BRAVYI_kitaev"),
    ],
)
def test_building_hamiltonian(
    charge, mult, package, nact_els, nact_orbs, mapping, psi4_support, requires_babel, tmpdir,
):
    r"""Test that the generated Hamiltonian `built_hamiltonian` is an instance of the PennyLane
    Hamiltonian class and the correctness of the total number of qubits required to run the
    quantum simulation. The latter is tested for different values of the molecule's charge and
    for active spaces with different size"""

    if package == "psi4" and not psi4_support:
        pytest.skip("Skipped, no Psi4 support")

    built_hamiltonian, qubits = qchem.molecular_hamiltonian(
        "gdb3_mol5",
        symbols,
        coordinates,
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
