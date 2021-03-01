import os

import pytest

from pennylane import qchem

from pennylane.vqe import Hamiltonian

import numpy as np


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
