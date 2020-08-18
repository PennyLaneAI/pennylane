import os

import numpy as np
import pytest
from openfermion.hamiltonians import MolecularData

from pennylane import qchem

name = "h2"
geometry = [("H", [0.0, 0.0, -0.35]), ("H", [0.0, 0.0, 0.35])]


@pytest.mark.parametrize(
    ("package", "basis"), [("PySCF", "sto-3g"), ("PySCF", "6-31g"), ("Psi4", "sto-3g")]
)
def test_path_to_file(package, basis, tmpdir, psi4_support):
    r"""Test the correctness of the full path to the file containing the meanfield
    electronic structure"""

    if package == "Psi4" and not psi4_support:
        pytest.skip("Skipped, no Psi4 support")

    exp_path = os.path.join(tmpdir.strpath, package.lower(), basis.strip(), name)

    res_path = qchem.meanfield(name, geometry, basis=basis, package=package, outpath=tmpdir.strpath)

    assert res_path == exp_path


@pytest.mark.parametrize("package", ["Psi4", "PySCF"])
def test_hf_calculations(package, tmpdir, psi4_support, tol):
    r"""Test the correctness of the HF calculation"""

    if package == "Psi4" and not psi4_support:
        pytest.skip("Skipped, no Psi4 support")

    n_atoms = 2
    n_electrons = 2
    n_orbitals = 2
    hf_energy = -1.1173490350703152
    orbital_energies = np.array([-0.59546347, 0.71416528])

    one_body_integrals = np.array(
        [[-1.27785300e00, 1.11022302e-16], [0.00000000e00, -4.48299698e-01]]
    )

    two_body_integrals = np.array(
        [
            [
                [[6.82389533e-01, 0.00000000e00], [6.93889390e-17, 1.79000576e-01]],
                [[4.16333634e-17, 1.79000576e-01], [6.70732778e-01, 0.00000000e00]],
            ],
            [
                [[5.55111512e-17, 6.70732778e-01], [1.79000576e-01, 1.11022302e-16]],
                [[1.79000576e-01, 0.00000000e00], [2.77555756e-17, 7.05105632e-01]],
            ],
        ]
    )

    fullpath = qchem.meanfield(name, geometry, package=package, outpath=tmpdir.strpath)

    molecule = MolecularData(filename=fullpath)

    assert molecule.n_atoms == n_atoms
    assert molecule.n_electrons == n_electrons
    assert molecule.n_orbitals == n_orbitals
    assert np.allclose(molecule.hf_energy, hf_energy, **tol)
    assert np.allclose(molecule.orbital_energies, orbital_energies, **tol)
    assert np.allclose(molecule.one_body_integrals, one_body_integrals, **tol)
    assert np.allclose(molecule.two_body_integrals, two_body_integrals, **tol)


@pytest.mark.parametrize("package", ["not_available_package"])
def test_not_available_qc_package(package, tmpdir):
    r"""Test that an error is raised if the input quantum chemistry package
    is neither Psi4 nor PySCF"""

    with pytest.raises(TypeError, match="Integration with quantum chemistry package"):
        qchem.meanfield(name, geometry, package=package, outpath=tmpdir.strpath)
