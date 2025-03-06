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
Unit tests for the ``meanfield`` function.
"""
import os

import numpy as np
import pytest

from pennylane import qchem

name = "h2"
symbols, coordinates = (["H", "H"], np.array([0.0, 0.0, -0.66140414, 0.0, 0.0, 0.66140414]))


@pytest.mark.usefixtures("skip_if_no_openfermion_support")
@pytest.mark.parametrize(("package", "basis"), [("PySCF", "sto-3g"), ("PySCF", "6-31g")])
def test_path_to_file(package, basis, tmpdir):
    r"""Test the correctness of the full path to the file containing the meanfield
    electronic structure"""
    filename = name.strip() + "_" + package.lower() + "_" + basis.strip()
    exp_path = os.path.join(tmpdir.strpath, filename)

    res_path = qchem.meanfield(
        symbols, coordinates, name=name, basis=basis, package=package, outpath=tmpdir.strpath
    )

    assert res_path == exp_path


@pytest.mark.usefixtures("skip_if_no_openfermion_support")
@pytest.mark.parametrize("package", ["PySCF"])
def test_hf_calculations(package, tmpdir, tol):
    r"""Test the correctness of the HF calculation"""
    import openfermion

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

    fullpath = qchem.meanfield(
        symbols, coordinates, name=name, package=package, outpath=tmpdir.strpath
    )

    molecule = openfermion.MolecularData(filename=fullpath)

    assert molecule.n_atoms == n_atoms
    assert molecule.n_electrons == n_electrons
    assert molecule.n_orbitals == n_orbitals
    assert np.allclose(molecule.hf_energy, hf_energy, **tol)
    assert np.allclose(molecule.orbital_energies, orbital_energies, **tol)
    assert np.allclose(molecule.one_body_integrals, one_body_integrals, **tol)
    assert np.allclose(molecule.two_body_integrals, two_body_integrals, **tol)


@pytest.mark.usefixtures("skip_if_no_openfermion_support")
def test_not_available_qc_package(tmpdir):
    r"""Test that an error is raised if the input quantum chemistry package
    is not PySCF"""

    with pytest.raises(TypeError, match="Integration with quantum chemistry package"):
        qchem.meanfield(
            symbols, coordinates, name=name, package="not_available_package", outpath=tmpdir.strpath
        )


@pytest.mark.usefixtures("skip_if_no_openfermion_support")
def test_dimension_consistency(tmpdir):
    r"""Test that an error is raised if the size of the 'coordinates' array is
    not equal to ``3*len(symbols)``"""

    extra_coordinate = np.array([0.0, 0.0, -0.66140414, 0.0, 0.0, 0.66140414, -0.987])
    with pytest.raises(ValueError, match="The size of the array 'coordinates' has to be"):
        qchem.meanfield(symbols, extra_coordinate, name=name, outpath=tmpdir.strpath)
