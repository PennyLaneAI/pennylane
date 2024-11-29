# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

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
This module contains tests for functions needed to compute PES object.
"""
import sys

import numpy as np
import pytest

import pennylane as qml
from pennylane.qchem import vibrational

AU_TO_CM = 219475

# pylint: disable=too-many-arguments


@pytest.mark.parametrize(
    ("sym", "geom", "method", "expected_dipole"),
    # Expected energy was obtained using pyscf
    [
        (
            ["H", "F"],
            np.array([[0.0, 0.0, 0.03967368], [0.0, 0.0, 0.96032632]]),
            "RHF",
            [-3.78176692e-16, -3.50274735e-17, -9.05219767e-01],
        ),
        (
            ["H", "H", "S"],
            np.array(
                [
                    [0.0, -1.00688408, -0.9679942],
                    [0.0, 1.00688408, -0.9679942],
                    [0.0, 0.0, -0.0640116],
                ]
            ),
            "UHF",
            [1.95258747e-16, 5.62355462e-15, -7.34149703e-01],
        ),
    ],
)
@pytest.mark.usefixtures("skip_if_no_pyscf_support")
def test_get_dipole(sym, geom, method, expected_dipole):
    r"""Test that the get_dipole function produces correct results."""

    mol = qml.qchem.Molecule(sym, geom, basis_name="6-31g", unit="Angstrom", load_data=True)
    mol_scf = vibrational.single_point(mol, method=method)
    dipole = vibrational.get_dipole(mol_scf, method=method)
    assert np.allclose(dipole, expected_dipole)
