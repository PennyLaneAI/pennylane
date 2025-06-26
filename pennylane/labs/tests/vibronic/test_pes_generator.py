# Copyright 2025 Xanadu Quantum Technologies Inc.

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
import numpy as np

from pennylane import qchem

from pennylane.labs.vibronic.pes_vibronic import vibronic_pes


def test_onemode_pes():

    symbols = ["H", "O", "H"]
    geometry = np.array(
        [[-0.0399, -0.0038, 0.0000], [1.5780, 0.8540, 0.0000], [2.7909, -0.5159, 0.0000]]
    )

    mol = qchem.Molecule(symbols, geometry, basis_name="6-31g", unit="bohr", load_data=True)

    pes_obj = vibronic_pes(mol, n_points=3, optimize=False, rotate=False)

    assert np.allclose(pes_obj.pes_onemode[-1], np.array([-75.98551162, -75.70135941]))


def test_onemode_pes_tddft():

    symbols = ["H", "O", "H"]
    geometry = np.array(
        [[-0.0399, -0.0038, 0.0000], [1.5780, 0.8540, 0.0000], [2.7909, -0.5159, 0.0000]]
    )

    mol = qchem.Molecule(symbols, geometry, basis_name="6-31g", unit="bohr", load_data=True)

    pes_obj = vibronic_pes(mol, n_points=3, optimize=False, rotate=False)

    assert np.allclose(pes_obj.pes_onemode[-1], np.array([-548.5203213540069, -548.5143355031091]))
