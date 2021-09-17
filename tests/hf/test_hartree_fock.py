# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

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
Unit tests for for Hartree-Fock functions.
"""
import pytest
from pennylane import numpy as np
from pennylane.hf.hartree_fock import generate_hartree_fock, hf_energy
from pennylane.hf.molecule import Molecule


@pytest.mark.parametrize(
    ("symbols", "geometry", "e_ref"),
    [
        (
            ["H", "H"],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], requires_grad=False),
            # HF energy computed with pyscf using scf.hf.SCF(mol).kernel(numpy.eye(mol.nao_nr()))
            np.array([-1.06599931664376]),
        ),
        (
            ["H", "F"],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], requires_grad=False),
            # HF energy computed with pyscf using scf.hf.SCF(mol).kernel(numpy.eye(mol.nao_nr()))
            np.array([-97.8884541671664]),
        )
    ],
)
def test_hartree_fock(symbols, geometry, e_ref):
    r"""Test that overlap_matrix returns the correct matrix."""
    mol = Molecule(symbols, geometry)
    e = hf_energy(mol)()
    print(e)
    assert np.allclose(e, e_ref)
