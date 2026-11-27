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
Unit tests for functions needed for computing the Hamiltonian.
"""

import numpy as np
import pytest

import pennylane as qp


@pytest.mark.parametrize(
    "coordinates",
    [
        np.array([[0.0, 0.0, -0.6614], [0.0, 0.0, 0.6614]]),
        np.array([0.0, 0.0, -0.6614, 0.0, 0.0, 0.6614]),
    ],
)
def test_molecular_hamiltonian_numpy_data(coordinates):
    """Test that if numpy data is used with molecular hamiltonian, that numpy data is outputted."""
    symbols = ["H", "H"]

    H, _ = qp.qchem.molecular_hamiltonian(symbols, coordinates)
    assert all(qp.math.get_interface(d) == "numpy" for d in H.data)


def test_dhf_backend_removed():
    """Reject the removed backend before importing optional packages."""
    with pytest.raises(ValueError, match=r"method='dhf'\) has been removed"):
        qp.qchem.molecular_hamiltonian(["H", "H"], np.zeros((2, 3)), method="dhf")


def test_differentiable_parameters_removed():
    """The Hamiltonian driver no longer accepts solver differentiation arguments."""
    mol = qp.qchem.Molecule(["H", "H"], np.zeros((2, 3)))
    with pytest.raises(TypeError, match="unexpected keyword argument 'args'"):
        qp.qchem.molecular_hamiltonian(mol, args=[mol.coordinates])


def test_solver_exports_removed():
    """The differentiable solver and its construction functions are absent from qchem."""
    removed = {
        "scf",
        "hf_energy",
        "nuclear_energy",
        "electron_integrals",
        "diff_hamiltonian",
        "fermionic_hamiltonian",
        "dipole_integrals",
        "fermionic_dipole",
        "dipole_moment",
        "BasisFunction",
        "load_basisset",
        "atom_basis_data",
        "mol_basis_data",
        "overlap_matrix",
        "repulsion_tensor",
        "overlap_integral",
        "primitive_norm",
    }
    assert removed.isdisjoint(vars(qp.qchem))


def test_pyscf_default(monkeypatch):
    """Default Hamiltonian construction calls PySCF without requiring OpenFermion."""

    def integrals(*args):
        assert args[0] == ["H", "H"]
        np.testing.assert_equal(args[1], [0, 0, 0, 0, 0, 1.4])
        assert args[2:] == (0, 1, "sto-3g", None, None)
        return np.array([0.7]), np.diag([-1.2, -0.4]), np.zeros((2, 2, 2, 2))

    monkeypatch.setattr(qp.qchem.openfermion_pyscf, "_pyscf_integrals", integrals)
    mol = qp.qchem.Molecule(["H", "H"], np.array([[0, 0, 0], [0, 0, 1.4]]))
    actual, qubits = qp.qchem.molecular_hamiltonian(mol)
    assert qubits == 4
    assert np.isclose(qp.matrix(actual)[12, 12], -1.7)
