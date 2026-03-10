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
"""Tests for the perturbation theory function"""

import numpy as np
import pytest

import pennylane as qml
from pennylane.labs.trotter_error import ProductFormula, generic_fragments, perturbation_error
from pennylane.qchem import fermionic_observable

symbols = ["H", "H", "H", "H"]
geometry = qml.math.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0], [0.0, 0.0, 3.0]])

mol = qml.qchem.Molecule(symbols, geometry, unit="angstrom")
core_constant, one_body, two_body = qml.qchem.electron_integrals(mol)()

two_chem = qml.math.swapaxes(two_body, 1, 3)  # V_pqrs
one_chem = one_body - 0.5 * qml.math.einsum("pqss", two_body)  # T_pq

_, two_body_cores, two_body_leaves = qml.qchem.factorize(two_chem, tol_factor=1e-2)

one_body_eigvals, one_body_eigvecs = qml.math.linalg.eigh(one_chem)
one_body_cores = qml.math.diag(one_body_eigvals)
one_body_leaves = one_body_eigvecs

# CDF Hamiltonian
U0 = one_body_leaves
Z0 = one_body_cores
U = two_body_leaves
Z = two_body_cores

eri = np.einsum("tpk,tqk,tkl,trl,tsl->tpqrs", U, U, Z, U, U)  # regenerate V_pqrs
h1e = U0 @ Z0 @ U0.T
h1e = h1e + 0.5 * qml.math.einsum("pqss", two_body)  # regenerate h_pq

h_ferm = [fermionic_observable(constant=core_constant, one=h1e)]
for frag in eri:
    h_ferm.append(fermionic_observable(constant=np.array([0]), two=np.swapaxes(frag, 1, 3)))

cdf_frags = [np.array(item.to_mat(format="dense")) for item in h_ferm]

frags = generic_fragments(cdf_frags)
eigenvalues, eigenvectors = np.linalg.eigh(sum(cdf_frags))
state = eigenvectors.T[:, eigenvalues.argsort()][0]

frags = dict(enumerate(frags))
frag_labels = list(frags.keys()) + list(frags.keys())[::-1]
frag_coeffs = [1 / 2] * len(frag_labels)


params = [
    ("serial", 1, "state", 1),
    ("serial", 1, "state", 2),
    ("mp_pool", 2, "state", 2),
    ("mp_pool", 2, "commutator", 2),
    ("cf_procpool", 2, "state", 2),
    ("cf_procpool", 2, "commutator", 2),
    ("cf_threadpool", 2, "state", 2),
    ("cf_threadpool", 2, "commutator", 2),
    ("mpi4py_pool", 2, "state", 2),
    ("mpi4py_pool", 2, "commutator", 2),
    ("mpi4py_comm", 2, "state", 2),
    ("mpi4py_comm", 2, "commutator", 2),
]


@pytest.mark.parametrize("backend, num_workers, parallel_mode, n_states", params)
def test_perturbation_error(backend, num_workers, parallel_mode, n_states, mpi4py_support):
    """Test that perturbation_error returns the correct result. This is a precomputed example
    of perturbation theory on a CDF Hamiltonian."""

    if backend in {"mpi4py_pool", "mpi4py_comm"} and not mpi4py_support:
        pytest.skip(f"Skipping test: '{backend}' requires mpi4py, which is not installed.")

    second_order = ProductFormula(frag_labels, frag_coeffs)
    states = [state] * n_states
    max_order, timestep = 3, 1.0
    errors = perturbation_error(
        second_order, frags, states, max_order, timestep, num_workers, backend, parallel_mode
    )

    actual = [sum(d.values()) for d in errors]
    expected = np.array([0.009947958260807521j] * n_states)  # computed using effective_hamiltonian

    assert np.allclose(actual, expected)
