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

import pennylane as qp
from pennylane.labs.trotter_error import (
    NumpyFragment,
    NumpyState,
    ProductFormula,
    perturbation_error,
)
from pennylane.qchem import fermionic_observable

symbols = ["H", "H", "H", "H"]
geometry = qp.math.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0], [0.0, 0.0, 3.0]])

mol = qp.qchem.Molecule(symbols, geometry, unit="angstrom")
core_constant, one_body, two_body = qp.qchem.electron_integrals(mol)()

two_chem = qp.math.swapaxes(two_body, 1, 3)  # V_pqrs
one_chem = one_body - 0.5 * qp.math.einsum("pqss", two_body)  # T_pq

_, two_body_cores, two_body_leaves = qp.qchem.factorize(two_chem, tol_factor=1e-2)

one_body_eigvals, one_body_eigvecs = qp.math.linalg.eigh(one_chem)
one_body_cores = qp.math.diag(one_body_eigvals)
one_body_leaves = one_body_eigvecs

# CDF Hamiltonian
U0 = one_body_leaves
Z0 = one_body_cores
U = two_body_leaves
Z = two_body_cores

eri = np.einsum("tpk,tqk,tkl,trl,tsl->tpqrs", U, U, Z, U, U)  # regenerate V_pqrs
h1e = U0 @ Z0 @ U0.T
h1e = h1e + 0.5 * qp.math.einsum("pqss", two_body)  # regenerate h_pq

h_ferm = [fermionic_observable(constant=core_constant, one=h1e)]
for frag in eri:
    h_ferm.append(fermionic_observable(constant=np.array([0]), two=np.swapaxes(frag, 1, 3)))

cdf_frags = [np.array(item.to_mat(format="dense")) for item in h_ferm]

frags = [NumpyFragment(cdf_frag) for cdf_frag in cdf_frags]
eigenvalues, eigenvectors = np.linalg.eigh(sum(cdf_frags))
state = NumpyState(eigenvectors.T[:, eigenvalues.argsort()][0])

frags = dict(enumerate(frags))
frag_labels = list(frags.keys()) + list(frags.keys())[::-1]
frag_coeffs = [1 / 2] * len(frag_labels)
order = 3

params = [
    ("serial", 1),
    ("mp_pool", 1),
    ("mp_pool", 2),
    ("cf_procpool", 1),
    ("cf_procpool", 2),
    ("cf_threadpool", 1),
    ("cf_threadpool", 2),
    ("mpi4py_pool", 1),
    ("mpi4py_pool", 1),
    ("mpi4py_comm", 1),
    ("mpi4py_comm", 2),
]


@pytest.mark.parametrize("backend, num_workers", params)
def test_perturbation_error(backend, num_workers, mpi4py_support):
    """Test that perturbation_error returns the correct result. This is a precomputed example
    of perturbation theory on a CDF Hamiltonian."""

    if backend in {"mpi4py_pool", "mpi4py_comm"} and not mpi4py_support:
        pytest.skip(f"Skipping test: '{backend}' requires mpi4py, which is not installed.")

    pf = ProductFormula(list(zip(frag_labels, frag_coeffs)))
    actual_dict = perturbation_error(
        pf,
        frags,
        state,
        order=order,
        timestep=1.0,
        backend=backend,
        num_workers=num_workers,
    )

    actual = actual_dict[3]
    expected = 0.009947958260807521j

    assert np.allclose(actual, expected)
