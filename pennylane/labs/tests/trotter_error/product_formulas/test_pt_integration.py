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
geometry = qml.math.array([[0.0, 0.0, -0.2], [0.0, 0.0, -0.1], [0.0, 0.0, 0.1], [0.0, 0.0, 0.2]])

mol = qml.qchem.Molecule(symbols, geometry)
nuc_core, one_body, two_body = qml.qchem.electron_integrals(mol)()

two_chem = 0.5 * qml.math.swapaxes(two_body, 1, 3)  # V_pqrs
one_chem = one_body - 0.5 * qml.math.einsum("pqss", two_body)  # T_pq

core_shift, one_shift, two_shift = qml.qchem.symmetry_shift(
    nuc_core, one_chem, two_chem, n_elec=mol.n_electrons
)  # symmetry-shifted terms of the Hamiltonian

_, two_body_cores, two_body_leaves = qml.qchem.factorize(
    two_shift, tol_factor=1e-2, cholesky=True, compressed=True, regularization="L2"
)  # compressed double-factorized shifted two-body terms with "L2" regularization

two_core_prime = qml.math.eye(mol.n_orbitals) * two_body_cores.sum(axis=-1)[:, None, :]
one_body_extra = qml.math.einsum(
    "tpk,tkk,tqk->pq", two_body_leaves, two_core_prime, two_body_leaves
)  # one-body correction

one_body_eigvals, one_body_eigvecs = qml.math.linalg.eigh(one_shift + one_body_extra)
one_body_cores = qml.math.expand_dims(qml.math.diag(one_body_eigvals), axis=0)
one_body_leaves = qml.math.expand_dims(one_body_eigvecs, axis=0)
cdf_hamiltonian = {
    "nuc_constant": core_shift[0],
    "core_tensors": qml.math.concatenate((one_body_cores, two_body_cores), axis=0),
    "leaf_tensors": qml.math.concatenate((one_body_leaves, two_body_leaves), axis=0),
}  # CDF Hamiltonian

circ_wires = range(2 * mol.n_orbitals)
hf_state = qml.qchem.hf_state(electrons=mol.n_electrons, orbitals=len(circ_wires))


@qml.qnode(qml.device("default.qubit", wires=circ_wires))
def create_state():
    """Create a basis state"""
    qml.BasisState(hf_state, wires=circ_wires)
    return qml.state()


state = create_state()
h = 0.1

# Load the CDF integrals
U = np.array(cdf_hamiltonian["leaf_tensors"][1:])
Z = np.array(cdf_hamiltonian["core_tensors"][1:])
U0 = np.array(cdf_hamiltonian["leaf_tensors"][0])
Z0 = np.array(cdf_hamiltonian["core_tensors"][0])

# Recreate the fragments
two_body_fragments = np.einsum("tpk,tqk,tkl,trl,tsl->tpqrs", U, U, Z, U, U)
one_body = U0 @ Z0 @ U0.T

fermionic_fragments = [fermionic_observable(one=one_body, constant=[0.0]).to_mat(format="dense")]
for i in range(two_body_fragments.shape[0]):
    obc = qml.math.einsum("pk,kk,qk->pq", U[i], Z[i], U[i])
    fermionic_fragments.append(
        fermionic_observable(one=-obc, two=two_body_fragments[i], constant=[0.0]).to_mat(
            format="dense"
        )
    )

fragments = dict(enumerate(generic_fragments(fermionic_fragments)))

frag_labels = list(fragments.keys())
frag_labels = frag_labels + frag_labels[::-1]
frag_coeffs = [1 / 2] * len(frag_labels)
order = 3

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

    pf = ProductFormula(frag_labels, coeffs=frag_coeffs)
    states = [state] * n_states
    actual_dicts = perturbation_error(
        pf,
        fragments,
        states,
        max_order=order,
        timestep=h,
        backend=backend,
        num_workers=num_workers,
        parallel_mode=parallel_mode,
    )
    actual = [sum(d.values()) for d in actual_dicts]
    expected = np.array([1.5815039950517473e-05j] * n_states)

    assert np.allclose(actual, expected)
