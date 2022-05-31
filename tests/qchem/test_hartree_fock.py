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
import autograd
import pytest

from pennylane import numpy as np
from pennylane import qchem


@pytest.mark.parametrize(
    ("symbols", "geometry", "v_fock", "coeffs", "fock_matrix", "h_core", "repulsion_tensor"),
    [
        (
            ["H", "H"],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], requires_grad=False),
            np.array([-0.67578019, 0.94181155]),
            np.array([[-0.52754647, -1.56782303], [-0.52754647, 1.56782303]]),
            np.array([[-0.51126165, -0.70283714], [-0.70283714, -0.51126165]]),
            np.array([[-1.27848869, -1.21916299], [-1.21916299, -1.27848869]]),
            np.array(
                [
                    [
                        [[0.77460595, 0.56886144], [0.56886144, 0.65017747]],
                        [[0.56886144, 0.45590152], [0.45590152, 0.56886144]],
                    ],
                    [
                        [[0.56886144, 0.45590152], [0.45590152, 0.56886144]],
                        [[0.65017747, 0.56886144], [0.56886144, 0.77460595]],
                    ],
                ]
            ),
        )
    ],
)
def test_scf(symbols, geometry, v_fock, coeffs, fock_matrix, h_core, repulsion_tensor):
    r"""Test that scf returns the correct values."""
    mol = qchem.Molecule(symbols, geometry)
    v, c, f, h, e = qchem.scf(mol)()

    assert np.allclose(v, v_fock)
    assert np.allclose(c, coeffs)
    assert np.allclose(f, fock_matrix)
    assert np.allclose(h, h_core)
    assert np.allclose(e, repulsion_tensor)


@pytest.mark.parametrize(
    ("symbols", "geometry", "charge", "basis_name", "e_ref"),
    [
        (
            ["H", "H"],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], requires_grad=False),
            0,
            "sto-3g",
            # HF energy computed with pyscf using scf.SCF(mol).kernel(numpy.eye(mol.nao_nr()))
            np.array([-1.06599931664376]),
        ),
        (
            ["H", "H", "H"],
            np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], requires_grad=False),
            1,
            "sto-3g",
            # HF energy computed with pyscf using scf.SCF(mol).kernel(numpy.eye(mol.nao_nr()))
            np.array([-0.948179228995941]),
        ),
        (
            ["H", "F"],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], requires_grad=False),
            0,
            "sto-3g",
            # HF energy computed with pyscf using scf.SCF(mol).kernel(numpy.eye(mol.nao_nr()))
            np.array([-97.8884541671664]),
        ),
        (
            ["H", "He"],
            np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], requires_grad=False),
            1,
            "6-31G",
            # HF energy computed with pyscf using scf.hf.SCF(mol).kernel(numpy.eye(mol.nao_nr()))
            np.array([-2.83655236013837]),
        ),
    ],
)
def test_hf_energy(symbols, geometry, charge, basis_name, e_ref):
    r"""Test that hf_energy returns the correct energy."""
    mol = qchem.Molecule(symbols, geometry, charge=charge, basis_name=basis_name)
    e = qchem.hf_energy(mol)()

    assert np.allclose(e, e_ref)


@pytest.mark.parametrize(
    ("symbols", "geometry", "g_ref"),
    [
        (
            ["H", "H"],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], requires_grad=True),
            # HF gradient computed with pyscf using rnuc_grad_method().kernel()
            np.array([[0.0, 0.0, 0.3650435], [0.0, 0.0, -0.3650435]]),
        ),
        (
            ["H", "Li"],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]], requires_grad=True),
            # HF gradient computed with pyscf using rnuc_grad_method().kernel()
            np.array([[0.0, 0.0, 0.21034957], [0.0, 0.0, -0.21034957]]),
        ),
    ],
)
def test_hf_energy_gradient(symbols, geometry, g_ref):
    r"""Test that the gradient of the Hartree-Fock energy wrt differentiable parameters is
    correct."""
    mol = qchem.Molecule(symbols, geometry)
    args = [mol.coordinates]
    g = autograd.grad(qchem.hf_energy(mol))(*args)

    assert np.allclose(g, g_ref)


@pytest.mark.parametrize(
    ("symbols", "geometry", "e_ref"),
    [
        # e_repulsion = \sum_{ij} (q_i * q_j / r_{ij})
        (
            ["H", "H"],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], requires_grad=False),
            np.array([1.0]),
        ),
        (
            ["H", "F"],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]], requires_grad=True),
            np.array([4.5]),
        ),
        (
            ["H", "O", "H"],
            np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], requires_grad=True),
            np.array([16.707106781186546]),
        ),
    ],
)
def test_nuclear_energy(symbols, geometry, e_ref):
    r"""Test that nuclear_energy returns the correct energy."""
    mol = qchem.Molecule(symbols, geometry)
    args = [mol.coordinates]
    e = qchem.nuclear_energy(mol.nuclear_charges, mol.coordinates)(*args)
    assert np.allclose(e, e_ref)


@pytest.mark.parametrize(
    ("symbols", "geometry", "g_ref"),
    [
        # gradient = d(q_i * q_j / (xi - xj)) / dxi, ...
        (
            ["H", "H"],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], requires_grad=True),
            np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]),
        ),
        (
            ["H", "F"],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]], requires_grad=True),
            np.array([[0.0, 0.0, 2.25], [0.0, 0.0, -2.25]]),
        ),
    ],
)
def test_nuclear_energy_gradient(symbols, geometry, g_ref):
    r"""Test that nuclear energy gradients are correct."""
    mol = qchem.Molecule(symbols, geometry)
    args = [mol.coordinates]
    g = autograd.grad(qchem.nuclear_energy(mol.nuclear_charges, mol.coordinates))(*args)
    assert np.allclose(g, g_ref)
