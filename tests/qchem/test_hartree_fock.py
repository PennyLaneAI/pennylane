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
Unit tests for Hartree-Fock functions.
"""
# pylint: disable=too-many-arguments,too-few-public-methods
import pytest

import pennylane as qml
from pennylane import numpy as np
from pennylane import qchem


def test_scf_leaves_random_seed_unchanged():
    """Tests that the scf function leaves the global numpy sampling state unchanged."""

    symbols = ["H", "H"]
    geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], requires_grad=False)
    alpha = np.array(
        [[3.42525091, 0.62391373, 0.1688554], [3.42525091, 0.62391373, 0.1688554]],
        requires_grad=True,
    )
    mol = qchem.Molecule(symbols, geometry, alpha=alpha)
    args = [alpha]

    initial_numpy_state = np.random.get_state()
    qchem.scf(mol)(*args)
    final_numpy_state = np.random.get_state()

    assert initial_numpy_state[0] == final_numpy_state[0]
    assert np.all(initial_numpy_state[1] == final_numpy_state[1])


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


def test_scf_openshell_error():
    r"""Test that scf raises an error when an open-shell molecule is provided."""
    symbols = ["H", "H", "H"]
    geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0]], requires_grad=False)
    mol = qchem.Molecule(symbols, geometry)

    with pytest.raises(ValueError, match="Open-shell systems are not supported."):
        qchem.scf(mol)()


@pytest.mark.parametrize(
    ("symbols", "geometry", "charge", "basis_name", "e_ref"),
    [
        (
            ["H", "H"],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], requires_grad=False),
            0,
            "sto-3g",
            # HF energy computed with pyscf using scf.hf.SCF(mol_pyscf).kernel()
            np.array([-1.06599931664376]),
        ),
        (
            ["H", "H", "H"],
            np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], requires_grad=False),
            1,
            "sto-3g",
            # HF energy computed with pyscf using scf.hf.SCF(mol_pyscf).kernel()
            np.array([-0.948179228995941]),
        ),
        (
            ["H", "F"],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], requires_grad=False),
            0,
            "sto-3g",
            # HF energy computed with pyscf using scf.hf.SCF(mol_pyscf).kernel()
            np.array([-97.8884541671664]),
        ),
        (
            ["H", "He"],
            np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], requires_grad=False),
            1,
            "6-31G",
            # HF energy computed with pyscf using scf.hf.SCF(mol_pyscf).kernel()
            np.array([-2.83655236013837]),
        ),
        (
            ["H", "He"],
            np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], requires_grad=False),
            1,
            "6-311G",
            # HF energy computed with pyscf using scf.hf.SCF(mol_pyscf).kernel()
            np.array([-2.84429553346549]),
        ),
        (
            ["H", "He"],
            np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], requires_grad=False),
            1,
            "cc-pvdz",
            # HF energy computed with pyscf using scf.hf.SCF(mol_pyscf).kernel()
            np.array([-2.84060925839206]),
        ),
    ],
)
def test_hf_energy(symbols, geometry, charge, basis_name, e_ref):
    r"""Test that hf_energy returns the correct energy."""
    mol = qchem.Molecule(symbols, geometry, charge=charge, basis_name=basis_name)
    e = qchem.hf_energy(mol)()

    assert np.allclose(e, e_ref)


@pytest.mark.parametrize(
    ("symbols", "geometry", "alpha", "charge", "basis_name", "e_ref"),
    [
        (
            ["H", "H"],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], requires_grad=True),
            [
                np.array([3.42525091, 0.62391373, 0.1688554], requires_grad=True),
                np.array([3.42525091, 0.62391373, 0.1688554], requires_grad=True),
            ],
            0,
            "sto-3g",
            # HF energy computed with pyscf using scf.hf.SCF(mol_pyscf).kernel()
            np.array([-1.06599931664376]),
        ),
        (
            ["H", "H", "H"],
            np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], requires_grad=True),
            [
                np.array([18.73113696, 2.82539437, 0.64012169], requires_grad=True),
                np.array([0.16127776], requires_grad=True),
                np.array([18.73113696, 2.82539437, 0.64012169], requires_grad=True),
                np.array([0.16127776], requires_grad=True),
                np.array([18.73113696, 2.82539437, 0.64012169], requires_grad=True),
                np.array([0.16127776], requires_grad=True),
            ],
            1,
            "6-31g",
            # HF energy computed with pyscf using scf.hf.SCF(mol_pyscf).kernel()
            np.array([-1.11631458846075]),
        ),
    ],
)
def test_hf_energy_diff(symbols, geometry, alpha, charge, basis_name, e_ref):
    r"""Test that hf_energy returns the correct energy with differentiable parameters."""
    mol = qchem.Molecule(symbols, geometry, alpha=alpha, charge=charge, basis_name=basis_name)
    args = [geometry, alpha]
    e = qchem.hf_energy(mol)(*args)

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
    g = qml.grad(qchem.hf_energy(mol))(*args)

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
    g = qml.grad(qchem.nuclear_energy(mol.nuclear_charges, mol.coordinates))(*args)
    assert np.allclose(g, g_ref)


@pytest.mark.jax
class TestJax:
    @pytest.mark.parametrize(
        ("symbols", "geometry", "e_ref"),
        [
            # e_repulsion = \sum_{ij} (q_i * q_j / r_{ij})
            (
                ["H", "H"],
                np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
                np.array([1.0]),
            ),
            (
                ["H", "F"],
                np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]),
                np.array([4.5]),
            ),
            (
                ["H", "O", "H"],
                np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
                np.array([16.707106781186546]),
            ),
        ],
    )
    def test_nuclear_energy_jax(self, symbols, geometry, e_ref):
        r"""Test that nuclear_energy returns the correct energy when using jax."""
        geometry = qml.math.array(geometry, like="jax")
        mol = qchem.Molecule(symbols, geometry)
        args = [mol.coordinates]
        e = qchem.nuclear_energy(mol.nuclear_charges, mol.coordinates)(*args)
        assert qml.math.allclose(e, e_ref)

    @pytest.mark.parametrize(
        ("symbols", "geometry", "g_ref"),
        [
            # gradient = d(q_i * q_j / (xi - xj)) / dxi, ...
            (
                ["H", "H"],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]],
            ),
            (
                ["H", "F"],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]],
                [[0.0, 0.0, 2.25], [0.0, 0.0, -2.25]],
            ),
        ],
    )
    def test_nuclear_energy_gradient_jax(self, symbols, geometry, g_ref):
        r"""Test that nuclear energy gradients are correct for jax."""
        import jax

        geometry = qml.math.array(geometry, like="jax")
        mol = qchem.Molecule(symbols, geometry)
        args = [geometry, mol.coeff, mol.alpha]
        g = jax.jacobian(qchem.nuclear_energy(mol.nuclear_charges, mol.coordinates), argnums=0)(
            *args
        )
        assert qml.math.allclose(g, g_ref)

    @pytest.mark.parametrize(
        ("symbols", "geometry", "g_ref"),
        [
            (
                ["H", "H"],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                # HF gradient computed with pyscf using rnuc_grad_method().kernel()
                [[0.0, 0.0, 0.3650435], [0.0, 0.0, -0.3650435]],
            ),
            (
                ["H", "Li"],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]],
                # HF gradient computed with pyscf using rnuc_grad_method().kernel()
                [[0.0, 0.0, 0.21034957], [0.0, 0.0, -0.21034957]],
            ),
        ],
    )
    def test_hf_energy_gradient_jax(self, symbols, geometry, g_ref):
        r"""Test that the gradient of the Hartree-Fock energy wrt differentiable parameters is
        correct with jax."""
        import jax

        geometry = qml.math.array(geometry, like="jax")

        mol = qchem.Molecule(symbols, geometry)
        args = [geometry, mol.coeff, mol.alpha]
        g = jax.grad(qchem.hf_energy(mol), argnums=[0])(*args)[0]
        assert qml.math.allclose(g, g_ref)
