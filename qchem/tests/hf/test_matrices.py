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
Unit tests for functions needed for computing matrices.
"""
import autograd
import pytest
from pennylane import numpy as np
from pennylane import qchem


@pytest.mark.parametrize(
    ("n_electron", "c", "p_ref"),
    [
        (
            2,
            np.array([[-0.54828771, 1.21848441], [-0.54828771, -1.21848441]]),
            # all P elements are computed as 0.54828771**2 = 0.3006194129370441
            np.array([[0.30061941, 0.30061941], [0.30061941, 0.30061941]]),
        ),
    ],
)
def test_molecular_density_matrix(n_electron, c, p_ref):
    r"""Test that molecular_density_matrix returns the correct matrix."""
    p = qchem.hf.mol_density_matrix(n_electron, c)
    assert np.allclose(p, p_ref)


@pytest.mark.parametrize(
    ("symbols", "geometry", "alpha", "s_ref"),
    [
        (
            ["H", "H"],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], requires_grad=False),
            np.array(
                [[3.42525091, 0.62391373, 0.1688554], [3.42525091, 0.62391373, 0.1688554]],
                requires_grad=True,
            ),
            np.array([[1.0, 0.7965883009074122], [0.7965883009074122, 1.0]]),
        )
    ],
)
def test_overlap_matrix(symbols, geometry, alpha, s_ref):
    r"""Test that overlap_matrix returns the correct matrix."""
    mol = qchem.hf.Molecule(symbols, geometry, alpha=alpha)
    args = [alpha]
    s = qchem.hf.overlap_matrix(mol.basis_set)(*args)
    assert np.allclose(s, s_ref)


@pytest.mark.parametrize(
    ("symbols", "geometry", "s_ref"),
    [
        (
            ["H", "H"],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], requires_grad=False),
            np.array([[1.0, 0.7965883009074122], [0.7965883009074122, 1.0]]),
        )
    ],
)
def test_overlap_matrix_nodiff(symbols, geometry, s_ref):
    r"""Test that overlap_matrix returns the correct matrix when no differentiable parameter is
    used."""
    mol = qchem.hf.Molecule(symbols, geometry)
    s = qchem.hf.overlap_matrix(mol.basis_set)()
    assert np.allclose(s, s_ref)


@pytest.mark.parametrize(
    ("symbols", "geometry", "alpha", "coeff", "g_alpha_ref", "g_coeff_ref"),
    [
        (
            ["H", "H"],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], requires_grad=False),
            np.array(
                [[3.42525091, 0.62391373, 0.1688554], [3.42525091, 0.62391373, 0.1688554]],
                requires_grad=True,
            ),
            np.array(
                [[0.15432897, 0.53532814, 0.44463454], [0.15432897, 0.53532814, 0.44463454]],
                requires_grad=True,
            ),
            # Jacobian matrix contains gradient of S11, S12, S21, S22 wrt arg_1, arg_2.
            np.array(
                [
                    [
                        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                        [
                            [-0.00043783, -0.09917143, -0.11600206],
                            [-0.00043783, -0.09917143, -0.11600206],
                        ],
                    ],
                    [
                        [
                            [-0.00043783, -0.09917143, -0.11600206],
                            [-0.00043783, -0.09917143, -0.11600206],
                        ],
                        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    ],
                ]
            ),
            np.array(
                [
                    [
                        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                        [
                            [-0.15627637, -0.02812029, 0.08809831],
                            [-0.15627637, -0.02812029, 0.08809831],
                        ],
                    ],
                    [
                        [
                            [-0.15627637, -0.02812029, 0.08809831],
                            [-0.15627637, -0.02812029, 0.08809831],
                        ],
                        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    ],
                ]
            ),
        )
    ],
)
def test_gradient_overlap_matrix(symbols, geometry, alpha, coeff, g_alpha_ref, g_coeff_ref):
    r"""Test that the overlap gradients are correct."""
    mol = qchem.hf.Molecule(symbols, geometry, alpha=alpha, coeff=coeff)
    args = [mol.alpha, mol.coeff]
    g_alpha = autograd.jacobian(qchem.hf.overlap_matrix(mol.basis_set), argnum=0)(*args)
    g_coeff = autograd.jacobian(qchem.hf.overlap_matrix(mol.basis_set), argnum=1)(*args)
    assert np.allclose(g_alpha, g_alpha_ref)
    assert np.allclose(g_coeff, g_coeff_ref)


@pytest.mark.parametrize(
    ("symbols", "geometry", "alpha", "e", "idx", "s_ref"),
    [
        (
            ["H", "H"],
            np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], requires_grad=False),
            np.array(
                [[3.42525091, 0.62391373, 0.1688554], [3.42525091, 0.62391373, 0.1688554]],
                requires_grad=True,
            ),
            1,
            0,
            np.array([[0.0, 0.4627777], [0.4627777, 2.0]]),
        )
    ],
)
def test_moment_matrix(symbols, geometry, alpha, e, idx, s_ref):
    r"""Test that moment_matrix returns the correct matrix."""
    mol = qchem.hf.Molecule(symbols, geometry, alpha=alpha)
    args = [alpha]
    s = qchem.hf.moment_matrix(mol.basis_set, e, idx)(*args)
    assert np.allclose(s, s_ref)


@pytest.mark.parametrize(
    ("symbols", "geometry", "e", "idx", "s_ref"),
    [
        (
            ["H", "H"],
            np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], requires_grad=False),
            1,
            0,
            np.array([[0.0, 0.4627777], [0.4627777, 2.0]]),
        )
    ],
)
def test_moment_matrix_nodiff(symbols, geometry, e, idx, s_ref):
    r"""Test that moment_matrix returns the correct matrix when no differentiable parameter is
    used."""
    mol = qchem.hf.Molecule(symbols, geometry)
    s = qchem.hf.moment_matrix(mol.basis_set, e, idx)()
    assert np.allclose(s, s_ref)


@pytest.mark.parametrize(
    ("symbols", "geometry", "alpha", "coeff", "e", "idx", "g_alpha_ref", "g_coeff_ref"),
    [
        (
            ["H", "H"],
            np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], requires_grad=False),
            np.array(
                [[3.42525091, 0.62391373, 0.1688554], [3.42525091, 0.62391373, 0.1688554]],
                requires_grad=True,
            ),
            np.array(
                [[0.15432897, 0.53532814, 0.44463454], [0.15432897, 0.53532814, 0.44463454]],
                requires_grad=True,
            ),
            1,
            0,
            # Jacobian matrix contains gradient of S11, S12, S21, S22 wrt arg_1, arg_2, computed
            # with finite difference.
            np.array(
                [
                    [
                        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                        [
                            [3.87296664e-03, -2.29246093e-01, -9.93852751e-01],
                            [-4.86326933e-04, -6.72924734e-02, 2.47919030e-01],
                        ],
                    ],
                    [
                        [
                            [3.87296664e-03, -2.29246093e-01, -9.93852751e-01],
                            [-4.86326933e-04, -6.72924734e-02, 2.47919030e-01],
                        ],
                        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    ],
                ]
            ),
            np.array(
                [
                    [
                        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                        [
                            [-0.26160753, -0.18843804, 0.3176762],
                            [-0.09003791, 0.01797702, 0.00960757],
                        ],
                    ],
                    [
                        [
                            [-0.26160753, -0.18843804, 0.3176762],
                            [-0.09003791, 0.01797702, 0.00960757],
                        ],
                        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    ],
                ]
            ),
        )
    ],
)
def test_gradient_moment_matrix(symbols, geometry, alpha, coeff, e, idx, g_alpha_ref, g_coeff_ref):
    r"""Test that the moment matrix gradients are correct."""
    mol = qchem.hf.Molecule(symbols, geometry, alpha=alpha, coeff=coeff)
    args = [mol.alpha, mol.coeff]
    g_alpha = autograd.jacobian(qchem.hf.moment_matrix(mol.basis_set, e, idx), argnum=0)(*args)
    g_coeff = autograd.jacobian(qchem.hf.moment_matrix(mol.basis_set, e, idx), argnum=1)(*args)

    assert np.allclose(g_alpha, g_alpha_ref)
    assert np.allclose(g_coeff, g_coeff_ref)


@pytest.mark.parametrize(
    ("symbols", "geometry", "alpha", "t_ref"),
    [
        (
            ["H", "H"],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], requires_grad=False),
            np.array(
                [[3.42525091, 0.62391373, 0.1688554], [3.42525091, 0.62391373, 0.1688554]],
                requires_grad=True,
            ),
            np.array(
                [
                    [0.7600318862777408, 0.38325367405372557],
                    [0.38325367405372557, 0.7600318862777408],
                ]
            ),
        )
    ],
)
def test_kinetic_matrix(symbols, geometry, alpha, t_ref):
    r"""Test that kinetic_matrix returns the correct matrix."""
    mol = qchem.hf.Molecule(symbols, geometry, alpha=alpha)
    args = [alpha]
    t = qchem.hf.kinetic_matrix(mol.basis_set)(*args)
    assert np.allclose(t, t_ref)


@pytest.mark.parametrize(
    ("symbols", "geometry", "t_ref"),
    [
        (
            ["H", "H"],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], requires_grad=False),
            np.array(
                [
                    [0.7600318862777408, 0.38325367405372557],
                    [0.38325367405372557, 0.7600318862777408],
                ]
            ),
        )
    ],
)
def test_kinetic_matrix_nodiff(symbols, geometry, t_ref):
    r"""Test that kinetic_matrix returns the correct matrix when no differentiable parameter is
    used."""
    mol = qchem.hf.Molecule(symbols, geometry)
    t = qchem.hf.kinetic_matrix(mol.basis_set)()
    assert np.allclose(t, t_ref)


@pytest.mark.parametrize(
    ("symbols", "geometry", "alpha", "coeff", "g_alpha_ref", "g_coeff_ref"),
    [
        (
            ["H", "H"],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], requires_grad=False),
            np.array(
                [[3.42525091, 0.62391373, 0.1688554], [3.42525091, 0.62391373, 0.1688554]],
                requires_grad=True,
            ),
            np.array(
                [[0.15432897, 0.53532814, 0.44463454], [0.15432897, 0.53532814, 0.44463454]],
                requires_grad=True,
            ),
            # Jacobian matrix contains gradient of T11, T12, T21, T22 wrt arg_1, arg_2.
            np.array(
                [
                    [
                        [[0.03263157, 0.85287851, 0.68779528], [0.0, 0.0, 0.0]],
                        [
                            [-0.00502729, 0.08211579, 0.3090185],
                            [-0.00502729, 0.08211579, 0.3090185],
                        ],
                    ],
                    [
                        [
                            [-0.00502729, 0.08211579, 0.3090185],
                            [-0.00502729, 0.08211579, 0.3090185],
                        ],
                        [[0.0, 0.0, 0.0], [0.03263157, 0.85287851, 0.68779528]],
                    ],
                ]
            ),
            np.array(
                [
                    [
                        [[1.824217, 0.10606991, -0.76087597], [0.0, 0.0, 0.0]],
                        [
                            [-0.00846016, 0.08488012, -0.09925695],
                            [-0.00846016, 0.08488012, -0.09925695],
                        ],
                    ],
                    [
                        [
                            [-0.00846016, 0.08488012, -0.09925695],
                            [-0.00846016, 0.08488012, -0.09925695],
                        ],
                        [[0.0, 0.0, 0.0], [1.824217, 0.10606991, -0.76087597]],
                    ],
                ]
            ),
        )
    ],
)
def test_gradient_kinetic_matrix(symbols, geometry, alpha, coeff, g_alpha_ref, g_coeff_ref):
    r"""Test that the kinetic gradients are correct."""
    mol = qchem.hf.Molecule(symbols, geometry, alpha=alpha, coeff=coeff)
    args = [mol.alpha, mol.coeff]
    g_alpha = autograd.jacobian(qchem.hf.kinetic_matrix(mol.basis_set), argnum=0)(*args)
    g_coeff = autograd.jacobian(qchem.hf.kinetic_matrix(mol.basis_set), argnum=1)(*args)
    assert np.allclose(g_alpha, g_alpha_ref)
    assert np.allclose(g_coeff, g_coeff_ref)


@pytest.mark.parametrize(
    ("symbols", "geometry", "alpha", "v_ref"),
    [
        (
            ["H", "H"],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], requires_grad=False),
            np.array(
                [[3.42525091, 0.62391373, 0.1688554], [3.42525091, 0.62391373, 0.1688554]],
                requires_grad=True,
            ),
            # attraction matrix obtained from pyscf using mol.intor('int1e_nuc')
            np.array(
                [
                    [-2.03852075, -1.6024171],
                    [-1.6024171, -2.03852075],
                ]
            ),
        )
    ],
)
def test_attraction_matrix(symbols, geometry, alpha, v_ref):
    r"""Test that attraction_matrix returns the correct matrix."""
    mol = qchem.hf.Molecule(symbols, geometry, alpha=alpha)
    args = [mol.alpha]
    v = qchem.hf.attraction_matrix(mol.basis_set, mol.nuclear_charges, mol.coordinates)(*args)
    assert np.allclose(v, v_ref)


@pytest.mark.parametrize(
    ("symbols", "geometry", "alpha", "v_ref"),
    [
        (
            ["H", "H"],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], requires_grad=True),
            np.array(
                [[3.42525091, 0.62391373, 0.1688554], [3.42525091, 0.62391373, 0.1688554]],
                requires_grad=True,
            ),
            # attraction matrix obtained from pyscf using mol.intor('int1e_nuc')
            np.array(
                [
                    [-2.03852075, -1.6024171],
                    [-1.6024171, -2.03852075],
                ]
            ),
        )
    ],
)
def test_attraction_matrix_diffR(symbols, geometry, alpha, v_ref):
    r"""Test that attraction_matrix returns the correct matrix when positions are differentiable."""
    mol = qchem.hf.Molecule(symbols, geometry, alpha=alpha)
    r_basis = mol.coordinates
    args = [mol.coordinates, mol.alpha, r_basis]
    v = qchem.hf.attraction_matrix(mol.basis_set, mol.nuclear_charges, mol.coordinates)(*args)
    assert np.allclose(v, v_ref)


@pytest.mark.parametrize(
    ("symbols", "geometry", "v_ref"),
    [
        (
            ["H", "H"],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], requires_grad=False),
            # attraction matrix obtained from pyscf using mol.intor('int1e_nuc')
            np.array(
                [
                    [-2.03852075, -1.6024171],
                    [-1.6024171, -2.03852075],
                ]
            ),
        )
    ],
)
def test_attraction_matrix_nodiff(symbols, geometry, v_ref):
    r"""Test that attraction_matrix returns the correct matrix."""
    mol = qchem.hf.Molecule(symbols, geometry)
    v = qchem.hf.attraction_matrix(mol.basis_set, mol.nuclear_charges, mol.coordinates)()
    assert np.allclose(v, v_ref)


@pytest.mark.parametrize(
    ("symbols", "geometry", "alpha", "coeff", "g_r_ref"),
    [
        (
            ["H", "H"],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], requires_grad=True),
            np.array(
                [[3.42525091, 0.62391373, 0.1688554], [3.42525091, 0.62391373, 0.1688554]],
                requires_grad=True,
            ),
            np.array(
                [[0.15432897, 0.53532814, 0.44463454], [0.15432897, 0.53532814, 0.44463454]],
                requires_grad=True,
            ),
            np.array(
                [
                    [
                        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.44900112]],
                        [
                            [0.0, 0.0, -0.26468668],
                            [0.0, 0.0, 0.26468668],
                        ],
                    ],
                    [
                        [
                            [0.0, 0.0, -0.26468668],
                            [0.0, 0.0, 0.26468668],
                        ],
                        [[0.0, 0.0, -0.44900112], [0.0, 0.0, 0.0]],
                    ],
                ]
            ),
        )
    ],
)
def test_gradient_attraction_matrix(symbols, geometry, alpha, coeff, g_r_ref):
    r"""Test that the attraction gradients are correct."""
    mol = qchem.hf.Molecule(symbols, geometry, alpha=alpha, coeff=coeff)
    r_basis = mol.coordinates
    args = [mol.coordinates, mol.alpha, mol.coeff, r_basis]

    g_r = autograd.jacobian(
        qchem.hf.attraction_matrix(mol.basis_set, mol.nuclear_charges, mol.coordinates), argnum=0
    )(*args)
    assert np.allclose(g_r, g_r_ref)


@pytest.mark.parametrize(
    ("symbols", "geometry", "alpha", "e_ref"),
    [
        (
            ["H", "H"],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], requires_grad=False),
            np.array(
                [[3.42525091, 0.62391373, 0.1688554], [3.42525091, 0.62391373, 0.1688554]],
                requires_grad=True,
            ),
            # electron repulsion tensor obtained from pyscf with mol.intor('int2e')
            np.array(
                [
                    [
                        [[0.77460594, 0.56886157], [0.56886157, 0.65017755]],
                        [[0.56886157, 0.45590169], [0.45590169, 0.56886157]],
                    ],
                    [
                        [[0.56886157, 0.45590169], [0.45590169, 0.56886157]],
                        [[0.65017755, 0.56886157], [0.56886157, 0.77460594]],
                    ],
                ]
            ),
        )
    ],
)
def test_repulsion_tensor(symbols, geometry, alpha, e_ref):
    r"""Test that repulsion_tensor returns the correct matrix."""
    mol = qchem.hf.Molecule(symbols, geometry, alpha=alpha)
    args = [mol.alpha]
    e = qchem.hf.repulsion_tensor(mol.basis_set)(*args)
    assert np.allclose(e, e_ref)


@pytest.mark.parametrize(
    ("symbols", "geometry", "e_ref"),
    [
        (
            ["H", "H"],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], requires_grad=False),
            # electron repulsion tensor obtained from pyscf with mol.intor('int2e')
            np.array(
                [
                    [
                        [[0.77460594, 0.56886157], [0.56886157, 0.65017755]],
                        [[0.56886157, 0.45590169], [0.45590169, 0.56886157]],
                    ],
                    [
                        [[0.56886157, 0.45590169], [0.45590169, 0.56886157]],
                        [[0.65017755, 0.56886157], [0.56886157, 0.77460594]],
                    ],
                ]
            ),
        )
    ],
)
def test_repulsion_tensor_nodiff(symbols, geometry, e_ref):
    r"""Test that repulsion_tensor returns the correct matrix when no differentiable parameter is
    used."""
    mol = qchem.hf.Molecule(symbols, geometry)
    e = qchem.hf.repulsion_tensor(mol.basis_set)()
    assert np.allclose(e, e_ref)


@pytest.mark.parametrize(
    ("symbols", "geometry", "alpha", "c_ref"),
    [
        (
            ["H", "H"],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], requires_grad=False),
            np.array(
                [[3.42525091, 0.62391373, 0.1688554], [3.42525091, 0.62391373, 0.1688554]],
                requires_grad=True,
            ),
            # core matrix obtained from pyscf using scf.RHF(mol).get_hcore()
            np.array(
                [
                    [-1.27848886, -1.21916326],
                    [-1.21916326, -1.27848886],
                ]
            ),
        )
    ],
)
def test_core_matrix(symbols, geometry, alpha, c_ref):
    r"""Test that core_matrix returns the correct matrix."""
    mol = qchem.hf.Molecule(symbols, geometry, alpha=alpha)
    args = [mol.alpha]
    c = qchem.hf.core_matrix(mol.basis_set, mol.nuclear_charges, mol.coordinates)(*args)
    assert np.allclose(c, c_ref)


@pytest.mark.parametrize(
    ("symbols", "geometry", "c_ref"),
    [
        (
            ["H", "H"],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], requires_grad=False),
            # core matrix obtained from pyscf using scf.RHF(mol).get_hcore()
            np.array(
                [
                    [-1.27848886, -1.21916326],
                    [-1.21916326, -1.27848886],
                ]
            ),
        )
    ],
)
def test_core_matrix_nodiff(symbols, geometry, c_ref):
    r"""Test that core_matrix returns the correct matrix when no differentiable parameter is
    used."""
    mol = qchem.hf.Molecule(symbols, geometry)
    c = qchem.hf.core_matrix(mol.basis_set, mol.nuclear_charges, mol.coordinates)()
    assert np.allclose(c, c_ref)


@pytest.mark.parametrize(
    ("symbols", "geometry", "alpha", "c_ref"),
    [
        (
            ["H", "H"],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], requires_grad=True),
            np.array(
                [[3.42525091, 0.62391373, 0.1688554], [3.42525091, 0.62391373, 0.1688554]],
                requires_grad=True,
            ),
            # core matrix obtained from pyscf using scf.RHF(mol).get_hcore()
            np.array(
                [
                    [-1.27848886, -1.21916326],
                    [-1.21916326, -1.27848886],
                ]
            ),
        )
    ],
)
def test_core_matrix_diff_positions(symbols, geometry, alpha, c_ref):
    r"""Test that core_matrix returns the correct matrix when positions are differentiable."""
    mol = qchem.hf.Molecule(symbols, geometry, alpha=alpha)
    r_basis = mol.coordinates
    args = [mol.coordinates, mol.alpha, r_basis]
    c = qchem.hf.core_matrix(mol.basis_set, mol.nuclear_charges, mol.coordinates)(*args)
    assert np.allclose(c, c_ref)
