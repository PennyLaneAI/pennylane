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
from pennylane.hf.matrices import kinetic_matrix, molecular_density_matrix, overlap_matrix
from pennylane.hf.molecule import Molecule


@pytest.mark.parametrize(
    ("n_electron", "c", "p_ref"),
    [
        # all P elements are 0.54828771**2 = 0.3006194129370441
        (
            2,
            np.array([[-0.54828771, 1.21848441], [-0.54828771, -1.21848441]]),
            np.array([[0.30061941, 0.30061941], [0.30061941, 0.30061941]]),
        ),
    ],
)
def test_molecular_density_matrix(n_electron, c, p_ref):
    r"""Test that molecular_density_matrix returns the correct matrix."""
    p = molecular_density_matrix(n_electron, c)
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
    mol = Molecule(symbols, geometry, alpha=alpha)
    args = [alpha]
    s = overlap_matrix(mol.basis_set)(*args)
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
    mol = Molecule(symbols, geometry)
    s = overlap_matrix(mol.basis_set)()
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
    mol = Molecule(symbols, geometry, alpha=alpha, coeff=coeff)
    args = [mol.alpha, mol.coeff]
    g_alpha = autograd.jacobian(overlap_matrix(mol.basis_set), argnum=0)(*args)
    g_coeff = autograd.jacobian(overlap_matrix(mol.basis_set), argnum=1)(*args)
    assert np.allclose(g_alpha, g_alpha_ref)
    assert np.allclose(g_coeff, g_coeff_ref)


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
            # Jacobian matrix contains gradient of K11, K12, K21, K22 wrt arg_1, arg_2.
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
    mol = Molecule(symbols, geometry, alpha=alpha, coeff=coeff)
    args = [mol.alpha, mol.coeff]
    g_alpha = autograd.jacobian(kinetic_matrix(mol.basis_set), argnum=0)(*args)
    g_coeff = autograd.jacobian(kinetic_matrix(mol.basis_set), argnum=1)(*args)
    assert np.allclose(g_alpha, g_alpha_ref)
    assert np.allclose(g_coeff, g_coeff_ref)
