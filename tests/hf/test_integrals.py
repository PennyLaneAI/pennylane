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
Unit tests for functions needed to computing integrals over basis functions.
"""
import autograd
import pytest
from pennylane import numpy as np
from pennylane.hf.integrals import (
    _boys,
    _diff2,
    _generate_params,
    _hermite_coulomb,
    contracted_norm,
    expansion,
    gaussian_kinetic,
    gaussian_overlap,
    generate_attraction,
    generate_kinetic,
    generate_overlap,
    generate_repulsion,
    primitive_norm,
)
from pennylane.hf.molecule import Molecule


@pytest.mark.parametrize(
    ("l", "alpha", "n"),
    [
        # normalization constant for an s orbital is :math:`(\frac {2 \alpha}{\pi})^{{3/4}}`.
        ((0, 0, 0), np.array([3.425250914]), np.array([1.79444183])),
    ],
)
def test_gaussian_norm(l, alpha, n):
    r"""Test that the computed normalization constant of a Gaussian function is correct."""
    assert np.allclose(primitive_norm(l, alpha), n)


@pytest.mark.parametrize(
    ("l", "alpha", "a", "n"),
    [
        # normalization constant for a contracted Gaussian function composed of three normalized
        # s orbital is :math:`1/3`.
        (
            (0, 0, 0),
            np.array([3.425250914, 3.425250914, 3.425250914]),
            np.array([1.79444183, 1.79444183, 1.79444183]),
            np.array([0.33333333]),
        )
    ],
)
def test_contraction_norm(l, alpha, a, n):
    r"""Test that the computed normalization constant of a contracted Gaussian function is correct."""
    assert np.allclose(contracted_norm(l, alpha, a), n)


@pytest.mark.parametrize(
    ("alpha", "coeff", "r"),
    [
        (
            np.array([3.42525091, 0.62391373, 0.1688554], requires_grad=True),
            np.array([0.15432897, 0.53532814, 0.44463454], requires_grad=True),
            np.array([0.0, 0.0, 0.0], requires_grad=False),
        ),
        (
            np.array([3.42525091, 0.62391373, 0.1688554], requires_grad=False),
            np.array([0.15432897, 0.53532814, 0.44463454], requires_grad=False),
            np.array([0.0, 0.0, 0.0], requires_grad=True),
        ),
    ],
)
def test_generate_params(alpha, coeff, r):
    r"""Test that test_generate_params returns correct basis set parameters."""
    params = [alpha, coeff, r]
    args = [p for p in [alpha, coeff, r] if p.requires_grad]
    basis_params = _generate_params(params, args)

    assert np.allclose(basis_params, (alpha, coeff, r))


@pytest.mark.parametrize(
    ("la", "lb", "ra", "rb", "alpha", "beta", "t", "c"),
    [
        (
            0,
            0,
            np.array([1.2]),
            np.array([1.2]),
            np.array([3.42525091]),
            np.array([3.42525091]),
            0,
            np.array([1.0]),
        ),
        (
            1,
            0,
            np.array([0.0]),
            np.array([0.0]),
            np.array([3.42525091]),
            np.array([3.42525091]),
            0,
            np.array([0.0]),
        ),
        (
            1,
            1,
            np.array([0.0]),
            np.array([10.0]),
            np.array([3.42525091]),
            np.array([3.42525091]),
            0,
            np.array([0.0]),
        ),
    ],
)
def test_expansion(la, lb, ra, rb, alpha, beta, t, c):
    r"""Test that expansion function returns correct value."""
    assert np.allclose(expansion(la, lb, ra, rb, alpha, beta, t), c)
    assert np.allclose(expansion(la, lb, ra, rb, alpha, beta, -1), np.array([0.0]))
    assert np.allclose(expansion(0, 1, ra, rb, alpha, beta, 2), np.array([0.0]))


@pytest.mark.parametrize(
    ("la", "lb", "ra", "rb", "alpha", "beta", "o"),
    [
        # two normalized s orbitals
        (
            (0, 0, 0),
            (0, 0, 0),
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0]),
            np.array([np.pi / 2]),
            np.array([np.pi / 2]),
            np.array([1.0]),
        ),
        (
            (0, 0, 0),
            (0, 0, 0),
            np.array([0.0, 0.0, 0.0]),
            np.array([20.0, 0.0, 0.0]),
            np.array([3.42525091]),
            np.array([3.42525091]),
            np.array([0.0]),
        ),
        (
            (1, 0, 0),
            (0, 0, 1),
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0]),
            np.array([6.46480325]),
            np.array([6.46480325]),
            np.array([0.0]),
        ),
    ],
)
def test_gaussian_overlap(la, lb, ra, rb, alpha, beta, o):
    r"""Test that gaussian overlap function returns a correct value."""
    assert np.allclose(gaussian_overlap(la, lb, ra, rb, alpha, beta), o)


@pytest.mark.parametrize(
    ("symbols", "geometry", "alpha", "coef", "r", "o_ref"),
    [
        (
            ["H", "H"],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 20.0]], requires_grad=False),
            np.array(
                [[3.42525091, 0.62391373, 0.1688554], [3.42525091, 0.62391373, 0.1688554]],
                requires_grad=False,
            ),
            np.array(
                [[0.15432897, 0.53532814, 0.44463454], [0.15432897, 0.53532814, 0.44463454]],
                requires_grad=True,
            ),
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 20.0]], requires_grad=True),
            np.array([0.0]),
        ),
        (
            ["H", "H"],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], requires_grad=False),
            np.array(
                [[3.42525091, 0.62391373, 0.1688554], [3.42525091, 0.62391373, 0.1688554]],
                requires_grad=False,
            ),
            np.array(
                [[0.15432897, 0.53532814, 0.44463454], [0.15432897, 0.53532814, 0.44463454]],
                requires_grad=False,
            ),
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], requires_grad=True),
            np.array([1.0]),
        ),
        (
            ["H", "H"],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], requires_grad=True),
            np.array(
                [[3.42525091, 0.62391373, 0.1688554], [3.42525091, 0.62391373, 0.1688554]],
                requires_grad=True,
            ),
            np.array(
                [[0.15432897, 0.53532814, 0.44463454], [0.15432897, 0.53532814, 0.44463454]],
                requires_grad=True,
            ),
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], requires_grad=True),
            np.array([1.0]),
        ),
    ],
)
def test_generate_overlap(symbols, geometry, alpha, coef, r, o_ref):
    r"""Test that generate_overlap function returns a correct value for the overlap integral."""
    mol = Molecule(symbols, geometry)
    basis_a = mol.basis_set[0]
    basis_b = mol.basis_set[1]
    args = [p for p in [alpha, coef, r] if p.requires_grad]

    o = generate_overlap(basis_a, basis_b)(*args)
    assert np.allclose(o, o_ref)


@pytest.mark.parametrize(
    ("symbols", "geometry", "alpha", "coeff"),
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
        ),
    ],
)
def test_gradient(symbols, geometry, alpha, coeff):
    r"""Test that the overlap gradient computed with respect to the basis parameters is correct."""
    mol = Molecule(symbols, geometry, alpha=alpha, coeff=coeff)
    basis_a = mol.basis_set[0]
    basis_b = mol.basis_set[1]
    args = [mol.alpha, mol.coeff]

    g_alpha = autograd.grad(generate_overlap(basis_a, basis_b), argnum=0)(*args)
    g_coeff = autograd.grad(generate_overlap(basis_a, basis_b), argnum=1)(*args)

    # compute overlap gradients with respect to alpha and coeff using finite diff
    delta = 0.0001
    g_ref_alpha = np.zeros(6).reshape(alpha.shape)
    g_ref_coeff = np.zeros(6).reshape(coeff.shape)

    for i in range(len(alpha)):
        for j in range(len(alpha[0])):

            alpha_minus = alpha.copy()
            alpha_plus = alpha.copy()
            alpha_minus[i][j] = alpha_minus[i][j] - delta
            alpha_plus[i][j] = alpha_plus[i][j] + delta
            o_minus = generate_overlap(basis_a, basis_b)(*[alpha_minus, coeff])
            o_plus = generate_overlap(basis_a, basis_b)(*[alpha_plus, coeff])
            g_ref_alpha[i][j] = (o_plus - o_minus) / (2 * delta)

            coeff_minus = coeff.copy()
            coeff_plus = coeff.copy()
            coeff_minus[i][j] = coeff_minus[i][j] - delta
            coeff_plus[i][j] = coeff_plus[i][j] + delta
            o_minus = generate_overlap(basis_a, basis_b)(*[alpha, coeff_minus])
            o_plus = generate_overlap(basis_a, basis_b)(*[alpha, coeff_plus])
            g_ref_coeff[i][j] = (o_plus - o_minus) / (2 * delta)

    assert np.allclose(g_alpha, g_ref_alpha)
    assert np.allclose(g_coeff, g_ref_coeff)


@pytest.mark.parametrize(
    ("i", "j", "ri", "rj", "alpha", "beta", "d"),
    [
        # _diff2 must return 0.0 for two Gaussians centered far apart at 0.0 and 20.0
        (
            0,
            1,
            np.array([0.0]),
            np.array([20.0]),
            np.array([3.42525091]),
            np.array([3.42525091]),
            np.array([0.0]),
        ),
        # computed manually
        (
            0,
            0,
            np.array([0.0]),
            np.array([1.0]),
            np.array([3.42525091]),
            np.array([3.42525091]),
            np.array([1.01479665]),
        ),
    ],
)
def test_diff2(i, j, ri, rj, alpha, beta, d):
    r"""Test that _diff2 function returns a correct value."""
    assert np.allclose(_diff2(i, j, ri, rj, alpha, beta), d)


@pytest.mark.parametrize(
    ("la", "lb", "ra", "rb", "alpha", "beta", "t"),
    [
        # gaussian_kinetic must return 0.0 for two Gaussians centered far apart
        (
            (0, 0, 0),
            (0, 0, 0),
            np.array([0.0, 0.0, 0.0]),
            np.array([20.0, 0.0, 0.0]),
            np.array([3.42525091]),
            np.array([3.42525091]),
            np.array([0.0]),
        ),
    ],
)
def test_gaussian_kinetic(la, lb, ra, rb, alpha, beta, t):
    r"""Test that gaussian_kinetic function returns a correct value."""
    assert np.allclose(gaussian_kinetic(la, lb, ra, rb, alpha, beta), t)


@pytest.mark.parametrize(
    ("symbols", "geometry", "alpha", "coeff", "t_ref"),
    [
        # generate_kinetic must return 0.0 for two Gaussians centered far apart
        (
            ["H", "H"],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 20.0]], requires_grad=False),
            np.array(
                [[3.42525091, 0.62391373, 0.1688554], [3.42525091, 0.62391373, 0.1688554]],
                requires_grad=False,
            ),
            np.array(
                [[0.15432897, 0.53532814, 0.44463454], [0.15432897, 0.53532814, 0.44463454]],
                requires_grad=True,
            ),
            np.array([0.0]),
        ),
        # kinetic integral obtained from pyscf using mol.intor('int1e_kin')
        (
            ["H", "H"],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], requires_grad=False),
            np.array(
                [[3.42525091, 0.62391373, 0.1688554], [3.42525091, 0.62391373, 0.1688554]],
                requires_grad=False,
            ),
            np.array(
                [[0.15432897, 0.53532814, 0.44463454], [0.15432897, 0.53532814, 0.44463454]],
                requires_grad=True,
            ),
            np.array([0.38325384]),
        ),
    ],
)
def test_generate_kinetic(symbols, geometry, alpha, coeff, t_ref):
    r"""Test that generate_kinetic function returns a correct value for the kinetic integral."""
    mol = Molecule(symbols, geometry, alpha=alpha, coeff=coeff)
    basis_a = mol.basis_set[0]
    basis_b = mol.basis_set[1]
    args = [p for p in [alpha, coeff] if p.requires_grad]

    t = generate_kinetic(basis_a, basis_b)(*args)
    assert np.allclose(t, t_ref)


@pytest.mark.parametrize(
    ("symbols", "geometry", "alpha", "coeff"),
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
        ),
    ],
)
def test_gradient_kinetic(symbols, geometry, alpha, coeff):
    r"""Test that the kinetic gradient computed with respect to the basis parameters is correct."""
    mol = Molecule(symbols, geometry, alpha=alpha, coeff=coeff)
    basis_a = mol.basis_set[0]
    basis_b = mol.basis_set[1]
    args = [mol.alpha, mol.coeff]

    g_alpha = autograd.grad(generate_kinetic(basis_a, basis_b), argnum=0)(*args)
    g_coeff = autograd.grad(generate_kinetic(basis_a, basis_b), argnum=1)(*args)

    # compute kinetic gradients with respect to alpha, coeff and r using finite diff
    delta = 0.0001
    g_ref_alpha = np.zeros(6).reshape(alpha.shape)
    g_ref_coeff = np.zeros(6).reshape(coeff.shape)

    for i in range(len(alpha)):
        for j in range(len(alpha[0])):

            alpha_minus = alpha.copy()
            alpha_plus = alpha.copy()
            alpha_minus[i][j] = alpha_minus[i][j] - delta
            alpha_plus[i][j] = alpha_plus[i][j] + delta
            t_minus = generate_kinetic(basis_a, basis_b)(*[alpha_minus, coeff])
            t_plus = generate_kinetic(basis_a, basis_b)(*[alpha_plus, coeff])
            g_ref_alpha[i][j] = (t_plus - t_minus) / (2 * delta)

            coeff_minus = coeff.copy()
            coeff_plus = coeff.copy()
            coeff_minus[i][j] = coeff_minus[i][j] - delta
            coeff_plus[i][j] = coeff_plus[i][j] + delta
            t_minus = generate_kinetic(basis_a, basis_b)(*[alpha, coeff_minus])
            t_plus = generate_kinetic(basis_a, basis_b)(*[alpha, coeff_plus])
            g_ref_coeff[i][j] = (t_plus - t_minus) / (2 * delta)

    assert np.allclose(g_alpha, g_ref_alpha)
    assert np.allclose(g_coeff, g_ref_coeff)


@pytest.mark.parametrize(
    ("symbols", "geometry", "alpha", "coeff", "a_ref"),
    [
        # trivial case: integral should be zero since atoms are located very far apart
        (
            ["H", "H"],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 20.0]], requires_grad=False),
            np.array(
                [[3.42525091, 0.62391373, 0.1688554], [3.42525091, 0.62391373, 0.1688554]],
                requires_grad=True,
            ),
            np.array(
                [[0.15432897, 0.53532814, 0.44463454], [0.15432897, 0.53532814, 0.44463454]],
                requires_grad=True,
            ),
            np.array([0.0]),
        ),
        # nuclear attraction integral obtained from pyscf using mol.intor('int1e_nuc')
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
            np.array([0.80120855]),
        ),
    ],
)
def test_generate_attraction(symbols, geometry, alpha, coeff, a_ref):
    r"""Test that generate_attraction function returns a correct value for the kinetic integral."""
    mol = Molecule(symbols, geometry, alpha=alpha, coeff=coeff)
    basis_a = mol.basis_set[0]
    basis_b = mol.basis_set[1]
    args = [p for p in [alpha, coeff] if p.requires_grad]

    if geometry.requires_grad:
        args = [geometry[0]] + args + [geometry]

    a = generate_attraction(geometry[0], basis_a, basis_b)(*args)
    assert np.allclose(a, a_ref)


@pytest.mark.parametrize(
    ("symbols", "geometry", "alpha", "coeff"),
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
        ),
    ],
)
def test_gradient_attraction(symbols, geometry, alpha, coeff):
    r"""Test that the attraction gradient computed with respect to the basis parameters is correct."""
    mol = Molecule(symbols, geometry, alpha=alpha, coeff=coeff)
    basis_a = mol.basis_set[0]
    basis_b = mol.basis_set[1]
    args = [mol.alpha, mol.coeff]
    r_nuc = geometry[0]

    g_alpha = autograd.grad(generate_attraction(r_nuc, basis_a, basis_b), argnum=0)(*args)
    g_coeff = autograd.grad(generate_attraction(r_nuc, basis_a, basis_b), argnum=1)(*args)

    # compute attraction gradients with respect to alpha and coeff using finite diff
    delta = 0.0001
    g_ref_alpha = np.zeros(6).reshape(alpha.shape)
    g_ref_coeff = np.zeros(6).reshape(coeff.shape)

    for i in range(len(alpha)):
        for j in range(len(alpha[0])):

            alpha_minus = alpha.copy()
            alpha_plus = alpha.copy()
            alpha_minus[i][j] = alpha_minus[i][j] - delta
            alpha_plus[i][j] = alpha_plus[i][j] + delta
            a_minus = generate_attraction(r_nuc, basis_a, basis_b)(*[alpha_minus, coeff])
            a_plus = generate_attraction(r_nuc, basis_a, basis_b)(*[alpha_plus, coeff])
            g_ref_alpha[i][j] = (a_plus - a_minus) / (2 * delta)

            coeff_minus = coeff.copy()
            coeff_plus = coeff.copy()
            coeff_minus[i][j] = coeff_minus[i][j] - delta
            coeff_plus[i][j] = coeff_plus[i][j] + delta
            a_minus = generate_attraction(r_nuc, basis_a, basis_b)(*[alpha, coeff_minus])
            a_plus = generate_attraction(r_nuc, basis_a, basis_b)(*[alpha, coeff_plus])
            g_ref_coeff[i][j] = (a_plus - a_minus) / (2 * delta)

    assert np.allclose(g_alpha, g_ref_alpha)
    assert np.allclose(g_coeff, g_ref_coeff)


@pytest.mark.parametrize(
    ("symbols", "geometry", "alpha", "coeff", "e_ref"),
    [
        (
            ["H", "H"],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 20.0]], requires_grad=False),
            np.array(
                [
                    [3.42525091, 0.62391373, 0.1688554],
                    [3.42525091, 0.62391373, 0.1688554],
                    [3.42525091, 0.62391373, 0.1688554],
                    [3.42525091, 0.62391373, 0.1688554],
                ],
                requires_grad=False,
            ),
            np.array(
                [
                    [0.15432897, 0.53532814, 0.44463454],
                    [0.15432897, 0.53532814, 0.44463454],
                    [0.15432897, 0.53532814, 0.44463454],
                    [0.15432897, 0.53532814, 0.44463454],
                ],
                requires_grad=True,
            ),
            np.array([0.0]),
        ),
        (
            ["H", "H"],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], requires_grad=False),
            np.array(
                [
                    [3.42525091, 0.62391373, 0.1688554],
                    [3.42525091, 0.62391373, 0.1688554],
                    [3.42525091, 0.62391373, 0.1688554],
                    [3.42525091, 0.62391373, 0.1688554],
                ],
                requires_grad=False,
            ),
            np.array(
                [
                    [0.15432897, 0.53532814, 0.44463454],
                    [0.15432897, 0.53532814, 0.44463454],
                    [0.15432897, 0.53532814, 0.44463454],
                    [0.15432897, 0.53532814, 0.44463454],
                ],
                requires_grad=True,
            ),
            np.array([0.45590169]),
        ),
    ],
)
def test_generate_repulsion(symbols, geometry, alpha, coeff, e_ref):
    r"""Test that generate_repulsion function returns a correct value for the repulsion integral."""
    mol = Molecule(symbols, geometry, alpha=alpha, coeff=coeff)
    basis_a = mol.basis_set[0]
    basis_b = mol.basis_set[1]
    args = [p for p in [alpha, coeff] if p.requires_grad]

    a = generate_repulsion(basis_a, basis_b, basis_a, basis_b)(*args)

    assert np.allclose(a, e_ref)


@pytest.mark.parametrize(
    ("symbols", "geometry", "alpha", "coeff"),
    [
        (
            ["H", "H"],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], requires_grad=False),
            np.array(
                [
                    [3.42525091, 0.62391373, 0.1688554],
                    [3.42525091, 0.62391373, 0.1688554],
                    [3.42525091, 0.62391373, 0.1688554],
                    [3.42525091, 0.62391373, 0.1688554],
                ],
                requires_grad=True,
            ),
            np.array(
                [
                    [0.15432897, 0.53532814, 0.44463454],
                    [0.15432897, 0.53532814, 0.44463454],
                    [0.15432897, 0.53532814, 0.44463454],
                    [0.15432897, 0.53532814, 0.44463454],
                ],
                requires_grad=True,
            ),
        ),
    ],
)
def test_gradient_repulsion(symbols, geometry, alpha, coeff):
    r"""Test that the repulsion gradient computed with respect to the basis parameters is correct."""
    mol = Molecule(symbols, geometry, alpha=alpha, coeff=coeff)
    basis_a = mol.basis_set[0]
    basis_b = mol.basis_set[1]
    args = [mol.alpha, mol.coeff]

    g_alpha = autograd.grad(generate_repulsion(basis_a, basis_b, basis_a, basis_b), argnum=0)(*args)
    g_coeff = autograd.grad(generate_repulsion(basis_a, basis_b, basis_a, basis_b), argnum=1)(*args)

    # compute repulsion gradients with respect to alpha and coeff using finite diff
    delta = 0.0001
    g_ref_alpha = np.zeros(12).reshape(alpha.shape)
    g_ref_coeff = np.zeros(12).reshape(coeff.shape)

    for i in range(len(alpha)):
        for j in range(len(alpha[0])):

            alpha_minus = alpha.copy()
            alpha_plus = alpha.copy()
            alpha_minus[i][j] = alpha_minus[i][j] - delta
            alpha_plus[i][j] = alpha_plus[i][j] + delta
            e_minus = generate_repulsion(basis_a, basis_b, basis_a, basis_b)(*[alpha_minus, coeff])
            e_plus = generate_repulsion(basis_a, basis_b, basis_a, basis_b)(*[alpha_plus, coeff])
            g_ref_alpha[i][j] = (e_plus - e_minus) / (2 * delta)

            coeff_minus = coeff.copy()
            coeff_plus = coeff.copy()
            coeff_minus[i][j] = coeff_minus[i][j] - delta
            coeff_plus[i][j] = coeff_plus[i][j] + delta
            e_minus = generate_repulsion(basis_a, basis_b, basis_a, basis_b)(*[alpha, coeff_minus])
            e_plus = generate_repulsion(basis_a, basis_b, basis_a, basis_b)(*[alpha, coeff_plus])
            g_ref_coeff[i][j] = (e_plus - e_minus) / (2 * delta)

    assert np.allclose(g_alpha, g_ref_alpha)
    assert np.allclose(g_coeff, g_ref_coeff)


@pytest.mark.parametrize(
    ("n", "t", "f_ref"),
    [(2.75, np.array([0.0, 1.23]), np.array([0.15384615384615385, 0.061750771828252976]))],
)
def test_boys(n, t, f_ref):
    r"""Test that the Boys function is evaluated correctly."""
    f = _boys(n, t)
    assert np.allclose(f, f_ref)


@pytest.mark.parametrize(
    ("t", "u", "v", "n", "p", "dr", "h_ref"),
    [
        (0, 0, 0, 0, 6.85050183, np.array([0.0, 0.0, 0.0]), 1.0),
        (0, 0, 1, 0, 6.85050183, np.array([0.0, 0.0, 0.0]), 0.0),
        (0, 0, 2, 0, 6.85050183, np.array([0.0, 0.0, 0.0]), -4.56700122),
        (0, 2, 0, 0, 6.85050183, np.array([0.0, 0.0, 0.0]), -4.56700122),
        (0, 1, 0, 0, 6.85050183, np.array([0.0, 0.0, 0.0]), 0.0),
        (2, 1, 0, 0, 6.85050183, np.array([0.0, 0.0, 0.0]), 0.0),
        (1, 1, 0, 0, 6.85050183, np.array([0.0, 0.0, 0.0]), 0.0),
    ],
)
def test_hermite_coulomb(t, u, v, n, p, dr, h_ref):
    r"""Test that the _hermite_coulomb function returns a correct value."""
    h = _hermite_coulomb(t, u, v, n, p, dr)
    assert np.allclose(h, h_ref)
