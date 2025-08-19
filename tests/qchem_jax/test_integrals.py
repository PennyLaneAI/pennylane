# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

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
import pytest


import numpy as np

from pennylane.qchem import Molecule
from pennylane import qchem_jax as qchem


import jax
from jax import numpy as jnp
from jax import config

config.update("jax_enable_x64", True)  # Need this for better precision


@pytest.mark.parametrize(
    ("l", "alpha", "n"),
    [
        ((0, 0, 0), jnp.array([3.425250914]), jnp.array([1.79444183])),
    ],
)
def test_primitive_norm(l, alpha, n):
    """Tests primitive norm calculation and if it can be JIT compiled."""
    n_computed = qchem.integrals.primitive_norm(l, alpha)
    assert np.allclose(n_computed, n)

    # JIT
    n_computed_jit = jax.jit(qchem.integrals.primitive_norm, static_argnums=0)(l, alpha)
    assert np.allclose(n_computed_jit, n)


@pytest.mark.parametrize(
    ("l", "alpha", "a", "n"),
    [
        # normalization constant for a contracted Gaussian function composed of three normalized
        # s orbital is :math:`1/3`.
        (
            (0, 0, 0),
            jnp.array([3.425250914, 3.425250914, 3.425250914]),
            jnp.array([1.79444183, 1.79444183, 1.79444183]),
            jnp.array([0.33333333]),
        )
    ],
)
def test_contraction_norm(l, alpha, a, n):
    r"""Test that the computed normalization constant of a contracted Gaussian function is
    correct and it can be JIT compiled."""
    assert np.allclose(qchem.contracted_norm(l, alpha, a), n)
    assert np.allclose(jax.jit(qchem.contracted_norm, static_argnums=0)(l, alpha, a), n)


@pytest.mark.parametrize(
    ("la", "lb", "ra", "rb", "alpha", "beta", "t", "c"),
    [
        (
            0,
            0,
            jnp.array([1.2]),
            jnp.array([1.2]),
            jnp.array([3.42525091]),
            jnp.array([3.42525091]),
            0,
            jnp.array([1.0]),
        ),
        (
            1,
            0,
            jnp.array([0.0]),
            jnp.array([0.0]),
            jnp.array([3.42525091]),
            jnp.array([3.42525091]),
            0,
            jnp.array([0.0]),
        ),
        (
            1,
            1,
            jnp.array([0.0]),
            jnp.array([10.0]),
            jnp.array([3.42525091]),
            jnp.array([3.42525091]),
            0,
            jnp.array([0.0]),
        ),
    ],
)
def test_expansion(la, lb, ra, rb, alpha, beta, t, c):
    r"""Test that expansion function returns correct value."""
    assert np.allclose(qchem.expansion(la, lb, ra, rb, alpha, beta, t), c)
    assert np.allclose(qchem.expansion(la, lb, ra, rb, alpha, beta, -1), jnp.array([0.0]))
    assert np.allclose(qchem.expansion(0, 1, ra, rb, alpha, beta, 2), jnp.array([0.0]))

    # JIT - Note that changing the angular momenta will trigger recompilation
    assert np.allclose(
        jax.jit(qchem.expansion, static_argnums=(0, 1, 6))(la, lb, ra, rb, alpha, beta, t), c
    )
    assert np.allclose(
        jax.jit(qchem.expansion, static_argnums=(0, 1, 6))(la, lb, ra, rb, alpha, beta, -1),
        jnp.array([0.0]),
    )
    assert np.allclose(
        jax.jit(qchem.expansion, static_argnums=(0, 1, 6))(0, 1, ra, rb, alpha, beta, 2),
        jnp.array([0.0]),
    )


@pytest.mark.parametrize(
    ("la", "lb", "ra", "rb", "alpha", "beta", "o"),
    [
        # two normalized s orbitals
        (
            (0, 0, 0),
            (0, 0, 0),
            jnp.array([0.0, 0.0, 0.0]),
            jnp.array([0.0, 0.0, 0.0]),
            jnp.array([jnp.pi / 2]),
            jnp.array([jnp.pi / 2]),
            jnp.array([1.0]),
        ),
        (
            (0, 0, 0),
            (0, 0, 0),
            jnp.array([0.0, 0.0, 0.0]),
            jnp.array([20.0, 0.0, 0.0]),
            jnp.array([3.42525091]),
            jnp.array([3.42525091]),
            jnp.array([0.0]),
        ),
        (
            (1, 0, 0),
            (0, 0, 1),
            jnp.array([0.0, 0.0, 0.0]),
            jnp.array([0.0, 0.0, 0.0]),
            jnp.array([6.46480325]),
            jnp.array([6.46480325]),
            jnp.array([0.0]),
        ),
    ],
)
def test_gaussian_overlap(la, lb, ra, rb, alpha, beta, o):
    r"""Test that gaussian overlap function returns a correct value and can be JIT compiled."""
    assert np.allclose(qchem.gaussian_overlap(la, lb, ra, rb, alpha, beta), o)
    assert np.allclose(
        jax.jit(qchem.gaussian_overlap, static_argnums=(0, 1))(la, lb, ra, rb, alpha, beta), o
    )


@pytest.mark.parametrize(
    ("symbols", "geometry", "alphas", "coeffs", "rs", "o_ref"),
    [
        (
            ["H", "H"],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 20.0]]),
            np.array([[3.42525091, 0.62391373, 0.1688554], [3.42525091, 0.62391373, 0.1688554]]),
            np.array([[0.15432897, 0.53532814, 0.44463454], [0.15432897, 0.53532814, 0.44463454]]),
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 20.0]]),
            np.array([0.0]),
        ),
        (
            ["H", "H"],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            np.array([[3.42525091, 0.62391373, 0.1688554], [3.42525091, 0.62391373, 0.1688554]]),
            np.array([[0.15432897, 0.53532814, 0.44463454], [0.15432897, 0.53532814, 0.44463454]]),
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            np.array([1.0]),
        ),
        (
            ["H", "H"],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            np.array([[3.42525091, 0.62391373, 0.1688554], [3.42525091, 0.62391373, 0.1688554]]),
            np.array([[0.15432897, 0.53532814, 0.44463454], [0.15432897, 0.53532814, 0.44463454]]),
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            np.array([1.0]),
        ),
    ],
)
def test_overlap_integral(symbols, geometry, alphas, coeffs, rs, o_ref):
    r"""Test that overlap_integral function returns a correct value for the overlap integral."""
    mol = Molecule(symbols, geometry)

    basis_a = mol.basis_set[0]
    basis_b = mol.basis_set[1]

    o = qchem.overlap_integral(alphas, coeffs, rs, basis_a, basis_b)
    assert np.allclose(o, o_ref)

    # JIT version - this will get tricky perhaps
    overlap_integral_jitted = jax.jit(qchem.overlap_integral, static_argnums=(3, 4))
    o_jitted = overlap_integral_jitted(alphas, coeffs, rs, basis_a, basis_b)
    assert np.allclose(o_jitted, o_ref)


@pytest.mark.parametrize(
    ("symbols", "geometry", "alphas", "coeffs"),
    [
        (
            ["H", "H"],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),  # Assume non differentiable
            np.array([[3.42525091, 0.62391373, 0.1688554], [3.42525091, 0.62391373, 0.1688554]]),
            np.array([[0.15432897, 0.53532814, 0.44463454], [0.15432897, 0.53532814, 0.44463454]]),
        ),
    ],
)
def test_gradient_overlap(symbols, geometry, alphas, coeffs):
    r"""Test that the overlap gradient computed with respect to the basis parameters is
    correct."""
    mol = Molecule(symbols, geometry, alpha=alphas, coeff=coeffs)

    basis_a = mol.basis_set[0]
    basis_b = mol.basis_set[1]

    rs = jnp.array(mol.r)  # We keep the signature of the overlap integrals the same

    g_alpha = jax.grad(qchem.overlap_integral, argnums=0)(alphas, coeffs, rs, basis_a, basis_b)
    g_coeff = jax.grad(qchem.overlap_integral, argnums=1)(alphas, coeffs, rs, basis_a, basis_b)

    # compute overlap gradients with respect to alpha and coeff using finite diff
    delta = 0.00001  # If delta is too small this will not work
    g_ref_alpha = np.zeros(6).reshape(alphas.shape)
    g_ref_coeff = np.zeros(6).reshape(coeffs.shape)

    for i in range(len(alphas)):
        for j in range(len(alphas[0])):

            alpha_minus = alphas.copy()
            alpha_plus = alphas.copy()

            alpha_minus[i][j] = alpha_minus[i][j] - delta
            alpha_plus[i][j] = alpha_plus[i][j] + delta

            o_minus = qchem.overlap_integral(alpha_minus, coeffs, rs, basis_a, basis_b)
            o_plus = qchem.overlap_integral(alpha_plus, coeffs, rs, basis_a, basis_b)

            g_ref_alpha[i][j] = (o_plus - o_minus) / (2 * delta)

            coeff_minus = coeffs.copy()
            coeff_plus = coeffs.copy()

            coeff_minus[i][j] = coeff_minus[i][j] - delta
            coeff_plus[i][j] = coeff_plus[i][j] + delta

            o_minus = qchem.overlap_integral(alphas, coeff_minus, rs, basis_a, basis_b)
            o_plus = qchem.overlap_integral(alphas, coeff_plus, rs, basis_a, basis_b)

            g_ref_coeff[i][j] = (o_plus - o_minus) / (2 * delta)

    assert np.allclose(g_alpha, g_ref_alpha)
    assert np.allclose(g_coeff, g_ref_coeff)

    # JIT
    overlap_integral_jitted = jax.jit(qchem.overlap_integral, static_argnums=(3, 4))

    g_alpha_jitted = jax.grad(overlap_integral_jitted, argnums=0)(
        alphas, coeffs, rs, basis_a, basis_b
    )
    g_coeff_jitted = jax.grad(overlap_integral_jitted, argnums=1)(
        alphas, coeffs, rs, basis_a, basis_b
    )

    assert np.allclose(g_alpha_jitted, g_ref_alpha)
    assert np.allclose(g_coeff_jitted, g_ref_coeff)


@pytest.mark.parametrize(
    ("alpha", "beta", "t", "e", "rc", "ref"),
    [
        (  # trivial case, ref = 0.0 for t > e
            jnp.array([3.42525091]),
            jnp.array([3.42525091]),
            2,
            1,
            jnp.array([1.5]),
            jnp.array([0.0]),
        ),
        (  # trivial case, ref = 0.0 for e == 0 and t != 0
            jnp.array([3.42525091]),
            jnp.array([3.42525091]),
            -1,
            0,
            jnp.array([1.5]),
            jnp.array([0.0]),
        ),
        (  # trivial case, ref = jnp.sqrt(jnp.pi / (alpha + beta))
            jnp.array([3.42525091]),
            jnp.array([3.42525091]),
            0,
            0,
            jnp.array([1.5]),
            jnp.array([0.677195]),
        ),
        (  # manually computed, ref = 1.0157925
            jnp.array([3.42525091]),
            jnp.array([3.42525091]),
            0,
            1,
            jnp.array([1.5]),
            jnp.array([1.0157925]),
        ),
    ],
)
def test_hermite_moment(alpha, beta, t, e, rc, ref):
    r"""Test that hermite_moment function returns correct values."""
    assert np.allclose(qchem.hermite_moment(alpha, beta, t, e, rc), ref)

    # JIT
    jitted_hermite_moment = jax.jit(qchem.hermite_moment, static_argnums=(2, 3))
    assert np.allclose(jitted_hermite_moment(alpha, beta, t, e, rc), ref)


@pytest.mark.parametrize(
    ("la", "lb", "ra", "rb", "alpha", "beta", "e", "rc", "ref"),
    [
        (  # manually computed, ref = 1.0157925
            0,
            0,
            jnp.array([2.0]),
            jnp.array([2.0]),
            jnp.array([3.42525091]),
            jnp.array([3.42525091]),
            1,
            jnp.array([1.5]),
            jnp.array([1.0157925]),
        ),
    ],
)
def test_gaussian_moment(la, lb, ra, rb, alpha, beta, e, rc, ref):
    r"""Test that gaussian_moment function returns correct values."""
    assert jnp.allclose(qchem.gaussian_moment(la, lb, ra, rb, alpha, beta, e, rc), ref)

    # JIT
    jitted_gaussian_moment = jax.jit(qchem.gaussian_moment, static_argnums=(0, 1, 6))
    assert np.allclose(jitted_gaussian_moment(la, lb, ra, rb, alpha, beta, e, rc), ref)


@pytest.mark.parametrize(
    ("symbols", "geometry", "e", "idx", "ref"),
    [
        (
            ["H", "Li"],
            jnp.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
            1,
            0,  # 'x' component
            3.12846324e-01,  # obtained from pyscf using mol.intor_symmetric("int1e_r")
        ),
        (
            ["H", "Li"],
            jnp.array([[0.5, 0.1, -0.2], [2.1, -0.3, 0.1]]),
            1,
            0,  # 'x' component
            4.82090830e-01,  # obtained from pyscf using mol.intor_symmetric("int1e_r")
        ),
        (
            ["N", "N"],
            jnp.array([[0.5, 0.1, -0.2], [2.1, -0.3, 0.1]]),
            1,
            2,  # 'z' component
            -4.70075530e-02,  # obtained from pyscf using mol.intor_symmetric("int1e_r")
        ),
    ],
)
def test_moment_integral(symbols, geometry, e, idx, ref):
    r"""Test that moment_integral function returns a correct value for the moment integral."""
    mol = Molecule(symbols, geometry)

    basis_a = mol.basis_set[0]
    basis_b = mol.basis_set[1]

    alphas = jnp.array(mol.alpha)  # We keep the signature for inputs the same - pairs of parameters
    coeffs = jnp.array(mol.coeff)
    rs = jnp.array(mol.r)

    s = qchem.moment_integral(alphas, coeffs, rs, basis_a, basis_b, e, idx)

    assert np.allclose(s, ref)

    # JIT
    jitted_moment_integral = jax.jit(qchem.moment_integral, static_argnums=(3, 4, 5, 6))
    s_jitted = jitted_moment_integral(alphas, coeffs, rs, basis_a, basis_b, e, idx)
    assert np.allclose(s_jitted, ref)


@pytest.mark.parametrize(
    ("symbols", "geometry", "alpha", "coeff", "e", "idx"),
    [
        (
            ["H", "H"],
            jnp.array([[0.1, 0.2, 0.3], [2.0, 0.1, 0.2]]),
            jnp.array(
                [[3.42525091, 0.62391373, 0.1688554], [3.42525091, 0.62391373, 0.1688554]],
            ),
            jnp.array([[0.15432897, 0.53532814, 0.44463454], [0.15432897, 0.53532814, 0.44463454]]),
            1,
            0,
        ),
    ],
)
def test_gradient_moment(symbols, geometry, alpha, coeff, e, idx):
    r"""Test that the moment gradient computed with respect to the basis parameters is
    correct."""
    mol = Molecule(symbols, geometry, alpha=alpha, coeff=coeff)

    basis_a = mol.basis_set[0]
    basis_b = mol.basis_set[1]

    alphas = mol.alpha
    coeffs = mol.coeff
    rs = mol.r

    g_alpha = jax.grad(qchem.moment_integral, argnums=0)(
        alphas, coeffs, rs, basis_a, basis_b, e, idx
    )
    g_coeff = jax.grad(qchem.moment_integral, argnums=1)(
        alphas, coeffs, rs, basis_a, basis_b, e, idx
    )

    # compute moment gradients with respect to alpha and coeff using finite diff
    delta = 0.00001
    g_ref_alpha = np.zeros(6).reshape(alpha.shape)
    g_ref_coeff = np.zeros(6).reshape(coeff.shape)

    for i in range(len(alpha)):
        for j in range(len(alpha[0])):

            alpha_minus = np.array(alpha.copy())
            alpha_plus = np.array(alpha.copy())

            alpha_minus[i][j] = alpha_minus[i][j] - delta
            alpha_plus[i][j] = alpha_plus[i][j] + delta

            o_minus = qchem.moment_integral(alpha_minus, coeff, rs, basis_a, basis_b, e, idx)
            o_plus = qchem.moment_integral(alpha_plus, coeff, rs, basis_a, basis_b, e, idx)

            g_ref_alpha[i][j] = (o_plus - o_minus) / (2 * delta)

            coeff_minus = np.array(coeff.copy())
            coeff_plus = np.array(coeff.copy())

            coeff_minus[i][j] = coeff_minus[i][j] - delta
            coeff_plus[i][j] = coeff_plus[i][j] + delta

            o_minus = qchem.moment_integral(alpha, coeff_minus, rs, basis_a, basis_b, e, idx)
            o_plus = qchem.moment_integral(alpha, coeff_plus, rs, basis_a, basis_b, e, idx)

            g_ref_coeff[i][j] = (o_plus - o_minus) / (2 * delta)

    assert np.allclose(g_alpha, g_ref_alpha)
    assert np.allclose(g_coeff, g_ref_coeff)

    # JIT
    moment_integral_jitted = jax.jit(qchem.moment_integral, static_argnums=(3, 4, 5, 6))

    g_alpha_jitted = jax.grad(moment_integral_jitted, argnums=0)(
        alphas, coeffs, rs, basis_a, basis_b, e, idx
    )
    g_coeff_jitted = jax.grad(moment_integral_jitted, argnums=1)(
        alphas, coeffs, rs, basis_a, basis_b, e, idx
    )

    assert np.allclose(g_alpha_jitted, g_ref_alpha)
    assert np.allclose(g_coeff_jitted, g_ref_coeff)


@pytest.mark.parametrize(
    ("i", "j", "ri", "rj", "alpha", "beta", "d"),
    [
        # _diff2 must return 0.0 for two Gaussians centered far apart at 0.0 and 20.0
        (
            0,
            1,
            jnp.array([0.0]),
            jnp.array([20.0]),
            jnp.array([3.42525091]),
            jnp.array([3.42525091]),
            jnp.array([0.0]),
        ),
        # computed manually
        (
            0,
            0,
            jnp.array([0.0]),
            jnp.array([1.0]),
            jnp.array([3.42525091]),
            jnp.array([3.42525091]),
            jnp.array([1.01479665]),
        ),
    ],
)
def test_diff2(i, j, ri, rj, alpha, beta, d):
    r"""Test that _diff2 function returns a correct value."""
    assert jnp.allclose(qchem.integrals._diff2(i, j, ri, rj, alpha, beta), d)

    # JIT
    jitted_diff_2 = jax.jit(qchem.integrals._diff2, static_argnums=(0, 1))
    assert jnp.allclose(jitted_diff_2(i, j, ri, rj, alpha, beta), d)


@pytest.mark.parametrize(
    ("la", "lb", "ra", "rb", "alpha", "beta", "t"),
    [
        # gaussian_kinetic must return 0.0 for two Gaussians centered far apart
        (
            (0, 0, 0),
            (0, 0, 0),
            jnp.array([0.0, 0.0, 0.0]),
            jnp.array([20.0, 0.0, 0.0]),
            jnp.array([3.42525091]),
            jnp.array([3.42525091]),
            jnp.array([0.0]),
        ),
    ],
)
def test_gaussian_kinetic(la, lb, ra, rb, alpha, beta, t):
    r"""Test that gaussian_kinetic function returns a correct value."""
    assert np.allclose(qchem.gaussian_kinetic(la, lb, ra, rb, alpha, beta), t)

    # JIT
    jitted_gaussian_kinetic = jax.jit(qchem.gaussian_kinetic, static_argnums=(0, 1))
    assert np.allclose(jitted_gaussian_kinetic(la, lb, ra, rb, alpha, beta), t)


@pytest.mark.parametrize(
    ("symbols", "geometry", "alpha", "coeff", "t_ref"),
    [
        # kinetic_integral must return 0.0 for two Gaussians centered far apart
        (
            ["H", "H"],
            jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 20.0]]),
            jnp.array([[3.42525091, 0.62391373, 0.1688554], [3.42525091, 0.62391373, 0.1688554]]),
            jnp.array([[0.15432897, 0.53532814, 0.44463454], [0.15432897, 0.53532814, 0.44463454]]),
            jnp.array([0.0]),
        ),
        # kinetic integral obtained from pyscf using mol.intor('int1e_kin')
        (
            ["H", "H"],
            jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
            jnp.array([[3.42525091, 0.62391373, 0.1688554], [3.42525091, 0.62391373, 0.1688554]]),
            jnp.array([[0.15432897, 0.53532814, 0.44463454], [0.15432897, 0.53532814, 0.44463454]]),
            jnp.array([0.38325384]),
        ),
    ],
)
def test_kinetic_integral(symbols, geometry, alpha, coeff, t_ref):
    r"""Test that kinetic_integral function returns a correct value for the kinetic integral."""
    mol = Molecule(symbols, geometry, alpha=alpha, coeff=coeff)

    basis_a = mol.basis_set[0]
    basis_b = mol.basis_set[1]

    alphas = alpha
    coeffs = coeff
    rs = mol.r

    t = qchem.kinetic_integral(alphas, coeffs, rs, basis_a, basis_b)
    assert np.allclose(t, t_ref)

    # JIT
    jitted_kinetic_integral = jax.jit(qchem.kinetic_integral, static_argnums=(3, 4))
    t_jitted = jitted_kinetic_integral(alphas, coeffs, rs, basis_a, basis_b)
    assert np.allclose(t_jitted, t_ref)


@pytest.mark.parametrize(
    ("symbols", "geometry", "alpha", "coeff"),
    [
        (
            ["H", "H"],
            jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
            jnp.array([[3.42525091, 0.62391373, 0.1688554], [3.42525091, 0.62391373, 0.1688554]]),
            jnp.array([[0.15432897, 0.53532814, 0.44463454], [0.15432897, 0.53532814, 0.44463454]]),
        ),
    ],
)
def test_gradient_kinetic(symbols, geometry, alpha, coeff):
    r"""Test that the kinetic gradient computed with respect to the basis parameters is
    correct."""
    mol = Molecule(symbols, geometry, alpha=alpha, coeff=coeff)

    basis_a = mol.basis_set[0]
    basis_b = mol.basis_set[1]

    alphas = mol.alpha
    coeffs = mol.coeff
    rs = mol.r

    g_alpha = jax.grad(qchem.kinetic_integral, argnums=0)(
        alphas,
        coeffs,
        rs,
        basis_a,
        basis_b,
    )
    g_coeff = jax.grad(qchem.kinetic_integral, argnums=1)(
        alphas,
        coeffs,
        rs,
        basis_a,
        basis_b,
    )

    # compute kinetic gradients with respect to alpha and coeff using finite diff
    delta = 0.00001
    g_ref_alpha = np.zeros(6).reshape(alpha.shape)
    g_ref_coeff = np.zeros(6).reshape(coeff.shape)

    for i in range(len(alpha)):
        for j in range(len(alpha[0])):

            alpha_minus = np.array(alpha.copy())
            alpha_plus = np.array(alpha.copy())

            alpha_minus[i][j] = alpha_minus[i][j] - delta
            alpha_plus[i][j] = alpha_plus[i][j] + delta

            o_minus = qchem.kinetic_integral(
                alpha_minus,
                coeff,
                rs,
                basis_a,
                basis_b,
            )
            o_plus = qchem.kinetic_integral(
                alpha_plus,
                coeff,
                rs,
                basis_a,
                basis_b,
            )

            g_ref_alpha[i][j] = (o_plus - o_minus) / (2 * delta)

            coeff_minus = np.array(coeff.copy())
            coeff_plus = np.array(coeff.copy())

            coeff_minus[i][j] = coeff_minus[i][j] - delta
            coeff_plus[i][j] = coeff_plus[i][j] + delta

            o_minus = qchem.kinetic_integral(
                alpha,
                coeff_minus,
                rs,
                basis_a,
                basis_b,
            )
            o_plus = qchem.kinetic_integral(
                alpha,
                coeff_plus,
                rs,
                basis_a,
                basis_b,
            )

            g_ref_coeff[i][j] = (o_plus - o_minus) / (2 * delta)

    assert np.allclose(g_alpha, g_ref_alpha)
    assert np.allclose(g_coeff, g_ref_coeff)

    # JIT
    kinetic_integral_jitted = jax.jit(qchem.kinetic_integral, static_argnums=(3, 4, 5, 6))

    g_alpha_jitted = jax.grad(kinetic_integral_jitted, argnums=0)(
        alphas,
        coeffs,
        rs,
        basis_a,
        basis_b,
    )
    g_coeff_jitted = jax.grad(kinetic_integral_jitted, argnums=1)(
        alphas,
        coeffs,
        rs,
        basis_a,
        basis_b,
    )

    assert np.allclose(g_alpha_jitted, g_ref_alpha)
    assert np.allclose(g_coeff_jitted, g_ref_coeff)


@pytest.mark.parametrize(
    ("symbols", "geometry", "alpha", "coeff", "a_ref"),
    [
        # trivial case: integral should be zero since atoms are located very far apart
        (
            ["H", "H"],
            jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 20.0]]),
            jnp.array([[3.42525091, 0.62391373, 0.1688554], [3.42525091, 0.62391373, 0.1688554]]),
            jnp.array([[0.15432897, 0.53532814, 0.44463454], [0.15432897, 0.53532814, 0.44463454]]),
            jnp.array([0.0]),
        ),
        # nuclear attraction integral obtained from pyscf using mol.intor('int1e_nuc')
        (
            ["H", "H"],
            jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
            jnp.array([[3.42525091, 0.62391373, 0.1688554], [3.42525091, 0.62391373, 0.1688554]]),
            jnp.array([[0.15432897, 0.53532814, 0.44463454], [0.15432897, 0.53532814, 0.44463454]]),
            np.array([0.80120855]),
        ),
    ],
)
def test_attraction_integral(symbols, geometry, alpha, coeff, a_ref):
    r"""Test that attraction_integral function returns a correct value for the kinetic
    integral."""
    mol = Molecule(symbols, geometry, alpha=alpha, coeff=coeff)

    basis_a = mol.basis_set[0]
    basis_b = mol.basis_set[1]

    alphas = mol.alpha
    coeffs = mol.coeff
    rs = mol.r

    coor = geometry[0]

    a = qchem.attraction_integral(alphas, coeffs, rs, coor, basis_a, basis_b)
    assert np.allclose(a, a_ref)

    # JIT
    jitted_attraction_integral = jax.jit(qchem.attraction_integral, static_argnums=(4, 5))
    a_jitted = jitted_attraction_integral(alphas, coeffs, rs, coor, basis_a, basis_b)
    assert np.allclose(a_jitted, a_ref)


@pytest.mark.parametrize(
    ("symbols", "geometry", "alpha", "coeff"),
    [
        (
            ["H", "H"],
            jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
            jnp.array(
                [[3.42525091, 0.62391373, 0.1688554], [3.42525091, 0.62391373, 0.1688554]],
            ),
            jnp.array([[0.15432897, 0.53532814, 0.44463454], [0.15432897, 0.53532814, 0.44463454]]),
        ),
    ],
)
def test_gradient_attraction(symbols, geometry, alpha, coeff):
    r"""Test that the attraction gradient computed with respect to the basis parameters is
    correct."""
    mol = Molecule(symbols, geometry, alpha=alpha, coeff=coeff)

    basis_a = mol.basis_set[0]
    basis_b = mol.basis_set[1]

    alphas = alpha
    coeffs = coeff
    rs = mol.r

    r_nuc = geometry[0]

    g_alpha = jax.grad(qchem.attraction_integral, argnums=0)(
        alphas, coeffs, rs, r_nuc, basis_a, basis_b
    )
    g_coeff = jax.grad(qchem.attraction_integral, argnums=1)(
        alphas, coeffs, rs, r_nuc, basis_a, basis_b
    )

    # compute attraction gradients with respect to alpha and coeff using finite diff
    delta = 0.00001
    g_ref_alpha = np.zeros(6).reshape(alpha.shape)
    g_ref_coeff = np.zeros(6).reshape(coeff.shape)

    for i in range(len(alpha)):
        for j in range(len(alpha[0])):

            alpha_minus = np.array(alpha.copy())
            alpha_plus = np.array(alpha.copy())

            alpha_minus[i][j] = alpha_minus[i][j] - delta
            alpha_plus[i][j] = alpha_plus[i][j] + delta

            o_minus = qchem.attraction_integral(
                alpha_minus,
                coeff,
                rs,
                r_nuc,
                basis_a,
                basis_b,
            )
            o_plus = qchem.attraction_integral(
                alpha_plus,
                coeff,
                rs,
                r_nuc,
                basis_a,
                basis_b,
            )

            g_ref_alpha[i][j] = (o_plus - o_minus) / (2 * delta)

            coeff_minus = np.array(coeff.copy())
            coeff_plus = np.array(coeff.copy())

            coeff_minus[i][j] = coeff_minus[i][j] - delta
            coeff_plus[i][j] = coeff_plus[i][j] + delta

            o_minus = qchem.attraction_integral(
                alpha,
                coeff_minus,
                rs,
                r_nuc,
                basis_a,
                basis_b,
            )
            o_plus = qchem.attraction_integral(
                alpha,
                coeff_plus,
                rs,
                r_nuc,
                basis_a,
                basis_b,
            )

            g_ref_coeff[i][j] = (o_plus - o_minus) / (2 * delta)

    assert np.allclose(g_alpha, g_ref_alpha)
    assert np.allclose(g_coeff, g_ref_coeff)

    # JIT
    attraction_integral_jitted = jax.jit(qchem.attraction_integral, static_argnums=(4, 5))

    g_alpha_jitted = jax.grad(attraction_integral_jitted, argnums=0)(
        alphas,
        coeffs,
        rs,
        r_nuc,
        basis_a,
        basis_b,
    )
    g_coeff_jitted = jax.grad(attraction_integral_jitted, argnums=1)(
        alphas,
        coeffs,
        rs,
        r_nuc,
        basis_a,
        basis_b,
    )

    assert np.allclose(g_alpha_jitted, g_ref_alpha)
    assert np.allclose(g_coeff_jitted, g_ref_coeff)


@pytest.mark.parametrize(
    ("symbols", "geometry", "alpha", "coeff", "e_ref"),
    [
        (
            ["H", "H"],
            jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 20.0]]),
            jnp.array(
                [
                    [3.42525091, 0.62391373, 0.1688554],
                    [3.42525091, 0.62391373, 0.1688554],
                    [3.42525091, 0.62391373, 0.1688554],
                    [3.42525091, 0.62391373, 0.1688554],
                ]
            ),
            jnp.array(
                [
                    [0.15432897, 0.53532814, 0.44463454],
                    [0.15432897, 0.53532814, 0.44463454],
                    [0.15432897, 0.53532814, 0.44463454],
                    [0.15432897, 0.53532814, 0.44463454],
                ]
            ),
            jnp.array([0.0]),
        ),
        (
            ["H", "H"],
            jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
            jnp.array(
                [
                    [3.42525091, 0.62391373, 0.1688554],
                    [3.42525091, 0.62391373, 0.1688554],
                    [3.42525091, 0.62391373, 0.1688554],
                    [3.42525091, 0.62391373, 0.1688554],
                ]
            ),
            jnp.array(
                [
                    [0.15432897, 0.53532814, 0.44463454],
                    [0.15432897, 0.53532814, 0.44463454],
                    [0.15432897, 0.53532814, 0.44463454],
                    [0.15432897, 0.53532814, 0.44463454],
                ]
            ),
            jnp.array([0.45590169]),
        ),
    ],
)
def test_repulsion_integral(symbols, geometry, alpha, coeff, e_ref):
    r"""Test that repulsion_integral function returns a correct value for the repulsion
    integral."""
    mol = Molecule(symbols, geometry, alpha=alpha, coeff=coeff)

    alphas = alpha
    coeffs = coeff

    # We repeat the basis
    basis_a = mol.basis_set[0]
    basis_b = mol.basis_set[1]

    rs = jnp.array([basis.r for basis in [basis_a, basis_b, basis_a, basis_b]])
    ls = tuple([tuple(basis.l) for basis in [basis_a, basis_b, basis_a, basis_b]])
    print(ls)

    # Here alphas, coeffs should have shape (4, N) for all four basis
    a = qchem.repulsion_integral(alphas, coeffs, rs, ls)

    assert np.allclose(a, e_ref)

    # JIT
    jitted_repuslion_integral = jax.jit(qchem.repulsion_integral, static_argnums=(3))
    a_jitted = jitted_repuslion_integral(alphas, coeffs, rs, ls)
    assert np.allclose(a_jitted, e_ref)


@pytest.mark.parametrize(
    ("symbols", "geometry", "alpha", "coeff"),
    [
        (
            ["H", "H"],
            jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
            jnp.array(
                [
                    [3.42525091, 0.62391373, 0.1688554],
                    [3.42525091, 0.62391373, 0.1688554],
                    [3.42525091, 0.62391373, 0.1688554],
                    [3.42525091, 0.62391373, 0.1688554],
                ]
            ),
            jnp.array(
                [
                    [0.15432897, 0.53532814, 0.44463454],
                    [0.15432897, 0.53532814, 0.44463454],
                    [0.15432897, 0.53532814, 0.44463454],
                    [0.15432897, 0.53532814, 0.44463454],
                ]
            ),
        ),
    ],
)
def test_gradient_repulsion(symbols, geometry, alpha, coeff):
    r"""Test that the repulsion gradient computed with respect to the basis parameters is
    correct."""
    mol = Molecule(symbols, geometry, alpha=alpha, coeff=coeff)

    alphas = alpha
    coeffs = coeff

    # We repeat the basis
    basis_a = mol.basis_set[0]
    basis_b = mol.basis_set[1]

    rs = jnp.array([basis.r for basis in [basis_a, basis_b, basis_a, basis_b]])
    ls = tuple([tuple(basis.l) for basis in [basis_a, basis_b, basis_a, basis_b]])

    g_alpha = jax.grad(qchem.repulsion_integral, argnums=0)(alphas, coeffs, rs, ls)
    g_coeff = jax.grad(qchem.repulsion_integral, argnums=1)(alphas, coeffs, rs, ls)

    # compute repulsion gradients with respect to alpha and coeff using finite diff
    delta = 0.00001
    g_ref_alpha = np.zeros(12).reshape(alpha.shape)
    g_ref_coeff = np.zeros(12).reshape(coeff.shape)

    for i in range(len(alpha)):
        for j in range(len(alpha[0])):

            alpha_minus = np.array(alpha.copy())
            alpha_plus = np.array(alpha.copy())

            alpha_minus[i][j] = alpha_minus[i][j] - delta
            alpha_plus[i][j] = alpha_plus[i][j] + delta

            o_minus = qchem.repulsion_integral(alpha_minus, coeff, rs, ls)
            o_plus = qchem.repulsion_integral(alpha_plus, coeff, rs, ls)

            g_ref_alpha[i][j] = (o_plus - o_minus) / (2 * delta)

            coeff_minus = np.array(coeff.copy())
            coeff_plus = np.array(coeff.copy())

            coeff_minus[i][j] = coeff_minus[i][j] - delta
            coeff_plus[i][j] = coeff_plus[i][j] + delta

            o_minus = qchem.repulsion_integral(alpha, coeff_minus, rs, ls)
            o_plus = qchem.repulsion_integral(alpha, coeff_plus, rs, ls)

            g_ref_coeff[i][j] = (o_plus - o_minus) / (2 * delta)

    assert np.allclose(g_alpha, g_ref_alpha)
    assert np.allclose(g_coeff, g_ref_coeff)

    # JIT
    repulsion_integral_jitted = jax.jit(qchem.repulsion_integral, static_argnums=(3))

    g_alpha_jitted = jax.grad(repulsion_integral_jitted, argnums=0)(alphas, coeffs, rs, ls)
    g_coeff_jitted = jax.grad(repulsion_integral_jitted, argnums=1)(alphas, coeffs, rs, ls)

    assert np.allclose(g_alpha_jitted, g_ref_alpha)
    assert np.allclose(g_coeff_jitted, g_ref_coeff)
