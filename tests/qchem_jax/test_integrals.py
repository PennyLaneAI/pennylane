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

    rs = jnp.zeros(alphas.shape)  # We keep the signature of the overlap integrals the same

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

    # JIT computation
    overlap_integral_jitted = jax.jit(qchem.overlap_integral, static_argnums=(3, 4))

    g_alpha_jitted = jax.grad(overlap_integral_jitted, argnums=0)(
        alphas, coeffs, rs, basis_a, basis_b
    )
    g_coeff_jitted = jax.grad(overlap_integral_jitted, argnums=1)(
        alphas, coeffs, rs, basis_a, basis_b
    )

    assert np.allclose(g_alpha_jitted, g_ref_alpha)
    assert np.allclose(g_coeff_jitted, g_ref_coeff)
