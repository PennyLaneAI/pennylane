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

import pytest
import numpy as np
from pennylane import numpy as pnp
from pennylane.hf.integrals import generate_params, expansion, gaussian_overlap, generate_overlap
from pennylane.hf.molecule import Molecule

@pytest.mark.parametrize(
    ("alpha", "coeff", "r"),
    [
        (
            pnp.array([3.42525091, 0.62391373, 0.1688554], requires_grad=True),
            pnp.array([0.15432897, 0.53532814, 0.44463454], requires_grad=True),
            pnp.array([0.0, 0.0, 0.0], requires_grad=False),
        ),
        (
            pnp.array([3.42525091, 0.62391373, 0.1688554], requires_grad=False),
            pnp.array([0.15432897, 0.53532814, 0.44463454], requires_grad=False),
            pnp.array([0.0, 0.0, 0.0], requires_grad=True),
        ),
    ],
)
def test_generate_params(alpha, coeff, r):
    r"""Test that test_generate_params returns correct basis set parameters."""
    params = [alpha, coeff, r]
    args = [p for p in [alpha, coeff, r] if p.requires_grad]
    basis_params = generate_params(params, args)

    assert np.allclose(basis_params, (alpha, coeff, r))


@pytest.mark.parametrize(
    ("la", "lb", "ra", "rb", "alpha", "beta", "t", "c"),
    [
        (
            0,
            0,
            pnp.array([1.2]),
            pnp.array([1.2]),
            pnp.array([3.42525091]),
            pnp.array([3.42525091]),
            0,
            pnp.array([1.0]),
        ),
        (
            1,
            0,
            pnp.array([0.0]),
            pnp.array([0.0]),
            pnp.array([3.42525091]),
            pnp.array([3.42525091]),
            0,
            pnp.array([0.0]),
        ),
        (
            1,
            1,
            pnp.array([0.0]),
            pnp.array([10.0]),
            pnp.array([3.42525091]),
            pnp.array([3.42525091]),
            0,
            pnp.array([0.0]),
        ),
    ],
)
def test_expansion(la, lb, ra, rb, alpha, beta, t, c):
    r"""Test that expansion function returns correct value."""
    assert np.allclose(expansion(la, lb, ra, rb, alpha, beta, t), c)
    assert np.allclose(expansion(la, lb, ra, rb, alpha, beta, -1), pnp.array([0.0]))
    assert np.allclose(expansion(0, 1, ra, rb, alpha, beta, 2), pnp.array([0.0]))


@pytest.mark.parametrize(
    ("la", "lb", "ra", "rb", "alpha", "beta", "o"),
    [
        # two normalized s orbitals
        (
            (0, 0, 0),
            (0, 0, 0),
            pnp.array([0.0, 0.0, 0.0]),
            pnp.array([0.0, 0.0, 0.0]),
            pnp.array([pnp.pi / 2]),
            pnp.array([pnp.pi / 2]),
            pnp.array([1.0]),
        ),
        (
            (0, 0, 0),
            (0, 0, 0),
            pnp.array([0.0, 0.0, 0.0]),
            pnp.array([20.0, 0.0, 0.0]),
            pnp.array([3.42525091]),
            pnp.array([3.42525091]),
            pnp.array([0.0]),
        ),
        (
            (1, 0, 0),
            (0, 0, 1),
            pnp.array([0.0, 0.0, 0.0]),
            pnp.array([0.0, 0.0, 0.0]),
            pnp.array([6.46480325]),
            pnp.array([6.46480325]),
            pnp.array([0.0]),
        ),
    ],
)
def test_gaussian_overlap(la, lb, ra, rb, alpha, beta, o):
    r"""Test that gaussian overlap function returns a correct value."""
    assert np.allclose(gaussian_overlap(la, lb, ra, rb, alpha, beta), o)


@pytest.mark.parametrize(
    ("symbols", "geometry", "params", "o_ref"),
    [
        (
            ["H", "H"],
            pnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], requires_grad=True),
            pnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], requires_grad=True),
            pnp.array([1.0]),
        ),
    ],
)
def test_generate_overlap(symbols, geometry, params, o_ref):
    r"""Test that generate overlap function returns a correct value."""
    mol = Molecule(symbols, geometry)
    basis_a = mol.basis_set[0]
    basis_b = mol.basis_set[0]
    o = generate_overlap(basis_a, basis_b)(*params)
    assert np.allclose(o, o_ref)
