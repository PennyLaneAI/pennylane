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
"""Tests for the BCH approximation"""

from itertools import product

import numpy as np
import pytest
from scipy.linalg import expm, logm

from pennylane.labs.trotter_error import ProductFormula, effective_hamiltonian
from pennylane.labs.trotter_error.abstract import commutator, nested_commutator
from pennylane.labs.trotter_error.product_formulas.bch import bch_expansion

deltas = [1, 0.1, 0.01]

fragment_list = [
    {0: np.zeros(shape=(3, 3)), 1: np.zeros(shape=(3, 3)), 2: np.zeros(shape=(3, 3))},
    {0: np.random.random(size=(3, 3)), 1: np.random.random(size=(3, 3))},
    {
        0: np.random.random(size=(3, 3)),
        1: np.random.random(size=(3, 3)),
        2: np.random.random(size=(3, 3)),
    },
]


@pytest.mark.parametrize("fragments, r, delta", product(fragment_list, range(1, 4), deltas))
def test_first_order(fragments, r, delta):
    """Test that the effective_hamiltonian function returns the correct result for first order Trotter.
    See Proposition 4 of https://arxiv.org/pdf/2408.03891 for the expression."""

    n_frags = len(fragments)
    expected = np.zeros(shape=(3, 3), dtype=np.complex128)
    ham = sum(fragments.values(), np.zeros(shape=(3, 3)))

    for j in range(n_frags - 1):
        for k in range(j + 1, n_frags):
            expected += (1 / 2) * commutator(fragments[j], fragments[k])

    expected *= 1j / r * delta

    first_order = ProductFormula(list(range(n_frags)), coeffs=[delta / r] * n_frags) ** r
    actual = effective_hamiltonian(first_order, fragments, order=2)

    assert np.allclose(1j * delta * (expected + ham), actual)


@pytest.mark.parametrize("fragments, r, delta", product(fragment_list, range(1, 4), deltas))
def test_second_order(fragments, r, delta):
    """Test that the effective_hamiltonian function returns the correct result for second order Trotter.
    See Proposition 4 of https://arxiv.org/pdf/2408.03891 for the expression."""

    n_frags = len(fragments)
    frag_labels = list(range(n_frags)) + list(range(n_frags - 1, -1, -1))
    coeffs = [delta / (2 * r)] * n_frags * 2

    pf = ProductFormula(frag_labels, coeffs=coeffs) ** r
    actual = effective_hamiltonian(pf, fragments, order=3)

    expected = np.zeros(shape=(3, 3), dtype=np.complex128)

    for i in range(n_frags - 1):
        for j in range(i + 1, n_frags):
            expected += nested_commutator([fragments[i], fragments[i], fragments[j]])
            for k in range(i + 1, n_frags):
                expected += 2 * nested_commutator([fragments[k], fragments[i], fragments[j]])

    expected *= -1 / 24
    expected *= (1j * delta / r) ** 2

    ham = sum(fragments.values(), np.zeros(shape=(3, 3)))
    eff = 1j * delta * (expected + ham)

    assert np.allclose(eff, actual)


fragment_list = [
    {"X": np.zeros(shape=(3, 3)), "Y": np.zeros(shape=(3, 3))},
    {"X": np.random.random(size=(3, 3)), "Y": np.random.random(size=(3, 3))},
]


X = "X"
Y = "Y"
Z = "Z"


@pytest.mark.parametrize(
    "frag_labels, frag_coeffs, max_order, expected",
    [
        (
            [X, Y],
            [1, 1],
            5,
            [
                {(X,): 1, (Y,): 1},
                {(X, Y): 1 / 2},
                {(X, X, Y): 1 / 12, (Y, X, Y): -1 / 12},
                {(X, Y, X, Y): -1 / 24},
                {
                    (X, X, X, X, Y): -1 / 720,
                    (X, X, Y, X, Y): -1 / 120,
                    (X, Y, Y, X, Y): -1 / 360,
                    (Y, X, X, X, Y): 1 / 360,
                    (Y, X, Y, X, Y): 1 / 120,
                    (Y, Y, Y, X, Y): 1 / 720,
                },
            ],
        ),
        (
            [X, Y, X],
            [1, 1, 1],
            3,
            [
                {(X,): 2, (Y,): 1},
                {},
                {(X, X, Y): -1 / 6, (Y, X, Y): -1 / 6},
            ],
        ),
        (
            [X, Y, X],
            [1 / 2, 1, 1 / 2],
            3,
            [{(X,): 1, (Y,): 1}, {}, {(X, X, Y): -1 / 24, (Y, X, Y): -1 / 12}],
        ),
    ],
)
def test_bch_expansion(frag_labels, frag_coeffs, max_order, expected):
    """Test against BCH expansion. The expected values come from Sections 4 and 5 of `arXiv:2006.15869 <https://arxiv.org/pdf/2006.15869>`"""

    product_formula = ProductFormula(frag_labels, coeffs=frag_coeffs)
    actual = bch_expansion(product_formula, max_order)

    for i, order in enumerate(actual):
        for comm in order:
            assert np.isclose(
                order[comm], expected[i][comm]
            ), f"Coefficient for commutator {comm} did not match its expected value."


fragment_list = [
    ({"X": np.zeros(shape=(3, 3)), "Y": np.zeros(shape=(3, 3))}),
    ({"X": np.random.random(size=(3, 3)), "Y": np.random.random(size=(3, 3))}),
    ({"X": np.ones(shape=(3, 3)), "Y": np.ones(shape=(3, 3))}),
]

second_order = ProductFormula(["X", "Y", "X"], coeffs=[1 / 2, 1, 1 / 2])
u = 1 / (4 - 4 ** (1 / 3))
fourth_order1 = second_order(u) ** 2 @ second_order((1 - 4 * u)) @ second_order(u) ** 2
frag_labels = ["X", "Y", "X", "Y", "X", "Y", "X", "Y", "X", "Y", "X"]
frag_coeffs = [
    u / 2,
    u,
    u,
    u,
    (1 - (3 * u)) / 2,
    (1 - (4 * u)),
    (1 - (3 * u)) / 2,
    u,
    u,
    u,
    u / 2,
]

fourth_order2 = ProductFormula(frag_labels, coeffs=frag_coeffs)
product_formulas = [(second_order, 3), (fourth_order1, 5), (fourth_order2, 5)]


@pytest.mark.parametrize("fragments, product_formula", product(fragment_list, product_formulas))
def test_against_matrix_log(fragments, product_formula):
    """Test that the BCH expansion converges to the matrix log of the product formula."""
    product_formula, order = product_formula
    t = 0.007
    ham = sum(fragments.values())

    pf_mat = product_formula(1j * t).to_matrix(fragments)

    bch = effective_hamiltonian(product_formula, fragments, order, t)
    log = logm(pf_mat)

    bch_error = bch - (1j * t * ham) / t**order
    log_error = log - (1j * t * ham) / t**order

    assert np.allclose(np.linalg.norm(bch_error - log_error), 0)
