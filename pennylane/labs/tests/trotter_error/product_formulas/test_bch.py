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
    """Test against first order Trotter"""

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
    """Test against the second order Trotter error formula"""

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


@pytest.mark.parametrize("fragments, t", product(fragment_list, [1, 0.1, 0.01]))
def test_fourth_order_norm_two_fragments(fragments, t):
    """Test against the fourth order Trotter error formula on two fragments"""

    u = 1 / (4 - 4 ** (1 / 3))
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

    fourth_order = ProductFormula(frag_labels, coeffs=frag_coeffs)(t)
    fourth_order_approx = fourth_order.to_matrix(fragments)
    actual = expm(1j * t * sum(fragments.values(), np.zeros(shape=(3, 3), dtype=np.complex128)))

    commutator_coeffs = {
        ("X", "X", "X", "Y", "X"): 0.0047,
        ("X", "X", "Y", "Y", "X"): 0.0057,
        ("X", "Y", "X", "Y", "X"): 0.0046,
        ("X", "Y", "Y", "Y", "X"): 0.0074,
        ("Y", "X", "X", "Y", "X"): 0.0097,
        ("Y", "X", "Y", "Y", "X"): 0.0097,
        ("Y", "Y", "X", "Y", "X"): 0.0173,
        ("Y", "Y", "Y", "Y", "X"): 0.0284,
    }

    upper_bound = 0
    for comm, coeff in commutator_coeffs.items():
        mat = nested_commutator([fragments[frag] for frag in comm])
        upper_bound += coeff * np.linalg.norm(mat)

    assert np.linalg.norm(fourth_order_approx - actual) <= (t**5) * upper_bound


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
                # {
                #    (X, X, Y, Y, X, Y): -1 / 720,
                #    (X, Y, X, Y, X, Y): 1 / 240,
                #    (X, Y, Y, Y, X, Y): 1 / 1440,
                #    (Y, X, X, X, X, Y): 1 / 1440,
                # },
                # {
                #    (X, X, X, X, X, X, Y): 1 / 30240,
                #    (X, X, Y, X, X, X, Y): 1 / 5040,
                #    (X, X, Y, Y, X, X, Y): -1 / 10080,
                #    (X, Y, X, X, X, X, Y): 1 / 10080,
                #    (X, Y, X, Y, X, X, Y): 1 / 1008,
                #    (X, Y, X, Y, Y, X, Y): 1 / 5040,
                #    (X, Y, Y, X, X, X, Y): -1 / 7560,
                #    (X, Y, Y, Y, X, X, Y): 1 / 3360,
                #    (X, Y, Y, Y, Y, X, Y): 1 / 10080,
                #    (Y, X, X, X, X, X, Y): -1 / 10080,
                #    (Y, X, Y, X, X, X, Y): -1 / 1260,
                #    (Y, X, Y, Y, X, X, Y): -1 / 1680,
                #    (Y, Y, X, X, X, X, Y): 1 / 3360,
                #    (Y, Y, X, Y, X, X, Y): -1 / 3360,
                #    (Y, Y, X, Y, Y, X, Y): -1 / 2520,
                #    (Y, Y, Y, X, X, X, Y): 1 / 7560,
                #    (Y, Y, Y, Y, X, X, Y): 1 / 10080,
                #    (Y, Y, Y, Y, Y, X, Y): -1 / 30240,
                # },
            ],
        ),
        (
            [X, Y, Z],
            [1, 1, 1],
            3,
            [
                {(X,): 1, (Y,): 1, (Z,): 1},
                {
                    (X, Y): 1 / 2,
                    (X, Z): 1 / 2,
                    (Y, Z): 1 / 2,
                },
                {
                    (X, X, Y): 1 / 12,
                    (X, X, Z): 1 / 12,
                    (X, Y, Z): 1 / 3,
                    (Y, X, Y): -1 / 12,
                    (Y, X, Z): -1 / 6,
                    (Y, Y, Z): 1 / 12,
                    (Z, X, Z): -1 / 12,
                    (Z, Y, Z): -1 / 12,
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

    product_formula = ProductFormula(frag_labels, coeffs=frag_coeffs, include_i=False)
    actual = bch_expansion(product_formula, max_order)

    for i, order in enumerate(actual):
        for comm in order:
            assert np.isclose(order[comm], expected[i][comm])


fragment_list = [
    ({"X": np.zeros(shape=(3, 3)), "Y": np.zeros(shape=(3, 3))}),
    ({"X": np.random.random(size=(3, 3)), "Y": np.random.random(size=(3, 3))}),
    ({"X": np.ones(shape=(3, 3)), "Y": np.ones(shape=(3, 3))}),
]


@pytest.mark.parametrize("fragments", fragment_list)
def test_second_order_against_matrix_log(fragments):
    """Test that the BCH expansion converges to the matrix log"""
    t = 0.01
    second_order = ProductFormula(["X", "Y", "X"], coeffs=[1 / 2, 1, 1 / 2])(t)
    ham = sum(fragments.values())

    pf_mat = second_order.to_matrix(fragments)

    bch = effective_hamiltonian(second_order, fragments, order=3)
    log = logm(pf_mat)

    bch_error = bch - (1j * t * ham) / t**3
    log_error = log - (1j * t * ham) / t**3

    assert np.allclose(np.linalg.norm(bch_error - log_error), 0)


@pytest.mark.parametrize("fragments", fragment_list)
def test_fourth_order_against_matrix_log(fragments):
    """Test that the BCH expansion converges to the matrix log"""
    t = 0.007
    u = 1 / (4 - 4 ** (1 / 3))
    second_order = ProductFormula(["X", "Y", "X"], coeffs=[1 / 2, 1, 1 / 2])
    fourth_order = (
        second_order(t * u) ** 2 @ second_order(t * (1 - 4 * u)) @ second_order(t * u) ** 2
    )
    ham = sum(fragments.values())

    pf_mat = fourth_order.to_matrix(fragments)

    bch = effective_hamiltonian(fourth_order, fragments, order=5)
    log = logm(pf_mat)

    bch_error = bch - (1j * t * ham) / t**5
    log_error = log - (1j * t * ham) / t**5

    assert np.allclose(np.linalg.norm(bch_error - log_error), 0)


@pytest.mark.parametrize("fragments", fragment_list)
def test_fourth_order_against_matrix_log_2(fragments):
    """Test that the BCH expansion converges to the matrix log"""
    t = 0.007
    u = 1 / (4 - 4 ** (1 / 3))
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

    fourth_order = ProductFormula(frag_labels, coeffs=frag_coeffs)(t)
    ham = sum(fragments.values())

    pf_mat = fourth_order.to_matrix(fragments)

    bch = effective_hamiltonian(fourth_order, fragments, order=5)
    log = logm(pf_mat)

    bch_error = bch - (1j * t * ham) / t**5
    log_error = log - (1j * t * ham) / t**5

    assert np.allclose(np.linalg.norm(bch_error - log_error), 0)
