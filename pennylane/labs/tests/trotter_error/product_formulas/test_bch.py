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


fragments = [
    {"X": np.zeros(shape=(3, 3)), "Y": np.zeros(shape=(3, 3))},
    {"X": np.random.random(size=(3, 3)), "Y": np.random.random(size=(3, 3))},
]


@pytest.mark.parametrize("fragments, t", product(fragments, [1, 0.1, 0.01]))
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
    fourth_order_approx = _pf_to_matrix(fourth_order, fragments, np.eye(3, dtype=np.complex128))
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
            ["X", "Y"],
            [1, 1],
            4,
            [
                {(X,): 1, (Y,): 1},
                {(X, Y): 1 / 2},
                {(X, X, Y): 1 / 12, (Y, X, Y): -1 / 12},
                {(X, Y, X, Y): -1 / 24},
            ],
        ),
        (
            ["X", "Y", "Z"],
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
            ["X", "Y", "X"],
            [1, 1, 1],
            3,
            [
                {(X,): 2, (Y,): 1},
                {},
                {(X, X, Y): -1 / 6, (Y, X, Y): -1 / 6},
            ],
        ),
    ],
)
def test_bch_expansion(frag_labels, frag_coeffs, max_order, expected):
    """Test against BCH expansion. The expected values come from Sections 4 and 5 of `arXiv:2006.15869 <https://arxiv.org/pdf/2006.15869>`"""

    actual = ProductFormula(frag_labels, coeffs=frag_coeffs, include_i=False).bch_approx(
        max_order=max_order
    )

    for i, order in enumerate(actual):
        for commutator in order:
            assert np.isclose(order[commutator], expected[i][commutator])


@pytest.mark.parametrize(
    "fragments, t",
    product(
        [
            ({"X": np.zeros(shape=(3, 3)), "Y": np.zeros(shape=(3, 3))}),
            # ({"X": np.random.random(size=(3, 3)), "Y": np.random.random(size=(3, 3))}),
            ({"X": np.ones(shape=(3, 3)), "Y": np.ones(shape=(3, 3))}),
        ],
        [0.1, 0.01, 0.001],
    ),
)
def test_second_order_against_matrix_log(fragments, t):
    """Test the BCH expansion against directly computing the matrix logarithm."""

    second_order_pf = ProductFormula(["X", "Y", "X"], coeffs=[1 / 2, 1, 1 / 2])(t)
    bch_expansion = effective_hamiltonian(second_order_pf, fragments, order=2)

    pf_matrix = _pf_to_matrix(second_order_pf, fragments, np.eye(3, dtype=np.complex128))

    print(pf_matrix)
    print(expm(bch_expansion))

    assert np.allclose(pf_matrix, expm(bch_expansion))


def _pf_to_matrix(product_formula, fragments, accumulator):
    for frag, coeff in zip(product_formula.terms, product_formula.coeffs):
        accumulator @= expm(coeff * fragments[frag])

    return accumulator
