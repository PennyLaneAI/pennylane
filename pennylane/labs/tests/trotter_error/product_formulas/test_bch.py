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

import copy
from itertools import product

import numpy as np
import pytest
from scipy.linalg import logm

from pennylane.labs.trotter_error import ProductFormula, effective_hamiltonian
from pennylane.labs.trotter_error.abstract import nested_commutator
from pennylane.labs.trotter_error.product_formulas.bch import bch_expansion

deltas = [0.5, 0.1, 0.01]

np.random.seed(42)
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
            expected += (1 / 2) * nested_commutator([fragments[j], fragments[k]])

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


@pytest.mark.parametrize("fragments, delta", product(fragment_list, deltas))
def test_fourth_order(fragments, delta):
    """Test that the effective_hamiltonian function returns the correct result for fourth order Trotter.
    The expected Hamiltonian was generated via Sympy."""
    u = 1 / (4 - 4 ** (1 / 3))
    v = 1 - 4 * u
    y3 = -0.00405944185443219
    y5 = -0.074375995396295

    frag_labels = list(range(len(fragments))) + list(reversed(range(len(fragments))))
    frag_coeffs = [1 / 2] * len(frag_labels)

    second_order = ProductFormula(frag_labels, frag_coeffs)
    fourth_order = (
        second_order(delta * u) ** 2 @ second_order(delta * v) @ second_order(delta * u) ** 2
    )

    ham = 1j * delta * (sum(fragments.values(), np.zeros_like(fragments[0])))
    expected = copy.copy(ham)

    bch = bch_expansion(second_order, order=5)
    for commutator, coeff in bch[2].items():
        commutator = tuple(1j * delta * fragments[x] for x in commutator)
        commutator = (ham, commutator, ham)
        expected += y3 * coeff * nested_commutator(commutator)
    for commutator, coeff in bch[4].items():
        commutator = tuple(1j * delta * fragments[x] for x in commutator)
        expected += y5 * coeff * nested_commutator(commutator)

    actual = effective_hamiltonian(fourth_order, fragments, order=5)

    assert np.allclose(expected, actual)


@pytest.mark.parametrize("fragments, delta", product(fragment_list, deltas))
def test_sixth_order(fragments, delta):
    """Test that the effective_hamiltonian function returns the correct result for sixth order Trotter.
    The expected Hamiltonian was generated via Sympy."""
    u4 = 1 / (4 - 4 ** (1 / 3))
    u6 = 1 / (4 - 4 ** (1 / 5))
    v4 = 1 - 4 * u4
    v6 = 1 - 4 * u6
    h_y3_hhh = -2.76996810612187 * 10e-6
    h_y3_y3 = -1.0079377060157 * 10e-5
    h_y5_h = 5.28018804117902 * 10e-5
    y7 = 0.000134097740473571

    frag_labels = list(range(len(fragments))) + list(reversed(range(len(fragments))))
    frag_coeffs = [1 / 2] * len(frag_labels)
    second_order = ProductFormula(frag_labels, frag_coeffs)
    fourth_order = second_order(u4) ** 2 @ second_order(v4) @ second_order(u4) ** 2
    sixth_order = (
        fourth_order(delta * u6) ** 2 @ fourth_order(delta * v6) @ fourth_order(delta * u6) ** 2
    )

    ham = 1j * delta * (sum(fragments.values(), np.zeros_like(fragments[0])))
    expected = copy.copy(ham)

    bch = bch_expansion(second_order, order=7)
    for commutator, coeff in bch[2].items():
        commutator = tuple(1j * delta * fragments[x] for x in commutator)
        new_commutator = (ham, commutator, commutator)
        expected += h_y3_y3 * coeff * nested_commutator(new_commutator)

        new_commutator = (ham, commutator, ham, ham, ham)
        expected += h_y3_hhh * coeff * nested_commutator(new_commutator)

    for commutator, coeff in bch[4].items():
        commutator = tuple(1j * delta * fragments[x] for x in commutator)
        new_commutator = (ham, commutator, ham)
        expected += h_y5_h * coeff * nested_commutator(new_commutator)

    for commutator, coeff in bch[6].items():
        commutator = tuple(1j * delta * fragments[x] for x in commutator)
        expected += y7 * coeff * nested_commutator(commutator)

    actual = effective_hamiltonian(sixth_order, fragments, order=7)

    assert np.allclose(expected, actual)


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

second_order_pf = ProductFormula(["X", "Y", "X"], coeffs=[1 / 2, 1, 1 / 2])
c = 1 / (4 - 4 ** (1 / 3))
fourth_order1 = second_order_pf(c) ** 2 @ second_order_pf(1 - 4 * c) @ second_order_pf(c) ** 2
fourth_order_labels = ["X", "Y", "X", "Y", "X", "Y", "X", "Y", "X", "Y", "X"]
fourth_order_coeffs = [
    c / 2,
    c,
    c,
    c,
    (1 - (3 * c)) / 2,
    (1 - (4 * c)),
    (1 - (3 * c)) / 2,
    c,
    c,
    c,
    c / 2,
]

fourth_order2 = ProductFormula(fourth_order_labels, coeffs=fourth_order_coeffs)
product_formulas = [(second_order_pf, 3), (fourth_order1, 5), (fourth_order2, 5)]


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


params = [
    ("serial", 1),
    ("mp_pool", 1),
    ("mp_pool", 2),
    ("cf_procpool", 1),
    ("cf_procpool", 2),
    ("cf_threadpool", 1),
    ("cf_threadpool", 2),
    ("mpi4py_pool", 1),
    ("mpi4py_pool", 2),
    ("mpi4py_comm", 1),
    ("mpi4py_comm", 2),
]


@pytest.mark.parametrize("backend, num_workers", params)
def test_effective_hamiltonian_backend(backend, num_workers, mpi4py_support):
    """Test that effective_hamiltonian function runs without errors for different backends."""

    if backend in {"mpi4py_pool", "mpi4py_comm"} and not mpi4py_support:
        pytest.skip(f"Skipping test: '{backend}' requires mpi4py, which is not installed.")

    r, delta = 1, 0.5

    fragments = {0: np.random.random(size=(3, 3)), 1: np.random.random(size=(3, 3))}
    n_frags = len(fragments)

    first_order = ProductFormula(list(range(n_frags)), coeffs=[delta / r] * n_frags) ** r
    actual = effective_hamiltonian(
        first_order, fragments, order=2, num_workers=num_workers, backend=backend
    )

    ham = sum(fragments.values(), np.zeros(shape=(3, 3)))
    expected = np.zeros(shape=(3, 3), dtype=np.complex128)
    for j in range(n_frags - 1):
        for k in range(j + 1, n_frags):
            expected += (1 / 2) * nested_commutator([fragments[j], fragments[k]])
    expected *= 1j / r * delta

    assert np.allclose(1j * delta * (expected + ham), actual)
