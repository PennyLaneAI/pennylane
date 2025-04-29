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

import math

import numpy as np
import pytest
from scipy.linalg import expm, logm

from pennylane.labs.trotter_error import ProductFormula, effective_hamiltonian
from pennylane.labs.trotter_error.abstract import nested_commutator


@pytest.mark.parametrize(
    "fragments",
    [
        {0: np.zeros(shape=(3, 3)), 1: np.zeros(shape=(3, 3)), 2: np.zeros(shape=(3, 3))},
        {0: np.random.random(size=(3, 3)), 1: np.random.random(size=(3, 3))},
        {
            0: np.random.random(size=(3, 3)),
            1: np.random.random(size=(3, 3)),
            2: np.random.random(size=(3, 3)),
        },
    ],
)
def test_second_order(fragments):
    """Test against the second order Trotter error formula"""

    n_frags = len(fragments)
    frag_labels = list(range(n_frags)) + list(range(n_frags - 1, -1, -1))
    coeffs = [1 / 2] * len(frag_labels)

    pf = ProductFormula(frag_labels, coeffs=coeffs)
    actual = effective_hamiltonian(pf, fragments, 3)

    expected = np.zeros(shape=(3, 3))

    for i in range(n_frags - 1):
        for j in range(i + 1, n_frags):
            expected += nested_commutator([fragments[i], fragments[i], fragments[j]])
            for k in range(i + 1, n_frags):
                expected += 2 * nested_commutator([fragments[k], fragments[i], fragments[j]])

    expected = -(1 / 24) * expected

    for fragment in fragments.values():
        expected += fragment

    assert np.allclose(actual, expected)


@pytest.mark.parametrize(
    "fragments",
    [
        {"X": np.zeros(shape=(3, 3)), "Y": np.zeros(shape=(3, 3)), 2: np.zeros(shape=(3, 3))},
        {"X": np.random.random(size=(3, 3)), "Y": np.random.random(size=(3, 3))},
    ],
)
def test_fourth_order_two_fragments(fragments):
    """Test against the fourth order Trotter error formula on two fragments"""

    u = 1 / (4 - (4 ** (1 / 3)))
    second_order = ProductFormula(["X", "Y", "X"], coeffs=[1 / 2, 1, 1 / 2])
    fourth_order = ProductFormula(
        [second_order(u) ** 2, second_order(1 - 4 * u), second_order(u) ** 2]
    )

    eff = effective_hamiltonian(fourth_order, fragments, order=5)
    error = np.linalg.norm(eff - sum(fragments.values()))

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

    print(error)
    print(upper_bound)

    assert error <= upper_bound


X = ("X", 1)
Y = ("Y", 1)
Z = ("Z", 1)


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

    actual = ProductFormula(frag_labels, coeffs=frag_coeffs).bch_approx(max_order=max_order)

    for i, order in enumerate(actual):
        for commutator in order:
            assert np.isclose(order[commutator], expected[i][commutator])


@pytest.mark.parametrize(
    "fragments",
    [
        ({"X": np.zeros(shape=(3, 3)), "Y": np.zeros(shape=(3, 3))}),
        ({"X": np.random.random(size=(3, 3)), "Y": np.random.random(size=(3, 3))}),
    ],
)
def test_second_order_against_matrix_log(fragments):
    """Test the BCH expansion against directly computing the matrix logarithm."""
    t = 1

    second_order_pf = ProductFormula(["X", "Y", "X"], coeffs=[t / 2, t, t / 2])
    bch_expansion = effective_hamiltonian(second_order_pf, fragments, order=9)

    second_order_matrix = (
        expm(-0.5j * t * fragments["X"])
        @ expm(-1j * t * fragments["Y"])
        @ expm(-1j * t * fragments["X"])
    )
    evals, evecs = np.linalg.eig(second_order_matrix)
    theta = np.angle(evals)
    matrix_log = evecs @ np.diag(theta) @ np.linalg.inv(evecs)

    matrix_log2 = logm(second_order_matrix)

    print(expm(bch_expansion))
    print(expm(matrix_log))
    print(expm(matrix_log2))

    assert np.allclose(bch_expansion, matrix_log)
