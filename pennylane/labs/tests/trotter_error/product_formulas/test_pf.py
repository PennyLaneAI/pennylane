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
"""Tests for the product formula representations"""

from itertools import product

import numpy as np
import pytest
from scipy.sparse.linalg import expm

from pennylane.labs.trotter_error import ProductFormula, effective_hamiltonian
from pennylane.labs.trotter_error.abstract import nested_commutator


def _hermitian(mat):
    return mat + np.conj(mat).T


fragment_dicts = [
    {0: np.zeros(shape=(3, 3)), 1: np.zeros(shape=(3, 3))},
    {0: np.ones(shape=(3, 3)), 1: np.ones(shape=(3, 3))},
    {0: np.diag([2, 2, 2]), 1: np.diag([3, 3, 3])},
    {0: np.array([[0, 1], [1, 0]]), 1: np.array([[1, 0], [0, -1]])},
    {0: _hermitian(np.random.random(size=(3, 3))), 1: _hermitian(np.random.random(size=(3, 3)))},
    {0: np.random.random(size=(3, 3)), 1: np.random.random(size=(3, 3))},
    {0: np.random.random(size=(2, 2)), 1: np.random.random(size=(2, 2))},
    {
        0: np.random.random(size=(3, 3)),
        1: np.random.random(size=(3, 3)),
        2: np.random.random(size=(3, 3)),
    },
]


@pytest.mark.parametrize("fragment_dict", fragment_dicts)
def test_second_order_representations(fragment_dict):
    """Test that two representations of second order Trotter produce the same effective Hamiltonian"""

    pf1 = ProductFormula([0, 1, 0], coeffs=[1 / 2, 1, 1 / 2])
    pf2 = ProductFormula([0, 1, 1, 0], coeffs=[1 / 2, 1 / 2, 1 / 2, 1 / 2])

    eff1 = effective_hamiltonian(pf1, fragment_dict, order=5)
    eff2 = effective_hamiltonian(pf2, fragment_dict, order=5)

    assert np.allclose(eff1, eff2)


@pytest.mark.parametrize("fragment_dict", fragment_dicts)
def test_second_order_recursive(fragment_dict):
    """Test that the recursively defined product formula yields the same effective Hamiltonian as the unraveled product formula"""

    first_order_1 = ProductFormula([0, 1], coeffs=[1 / 2, 1 / 2])
    first_order_2 = ProductFormula([1, 0], coeffs=[1 / 2, 1 / 2])

    explicit = ProductFormula([0, 1, 1, 0], coeffs=[1 / 2, 1 / 2, 1 / 2, 1 / 2])
    constructed = first_order_1 @ first_order_2

    eff1 = effective_hamiltonian(explicit, fragment_dict, order=3)
    eff2 = effective_hamiltonian(constructed, fragment_dict, order=3)

    assert np.allclose(eff1, eff2)


@pytest.mark.parametrize("fragment_dict", fragment_dicts)
def test_fourth_order_recursive(fragment_dict):
    """Test that the recursively defined product formula yields the same effective Hamiltonian as the unraveled product formula"""

    u = 1 / (4 - 4 ** (1 / 3))
    v = 1 - 4 * u

    n_frags = len(fragment_dict)

    frag_labels = (list(range(n_frags)) + list(reversed(range(n_frags)))) * 5
    frag_coeffs = [u / 2] * (n_frags * 4) + [v / 2] * (n_frags * 2) + [u / 2] * (n_frags * 4)

    fourth_order1 = ProductFormula(frag_labels, coeffs=frag_coeffs)

    frag_labels = list(range(n_frags)) + list(reversed(range(n_frags)))
    frag_coeffs = [1 / 2] * len(frag_labels)

    second_order = ProductFormula(frag_labels, frag_coeffs)
    fourth_order2 = second_order(u) ** 2 @ second_order(1 - 4 * u) @ second_order(u) ** 2

    eff1 = effective_hamiltonian(fourth_order1, fragment_dict, order=5)
    eff2 = effective_hamiltonian(fourth_order2, fragment_dict, order=5)

    assert np.allclose(eff1, eff2)


@pytest.mark.parametrize("fragment_dict", fragment_dicts)
def test_fourth_order_mult(fragment_dict):
    """Test that fourth order Trotter can be built from multiplying second order Trotters"""
    t = 0.1
    u = 1 / (4 - 4 ** (1 / 3))

    second_order_labels = [0, 1, 1, 0]
    second_order_coeffs = [1 / 2, 1 / 2, 1 / 2, 1 / 2]
    second_order = ProductFormula(second_order_labels, coeffs=second_order_coeffs, label="U2") ** 2

    pfs = [(second_order**2)(t * u), second_order(t * (1 - 4 * u)), (second_order**2)(t * u)]

    fourth_order_1 = ProductFormula(pfs)
    fourth_order_2 = (
        (second_order**2)(t * u) @ second_order(t * (1 - 4 * u)) @ (second_order**2)(t * u)
    )

    eff_1 = effective_hamiltonian(fourth_order_1, fragment_dict, order=5)
    eff_2 = effective_hamiltonian(fourth_order_2, fragment_dict, order=5)

    assert np.allclose(eff_1, eff_2)


@pytest.mark.parametrize("fragment_dict", fragment_dicts)
def test_mul_and_pow(fragment_dict):
    """Test that multiplying a product formula by itself and raising it to a power give the same result"""

    n_frags = len(fragment_dict)
    second_order_labels = list(range(n_frags)) + list(reversed(range(n_frags)))
    second_order_coeffs = [1 / 2] * len(second_order_labels)
    second_order = ProductFormula(second_order_labels, coeffs=second_order_coeffs, label="U2")

    pf1 = second_order @ second_order @ second_order
    pf2 = second_order**3
    pf3 = ProductFormula([second_order, second_order, second_order])
    pf4 = ProductFormula(second_order_labels * 3, second_order_coeffs * 3)

    eff1 = effective_hamiltonian(pf1, fragment_dict, order=7)
    eff2 = effective_hamiltonian(pf2, fragment_dict, order=7)
    eff3 = effective_hamiltonian(pf3, fragment_dict, order=7)
    eff4 = effective_hamiltonian(pf4, fragment_dict, order=7)

    assert np.allclose(eff1, eff2)
    assert np.allclose(eff2, eff3)
    assert np.allclose(eff3, eff4)


@pytest.mark.parametrize("fragment_dict", fragment_dicts)
def test_matrix(fragment_dict):
    """Test that two ways of writing fourth order Trotter produce the same matrix."""

    u = 1 / (4 - 4 ** (1 / 3))
    frag_labels = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
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

    fourth_order_1 = ProductFormula(frag_labels, coeffs=frag_coeffs, label="U4")

    second_order_labels = [0, 1, 0]
    second_order_coeffs = [1 / 2, 1, 1 / 2]

    second_order = ProductFormula(second_order_labels, coeffs=second_order_coeffs, label="U2")

    pfs = [
        second_order(u) ** 2,
        second_order(1 - (4 * u)),
        second_order(u) ** 2,
    ]

    fourth_order_2 = ProductFormula(pfs, label="U4")

    mat1 = fourth_order_1.to_matrix(fragment_dict)
    mat2 = fourth_order_2.to_matrix(fragment_dict)

    assert np.allclose(mat1, mat2)


@pytest.mark.parametrize("fragments, t", product(fragment_dicts[:-1], [1, 0.1, 0.01]))
def test_fourth_order_norm_two_fragments(fragments, t):
    """Tests against an upper bound on the norm of the fourth order Trotter formula. This test comes from
    Proposition J.1 of https://arxiv.org/pdf/1912.08854"""

    u = 1 / (4 - 4 ** (1 / 3))
    frag_labels = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
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

    fourth_order = ProductFormula(frag_labels, coeffs=frag_coeffs)(1j * t)
    fourth_order_approx = fourth_order.to_matrix(fragments)
    actual = expm(1j * t * sum(fragments.values(), np.zeros_like(fragments[0])))

    commutator_coeffs = {
        (0, 0, 0, 1, 0): 0.0047,
        (0, 0, 1, 1, 0): 0.0057,
        (0, 1, 0, 1, 0): 0.0046,
        (0, 1, 1, 1, 0): 0.0074,
        (1, 0, 0, 1, 0): 0.0097,
        (1, 0, 1, 1, 0): 0.0097,
        (1, 1, 0, 1, 0): 0.0173,
        (1, 1, 1, 1, 0): 0.0284,
    }

    upper_bound = 0
    for comm, coeff in commutator_coeffs.items():
        mat = nested_commutator([fragments[frag] for frag in comm])
        upper_bound += coeff * np.linalg.norm(mat)

    difference = np.linalg.norm(fourth_order_approx - actual)

    assert difference <= (t**5) * upper_bound or np.isclose(difference, (t**5) * upper_bound)
