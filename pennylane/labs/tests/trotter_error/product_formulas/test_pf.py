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

import numpy as np
import pytest

from trotter_error import NumpyFragment, ProductFormula, effective_hamiltonian


def _hermitian(mat):
    return mat + np.conj(mat).T


u = 1 / (4 - 4 ** (1 / 3))
v = 1 - 4 * u
pf1 = ProductFormula(list(zip([0, 1, 2, 1, 0], [1 / 2, 1 / 2, 1, 1 / 2, 1 / 2])))
pf2 = ProductFormula.prod([pf1(u) ** 2, pf1(v), pf1(u) ** 2])
pf3 = ProductFormula(list(zip([0, 1, 2], [1, 1, 1])))
pf4 = ProductFormula.prod([pf1(u), pf1(v)])


@pytest.mark.parametrize(
    "product_formula, is_symmetric",
    [
        (pf1, True),
        (pf1(u), True),
        (pf1(v), True),
        (pf2, True),
        (pf3, False),
        (pf4, False),
    ],
)
def test_symmetry(product_formula, is_symmetric):
    """Test the is_symmetric property"""
    assert product_formula.is_symmetric == is_symmetric


fragment_dicts = [
    {0: NumpyFragment(np.zeros(shape=(3, 3))), 1: NumpyFragment(np.zeros(shape=(3, 3)))},
    {0: NumpyFragment(np.ones(shape=(3, 3))), 1: NumpyFragment(np.ones(shape=(3, 3)))},
    {0: NumpyFragment(np.diag([2, 2, 2])), 1: NumpyFragment(np.diag([3, 3, 3]))},
    {0: NumpyFragment(np.array([[0, 1], [1, 0]])), 1: NumpyFragment(np.array([[1, 0], [0, -1]]))},
    {
        0: NumpyFragment(_hermitian(np.random.random(size=(3, 3)))),
        1: NumpyFragment(_hermitian(np.random.random(size=(3, 3)))),
    },
    {
        0: NumpyFragment(np.random.random(size=(3, 3))),
        1: NumpyFragment(np.random.random(size=(3, 3))),
    },
    {
        0: NumpyFragment(np.random.random(size=(2, 2))),
        1: NumpyFragment(np.random.random(size=(2, 2))),
    },
    {
        0: NumpyFragment(np.random.random(size=(3, 3))),
        1: NumpyFragment(np.random.random(size=(3, 3))),
        2: NumpyFragment(np.random.random(size=(3, 3))),
    },
]


@pytest.mark.parametrize("fragment_dict", fragment_dicts)
def test_second_order_representations(fragment_dict):
    """Test that two representations of second order Trotter produce the same effective
    Hamiltonian"""

    second_order1 = ProductFormula(list(zip([0, 1, 0], [1 / 2, 1, 1 / 2])))
    second_order2 = ProductFormula(list(zip([0, 1, 1, 0], [1 / 2, 1 / 2, 1 / 2, 1 / 2])))

    eff1 = effective_hamiltonian(second_order1, fragment_dict, order=5)
    eff2 = effective_hamiltonian(second_order2, fragment_dict, order=5)

    assert np.allclose(eff1.fragment, eff2.fragment)


@pytest.mark.parametrize("fragment_dict", fragment_dicts)
def test_second_order_recursive(fragment_dict):
    """Test that the recursively defined product formula yields the same effective Hamiltonian as
    the unraveled product formula"""

    first_order_1 = ProductFormula(list(zip([0, 1], [1 / 2, 1 / 2])))
    first_order_2 = ProductFormula(list(zip([1, 0], [1 / 2, 1 / 2])))

    explicit = ProductFormula(list(zip([0, 1, 1, 0], [1 / 2, 1 / 2, 1 / 2, 1 / 2])))
    constructed = ProductFormula.prod([first_order_1, first_order_2])

    eff1 = effective_hamiltonian(explicit, fragment_dict, order=3)
    eff2 = effective_hamiltonian(constructed, fragment_dict, order=3)

    assert np.allclose(eff1.fragment, eff2.fragment)


@pytest.mark.parametrize("fragment_dict", fragment_dicts)
def test_fourth_order_recursive(fragment_dict):
    """Test that the recursively defined product formula yields the same effective Hamiltonian as
    the unraveled product formula"""

    n_frags = len(fragment_dict)

    frag_labels = (list(range(n_frags)) + list(reversed(range(n_frags)))) * 5
    frag_coeffs = [u / 2] * (n_frags * 4) + [v / 2] * (n_frags * 2) + [u / 2] * (n_frags * 4)

    fourth_order1 = ProductFormula(list(zip(frag_labels, frag_coeffs)))

    frag_labels = list(range(n_frags)) + list(reversed(range(n_frags)))
    frag_coeffs = [1 / 2] * len(frag_labels)

    second_order = ProductFormula(list(zip(frag_labels, frag_coeffs)))
    fourth_order2 = ProductFormula.prod(
        [second_order(u) ** 2, second_order(v), second_order(u) ** 2]
    )

    eff1 = effective_hamiltonian(fourth_order1, fragment_dict, order=5)
    eff2 = effective_hamiltonian(fourth_order2, fragment_dict, order=5)

    assert np.allclose(eff1.fragment, eff2.fragment)


@pytest.mark.parametrize("fragment_dict", fragment_dicts)
def test_mul_and_pow(fragment_dict):
    """Test that multiplying a product formula by itself and raising it to a power give the
    same result"""

    n_frags = len(fragment_dict)
    second_order_labels = list(range(n_frags)) + list(reversed(range(n_frags)))
    second_order_coeffs = [1 / 2] * len(second_order_labels)
    second_order = ProductFormula(list(zip(second_order_labels, second_order_coeffs)))

    mul1 = ProductFormula.prod([second_order, second_order, second_order])
    mul2 = second_order**3
    mul3 = ProductFormula(list(zip(second_order_labels * 3, second_order_coeffs * 3)))

    eff1 = effective_hamiltonian(mul1, fragment_dict, order=5)
    eff2 = effective_hamiltonian(mul2, fragment_dict, order=5)
    eff3 = effective_hamiltonian(mul3, fragment_dict, order=5)

    assert np.allclose(eff1.fragment, eff2.fragment)
    assert np.allclose(eff2.fragment, eff3.fragment)
