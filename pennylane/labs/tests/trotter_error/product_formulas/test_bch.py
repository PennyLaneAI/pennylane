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

import numpy as np
import pytest

from pennylane.labs.trotter_error import ProductFormula, effective_hamiltonian
from pennylane.labs.trotter_error.abstract import nested_commutator

fragments_list = [
    {0: np.zeros(shape=(3, 3)), 1: np.zeros(shape=(3, 3)), 2: np.zeros(shape=(3, 3))},
    {0: np.random.random(size=(3, 3)), 1: np.random.random(size=(3, 3))},
    {
        0: np.random.random(size=(3, 3)),
        1: np.random.random(size=(3, 3)),
        2: np.random.random(size=(3, 3)),
    },
]


@pytest.mark.parametrize("fragments", fragments_list)
def test_second_order(fragments):
    """Test against the second order Trotter error formula"""

    n_frags = len(fragments)
    frag_labels = list(range(n_frags)) + list(range(n_frags - 1, -1, -1))
    coeffs = [1 / 2] * len(frag_labels)

    pf = ProductFormula(coeffs, frag_labels)
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


def test_two_fragments():
    """Test against BCH expansion on two fragments. The expected value comes from Section 4 of `arXiv:2006.15869 <https://arxiv.org/pdf/2006.15869>`"""

    frag_labels = [0, 1]
    frag_coeffs = [1, 1]

    actual = ProductFormula(frag_coeffs, frag_labels).bch_approx(max_order=4)

    expected = [
        {(0,): 1, (1,): 1},
        {(0, 1): 1 / 2},
        {(0, 0, 1): 1 / 12, (1, 0, 1): -1 / 12},
        {(0, 1, 0, 1): -1 / 24},
    ]

    assert actual == expected


def test_three_fragments():
    """Test against BCH expansion on three fragments. The expected value comes from Section 5 of `arXiv:2006.15869 <https://arxiv.org/pdf/2006.15869>`"""

    frag_labels = [0, 1, 2]
    frag_coeffs = [1, 1, 1]

    actual = ProductFormula(frag_coeffs, frag_labels).bch_approx(max_order=3)
    expected = [
        {(0,): 1, (1,): 1, (2,): 1},
        {(0, 1): 1 / 2, (0, 2): 1 / 2, (1, 2): 1 / 2},
        {
            (0, 0, 1): 1 / 12,
            (0, 0, 2): 1 / 12,
            (0, 1, 2): 1 / 3,
            (1, 0, 1): -1 / 12,
            (1, 0, 2): -1 / 6,
            (1, 1, 2): 1 / 12,
            (2, 0, 2): -1 / 12,
            (2, 1, 2): -1 / 12,
        },
    ]

    assert actual == expected
