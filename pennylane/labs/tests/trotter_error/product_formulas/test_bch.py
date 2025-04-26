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
    "frag_labels, frag_coeffs, max_order, expected",
    [
        (
            ["X", "Y"],
            [1, 1],
            4,
            [
                {(("X", 1),): 1, (("Y", 1),): 1},
                {(("X", 1), ("Y", 1)): 1 / 2},
                {(("X", 1), ("X", 1), ("Y", 1)): 1 / 12, (("Y", 1), ("X", 1), ("Y", 1)): -1 / 12},
                {(("X", 1), ("Y", 1), ("X", 1), ("Y", 1)): -1 / 24},
            ],
        ),
        (
            ["X", "Y", "Z"],
            [1, 1, 1],
            3,
            [
                {(("X", 1),): 1, (("Y", 1),): 1, (("Z", 1),): 1},
                {
                    (("X", 1), ("Y", 1)): 1 / 2,
                    (("X", 1), ("Z", 1)): 1 / 2,
                    (("Y", 1), ("Z", 1)): 1 / 2,
                },
                {
                    (("X", 1), ("X", 1), ("Y", 1)): 1 / 12,
                    (("X", 1), ("X", 1), ("Z", 1)): 1 / 12,
                    (("X", 1), ("Y", 1), ("Z", 1)): 1 / 3,
                    (("Y", 1), ("X", 1), ("Y", 1)): -1 / 12,
                    (("Y", 1), ("X", 1), ("Z", 1)): -1 / 6,
                    (("Y", 1), ("Y", 1), ("Z", 1)): 1 / 12,
                    (("Z", 1), ("X", 1), ("Z", 1)): -1 / 12,
                    (("Z", 1), ("Y", 1), ("Z", 1)): -1 / 12,
                },
            ],
        ),
        (
            ["X", "Y", "X"],
            [1, 1, 1],
            3,
            [
                {(("X", 1),): 2, (("Y", 1),): 1},
                {},
                {(("X", 1), ("X", 1), ("Y", 1)): -1 / 6, (("Y", 1), ("X", 1), ("Y", 1)): -1 / 6},
            ],
        ),
    ],
)
def test_bch(frag_labels, frag_coeffs, max_order, expected):
    """Test against BCH expansion. The expected values come from Sections 4 and 5 of `arXiv:2006.15869 <https://arxiv.org/pdf/2006.15869>`"""

    actual = ProductFormula(frag_labels, coeffs=frag_coeffs).bch_approx(max_order=max_order)

    for i, order in enumerate(actual):
        for commutator in order:
            assert np.isclose(order[commutator], expected[i][commutator])
