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
import pytest

import numpy as np

from pennylane.labs.trotter_error import ProductFormula, effective_hamiltonian

fragment_dicts = [
        #{ 0: np.zeros(shape=(3, 3)), 1: np.zeros(shape=(3, 3))},
        { 0: np.random.random(size=(3, 3)), 1: np.random.random(size=(3, 3))},
    ]

@pytest.mark.parametrize("fragment_dict", fragment_dicts)
def test_fourth_order_recursive(fragment_dict):
    """Test that the recursively defined product formula yields the same effective Hamiltonian as the unraveled product formula"""

    u = 1 / (4 - 4**(1/3))
    frag_labels1 = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    frag_coeffs1 = [u/2, u, u, u, (1-3*u)/2, 1 - 4*u, (1-3*u)/2, u, u, u, u/2]

    fourth_order_1 = ProductFormula(frag_coeffs1, frag_labels1)

    second_order =  ProductFormula([1/2, 1/2, 1/2, 1/2], [0, 1, 1, 0])

    frag_labels2 = [second_order**2, second_order, second_order**2]
    frag_coeffs2 = [u, 1 - 4*u, u]

    fourth_order_2 = ProductFormula(frag_coeffs2, frag_labels2)

    eff_1 = effective_hamiltonian(fourth_order_1, fragment_dict, order=5)
    eff_2 = effective_hamiltonian(fourth_order_2, fragment_dict, order=5)

    assert np.isclose(eff_1, eff_2)

fragment_dicts = [
        #{ 0: np.zeros(shape=(3, 3)), 1: np.zeros(shape=(3, 3))},
        { 0: np.random.random(size=(3, 3)), 1: np.random.random(size=(3, 3))},
    ]

@pytest.mark.parametrize("fragment_dict", fragment_dicts)
def test_fourth_order_mult(fragment_dict):
    u = 1 / (4 - 4**(1/3))
    frag_labels1 = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    frag_coeffs1 = [u/2, u, u, u, (1-3*u)/2, 1 - 4*u, (1-3*u)/2, u, u, u, u/2]

    second_order =  ProductFormula([1/2, 1/2, 1/2, 1/2], [0, 1, 1, 0])

    frag_labels = [second_order**2, second_order, second_order**2]
    frag_coeffs = [u, 1 - 4*u, u]

    fourth_order_1 = ProductFormula(frag_coeffs, frag_labels)

    fourth_order_2 = second_order**2 * second_order * second_order**2

    eff_1 = effective_hamiltonian(fourth_order_1, fragment_dict, order=5)
    eff_2 = effective_hamiltonian(fourth_order_2, fragment_dict, order=5)

    assert np.isclose(eff_1, eff_2)
