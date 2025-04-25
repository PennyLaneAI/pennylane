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

from pennylane.labs.trotter_error import ProductFormula, effective_hamiltonian

fragment_dicts = [
    {0: np.zeros(shape=(3, 3)), 1: np.zeros(shape=(3, 3))},
    {0: np.random.random(size=(3, 3)), 1: np.random.random(size=(3, 3))},
]


@pytest.mark.parametrize("fragment_dict", fragment_dicts)
def test_second_order_representations(fragment_dict):

    pf1 = ProductFormula([0, 1, 0], coeffs=[1 / 2, 1, 1 / 2])
    pf2 = ProductFormula([0, 1, 1, 0], coeffs=[1 / 2, 1 / 2, 1 / 2, 1 / 2])

    eff1 = effective_hamiltonian(pf1, fragment_dict, order=5)
    eff2 = effective_hamiltonian(pf2, fragment_dict, order=5)

    assert np.allclose(eff1, eff2)


@pytest.mark.parametrize("fragment_dict", fragment_dicts)
def test_second(fragment_dict):

    t = 1
    u = 1 / (4 - 4 ** (1 / 3))
    second_order_labels = [0, 1, 0]
    second_order_coeffs = [1 / 2, 1, 1 / 2]
    second_order = ProductFormula(second_order_labels, coeffs=second_order_coeffs, label="U")

    # pf1 = ProductFormula([0, 1, 0, 0, 1, 0], coeffs=[u/2, u, u/2, -u/2, -u, -u/2])
    # pf2 = second_order(u) @ second_order(-u)

    pf1 = ProductFormula(
        [0, 1, 0, 0, 1, 0], coeffs=[t * u / 2, t * u, t * u / 2, -t * u / 2, -t * u, -t * u / 2]
    )
    pf2 = second_order(t * u) @ second_order(-t * u)

    # f1 = effective_hamiltonian(second_order(t*u), fragment_dict, order=5)
    # f2 = effective_hamiltonian(second_order(t*u), fragment_dict, order=5)
    # print(f1)
    # print(f2)
    # print(f1 + f2)

    print("eff1")
    eff1 = effective_hamiltonian(pf1, fragment_dict, order=7)
    print("eff2")
    eff2 = effective_hamiltonian(pf2, fragment_dict, order=7)

    assert np.allclose(eff1, eff2)


@pytest.mark.parametrize("fragment_dict", fragment_dicts)
def test_fourth_order_recursive(fragment_dict):
    """Test that the recursively defined product formula yields the same effective Hamiltonian as the unraveled product formula"""

    t = 0.01
    u = 1 / (4 - 4 ** (1 / 3))
    frag_labels = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    frag_coeffs = [
        t * u / 2,
        t * u,
        t * u,
        t * u,
        t * (1 - (3 * u)) / 2,
        t * (1 - (4 * u)),
        t * (1 - (3 * u)) / 2,
        t * u,
        t * u,
        t * u,
        t * u / 2,
    ]

    fourth_order_1 = ProductFormula(frag_labels, coeffs=frag_coeffs, label="U4")

    second_order_labels = [0, 1, 0]
    second_order_coeffs = [1 / 2, 1, 1 / 2]

    second_order = ProductFormula(second_order_labels, coeffs=second_order_coeffs, label="U2")

    pfs = [
        second_order(t * u) ** 2,
        second_order(t * (1 - (4 * u))),
        second_order(t * u) ** 2,
    ]

    fourth_order_2 = ProductFormula(pfs, label="U4")

    eff_1 = effective_hamiltonian(fourth_order_1, fragment_dict, order=5)
    eff_2 = effective_hamiltonian(fourth_order_2, fragment_dict, order=5)

    print(eff_1)
    print(eff_2)

    assert np.allclose(eff_1, eff_2)


@pytest.mark.parametrize("fragment_dict", fragment_dicts)
def test_fourth_order_mult(fragment_dict):
    t = 0.1
    u = 1 / (4 - 4 ** (1 / 3))

    second_order_labels = [0, 1, 1, 0]
    second_order_coeffs = [1 / 2, 1 / 2, 1 / 2, 1 / 2]
    second_order = ProductFormula(second_order_labels, coeffs=second_order_coeffs, label="U2")

    pfs = [(second_order**2)(t * u), second_order(t * (1 - 4 * u)), (second_order**2)(t * u)]

    fourth_order_1 = ProductFormula(pfs)
    fourth_order_2 = (
        (second_order**2)(t * u) @ second_order(t * (1 - 4 * u)) @ (second_order**2)(t * u)
    )

    eff_1 = effective_hamiltonian(fourth_order_1, fragment_dict, order=5)
    eff_2 = effective_hamiltonian(fourth_order_2, fragment_dict, order=5)

    assert np.allclose(eff_1, eff_2)


@pytest.mark.parametrize("fragment_dict", fragment_dicts)
def test_pow(fragment_dict):
    second_order_labels = [0, 1, 1, 0]
    second_order_coeffs = [1 / 2, 1 / 2, 1 / 2, 1 / 2]

    second_order = ProductFormula(second_order_labels, coeffs=second_order_coeffs, label="U2")

    pf1 = second_order @ second_order @ second_order
    pf2 = second_order**3
    pf3 = ProductFormula([second_order, second_order, second_order])

    eff1 = effective_hamiltonian(pf1, fragment_dict, order=5)
    eff2 = effective_hamiltonian(pf2, fragment_dict, order=5)
    eff3 = effective_hamiltonian(pf3, fragment_dict, order=5)

    assert np.allclose(eff1, eff2)
    assert np.allclose(eff1, eff3)
    assert np.allclose(eff2, eff3)


@pytest.mark.parametrize("fragment_dict", fragment_dicts)
def test_mul(fragment_dict):

    t = 0.01
    frags = [0, 1, 0]
    coeffs = [1 / 2, 1, 1 / 2]

    second_order_trotter = ProductFormula(frags, coeffs=coeffs, label="U")

    pf1 = second_order_trotter(t) @ second_order_trotter(t)
    pf2 = ProductFormula([0, 1, 0, 0, 1, 0], coeffs=[t / 2, t, t / 2, t / 2, t, t / 2])
    pf3 = ProductFormula([second_order_trotter(t), second_order_trotter(t)])

    eff1 = effective_hamiltonian(pf1, fragment_dict, order=5)
    eff2 = effective_hamiltonian(pf2, fragment_dict, order=5)
    eff3 = effective_hamiltonian(pf3, fragment_dict, order=5)

    assert np.allclose(eff1, eff2)
    assert np.allclose(eff1, eff3)
    assert np.allclose(eff2, eff3)
