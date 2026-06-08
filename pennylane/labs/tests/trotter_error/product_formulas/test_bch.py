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

from pennylane.labs.trotter_error import NumpyFragment, ProductFormula, effective_hamiltonian
from pennylane.labs.trotter_error.abstract import nested_commutator
from pennylane.labs.trotter_error.product_formulas.bch import bch_expansion
from pennylane.labs.trotter_error.product_formulas.commutator import CommutatorNode, SymbolNode


deltas = [1, 0.5, 0.1, 0.01]

np.random.seed(42)
fragment_list = [
    {
        0: NumpyFragment(np.zeros(shape=(3, 3))),
        1: NumpyFragment(np.zeros(shape=(3, 3))),
        2: NumpyFragment(np.zeros(shape=(3, 3))),
    },
    {
        0: NumpyFragment(np.random.random(size=(3, 3))),
        1: NumpyFragment(np.random.random(size=(3, 3))),
    },
    {
        0: NumpyFragment(np.random.random(size=(3, 3))),
        1: NumpyFragment(np.random.random(size=(3, 3))),
        2: NumpyFragment(np.random.random(size=(3, 3))),
    },
]


@pytest.mark.parametrize("fragments, r, delta", product(fragment_list, range(1, 4), deltas))
def test_first_order(fragments, r, delta):
    """Test that the effective_hamiltonian function returns the correct result for first order
    Trotter. See Proposition 4 of https://arxiv.org/pdf/2408.03891 for the expression."""

    n_frags = len(fragments)
    expected = NumpyFragment(np.zeros(shape=(3, 3), dtype=np.complex128))
    ham = sum(fragments.values(), NumpyFragment(np.zeros(shape=(3, 3))))

    for j in range(n_frags - 1):
        for k in range(j + 1, n_frags):
            expected += (1 / 2) * nested_commutator([fragments[j], fragments[k]])

    expected *= 1j / r * delta

    first_order = ProductFormula(list(zip(range(n_frags), [delta / r] * n_frags))) ** r
    actual = effective_hamiltonian(first_order, fragments, order=2)

    assert np.allclose(1j * delta * (expected + ham).fragment, actual.fragment)


@pytest.mark.parametrize("fragments, r, delta", product(fragment_list, range(1, 4), deltas))
def test_second_order(fragments, r, delta):
    """Test that the effective_hamiltonian function returns the correct result for second order
    Trotter. See Proposition 4 of https://arxiv.org/pdf/2408.03891 for the expression."""

    n_frags = len(fragments)
    frag_labels = list(range(n_frags)) + list(range(n_frags))[::-1]
    coeffs = [1 / (2 * r)] * n_frags * 2

    pf = ProductFormula(list(zip(frag_labels, coeffs))) ** r
    actual = effective_hamiltonian(pf, fragments, order=3, timestep=delta)

    expected = NumpyFragment(np.zeros(shape=(3, 3), dtype=np.complex128))

    for i in range(n_frags - 1):
        for j in range(i + 1, n_frags):
            expected += nested_commutator([fragments[i], fragments[i], fragments[j]])
            for k in range(i + 1, n_frags):
                expected += 2 * nested_commutator([fragments[k], fragments[i], fragments[j]])

    expected *= -1 / 24
    expected *= (1j * delta / r) ** 2

    ham = sum(fragments.values(), NumpyFragment(np.zeros(shape=(3, 3))))
    eff = 1j * delta * (expected + ham)

    assert np.allclose(eff.fragment, actual.fragment)


@pytest.mark.parametrize("fragments, delta", product(fragment_list, deltas))
def test_fourth_order(fragments, delta):
    """Test that the effective_hamiltonian function returns the correct result for fourth order
    Trotter. The expected Hamiltonian was generated via Sympy."""
    u = 1 / (4 - 4 ** (1 / 3))
    v = 1 - 4 * u
    y3 = -0.00405944185443219
    y5 = -0.074375995396295

    frag_labels = list(range(len(fragments))) + list(range(len(fragments)))[::-1]
    frag_coeffs = [1 / 2] * len(frag_labels)
    second_order = ProductFormula(list(zip(frag_labels, frag_coeffs)))
    fourth_order = ProductFormula.prod(
        [second_order(u) ** 2, second_order(v), second_order(u) ** 2]
    )

    ham = 1j * delta * (sum(fragments.values(), NumpyFragment(np.zeros_like(fragments[0]))))
    expected = copy.copy(ham)

    fragments = copy.deepcopy(fragments)
    fragments["ham"] = ham

    bch = bch_expansion(second_order, max_order=5)
    bch3 = {comm: coeff for comm, coeff in bch.items() if comm.order == 3}
    bch5 = {comm: coeff for comm, coeff in bch.items() if comm.order == 5}

    for commutator, coeff in bch3.items():
        new_commutator = CommutatorNode(
            CommutatorNode(SymbolNode("ham"), commutator), SymbolNode("ham")
        )
        expected += y3 * coeff * (delta * 1j) ** 3 * new_commutator.eval(fragments)
    for commutator, coeff in bch5.items():
        expected += y5 * coeff * (delta * 1j) ** 5 * commutator.eval(fragments)

    actual = effective_hamiltonian(fourth_order, fragments, order=5, timestep=delta)

    assert np.allclose(expected.fragment, actual.fragment)


@pytest.mark.skip(reason="Slow")
@pytest.mark.parametrize("fragments, delta", product(fragment_list, deltas))
def test_sixth_order(fragments, delta):
    """Test that the effective_hamiltonian function returns the correct result for sixth order
    Trotter. The expected Hamiltonian was generated via Sympy."""
    u4 = 1 / (4 - 4 ** (1 / 3))
    u6 = 1 / (4 - 4 ** (1 / 5))
    v4 = 1 - 4 * u4
    v6 = 1 - 4 * u6

    h_y3_hhh = -2.76996721648715 * 1e-6
    h_y3_y3 = -1.0079627372761 * 1e-5
    h_y5_h = 5.28014551264278 * 1e-5
    y7 = 1.34083888042369 * 1e-4

    frag_labels = list(range(len(fragments))) + list(reversed(range(len(fragments))))
    frag_coeffs = [1 / 2] * len(frag_labels)
    second_order = ProductFormula(list(zip(frag_labels, frag_coeffs)))
    fourth_order = ProductFormula.prod(
        [second_order(u4) ** 2, second_order(v4), second_order(u4) ** 2]
    )
    sixth_order = ProductFormula.prod(
        [fourth_order(u6) ** 2, fourth_order(v6), fourth_order(u6) ** 2]
    )

    ham = 1j * delta * (sum(fragments.values(), NumpyFragment(np.zeros_like(fragments[0]))))
    expected = copy.copy(ham)

    fragments = copy.deepcopy(fragments)
    fragments["ham"] = ham

    bch = bch_expansion(second_order, max_order=7)
    bch3 = {comm: coeff for comm, coeff in bch.items() if comm.order == 3}
    bch5 = {comm: coeff for comm, coeff in bch.items() if comm.order == 5}
    bch7 = {comm: coeff for comm, coeff in bch.items() if comm.order == 7}

    for (comm1, coeff1), (comm2, coeff2) in product(bch3.items(), repeat=2):
        new_commutator = CommutatorNode(CommutatorNode(SymbolNode("ham"), comm1), comm2)
        expected += h_y3_y3 * coeff1 * coeff2 * (1j * delta) ** 6 * new_commutator.eval(fragments)

    for commutator, coeff in bch3.items():
        new_commutator = CommutatorNode(
            CommutatorNode(
                CommutatorNode(
                    CommutatorNode(SymbolNode("ham"), commutator),
                    SymbolNode("ham"),
                ),
                SymbolNode("ham"),
            ),
            SymbolNode("ham"),
        )
        expected += (
            h_y3_hhh * coeff * (1j * delta) ** commutator.order * new_commutator.eval(fragments)
        )

    for commutator, coeff in bch5.items():
        new_commutator = CommutatorNode(
            SymbolNode("ham"), CommutatorNode(commutator, SymbolNode("ham"))
        )
        expected += (
            h_y5_h * coeff * (1j * delta) ** commutator.order * new_commutator.eval(fragments)
        )

    for commutator, coeff in bch7.items():
        expected += y7 * coeff * (1j * delta) ** commutator.order * commutator.eval(fragments)

    actual = effective_hamiltonian(sixth_order, fragments, order=7, timestep=delta)

    assert np.allclose(expected.fragment, actual.fragment)


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

    fragments = {
        0: NumpyFragment(np.random.random(size=(3, 3))),
        1: NumpyFragment(np.random.random(size=(3, 3))),
    }
    n_frags = len(fragments)

    first_order = ProductFormula(list(zip(list(range(n_frags)), [1 / r] * n_frags))) ** r
    actual = effective_hamiltonian(
        first_order, fragments, order=2, timestep=delta, num_workers=num_workers, backend=backend
    )

    ham = sum(fragments.values(), NumpyFragment(np.zeros(shape=(3, 3))))
    expected = NumpyFragment(np.zeros(shape=(3, 3), dtype=np.complex128))
    for j in range(n_frags - 1):
        for k in range(j + 1, n_frags):
            expected += (1 / 2) * nested_commutator([fragments[j], fragments[k]])
    expected *= 1j / r * delta

    assert np.allclose(1j * delta * (expected + ham).fragment, actual.fragment)
