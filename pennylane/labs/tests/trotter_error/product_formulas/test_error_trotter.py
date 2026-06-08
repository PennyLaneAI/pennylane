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
"""Tests for the perturbation error function."""

import numpy as np
import pytest

from trotter_error import (
    ImportanceConfig,
    NumpyFragment,
    NumpyState,
    ProductFormula,
    bch_expansion,
    perturbation_error,
)


symbols = ["A", "B", "C", "D"]
frag_labels = symbols + symbols[::-1]
frag_coeffs = [1 / 2] * len(frag_labels)
second_order = ProductFormula(list(zip(frag_labels, frag_coeffs)))
u = 1 / (4 - 4 ** (1 / 3))
v = 1 - 4 * u
fourth_order = ProductFormula.prod([second_order(u) ** 2, second_order(v), second_order(u) ** 2])


@pytest.mark.parametrize(
    "product_formula, order",
    [
        (second_order, 3),
        (fourth_order, 5),
    ],
)
def test_parallel_modes_match_serial(product_formula, order, mpi4py_support):
    """Test perturbation_error against manual computation"""

    if not mpi4py_support:
        pytest.skip("Skipping test: requires mpi4py, which is not installed.")

    np.random.seed(42)
    fragments = {symbol: NumpyFragment(np.random.random(size=(3, 3))) for symbol in symbols}
    state = NumpyState(np.random.random(3))

    bch = bch_expansion(product_formula, max_order=order)
    comms = {comm: coeff for comm, coeff in bch.items() if comm.order > 1}
    bch_error = sum(
        (1j) ** comm.order * coeff * comm.eval(fragments).expectation(state, state)
        for comm, coeff in comms.items()
    )

    serial_error = perturbation_error(product_formula, fragments, state, order=order)

    parallel_error = perturbation_error(
        product_formula,
        fragments,
        state,
        order=order,
        backend="mpi4py_comm",
    )

    assert np.isclose(bch_error, serial_error[order])
    assert np.isclose(bch_error, parallel_error[order])


@pytest.mark.parametrize(
    "backend", ["serial", "mp_pool", "cf_procpool", "mpi4py_pool", "mpi4py_comm"]
)
def test_convergence_log(backend, mpi4py_support):
    """Test that the convergence log is valid."""

    if backend in {"mpi4py_pool", "mpi4py_comm"} and not mpi4py_support:
        pytest.skip(f"Skipping test: '{backend}' requires mpi4py, which is not installed.")

    topk = 10
    importance = ImportanceConfig(topk=10, weights={symbol: 1 for symbol in symbols}, history=True)
    np.random.seed(42)
    fragments = {symbol: NumpyFragment(np.random.random(size=(3, 3))) for symbol in symbols}
    state = NumpyState(np.random.random(3))

    error = perturbation_error(
        fourth_order,
        fragments,
        state,
        order=5,
        importance=importance,
        backend=backend,
    )

    assert 5 in error.keys()
    assert np.isclose(error[5]["error"], error[5]["partial sums"][-1])
    assert len(error[5]["partial sums"]) == topk

    if 3 in error.keys():
        assert np.isclose(error[3]["error"], error[3]["partial sums"][-1])
        assert len(error[3]["partial sums"]) == topk
