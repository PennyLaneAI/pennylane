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

from pennylane.labs.trotter_error import (
    HOState,
    ProductFormula,
    perturbation_error,
    vibrational_fragments,
)
from pennylane.labs.trotter_error.product_formulas.error import _group_sums


@pytest.mark.parametrize(
    "backend", ["serial", "mp_pool", "cf_procpool", "mpi4py_pool", "mpi4py_comm"]
)
@pytest.mark.parametrize("parallel_mode", ["state", "commutator"])
def test_perturbation_error(backend, parallel_mode, mpi4py_support):
    """Test that perturbation error function runs without errors for different backends."""

    if backend in {"mpi4py_pool", "mpi4py_comm"} and not mpi4py_support:
        pytest.skip(f"Skipping test: '{backend}' requires mpi4py, which is not installed.")

    frag_labels = [0, 1, 1, 0]
    frag_coeffs = [1 / 2, 1 / 2, 1 / 2, 1 / 2]
    pf = ProductFormula(frag_labels, coeffs=frag_coeffs)
    n_modes = 2
    r_state = np.random.RandomState(42)
    freqs = r_state.random(n_modes)
    taylor_coeffs = [
        np.array(0),
        r_state.random(size=(n_modes,)),
        r_state.random(size=(n_modes, n_modes)),
        r_state.random(size=(n_modes, n_modes, n_modes)),
    ]
    frags = dict(enumerate(vibrational_fragments(n_modes, freqs, taylor_coeffs)))
    gridpoints = 5
    state1 = HOState(n_modes, gridpoints, {(0, 0): 1})
    state2 = HOState(n_modes, gridpoints, {(1, 1): 1})

    num_workers = 1 if backend == "serial" else 2
    errors = perturbation_error(
        pf,
        frags,
        [state1, state2],
        max_order=3,
        num_workers=num_workers,
        backend=backend,
        parallel_mode=parallel_mode,
    )

    assert isinstance(errors, list)
    assert len(errors) == 2


def test_perturbation_error_invalid_parallel_mode():
    """Test that perturbation error raises an error for invalid parallel mode."""
    frag_labels = [0, 1, 1, 0]
    frag_coeffs = [1 / 2, 1 / 2, 1 / 2, 1 / 2]
    pf = ProductFormula(frag_labels, coeffs=frag_coeffs)
    n_modes = 2
    r_state = np.random.RandomState(42)
    freqs = r_state.random(n_modes)
    taylor_coeffs = [
        np.array(0),
        r_state.random(size=(n_modes,)),
        r_state.random(size=(n_modes, n_modes)),
        r_state.random(size=(n_modes, n_modes, n_modes)),
    ]
    frags = dict(enumerate(vibrational_fragments(n_modes, freqs, taylor_coeffs)))
    gridpoints = 5
    state1 = HOState(n_modes, gridpoints, {(0, 0): 1})
    state2 = HOState(n_modes, gridpoints, {(1, 1): 1})

    with pytest.raises(ValueError, match="Invalid parallel mode"):
        perturbation_error(
            pf,
            frags,
            [state1, state2],
            max_order=3,
            num_workers=1,
            backend="mp_pool",
            parallel_mode="invalid_mode",
        )


@pytest.mark.parametrize(
    "term_dict, expected",
    [
        (
            {("X", "A", "B"): 4, ("Y", "A", "B"): 3},
            [(frozenset({("X", 4), ("Y", 3)}), "A", "B")],
        ),
        (
            {("X", "A", "B"): 4, ("Y", "A", "C"): 3},
            [(frozenset({("X", 4)}), "A", "B"), (frozenset({("Y", 3)}), "A", "C")],
        ),
    ],
)
def test_group_sums(term_dict, expected):
    """Test the private _group_sums method"""
    assert _group_sums(term_dict) == expected
