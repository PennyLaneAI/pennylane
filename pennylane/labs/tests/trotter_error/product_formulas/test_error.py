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

    print(f"{backend}, {mpi4py_support}")

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
    errors, convergence_info = perturbation_error(
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
    assert isinstance(convergence_info, dict)
    assert len(convergence_info) == 2  # Two states

    # Check convergence info structure
    for state_idx in [0, 1]:
        assert state_idx in convergence_info
        state_info = convergence_info[state_idx]
        assert isinstance(state_info, dict)

        # Check that each order has the required keys
        for _, order_info in state_info.items():
            assert isinstance(order_info, dict)
            required_keys = {"mean_history", "median_history", "std_history"}
            assert required_keys.issubset(order_info.keys())

            # Check that histories are lists
            assert isinstance(order_info["mean_history"], list)
            assert isinstance(order_info["median_history"], list)
            assert isinstance(order_info["std_history"], list)

            # All histories should have the same length
            hist_lengths = [
                len(order_info["mean_history"]),
                len(order_info["median_history"]),
                len(order_info["std_history"]),
            ]
            assert len(set(hist_lengths)) == 1  # All lengths should be equal


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
        _, _ = perturbation_error(
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


def test_perturbation_error_consistency():
    """Test that all execution modes give consistent results."""
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

    # Test serial mode
    errors_serial, conv_serial = perturbation_error(
        pf, frags, [state1, state2], max_order=3, backend="serial"
    )

    # Test parallel state mode
    errors_par_state, conv_par_state = perturbation_error(
        pf,
        frags,
        [state1, state2],
        max_order=3,
        backend="cf_threadpool",
        parallel_mode="state",
        num_workers=2,
    )

    # Test parallel commutator mode
    errors_par_comm, conv_par_comm = perturbation_error(
        pf,
        frags,
        [state1, state2],
        max_order=3,
        backend="cf_threadpool",
        parallel_mode="commutator",
        num_workers=2,
    )

    # All modes should give the same errors
    assert errors_serial == errors_par_state
    assert errors_serial == errors_par_comm

    # All modes should give the same convergence info structure
    assert list(conv_serial.keys()) == list(conv_par_state.keys()) == list(conv_par_comm.keys())

    for state_idx in conv_serial.keys():
        assert list(conv_serial[state_idx].keys()) == list(conv_par_state[state_idx].keys())
        assert list(conv_serial[state_idx].keys()) == list(conv_par_comm[state_idx].keys())


def test_convergence_info_structure():
    """Test the structure and content of convergence info."""
    frag_labels = [0, 1, 2]  # Multiple fragments for more complex convergence
    frag_coeffs = [1 / 3, 1 / 3, 1 / 3]
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

    # Create 3 fragments
    vibs = list(vibrational_fragments(n_modes, freqs, taylor_coeffs))
    frags = {}
    for i in range(3):
        frags[i] = vibs[i % len(vibs)]

    gridpoints = 5
    states = [HOState(n_modes, gridpoints, {(0, 0): 1})]

    _, convergence_info = perturbation_error(pf, frags, states, max_order=4)

    # Check basic structure
    assert len(convergence_info) == 1  # One state
    state_info = convergence_info[0]

    # Check that we have multiple orders with non-trivial histories
    for _, order_info in state_info.items():
        mean_hist = order_info["mean_history"]
        median_hist = order_info["median_history"]
        std_hist = order_info["std_history"]

        # All histories should be non-empty
        assert len(mean_hist) > 0
        assert len(median_hist) > 0
        assert len(std_hist) > 0

        # All histories should have the same length
        assert len(mean_hist) == len(median_hist) == len(std_hist)

        # For orders with multiple commutators, we should see evolution
        if len(mean_hist) > 1:
            # Standard deviation should start at 0 (first element)
            assert std_hist[0] == 0.0
            # Values should be complex numbers
            assert all(isinstance(val, (complex, float, int)) for val in mean_hist)
            assert all(isinstance(val, (complex, float, int)) for val in median_hist)
            assert all(isinstance(val, (float, int)) for val in std_hist)
