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
from pennylane.labs.trotter_error.product_formulas.error import (
    PerturbationErrorConfig,
    SamplingConfig,
    _CommutatorCache,
    _group_sums,
    _initialize_convergence_info,
)


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
    errors = perturbation_error(
        pf,
        frags,
        [state1, state2],
        order=3,
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
            order=3,
            num_workers=1,
            backend="mp_pool",
            parallel_mode="invalid_mode",
        )


@pytest.mark.parametrize(
    "term_dict, expected",
    [
        (
            [{("A",): 5}, {("X", "A", "B"): 4, ("Y", "A", "B"): 3}],
            [(frozenset({("X", 4), ("Y", 3)}), "A", "B")],
        ),
        (
            [{("A",): 5}, {("X", "A", "B"): 4, ("Y", "A", "C"): 3}],
            [(frozenset({("X", 4)}), "A", "B"), (frozenset({("Y", 3)}), "A", "C")],
        ),
    ],
)
def test_group_sums(term_dict, expected):
    """Test the private _group_sums method"""
    assert _group_sums(term_dict) == expected


def test_cache_functionality():
    """Test basic cache functionality and integration with error calculation."""
    cache = _CommutatorCache(max_size=5)

    # Test basic operations
    assert len(cache) == 0
    assert cache.get((0,), 1) is None  # Cache miss

    cache.put((0,), 1, "test_result")
    assert cache.get((0,), 1) == "test_result"  # Cache hit

    # Test statistics
    stats = cache.get_stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["cache_size"] == 1

    # Test cache key generation with different types
    # Test with simple tuple
    key1 = cache.get_cache_key((0, 1), 1)
    assert isinstance(key1, str)
    assert "comm:" in key1
    assert "state:1" in key1

    # Test with frozenset
    frozen_set = frozenset([("A", 1), ("B", 2)])
    key2 = cache.get_cache_key((frozen_set,), 2)
    assert isinstance(key2, str)
    assert "state:2" in key2

    # Test cache overflow (max_size=5)
    for i in range(10):
        cache.put((i,), i, f"result_{i}")

    # Cache should not exceed max_size
    assert len(cache) <= 5

    # Test clear functionality
    cache.clear()
    assert len(cache) == 0
    stats_after_clear = cache.get_stats()
    assert stats_after_clear["hits"] == 0
    assert stats_after_clear["misses"] == 0


def test_initialize_convergence_info():
    """Test _initialize_convergence_info function."""
    # Create a mock list of commutators
    commutators = [(0,), (1,), ((0,), (1,))]

    # Test when return_convergence_info is False
    sampling_config = SamplingConfig(sampling_method="importance")
    config_no_info = PerturbationErrorConfig(
        timestep=0.1, return_convergence_info=False, sampling=sampling_config
    )

    result = _initialize_convergence_info(commutators, config_no_info)
    assert result is None

    # Test when return_convergence_info is True
    config_with_info = PerturbationErrorConfig(
        timestep=0.5, return_convergence_info=True, sampling=sampling_config
    )

    result = _initialize_convergence_info(commutators, config_with_info)
    assert isinstance(result, dict)

    # Check structure
    assert "global" in result
    assert "states_info" in result

    # Check global info
    global_info = result["global"]
    assert global_info["total_commutators"] == 3
    assert global_info["sampling_method"] == "importance"
    assert global_info["order"] is None  # Should be set by caller
    assert global_info["timestep"] == 0.5

    # Check states_info is empty list initially
    assert result["states_info"] == []

    # Test with different sampling method
    sampling_config_random = SamplingConfig(sampling_method="random")
    config_random = PerturbationErrorConfig(
        timestep=1.0, return_convergence_info=True, sampling=sampling_config_random
    )

    result_random = _initialize_convergence_info(commutators, config_random)
    assert result_random["global"]["sampling_method"] == "random"
    assert result_random["global"]["timestep"] == 1.0
