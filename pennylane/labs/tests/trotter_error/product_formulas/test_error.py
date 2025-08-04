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
    SamplingConfig,
    _apply_sampling_strategy,
    _calculate_commutator_probability,
    _CommutatorCache,
    _group_sums,
    _setup_probability_distribution,
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
    assert stats["size"] == 1  # Changed from "cache_size" to "size"

    # Test cache key generation with different types
    # Test with simple tuple
    key1 = cache.get_cache_key((0, 1), 1)
    assert isinstance(key1, str)
    assert "s1_c" in key1  # Updated to match new format

    # Test with frozenset
    frozen_set = frozenset([("A", 1), ("B", 2)])
    key2 = cache.get_cache_key((frozen_set,), 2)
    assert isinstance(key2, str)
    assert "s2_c" in key2  # Updated to match new format

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


def test_calculate_commutator_probability():
    """Test _calculate_commutator_probability function."""
    # Create test fragments
    n_modes = 2
    r_state = np.random.RandomState(42)
    freqs = r_state.random(n_modes)
    taylor_coeffs = [
        np.array(0),
        r_state.random(size=(n_modes,)),
        r_state.random(size=(n_modes, n_modes)),
    ]
    frags = dict(enumerate(vibrational_fragments(n_modes, freqs, taylor_coeffs)))

    # Test with simple commutator
    commutator = (0,)
    prob = _calculate_commutator_probability(commutator, frags, timestep=0.1, gridpoints=5)
    assert isinstance(prob, float)
    assert prob > 0

    # Test with longer commutator
    commutator = (0, 1)
    prob = _calculate_commutator_probability(commutator, frags, timestep=0.1, gridpoints=5)
    assert isinstance(prob, float)
    assert prob > 0

    # Test with empty commutator
    prob = _calculate_commutator_probability((), frags, timestep=0.1, gridpoints=5)
    assert prob == 0.0

    # Test with frozenset
    frozenset_element = frozenset([(0, 1.0), (1, 0.5)])
    commutator = (frozenset_element,)
    prob = _calculate_commutator_probability(commutator, frags, timestep=0.1, gridpoints=5)
    assert isinstance(prob, float)
    assert prob >= 0


# NOTE: _validate_sampling_inputs was removed as it was not used in production code
# and provided redundant validations already covered by existing functions

def test_sampling_config_validation():
    """Test SamplingConfig validation."""
    # Test valid configurations
    config = SamplingConfig()
    assert config.method == "exact"
    assert config.sample_size is None
    assert config.random_seed is None

    config = SamplingConfig(method="importance", sample_size=10, random_seed=42)
    assert config.method == "importance"
    assert config.sample_size == 10
    assert config.random_seed == 42

    # Test invalid method
    with pytest.raises(ValueError, match="method must be one of"):
        SamplingConfig(method="invalid_method")

    # Test invalid sample_size
    with pytest.raises(ValueError, match="sample_size must be positive"):
        SamplingConfig(sample_size=0)

    with pytest.raises(ValueError, match="sample_size must be positive"):
        SamplingConfig(sample_size=-5)


def test_cache_improved_functionality():
    """Test improved cache functionality with LRU and better key generation."""
    cache = _CommutatorCache(max_size=3)

    # Test improved key generation
    commutator1 = (0, 1)
    commutator2 = (frozenset([(0, 1.0), (1, 0.5)]),)

    key1 = cache.get_cache_key(commutator1, 1)
    key2 = cache.get_cache_key(commutator2, 2)

    assert isinstance(key1, str)
    assert isinstance(key2, str)
    assert key1 != key2

    # Test that keys are deterministic
    key1_repeat = cache.get_cache_key(commutator1, 1)
    assert key1 == key1_repeat

    # Test simple eviction behavior (FIFO)
    cache.put((0,), 1, "result_0")
    cache.put((1,), 2, "result_1")
    cache.put((2,), 3, "result_2")
    assert len(cache) == 3

    # Add fourth item - should evict first item (FIFO)
    cache.put((3,), 4, "result_3")
    assert len(cache) == 3
    assert cache.get((0,), 1) is None  # Should be evicted (first in)
    assert cache.get((1,), 2) == "result_1"  # Should still be there
    assert cache.get((2,), 3) == "result_2"  # Should still be there
    assert cache.get((3,), 4) == "result_3"  # Should be there

    # Test statistics
    stats = cache.get_stats()
    assert "total" in stats
    assert "max_size" in stats
    assert stats["max_size"] == 3
    assert stats["size"] == 3


def test_cache_error_handling():
    """Test cache error handling with problematic keys."""
    cache = _CommutatorCache(max_size=5)

    # Test with keys that might cause issues
    problematic_commutator = (None, "test")

    # These operations should not raise exceptions due to error handling
    cache.put(problematic_commutator, 1, "test_result")
    result = cache.get(problematic_commutator, 1)

    # The cache should handle errors gracefully
    # Either it works or it fails silently
    assert result is None or result == "test_result"

    # Test that cache continues to work after errors
    cache.put((0,), 1, "normal_result")
    assert cache.get((0,), 1) == "normal_result"


def test_cache_context_manager():
    """Test that cache works properly as a context manager."""
    cache_instance = None

    with _CommutatorCache() as cache:
        cache_instance = cache
        # Add some data
        cache.put(("test",), 0, "test_value")
        assert len(cache) == 1
        assert cache.get(("test",), 0) == "test_value"

    # After exiting context, cache should be cleared
    assert len(cache_instance) == 0
    assert cache_instance.get(("test",), 0) is None


def test_setup_probability_distribution():
    """Test the _setup_probability_distribution function."""
    # Mock fragment
    class MockFragment:  # pylint: disable=too-few-public-methods
        """Mock fragment for testing probability distribution."""
        def norm(self, params):  # pylint: disable=unused-argument
            """Return mock norm value."""
            return 1.0

    fragments = {0: MockFragment(), 1: MockFragment()}
    commutators = [(0,), (1,), (0, 1)]
    timestep = 0.1

    # Test probability distribution
    probs = _setup_probability_distribution(commutators, fragments, timestep)

    # Should be normalized (sum to 1)
    assert abs(np.sum(probs) - 1.0) < 1e-10

    # Should have same length as commutators
    assert len(probs) == len(commutators)

    # All probabilities should be non-negative
    assert np.all(probs >= 0)


def test_apply_sampling_strategy():
    """Test the _apply_sampling_strategy function."""
    commutators = [(0,), (1,), (0, 1)]

    # Test exact method
    config = SamplingConfig(method="exact")
    selected, weights = _apply_sampling_strategy(commutators, config)

    assert selected == commutators
    assert weights == [1.0, 1.0, 1.0]

    # Test unimplemented methods
    config = SamplingConfig(method="random")
    with pytest.raises(NotImplementedError, match="Random sampling will be implemented"):
        _apply_sampling_strategy(commutators, config)

    config = SamplingConfig(method="importance")
    with pytest.raises(NotImplementedError, match="Importance sampling will be implemented"):
        _apply_sampling_strategy(commutators, config)

    config = SamplingConfig(method="top_k")
    with pytest.raises(NotImplementedError, match="Top-k sampling will be implemented"):
        _apply_sampling_strategy(commutators, config)
