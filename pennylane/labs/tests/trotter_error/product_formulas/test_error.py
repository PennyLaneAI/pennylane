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
from pennylane.labs.trotter_error.realspace import RealspaceCoeffs, RealspaceOperator, RealspaceSum
from pennylane.labs.trotter_error.product_formulas.error import (
    SamplingConfig,
    _apply_sampling_method,
    _apply_sampling_strategy,
    _calculate_commutator_probability,
    _CommutatorCache,
    _group_sums,
    _importance_sampling,
    _random_sampling,
    _setup_importance_probabilities,
    _setup_probability_distribution,
    _top_k_sampling,
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
    assert "comm:" in key1 and "state:1" in key1

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





# =============================================================================
# Tests for Importance Probability Functions
# =============================================================================


def test_calculate_commutator_probability():
    """Test _calculate_commutator_probability function."""
    # Create test fragments using real vibrational fragments
    n_modes = 2
    r_state = np.random.RandomState(42)
    freqs = r_state.random(n_modes)
    taylor_coeffs = [
        np.array(0),
        r_state.random(size=(n_modes,)),
        r_state.random(size=(n_modes, n_modes)),
    ]
    frags = dict(enumerate(vibrational_fragments(n_modes, freqs, taylor_coeffs)))

    # Test empty commutator
    prob = _calculate_commutator_probability((), frags, 0.1, 10)
    assert prob == 0.0

    # Test single element commutator
    prob = _calculate_commutator_probability((0,), frags, 0.1, 10)
    assert isinstance(prob, float)
    assert prob > 0

    # Test two element commutator
    prob = _calculate_commutator_probability((0, 1), frags, 0.1, 10)
    assert isinstance(prob, float)
    assert prob > 0

    # Test with frozenset element
    frozenset_element = frozenset([(0, 1.0), (1, 0.5)])
    prob = _calculate_commutator_probability((frozenset_element,), frags, 0.1, 10)
    assert isinstance(prob, float)
    assert prob >= 0


def test_setup_importance_probabilities():
    """Test _setup_importance_probabilities function."""
    # Create test fragments using real vibrational fragments
    n_modes = 2
    r_state = np.random.RandomState(42)
    freqs = r_state.random(n_modes)
    taylor_coeffs = [
        np.array(0),
        r_state.random(size=(n_modes,)),
        r_state.random(size=(n_modes, n_modes)),
    ]
    frags = dict(enumerate(vibrational_fragments(n_modes, freqs, taylor_coeffs)))
    commutators = [(0,), (1,), (0, 1)]

    probs = _setup_importance_probabilities(commutators, frags, 0.1, 10)

    # Check return type and shape
    assert isinstance(probs, np.ndarray)
    assert len(probs) == 3

    # All probabilities should be positive
    assert all(p >= 0 for p in probs)

    # Test with different gridpoints parameter
    probs_with_gridpoints = _setup_importance_probabilities(commutators, frags, 0.1, 20)
    assert isinstance(probs_with_gridpoints, np.ndarray)
    assert len(probs_with_gridpoints) == 3


# =============================================================================
# Tests for Sampling Methods
# =============================================================================


def test_random_sampling():
    """Test _random_sampling function."""
    commutators = [(0,), (1,), (0, 1), (1, 0)]

    # Test basic functionality
    sampled_comms, weights = _random_sampling(commutators, 3, random_seed=42)

    assert len(sampled_comms) == 3
    assert len(weights) == 3
    assert all(w == pytest.approx(1.0 / 3) for w in weights)

    # Test reproducibility
    sampled_comms2, weights2 = _random_sampling(commutators, 3, random_seed=42)
    assert sampled_comms == sampled_comms2
    assert np.allclose(weights, weights2)


def test_importance_sampling():
    """Test _importance_sampling function."""
    commutators = [(0,), (1,), (0, 1)]
    probabilities = np.array([0.1, 0.2, 0.7])  # Heavily weighted toward last commutator

    sampled_comms, weights = _importance_sampling(commutators, probabilities, 2, random_seed=42)

    assert len(sampled_comms) == 2
    assert len(weights) == 2

    # Check that weights are positive
    assert all(w > 0 for w in weights)

    # Test reproducibility
    sampled_comms2, weights2 = _importance_sampling(commutators, probabilities, 2, random_seed=42)
    assert sampled_comms == sampled_comms2
    assert np.allclose(weights, weights2)


def test_top_k_sampling():
    """Test _top_k_sampling function."""
    commutators = [(0,), (1,), (0, 1), (1, 0)]
    probabilities = np.array([0.1, 0.4, 0.2, 0.3])

    # Sample top 2
    sampled_comms, weights = _top_k_sampling(commutators, probabilities, 2)

    assert len(sampled_comms) == 2
    assert len(weights) == 2
    assert all(w == 1.0 for w in weights)

    # Should get the commutators with highest probabilities (indices 1 and 3)
    expected_comms = [(1,), (1, 0)]  # Indices 1 and 3 sorted by probability
    assert set(sampled_comms) == set(expected_comms)


def test_apply_sampling_method():
    """Test _apply_sampling_method dispatcher."""
    # Create test fragments using real vibrational fragments
    n_modes = 2
    r_state = np.random.RandomState(42)
    freqs = r_state.random(n_modes)
    taylor_coeffs = [
        np.array(0),
        r_state.random(size=(n_modes,)),
        r_state.random(size=(n_modes, n_modes)),
    ]
    frags = dict(enumerate(vibrational_fragments(n_modes, freqs, taylor_coeffs)))
    commutators = [(0,), (1,), (0, 1)]

    # Test random sampling
    config_random = SamplingConfig(
        sampling_method="random", sample_size=2, random_seed=42
    )

    sampled_comms, weights = _apply_sampling_method(commutators, frags, config_random, 0.1, 10)
    assert len(sampled_comms) == 2
    assert len(weights) == 2

    # Test importance sampling
    config_importance = SamplingConfig(
        sampling_method="importance", sample_size=2, random_seed=42
    )

    sampled_comms, weights = _apply_sampling_method(
        commutators, frags, config_importance, 0.1, 10
    )
    assert len(sampled_comms) == 2
    assert len(weights) == 2

    # Test top_k sampling
    config_top_k = SamplingConfig(
        sampling_method="top_k", sample_size=2
    )

    sampled_comms, weights = _apply_sampling_method(commutators, frags, config_top_k, 0.1, 10)
    assert len(sampled_comms) == 2
    assert len(weights) == 2

    # Test with different gridpoints parameter
    sampled_comms, weights = _apply_sampling_method(commutators, frags, config_top_k, 0.1, 20)
    assert len(sampled_comms) == 2
    assert len(weights) == 2

    # Test unknown method
    with pytest.raises(ValueError, match="sampling_method must be one of"):
        SamplingConfig(sampling_method="unknown")

    # Test default sample_size (should use all commutators)
    config_default = SamplingConfig(
        sampling_method="random", sample_size=None, random_seed=42
    )

    sampled_comms, weights = _apply_sampling_method(commutators, frags, config_default, 0.1, 10)
    assert len(sampled_comms) == len(commutators)


def test_sampling_config_validation():
    """Test SamplingConfig validation."""
    # Test valid configurations
    config = SamplingConfig()
    assert config.sampling_method == "importance"
    assert config.sample_size is None
    assert config.random_seed is None

    config = SamplingConfig(sampling_method="random", sample_size=10, random_seed=42)
    assert config.sampling_method == "random"
    assert config.sample_size == 10
    assert config.random_seed == 42

    # Test invalid method
    with pytest.raises(ValueError, match="sampling_method must be one of"):
        SamplingConfig(sampling_method="invalid_method")

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
    assert "hit_rate" in stats
    assert "cache_size" in stats
    assert stats["cache_size"] == 3


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
    # Create simple RealspaceSum fragments with constant coefficients
    n_modes = 2
    coeffs = RealspaceCoeffs(
        np.array(1.0), label="test"
    )  # Scalar coefficient for identity operator
    op = RealspaceOperator(n_modes, (), coeffs)
    fragment = RealspaceSum(n_modes, [op])

    fragments = {0: fragment, 1: fragment}
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

    # Test random method
    config = SamplingConfig(sampling_method="random", sample_size=2, random_seed=42)
    selected, weights = _apply_sampling_strategy(commutators, config)
    
    assert len(selected) == 2
    assert len(weights) == 2
    assert all(w > 0 for w in weights)

    # Test importance method - requires probabilities
    config = SamplingConfig(sampling_method="importance", sample_size=2, random_seed=42)
    probabilities = np.array([0.1, 0.3, 0.6])
    selected, weights = _apply_sampling_strategy(commutators, config, probabilities)
    
    assert len(selected) == 2
    assert len(weights) == 2
    assert all(w > 0 for w in weights)

    # Test top_k method - requires probabilities
    config = SamplingConfig(sampling_method="top_k", sample_size=2)
    selected, weights = _apply_sampling_strategy(commutators, config, probabilities)
    
    assert len(selected) == 2
    assert len(weights) == 2
    assert all(w == 1.0 for w in weights)

    # Test error cases
    config = SamplingConfig(sampling_method="importance")
    with pytest.raises(ValueError, match="Probabilities required for importance sampling"):
        _apply_sampling_strategy(commutators, config)

    config = SamplingConfig(sampling_method="top_k")
    with pytest.raises(ValueError, match="Probabilities required for top-k sampling"):
        _apply_sampling_strategy(commutators, config)

    # Test unknown method
    with pytest.raises(ValueError, match="sampling_method must be one of"):
        SamplingConfig(sampling_method="unknown")
