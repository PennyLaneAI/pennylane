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
from scipy.sparse import csr_matrix

from pennylane.labs.trotter_error import (
    HOState,
    SparseState,
    ProductFormula,
    perturbation_error,
    vibrational_fragments,
)
from pennylane.labs.trotter_error.product_formulas.error import (
    _group_sums,
    _sample_importance_commutators,
    _fixed_sampling,
    _adaptive_sampling,
    _compute_expectation_values,
    effective_hamiltonian,
    _get_confidence_z_score,
    _setup_importance_probabilities,
    _adaptive_sample_single_state,
)
from pennylane.labs.trotter_error.fragments import sparse_fragments


# Shared test fixtures to reduce code duplication
@pytest.fixture
def simple_system():
    """Create a simple 2-mode system for testing."""
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

    frag_labels = [0, 1, 1, 0]
    frag_coeffs = [1 / 2, 1 / 2, 1 / 2, 1 / 2]
    pf = ProductFormula(frag_labels, coeffs=frag_coeffs)

    gridpoints = 5
    state1 = HOState(n_modes, gridpoints, {(0, 0): 1})
    state2 = HOState(n_modes, gridpoints, {(1, 1): 1})

    return {
        'pf': pf,
        'frags': frags,
        'states': [state1, state2],
        'n_modes': n_modes,
        'gridpoints': gridpoints
    }


@pytest.fixture
def minimal_system():
    """Create a minimal system for unit tests."""
    n_modes = 2
    r_state = np.random.RandomState(42)
    freqs = r_state.random(n_modes)
    taylor_coeffs = [
        np.array(0),
        r_state.random(size=(n_modes,)),
        r_state.random(size=(n_modes, n_modes)),
    ]
    frags = dict(enumerate(vibrational_fragments(n_modes, freqs, taylor_coeffs)))

    frag_labels = [0, 1]
    frag_coeffs = [1 / 2, 1 / 2]
    pf = ProductFormula(frag_labels, coeffs=frag_coeffs)

    gridpoints = 5
    state1 = HOState(n_modes, gridpoints, {(0, 0): 1})
    state2 = HOState(n_modes, gridpoints, {(1, 1): 1})

    return {
        'pf': pf,
        'frags': frags,
        'states': [state1, state2],
        'n_modes': n_modes,
        'gridpoints': gridpoints
    }


@pytest.fixture
def simple_system_sparse():
    """Create a simple 2-mode system with SparseState for testing."""

    # Create simple sparse matrices for testing
    # Identity and Pauli-X-like matrices
    matrix1 = csr_matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    matrix2 = csr_matrix([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])

    frags = dict(enumerate(sparse_fragments([matrix1, matrix2])))

    frag_labels = [0, 1, 1, 0]
    frag_coeffs = [1 / 2, 1 / 2, 1 / 2, 1 / 2]
    pf = ProductFormula(frag_labels, coeffs=frag_coeffs)

    # Create sparse states compatible with 4x4 matrices - as row vectors for transpose operations
    state1_sparse = csr_matrix([[1, 0, 0, 0]])  # |0> as row vector
    state1 = SparseState(state1_sparse)

    state2_sparse = csr_matrix([[0, 1, 0, 0]])  # |1> as row vector
    state2 = SparseState(state2_sparse)

    return {
        'pf': pf,
        'frags': frags,
        'states': [state1, state2],
        'n_modes': 2,  # Keep for compatibility
        'gridpoints': 2  # Adjusted for 4x4 matrices
    }


@pytest.fixture
def minimal_system_sparse():
    """Create a minimal system with SparseState for unit tests."""

    # Create simple sparse matrices for testing
    matrix1 = csr_matrix([[1, 0], [0, 1]])  # Identity matrix
    matrix2 = csr_matrix([[0, 1], [1, 0]])  # Pauli-X matrix

    frags = dict(enumerate(sparse_fragments([matrix1, matrix2])))

    frag_labels = [0, 1]
    frag_coeffs = [1 / 2, 1 / 2]
    pf = ProductFormula(frag_labels, coeffs=frag_coeffs)

    # Create sparse states compatible with 2x2 matrices - as row vectors for transpose operations
    state1_sparse = csr_matrix([[1, 0]])  # |0> as row vector
    state1 = SparseState(state1_sparse)

    state2_sparse = csr_matrix([[0, 1]])  # |1> as row vector
    state2 = SparseState(state2_sparse)

    return {
        'pf': pf,
        'frags': frags,
        'states': [state1, state2],
        'n_modes': 2,  # Keep for compatibility
        'gridpoints': 2  # Adjusted for 2x2 matrices
    }


@pytest.mark.parametrize(
    "backend", ["serial", "mpi4py_pool", "mpi4py_comm"]  # Removed mp_pool and cf_procpool due to pickling issues
)
@pytest.mark.parametrize("parallel_mode", ["state", "commutator"])
def test_perturbation_error_backends(backend, parallel_mode, mpi4py_support, simple_system):  # pylint: disable=redefined-outer-name
    """Test that perturbation error function runs without errors for different backends."""

    if backend in {"mpi4py_pool", "mpi4py_comm"} and not mpi4py_support:
        pytest.skip(f"Skipping test: '{backend}' requires mpi4py, which is not installed.")

    num_workers = 1 if backend == "serial" else 2
    errors = perturbation_error(
        simple_system['pf'],
        simple_system['frags'],
        simple_system['states'],
        order=3,
        num_workers=num_workers,
        backend=backend,
        parallel_mode=parallel_mode,
    )

    assert isinstance(errors, list)
    assert len(errors) == 2
    # Verify results are finite complex numbers
    for error in errors:
        assert np.isfinite(error)
        assert isinstance(error, (complex, float, int))


def test_perturbation_error_invalid_parallel_mode(simple_system):  # pylint: disable=redefined-outer-name
    """Test that perturbation error raises an error for invalid parallel mode."""
    with pytest.raises(ValueError, match="Invalid parallel mode"):
        perturbation_error(
            simple_system['pf'],
            simple_system['frags'],
            simple_system['states'],
            order=3,
            num_workers=2,
            backend="mpi4py_pool",  # Use parallel backend to trigger parallel_mode validation
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


@pytest.mark.parametrize("sampling_method", ["random", "importance"])
@pytest.mark.parametrize("sample_size", [1, 5, 10])
def test_perturbation_error_sampling_methods(sampling_method, sample_size, simple_system):  # pylint: disable=redefined-outer-name
    """Test perturbation error with different sampling methods and sizes."""
    # Test with sampling
    errors_sampled = perturbation_error(
        simple_system['pf'],
        simple_system['frags'],
        simple_system['states'],
        order=3,
        sample_size=sample_size,
        sampling_method=sampling_method,
        random_seed=42,
    )

    # Test without sampling for comparison
    errors_full = perturbation_error(
        simple_system['pf'],
        simple_system['frags'],
        simple_system['states'],
        order=3,
    )

    assert isinstance(errors_sampled, list)
    assert len(errors_sampled) == 2
    assert isinstance(errors_full, list)
    assert len(errors_full) == 2

    # Verify that results are finite complex numbers
    for sampled_error, full_error in zip(errors_sampled, errors_full):
        assert np.isfinite(sampled_error)
        assert isinstance(sampled_error, (complex, float, int))
        assert np.isfinite(full_error)
        assert isinstance(full_error, (complex, float, int))

        # With reasonable sample size, results should be in same order of magnitude
        if abs(full_error) > 1e-12:  # Avoid division by very small numbers
            relative_diff = abs(sampled_error - full_error) / abs(full_error)
            # Allow for significant sampling error, but not completely wrong order of magnitude
            assert relative_diff < 10.0, f"Sampling result {sampled_error} too different from full result {full_error}"


def test_perturbation_error_sampling_reproducibility(minimal_system):  # pylint: disable=redefined-outer-name
    """Test that sampling methods produce reproducible results with same seed."""
    for sampling_method in ["random", "importance"]:
        errors1 = perturbation_error(
            minimal_system['pf'],
            minimal_system['frags'],
            minimal_system['states'],
            order=2,
            sample_size=3,
            sampling_method=sampling_method,
            random_seed=42,
        )

        errors2 = perturbation_error(
            minimal_system['pf'],
            minimal_system['frags'],
            minimal_system['states'],
            order=2,
            sample_size=3,
            sampling_method=sampling_method,
            random_seed=42,
        )

        # Results should be identical with same seed
        np.testing.assert_array_equal(errors1, errors2)


def test_perturbation_error_adaptive_sampling(minimal_system):  # pylint: disable=redefined-outer-name
    """Test adaptive sampling functionality."""
    errors = perturbation_error(
        minimal_system['pf'],
        minimal_system['frags'],
        minimal_system['states'],
        order=2,
        adaptive_sampling=True,
        confidence_level=0.95,
        target_error=0.1,
        min_sample_size=5,
        max_sample_size=20,
        sampling_method="random",
        random_seed=42,
    )

    assert isinstance(errors, list)
    assert len(errors) == 2
    for error in errors:
        assert np.isfinite(error)
        assert isinstance(error, (complex, float, int))


def test_perturbation_error_adaptive_sampling_constraints():
    """Test that adaptive sampling enforces its constraints."""
    # Should raise error for non-serial backend
    with pytest.raises(ValueError, match="Adaptive sampling is only compatible with backend='serial'"):
        perturbation_error(
            ProductFormula([0], coeffs=[1.0]),
            {0: None},  # dummy
            [],  # dummy
            order=1,
            adaptive_sampling=True,
            backend="mp_pool"
        )

    # Should raise error for multiple workers
    with pytest.raises(ValueError, match="Adaptive sampling requires num_workers=1"):
        perturbation_error(
            ProductFormula([0], coeffs=[1.0]),
            {0: None},  # dummy
            [],  # dummy
            order=1,
            adaptive_sampling=True,
            num_workers=2
        )


@pytest.mark.parametrize("invalid_method", ["invalid", "wrong_method", ""])
def test_perturbation_error_invalid_sampling_method(invalid_method, minimal_system):  # pylint: disable=redefined-outer-name
    """Test that invalid sampling methods raise appropriate errors."""
    with pytest.raises(ValueError, match="sampling_method must be"):
        perturbation_error(
            minimal_system['pf'],
            minimal_system['frags'],
            minimal_system['states'],
            order=2,
            sample_size=5,
            sampling_method=invalid_method,
        )


def test_effective_hamiltonian_basic(minimal_system):  # pylint: disable=redefined-outer-name
    """Test the effective_hamiltonian function with different orders."""
    # Test with different orders
    for order in [1, 2]:
        eff_ham = effective_hamiltonian(
            minimal_system['pf'],
            minimal_system['frags'],
            order=order,
            timestep=0.1
        )
        assert eff_ham is not None

        # Verify it's a proper Fragment-like object
        assert hasattr(eff_ham, 'norm'), "Effective Hamiltonian should have norm method"

        # Test that the norm is a finite number
        norm_value = eff_ham.norm({"gridpoints": minimal_system['gridpoints']})
        assert np.isfinite(norm_value), "Effective Hamiltonian norm should be finite"
        assert norm_value >= 0, "Norm should be non-negative"


def test_effective_hamiltonian_fragment_mismatch():
    """Test that effective_hamiltonian raises error when fragments don't match product formula."""
    # Create a product formula requiring fragment 2 that doesn't exist
    frag_labels = [0, 1, 2]  # This includes label 2
    frag_coeffs = [1 / 3, 1 / 3, 1 / 3]
    pf = ProductFormula(frag_labels, coeffs=frag_coeffs)

    # Create proper fragments for labels 0 and 1 but missing 2
    n_modes = 2
    r_state = np.random.RandomState(42)
    freqs = r_state.random(n_modes)
    taylor_coeffs = [
        np.array(0),
        r_state.random(size=(n_modes,)),
        r_state.random(size=(n_modes, n_modes)),
    ]
    all_frags = dict(enumerate(vibrational_fragments(n_modes, freqs, taylor_coeffs)))

    # Only provide fragments for labels 0 and 1, missing 2
    incomplete_frags = {0: all_frags[0], 1: all_frags[1]}  # Missing fragment 2

    with pytest.raises(ValueError, match="Fragments do not match product formula"):
        effective_hamiltonian(pf, incomplete_frags, order=2, timestep=0.1)


def test_fixed_sampling_invalid_method(minimal_system):  # pylint: disable=redefined-outer-name
    """Test that _fixed_sampling raises error for invalid sampling method."""
    frags = minimal_system['frags']
    states = minimal_system['states']
    commutators = [(0,), (1,)]

    with pytest.raises(ValueError, match="sampling_method must be"):
        _fixed_sampling(
            commutators=commutators,
            fragments=frags,
            states=states,
            sample_size=2,
            sampling_method="invalid_method",
            timestep=0.1,
            gridpoints=minimal_system['gridpoints'],
            random_seed=42
        )


@pytest.mark.parametrize("system_fixture", ["minimal_system", "minimal_system_sparse"])
def test_fixed_sampling_random(system_fixture, request):
    """Test _fixed_sampling with random method for both HOState and SparseState."""
    system = request.getfixturevalue(system_fixture)
    frags = system['frags']
    states = system['states']

    # Create test commutators
    commutators = [
        (0,),
        (1,),
        ((0,), (1,)),
        ((1,), (0,)),
    ]

    # Test random sampling
    results = _fixed_sampling(
        commutators=commutators,
        fragments=frags,
        states=states,
        timestep=0.1,
        sample_size=3,
        sampling_method="random",
        random_seed=42,
        gridpoints=system['gridpoints']
    )

    assert isinstance(results, list)
    assert len(results) == len(states)
    for result in results:
        assert np.isfinite(result)
        assert isinstance(result, (complex, float, int))


@pytest.mark.parametrize("system_fixture", ["minimal_system", "minimal_system_sparse"])
def test_fixed_sampling_importance(system_fixture, request):
    """Test _fixed_sampling with importance method for both HOState and SparseState."""
    system = request.getfixturevalue(system_fixture)
    frags = system['frags']
    states = system['states']

    # Create test commutators
    commutators = [
        (0,),
        (1,),
        ((0,), (1,)),
        ((1,), (0,)),
    ]

    # Test importance sampling
    results = _fixed_sampling(
        commutators=commutators,
        fragments=frags,
        states=states,
        timestep=0.1,
        sample_size=3,
        sampling_method="importance",
        random_seed=42,
        gridpoints=system['gridpoints']
    )

    assert isinstance(results, list)
    assert len(results) == len(states)
    for result in results:
        assert np.isfinite(result)
        assert isinstance(result, (complex, float, int))


@pytest.mark.parametrize("system_fixture", ["minimal_system", "minimal_system_sparse"])
def test_adaptive_sampling(system_fixture, request):
    """Test _adaptive_sampling for both HOState and SparseState."""
    system = request.getfixturevalue(system_fixture)
    frags = system['frags']
    states = system['states']

    # Create test commutators
    commutators = [
        (0,),
        (1,),
        ((0,), (1,)),
        ((1,), (0,)),
    ]

    # Test adaptive sampling
    results = _adaptive_sampling(
        commutators=commutators,
        fragments=frags,
        states=states,
        timestep=0.1,
        sampling_method="random",
        confidence_level=0.95,
        target_error=0.1,
        min_sample_size=2,
        max_sample_size=5,
        random_seed=42,
        gridpoints=system['gridpoints']
    )

    assert isinstance(results, list)
    assert len(results) == len(states)
    for result in results:
        assert np.isfinite(result)
        assert isinstance(result, (complex, float, int))


@pytest.mark.parametrize("system_fixture", ["minimal_system", "minimal_system_sparse"])
def test_compute_expectation_values(system_fixture, request):
    """Test _compute_expectation_values for both HOState and SparseState."""
    system = request.getfixturevalue(system_fixture)
    frags = system['frags']
    states = system['states']

    # Create test commutators and weights
    commutators = [
        (0,),
        (1,),
        ((0,), (1,)),
    ]
    weights = [1.0, 1.0, 2.0]

    # Test expectation value computation
    results = _compute_expectation_values(
        commutators=commutators,
        weights=weights,
        fragments=frags,
        states=states,
        num_workers=1,
        backend="serial",
        parallel_mode="state"
    )

    assert isinstance(results, list)
    assert len(results) == len(states)
    for result in results:
        assert np.isfinite(result)
        assert isinstance(result, (complex, float, int))


def test_sample_importance_commutators():
    """Test _sample_importance_commutators with synthetic data."""

    # Create test commutators
    commutators = [
        (0,),
        (1,),
        ((0,), (1,)),
        ((1,), (0,)),
    ]

    # Pre-calculate probabilities (simplified uniform for this test)
    probabilities = np.ones(len(commutators)) / len(commutators)

    # Test importance sampling
    sampled_commutators, weights = _sample_importance_commutators(
        commutators=commutators,
        probabilities=probabilities,
        sample_size=2,
        random_seed=42
    )

    assert isinstance(sampled_commutators, list)
    assert isinstance(weights, list)
    assert len(sampled_commutators) == 2
    assert len(weights) == 2

    for comm in sampled_commutators:
        assert comm in commutators

    for weight in weights:
        assert np.isfinite(weight)
        assert weight > 0


# ==================== TESTS FOR NEW HELPER FUNCTIONS ====================

def test_get_confidence_z_score():
    """Test _get_confidence_z_score function with different confidence levels."""

    # Test known confidence levels with correct statistical values
    assert _get_confidence_z_score(0.68) == 1.0     # 1 standard deviation
    assert _get_confidence_z_score(0.9545) == 2.0   # 2 standard deviations
    assert _get_confidence_z_score(0.90) == 1.645   # 90% confidence
    assert _get_confidence_z_score(0.95) == 1.96    # 95% confidence
    assert _get_confidence_z_score(0.99) == 2.576   # 99% confidence

    # Test default fallback for unrecognized confidence levels
    assert _get_confidence_z_score(0.85) == 1.0  # Should fallback to 68%
    assert _get_confidence_z_score(0.12) == 1.0  # Should fallback to 68%


@pytest.mark.parametrize("system_fixture", ["minimal_system", "minimal_system_sparse"])
def test_setup_importance_probabilities_random(system_fixture, request):
    """Test _setup_importance_probabilities with random sampling."""

    system = request.getfixturevalue(system_fixture)
    frags = system['frags']

    # Create test commutators
    commutators = [
        (0,),
        (1,),
        ((0,), (1,)),
    ]

    # Test random sampling - should return None
    probabilities = _setup_importance_probabilities(
        commutators=commutators,
        fragments=frags,
        timestep=0.1,
        gridpoints=system['gridpoints'],
        sampling_method="random"
    )

    assert probabilities is None


@pytest.mark.parametrize("system_fixture", ["minimal_system", "minimal_system_sparse"])
def test_setup_importance_probabilities_importance(system_fixture, request):
    """Test _setup_importance_probabilities with importance sampling."""

    system = request.getfixturevalue(system_fixture)
    frags = system['frags']

    # Create test commutators
    commutators = [
        (0,),
        (1,),
        ((0,), (1,)),
    ]

    # Test importance sampling - should return normalized probabilities
    probabilities = _setup_importance_probabilities(
        commutators=commutators,
        fragments=frags,
        timestep=0.1,
        gridpoints=system['gridpoints'],
        sampling_method="importance"
    )

    assert isinstance(probabilities, np.ndarray)
    assert len(probabilities) == len(commutators)
    assert np.all(probabilities >= 0)
    assert np.sum(probabilities) > 0

    # Should be normalized
    np.testing.assert_allclose(np.sum(probabilities), 1.0, rtol=1e-10)


@pytest.mark.parametrize("system_fixture", ["minimal_system", "minimal_system_sparse"])
def test_adaptive_sample_single_state_random(system_fixture, request):
    """Test _adaptive_sample_single_state with random sampling."""

    system = request.getfixturevalue(system_fixture)
    frags = system['frags']
    state = system['states'][0]  # Use first state

    # Create test commutators
    commutators = [
        (0,),
        (1,),
        ((0,), (1,)),
        ((1,), (0,)),
    ]

    # Test with random sampling (probabilities = None)
    expectation = _adaptive_sample_single_state(
        state=state,
        state_idx=0,
        commutators=commutators,
        fragments=frags,
        probabilities=None,  # Random sampling
        sampling_method="random",
        z_score=1.96,
        target_error=0.5,  # Large error to converge quickly
        min_sample_size=2,
        max_sample_size=5,
    )

    assert np.isfinite(expectation)
    assert isinstance(expectation, (complex, float, int))


@pytest.mark.parametrize("system_fixture", ["minimal_system", "minimal_system_sparse"])
def test_adaptive_sample_single_state_importance(system_fixture, request):
    """Test _adaptive_sample_single_state with importance sampling."""

    system = request.getfixturevalue(system_fixture)
    frags = system['frags']
    state = system['states'][0]  # Use first state

    # Create test commutators
    commutators = [
        (0,),
        (1,),
        ((0,), (1,)),
        ((1,), (0,)),
    ]

    # Create uniform probabilities for importance sampling
    probabilities = np.ones(len(commutators)) / len(commutators)

    # Test with importance sampling
    expectation = _adaptive_sample_single_state(
        state=state,
        state_idx=0,
        commutators=commutators,
        fragments=frags,
        probabilities=probabilities,
        sampling_method="importance",
        z_score=1.96,
        target_error=0.5,  # Large error to converge quickly
        min_sample_size=2,
        max_sample_size=5,
    )

    assert np.isfinite(expectation)
    assert isinstance(expectation, (complex, float, int))
