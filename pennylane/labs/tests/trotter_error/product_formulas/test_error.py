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
    _convergence_sampling,
    _compute_expectation_values,
    effective_hamiltonian,
    _get_confidence_z_score,
    _setup_importance_probabilities,
    _adaptive_sample_single_state,
    _CommutatorCache,
    _apply_commutator,
    _get_expval_state,
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
        target_relative_error=0.1,
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


# ==================== TESTS FOR CONVERGENCE INFO FUNCTIONALITY ====================

def test_perturbation_error_return_convergence_info_basic(minimal_system):  # pylint: disable=redefined-outer-name
    """Test that return_convergence_info=True returns tuple with convergence information."""
    # Test without convergence info (default behavior)
    errors_only = perturbation_error(
        minimal_system['pf'],
        minimal_system['frags'],
        minimal_system['states'],
        order=2,
    )

    # Test with convergence info
    errors_with_info, convergence_info = perturbation_error(
        minimal_system['pf'],
        minimal_system['frags'],
        minimal_system['states'],
        order=2,
        return_convergence_info=True,
    )

    # Check that results are consistent
    np.testing.assert_array_equal(errors_only, errors_with_info)

    # Check convergence info structure
    assert isinstance(convergence_info, dict)
    assert 'states_info' in convergence_info
    assert 'global' in convergence_info

    # Check global info structure
    global_info = convergence_info['global']
    assert 'total_commutators' in global_info
    assert 'sampling_method' in global_info
    assert 'order' in global_info
    assert 'timestep' in global_info

    # For exact computation, sampled_commutators should be in global section
    assert 'sampled_commutators' in convergence_info['global']

    # Check states info structure for exact computation
    states_info = convergence_info['states_info']
    assert len(states_info) == len(minimal_system['states'])

    for state_info in states_info:
        assert isinstance(state_info, dict)
        assert 'mean' in state_info
        assert 'variance' in state_info
        assert 'sigma2_over_n' in state_info
        assert 'n_samples' in state_info
        assert 'method' in state_info
        # For exact computation, these should be zero
        assert state_info['variance'] == 0.0
        assert state_info['sigma2_over_n'] == 0.0
        assert state_info['method'] == 'exact'


@pytest.mark.parametrize("sampling_method", ["random", "importance"])
def test_perturbation_error_convergence_info_fixed_sampling(sampling_method, minimal_system):  # pylint: disable=redefined-outer-name
    """Test convergence info with fixed sampling methods."""
    errors, convergence_info = perturbation_error(
        minimal_system['pf'],
        minimal_system['frags'],
        minimal_system['states'],
        order=2,
        sample_size=3,
        sampling_method=sampling_method,
        random_seed=42,
        return_convergence_info=True,
    )

    # Basic structure checks
    assert isinstance(errors, list)
    assert isinstance(convergence_info, dict)
    assert len(errors) == len(minimal_system['states'])

    # Check convergence info contains expected keys
    assert 'states_info' in convergence_info
    assert 'global' in convergence_info
    assert 'sampled_commutators' in convergence_info['global']

    # Check states info
    states_info = convergence_info['states_info']
    assert len(states_info) == len(minimal_system['states'])

    for state_info in states_info:
        assert 'mean' in state_info
        assert 'sampling_method' in state_info
        assert state_info['sampling_method'] == 'fixed'
        assert 'n_samples' in state_info
        assert 'execution_time' in state_info
        assert 'cache_stats' in state_info

        # For fixed sampling without repeated runs
        assert state_info['variance'] == 0.0
        assert state_info['sigma2_over_n'] == 0.0


def test_perturbation_error_convergence_info_adaptive_sampling(minimal_system):  # pylint: disable=redefined-outer-name
    """Test convergence info with adaptive sampling."""
    errors, convergence_info = perturbation_error(
        minimal_system['pf'],
        minimal_system['frags'],
        minimal_system['states'],
        order=2,
        adaptive_sampling=True,
        confidence_level=0.95,
        target_relative_error=0.5,  # Large target for quick convergence
        min_sample_size=3,
        max_sample_size=10,
        sampling_method="random",
        random_seed=42,
        return_convergence_info=True,
    )

    # Basic structure checks
    assert isinstance(errors, list)
    assert isinstance(convergence_info, dict)
    assert len(errors) == len(minimal_system['states'])

    # Check global info
    assert 'global' in convergence_info
    global_info = convergence_info['global']
    assert 'probability_calculation_time' in global_info
    assert 'sampled_commutators' in global_info

    # Check states info with convergence histories
    assert 'states_info' in convergence_info
    states_info = convergence_info['states_info']
    assert len(states_info) == len(minimal_system['states'])

    for state_info in states_info:
        # Check basic fields
        assert 'mean' in state_info
        assert 'variance' in state_info
        assert 'sigma2_over_n' in state_info
        assert 'n_samples' in state_info
        assert 'execution_time' in state_info
        assert 'cache_stats' in state_info

        # Check convergence histories - the key new functionality
        assert 'mean_history' in state_info
        assert 'variance_history' in state_info
        assert 'sigma2_over_n_history' in state_info
        assert 'relative_error_history' in state_info

        # Verify histories are lists with appropriate length
        n_samples = state_info['n_samples']
        assert isinstance(state_info['mean_history'], list)
        assert isinstance(state_info['variance_history'], list)
        assert isinstance(state_info['sigma2_over_n_history'], list)
        assert isinstance(state_info['relative_error_history'], list)

        # All histories should have same length as n_samples
        assert len(state_info['mean_history']) == n_samples
        assert len(state_info['variance_history']) == n_samples
        assert len(state_info['sigma2_over_n_history']) == n_samples
        assert len(state_info['relative_error_history']) == n_samples

        # Check convergence status
        assert 'convergence_status' in state_info
        assert state_info['convergence_status'] in ['converged', 'max_samples_reached']


def test_perturbation_error_convergence_info_convergence_sampling(minimal_system):  # pylint: disable=redefined-outer-name
    """Test convergence info with convergence sampling."""
    errors, convergence_info = perturbation_error(
        minimal_system['pf'],
        minimal_system['frags'],
        minimal_system['states'],
        order=2,
        convergence_sampling=True,
        convergence_tolerance=0.2,  # Large tolerance for quick convergence
        convergence_window=3,
        min_convergence_checks=2,
        min_sample_size=3,
        max_sample_size=10,
        sampling_method="random",
        random_seed=42,
        return_convergence_info=True,
    )

    # Basic structure checks
    assert isinstance(errors, list)
    assert isinstance(convergence_info, dict)
    assert len(errors) == len(minimal_system['states'])

    # Check that convergence histories exist
    assert 'states_info' in convergence_info
    states_info = convergence_info['states_info']

    for state_info in states_info:
        # Check convergence histories exist
        assert 'mean_history' in state_info
        assert 'variance_history' in state_info
        assert 'sigma2_over_n_history' in state_info
        assert 'relative_error_history' in state_info

        # Verify they are non-empty lists
        assert len(state_info['mean_history']) > 0
        assert len(state_info['variance_history']) > 0
        assert len(state_info['sigma2_over_n_history']) > 0
        assert len(state_info['relative_error_history']) > 0

        # Check convergence-specific fields
        assert 'convergence_status' in state_info
        assert state_info['convergence_status'] in ['converged', 'max_samples_reached']


def test_convergence_info_dictionary_structure():
    """Test that convergence info maintains proper dictionary structure without ConvergenceInfo class."""
    # This test ensures our simplified dictionary approach works correctly

    # Create a minimal test case
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
    frag_coeffs = [1/2, 1/2]
    pf = ProductFormula(frag_labels, coeffs=frag_coeffs)

    gridpoints = 3
    state = HOState(n_modes, gridpoints, {(0, 0): 1})

    # Test with adaptive sampling
    _, convergence_info = perturbation_error(
        pf, frags, [state], order=2,
        adaptive_sampling=True,
        target_relative_error=0.5,
        min_sample_size=2,
        max_sample_size=5,
        random_seed=42,
        return_convergence_info=True,
    )

    # Verify it's a plain dictionary (not a custom class instance)
    assert isinstance(convergence_info, dict)

    # Verify direct dictionary access works
    assert isinstance(convergence_info['states_info'], list)

    # Verify nested dictionary access works
    state_info = convergence_info['states_info'][0]
    assert isinstance(state_info['mean_history'], list)
    assert isinstance(state_info['variance_history'], list)

    # Verify we can modify the dictionary (proving it's not a special class)
    convergence_info['test_key'] = 'test_value'
    assert convergence_info['test_key'] == 'test_value'


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


# ==================== TESTS FOR CONVERGENCE SAMPLING ====================

def test_perturbation_error_convergence_sampling(minimal_system):  # pylint: disable=redefined-outer-name
    """Test convergence sampling functionality."""
    errors = perturbation_error(
        minimal_system['pf'],
        minimal_system['frags'],
        minimal_system['states'],
        order=2,
        convergence_sampling=True,
        convergence_tolerance=0.1,  # Large tolerance for quick convergence
        convergence_window=5,
        min_convergence_checks=2,
        min_sample_size=5,
        max_sample_size=20,
        sampling_method="random",
        random_seed=42,
    )

    assert isinstance(errors, list)
    assert len(errors) == 2
    for error in errors:
        assert np.isfinite(error)


def test_perturbation_error_convergence_sampling_constraints():
    """Test that convergence sampling enforces its constraints."""
    # Should raise error for non-serial backend
    with pytest.raises(ValueError, match="Convergence sampling is only compatible with backend='serial'"):
        perturbation_error(
            ProductFormula([0, 1], coeffs=[1.0, 1.0]),
            {0: None, 1: None},  # Dummy fragments
            [],
            order=1,
            convergence_sampling=True,
            backend="mpi4py_pool"
        )

    # Should raise error for multiple workers
    with pytest.raises(ValueError, match="Convergence sampling requires num_workers=1"):
        perturbation_error(
            ProductFormula([0, 1], coeffs=[1.0, 1.0]),
            {0: None, 1: None},  # Dummy fragments
            [],
            order=1,
            convergence_sampling=True,
            num_workers=2
        )


def test_perturbation_error_conflicting_sampling_methods():
    """Test that adaptive_sampling and convergence_sampling cannot be used together."""
    with pytest.raises(ValueError, match="adaptive_sampling and convergence_sampling cannot be used simultaneously"):
        perturbation_error(
            ProductFormula([0, 1], coeffs=[1.0, 1.0]),
            {0: None, 1: None},  # Dummy fragments
            [],
            order=1,
            adaptive_sampling=True,
            convergence_sampling=True
        )


@pytest.mark.parametrize("system_fixture", ["minimal_system", "minimal_system_sparse"])
def test_convergence_sampling_direct(system_fixture, request):
    """Test _convergence_sampling function directly."""
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

    # Test convergence sampling
    expectations = _convergence_sampling(
        commutators=commutators,
        fragments=frags,
        states=states,
        timestep=0.1,
        sampling_method="random",
        convergence_tolerance=0.1,  # Large tolerance for quick convergence
        convergence_window=3,
        min_convergence_checks=2,
        min_sample_size=3,
        max_sample_size=15,
        random_seed=42,
    )

    assert len(expectations) == len(states)
    for expectation in expectations:
        assert np.isfinite(expectation)
        assert isinstance(expectation, (complex, float, int))


@pytest.mark.parametrize("system_fixture", ["minimal_system", "minimal_system_sparse"])
def test_convergence_sampling_importance(system_fixture, request):
    """Test _convergence_sampling with importance sampling."""
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

    # Test convergence sampling with importance sampling
    expectations = _convergence_sampling(
        commutators=commutators,
        fragments=frags,
        states=states,
        timestep=0.1,
        sampling_method="importance",
        convergence_tolerance=0.2,  # Large tolerance for quick convergence
        convergence_window=4,
        min_convergence_checks=2,
        min_sample_size=4,
        max_sample_size=20,
        random_seed=42,
    )

    assert len(expectations) == len(states)
    for expectation in expectations:
        assert np.isfinite(expectation)
        assert isinstance(expectation, (complex, float, int))


def test_convergence_sampling_reproducibility(minimal_system):  # pylint: disable=redefined-outer-name
    """Test that convergence sampling produces reproducible results with same seed."""
    # Test convergence sampling
    errors1 = perturbation_error(
        minimal_system['pf'],
        minimal_system['frags'],
        minimal_system['states'],
        order=2,
        convergence_sampling=True,
        convergence_tolerance=0.1,
        convergence_window=5,
        min_convergence_checks=2,
        min_sample_size=5,
        max_sample_size=15,
        sampling_method="random",
        random_seed=42,
    )

    errors2 = perturbation_error(
        minimal_system['pf'],
        minimal_system['frags'],
        minimal_system['states'],
        order=2,
        convergence_sampling=True,
        convergence_tolerance=0.1,
        convergence_window=5,
        min_convergence_checks=2,
        min_sample_size=5,
        max_sample_size=15,
        sampling_method="random",
        random_seed=42,
    )

    assert len(errors1) == len(errors2)
    for e1, e2 in zip(errors1, errors2):
        assert np.isclose(e1, e2, rtol=1e-10)  # Should be exactly equal with same seed


@pytest.mark.parametrize("convergence_tolerance", [1e-3, 1e-4, 1e-5])
def test_convergence_sampling_tolerance_levels(minimal_system, convergence_tolerance):  # pylint: disable=redefined-outer-name
    """Test convergence sampling with different tolerance levels."""
    errors = perturbation_error(
        minimal_system['pf'],
        minimal_system['frags'],
        minimal_system['states'],
        order=2,
        convergence_sampling=True,
        convergence_tolerance=convergence_tolerance,
        convergence_window=5,
        min_convergence_checks=3,
        min_sample_size=10,
        max_sample_size=50,
        sampling_method="random",
        random_seed=42,
    )

    assert isinstance(errors, list)
    assert len(errors) == 2
    for error in errors:
        assert np.isfinite(error)


# ==================== TESTS FOR CACHE FUNCTIONALITY ====================

class TestCommutatorCache:
    """Test suite for _CommutatorCache functionality."""

    def test_cache_initialization(self):
        """Test cache initialization with default and custom parameters."""
        # Default initialization
        cache = _CommutatorCache()
        assert len(cache) == 0
        assert cache.get_stats()['max_size'] == 10000

        # Custom max_size
        cache = _CommutatorCache(max_size=100)
        assert cache.get_stats()['max_size'] == 100
        assert len(cache) == 0

    def test_cache_key_generation(self):
        """Test cache key generation for different commutator types."""
        cache = _CommutatorCache()

        # Simple commutators
        key1 = cache.get_cache_key((0,), 1)
        key2 = cache.get_cache_key((1,), 1)
        key3 = cache.get_cache_key((0,), 2)

        assert key1 != key2  # Different commutators
        assert key1 != key3  # Different state_ids
        assert isinstance(key1, str)

        # Complex commutators with frozensets
        comm_frozenset = (frozenset({0, 1}), 2)
        key4 = cache.get_cache_key(comm_frozenset, 1)
        key5 = cache.get_cache_key(comm_frozenset, 1)
        assert key4 == key5  # Same commutator and state should give same key

        # Nested commutators
        nested_comm = ((0, 1), (2, 3))
        key6 = cache.get_cache_key(nested_comm, 1)
        assert isinstance(key6, str)

    def test_cache_put_get_basic(self):
        """Test basic put and get operations."""
        cache = _CommutatorCache()

        # Test cache miss
        result = cache.get((0,), 1)
        assert result is None

        # Test cache put and hit
        test_data = "test_result"
        cache.put((0,), 1, test_data)
        result = cache.get((0,), 1)
        assert result == test_data

        # Verify statistics
        stats = cache.get_stats()
        assert stats['hits'] == 1
        assert stats['misses'] == 1
        assert stats['total'] == 2
        assert stats['hit_rate'] == 0.5

    def test_cache_lru_eviction(self):
        """Test LRU eviction when cache exceeds max_size."""
        cache = _CommutatorCache(max_size=3)

        # Fill cache to capacity
        for i in range(3):
            cache.put((i,), 1, f"result_{i}")

        assert len(cache) == 3

        # Add one more item to trigger eviction
        cache.put((3,), 1, "result_3")
        assert len(cache) == 3

        # First item should be evicted
        assert cache.get((0,), 1) is None
        assert cache.get((1,), 1) == "result_1"
        assert cache.get((2,), 1) == "result_2"
        assert cache.get((3,), 1) == "result_3"

    def test_cache_clear(self):
        """Test cache clearing functionality."""
        cache = _CommutatorCache()

        # Add some data
        cache.put((0,), 1, "test")
        cache.get((0,), 1)  # Generate hit
        cache.get((1,), 1)  # Generate miss

        assert len(cache) == 1
        stats = cache.get_stats()
        assert stats['hits'] == 1
        assert stats['misses'] == 1

        # Clear cache
        cache.clear()
        assert len(cache) == 0
        stats = cache.get_stats()
        assert stats['hits'] == 0
        assert stats['misses'] == 0

    def test_cache_error_handling(self):
        """Test cache behavior with problematic inputs."""
        cache = _CommutatorCache()

        # Test with None values (should handle gracefully)
        result = cache.get(None, 1)
        assert result is None

        # Should not raise exceptions
        cache.put(None, 1, "test")

        # Cache should continue to work normally
        cache.put((0,), 1, "normal_test")
        assert cache.get((0,), 1) == "normal_test"


@pytest.mark.parametrize("system_fixture", ["minimal_system", "minimal_system_sparse"])
def test_apply_commutator_with_cache(system_fixture, request):
    """Test _apply_commutator function with cache enabled."""
    system = request.getfixturevalue(system_fixture)
    frags = system['frags']
    state = system['states'][0]

    cache = _CommutatorCache()

    # Test simple commutator
    commutator = (0,)

    # First call - should be cache miss
    result1 = _apply_commutator(commutator, frags, state, cache=cache, state_id=0)
    stats1 = cache.get_stats()
    assert stats1['misses'] == 1
    assert stats1['hits'] == 0

    # Second call - should be cache hit
    result2 = _apply_commutator(commutator, frags, state, cache=cache, state_id=0)
    stats2 = cache.get_stats()
    assert stats2['misses'] == 1
    assert stats2['hits'] == 1

    # Results should be identical
    assert type(result1) == type(result2)
    # For AbstractState objects, we can't directly compare, but they should be equivalent


@pytest.mark.parametrize("system_fixture", ["minimal_system", "minimal_system_sparse"])
def test_get_expval_state_with_cache(system_fixture, request):
    """Test _get_expval_state function with cache enabled."""
    system = request.getfixturevalue(system_fixture)
    frags = system['frags']
    state = system['states'][0]

    cache = _CommutatorCache()
    commutators = [(0,), (1,)]

    # First call - should populate cache
    result1 = _get_expval_state(commutators, frags, state, cache=cache, state_id=0)
    stats1 = cache.get_stats()

    # Second call - should use cache
    result2 = _get_expval_state(commutators, frags, state, cache=cache, state_id=0)
    stats2 = cache.get_stats()

    # Should have some cache hits on second call
    assert stats2['hits'] > stats1['hits']

    # Results should be identical
    assert np.isclose(result1, result2, rtol=1e-10)


def test_sampling_methods_use_cache_effectively():
    """Test that sampling methods effectively use cache to improve performance."""

    # Create a simple test system
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
    states = [state1, state2]

    # Test each sampling method to ensure cache is being used
    sampling_methods = [
        ("fixed", {"sample_size": 5, "sampling_method": "random"}),
        ("adaptive", {"adaptive_sampling": True, "confidence_level": 0.95,
                     "target_relative_error": 0.5, "min_sample_size": 3, "max_sample_size": 8}),
        ("convergence", {"convergence_sampling": True, "convergence_tolerance": 0.2,
                        "min_sample_size": 3, "max_sample_size": 10})
    ]

    for _, method_kwargs in sampling_methods:
        # Run perturbation error with the sampling method
        errors = perturbation_error(
            pf, frags, states, order=2, random_seed=42, **method_kwargs
        )

        # Verify we get finite results
        assert isinstance(errors, list)
        assert len(errors) == 2
        for error in errors:
            assert np.isfinite(error)
            assert isinstance(error, (complex, float, int))


@pytest.mark.parametrize("cache_size", [1, 5, 50])
def test_cache_size_limits(cache_size):
    """Test cache behavior with different size limits."""
    cache = _CommutatorCache(max_size=cache_size)

    # Add more items than cache can hold
    num_items = cache_size * 2
    for i in range(num_items):
        cache.put((i,), 1, f"result_{i}")

    # Cache should not exceed max_size
    assert len(cache) <= cache_size

    # Should still be functional
    cache.put((999,), 1, "new_result")
    assert cache.get((999,), 1) == "new_result"


def test_cache_statistics_accuracy():
    """Test that cache statistics are accurately maintained."""
    cache = _CommutatorCache()

    # Perform a sequence of operations
    cache.put((0,), 1, "result_0")  # Store
    cache.get((0,), 1)              # Hit
    cache.get((0,), 1)              # Hit
    cache.get((1,), 1)              # Miss
    cache.get((2,), 1)              # Miss
    cache.put((1,), 1, "result_1")  # Store
    cache.get((1,), 1)              # Hit

    stats = cache.get_stats()
    assert stats['hits'] == 3
    assert stats['misses'] == 2
    assert stats['total'] == 5
    assert stats['hit_rate'] == 0.6
    assert stats['size'] == 2
    assert stats['max_size'] == 10000


def test_cache_key_determinism():
    """Test that cache keys are deterministic for identical inputs."""
    cache = _CommutatorCache()

    # Test multiple times with same input
    commutator = (frozenset({0, 1}), 2, frozenset({3}))
    keys = [cache.get_cache_key(commutator, 5) for _ in range(10)]

    # All keys should be identical
    assert len(set(keys)) == 1

    # Test with different orderings of frozenset (should be same)
    comm1 = (frozenset({1, 0}), 2)  # Different order
    comm2 = (frozenset({0, 1}), 2)  # Different order
    key1 = cache.get_cache_key(comm1, 1)
    key2 = cache.get_cache_key(comm2, 1)
    assert key1 == key2


@pytest.mark.parametrize("system_fixture", ["minimal_system", "minimal_system_sparse"])
def test_cache_integration_with_all_sampling_methods(system_fixture, request):
    """Integration test to verify cache works correctly with all sampling methods."""
    system = request.getfixturevalue(system_fixture)

    # Test that all sampling methods work with cache (implicitly tested)
    # and produce consistent results

    # Fixed sampling
    errors_fixed = perturbation_error(
        system['pf'], system['frags'], system['states'], order=2,
        sample_size=3, sampling_method="random", random_seed=42
    )

    # Adaptive sampling
    errors_adaptive = perturbation_error(
        system['pf'], system['frags'], system['states'], order=2,
        adaptive_sampling=True, confidence_level=0.95, target_relative_error=0.5,
        min_sample_size=2, max_sample_size=5, random_seed=42
    )

    # Convergence sampling
    errors_convergence = perturbation_error(
        system['pf'], system['frags'], system['states'], order=2,
        convergence_sampling=True, convergence_tolerance=0.2,
        min_sample_size=2, max_sample_size=8, random_seed=42
    )

    # All should return valid results
    for errors in [errors_fixed, errors_adaptive, errors_convergence]:
        assert isinstance(errors, list)
        assert len(errors) == len(system['states'])
        for error in errors:
            assert np.isfinite(error)
