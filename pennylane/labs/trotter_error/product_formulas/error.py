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
"""Functions for retreiving effective error from fragments"""

import copy
from collections import Counter, defaultdict
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from pennylane import concurrency
from pennylane.labs.trotter_error import AbstractState, Fragment
from pennylane.labs.trotter_error.abstract import nested_commutator
from pennylane.labs.trotter_error.product_formulas.bch import bch_expansion
from pennylane.labs.trotter_error.product_formulas.product_formula import ProductFormula

# Constants
DEFAULT_CACHE_SIZE = 1000
DEFAULT_GRIDPOINTS = 10
MIN_PROBABILITY_THRESHOLD = 1e-12


@dataclass
class SamplingConfig:
    """Configuration for sampling parameters.

    Args:
        sample_size (Optional[int]): Number of commutators to sample for fixed-size sampling.
        sampling_method (str): Sampling strategy ("random", "importance", "top_k").
        random_seed (Optional[int]): Random seed for reproducibility.
    """

    sample_size: Optional[int] = None
    sampling_method: str = "importance"
    random_seed: Optional[int] = None

    def __post_init__(self):
        """Validate configuration parameters."""
        valid_methods = {"random", "importance", "top_k"}
        if self.sampling_method not in valid_methods:
            raise ValueError(
                f"sampling_method must be one of {valid_methods}, got '{self.sampling_method}'"
            )

        if self.sample_size is not None and self.sample_size <= 0:
            raise ValueError(f"sample_size must be positive, got {self.sample_size}")


class _CommutatorCache:
    """Cache for storing computed commutator applications to avoid redundant calculations.

    This cache stores the results of applying commutators to states to avoid redundant
    calculations during sampling. The cache uses a robust key generation mechanism
    and includes memory management to prevent unbounded growth.
    """

    def __init__(self, max_size: int = DEFAULT_CACHE_SIZE):
        self.cache = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - clear cache."""
        self.clear()
        return False

    @staticmethod
    def get_cache_key(commutator: Tuple[Hashable | Set], state_id: int) -> str:
        """Generate a robust cache key for commutator and state combination."""
        try:
            # Convert commutator to a hashable representation
            if isinstance(commutator, tuple):
                key_parts = []
                for item in commutator:
                    if isinstance(item, frozenset):
                        # Sort frozenset items for consistent key generation
                        sorted_items = tuple(sorted(item, key=str))
                        key_parts.append(f"frozenset({sorted_items})")
                    else:
                        key_parts.append(str(item))
                commutator_str = f"({','.join(key_parts)})"
            else:
                commutator_str = str(commutator)

            return f"comm:{commutator_str}|state:{state_id}"
        except (TypeError, ValueError, AttributeError):
            # Fallback to a simpler key if conversion fails
            return f"comm:{id(commutator)}|state:{state_id}"

    def get(self, commutator: Tuple[Hashable | Set], state_id: int):
        """Get cached result for commutator applied to state."""
        key = self.get_cache_key(commutator, state_id)
        if key in self.cache:
            self.hits += 1
            return self.cache[key]

        self.misses += 1
        return None

    def put(self, commutator: Tuple[Hashable | Set], state_id: int, result):
        """Store result in cache."""
        if len(self.cache) >= self.max_size:
            # Simple LRU: remove first item
            first_key = next(iter(self.cache))
            del self.cache[first_key]

        key = self.get_cache_key(commutator, state_id)
        self.cache[key] = result

    def clear(self):
        """Clear the cache."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0

    def get_stats(self):
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "cache_size": len(self.cache),
        }

    def __len__(self):
        return len(self.cache)


class _AdditiveIdentity:
    r"""Only used to initialize accumulators for summing Fragments"""

    def __add__(self, other):
        return other

    def __radd__(self, other):
        return other

    def __mul__(self, other):
        return self

    def __rmul__(self, other):
        return self


def effective_hamiltonian(
    product_formula: ProductFormula,
    fragments: dict[Hashable, Fragment],
    order: int,
    timestep: float = 1.0,
):
    r"""Compute the effective Hamiltonian :math:`\hat{H}_{eff} = \hat{H} + \hat{\epsilon}` that corresponds to a given product formula.

    Args:
        product_formula (ProductFormula): A product formula used to approximate the time-evolution operator for a Hamiltonian.
        fragments (Dict[Hashable, :class:`~.pennylane.labs.trotter_error.Fragment`): The fragments that sum to the Hamiltonian. The keys in the dictionary must match the labels used to build the :class:`~.pennylane.labs.trotter_error.ProductFormula` object.
        order (int): The order of the approximatation.
        timestep (float): The timestep for simulation.

    **Example**

    >>> import numpy as np
    >>> from pennylane.labs.trotter_error.fragments import vibrational_fragments
    >>> from pennylane.labs.trotter_error.product_formulas import ProductFormula, effective_hamiltonian
    >>>
    >>> n_modes = 4
    >>> r_state = np.random.RandomState(42)
    >>> freqs = r_state.random(4)
    >>> taylor_coeffs = [
    >>>     np.array(0),
    >>>     r_state.random(size=(n_modes, )),
    >>>     r_state.random(size=(n_modes, n_modes)),
    >>>     r_state.random(size=(n_modes, n_modes, n_modes))
    >>> ]
    >>>
    >>> delta = 0.001
    >>> frag_labels = [0, 1, 1, 0]
    >>> frag_coeffs = [1/2, 1/2, 1/2, 1/2]
    >>>
    >>> pf = ProductFormula(frag_labels, coeffs=frag_coeffs)
    >>> frags = dict(enumerate(vibrational_fragments(n_modes, freqs, taylor_coeffs)))
    >>> type(effective_hamiltonian(pf, frags, order=5, timestep=delta))
    <class 'pennylane.labs.trotter_error.realspace.realspace_operator.RealspaceSum'>
    """

    _validate_fragments(product_formula, fragments)

    bch = bch_expansion(product_formula(1j * timestep), order)
    eff = _AdditiveIdentity()

    for ith_order in bch:
        for commutator, coeff in ith_order.items():
            eff += coeff * nested_commutator(_insert_fragments(commutator, fragments))

    return eff


def _insert_fragments(
    commutator: tuple[Hashable], fragments: dict[Hashable, Fragment]
) -> tuple[Fragment]:
    """This function transforms a commutator of labels to a commutator of concrete `Fragment` objects.
    The function recurses through the nested structure of the tuple replacing each hashable `label` with
    the concrete value `fragments[label]`."""

    return tuple(
        _insert_fragments(term, fragments) if isinstance(term, tuple) else fragments[term]
        for term in commutator
    )


def _validate_fragments(product_formula: ProductFormula, fragments: dict[Hashable, Fragment]):
    """Validate that fragments match the product formula.

    Args:
        product_formula: The product formula to validate against
        fragments: Dictionary of fragments

    Raises:
        ValueError: If fragments do not match product formula
    """
    if not product_formula.fragments.issubset(fragments.keys()):
        raise ValueError("Fragments do not match product formula")


# =============================================================================
# Importance Probability Calculation Functions
# =============================================================================


def _get_element_norm(element, fragments: Dict[Hashable, Fragment], gridpoints: int) -> float:
    """Calculate the norm of a single commutator element."""
    if isinstance(element, frozenset):
        # Handle frozenset of weighted fragments
        weighted_fragment = sum(
            (coeff * fragments[key] for key, coeff in element if key in fragments),
            _AdditiveIdentity(),
        )
        # Check if weighted_fragment is still _AdditiveIdentity (empty sum case)
        if isinstance(weighted_fragment, _AdditiveIdentity):
            return 0.0
        return weighted_fragment.norm({"gridpoints": gridpoints})

    return fragments[element].norm({"gridpoints": gridpoints}) if element in fragments else 1.0


def _calculate_commutator_probability(commutator, fragments, timestep, gridpoints):
    r"""Calculate importance probability using:

    .. math::
        prob = 2^{k-1} \cdot timestep^k \cdot \prod \|H_i\|

    where k is commutator order and H_i are the fragment operators.
    """
    if not commutator:
        return 0.0

    # Get norms for each fragment in commutator using helper function
    fragment_norms = [_get_element_norm(element, fragments, gridpoints) for element in commutator]

    # Apply formula: 2^(k-1) * timestep^k * ∏norms
    k = len(commutator)
    prob = 2 ** (k - 1) * timestep**k * np.prod(fragment_norms)
    return max(prob, MIN_PROBABILITY_THRESHOLD)  # Use MIN_PROBABILITY_THRESHOLD


def _setup_probability_distribution(
    commutators: List[Tuple[Hashable | Set]],
    fragments: Dict[Hashable, Fragment],
    timestep: float,
    gridpoints: int = DEFAULT_GRIDPOINTS,
) -> np.ndarray:
    """Setup normalized probability distribution for importance sampling.

    This function calculates and normalizes probabilities for all commutators,
    providing a foundation for importance sampling methods.

    Args:
        commutators: List of commutator tuples
        fragments: Fragment dictionary
        timestep: Time step for probability calculation
        gridpoints: Number of gridpoints for norm calculations

    Returns:
        Normalized probability array
    """
    # Calculate raw probabilities using vectorized operations
    probabilities = np.array(
        [
            _calculate_commutator_probability(comm, fragments, timestep, gridpoints)
            for comm in commutators
        ]
    )

    # Normalize with fallback to uniform distribution
    total_prob = np.sum(probabilities)
    return (
        probabilities / total_prob
        if total_prob > 0
        else np.ones(len(commutators)) / len(commutators)
    )


def _setup_importance_probabilities(commutators, fragments, timestep, gridpoints):
    """Setup importance probabilities for all commutators (non-normalized)."""
    return np.array(
        [
            _calculate_commutator_probability(comm, fragments, timestep, gridpoints)
            for comm in commutators
        ]
    )


# =============================================================================
# Sampling Methods Layer
# =============================================================================


def _random_sampling(commutators, sample_size, random_seed=None):
    """Random uniform sampling with replacement. Weights = 1/n."""
    if random_seed is not None:
        np.random.seed(random_seed)

    n_total = len(commutators)
    indices = np.random.choice(n_total, size=sample_size, replace=True)
    sampled_commutators = [commutators[i] for i in indices]
    weights = np.full(sample_size, 1.0 / sample_size)

    return sampled_commutators, weights


def _importance_sampling(commutators, probabilities, sample_size, random_seed=None):
    """Importance sampling based on probabilities. Weights = 1/(prob_i * n)."""
    if random_seed is not None:
        np.random.seed(random_seed)

    # Normalize probabilities
    probs_normalized = probabilities / np.sum(probabilities)

    # Sample according to probabilities
    indices = np.random.choice(len(commutators), size=sample_size, replace=True, p=probs_normalized)
    sampled_commutators = [commutators[i] for i in indices]

    # Importance weights: 1 / (prob_i * n_samples)
    weights = 1.0 / (probs_normalized[indices] * sample_size)

    return sampled_commutators, weights


def _top_k_sampling(commutators, probabilities, sample_size):
    """Deterministic top-k sampling. Weights = 1.0."""
    # Get indices of top-k probabilities
    top_indices = np.argsort(probabilities)[-sample_size:]
    sampled_commutators = [commutators[i] for i in top_indices]
    weights = np.ones(sample_size)

    return sampled_commutators, weights


def _apply_sampling(commutators, fragments, config, timestep, gridpoints):
    """Dispatch to specific sampling method based on configuration."""
    method = config.sampling_method
    sample_size = config.sample_size or len(commutators)

    # For random sampling, no probabilities needed
    if method == "random":
        return _random_sampling(commutators, sample_size, config.random_seed)

    # For importance and top_k, calculate probabilities
    probabilities = _setup_importance_probabilities(commutators, fragments, timestep, gridpoints)

    if method == "importance":
        return _importance_sampling(commutators, probabilities, sample_size, config.random_seed)

    if method == "top_k":
        return _top_k_sampling(commutators, probabilities, sample_size)

    raise ValueError(f"Unknown sampling method: {method}")


def _apply_sampling_strategy(
    commutators: List[Tuple[Hashable | Set]],
    config: SamplingConfig,
    probabilities: Optional[np.ndarray] = None,
) -> Tuple[List[Tuple[Hashable | Set]], List[float]]:
    """Apply sampling strategy to select commutators and compute weights.

    This function provides a unified interface for different sampling methods.
    Currently supports 'random', 'importance', and 'top_k' methods.

    Args:
        commutators: List of all available commutators
        config: Sampling configuration
        probabilities: Pre-computed probabilities (for importance/top_k methods)

    Returns:
        Tuple of (selected_commutators, weights)

    Raises:
        ValueError: For unknown sampling methods
    """
    method = config.sampling_method
    sample_size = config.sample_size or len(commutators)

    if method == "random":
        return _random_sampling(commutators, sample_size, config.random_seed)

    if method == "importance":
        if probabilities is None:
            raise ValueError("Probabilities required for importance sampling")
        return _importance_sampling(commutators, probabilities, sample_size, config.random_seed)

    if method == "top_k":
        if probabilities is None:
            raise ValueError("Probabilities required for top-k sampling")
        return _top_k_sampling(commutators, probabilities, sample_size)

    raise ValueError(f"Unknown sampling method: {method}")


# =============================================================================
# Parallel Computation Functions
# =============================================================================


def _compute_serial(commutators, fragments, states):
    """Compute perturbation error serially."""
    expectations = []
    for state in states:
        new_state = _AdditiveIdentity()
        for commutator in commutators:
            new_state += _apply_commutator(commutator, fragments, state)
        expectations.append(state.dot(new_state))
    return expectations


def _compute_parallel_by_state(commutators, fragments, states, backend, num_workers):
    """Compute perturbation error with parallelization by state."""
    executor = concurrency.backends.get_executor(backend)
    with executor(max_workers=num_workers) as ex:
        expectations = ex.starmap(
            _get_expval_state,
            [(commutators, fragments, state) for state in states],
        )
    return expectations


def _compute_parallel_by_commutator(commutators, fragments, states, backend, num_workers):
    """Compute perturbation error with parallelization by commutator."""
    executor = concurrency.backends.get_executor(backend)
    expectations = []
    for state in states:
        with executor(max_workers=num_workers) as ex:
            applied_commutators = ex.starmap(
                _apply_commutator,
                [(commutator, fragments, state) for commutator in commutators],
            )

        new_state = _AdditiveIdentity()
        for applied_state in applied_commutators:
            new_state += applied_state

        expectations.append(state.dot(new_state))
    return expectations


def _compute_expectation_values_with_cache(
    commutators: List[Tuple[Hashable | Set]],
    weights: List[float],
    fragments: Dict[Hashable, Fragment],
    states: Sequence[AbstractState],
    use_cache: bool = True,
) -> List[float]:
    """Compute expectation values with optional caching.

    This function provides an optimized path for computing expectation values
    with caching support, which will be essential for sampling methods.

    Args:
        commutators: List of commutator tuples
        weights: Weights for each commutator
        fragments: Fragment dictionary
        states: States to compute expectation values for
        use_cache: Whether to use caching for commutator applications

    Returns:
        List of expectation values for each state
    """
    commutator_weight_pairs = list(zip(commutators, weights))
    expectations = []

    for state_idx, state in enumerate(states):
        cache = _CommutatorCache() if use_cache else None
        state_id = state_idx if use_cache else None

        expectation = _get_expval_state_with_cache(
            commutator_weight_pairs, fragments, state, cache, state_id
        )
        expectations.append(expectation)

        if cache is not None:
            cache.clear()  # Clean up cache for each state

    return expectations


# pylint: disable=too-many-arguments, too-many-positional-arguments
def perturbation_error(
    product_formula: ProductFormula,
    fragments: dict[Hashable, Fragment],
    states: Sequence[AbstractState],
    order: int,
    timestep: float = 1.0,
    num_workers: int = 1,
    backend: str = "serial",
    parallel_mode: str = "state",
) -> list[float]:
    r"""Computes the perturbation theory error using the effective Hamiltonian :math:`\hat{\epsilon} = \hat{H}_{eff} - \hat{H}` for a  given product formula.


    For a state :math:`\left| \psi \right\rangle` the perturbation theory error is given by the expectation value :math:`\left\langle \psi \right| \hat{\epsilon} \left| \psi \right\rangle`.

    Args:
        product_formula (ProductFormula): the :class:`~.pennylane.labs.trotter_error.ProductFormula` used to obtain the effective Hamiltonian
        fragments (Sequence[Fragments]): the set of :class:`~.pennylane.labs.trotter_error.Fragment`
            objects to compute the perturbation error from
        states (Sequence[AbstractState]): the states to compute expectation values from
        delta (float): time step for the trotter error operator.
        num_workers (int): the number of concurrent units used for the computation. Default value is set to 1.
        backend (string): the executor backend from the list of supported backends.
            Available options : "mp_pool", "cf_procpool", "cf_threadpool", "serial", "mpi4py_pool", "mpi4py_comm". Default value is set to "serial".
        parallel_mode (str): the mode of parallelization to use.
            Options are "state" or "commutator".
            "state" parallelizes the computation of expectation values per state,
            while "commutator" parallelizes the application of commutators to each state.
            Default value is set to "state".

    Returns:
        List[float]: the list of expectation values computed from the Trotter error operator and the input states

    **Example**

    >>> import numpy as np
    >>> from pennylane.labs.trotter_error import HOState, ProductFormula, vibrational_fragments, perturbation_error
    >>>
    >>> frag_labels = [0, 1, 1, 0]
    >>> frag_coeffs = [1/2, 1/2, 1/2, 1/2]
    >>> pf = ProductFormula(frag_labels, coeffs=frag_coeffs)
    >>>
    >>> n_modes = 2
    >>> r_state = np.random.RandomState(42)
    >>> freqs = r_state.random(n_modes)
    >>> taylor_coeffs = [
    >>>     np.array(0),
    >>>     r_state.random(size=(n_modes, )),
    >>>     r_state.random(size=(n_modes, n_modes)),
    >>>     r_state.random(size=(n_modes, n_modes, n_modes))
    >>> ]
    >>> frags = dict(enumerate(vibrational_fragments(n_modes, freqs, taylor_coeffs)))
    >>>
    >>> gridpoints = 5
    >>> state1 = HOState(n_modes, gridpoints, {(0, 0): 1})
    >>> state2 = HOState(n_modes, gridpoints, {(1, 1): 1})
    >>>
    >>> errors = perturbation_error(pf, frags, [state1, state2], order=3)
    >>> print(errors)
    [0.9189251160920877j, 4.797716682426847j]
    >>>
    >>> errors = perturbation_error(pf, frags, [state1, state2], order=3, num_workers=4, backend="mp_pool", parallel_mode="commutator")
    >>> print(errors)
    [0.9189251160920877j, 4.797716682426847j]
    """

    _validate_fragments(product_formula, fragments)

    commutators = _group_sums(bch_expansion(product_formula(1j * timestep), order))

    if backend == "serial":
        assert num_workers == 1, "num_workers must be set to 1 for serial execution."
        return _compute_serial(commutators, fragments, states)

    if parallel_mode == "state":
        return _compute_parallel_by_state(commutators, fragments, states, backend, num_workers)

    if parallel_mode == "commutator":
        return _compute_parallel_by_commutator(commutators, fragments, states, backend, num_workers)

    raise ValueError("Invalid parallel mode. Choose 'state' or 'commutator'.")


def _get_expval_state(commutators, fragments, state: AbstractState) -> float:
    """Returns the expectation value of ``state`` with respect to the operator obtained by substituting ``fragments`` into ``commutators``."""

    new_state = _AdditiveIdentity()
    for commutator in commutators:
        new_state += _apply_commutator(commutator, fragments, state)

    return state.dot(new_state)


def _get_expval_state_with_cache(
    commutators,
    fragments,
    state: AbstractState,
    cache: Optional[_CommutatorCache] = None,
    state_id: Optional[int] = None,
    weights: Optional[List[float]] = None,
) -> float:
    """Returns the expectation value of a state with respect to the operator obtained
    by substituting fragments into commutators.

    Args:
        commutators: List of commutator tuples or list of (commutator, weight) tuples
        fragments: Dictionary mapping fragment keys to Fragment objects
        state: The quantum state to compute expectation value for
        cache: Optional cache to store/retrieve computed results
        state_id: Optional state identifier for caching
        weights: Optional list of weights for commutators. If None, uniform weights of 1.0 are used.
                If commutators contains tuples, this parameter is ignored.

    Returns:
        float: The expectation value
    """
    # Compute weighted sum of applied commutators
    new_state = _AdditiveIdentity()
    for i, item in enumerate(commutators):
        if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], (int, float)):
            commutator, weight = item
        else:
            commutator = item
            weight = weights[i] if weights is not None else 1.0

        applied_state = _apply_commutator_with_cache(commutator, fragments, state, cache, state_id)

        # Skip if applied_state is _AdditiveIdentity (no contribution)
        if isinstance(applied_state, _AdditiveIdentity):
            continue

        weighted_applied_state = weight * applied_state
        new_state += weighted_applied_state

    # Return 0 if no commutators were applied, otherwise compute expectation
    return 0.0 if isinstance(new_state, _AdditiveIdentity) else state.dot(new_state)


def _apply_commutator(
    commutator: tuple[Hashable], fragments: dict[Hashable, Fragment], state: AbstractState
) -> AbstractState:
    """Returns the state obtained from applying ``commutator`` to ``state``."""

    new_state = _AdditiveIdentity()

    for term, coeff in _op_list(commutator).items():
        tmp_state = copy.copy(state)
        for frag in reversed(term):
            if isinstance(frag, frozenset):
                frag = sum(
                    (frag_coeff * fragments[x] for x, frag_coeff in frag), _AdditiveIdentity()
                )
            else:
                frag = fragments[frag]

            tmp_state = frag.apply(tmp_state)

        new_state += coeff * tmp_state

    return new_state


def _apply_commutator_with_cache(
    commutator: Tuple[Hashable],
    fragments: Dict[Hashable, Fragment],
    state: AbstractState,
    cache: Optional[_CommutatorCache] = None,
    state_id: Optional[int] = None,
) -> AbstractState:
    """Returns the state obtained from applying a commutator to a state.

    Args:
        commutator: Tuple representing a commutator structure
        fragments: Dictionary mapping fragment keys to Fragment objects
        state: The quantum state to apply the commutator to
        cache: Optional cache to store/retrieve computed results
        state_id: Optional state identifier for caching (required if cache is provided)

    Returns:
        AbstractState: The state after applying the commutator
    """
    # Try to get from cache first
    if cache is not None and state_id is not None:
        cached_result = cache.get(commutator, state_id)
        if cached_result is not None:
            return cached_result

    new_state = _AdditiveIdentity()

    for term, coeff in _op_list(commutator).items():
        tmp_state = copy.copy(state)
        for frag in reversed(term):
            if isinstance(frag, frozenset):
                frag = sum(
                    (frag_coeff * fragments[x] for x, frag_coeff in frag), _AdditiveIdentity()
                )
            else:
                frag = fragments[frag]

            tmp_state = frag.apply(tmp_state)

        new_state += coeff * tmp_state

    # Store in cache if available
    if cache is not None and state_id is not None:
        cache.put(commutator, state_id, new_state)

    return new_state


def _op_list(commutator) -> dict[tuple[Hashable], complex]:
    """Returns the operations needed to apply the commutator to a state."""

    if not commutator:
        return Counter()

    head, *tail = commutator

    if not tail:
        return Counter({(head,): 1})

    tail_ops_coeffs = _op_list(tuple(tail))

    ops1 = Counter({(head, *ops): coeff for ops, coeff in tail_ops_coeffs.items()})
    ops2 = Counter({(*ops, head): -coeff for ops, coeff in tail_ops_coeffs.items()})

    ops1.update(ops2)

    return ops1


def _group_sums(
    term_dicts: list[dict[tuple[Hashable], complex]],
) -> list[tuple[Hashable | set]]:
    """Reduce the number of commutators by grouping them using linearity in the first argument. For example,
    two commutators a*[X, A, B] and b*Y[A, B] will be merged into one commutator [a*X + b*Y, A, B].
    """
    return [
        x for xs in [_group_sums_in_dict(term_dict) for term_dict in term_dicts[1:]] for x in xs
    ]


def _group_sums_in_dict(term_dict: dict[tuple[Hashable], complex]) -> list[tuple[Hashable | set]]:
    grouped_comms = defaultdict(set)
    for commutator, coeff in term_dict.items():
        head, *tail = commutator
        tail = tuple(tail)
        grouped_comms[tail].add((head, coeff))

    return [(frozenset(heads), *tail) for tail, heads in grouped_comms.items()]
