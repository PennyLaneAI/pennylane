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


@dataclass
class SamplingConfig:
    """Configuration for sampling methods (for future use).

    This class provides a unified configuration interface for different
    sampling strategies that will be implemented in future versions.
    """

    method: str = "exact"
    sample_size: Optional[int] = None
    random_seed: Optional[int] = None

    def __post_init__(self):
        """Validate configuration parameters."""
        valid_methods = {"exact", "random", "importance", "top_k"}
        if self.method not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}, got '{self.method}'")

        if self.sample_size is not None and self.sample_size <= 0:
            raise ValueError(f"sample_size must be positive, got {self.sample_size}")


class _CommutatorCache:
    """Cache for storing computed commutator applications to avoid redundant calculations.

    This cache stores the results of applying commutators to states to avoid redundant
    calculations during sampling. Uses simple key generation and basic eviction.
    """

    def __init__(self, max_size: int = 1000):
        """Initialize the cache with optional size limit.

        Args:
            max_size: Maximum number of entries to store. When exceeded,
                     oldest entries are removed (FIFO).
        """
        self.cache = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def get_cache_key(self, commutator: Tuple[Hashable | Set], state_id: int) -> str:
        """Generate a simple cache key for a commutator-state pair."""
        try:
            # Simple string representation for key
            comm_str = str(commutator)
            return f"s{state_id}_c{comm_str}"
        except (TypeError, ValueError, AttributeError):
            # Fallback to id-based key if conversion fails
            return f"s{state_id}_c{id(commutator)}"

    def get(self, commutator: Tuple[Hashable | Set], state_id: int):
        """Retrieve cached result if available."""
        key = self.get_cache_key(commutator, state_id)
        if key in self.cache:
            self.hits += 1
            return self.cache[key]

        self.misses += 1
        return None

    def put(self, commutator: Tuple[Hashable | Set], state_id: int, result):
        """Store result in cache with simple eviction if needed."""
        # Simple eviction: remove first item if cache is full
        if len(self.cache) >= self.max_size:
            first_key = next(iter(self.cache))
            del self.cache[first_key]

        key = self.get_cache_key(commutator, state_id)
        self.cache[key] = result

    def clear(self):
        """Clear the cache and reset statistics."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0

    def get_stats(self):
        """Get cache hit/miss statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "total": total,
            "hit_rate": hit_rate,
            "size": len(self.cache),
            "max_size": self.max_size,
        }

    def __len__(self):
        """Return current cache size."""
        return len(self.cache)


class _AdditiveIdentity:
    """Only used to initialize accumulators for summing Fragments"""

    def __add__(self, other):
        return other

    def __radd__(self, other):
        return other


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

    if not product_formula.fragments.issubset(fragments.keys()):
        raise ValueError("Fragments do not match product formula")

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

    if not product_formula.fragments.issubset(fragments.keys()):
        raise ValueError("Fragments do not match product formula")

    commutators = _group_sums(bch_expansion(product_formula(1j * timestep), order))

    if backend == "serial":
        assert num_workers == 1, "num_workers must be set to 1 for serial execution."
        expectations = []
        for state in states:
            new_state = _AdditiveIdentity()
            for commutator in commutators:
                new_state += _apply_commutator(commutator, fragments, state)

            expectations.append(state.dot(new_state))

        return expectations

    if parallel_mode == "state":
        executor = concurrency.backends.get_executor(backend)
        with executor(max_workers=num_workers) as ex:
            expectations = ex.starmap(
                _get_expval_state,
                [(commutators, fragments, state) for state in states],
            )

        return expectations

    if parallel_mode == "commutator":
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

    raise ValueError("Invalid parallel mode. Choose 'state' or 'commutator'.")


def _get_expval_state(
    commutators,
    fragments,
    state: AbstractState,
    cache: Optional[_CommutatorCache] = None,
    state_id: Optional[int] = None,
) -> float:
    """Returns the expectation value of a state with respect to the operator obtained
    by substituting fragments into commutators.

    Args:
        commutators: List of commutator tuples
        fragments: Dictionary mapping fragment keys to Fragment objects
        state: The quantum state to compute expectation value for
        cache: Optional cache to store/retrieve computed results
        state_id: Optional state identifier for caching

    Returns:
        float: The expectation value
    """

    new_state = _AdditiveIdentity()
    for commutator in commutators:
        new_state += _apply_commutator(commutator, fragments, state, cache, state_id)

    # Handle case where new_state is still _AdditiveIdentity (no commutators applied)
    if isinstance(new_state, _AdditiveIdentity):
        return 0.0

    return state.dot(new_state)


def _apply_commutator(
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


def _get_gridpoints_from_states(states: Sequence[AbstractState]) -> int:
    """Extract gridpoints from states, with fallback to default value."""
    if states and hasattr(states[0], "gridpoints"):
        return states[0].gridpoints
    return 0  # Default gridpoints for norm calculations


def _calculate_commutator_probability(
    commutator: Tuple[Hashable | Set],
    fragments: Dict[Hashable, Fragment],
    timestep: float,
    gridpoints: int = 0,
) -> float:
    r"""Calculate the unnormalized probability for importance sampling a commutator.

    The unnormalized probability for a commutator :math:`C_k` of order :math:`k` is computed as:

    .. math::
        p_k = 2^{k-1} \cdot \tau^k \cdot \prod_{i=1}^{k} \|\hat{H}_i\|

    where:
    - :math:`k` is the commutator order (number of operators in the commutator)
    - :math:`\tau` is the timestep parameter
    - :math:`\hat{H}_i` are the fragment operators appearing in the commutator
    - :math:`\|\hat{H}_i\|` is the operator norm of fragment :math:`i`

    Args:
        commutator: Tuple representing a commutator structure
        fragments: Dictionary mapping fragment keys to Fragment objects
        timestep: Time step for the simulation
        gridpoints: Number of gridpoints for norm calculation

    Returns:
        float: Unnormalized probability for importance sampling
    """
    if len(commutator) == 0:
        return 0.0

    # Calculate norms for each element in the commutator
    fragment_norms = [_get_element_norm(element, fragments, gridpoints) for element in commutator]

    if not fragment_norms:
        return 0.0

    # Calculate probability: p_k = 2^(k-1) * τ^k * ∏||H_i||
    commutator_order = len(commutator)
    norm_product = np.prod(fragment_norms)
    prob = 2 ** (commutator_order - 1) * timestep**commutator_order * norm_product

    return max(prob, 1e-12)  # Avoid zero probabilities


def _get_element_norm(element, fragments: Dict[Hashable, Fragment], gridpoints: int) -> float:
    """Calculate the norm of a single commutator element.

    Args:
        element: Either a fragment key, frozenset of weighted fragments, or other element
        fragments: Dictionary mapping fragment keys to Fragment objects
        gridpoints: Number of gridpoints for norm calculation

    Returns:
        float: Norm of the element
    """
    if isinstance(element, frozenset):
        return _get_frozenset_norm(element, fragments, gridpoints)

    if element in fragments:
        return fragments[element].norm({"gridpoints": gridpoints})

    # Default norm for unrecognized elements
    return 1.0


def _get_frozenset_norm(frozenset_element, fragments: Dict[Hashable, Fragment], gridpoints: int) -> float:
    """Calculate norm for frozenset of weighted fragments.

    Args:
        frozenset_element: Frozenset containing (fragment_key, coefficient) tuples
        fragments: Dictionary mapping fragment keys to Fragment objects
        gridpoints: Number of gridpoints for norm calculation

    Returns:
        float: Norm of the combined weighted fragments
    """
    weighted_fragment = None

    for frag_data in frozenset_element:
        if not (isinstance(frag_data, tuple) and len(frag_data) == 2):
            continue

        frag_key, frag_coeff = frag_data
        if frag_key not in fragments:
            continue

        scaled_fragment = frag_coeff * fragments[frag_key]
        weighted_fragment = (
            scaled_fragment if weighted_fragment is None
            else weighted_fragment + scaled_fragment
        )

    return weighted_fragment.norm({"gridpoints": gridpoints}) if weighted_fragment is not None else 0.0


def _validate_sampling_inputs(
    product_formula: ProductFormula,
    fragments: Dict[Hashable, Fragment],
    states: Sequence[AbstractState],
    order: int,
    config: Optional[SamplingConfig] = None,
):
    """Validate basic inputs for perturbation error calculation with sampling.

    Args:
        product_formula: The product formula to analyze
        fragments: Fragment dictionary
        states: States to compute expectation values for
        order: Order of the BCH expansion
        config: Optional sampling configuration

    Raises:
        ValueError: If inputs are invalid
    """
    if not product_formula.fragments.issubset(fragments.keys()):
        raise ValueError("Fragments do not match product formula")

    if order <= 0:
        raise ValueError(f"order must be positive, got {order}")

    if not states:
        raise ValueError("states cannot be empty")

    if config is not None:
        # Additional validation for sampling config can be added here
        if config.sample_size is not None and config.method == "exact":
            raise ValueError("sample_size should not be specified when method='exact'")


def _setup_probability_distribution(
    commutators: List[Tuple[Hashable | Set]],
    fragments: Dict[Hashable, Fragment],
    timestep: float,
    gridpoints: int = 10,
) -> np.ndarray:
    """Setup normalized probability distribution for importance sampling.

    This function calculates and normalizes probabilities for all commutators,
    providing a foundation for importance sampling methods in future PRs.

    Args:
        commutators: List of commutator tuples
        fragments: Fragment dictionary
        timestep: Time step for probability calculation
        gridpoints: Number of gridpoints for norm calculations

    Returns:
        Normalized probability array
    """
    # Calculate raw probabilities
    probabilities = []
    for commutator in commutators:
        prob = _calculate_commutator_probability(commutator, fragments, timestep, gridpoints)
        probabilities.append(prob)

    probabilities = np.array(probabilities)

    # Normalize probabilities
    total_prob = np.sum(probabilities)
    if total_prob == 0:
        # Uniform distribution fallback
        return np.ones(len(commutators)) / len(commutators)

    return probabilities / total_prob


def _apply_sampling_strategy(
    commutators: List[Tuple[Hashable | Set]],
    config: SamplingConfig,
    probabilities: Optional[np.ndarray] = None,  # pylint: disable=unused-argument
) -> Tuple[List[Tuple[Hashable | Set]], List[float]]:
    """Apply sampling strategy to select commutators and compute weights.

    This function provides a unified interface for different sampling methods
    that will be implemented in future PRs. Currently supports only 'exact' method.

    Args:
        commutators: List of all available commutators
        config: Sampling configuration
        probabilities: Pre-computed probabilities (for importance/top_k methods)

    Returns:
        Tuple of (selected_commutators, weights)

    Raises:
        NotImplementedError: For sampling methods not yet implemented
    """
    if config.method == "exact":
        # Return all commutators with uniform weights
        return commutators, [1.0] * len(commutators)

    # Future PRs will implement these methods
    if config.method == "random":
        raise NotImplementedError("Random sampling will be implemented in future PR")
    if config.method == "importance":
        raise NotImplementedError("Importance sampling will be implemented in future PR")
    if config.method == "top_k":
        raise NotImplementedError("Top-k sampling will be implemented in future PR")

    raise ValueError(f"Unknown sampling method: {config.method}")


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
    expectations = []

    for state_idx, state in enumerate(states):
        if use_cache:
            cache = _CommutatorCache()
            state_id = state_idx
        else:
            cache = None
            state_id = None

        # Compute expectation value for this state
        expectation = _get_expval_state(
            list(zip(commutators, weights)),
            fragments,
            state,
            cache,
            state_id,
        )
        expectations.append(expectation)

    return expectations


# Helper function to unpack weighted commutators
def _get_expval_state_weighted(
    weighted_commutators: List[Tuple[Tuple[Hashable | Set], float]],
    fragments: Dict[Hashable, Fragment],
    state: AbstractState,
    cache: Optional[_CommutatorCache] = None,
    state_id: Optional[int] = None,
) -> float:
    """Compute expectation value with weighted commutators.

    Args:
        weighted_commutators: List of (commutator, weight) tuples
        fragments: Fragment dictionary
        state: State to compute expectation value for
        cache: Optional cache
        state_id: Optional state ID for caching

    Returns:
        Expectation value
    """
    result_state = _AdditiveIdentity()

    for commutator, weight in weighted_commutators:
        applied_state = _apply_commutator(commutator, fragments, state, cache, state_id)
        result_state += weight * applied_state

    if isinstance(result_state, _AdditiveIdentity):
        return 0.0

    return state.dot(result_state)
