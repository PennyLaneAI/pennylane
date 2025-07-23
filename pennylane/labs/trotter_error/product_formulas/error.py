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
import time
import warnings
from collections import Counter, defaultdict
from typing import List, Sequence, Tuple, Hashable, Dict, Set, Optional

from tqdm import tqdm
import numpy as np

from pennylane import concurrency
from pennylane.labs.trotter_error import AbstractState, Fragment
from pennylane.labs.trotter_error.abstract import nested_commutator
from pennylane.labs.trotter_error.product_formulas.bch import bch_expansion
from pennylane.labs.trotter_error.product_formulas.product_formula import ProductFormula


class _CommutatorCache:
    """Cache for storing computed commutator applications to avoid redundant calculations.

    This cache stores the results of applying commutators to states to avoid redundant
    calculations during sampling. The cache uses a robust key generation mechanism
    and includes memory management to prevent unbounded growth.
    """

    def __init__(self, max_size: int = 10000):
        """Initialize the cache with optional size limit.

        Args:
            max_size: Maximum number of entries to store. When exceeded,
                     oldest entries are removed (LRU-like behavior).
        """
        self._cache = {}
        self._access_order = []  # Track access order for LRU eviction
        self._hits = 0
        self._misses = 0
        self._max_size = max_size

    def get_cache_key(self, commutator: Tuple[Hashable | Set], state_id: int) -> str:
        """Generate a unique cache key for a commutator-state pair.

        Uses a more robust key generation that avoids hash collisions by
        creating a deterministic string representation.
        """
        # Convert commutator to a deterministic hashable representation
        def make_hashable(item):
            if isinstance(item, frozenset):
                # Sort frozenset items for deterministic ordering
                sorted_items = sorted(str(x) for x in item)
                return f"fs({','.join(sorted_items)})"
            if isinstance(item, tuple):
                return f"t({','.join(make_hashable(x) for x in item)})"
            return str(item)

        # Create deterministic string key instead of using hash()
        comm_str = make_hashable(commutator)
        return f"s{state_id}_c{comm_str}"

    def get(self, commutator: Tuple[Hashable | Set], state_id: int):
        """Retrieve cached result if available."""
        try:
            key = self.get_cache_key(commutator, state_id)
            if key in self._cache:
                self._hits += 1
                # Update access order for LRU
                if key in self._access_order:
                    self._access_order.remove(key)
                self._access_order.append(key)
                return self._cache[key]
            self._misses += 1
            return None
        except (TypeError, ValueError, AttributeError):
            # If key generation fails, count as miss and continue
            self._misses += 1
            return None

    def put(self, commutator: Tuple[Hashable | Set], state_id: int, result):
        """Store result in cache with LRU eviction if needed."""
        try:
            key = self.get_cache_key(commutator, state_id)

            # Remove oldest entries if cache is full
            while len(self._cache) >= self._max_size and self._access_order:
                oldest_key = self._access_order.pop(0)
                if oldest_key in self._cache:
                    del self._cache[oldest_key]

            # Store new entry
            self._cache[key] = result
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)

        except (TypeError, ValueError, AttributeError):
            # If caching fails, silently continue (cache is optional optimization)
            pass

    def clear(self):
        """Clear the cache and reset statistics."""
        self._cache.clear()
        self._access_order.clear()
        self._hits = 0
        self._misses = 0

    def get_stats(self):
        """Get cache hit/miss statistics."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        return {
            'hits': self._hits,
            'misses': self._misses,
            'total': total,
            'hit_rate': hit_rate,
            'size': len(self._cache),
            'max_size': self._max_size
        }

    def __len__(self):
        """Return current cache size."""
        return len(self._cache)


class _AdditiveIdentity:
    """Only used to initialize accumulators for summing Fragments"""

    def __add__(self, other):
        return other

    def __radd__(self, other):
        return other


def effective_hamiltonian(
    product_formula: ProductFormula,
    fragments: Dict[Hashable, Fragment],
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
    commutator: Tuple[Hashable], fragments: Dict[Hashable, Fragment]
) -> Tuple[Fragment]:
    """This function transforms a commutator of labels to a commutator of concrete `Fragment` objects.
    The function recurses through the nested structure of the tuple replacing each hashable `label` with
    the concrete value `fragments[label]`."""

    return tuple(
        _insert_fragments(term, fragments) if isinstance(term, tuple) else fragments[term]
        for term in commutator
    )


def _handle_return_value(expectations, convergence_info, return_convergence_info):
    """Helper function to handle return values consistently."""
    if return_convergence_info:
        return expectations, convergence_info
    return expectations


# pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-branches, too-many-statements
def perturbation_error(
    product_formula: ProductFormula,
    fragments: Dict[Hashable, Fragment],
    states: Sequence[AbstractState],
    order: int,
    timestep: float = 1.0,
    num_workers: int = 1,
    backend: str = "serial",
    parallel_mode: str = "state",
    sample_size: Optional[int] = None,
    sampling_method: str = "importance",
    random_seed: Optional[int] = None,
    adaptive_sampling: bool = False,
    confidence_level: float = 0.95,
    target_relative_error: float = 1.,
    min_sample_size: int = 10,
    max_sample_size: int = 10000,
    convergence_sampling: bool = False,
    convergence_tolerance: float = 1e-6,
    convergence_window: int = 10,
    min_convergence_checks: int = 3,
    return_convergence_info: bool = False,
):
    r"""Computes the perturbation theory error using the effective Hamiltonian :math:`\hat{\epsilon} = \hat{H}_{eff} - \hat{H}` for a  given product formula.


    For a state :math:`\left| \psi \right\rangle` the perturbation theory error is given by the expectation value :math:`\left\langle \psi \right| \hat{\epsilon} \left| \psi \right\rangle`.

    Args:
        product_formula (ProductFormula): the :class:`~.pennylane.labs.trotter_error.ProductFormula` used to obtain the effective Hamiltonian
        fragments (Sequence[Fragments]): the set of :class:`~.pennylane.labs.trotter_error.Fragment`
            objects to compute the perturbation error from
        states (Sequence[AbstractState]): the states to compute expectation values from
        order (int): the order of the BCH expansion
        timestep (float): time step for the trotter error operator.
        num_workers (int): the number of concurrent units used for the computation. Default value is set to 1.
        backend (string): the executor backend from the list of supported backends.
            Available options : "serial", "mpi4py_pool", "mpi4py_comm". Default value is set to "serial".
        parallel_mode (str): the mode of parallelization to use.
            Options are "state" or "commutator".
            "state" parallelizes the computation of expectation values per state,
            while "commutator" parallelizes the application of commutators to each state.
            Default value is set to "state".
        sample_size (Optional[int]): Number of commutators to sample for fixed-size sampling.
            If None (default), uses all commutators exactly without sampling for maximum accuracy.
            Only when a specific number is provided will fixed-size sampling be applied.
            This parameter is ignored if adaptive_sampling or convergence_sampling is True.
        sampling_method (str): Sampling strategy to use. Options are "random" for uniform random sampling
            or "importance" for importance sampling based on commutator magnitudes.
            Only used when sample_size is specified, adaptive_sampling is True, or convergence_sampling is True.
        random_seed (Optional[int]): Random seed for reproducibility in sampling methods.
        adaptive_sampling (bool): If True, uses adaptive sampling that dynamically determines
            the optimal sample size based on variance convergence criteria. When enabled,
            the sample_size parameter is ignored since the algorithm determines the required
            number of samples automatically. Note: adaptive sampling is only compatible with
            backend='serial' and num_workers=1. Cannot be used simultaneously with convergence_sampling.
        confidence_level (float): Confidence level for adaptive sampling (e.g., 0.95 for 95% confidence).
            Only used when adaptive_sampling is True.
        target_relative_error (float): Target relative error for adaptive sampling (epsilon in :math:`N \geq z^2\sigma^2/\varepsilon^2`).
            Only used when adaptive_sampling is True.
        min_sample_size (int): Minimum sample size for adaptive sampling or convergence sampling.
            Only used when adaptive_sampling or convergence_sampling is True.
        max_sample_size (int): Maximum sample size for adaptive sampling or convergence sampling to prevent infinite loops.
            Only used when adaptive_sampling or convergence_sampling is True.
        convergence_sampling (bool): If True, uses convergence sampling that stops when the mean has converged
            to a certain precision. When enabled, the sample_size parameter is ignored since the algorithm
            determines when to stop based on mean convergence. Note: convergence sampling is only compatible with
            backend='serial' and num_workers=1. Cannot be used simultaneously with adaptive_sampling.
        convergence_tolerance (float): Relative tolerance for mean convergence in convergence sampling (default: 1e-6).
            The algorithm stops when the relative change in the mean over the convergence window is below this threshold.
            Only used when convergence_sampling is True.
        convergence_window (int): Number of samples to look back for convergence check in convergence sampling (default: 10).
            Only used when convergence_sampling is True.
        min_convergence_checks (int): Minimum number of consecutive convergence checks that must pass before stopping
            in convergence sampling (default: 3). Only used when convergence_sampling is True.
        return_convergence_info (bool): If True, returns a tuple (expectation_values, convergence_info)
            where convergence_info is a dictionary containing detailed statistics about the sampling process including
            mean and variance histories, execution times, cache performance, and distribution information.
            If False (default), returns only the expectation values for backward compatibility.

    Returns:
        List[float] or Tuple[List[float], Dict]: If return_convergence_info is False,
            returns the list of expectation values. If True, returns a tuple containing both
            the expectation values and a dictionary with detailed convergence information.

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
    >>> errors, conv_info = perturbation_error(pf, frags, [state1, state2], order=3,
    ...                                         adaptive_sampling=True, return_convergence_info=True)
    >>> print(conv_info['states'][0]['mean_history'])  # Mean evolution for first state
    """

    if not product_formula.fragments.issubset(fragments.keys()):
        raise ValueError("Fragments do not match product formula")

    commutators = _group_sums(bch_expansion(product_formula(1j * timestep), order))

    # Initialize convergence info if requested
    convergence_info = None
    if return_convergence_info:
        convergence_info = {
            'global': {
                'total_commutators': len(commutators),
                'sampling_method': sampling_method,
                'order': order,
                'timestep': timestep,
            },
            'states_info': []
        }

    # Check for conflicting sampling methods
    if adaptive_sampling and convergence_sampling:
        raise ValueError("adaptive_sampling and convergence_sampling cannot be used simultaneously")

    # Handle adaptive sampling first (it has priority)
    if adaptive_sampling:
        if sample_size is not None:
            warnings.warn("sample_size is ignored when adaptive_sampling=True", UserWarning)

        # Adaptive sampling is only compatible with serial execution
        if backend != "serial":
            raise ValueError("Adaptive sampling is only compatible with backend='serial'")
        if num_workers != 1:
            raise ValueError("Adaptive sampling requires num_workers=1")

        # Get gridpoints from the first state (all states should have the same gridpoints)
        if states and hasattr(states[0], 'gridpoints'):
            gridpoints = states[0].gridpoints
        else:
            gridpoints = None

        # Perform adaptive sampling for each state
        expectations = _adaptive_sampling(
            commutators=commutators,
            fragments=fragments,
            states=states,
            timestep=timestep,
            sampling_method=sampling_method,
            confidence_level=confidence_level,
            target_error=target_relative_error,
            min_sample_size=min_sample_size,
            max_sample_size=max_sample_size,
            random_seed=random_seed,
            gridpoints=gridpoints or 10,  # Default gridpoints if not found
            convergence_info=convergence_info,
        )

        # Return results for adaptive sampling
        return _handle_return_value(expectations, convergence_info, return_convergence_info)

    # Handle convergence sampling
    if convergence_sampling:
        if sample_size is not None:
            warnings.warn("sample_size is ignored when convergence_sampling=True", UserWarning)

        # Convergence sampling is only compatible with serial execution
        if backend != "serial":
            raise ValueError("Convergence sampling is only compatible with backend='serial'")
        if num_workers != 1:
            raise ValueError("Convergence sampling requires num_workers=1")

        # Get gridpoints from the first state (all states should have the same gridpoints)
        if states and hasattr(states[0], 'gridpoints'):
            gridpoints = states[0].gridpoints
        else:
            gridpoints = None

        # Perform convergence sampling for each state
        expectations = _convergence_sampling(
            commutators=commutators,
            fragments=fragments,
            states=states,
            timestep=timestep,
            sampling_method=sampling_method,
            convergence_tolerance=convergence_tolerance,
            convergence_window=convergence_window,
            min_convergence_checks=min_convergence_checks,
            min_sample_size=min_sample_size,
            max_sample_size=max_sample_size,
            random_seed=random_seed,
            gridpoints=gridpoints or 10,  # Default gridpoints if not found
            convergence_info=convergence_info,
        )

        # Return results for convergence sampling
        return _handle_return_value(expectations, convergence_info, return_convergence_info)

    # Handle fixed-size sampling if explicitly requested
    if sample_size is not None:
        # Get gridpoints from the first state (all states should have the same gridpoints)
        if states and hasattr(states[0], 'gridpoints'):
            gridpoints = states[0].gridpoints
        else:
            gridpoints = None
        expectations = _fixed_sampling(
            commutators=commutators,
            fragments=fragments,
            states=states,
            timestep=timestep,
            sample_size=sample_size,
            sampling_method=sampling_method,
            random_seed=random_seed,
            gridpoints=gridpoints,
            num_workers=num_workers,
            backend=backend,
            parallel_mode=parallel_mode,
            convergence_info=convergence_info,
        )

        # Return results for fixed sampling
        return _handle_return_value(expectations, convergence_info, return_convergence_info)

    # Use all commutators exactly (no sampling)
    commutator_weights = [1.0] * len(commutators)

    # Use the shared expectation value computation function
    expectations = _compute_expectation_values(
        commutators, commutator_weights, fragments, states,
        num_workers, backend, parallel_mode, convergence_info
    )

    # Return based on whether convergence info was requested
    if return_convergence_info:
        # For exact computation, update the method field for each state
        if convergence_info and convergence_info.get('states_info'):
            for state_info in convergence_info['states_info']:
                state_info['method'] = 'exact'
                state_info['variance'] = 0.0  # No variance in exact computation
                state_info['sigma2_over_n'] = 0.0  # No sampling uncertainty
            convergence_info['global']['sampled_commutators'] = len(commutators)

    return _handle_return_value(expectations, convergence_info, return_convergence_info)


def _get_expval_state(
    commutators,
    fragments,
    state: AbstractState,
    cache: Optional[_CommutatorCache] = None,
    state_id: Optional[int] = None
) -> float:
    """
    Returns the expectation value of a state with respect to the operator obtained
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
        result = 0.0
    else:
        result = state.dot(new_state)

    return result


def _apply_commutator(
    commutator: Tuple[Hashable],
    fragments: Dict[Hashable, Fragment],
    state: AbstractState,
    cache: Optional[_CommutatorCache] = None,
    state_id: Optional[int] = None
) -> AbstractState:
    """
    Returns the state obtained from applying a commutator to a state.

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
                # Handle tuple wrapping from _op_list structure
                if isinstance(frag, tuple) and len(frag) == 1:
                    frag_key = frag[0]
                else:
                    frag_key = frag
                frag = fragments[frag_key]

            tmp_state = frag.apply(tmp_state)

        new_state += coeff * tmp_state

    # Store in cache if available
    if cache is not None and state_id is not None:
        cache.put(commutator, state_id, new_state)

    return new_state


def _op_list(commutator) -> Dict[Tuple[Hashable], complex]:
    """
    Returns the operations needed to apply the commutator to a state.

    Args:
        commutator: Tuple representing a commutator structure

    Returns:
        Dict[Tuple[Hashable], complex]: Dictionary mapping operation sequences to coefficients
    """

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
    term_dicts: List[Dict[Tuple[Hashable], complex]],
) -> List[Tuple[Hashable | Set]]:
    """
    Reduce the number of commutators by grouping them using linearity in the first argument.

    For example, two commutators :math:`a \\cdot [X, A, B]` and :math:`b \\cdot [Y, A, B]`
    will be merged into one commutator :math:`[a \\cdot X + b \\cdot Y, A, B]`.

    Args:
        term_dicts: List of dictionaries mapping commutator tuples to complex coefficients

    Returns:
        List of grouped commutator tuples
    """
    return [
        x for xs in [_group_sums_in_dict(term_dict) for term_dict in term_dicts[1:]] for x in xs
    ]


def _group_sums_in_dict(term_dict: Dict[Tuple[Hashable], complex]) -> List[Tuple[Hashable | Set]]:
    """
    Group commutators in a single dictionary by their tail structure.

    Args:
        term_dict: Dictionary mapping commutator tuples to complex coefficients

    Returns:
        List of grouped commutator tuples with frozensets for combined heads
    """
    grouped_comms = defaultdict(set)
    for commutator, coeff in term_dict.items():
        head, *tail = commutator
        tail = tuple(tail)
        grouped_comms[tail].add((head, coeff))

    return [(frozenset(heads), *tail) for tail, heads in grouped_comms.items()]


def _calculate_commutator_probability(
    commutator: Tuple[Hashable | Set],
    fragments: Dict[Hashable, Fragment],
    timestep: float,
    gridpoints: int = 10
) -> float:
    r"""
    Calculate the unnormalized probability for importance sampling a commutator.

    The unnormalized probability for a commutator :math:`C_k` of order :math:`k` is computed as:

    .. math::
        p_k = 2^{k-1} \cdot \tau^k \cdot \prod_{i=1}^{k} \|\hat{H}_i\|

    where:

    - :math:`k` is the commutator order (number of operators in the commutator)
    - :math:`\tau` is the timestep parameter
    - :math:`\hat{H}_i` are the fragment operators appearing in the commutator
    - :math:`\|\hat{H}_i\|` is the operator norm of fragment :math:`i`

    The factor :math:`2^{k-1}` accounts for the number of terms generated when expanding
    the nested commutator structure, and :math:`\tau^k` reflects the scaling of
    :math:`k`-th order terms in the Baker-Campbell-Hausdorff expansion.

    For commutators containing weighted combinations of fragments (frozensets), the norm
    is computed for the weighted sum :math:`\|\sum_j c_j \hat{H}_j\|` where :math:`c_j`
    are the coefficients and :math:`\hat{H}_j` are the individual fragments.

    Args:
        commutator: Tuple representing a commutator structure
        fragments: Dictionary mapping fragment keys to Fragment objects
        timestep: Time step for the simulation
        gridpoints: Number of gridpoints for norm calculation

    Returns:
        float: Unnormalized probability for importance sampling
    """
    # Handle empty commutator
    if len(commutator) == 0:
        return 0.0

    # Extract fragment norms from the commutator structure
    fragment_norms = []
    commutator_order = len(commutator)

    for element in commutator:
        if isinstance(element, frozenset):
            # Handle frozenset of weighted fragments like {(1, coeff1), (0, coeff2)}
            weighted_fragment = None
            for frag_data in element:
                if isinstance(frag_data, tuple) and len(frag_data) == 2:
                    frag_key, frag_coeff = frag_data
                    if frag_key in fragments:
                        scaled_fragment = frag_coeff * fragments[frag_key]
                        weighted_fragment = scaled_fragment if weighted_fragment is None else weighted_fragment + scaled_fragment

            # Compute the norm of the weighted sum
            if weighted_fragment is not None:
                frozenset_norm = weighted_fragment.norm({"gridpoints": gridpoints})
                fragment_norms.append(frozenset_norm)
            else:
                fragment_norms.append(0.0)
        elif element in fragments:
            # Handle direct fragment keys
            fragment_norms.append(fragments[element].norm({"gridpoints": gridpoints}))
        else:
            # For other elements (like indices), assign a default norm
            fragment_norms.append(1.0)

    if not fragment_norms:
        return 0.0

    # Calculate probability based on commutator structure and fragment norms
    norm_product = np.prod(fragment_norms)
    prob = 2**(commutator_order-1) * timestep**commutator_order * norm_product

    return max(prob, 1e-12)  # Avoid zero probabilities


# pylint: disable=too-many-branches, too-many-statements
# pylint: disable=too-many-arguments
def _adaptive_sampling(
    commutators: List[Tuple[Hashable | Set]],
    fragments: Dict[Hashable, Fragment],
    states: Sequence[AbstractState],
    timestep: float,
    sampling_method: str,
    confidence_level: float,
    target_error: float,
    min_sample_size: int,
    max_sample_size: int,
    random_seed: Optional[int] = None,
    gridpoints: int = 10,
    convergence_info: Optional[Dict] = None,
) -> List[float]:
    r"""
    Adaptive sampling that continues until variance-based stopping criterion is met.

    Uses the criterion: :math:`N \geq z^2\sigma^2/\varepsilon^2` where:

    - :math:`N` is the sample size
    - :math:`z` is the z-score for the given confidence level
    - :math:`\sigma` is the sample standard deviation
    - :math:`\varepsilon` is the target error (relative to the mean)

    Args:
        commutators: List of all available commutators
        fragments: Dictionary mapping fragment keys to Fragment objects
        states: States to compute expectation values for
        timestep: Time step for simulation
        sampling_method: "random" or "importance"
        confidence_level: Confidence level for z-score calculation
        target_error: Target relative error for adaptive sampling (epsilon in :math:`N \geq z^2\sigma^2/\varepsilon^2`).
            Only used when adaptive_sampling is True.
        min_sample_size: Minimum sample size for adaptive sampling.
            Only used when adaptive_sampling is True.
        max_sample_size: Maximum sample size for adaptive sampling to prevent infinite loops.
            Only used when adaptive_sampling is True.
        random_seed: Random seed for reproducibility

    Returns:
        List of expectation values for each state
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Get z-score for confidence level
    z_score = _get_confidence_z_score(confidence_level)

    # Setup probabilities for sampling
    prob_start_time = time.time()
    probabilities = _setup_importance_probabilities(
        commutators, fragments, timestep, gridpoints, sampling_method
    )
    prob_time = time.time() - prob_start_time

    # Record probability calculation time
    if convergence_info:
        convergence_info['global']['probability_calculation_time'] = prob_time

    # Process each state with adaptive sampling
    expectations = []

    for state_idx, state in enumerate(states):
        expectation = _adaptive_sample_single_state(
            state=state,
            state_idx=state_idx,
            commutators=commutators,
            fragments=fragments,
            probabilities=probabilities,
            sampling_method=sampling_method,
            z_score=z_score,
            target_error=target_error,
            min_sample_size=min_sample_size,
            max_sample_size=max_sample_size,
            convergence_info=convergence_info,
        )
        expectations.append(expectation)

    # Record total samples information for adaptive sampling
    if convergence_info:
        total_samples = sum(state_info['n_samples'] for state_info in convergence_info['states_info'])
        convergence_info['global']['sampled_commutators'] = total_samples

    return expectations


def _get_confidence_z_score(confidence_level: float) -> float:
    """
    Get the z-score for a given confidence level.

    Args:
        confidence_level: Confidence level (e.g., 0.95 for 95% confidence)

    Returns:
        float: Corresponding z-score
    """
    # Avoid scipy dependency by using a lookup table for common confidence levels
    z_score_dict = {
        0.68: 1.0,      # 68% confidence interval (1 standard deviation)
        0.90: 1.645,    # 90% confidence interval
        0.95: 1.96,     # 95% confidence interval
        0.9545: 2.0,    # ~95.45% confidence interval (2 standard deviations)
        0.99: 2.576     # 99% confidence interval
    }
    return z_score_dict.get(confidence_level, 1.0)  # Default to 68% confidence (z=1.0)


def _setup_importance_probabilities(
    commutators: List[Tuple[Hashable | Set]],
    fragments: Dict[Hashable, Fragment],
    timestep: float,
    gridpoints: int,
    sampling_method: str,
) -> Optional[np.ndarray]:
    r"""
    Setup probabilities for importance sampling.

    Args:
        commutators: List of all available commutators
        fragments: Dictionary mapping fragment keys to Fragment objects
        timestep: Time step for simulation
        gridpoints: Number of gridpoints for norm calculations
        sampling_method: "random" or "importance"

    Returns:
        np.ndarray or None: Normalized probabilities for importance sampling,
                           or None for random sampling
    """
    if sampling_method == "random":
        return None

    # Pre-calculate importance sampling probabilities
    probabilities = []

    for commutator in tqdm(commutators, desc="Calculating probabilities", unit="commutator"):
        prob = _calculate_commutator_probability(commutator, fragments, timestep, gridpoints)
        probabilities.append(prob)

    probabilities = np.array(probabilities)
    total_prob_sum = np.sum(probabilities)

    if total_prob_sum == 0:
        warnings.warn("All probabilities are zero, falling back to uniform distribution", UserWarning)
        probabilities = np.ones(len(commutators)) / len(commutators)
    else:
        probabilities = probabilities / total_prob_sum

    return probabilities


def _adaptive_sample_single_state(
    state: AbstractState,
    state_idx: int,
    commutators: List[Tuple[Hashable | Set]],
    fragments: Dict[Hashable, Fragment],
    probabilities: Optional[np.ndarray],
    sampling_method: str,
    z_score: float,
    target_error: float,
    min_sample_size: int,
    max_sample_size: int,
    convergence_info: Optional[Dict] = None,
) -> float:
    """
    Perform adaptive sampling for a single state until convergence.

    Args:
        state: The quantum state to compute expectation value for
        state_idx: Index of the state (for logging)
        commutators: List of all commutators
        fragments: Dictionary mapping fragment keys to Fragment objects
        probabilities: Pre-calculated probabilities for importance sampling (None for random)
        sampling_method: "random" or "importance"
        z_score: Z-score for confidence interval calculation
        target_error: Target relative error for adaptive sampling (epsilon in :math:`N \\geq z^2\\sigma^2/\\varepsilon^2`).
            Only used when adaptive_sampling is True.
        min_sample_size: Minimum sample size for adaptive sampling.
            Only used when adaptive_sampling is True.
        max_sample_size: Maximum sample size for adaptive sampling to prevent infinite loops.
            Only used when adaptive_sampling is True.

    Returns:
        float: Final expectation value estimate
    """
    state_start_time = time.time()
    print(f"\n=== Processing State {state_idx + 1} ===")

    # Initialize cache for this state
    cache = _CommutatorCache()

    # Initialize statistics for this state
    n_samples = 0
    sum_values = 0.0
    sum_squared = 0.0
    last_report_time = time.time()
    report_interval = 10  # Report every 10 samples initially

    # Initialize convergence tracking histories
    mean_history = []
    variance_history = []
    sigma2_over_n_history = []
    relative_error_history = []

    # Start sampling
    while n_samples < max_sample_size:
        sample_start_time = time.time()

        # Sample one commutator and compute its contribution
        if sampling_method == "random":
            idx = np.random.choice(len(commutators))
            weight = len(commutators)  # Scaling factor for uniform sampling
            commutator = commutators[idx]
        else:  # importance sampling
            idx = np.random.choice(len(commutators), p=probabilities)
            weight = 1.0 / probabilities[idx]  # Importance weight
            commutator = commutators[idx]

        # Compute the contribution of this commutator using cache
        applied_state = _apply_commutator(commutator, fragments, state, cache, state_idx)
        if isinstance(applied_state, _AdditiveIdentity):
            contribution = 0.0
        else:
            contribution = weight * state.dot(applied_state)

        # Update running statistics
        n_samples += 1
        sum_values += contribution
        sum_squared += contribution * contribution

        # Calculate current statistics for convergence tracking
        current_mean = sum_values / n_samples
        current_variance = (sum_squared - n_samples * current_mean * current_mean) / (n_samples - 1) if n_samples > 1 else 0.0
        current_variance = abs(current_variance)  # Handle complex variance
        current_sigma2_over_n = current_variance / n_samples

        # Store convergence history
        mean_history.append(current_mean)
        variance_history.append(current_variance)
        sigma2_over_n_history.append(current_sigma2_over_n)

        # Calculate and store relative error history
        if abs(current_mean) > 1e-12:
            current_relative_error = np.sqrt(current_variance) / abs(current_mean)
            relative_error_history.append(current_relative_error)
        else:
            relative_error_history.append(float('inf'))  # Handle near-zero mean

        sample_time = time.time() - sample_start_time

        # Progress reporting
        current_time = time.time()
        if (n_samples % report_interval == 0) or (current_time - last_report_time > 5.0):
            std_dev = np.sqrt(current_variance)

            # Add cache statistics to progress reporting
            cache_stats = cache.get_stats()
            print(f"  Sample {n_samples:4d}: "
                  f"mean={current_mean:8.3e}, "
                  f"std={std_dev:8.3e}, "
                  f"cache_hit_rate={cache_stats['hit_rate']:.1%}, "
                  f"time={sample_time:.3f}s")

            last_report_time = current_time

            # Adaptive reporting interval: start with 10, then 50, then 100
            if n_samples >= 50 and report_interval == 10:
                report_interval = 50
            elif n_samples >= 200 and report_interval == 50:
                report_interval = 100

        # Check stopping criterion after minimum samples
        if n_samples >= min_sample_size:
            mean = sum_values / n_samples
            variance = (sum_squared - n_samples * mean * mean) / (n_samples - 1) if n_samples > 1 else 0
            variance_abs = abs(variance)
            std_dev = np.sqrt(variance_abs)

            # Avoid division by zero
            if abs(mean) > 1e-12:
                relative_error = std_dev / abs(mean)
                required_samples = (z_score * relative_error / target_error) ** 2

                if n_samples >= required_samples:
                    convergence_time = time.time() - state_start_time
                    cache_stats = cache.get_stats()
                    print(f"  ✓ CONVERGED after {n_samples} samples in {convergence_time:.2f}s")
                    print(f"    Required samples: {required_samples:.1f}")
                    print(f"    Final mean: {mean:.6e}")
                    print(f"    Final std: {std_dev:.6e}")
                    print(f"    Relative error: {relative_error:.6e} (target: {target_error:.6e})")
                    print(f"    Cache performance: {cache_stats['hits']} hits, {cache_stats['misses']} misses ({cache_stats['hit_rate']:.1%} hit rate)")
                    break
            else:
                # If mean is very close to zero, use absolute error criterion
                if std_dev < target_error:
                    convergence_time = time.time() - state_start_time
                    cache_stats = cache.get_stats()
                    print(f"  ✓ CONVERGED after {n_samples} samples in {convergence_time:.2f}s (mean ≈ 0)")
                    print(f"    Final std: {std_dev:.6e} (target: {target_error:.6e})")
                    print(f"    Cache performance: {cache_stats['hits']} hits, {cache_stats['misses']} misses ({cache_stats['hit_rate']:.1%} hit rate)")
                    break

    final_mean = sum_values / n_samples if n_samples > 0 else 0.0
    state_time = time.time() - state_start_time

    if n_samples >= max_sample_size:
        cache_stats = cache.get_stats()
        print(f"  ⚠ STOPPED at maximum samples ({max_sample_size}) for state {state_idx + 1}")
        print(f"    Time spent: {state_time:.2f}s")
        print(f"    Final mean: {final_mean:.6e}")
        variance = (sum_squared - n_samples * final_mean * final_mean) / (n_samples - 1) if n_samples > 1 else 0
        variance_abs = abs(variance)
        std_dev = np.sqrt(variance_abs)
        print(f"    Final std: {std_dev:.6e}")
        if abs(final_mean) > 1e-12:
            rel_err = std_dev / abs(final_mean)
            print(f"    Achieved relative error: {rel_err:.6e} (target: {target_error:.6e})")
        print(f"    Cache performance: {cache_stats['hits']} hits, {cache_stats['misses']} misses ({cache_stats['hit_rate']:.1%} hit rate)")

    print(f"  State {state_idx + 1} completed in {state_time:.2f}s")

    # Add convergence information if requested
    if convergence_info:
        # Calculate final statistics
        final_variance = (sum_squared - n_samples * final_mean * final_mean) / (n_samples - 1) if n_samples > 1 else 0.0
        final_variance = abs(final_variance)  # Handle complex variance
        final_sigma2_over_n = final_variance / n_samples if n_samples > 0 else 0.0
        final_cache_stats = cache.get_stats()

        # Determine convergence method and add relevant info
        convergence_details = {
            'sampling_method': sampling_method,
            'z_score': z_score,
            'target_error': target_error,
            'min_sample_size': min_sample_size,
            'max_sample_size': max_sample_size,
        }

        # Add final statistics based on convergence outcome
        if n_samples < max_sample_size:
            convergence_details['convergence_status'] = 'converged'
            if abs(final_mean) > 1e-12:
                final_relative_error = np.sqrt(final_variance) / abs(final_mean)
                required_samples = (z_score * final_relative_error / target_error) ** 2
                convergence_details.update({
                    'relative_error': final_relative_error,
                    'required_samples': required_samples,
                    'z2_sigma2_over_eps2': z_score**2 * final_variance / target_error**2,
                })
        else:
            convergence_details['convergence_status'] = 'max_samples_reached'

        # Add state info using dictionary approach
        state_info = {
            'mean': final_mean,
            'variance': final_variance,
            'sigma2_over_n': final_sigma2_over_n,
            'n_samples': n_samples,
            'execution_time': state_time,
            'cache_stats': final_cache_stats,
            'mean_history': mean_history,
            'variance_history': variance_history,
            'sigma2_over_n_history': sigma2_over_n_history,
            'relative_error_history': relative_error_history,
            **convergence_details
        }
        if 'states_info' not in convergence_info:
            convergence_info['states_info'] = []
        convergence_info['states_info'].append(state_info)

    return final_mean


def _convergence_sampling(
    commutators: List[Tuple[Hashable | Set]],
    fragments: Dict[Hashable, Fragment],
    states: Sequence[AbstractState],
    timestep: float,
    sampling_method: str,
    convergence_tolerance: float = 1e-6,
    convergence_window: int = 10,
    min_convergence_checks: int = 3,
    min_sample_size: int = 10,
    max_sample_size: int = 10000,
    random_seed: Optional[int] = None,
    gridpoints: int = 10,
    convergence_info: Optional[Dict] = None,
) -> List[float]:
    r"""
    Convergence-based sampling that stops when the mean has converged to a certain precision.

    Uses the criterion: :math:`|\mu_n - \mu_{n-w}| < \varepsilon \cdot |\mu_n|` where:

    - :math:`\mu_n` is the current mean estimate at sample n
    - :math:`\mu_{n-w}` is the mean estimate w samples ago (where w is the convergence window)
    - :math:`\varepsilon` is the convergence tolerance (relative to current mean)

    Args:
        commutators: List of all available commutators
        fragments: Dictionary mapping fragment keys to Fragment objects
        states: States to compute expectation values for
        timestep: Time step for simulation
        sampling_method: "random" or "importance"
        convergence_tolerance: Relative tolerance for mean convergence (default: 1e-6)
        convergence_window: Number of samples to look back for convergence check (default: 10)
        min_convergence_checks: Minimum number of convergence checks that must pass (default: 3)
        min_sample_size: Minimum number of samples before checking convergence
        max_sample_size: Maximum number of samples (stopping criterion)
        random_seed: Random seed for reproducibility
        gridpoints: Number of gridpoints for norm calculations

    Returns:
        List of expectation values for each state
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Setup probabilities for sampling
    prob_start_time = time.time()
    probabilities = _setup_importance_probabilities(
        commutators, fragments, timestep, gridpoints, sampling_method
    )
    prob_time = time.time() - prob_start_time

    # Record probability calculation time
    if convergence_info:
        convergence_info['probability_calculation_time'] = prob_time

    # Process each state with convergence-based sampling
    expectations = []

    for state_idx, state in enumerate(states):
        expectation = _convergence_sample_single_state(
            state=state,
            state_idx=state_idx,
            commutators=commutators,
            fragments=fragments,
            probabilities=probabilities,
            sampling_method=sampling_method,
            convergence_tolerance=convergence_tolerance,
            convergence_window=convergence_window,
            min_convergence_checks=min_convergence_checks,
            min_sample_size=min_sample_size,
            max_sample_size=max_sample_size,
            convergence_info=convergence_info,
        )
        expectations.append(expectation)

    # Record total samples information for convergence sampling
    if convergence_info:
        total_samples = sum(state_info['n_samples'] for state_info in convergence_info['states_info'])
        convergence_info['global']['sampled_commutators'] = total_samples

    return expectations


def _convergence_sample_single_state(
    state: AbstractState,
    state_idx: int,
    commutators: List[Tuple[Hashable | Set]],
    fragments: Dict[Hashable, Fragment],
    probabilities: Optional[np.ndarray],
    sampling_method: str,
    convergence_tolerance: float,
    convergence_window: int,
    min_convergence_checks: int,
    min_sample_size: int,
    max_sample_size: int,
    convergence_info: Optional[Dict] = None,
) -> float:
    """
    Perform convergence-based sampling for a single state until mean converges.

    Args:
        state: The quantum state to compute expectation value for
        state_idx: Index of the state (for logging)
        commutators: List of all commutators
        fragments: Dictionary mapping fragment keys to Fragment objects
        probabilities: Pre-calculated probabilities for importance sampling (None for random)
        sampling_method: "random" or "importance"
        convergence_tolerance: Relative tolerance for mean convergence
        convergence_window: Number of samples to look back for convergence check
        min_convergence_checks: Minimum number of convergence checks that must pass
        min_sample_size: Minimum number of samples before checking convergence
        max_sample_size: Maximum number of samples (stopping criterion)

    Returns:
        float: Final expectation value estimate
    """
    state_start_time = time.time()
    print(f"\n=== Processing State {state_idx + 1} (Convergence Sampling) ===")

    # Initialize cache for this state
    cache = _CommutatorCache()

    # Initialize statistics for this state
    n_samples = 0
    sum_values = 0.0
    sum_squared = 0.0  # Add for variance calculation
    last_report_time = time.time()
    report_interval = 10  # Report every 10 samples initially

    # Keep track of convergence histories
    mean_history = []
    variance_history = []
    sigma2_over_n_history = []
    relative_error_history = []
    convergence_checks_passed = 0

    # Start sampling
    while n_samples < max_sample_size:
        sample_start_time = time.time()

        # Sample one commutator and compute its contribution
        if sampling_method == "random":
            idx = np.random.choice(len(commutators))
            weight = len(commutators)  # Scaling factor for uniform sampling
            commutator = commutators[idx]
        else:  # importance sampling
            idx = np.random.choice(len(commutators), p=probabilities)
            weight = 1.0 / probabilities[idx]  # Importance weight
            commutator = commutators[idx]

        # Compute the contribution of this commutator using cache
        applied_state = _apply_commutator(commutator, fragments, state, cache, state_idx)
        if isinstance(applied_state, _AdditiveIdentity):
            contribution = 0.0
        else:
            contribution = weight * state.dot(applied_state)

        # Update running statistics
        n_samples += 1
        sum_values += contribution
        sum_squared += contribution * contribution
        current_mean = sum_values / n_samples
        mean_history.append(current_mean)

        # Calculate additional convergence metrics
        current_variance = (sum_squared - n_samples * current_mean * current_mean) / (n_samples - 1) if n_samples > 1 else 0.0
        current_variance = abs(current_variance)  # Handle complex variance
        current_sigma2_over_n = current_variance / n_samples

        # Store convergence histories
        variance_history.append(current_variance)
        sigma2_over_n_history.append(current_sigma2_over_n)

        # Calculate and store relative error history
        if abs(current_mean) > 1e-12:
            current_relative_error = np.sqrt(current_variance) / abs(current_mean)
            relative_error_history.append(current_relative_error)
        else:
            relative_error_history.append(float('inf'))  # Handle near-zero mean

        sample_time = time.time() - sample_start_time

        # Progress reporting
        current_time = time.time()
        if (n_samples % report_interval == 0) or (current_time - last_report_time > 5.0):
            # Add cache statistics to progress reporting
            cache_stats = cache.get_stats()
            print(f"  Sample {n_samples:4d}: "
                  f"mean={current_mean:8.3e}, "
                  f"cache_hit_rate={cache_stats['hit_rate']:.1%}, "
                  f"time={sample_time:.3f}s")

            last_report_time = current_time

            # Adaptive reporting interval: start with 10, then 50, then 100
            if n_samples >= 50 and report_interval == 10:
                report_interval = 50
            elif n_samples >= 200 and report_interval == 50:
                report_interval = 100

        # Check convergence criterion after minimum samples
        if n_samples >= min_sample_size and len(mean_history) > convergence_window:
            # Get mean from convergence_window samples ago
            previous_mean = mean_history[-convergence_window-1]

            # Calculate relative change in mean
            if abs(current_mean) > 1e-12:
                relative_change = abs(current_mean - previous_mean) / abs(current_mean)
            else:
                # For very small means, use absolute change
                relative_change = abs(current_mean - previous_mean)

            # Check if convergence criterion is satisfied
            if relative_change < convergence_tolerance:
                convergence_checks_passed += 1
            else:
                convergence_checks_passed = 0  # Reset counter if criterion not met

            # Stop if we've had enough consecutive convergence checks
            if convergence_checks_passed >= min_convergence_checks:
                convergence_time = time.time() - state_start_time
                cache_stats = cache.get_stats()
                print(f"  ✓ CONVERGED after {n_samples} samples in {convergence_time:.2f}s")
                print(f"    Convergence checks passed: {convergence_checks_passed}")
                print(f"    Final mean: {current_mean:.6e}")
                print(f"    Previous mean: {previous_mean:.6e}")
                print(f"    Relative change: {relative_change:.6e} (tolerance: {convergence_tolerance:.6e})")
                print(f"    Cache performance: {cache_stats['hits']} hits, {cache_stats['misses']} misses ({cache_stats['hit_rate']:.1%} hit rate)")
                break

    final_mean = sum_values / n_samples if n_samples > 0 else 0.0
    state_time = time.time() - state_start_time

    if n_samples >= max_sample_size:
        cache_stats = cache.get_stats()
        print(f"  ⚠ STOPPED at maximum samples ({max_sample_size}) for state {state_idx + 1}")
        print(f"    Time spent: {state_time:.2f}s")
        print(f"    Final mean: {final_mean:.6e}")
        if len(mean_history) > convergence_window:
            previous_mean = mean_history[-convergence_window-1]
            if abs(final_mean) > 1e-12:
                rel_change = abs(final_mean - previous_mean) / abs(final_mean)
            else:
                rel_change = abs(final_mean - previous_mean)
            print(f"    Final relative change: {rel_change:.6e} (tolerance: {convergence_tolerance:.6e})")
        print(f"    Cache performance: {cache_stats['hits']} hits, {cache_stats['misses']} misses ({cache_stats['hit_rate']:.1%} hit rate)")

    print(f"  State {state_idx + 1} completed in {state_time:.2f}s")

    # Add convergence information if requested
    if convergence_info:
        # Calculate final statistics
        final_variance = (sum_squared - n_samples * final_mean * final_mean) / (n_samples - 1) if n_samples > 1 else 0.0
        final_variance = abs(final_variance)  # Handle complex variance
        final_sigma2_over_n = final_variance / n_samples if n_samples > 0 else 0.0
        final_cache_stats = cache.get_stats()

        # Determine convergence method and add relevant info
        convergence_details = {
            'sampling_method': sampling_method,
            'convergence_tolerance': convergence_tolerance,
            'convergence_window': convergence_window,
            'min_convergence_checks': min_convergence_checks,
            'min_sample_size': min_sample_size,
            'max_sample_size': max_sample_size,
        }

        # Add final statistics based on convergence outcome
        if n_samples < max_sample_size:
            convergence_details['convergence_status'] = 'converged'
            convergence_details['convergence_checks_passed'] = convergence_checks_passed
            if len(mean_history) > convergence_window:
                previous_mean = mean_history[-convergence_window-1]
                if abs(final_mean) > 1e-12:
                    final_relative_change = abs(final_mean - previous_mean) / abs(final_mean)
                else:
                    final_relative_change = abs(final_mean - previous_mean)
                convergence_details['final_relative_change'] = final_relative_change
        else:
            convergence_details['convergence_status'] = 'max_samples_reached'

        # Add state info using dictionary approach
        state_info = {
            'mean': final_mean,
            'variance': final_variance,
            'sigma2_over_n': final_sigma2_over_n,
            'n_samples': n_samples,
            'execution_time': state_time,
            'cache_stats': final_cache_stats,
            'mean_history': mean_history,
            'variance_history': variance_history,
            'sigma2_over_n_history': sigma2_over_n_history,
            'relative_error_history': relative_error_history,
            **convergence_details
        }
        if 'states_info' not in convergence_info:
            convergence_info['states_info'] = []
        convergence_info['states_info'].append(state_info)

    return final_mean


def _fixed_sampling(
    commutators: List[Tuple[Hashable | Set]],
    fragments: Dict[Hashable, Fragment],
    states: Sequence[AbstractState],
    timestep: float,
    sample_size: int,
    sampling_method: str,
    random_seed: Optional[int] = None,
    gridpoints: int = 10,
    num_workers: int = 1,
    backend: str = "serial",
    parallel_mode: str = "state",
    convergence_info: Optional[Dict] = None,
) -> List[float]:
    """
    Fixed-size sampling for perturbation error calculation.

    Samples a fixed number of commutators using either random or importance sampling,
    then computes expectation values for all states.

    Args:
        commutators: List of all available commutators
        fragments: Dictionary mapping fragment keys to Fragment objects
        states: States to compute expectation values for
        timestep: Time step for simulation
        sample_size: Number of commutators to sample
        sampling_method: "random" or "importance"
        random_seed: Random seed for reproducibility
        gridpoints: Number of gridpoints for norm calculations
        num_workers: Number of workers for parallel processing
        backend: Backend for parallel processing
        parallel_mode: Mode of parallelization ("state" or "commutator")
        convergence_info: Optional convergence info container to populate

    Returns:
        List of expectation values for each state
    """
    print(f"Using fixed-size sampling with {sample_size} commutators out of {len(commutators)} total")

    # Pre-calculate sampling probabilities and sample commutators
    if sampling_method == "random":
        if random_seed is not None:
            np.random.seed(random_seed)

        # Sample uniformly at random with replacement
        effective_sample_size = min(sample_size, len(commutators))
        sampled_indices = np.random.choice(len(commutators), size=effective_sample_size, replace=True)
        sampled_commutators = [commutators[idx] for idx in sampled_indices]

        scaling_factor = len(commutators) / len(sampled_commutators) if sampled_commutators else 1.0
        commutator_weights = [scaling_factor] * len(sampled_commutators)

    elif sampling_method == "importance":
        print("=== Using Importance Sampling ===")
        # Pre-calculate probabilities once
        start_time = time.time()
        probabilities = _setup_importance_probabilities(
            commutators, fragments, timestep, gridpoints, sampling_method
        )
        prob_time = time.time() - start_time
        print(f"Probability calculation completed in {prob_time:.2f} seconds")

        # Record probability calculation time
        if convergence_info:
            convergence_info['probability_calculation_time'] = prob_time

        sampled_commutators, commutator_weights = _sample_importance_commutators(
            commutators, probabilities, sample_size, random_seed
        )
    else:
        raise ValueError("sampling_method must be 'random' or 'importance'")

    # Record sampling information
    if convergence_info:
        convergence_info['global']['sampled_commutators'] = len(sampled_commutators)

    # Compute expectation values using the sampled commutators
    return _compute_expectation_values(
        sampled_commutators, commutator_weights, fragments, states,
        num_workers, backend, parallel_mode, convergence_info
    )


def _sample_importance_commutators(
    commutators: List[Tuple[Hashable | Set]],
    probabilities: np.ndarray,
    sample_size: int,
    random_seed: Optional[int] = None
) -> Tuple[List[Tuple[Hashable | Set]], List[float]]:
    """
    Sample commutators using importance sampling with pre-calculated probabilities.

    Args:
        commutators: List of commutator tuples to sample from
        probabilities: Pre-calculated normalized probabilities for each commutator
        sample_size: Number of commutators to sample
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (sampled_commutators, weights)
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Sample with replacement using the calculated probabilities
    sampled_indices = np.random.choice(
        len(commutators),
        size=sample_size,
        replace=True,
        p=probabilities
    )

    # Create lists of sampled commutators and their importance weights
    sampled_commutators = []
    weights = []

    for idx in sampled_indices:
        commutator = commutators[idx]
        # Calculate importance sampling weight
        if probabilities[idx] > 0:
            weight = 1.0 / (probabilities[idx] * sample_size)
            sampled_commutators.append(commutator)
            weights.append(weight)

    return sampled_commutators, weights


def _compute_expectation_values(
    commutators: List[Tuple[Hashable | Set]],
    weights: List[float],
    fragments: Dict[Hashable, Fragment],
    states: Sequence[AbstractState],
    num_workers: int = 1,
    backend: str = "serial",
    parallel_mode: str = "state",
    convergence_info: Optional[Dict] = None,
) -> List[float]:
    """
    Compute expectation values for all states using the sampled commutators.

    Args:
        commutators: List of sampled commutators
        weights: List of weights for each commutator
        fragments: Dictionary mapping fragment keys to Fragment objects
        states: States to compute expectation values for
        num_workers: Number of workers for parallel processing
        backend: Backend for parallel processing
        parallel_mode: Mode of parallelization ("state" or "commutator")
        convergence_info: Optional convergence info container to populate

    Returns:
        List of expectation values for each state
    """
    if backend == "serial":
        assert num_workers == 1, "num_workers must be set to 1 for serial execution."
        expectations = []

        # Create a cache for each state to store computed commutator applications
        for state_idx, state in enumerate(states):
            state_start_time = time.time()
            cache = _CommutatorCache()
            new_state = _AdditiveIdentity()

            for weight, commutator in tqdm(zip(weights, commutators), desc=f"Processing commutators for state {state_idx+1}", total=len(commutators)):
                weighted_state = weight * _apply_commutator(commutator, fragments, state, cache, state_idx)
                new_state += weighted_state

            # Handle case where new_state is still _AdditiveIdentity (no commutators applied)
            if isinstance(new_state, _AdditiveIdentity):
                expectation = 0.0
            else:
                expectation = state.dot(new_state)

            expectations.append(expectation)

            # Add convergence info for fixed sampling if requested
            if convergence_info:
                state_time = time.time() - state_start_time
                cache_stats = cache.get_stats()
                state_info = {
                    'mean': expectation,
                    'variance': 0.0,  # No variance in fixed sampling without repeated runs
                    'sigma2_over_n': 0.0,  # No sampling uncertainty for fixed sampling
                    'n_samples': len(commutators),
                    'execution_time': state_time,
                    'cache_stats': cache_stats,
                    'sampling_method': 'fixed',
                }
                if 'states_info' not in convergence_info:
                    convergence_info['states_info'] = []
                convergence_info['states_info'].append(state_info)

            # Print cache statistics for this state (only if not tracking convergence info)
            if not convergence_info:
                cache_stats = cache.get_stats()
                print(f"State {state_idx+1} cache performance: {cache_stats['hits']} hits, {cache_stats['misses']} misses ({cache_stats['hit_rate']:.1%} hit rate)")

        return expectations

    if parallel_mode == "state":
        executor = concurrency.backends.get_executor(backend)
        with executor(max_workers=num_workers) as ex:
            expectations = ex.starmap(
                _compute_state_expectation,
                [(commutators, weights, fragments, state) for state in states],
            )

        return expectations

    if parallel_mode == "commutator":
        executor = concurrency.backends.get_executor(backend)
        expectations = []
        for state in states:
            with executor(max_workers=num_workers) as ex:
                applied_commutators = ex.starmap(
                    _apply_weighted_commutator,
                    [(commutator, weight, fragments, state) for commutator, weight in zip(commutators, weights)],
                )

            new_state = _AdditiveIdentity()
            for applied_state in applied_commutators:
                new_state += applied_state

            # Handle case where new_state is still _AdditiveIdentity (no commutators applied)
            if isinstance(new_state, _AdditiveIdentity):
                expectations.append(0.0)
            else:
                expectations.append(state.dot(new_state))

        return expectations

    raise ValueError("Invalid parallel mode. Choose 'state' or 'commutator'.")


def _compute_state_expectation(commutators, weights, fragments, state):
    """Helper function to compute state expectation for parallel processing.

    This replaces the lambda function to make it pickleable for MPI.
    Note: Caching is disabled in parallel mode for thread safety.
    """
    new_state = sum((weight * _apply_commutator(commutator, fragments, state)
                    for commutator, weight in zip(commutators, weights)),
                   start=_AdditiveIdentity())

    # Handle case where new_state is still _AdditiveIdentity (no commutators applied)
    if isinstance(new_state, _AdditiveIdentity):
        result = 0.0
    else:
        result = state.dot(new_state)

    return result


def _apply_weighted_commutator(commutator, weight, fragments, state):
    """Helper function to apply weighted commutator for parallel processing.

    This replaces the lambda function to make it pickleable for MPI.
    Note: Caching is disabled in parallel mode for thread safety.
    """
    return weight * _apply_commutator(commutator, fragments, state)
