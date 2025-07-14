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
from collections import Counter, defaultdict
from typing import List, Sequence, Tuple, Hashable, Dict, Set, Optional

import numpy as np
from tqdm import tqdm

from pennylane import concurrency
from pennylane.labs.trotter_error import AbstractState, Fragment
from pennylane.labs.trotter_error.abstract import nested_commutator
from pennylane.labs.trotter_error.product_formulas.bch import bch_expansion
from pennylane.labs.trotter_error.product_formulas.product_formula import ProductFormula


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


# pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-branches
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
    target_error: float = 0.01,
    min_sample_size: int = 10,
    max_sample_size: int = 10000,
) -> List[float]:
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
            Available options : "mp_pool", "cf_procpool", "cf_threadpool", "serial", "mpi4py_pool", "mpi4py_comm". Default value is set to "serial".
        parallel_mode (str): the mode of parallelization to use.
            Options are "state" or "commutator".
            "state" parallelizes the computation of expectation values per state,
            while "commutator" parallelizes the application of commutators to each state.
            Default value is set to "state".
        sample_size (Optional[int]): Number of commutators to sample for fixed-size sampling.
            If None (default), uses all commutators exactly without sampling for maximum accuracy.
            Only when a specific number is provided will fixed-size sampling be applied.
            This parameter is ignored if adaptive_sampling is True.
        sampling_method (str): Sampling strategy to use. Options are "random" for uniform random sampling
            or "importance" for importance sampling based on commutator magnitudes.
            Only used when sample_size is specified or adaptive_sampling is True.
        random_seed (Optional[int]): Random seed for reproducibility in sampling methods.
        adaptive_sampling (bool): If True, uses adaptive sampling that dynamically determines
            the optimal sample size based on variance convergence criteria. When enabled,
            the sample_size parameter is ignored since the algorithm determines the required
            number of samples automatically. Note: adaptive sampling is only compatible with
            backend='serial' and num_workers=1.
        confidence_level (float): Confidence level for adaptive sampling (e.g., 0.95 for 95% confidence).
            Only used when adaptive_sampling is True.
        target_error (float): Target relative error for adaptive sampling (epsilon in :math:`N \geq z^2\sigma^2/\varepsilon^2`).
            Only used when adaptive_sampling is True.
        min_sample_size (int): Minimum sample size for adaptive sampling.
            Only used when adaptive_sampling is True.
        max_sample_size (int): Maximum sample size for adaptive sampling to prevent infinite loops.
            Only used when adaptive_sampling is True.

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

    start_time = time.time()
    commutators = _group_sums(bch_expansion(product_formula(1j * timestep), order))
    print(f"BCH expansion time: {time.time() - start_time:.4f} seconds")

    # Handle adaptive sampling first (it has priority)
    if adaptive_sampling:
        if sample_size is not None:
            print("Warning: sample_size is ignored when adaptive_sampling=True")

        print("Using adaptive sampling (dynamic sample size determination)")

        # Adaptive sampling is only compatible with serial execution
        if backend != "serial":
            raise ValueError("Adaptive sampling is only compatible with backend='serial'")
        if num_workers != 1:
            raise ValueError("Adaptive sampling requires num_workers=1")

        # Get gridpoints from the first state (all states should have the same gridpoints)
        gridpoints = getattr(states[0], 'gridpoints', 10) if states else 10
        return _adaptive_sampling(
            commutators=commutators,
            fragments=fragments,
            states=states,
            timestep=timestep,
            sampling_method=sampling_method,
            confidence_level=confidence_level,
            target_error=target_error,
            min_sample_size=min_sample_size,
            max_sample_size=max_sample_size,
            random_seed=random_seed,
            gridpoints=gridpoints,
        )

    # Handle fixed-size sampling if explicitly requested
    if sample_size is not None:
        print(f"Using fixed-size sampling with {sample_size} commutators out of {len(commutators)} total")
        # Get gridpoints from the first state (all states should have the same gridpoints)
        gridpoints = getattr(states[0], 'gridpoints', 10) if states else 10
        commutators, commutator_weights = _sample_commutators(
            commutators, fragments, timestep, sample_size, sampling_method, random_seed, gridpoints
        )
    else:
        print(f"Using all {len(commutators)} commutators exactly (no sampling)")
        commutator_weights = [1.0] * len(commutators)

    if backend == "serial":
        assert num_workers == 1, "num_workers must be set to 1 for serial execution."
        expectations = []
        for state in states:
            new_state = _AdditiveIdentity()
            for weight, commutator in tqdm(zip(commutator_weights, commutators), desc="Processing commutators", total=len(commutators)):
                weighted_state = weight * _apply_commutator(commutator, fragments, state)
                new_state += weighted_state

            # Handle case where new_state is still _AdditiveIdentity (no commutators applied)
            if isinstance(new_state, _AdditiveIdentity):
                expectations.append(0.0)
            else:
                expectations.append(state.dot(new_state))

        return expectations

    if parallel_mode == "state":
        executor = concurrency.backends.get_executor(backend)
        with executor(max_workers=num_workers) as ex:
            expectations = ex.starmap(
                _compute_state_expectation,
                [(commutators, commutator_weights, fragments, state) for state in states],
            )

        return expectations

    if parallel_mode == "commutator":
        executor = concurrency.backends.get_executor(backend)
        expectations = []
        for state in states:
            with executor(max_workers=num_workers) as ex:
                applied_commutators = ex.starmap(
                    _apply_weighted_commutator,
                    [(commutator, weight, fragments, state) for commutator, weight in zip(commutators, commutator_weights)],
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


def _get_expval_state(commutators, fragments, state: AbstractState) -> float:
    """
    Returns the expectation value of a state with respect to the operator obtained
    by substituting fragments into commutators.

    Args:
        commutators: List of commutator tuples
        fragments: Dictionary mapping fragment keys to Fragment objects
        state: The quantum state to compute expectation value for

    Returns:
        float: The expectation value
    """

    new_state = _AdditiveIdentity()
    for commutator in commutators:
        new_state += _apply_commutator(commutator, fragments, state)

    # Handle case where new_state is still _AdditiveIdentity (no commutators applied)
    if isinstance(new_state, _AdditiveIdentity):
        return 0.0
    else:
        return state.dot(new_state)


def _apply_commutator(
    commutator: Tuple[Hashable], fragments: Dict[Hashable, Fragment], state: AbstractState
) -> AbstractState:
    """
    Returns the state obtained from applying a commutator to a state.

    Args:
        commutator: Tuple representing a commutator structure
        fragments: Dictionary mapping fragment keys to Fragment objects
        state: The quantum state to apply the commutator to

    Returns:
        AbstractState: The state after applying the commutator
    """

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


def random_sample_commutators(
    commutators: List[Tuple[Hashable | Set]],
    sample_size: int,
    random_seed: Optional[int] = None
) -> List[Tuple[Hashable | Set]]:
    """
    Randomly sample commutators uniformly without replacement.

    Args:
        commutators: List of commutator tuples to sample from
        sample_size: Number of commutators to sample
        random_seed: Random seed for reproducibility

    Returns:
        List of sampled commutator tuples
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    if len(commutators) <= sample_size:
        return commutators

    sampled_indices = np.random.choice(
        len(commutators),
        size=sample_size,
        replace=False
    )

    return [commutators[i] for i in sampled_indices]


def importance_sample_commutators(
    commutators: List[Tuple[Hashable | Set]],
    fragments: Dict[Hashable, Fragment],
    timestep: float,
    sample_size: int,
    random_seed: Optional[int] = None,
    gridpoints: int = 10
) -> List[Tuple[Tuple[Hashable | Set], float]]:
    r"""
    Importance sample commutators based on their magnitude and structure.

    Importance sampling is a variance reduction technique that samples commutators according to
    a probability distribution proportional to their expected contribution to the final result.

    For a commutator :math:`C_k` of order :math:`k`, the unnormalized probability is given by:

    .. math::
        p_k \propto 2^{k-1} \cdot \tau^k \cdot \prod_{i} \|\hat{H}_i\|

    where :math:`\tau` is the timestep, :math:`\hat{H}_i` are the fragment operators in the commutator,
    and :math:`\|\hat{H}_i\|` are their operator norms.

    The final sampling probability is :math:`P_k = p_k / \sum_j p_j`, and each sampled commutator
    receives an importance weight :math:`w_k = 1/(P_k \cdot N_{\text{sample}})` to ensure the estimator
    remains unbiased:

    .. math::
        \langle \hat{\varepsilon} \rangle \approx \frac{1}{N_{\text{sample}}} \sum_{k \in \text{sampled}} \frac{\langle \psi | C_k | \psi \rangle}{P_k}

    This approach typically provides better convergence compared to uniform random sampling
    when estimating perturbation theory errors.

    Args:
        commutators: List of commutator tuples where each tuple represents a commutator
        fragments: Dictionary mapping fragment keys to Fragment objects
        timestep: Time step for the simulation
        sample_size: Number of commutators to sample
        random_seed: Random seed for reproducibility
        gridpoints: Number of gridpoints for norm calculation

    Returns:
        List of tuples (commutator, weight) where weight is the importance sampling weight
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    if not commutators:
        return []

    # Calculate unnormalized probabilities for each commutator
    probabilities = []

    for commutator in tqdm(commutators, desc="Calculating commutator probabilities"):
        prob = _calculate_commutator_probability(commutator, fragments, timestep, gridpoints)
        probabilities.append(prob)

    # Normalize probabilities
    probabilities = np.array(probabilities)
    total_prob_sum = np.sum(probabilities)

    if total_prob_sum == 0:
        # Handle the case where all probabilities are zero
        # Fallback to uniform sampling
        probabilities = np.ones(len(commutators)) / len(commutators)
    else:
        probabilities = probabilities / total_prob_sum

    # Sample with replacement using the calculated probabilities
    sampled_indices = np.random.choice(
        len(commutators),
        size=sample_size,
        replace=True,
        p=probabilities
    )

    # Create the list of sampled commutators with importance weights
    sampled_commutators = []

    for idx in sampled_indices:
        commutator = commutators[idx]

        # Calculate importance sampling weight
        if probabilities[idx] > 0:
            weight = 1.0 / (probabilities[idx] * sample_size)
            sampled_commutators.append((commutator, weight))

    return sampled_commutators


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
        target_error: Target relative error (epsilon)
        min_sample_size: Minimum number of samples
        max_sample_size: Maximum number of samples
        random_seed: Random seed for reproducibility

    Returns:
        List of expectation values for each state
    """
    print("\n=== Adaptive Sampling Configuration ===")
    print(f"Total commutators available: {len(commutators)}")
    print(f"Sampling method: {sampling_method}")
    print(f"Confidence level: {confidence_level}")
    print(f"Target relative error: {target_error}")
    print(f"Min sample size: {min_sample_size}")
    print(f"Max sample size: {max_sample_size}")
    print(f"Number of states: {len(states)}")

    if random_seed is not None:
        np.random.seed(random_seed)
        print(f"Random seed: {random_seed}")

    # Calculate z-score for given confidence level (avoid scipy dependency)
    # For 95% confidence: z ≈ 1.96, for 99%: z ≈ 2.576
    z_score_dict = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    z_score = z_score_dict.get(confidence_level, 1.96)  # Default to 95%
    print(f"Z-score for {confidence_level} confidence: {z_score}")

    # Pre-calculate sampling probabilities for importance sampling (only once!)
    probabilities = None
    if sampling_method == "importance":
        print("\n=== Pre-calculating Importance Sampling Probabilities ===")
        prob_start_time = time.time()
        probabilities = []

        for commutator in tqdm(commutators, desc="Calculating probabilities", unit="commutator"):
            prob = _calculate_commutator_probability(commutator, fragments, timestep, gridpoints)
            probabilities.append(prob)

        probabilities = np.array(probabilities)
        total_prob_sum = np.sum(probabilities)

        if total_prob_sum == 0:
            print("Warning: All probabilities are zero, falling back to uniform distribution")
            probabilities = np.ones(len(commutators)) / len(commutators)
        else:
            probabilities = probabilities / total_prob_sum

        prob_time = time.time() - prob_start_time
        print(f"Probability calculation completed in {prob_time:.2f} seconds")
        print(f"Probability stats: min={np.min(probabilities):.2e}, max={np.max(probabilities):.2e}, mean={np.mean(probabilities):.2e}")
    else:
        print("\n=== Using Random Sampling ===")

    # Initialize result storage for each state
    expectations = []
    total_start_time = time.time()

    for state_idx, state in enumerate(states):
        state_start_time = time.time()
        print(f"\n=== Processing State {state_idx + 1}/{len(states)} ===")

        # Initialize statistics for this state
        n_samples = 0
        sum_values = 0.0
        sum_squared = 0.0
        last_report_time = time.time()
        report_interval = 10  # Report every 10 samples initially

        # Start sampling
        while n_samples < max_sample_size:
            sample_start_time = time.time()

            # Sample one commutator
            if sampling_method == "random":
                idx = np.random.choice(len(commutators))
                weight = len(commutators)  # Scaling factor for uniform sampling
                commutator = commutators[idx]
            else:  # importance sampling
                idx = np.random.choice(len(commutators), p=probabilities)
                weight = 1.0 / probabilities[idx]  # Importance weight
                commutator = commutators[idx]

            # Apply commutator and get contribution
            applied_state = _apply_commutator(commutator, fragments, state)
            
            # Handle case where applied_state is _AdditiveIdentity
            if isinstance(applied_state, _AdditiveIdentity):
                contribution = 0.0
            else:
                contribution = weight * state.dot(applied_state)

            # Update running statistics
            n_samples += 1
            sum_values += contribution
            sum_squared += contribution * contribution

            sample_time = time.time() - sample_start_time

            # Progress reporting
            current_time = time.time()
            if (n_samples % report_interval == 0) or (current_time - last_report_time > 5.0):  # Report every interval or every 5 seconds
                mean = sum_values / n_samples
                variance = (sum_squared - n_samples * mean * mean) / (n_samples - 1) if n_samples > 1 else 0
                # Handle complex variance by taking the absolute value
                variance_abs = abs(variance)
                std_dev = np.sqrt(variance_abs)

                print(f"  Sample {n_samples:4d}: "
                      f"mean={mean:8.3e}, "
                      f"std={std_dev:8.3e}, "
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
                # Handle complex variance by taking the absolute value
                variance_abs = abs(variance)
                std_dev = np.sqrt(variance_abs)

                # Avoid division by zero
                if abs(mean) > 1e-12:
                    relative_error = std_dev / abs(mean)
                    required_samples = (z_score * relative_error / target_error) ** 2

                    if n_samples >= required_samples:
                        convergence_time = time.time() - state_start_time
                        print(f"  ✓ CONVERGED after {n_samples} samples in {convergence_time:.2f}s")
                        print(f"    Required samples: {required_samples:.1f}")
                        print(f"    Final mean: {mean:.6e}")
                        print(f"    Final std: {std_dev:.6e}")
                        print(f"    Relative error: {relative_error:.6e} (target: {target_error:.6e})")
                        break
                else:
                    # If mean is very close to zero, use absolute error criterion
                    if std_dev < target_error:
                        convergence_time = time.time() - state_start_time
                        print(f"  ✓ CONVERGED after {n_samples} samples in {convergence_time:.2f}s (mean ≈ 0)")
                        print(f"    Final std: {std_dev:.6e} (target: {target_error:.6e})")
                        break

        final_mean = sum_values / n_samples if n_samples > 0 else 0.0
        expectations.append(final_mean)

        state_time = time.time() - state_start_time
        if n_samples >= max_sample_size:
            print(f"  ⚠ STOPPED at maximum samples ({max_sample_size}) for state {state_idx + 1}")
            print(f"    Time spent: {state_time:.2f}s")
            print(f"    Final mean: {final_mean:.6e}")
            variance = (sum_squared - n_samples * final_mean * final_mean) / (n_samples - 1) if n_samples > 1 else 0
            # Handle complex variance by taking the absolute value
            variance_abs = abs(variance)
            std_dev = np.sqrt(variance_abs)
            print(f"    Final std: {std_dev:.6e}")
            if abs(final_mean) > 1e-12:
                rel_err = std_dev / abs(final_mean)
                print(f"    Achieved relative error: {rel_err:.6e} (target: {target_error:.6e})")

        print(f"  State {state_idx + 1} completed in {state_time:.2f}s")

    total_time = time.time() - total_start_time
    print("\n=== Adaptive Sampling Summary ===")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average time per state: {total_time/len(states):.2f}s")
    print(f"Final expectation values: {[f'{exp:.6e}' for exp in expectations]}")

    return expectations


def _sample_commutators(
    commutators: List[Tuple[Hashable | Set]],
    fragments: Dict[Hashable, Fragment],
    timestep: float,
    sample_size: int,
    sampling_method: str,
    random_seed: Optional[int] = None,
    gridpoints: int = 10
) -> Tuple[List[Tuple[Hashable | Set]], List[float]]:
    """
    Sample commutators and return them with their corresponding weights.

    Args:
        commutators: List of commutator tuples to sample from
        fragments: Dictionary mapping fragment keys to Fragment objects
        timestep: Time step for the simulation
        sample_size: Number of commutators to sample
        sampling_method: "random" for uniform random sampling, "importance" for importance sampling
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (sampled_commutators, weights)
    """
    if sampling_method == "random":
        sampled_commutators = random_sample_commutators(commutators, sample_size, random_seed)
        scaling_factor = len(commutators) / len(sampled_commutators) if sampled_commutators else 1.0
        weights = [scaling_factor] * len(sampled_commutators)
        return sampled_commutators, weights
    if sampling_method == "importance":
        sampled_data = importance_sample_commutators(
            commutators, fragments, timestep, sample_size, random_seed, gridpoints
        )
        sampled_commutators = [data[0] for data in sampled_data]
        weights = [data[1] for data in sampled_data]
        return sampled_commutators, weights

    raise ValueError("sampling_method must be 'random' or 'importance'")


def _compute_state_expectation(commutators, weights, fragments, state):
    """Helper function to compute state expectation for parallel processing.
    
    This replaces the lambda function to make it pickleable for MPI.
    """
    new_state = sum((weight * _apply_commutator(commutator, fragments, state)
                    for commutator, weight in zip(commutators, weights)),
                   start=_AdditiveIdentity())
    
    # Handle case where new_state is still _AdditiveIdentity (no commutators applied)
    if isinstance(new_state, _AdditiveIdentity):
        return 0.0
    else:
        return state.dot(new_state)


def _apply_weighted_commutator(commutator, weight, fragments, state):
    """Helper function to apply weighted commutator for parallel processing.
    
    This replaces the lambda function to make it pickleable for MPI.
    """
    return weight * _apply_commutator(commutator, fragments, state)
