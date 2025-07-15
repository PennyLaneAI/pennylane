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
from scipy.sparse import csr_matrix

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
            Available options : "serial", "mpi4py_pool", "mpi4py_comm". Default value is set to "serial".
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
    >>> errors = perturbation_error(pf, frags, [state1, state2], order=3, num_workers=2, backend="mpi4py_pool", parallel_mode="commutator")
    >>> print(errors)
    [0.9189251160920877j, 4.797716682426847j]
    """

    if not product_formula.fragments.issubset(fragments.keys()):
        raise ValueError("Fragments do not match product formula")

    commutators = _group_sums(bch_expansion(product_formula(1j * timestep), order))

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
            gridpoints=gridpoints or 10,  # Default gridpoints if not found
        )

    # Handle fixed-size sampling if explicitly requested
    if sample_size is not None:
        # Get gridpoints from the first state (all states should have the same gridpoints)
        if states and hasattr(states[0], 'gridpoints'):
            gridpoints = states[0].gridpoints
        else:
            gridpoints = None
        return _fixed_sampling(
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
        )

    # Use all commutators exactly (no sampling)
    commutator_weights = [1.0] * len(commutators)

    # Use the shared expectation value computation function
    return _compute_expectation_values(
        commutators, commutator_weights, fragments, states,
        num_workers, backend, parallel_mode
    )


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
        result = 0.0
    else:
        result = state.dot(new_state)

    return result


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
                # Handle tuple wrapping from _op_list structure
                if isinstance(frag, tuple) and len(frag) == 1:
                    frag_key = frag[0]
                else:
                    frag_key = frag
                frag = fragments[frag_key]

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
    if random_seed is not None:
        np.random.seed(random_seed)

    # Get z-score for confidence level
    z_score = _get_confidence_z_score(confidence_level)

    # Setup probabilities for sampling
    probabilities = _setup_importance_probabilities(
        commutators, fragments, timestep, gridpoints, sampling_method
    )

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
        )
        expectations.append(expectation)

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
    """
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
        target_error: Target relative error for convergence
        min_sample_size: Minimum number of samples before checking convergence
        max_sample_size: Maximum number of samples (stopping criterion)

    Returns:
        float: Final expectation value estimate
    """
    state_start_time = time.time()
    print(f"\n=== Processing State {state_idx + 1} ===")

    # Initialize statistics for this state
    n_samples = 0
    sum_values = 0.0
    sum_squared = 0.0
    last_report_time = time.time()
    report_interval = 10  # Report every 10 samples initially

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

        # Compute the contribution of this commutator
        applied_state = _apply_commutator(commutator, fragments, state)
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
        if (n_samples % report_interval == 0) or (current_time - last_report_time > 5.0):
            mean = sum_values / n_samples
            variance = (sum_squared - n_samples * mean * mean) / (n_samples - 1) if n_samples > 1 else 0
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
    state_time = time.time() - state_start_time

    if n_samples >= max_sample_size:
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

    print(f"  State {state_idx + 1} completed in {state_time:.2f}s")
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

        sampled_commutators, commutator_weights = _sample_importance_commutators(
            commutators, probabilities, sample_size, random_seed
        )
    else:
        raise ValueError("sampling_method must be 'random' or 'importance'")

    # Compute expectation values using the sampled commutators
    return _compute_expectation_values(
        sampled_commutators, commutator_weights, fragments, states,
        num_workers, backend, parallel_mode
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

    Returns:
        List of expectation values for each state
    """
    if backend == "serial":
        assert num_workers == 1, "num_workers must be set to 1 for serial execution."
        expectations = []
        for state in states:
            new_state = _AdditiveIdentity()
            for weight, commutator in tqdm(zip(weights, commutators), desc="Processing commutators", total=len(commutators)):
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
    """
    return weight * _apply_commutator(commutator, fragments, state)
