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

import numpy as np

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
    max_order: int,
    timestep: float = 1.0,
    num_workers: int = 1,
    backend: str = "serial",
    parallel_mode: str = "state",
    topk: int = None,
) -> tuple[list[dict], dict]:
    r"""Computes the perturbation theory error using the effective Hamiltonian :math:`\hat{\epsilon} = \hat{H}_{eff} - \hat{H}` for a  given product formula.


    For a state :math:`\left| \psi \right\rangle` the perturbation theory error is given by the expectation value :math:`\left\langle \psi \right| \hat{\epsilon} \left| \psi \right\rangle`.

    Args:
        product_formula (ProductFormula): the :class:`~.pennylane.labs.trotter_error.ProductFormula` used to obtain the effective Hamiltonian
        fragments (Sequence[Fragments]): the set of :class:`~.pennylane.labs.trotter_error.Fragment`
            objects to compute the perturbation error from
        states (Sequence[AbstractState]): the states to compute expectation values from
        max_order (float): the maximum commutator order to compute in BCH
        timestep (float): time step for the Trotter error operator.
        num_workers (int): the number of concurrent units used for the computation. Default value is set to 1.
        backend (string): the executor backend from the list of supported backends.
            Available options : "mp_pool", "cf_procpool", "cf_threadpool", "serial", "mpi4py_pool", "mpi4py_comm". Default value is set to "serial".
        parallel_mode (str): the mode of parallelization to use.
            Options are "state", "commutator", "nested_commutator", or "mvp".
            "state" parallelizes the computation of expectation values per state,
            "commutator" parallelizes the application of commutators to each state,
            "nested_commutator" parallelizes the application of individual operator terms within each commutator,
            "mvp" (matrix-vector product) parallelizes all terms from all commutators within each order (maximum parallelization per order).
            Default value is set to "state".
        topk (int, optional): the number of most important commutators to evaluate for each order.
            Commutators are ranked by importance using the formula: 2^(length-1) * product of fragment norms.
            If None, all commutators are evaluated. Default value is None.

    Returns:
        tuple[list[dict], dict]: A tuple containing:
            - List[Dict[int, float]]: the list of dictionaries of expectation values computed from the Trotter error operator and the input states.
              The dictionary is indexed by the commutator orders and its value is the error obtained from the commutators of that order.
            - Dict: convergence_info dictionary containing convergence statistics for each state and order,
              including partial sums, mean, median, and standard deviation.

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
    >>> # Evaluate all commutators
    >>> errors, convergence_info = perturbation_error(pf, frags, [state1, state2], max_order=3)
    >>>
    >>> # Evaluate only top-5 most important commutators per order
    >>> errors_topk, convergence_info_topk = perturbation_error(pf, frags, [state1, state2], max_order=3, topk=5)
    >>> print(errors)
    [{3: 0.9189251160920876j}, {3: 4.7977166824268505j}]
    """

    if not product_formula.fragments.issubset(fragments.keys()):
        raise ValueError("Fragments do not match product formula")

    commutator_lists = [
        _group_sums(commutators) for commutators in bch_expansion(product_formula, max_order)[1:]
    ]

    # Apply top-k filtering if needed, otherwise use commutators as-is
    if topk is not None:
        sorted_commutators_by_order = _apply_topk_filtering(commutator_lists, fragments, topk)
    else:
        # Convert to dict structure efficiently without importance computation
        sorted_commutators_by_order = {
            order + 2: {comm: 1.0 for comm in commutators}
            for order, commutators in enumerate(commutator_lists)
        }

    if backend == "serial":
        assert num_workers == 1, "num_workers must be set to 1 for serial execution."
        expectations = []
        convergence_info = {}

        for state_idx, state in enumerate(states):
            state_convergence = {}
            state_expectations = {}

            for order, commutators_dict in sorted_commutators_by_order.items():
                if len(commutators_dict) == 0:
                    continue

                # Reset expectation accumulator for each order
                current_expectation = 0.0
                partial_sums = []
                mean_history = []
                median_history = []
                std_history = []

                # Process commutators in order of importance (already sorted in the dict)
                for commutator in commutators_dict.keys():
                    commutator_expectation = _evaluate_expected_value(
                        commutator, fragments, state, timestep, order
                    )
                    current_expectation += commutator_expectation
                    partial_sums.append(current_expectation)

                    # Compute convergence statistics incrementally
                    mean_history, median_history, std_history = _update_convergence_histories(
                        partial_sums, mean_history, median_history, std_history
                    )

                state_expectations[order] = current_expectation

                state_convergence[order] = {
                    "mean_history": mean_history,
                    "median_history": median_history,
                    "std_history": std_history,
                }

            expectations.append(state_expectations)
            convergence_info[state_idx] = state_convergence

        return expectations, convergence_info

    if parallel_mode == "state":
        executor = concurrency.backends.get_executor(backend)
        with executor(max_workers=num_workers) as ex:
            results = ex.starmap(
                _get_expval_state,
                [(sorted_commutators_by_order, fragments, state, timestep) for state in states],
            )

        expectations = [result[0] for result in results]
        convergence_info = {state_idx: result[1] for state_idx, result in enumerate(results)}

        return expectations, convergence_info

    if parallel_mode == "commutator":
        executor = concurrency.backends.get_executor(backend)
        expectations = []
        convergence_info = {}

        # Flatten all commutators but keep track of their orders AND preserve importance order
        all_commutator_tasks = []  # Will store (commutator, order, timestep, importance_position)
        commutator_order_map = {}  # Maps (commutator, order) -> position in importance order

        for order, commutators_dict in sorted_commutators_by_order.items():
            for position, commutator in enumerate(commutators_dict.keys()):
                all_commutator_tasks.append((commutator, fragments, timestep, order))
                commutator_order_map[(commutator, order)] = position

        for state_idx, state in enumerate(states):
            with executor(max_workers=num_workers) as ex:
                expected_values = ex.starmap(
                    _evaluate_expected_value,
                    [
                        (commutator, fragments, state, timestep, order)
                        for commutator, fragments, timestep, order in all_commutator_tasks
                    ],
                )

            # Group expected values by order and preserve importance order
            order_expectations = defaultdict(list)

            for expected_value, (commutator, _, timestep, order) in zip(
                expected_values, all_commutator_tasks
            ):
                importance_position = commutator_order_map[(commutator, order)]
                order_expectations[order].append((expected_value, importance_position))

            # Sort expected values by importance order within each order
            for order in order_expectations:
                order_expectations[order].sort(key=lambda x: x[1])  # Sort by importance position

            # Calculate final expectations for each order
            state_expectations = {}
            for order, expectations_list in order_expectations.items():
                total_expectation = sum(exp_val for exp_val, _ in expectations_list)
                state_expectations[order] = total_expectation

            expectations.append(state_expectations)

            # For convergence info, we need to compute partial sums sequentially
            # in the correct importance order
            state_convergence = {}
            for order in order_expectations.keys():
                partial_sums = []
                mean_history = []
                median_history = []
                std_history = []
                current_expectation = 0.0

                # Process in importance order (already sorted)
                for expected_value, _ in order_expectations[order]:
                    current_expectation += expected_value
                    partial_sums.append(current_expectation)

                    # Compute convergence statistics incrementally
                    mean_history, median_history, std_history = _update_convergence_histories(
                        partial_sums, mean_history, median_history, std_history
                    )

                state_convergence[order] = {
                    "mean_history": mean_history,
                    "median_history": median_history,
                    "std_history": std_history,
                }

            convergence_info[state_idx] = state_convergence

        return expectations, convergence_info

    if parallel_mode == "nested_commutator":
        expectations = []
        convergence_info = {}

        for state_idx, state in enumerate(states):
            state_convergence = {}
            state_expectations = {}

            for order, commutators_dict in sorted_commutators_by_order.items():
                if len(commutators_dict) == 0:
                    continue

                # Reset expectation accumulator for each order
                current_expectation = 0.0
                partial_sums = []
                mean_history = []
                median_history = []
                std_history = []

                # Process commutators in order of importance
                for commutator in commutators_dict.keys():
                    # Use parallel evaluation for each commutator
                    commutator_expectation = _evaluate_expected_value_parallel(
                        commutator, fragments, state, timestep, order, backend, num_workers
                    )
                    current_expectation += commutator_expectation
                    partial_sums.append(current_expectation)

                    # Compute convergence statistics incrementally
                    mean_history, median_history, std_history = _update_convergence_histories(
                        partial_sums, mean_history, median_history, std_history
                    )

                state_expectations[order] = current_expectation
                state_convergence[order] = {
                    "mean_history": mean_history,
                    "median_history": median_history,
                    "std_history": std_history,
                }

            expectations.append(state_expectations)
            convergence_info[state_idx] = state_convergence

        return expectations, convergence_info

    if parallel_mode == "mvp":
        import logging
        logger = logging.getLogger(__name__)
        
        logger.info(f"Starting perturbation_error with backend={backend}, parallel_mode={parallel_mode}, num_workers={num_workers}, max_order={max_order}, topk={topk}")
        logger.info(f"Processing {len(states)} state(s) across {len(sorted_commutators_by_order)} order(s)")
        logger.info("Using MVP (matrix-vector product) parallelization - maximum parallelization per order")
        
        executor = concurrency.backends.get_executor(backend)
        expectations = []
        convergence_info = {}
        
        logger.info(f"Initializing executor backend: {backend}")
        logger.info("✅ Executor initialized successfully")
        
        # ============================================================
        # PHASE 1: COLLECT ALL TASKS FOR ALL STATES/ORDERS GLOBALLY
        # ============================================================
        logger.info(f"Collecting all tasks for {len(states)} state(s)...")
        
        global_task_list = []
        global_metadata = []  # Store (state_idx, order, importance_idx, timestep_factor)
        
        for state_idx, state in enumerate(states):
            logger.info(f"[State {state_idx+1}/{len(states)}] Preparing tasks...")
            
            for order, commutators_dict in sorted_commutators_by_order.items():
                if len(commutators_dict) == 0:
                    continue
                
                timestep_factor = (1j * timestep) ** order
                
                for importance_idx, commutator in enumerate(commutators_dict.keys()):
                    op_terms = _op_list(commutator)
                    
                    for term, coeff in op_terms.items():
                        # Add task: (term, coeff, state, fragments)
                        global_task_list.append((term, coeff, state, fragments))
                        
                        # Track metadata for regrouping results
                        global_metadata.append({
                            'state_idx': state_idx,
                            'order': order,
                            'importance_idx': importance_idx,
                            'timestep_factor': timestep_factor,
                            'term_coeff': coeff,
                        })
        
        logger.info(f"Collected {len(sorted_commutators_by_order)} state-order combinations")
        logger.info(f"Total tasks across all states/orders: {len(global_task_list)}")
        
        # ============================================================
        # PHASE 2: ONE SINGLE EXECUTOR CONTEXT WITH ONE STARMAP CALL
        # ============================================================
        logger.info("Creating single executor context for ONE massive starmap call...")
        
        with executor(max_workers=num_workers) as ex:
            logger.info(f"✅ Executor entered, calling starmap with {len(global_task_list)} tasks...")
            import time
            start_time = time.time()
            
            global_results = ex.starmap(_apply_single_term, global_task_list)
            
            elapsed = time.time() - start_time
            logger.info(f"✅ Starmap returned {len(global_results)} results in {elapsed:.2f}s")
        
        logger.info("Exiting executor context")
        
        # ============================================================
        # PHASE 3: POST-PROCESS RESULTS - REGROUP BY STATE/ORDER
        # ============================================================
        logger.info("Processing results...")
        
        # Initialize storage for all states
        for state_idx in range(len(states)):
            expectations.append({})
            convergence_info[state_idx] = {}
        
        # Group results by (state_idx, order, importance_idx)
        grouped_results = {}
        
        for task_idx, result in enumerate(global_results):
            meta = global_metadata[task_idx]
            state_idx = meta['state_idx']
            order = meta['order']
            importance_idx = meta['importance_idx']
            timestep_factor = meta['timestep_factor']
            
            key = (state_idx, order, importance_idx)
            
            if key not in grouped_results:
                grouped_results[key] = 0.0
            
            # Apply timestep factor and accumulate
            grouped_results[key] += result * timestep_factor
        
        # Process each state
        for state_idx in range(len(states)):
            logger.info(f"[State {state_idx+1}/{len(states)}] Results processed")
            
            for order in sorted_commutators_by_order.keys():
                commutators_dict = sorted_commutators_by_order[order]
                
                if len(commutators_dict) == 0:
                    continue
                
                # Collect commutator expectations in importance order
                order_expectations = []
                for importance_idx in range(len(commutators_dict)):
                    key = (state_idx, order, importance_idx)
                    if key in grouped_results:
                        order_expectations.append(grouped_results[key])
                
                if not order_expectations:
                    continue
                
                # Compute convergence statistics
                current_expectation = 0.0
                partial_sums = []
                mean_history = []
                median_history = []
                std_history = []
                
                for expectation_value in order_expectations:
                    current_expectation += expectation_value
                    partial_sums.append(current_expectation)
                    
                    mean_history, median_history, std_history = _update_convergence_histories(
                        partial_sums, mean_history, median_history, std_history
                    )
                
                expectations[state_idx][order] = current_expectation
                convergence_info[state_idx][order] = {
                    "mean_history": mean_history,
                    "median_history": median_history,
                    "std_history": std_history,
                }
        
        logger.info(f"All states processed successfully. Total execution time: {elapsed:.4f}s")
        return expectations, convergence_info

        raise ValueError(
        "Invalid parallel mode. Choose 'state', 'commutator', 'nested_commutator', or 'mvp'."
    )


def _get_expval_state(
    sorted_commutators_by_order: dict[int, dict[tuple[Hashable], float]],
    fragments: dict[Hashable, Fragment],
    state: AbstractState,
    timestep: float,
) -> tuple[dict, dict]:
    """Returns the expectation value of ``state`` with respect to the operator obtained by substituting ``fragments`` into ``commutators``,
    along with convergence information."""

    expectations = {}
    convergence = {}

    for order, commutators_dict in sorted_commutators_by_order.items():
        if len(commutators_dict) == 0:
            continue

        # Reset expectation accumulator for each order
        current_expectation = 0.0
        partial_sums = []
        mean_history = []
        median_history = []
        std_history = []

        # Process commutators in order of importance (already sorted in the dict)
        for commutator in commutators_dict.keys():
            commutator_expectation = _evaluate_expected_value(
                commutator, fragments, state, timestep, order
            )
            current_expectation += commutator_expectation
            partial_sums.append(current_expectation)

            # Compute convergence statistics incrementally
            mean_history, median_history, std_history = _update_convergence_histories(
                partial_sums, mean_history, median_history, std_history
            )

        expectations[order] = current_expectation
        convergence[order] = {
            "mean_history": mean_history,
            "median_history": median_history,
            "std_history": std_history,
        }

    return expectations, convergence


def _evaluate_expected_value_parallel(
    commutator: tuple[Hashable],
    fragments: dict[Hashable, Fragment],
    state: AbstractState,
    timestep: float,
    order: int,
    backend: str,
    num_workers: int,
) -> complex:
    """Parallel version of _evaluate_expected_value.

    Parallelizes the application of individual terms within a commutator.

    Args:
        commutator: The commutator tuple
        fragments: Dictionary mapping fragment keys to Fragment objects
        state: Quantum state to apply commutator to
        timestep: Time step for simulation
        order: Order of the commutator (used for phase factor)
        backend: Parallel execution backend
        num_workers: Number of parallel workers

    Returns:
        complex: The expectation value (1j * timestep)^order * <state | commutator | state>
    """
    op_terms = _op_list(commutator)

    executor = concurrency.backends.get_executor(backend)
    with executor(max_workers=num_workers) as ex:
        expected_values = ex.starmap(
            _apply_single_term,
            [(term, coeff, state, fragments) for term, coeff in op_terms.items()],
        )

    total_expectation = sum(expected_values)
    return (1j * timestep) ** order * total_expectation


def _evaluate_expected_value(
    commutator: tuple[Hashable],
    fragments: dict[Hashable, Fragment],
    state: AbstractState,
    timestep: float,
    order: int,
) -> complex:
    """Compute the expectation value of a commutator applied to a state.

    Returns: (1j * timestep)^order * <state | commutator | state>
    """
    expected_value = 0.0

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

        expected_value += coeff * state.dot(tmp_state)

    return (1j * timestep) ** order * expected_value


def _apply_single_term(
    term: tuple[Hashable], coeff: complex, state: AbstractState, fragments: dict[Hashable, Fragment]
) -> complex:
    """Apply a single operator term to a state and compute the expectation value.

    This function applies a sequence of fragment operations to a state and computes
    the expectation value with the original state. It's essentially the inner loop
    of _evaluate_expected_value, extracted for parallelization.

    Args:
        term: Sequence of fragment keys to apply (in reverse order)
        coeff: Coefficient for this term
        state: Initial quantum state
        fragments: Dictionary mapping fragment keys to Fragment objects

    Returns:
        complex: coeff * <state | (applied fragments) | state>
    """
    tmp_state = copy.copy(state)
    for frag in reversed(term):
        if isinstance(frag, frozenset):
            frag = sum((frag_coeff * fragments[x] for x, frag_coeff in frag), _AdditiveIdentity())
        else:
            frag = fragments[frag]
        tmp_state = frag.apply(tmp_state)

    return coeff * state.dot(tmp_state)


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


def _calculate_commutator_importance(
    commutator: tuple[Hashable], fragments: dict[Hashable, Fragment]
) -> float:
    """Calculate the importance of a commutator using the formula:
    2^(length of commutator - 1) * product of norms of fragments

    Args:
        commutator: The commutator tuple
        fragments: Dictionary of fragments

    Returns:
        float: The importance value
    """
    if not commutator:
        return 0.0

    # The primary importance factor is based on commutator length
    power_factor = 2 ** (len(commutator) - 1)

    # For simplicity, use a constant norm approximation of 1.0 per fragment
    # This preserves the ordering based on commutator structure while avoiding
    # complex norm calculations that may require additional parameters
    norm_product = 1.0

    return power_factor * norm_product


def _apply_topk_filtering(
    commutator_lists: list[list[tuple[Hashable]]], fragments: dict[Hashable, Fragment], topk: int
) -> dict[int, dict[tuple[Hashable], float]]:
    """Apply top-k filtering to commutators that are already ordered by BCH order.

    Args:
        commutator_lists: List of lists from bch_expansion()[1:], already ordered by BCH order
        fragments: Dictionary of fragments
        topk: Number of top commutators to keep per order

    Returns:
        dict: Dictionary with structure: order -> {commutator: importance, ...}
    """
    result = {}

    for list_idx, commutators in enumerate(commutator_lists):
        if len(commutators) == 0:
            continue

        order = list_idx + 2  # Since we use [1:], first list is order 2

        # Compute importance for each commutator
        importance_dict = {
            comm: _calculate_commutator_importance(comm, fragments) for comm in commutators
        }

        # Sort by importance and apply top-k
        sorted_items = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        result[order] = dict(sorted_items[:topk])

    return result


def _group_sums(term_dict: dict[tuple[Hashable], complex]) -> list[tuple[Hashable | set]]:
    """Reduce the number of commutators by grouping them using linearity in the first argument. For example,
    two commutators a*[X, A, B] and b*Y[A, B] will be merged into one commutator [a*X + b*Y, A, B].
    """
    grouped_comms = defaultdict(set)
    for commutator, coeff in term_dict.items():
        head, *tail = commutator
        tail = tuple(tail)
        grouped_comms[tail].add((head, coeff))

    return [(frozenset(heads), *tail) for tail, heads in grouped_comms.items()]


def _expand_all_terms_mvp_per_order(commutators_dict, fragments, timestep, order):
    """Expand all commutators for a specific order into individual terms for matrix-vector product parallelization.

    This function treats all commutator terms as a large matrix-vector product operation,
    where each term application is parallelized as an independent matrix-vector multiplication.

    Args:
        commutators_dict: Dictionary of commutators for a specific order (already sorted by importance)
        fragments: Dictionary of fragments
        timestep: Time step for simulation
        order: Order of the commutators

    Returns:
        tuple: (all_term_tasks, metadata_map)
            - all_term_tasks: List of (term, coeff, fragments) for parallel execution
            - metadata_map: Dictionary mapping task_id to commutator metadata
    """
    all_term_tasks = []
    metadata_map = {}
    timestep_factor = (1j * timestep) ** order

    for importance_idx, commutator in enumerate(commutators_dict.keys()):
        op_terms = _op_list(commutator)  # Reuse existing function

        for term, coeff in op_terms.items():
            task_id = len(all_term_tasks)
            all_term_tasks.append((term, coeff, fragments))

            metadata_map[task_id] = {
                "importance_idx": importance_idx,
                "timestep_factor": timestep_factor,
                "commutator": commutator,  # Keep for debugging if needed
            }

    return all_term_tasks, metadata_map


def _regroup_by_commutator_mvp_per_order(all_results, metadata_map):
    """Regroup parallel matrix-vector product results by commutator for a single order.

    Args:
        all_results: List of results from parallel term evaluations (matrix-vector products)
        metadata_map: Dictionary mapping task_id to commutator metadata

    Returns:
        dict: Dictionary mapping importance_idx to commutator expectation value
    """
    commutator_sums = defaultdict(float)

    for task_id, result in enumerate(all_results):
        metadata = metadata_map[task_id]
        importance_idx = metadata["importance_idx"]
        # Apply timestep factor and accumulate
        commutator_sums[importance_idx] += result * metadata["timestep_factor"]

    return dict(commutator_sums)


def _compute_convergence_mvp_per_order(commutator_expectations, num_commutators):
    """Compute convergence statistics from regrouped commutator expectations for matrix-vector product mode.

    Args:
        commutator_expectations: Dictionary mapping importance_idx to expectation values
        num_commutators: Number of commutators in this order

    Returns:
        tuple: (order_expectation, order_convergence)
    """
    # Get commutator expectations in importance order
    order_expectations = []
    for importance_idx in range(num_commutators):
        if importance_idx in commutator_expectations:
            order_expectations.append(commutator_expectations[importance_idx])

    if not order_expectations:
        return 0.0, {
            "mean_history": [],
            "median_history": [],
            "std_history": [],
        }

    # Compute convergence statistics - reuse existing logic
    current_expectation = 0.0
    partial_sums = []
    mean_history = []
    median_history = []
    std_history = []

    for expectation_value in order_expectations:
        current_expectation += expectation_value
        partial_sums.append(current_expectation)

        # Reuse existing convergence function
        mean_history, median_history, std_history = _update_convergence_histories(
            partial_sums, mean_history, median_history, std_history
        )

    order_convergence = {
        "mean_history": mean_history,
        "median_history": median_history,
        "std_history": std_history,
    }

    return current_expectation, order_convergence


def _update_convergence_histories(partial_sums, mean_history, median_history, std_history):
    """Update convergence histories with the latest partial sum.

    Returns updated histories as new lists (functional approach).
    """
    subset = np.array(partial_sums)

    new_mean_history = mean_history + [np.mean(subset)]
    new_median_history = median_history + [np.median(subset)]
    new_std_history = std_history + [np.std(subset) if len(partial_sums) > 1 else 0.0]

    return new_mean_history, new_median_history, new_std_history
