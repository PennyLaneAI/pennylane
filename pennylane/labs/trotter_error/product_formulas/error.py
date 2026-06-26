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
"""Functions for retrieving effective error from fragments"""

import copy
import math
from collections import defaultdict
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from itertools import groupby

import numpy as np

from pennylane import concurrency
from pennylane.labs.trotter_error import Fragment, TrotterState
from pennylane.labs.trotter_error.abstract import _AdditiveIdentity
from pennylane.labs.trotter_error.product_formulas.bch import bch_expansion
from pennylane.labs.trotter_error.product_formulas.commutator import (
    ASTNode,
    CommutatorNode,
    SymbolNode,
)
from pennylane.labs.trotter_error.product_formulas.product_formula import ProductFormula

# pylint: disable=too-many-arguments, too-many-positional-arguments


def effective_hamiltonian(
    product_formula: ProductFormula,
    fragments: dict[Hashable, Fragment],
    order: int,
    timestep: float = 1.0,
    num_workers: int = 1,
    backend: str = "serial",
):
    r"""Compute the effective Hamiltonian :math:`\hat{H}_{eff} = \hat{H} + \hat{\epsilon}` that
    corresponds to a given product formula.

    Args:
        product_formula (ProductFormula): A product formula used to approximate the time-evolution
            operator for a Hamiltonian.
        fragments (dict[Hashable, :class:`~.pennylane.labs.trotter_error.Fragment`): The fragments
            that sum to the Hamiltonian. The keys in the dictionary must match the labels used to
            build the :class:`~.pennylane.labs.trotter_error.ProductFormula` object.
        order (int): An order k approximation will compute the effective Hamiltonian up to order k commutators.
        timestep (float): time step for the Trotter error operator
        num_workers (int): the number of concurrent units used for the computation. Default value is
            set to 1.
        backend (string): the executor backend from the list of supported backends.
            Available options : "mp_pool", "cf_procpool", "cf_threadpool", "serial", "mpi4py_pool",
            "mpi4py_comm". Default value is set to "serial".

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
    >>> frag_data = list(zip(frag_labels, frag_coeffs))
    >>>
    >>> pf = ProductFormula(frag_data)
    >>> frags = dict(enumerate(vibrational_fragments(n_modes, freqs, taylor_coeffs)))
    >>> type(effective_hamiltonian(pf, frags, order=5, timestep=delta))
    <class 'pennylane.labs.trotter_error.realspace.realspace_operator.RealspaceSum'>
    """

    if not product_formula.symbol_set.issubset(fragments.keys()):
        raise ValueError("Fragments do not match product formula")

    bch = bch_expansion(product_formula(1j * timestep), order)

    executor = concurrency.backends.get_executor(backend)
    with executor(max_workers=num_workers) as ex:
        partial_sum = ex.starmap(
            _eval_commutator,
            [(commutator, coeff, fragments) for commutator, coeff in bch.items()],
        )

    eff = _AdditiveIdentity()
    for term in partial_sum:
        eff += term
    return eff


def _eval_commutator(commutator, coeff, fragments):
    r"""Computes a commutator after replacing symbols in the commutator with concrete fragments.

    Args:
        commutator: commutator to be evaluated
        coeff (complex): coefficient associated with the commutator
        fragments (dict): dictionary representing a sequence of fragments

    Returns:
        ndarray: the evaluated form of the commutator
    """

    return coeff * commutator.eval(fragments)


@dataclass
class ImportanceConfig:
    """This class is used as an optional argument to :func:`~.pennylane.labs.trotter_error.product_formulas.perturbation_error`
    to enable the importance sampling feature. When used the perturbation error will be computed using
    only the ``topk`` most important commutators. The importance of a commutator is induced from
    ``weights`` which stores a user-defined importance score for each fragment label. The keys in
    ``weights`` must be identical to the symbols used to build the :class:`~.pennylane.labs.trotter_error.product_formulas.product_formula.ProductFormula`
    object passed into :func:`~.pennylane.labs.trotter_error.product_formulas.perturbation_error`.


    """

    topk: int
    """samples the ``topk`` most important commutators to estimate the perturbation error."""

    weights: dict[Hashable, float]
    """a dictionary mapping fragment labels to their importance score."""

    history: bool = False
    """tracks the convergence history of the perturbation error per commutator evaluation.
       when true the output of :func:`~.pennylane.labs.trotter_error.product_formulas.perturbation_error`
       is modified to include the convergence history."""


# pylint: disable=too-many-branches
def perturbation_error(
    product_formula: ProductFormula,
    fragments: dict[Hashable, Fragment],
    state: TrotterState,
    order: int | Sequence[int],
    timestep: float = 1.0,
    num_workers: int = 1,
    backend: str = "serial",
    importance: ImportanceConfig = None,
) -> list[float] | list[dict[int, dict]]:
    r"""Computes the perturbation theory error using the effective Hamiltonian
    :math:`\hat{H}_{eff} = \hat{H} + \hat{\epsilon}` for a  given product formula.


    For a state :math:`\left| \psi \right\rangle` the perturbation theory error is given by the
    expectation value :math:`\left\langle \psi \right| \hat{\epsilon} \left| \psi \right\rangle`.

    Args:
        product_formula (ProductFormula): the :class:`~.pennylane.labs.trotter_error.ProductFormula` used to obtain
            the effective Hamiltonian
        fragments (Sequence[Fragments]): the set of :class:`~.pennylane.labs.trotter_error.Fragment`
            objects to compute the perturbation error from
        states (Sequence[TrotterState]): the states to compute expectation values from
        order (int | Sequence[int]): Computes the perturbation error using commutators of order `order`.
        timestep (float): time step for the Trotter error operator.
        num_workers (int): the number of concurrent units used for the computation. Default value
            is set to 1.
        backend (string): the executor backend from the list of supported backends.
            Available options : "mp_pool", "cf_procpool", "cf_threadpool", "serial", "mpi4py_pool",
            "mpi4py_comm". Default value is set to "serial".

    Returns:
        list[dict[int, float]]: the list of dictionaries of expectation values computed from the
            Trotter error operator and the input states. The dictionary is indexed by the commutator
            orders and its value is the error obtained from the commutators of that order.

    **Example**

    >>> import numpy as np
    >>> from pennylane.labs.trotter_error import HOState, ProductFormula, vibrational_fragments, perturbation_error

    >>> frag_labels = [0, 1, 1, 0]
    >>> frag_coeffs = [1/2, 1/2, 1/2, 1/2]
    >>> frag_data = list(zip(frag_labels, frag_coeffs))
    >>> pf = ProductFormula(frag_data)

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

    >>> gridpoints = 5
    >>> state = HOState(n_modes, gridpoints, {(0, 0): 1})

    >>> errors = perturbation_error(pf, frags, state, order=3)
    >>> print(errors)
     defaultdict(<class 'int'>, {3: np.complex128(0.9189251160920877j)})
    """

    if not product_formula.symbol_set.issubset(fragments.keys()):
        raise ValueError("Fragments do not match product formula")

    if not all(isinstance(fragment, Fragment) for fragment in fragments.values()):
        raise TypeError("Fragments must be an instance of Fragment.")

    if not isinstance(state, TrotterState):
        raise TypeError("State must be an instance of TrotterState.")

    if not isinstance(order, Sequence):
        order = [order]

    max_order = max(order)

    commutators = {
        comm: coeff
        for comm, coeff in bch_expansion(product_formula, max_order).items()
        if comm.order in order
    }

    track_history = False

    if importance is not None:
        if not product_formula.symbol_set.issubset(importance.weights.keys()):
            raise ValueError("Fragment weights do not match product formula")

        commutators = _get_topk(commutators, importance, order)
        track_history = importance.history

    if backend == "serial":
        assert num_workers == 1, "num_workers must be set to 1 for serial execution."

        expectations_by_order = defaultdict(int)
        partial_sums_by_order = defaultdict(list)

        for commutator, coeff in commutators.items():
            _, expectation = _compute_expectation(commutator, fragments, state, coeff)

            expectations_by_order[commutator.order] += (
                1j * timestep
            ) ** commutator.order * expectation
            partial_sums_by_order[commutator.order].append(expectations_by_order[commutator.order])

        return _format_output(expectations_by_order, partial_sums_by_order, track_history)

    state.initialize_parallel_job(backend)
    for fragment in fragments.values():
        fragment.initialize_parallel_job(backend)

    comms = []
    terms = []

    for commutator, coeff1 in commutators.items():
        for term, coeff2 in commutator.expand().items():
            comms.append(commutator)
            terms.append((term, coeff1 * coeff2))

    executor = concurrency.backends.get_executor(backend)

    with executor(max_workers=num_workers) as ex:
        applied_terms = ex.starmap(
            _apply_single_term, [(term, fragments, state, coeff, backend) for term, coeff in terms]
        )

    expectations = defaultdict(int)
    partial_sums = defaultdict(list)

    for comm, group in groupby(zip(comms, applied_terms), key=lambda x: x[0]):
        for _, expectation in group:
            expectations[comm.order] += (1j * timestep) ** comm.order * expectation
        partial_sums[comm.order].append(expectations[comm.order])

    return _format_output(expectations, partial_sums, track_history)


def _apply_single_term(
    term: tuple[SymbolNode],
    fragments: Sequence[Fragment],
    state: TrotterState,
    coeff: float,
    backend: str = None,
) -> float:
    state.start_parallel_job(backend)
    for fragment in fragments.values():
        fragment.start_parallel_job(state)

    new_state = copy.deepcopy(state)

    for symbol in reversed(term):
        new_state = symbol.eval(fragments).apply(new_state)

    return coeff * state.dot(new_state)


def _compute_expectation(
    commutator: CommutatorNode,
    fragments: dict[Hashable, Fragment],
    state: TrotterState,
    coeff: float,
) -> tuple[CommutatorNode, float]:
    """Returns the expectation value obtained from applying ``commutator`` to ``state``."""

    expectation = 0

    for term, exp_coeff in commutator.expand().items():
        new_state = copy.deepcopy(state)

        for symbol in reversed(term):
            new_state = symbol.eval(fragments).apply(new_state)

        expectation += exp_coeff * state.dot(new_state)

    return commutator, coeff * expectation


def _compute_importance(commutator: ASTNode, weights: dict[Hashable, float]) -> float:
    """Upper bound the importance of a commutator with the identity ||[A, B]|| < 2||AB||"""
    return np.abs(
        2 ** (commutator.order - 1) * math.prod([x.eval(weights) for x in commutator.leaves()])
    )


def _get_topk(
    commutators: dict[ASTNode, float], importance: ImportanceConfig, orders: Sequence[int]
) -> dict[ASTNode, float]:
    """Return only the top k most important commutators"""

    ret = {}

    for order in orders:
        comms_of_order = {comm: coeff for comm, coeff in commutators.items() if comm.order == order}
        sorted_comms = sorted(
            comms_of_order.keys(),
            key=lambda x: _compute_importance(x, importance.weights),
            reverse=True,
        )

        for comm in sorted_comms[: importance.topk]:
            ret[comm] = comms_of_order[comm]

    return ret


def _format_output(expectations, partial_sums, track_history):
    if not track_history:
        return expectations

    assert expectations.keys() == partial_sums.keys()

    return {
        order: {
            "error": expectations[order],
            "mean": np.mean(partial_sums[order]),
            "median": np.median(partial_sums[order]),
            "std": np.std(partial_sums[order]),
            "partial sums": partial_sums[order],
        }
        for order in expectations.keys()
    }
