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
from collections import Counter
from collections.abc import Hashable
from typing import Dict, List, Sequence, Tuple

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


# pylint: disable=too-many-arguments
def perturbation_error(
    product_formula: ProductFormula,
    fragments: Dict[Hashable, Fragment],
    states: Sequence[AbstractState],
    order: int,
    timestep: float = 1.0,
    num_workers: int = 1,
    backend: str = "serial",
    parallel_mode: str = "state",
) -> List[float]:
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

    bch = bch_expansion(product_formula(1j * timestep), order)
    commutators = {
        commutator: coeff for comm_dict in bch[1:] for commutator, coeff in comm_dict.items()
    }

    if backend == "serial":
        assert num_workers == 1, "num_workers must be set to 1 for serial execution."
        expectations = []
        for state in states:
            new_state = _AdditiveIdentity()
            for commutator, coeff in commutators.items():
                new_state += coeff * _apply_commutator(commutator, fragments, state)
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
                    _apply_commutator_coeff,
                    [
                        (commutator, coeff, fragments, state)
                        for commutator, coeff in commutators.items()
                    ],
                )

            new_state = _AdditiveIdentity()
            for applied_state in applied_commutators:
                new_state += applied_state

            expectations.append(state.dot(new_state))

        return expectations

    raise ValueError("Invalid parallel mode. Choose 'state' or 'commutator'.")


def _get_expval_state(commutators, fragments, state: AbstractState) -> float:
    """Returns the expectation value of ``state`` with respect to the operator obtained by substituting ``fragments`` into ``commutators``."""

    new_state = _AdditiveIdentity()
    for commutator, coeff in commutators.items():
        new_state += coeff * _apply_commutator(commutator, fragments, state)

    return state.dot(new_state)


def _apply_commutator(
    commutator: Tuple[Hashable], fragments: Dict[Hashable, Fragment], state: AbstractState
) -> AbstractState:
    """Returns the state obtained from applying ``commutator`` to ``state``."""

    new_state = _AdditiveIdentity()

    for term, coeff in _op_list(commutator).items():
        tmp_state = copy.copy(state)
        for frag in reversed([fragments[x] for x in term]):
            tmp_state = frag.apply(tmp_state)

        new_state += coeff * tmp_state

    return new_state


def _apply_commutator_coeff(
    commutator: Tuple[Hashable], coeff, fragments: Dict[Hashable, Fragment], state: AbstractState
) -> AbstractState:
    """Returns the state obtained from applying ``commutator`` to ``state`` and multiplying by the scalar ``coeff``."""

    new_state = _AdditiveIdentity()

    for term, co in _op_list(commutator).items():
        tmp_state = None
        for frag in reversed([fragments[x] for x in term]):
            tmp_state = frag.apply(tmp_state) if tmp_state is not None else frag.apply(state)
        new_state += co * tmp_state

    return coeff * new_state


def _op_list(commutator) -> Dict[Tuple[Hashable], complex]:
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
