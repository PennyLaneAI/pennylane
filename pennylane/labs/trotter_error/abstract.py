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
"""The abstract Fragment class that defines the API for Trotter error computations"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence


class _AdditiveIdentity:
    """Only used to initialize accumulators for summing Fragments"""

    def __add__(self, other):
        return other

    def __radd__(self, other):
        return other


class Fragment(ABC):
    r"""Abstract class used to define a fragment object for product formula error estimation. For
    Trotter error a Hamiltonian is expressed a sum of fragments :math:`H = \sum H_i`.

    A :class:`~.Fragment` is an object that has a well-defined notion of a commutator. To ensure
    the existence of commutators, the implementation requires the following arithmetic dunder
    methods:

    * :meth:`~.__add__`: implements addition

    * :meth:`~.__mul__`: implements multiplication

    * :meth:`~.__matmul__`: implements matrix multiplication

    In addition to the arithmetic operators, a ``norm`` method should be defined. The norm is
    required to compute error estimates of Trotter error operators.
    """

    @abstractmethod
    def __add__(self, other: Fragment) -> Fragment:
        raise NotImplementedError

    def __sub__(self, other: Fragment) -> Fragment:
        return self + (-1) * other

    @abstractmethod
    def __mul__(self, scalar: float) -> Fragment:
        raise NotImplementedError

    @abstractmethod
    def __matmul__(self, other: Fragment) -> Fragment:
        raise NotImplementedError

    def commutator(self, other: Fragment) -> Fragment:
        """Evaluates the commutator :math:`[A, B] = AB - BA` of two fragments"""
        return self @ other - other @ self

    @abstractmethod
    def apply(self, state: TrotterState) -> TrotterState:
        r"""For a fragment :math:`H` and state :math:`|\psi\rangle` return the state :math:`H|\psi\rangle`
        The type of ``state`` is determined by the implementation of :math:`H` as a ``Fragment`` object.
        Implementation of this function is mandatory for :func:`~.pennylane.labs.trotter_error.perturbation_error`.

        Args:
            state (TrotterState): an object representing a quantum state

        Returns:
            TrotterState: the result of applying the ``Fragment`` to ``state``

        """
        raise NotImplementedError

    def expectation(self, left: TrotterState, right: TrotterState) -> float:
        """Return the expectation value of a state. The type of ``state`` is determined by each
        class inheriting from ``Fragment``.

        Args:
            left (TrotterState): the state to be multiplied on the left of the ``Fragment``
            right (TrotterState): the state to be multiplied on the right of the ``Fragment``

        Returns:
            float: the expectation value obtained by applying ``Fragment`` to the given states
        """
        return left.dot(self.apply(right))

    def initialize_parallel_job(self, backend: str):
        """Set up required for parallel compatibility. This method is called in
        :func:`~.pennylane.labs.trotter_error.perturbation_error` before the parallel computations
        are called. Any required setup for parallel jobs goes into this function."""

    def start_parallel_job(self, state: TrotterState):
        """Start perturbation error computation in parallel. This method is the first operation called
        in each parallel job dispatched by :func:`~.pennylane.labs.trotter_error.perturbation_error`.
        Anything necessary to start the parallel job (such as reading from disk memory) goes into this
        function."""


def commutator(a: Fragment, b: Fragment) -> Fragment:
    """Return the commutator of two :class:`~.Fragment` objects

    Args:
        a (Fragment): the :class:`~.Fragment` on the left side of the commutator
        b (Fragment): the :class:`~.Fragment` on the right side of the commutator

    Returns:
        Fragment: the commutator ``[a, b]``
    """
    return a @ b - b @ a


def nested_commutator(fragments: Sequence[Fragment]) -> Fragment:
    """Return the nested commutator of a sequence of :class:`~.Fragment` objects

    Args:
        fragments (Sequence[Fragment]): a sequence of fragments

    Returns:
        Fragment: the nested commutator of the fragments
    """

    if len(fragments) == 0:
        return []

    if len(fragments) == 1:
        if isinstance(fragments[0], Sequence):
            return nested_commutator(fragments[0])

        return fragments[0]

    head, *tail = fragments

    if isinstance(head, Sequence):
        return commutator(nested_commutator(head), nested_commutator(tail))

    return commutator(head, nested_commutator(tail))


class TrotterState(ABC):
    """Abstract class used to define a state object for perturbation error estimation.

    A class inheriting from ``TrotterState`` must implement the following dunder methods.

    * ``__add__``: implements addition
    * ``__mul__``: implements multiplication

    Additionally, it requires the following methods.

    * ``zero_state``: returns a representation of the zero state
    * ``dot``: implements the dot product of two states
    """

    @abstractmethod
    def __add__(self, other: TrotterState) -> TrotterState:
        raise NotImplementedError

    def __sub__(self, other: TrotterState) -> TrotterState:
        return self + (-1) * other

    @abstractmethod
    def __mul__(self, scalar: float) -> TrotterState:
        raise NotImplementedError

    @abstractmethod
    def dot(self, other: TrotterState) -> float:
        """Compute the dot product of two states.

        Args:
            other (TrotterState): the state to take the dot product with

        Returns:
           float: the dot product of self and other
        """
        raise NotImplementedError

    def initialize_parallel_job(self, backend: str):
        """Set up required for parallel compatibility. This method is called in
        :func:`~.pennylane.labs.trotter_error.perturbation_error` before the parallel computations
        are called. Any required setup for parallel jobs goes into this function."""

    def start_parallel_job(self, backend: str):
        """Start perturbation error computation in parallel. This method is the first operation called
        in each parallel job dispatched by :func:`~.pennylane.labs.trotter_error.perturbation_error`.
        Anything necessary to start the parallel job (such as reading from disk memory) goes into this
        function."""
