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
"""The Fragment class"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence


class Fragment(ABC):
    """Abstract class used to define a fragment object for product formula error estimation.

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

    @abstractmethod
    def norm(self, params: dict) -> float:
        """Compute the norm of the fragment.

        Args:
            params (Dict): A dictionary of parameters needed to compute the norm. It should be
                specified for each class inheriting from :class:`~.Fragment`.

        Returns:
            float: the norm of the :class:`~.Fragment`

        """
        raise NotImplementedError

    @abstractmethod
    def apply(self, state: AbstractState) -> AbstractState:
        """Apply the Fragment to a state on the right. The type of ``state`` is determined by each class inheriting from ``Fragment``.

        Args:
            state (AbstractState): an object representing a quantum state

        Returns:
            AbstractState: the result of applying the ``Fragment`` to ``state``

        """
        raise NotImplementedError

    def expectation(self, left: AbstractState, right: AbstractState) -> float:
        """Return the expectation value of a state. The type of ``state`` is determined by each class inheriting from ``Fragment``.

        Args:
            left (AbstractState): the state to be multiplied on the left of the ``Fragment``
            right (AbstractState): the state to be multiplied on the right of the ``Fragment``

        Returns:
            float: the expectation value obtained by applying ``Fragment`` to the given states
        """
        return left.dot(self.apply(right))


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


class AbstractState(ABC):
    """Abstract class used to define a state object for product formula error estimation.

    A class inheriting from ``AbstractState`` must implement the following dunder methods.

    * ``__add__``: implements addition
    * ``__mul__``: implements multiplication

    Additionally, it requires the following methods.

    * ``zero_state``: returns a representation of the zero state
    * ``dot``: implments the dot product of two states
    """

    @abstractmethod
    def __add__(self, other: AbstractState) -> AbstractState:
        raise NotImplementedError

    def __sub__(self, other: AbstractState) -> AbstractState:
        return self + (-1) * other

    @abstractmethod
    def __mul__(self, scalar: float) -> AbstractState:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def zero_state(cls) -> AbstractState:
        """Return a representation of the zero state.

        Returns:
            AbstractState: an ``AbstractState`` representation of the zero state
        """
        raise NotImplementedError

    @abstractmethod
    def dot(self, other: AbstractState) -> float:
        """Compute the dot product of two states.

        Args:
            other (AbstractState): the state to take the dot product with

        Returns:
           float: the dot product of self and other
        """
        raise NotImplementedError
