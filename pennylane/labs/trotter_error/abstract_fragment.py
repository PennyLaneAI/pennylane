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
from typing import Dict, Sequence


class Fragment(ABC):
    """Abstract class used to define a fragment object for product formula error estimation.

    A :class:`~.Fragment` is an object that has a well-defined notion of a commutator. To ensure
    the existence of commutators, the implementation requires the following arithmetic dunder
    methods:

    * :meth:`~.__add__`: implements addition

    * :meth:`~.__sub__`: implements subtraction

    * :meth:`~.__mul__`: implements multiplication

    * :meth:`~.__matmul__`: implements matrix multiplication

    In addition to the arithmetic operators, a ``norm`` method should be defined. The norm is
    required to compute error estimates of error operators obtained by computing nested commutators.
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
    def norm(self, params: Dict) -> float:
        """Compute the norm of the fragment.

        Args:
            params (Dict): A dictionary of parameters needed to compute the norm. It should be
                specified for each class inheriting from :class:`~.Fragment`.

        Returns:
            float: the norm of the :class:`~.Fragment`

        """
        raise NotImplementedError


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

    if len(fragments) < 2:
        raise ValueError("Need at least two fragments to commute.")

    if len(fragments) == 2:
        return commutator(*fragments)

    head, *tail = fragments

    return commutator(head, nested_commutator(tail))
