# Copyright 2024 Xanadu Quantum Technologies Inc.

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
    """Abstract class specifying which methods a Fragment class should implement"""

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
        """Compute the norm of the fragment"""
        raise NotImplementedError

    @abstractmethod
    def apply(self, state: AbstractState) -> AbstractState:
        """Apply to a state on the right"""
        raise NotImplementedError

    def expectation(self, left: AbstractState, right: AbstractState) -> float:
        """Return the expectation value of a state"""
        return left.dot(self.apply(right))


def commutator(a: Fragment, b: Fragment) -> Fragment:
    """Return the commutator [a, b]"""
    return a @ b - b @ a


def nested_commutator(fragments: Sequence[Fragment]) -> Fragment:
    """Return [a, [b, [c, d]]]"""

    if len(fragments) < 2:
        raise ValueError("Need at least two fragments to commute.")

    if len(fragments) == 2:
        return commutator(*fragments)

    head, *tail = fragments

    return commutator(head, nested_commutator(tail))


class AbstractState(ABC):
    """Abstract class defining the methods a class needs to implement to be used to compute an expectation value"""

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
        """Return a representation of the zero state"""
        raise NotImplementedError

    @abstractmethod
    def dot(self, other: AbstractState):
        """Return the dot product of two states"""
        raise NotImplementedError
