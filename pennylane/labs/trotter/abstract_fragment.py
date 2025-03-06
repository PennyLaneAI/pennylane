"""The Fragment class"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence


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
    def norm(self, *args) -> float:
        """Compute the norm of the fragment"""
        raise NotImplementedError

    @abstractmethod
    def apply(self, state):
        """Apply to a state on the right"""
        raise NotImplementedError


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


class State(ABC):
    """Abstract class specifying which methods a State class should implement"""

    @abstractmethod
    def __add__(self, other: State) -> State:
        raise NotImplementedError

    def __sub__(self, other: State) -> State:
        return self + (-1) * other

    @abstractmethod
    def __mul__(self, scalar: float) -> State:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def zero_state(cls) -> State:
        """Return a representation of the zero state"""
        raise NotImplementedError

    @abstractmethod
    def dot(self, other: State):
        """Return the dot product self and other"""
        raise NotImplementedError
