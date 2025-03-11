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
