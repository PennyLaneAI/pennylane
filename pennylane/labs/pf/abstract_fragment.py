"""The Fragment class"""

from __future__ import annotations
from abc import ABC, abstractmethod

class Fragment(ABC):
    """Abstract class specifying which methods a Fragment class should implement"""

    @abstractmethod
    def __add__(self, other: Fragment) -> Fragment:
        raise NotImplementedError

    def __sub__(self, other: Fragment) -> Fragment:
        return self + (-1)*other

    @abstractmethod
    def __mul__(self, scalar: float) -> Fragment:
        raise NotImplementedError

    @abstractmethod
    def __matmul__(self, other: Fragment) -> Fragment:
        raise NotImplementedError

    def commutator(self, other: Fragment) -> Fragment:
        """Return the commutator [self, other]"""
        return (self @ other) - (other @ self)

    def nested_commutator(self, a: Fragment, b: Fragment) -> Fragment:
        """ Return the commutator [self, [a, b]]"""
        return self.commutator(a.commutator(b))

    @abstractmethod
    def norm(self) -> float:
        """Compute the norm of the fragment"""
        raise NotImplementedError

    @abstractmethod
    def mul_state(self, state):
        """Apply to a state on the right"""
        raise NotImplementedError
