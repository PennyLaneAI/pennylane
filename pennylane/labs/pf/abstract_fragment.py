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

    @abstractmethod
    def norm(self) -> float:
        """Compute the norm of the fragment"""
        raise NotImplementedError

    @abstractmethod
    def lmul_state(self, state):
        """Apply to a state on the left"""
        raise NotImplementedError

    @abstractmethod
    def rmul_state(self, state):
        """Apply to a state on the right"""
        raise NotImplementedError
