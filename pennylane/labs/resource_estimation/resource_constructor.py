from abc import ABC, abstractmethod
from typing import Callable

from .resource_container import CompressedResourceOp, Resources


class ResourceConstructor(ABC):
    @staticmethod
    @abstractmethod
    def compute_resources(*args, **kwargs) -> Resources:
        """Returns the Resource object associated with the Operator."""

    @classmethod
    def set_compute_resources(cls, new_func: Callable) -> None:
        """Override the compute_resources method."""
        cls.compute_resources = new_func

    @abstractmethod
    def resource_rep(self) -> CompressedResourceOp:
        """Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation."""

class ResourcesNotDefined(Exception):
    """Exception to be raised when a ResourceConstructor does not implement compute_resources"""
