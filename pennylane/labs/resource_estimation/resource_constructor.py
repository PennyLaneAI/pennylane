from abc import ABC, abstractmethod
from typing import Callable

import pennylane.labs.resource_estimation.resources_base as resources_base
import pennylane.labs.resource_estimation.resource_container as resource_container

CompressedResourceOp = resource_container.CompressedResourceOp
Resources = resources_base.Resources

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
