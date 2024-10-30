from abc import ABC, abstractmethod
from typing import Callable, Dict

from .resource_container import CompressedResourceOp


class ResourceConstructor(ABC):
    r"""This is an abstract class that defines the methods a PennyLane Operator
    must implement in order to be used for resource estimation.

    .. details::

        **Example**
        import pennylane as qml

        class ResourceQFT(qml.QFT, ResourceConstructor):
            def compute_resources(num_wires):
                return

            def resource_rep(self):
                return
    """

    @staticmethod
    @abstractmethod
    def _resource_decomp(*args, **kwargs) -> Dict[CompressedResourceOp, int]:
        """Returns the Resource object. This method is only to be used inside
        the methods of classes inheriting from ResourceConstructor."""

    @classmethod
    def resources(cls, *args, **kwargs):
        """Returns the Resource object. This method is intended to be user facing
        and overridable."""
        return cls._resource_decomp(*args, **kwargs)

    @classmethod
    def set_resources(cls, new_func: Callable) -> None:
        """Override the compute_resources method."""
        cls.resources = new_func

    @abstractmethod
    def resource_rep(self) -> CompressedResourceOp:
        """Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation."""


class ResourcesNotDefined(Exception):
    """Exception to be raised when a ResourceConstructor does not implement compute_resources"""
