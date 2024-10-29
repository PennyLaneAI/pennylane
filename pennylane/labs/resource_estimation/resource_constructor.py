from abc import ABC, abstractmethod
from typing import Callable, Dict

from .resource_container import CompressedResourceOp


class ResourceConstructor(ABC):
    r"""This is an abstract class that defines the methods a PennyLane Operator
    must implement in order to be used for resource estimation.

    .. details::

        **Example**

        A PennyLane Operator can be extended for resource estimation by creating a new class that inherits from both the Operator and Resource Constructor.
        Here is an example showing how to extend ``qml.QFT`` for resource estimation.

        .. code-block:: python

            import pennylane as qml
            from pennylane.labs.resource_estimation import CompressedResourceOp, ResourceConstructor

            class ResourceQFT(qml.QFT, ResourceConstructor):

                @staticmethod
                def _resource_decomp(num_wires) -> dict:
                    if not isinstance(num_wires, int):
                        raise TypeError("num_wires must be an int.")

                    if num_wires < 1:
                        raise ValueError("num_wires must be greater than 0.")

                    gate_types = {}

                    hadamard = CompressedResourceOp(qml.Hadamard, {})
                    swap = CompressedResourceOp(qml.SWAP, {})
                    ctrl_phase_shift = CompressedResourceOp(qml.ControlledPhaseShift, {})

                    gate_types[hadamard] = num_wires
                    gate_types[swap] = num_wires // 2
                    gate_types[ctrl_phase_shift] = num_wires*(num_wires - 1) // 2

                    return gate_types

                def resource_rep(self) -> CompressedResourceOp:
                    params = {"num_wires": len(self.wires)}
                    return CompressedResourceOp(qml.QFT, params)
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
        """Set a custom resource method."""
        cls.resources = new_func

    @staticmethod
    @abstractmethod
    def resource_rep() -> CompressedResourceOp:
        """Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation."""

class ResourcesNotDefined(Exception):
    """Exception to be raised when a ``ResourceConstructor`` does not implement _resource_decomp"""
