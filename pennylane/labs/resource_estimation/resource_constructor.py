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
r"""Abstract base class for resource operators."""
from abc import ABC, abstractmethod
from typing import Callable, Dict

import pennylane.labs.resource_estimation.resource_container as rc


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
    def _resource_decomp(*args, **kwargs) -> Dict[rc.CompressedResourceOp, int]:
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

    @abstractmethod
    def resource_params(self) -> dict:
        """Returns a dictionary containing the minimal information needed to
        compute a comparessed representation"""

    @classmethod
    @abstractmethod
    def resource_rep(cls, **kwargs) -> rc.CompressedResourceOp:
        """Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation."""

    def resource_rep_from_op(self) -> rc.CompressedResourceOp:
        """Returns a compressed representation directly from the operator"""
        params = self.resource_params()
        return self.__class__.resource_rep(**params)


class ResourcesNotDefined(Exception):
    """Exception to be raised when a ``ResourceConstructor`` does not implement _resource_decomp"""