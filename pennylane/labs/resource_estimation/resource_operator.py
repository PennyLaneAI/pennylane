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
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, Dict

if TYPE_CHECKING:
    from pennylane.labs.resource_estimation import CompressedResourceOp


class ResourceOperator(ABC):
    r"""Abstract class that defines the methods a PennyLane Operator
    must implement in order to be used for resource estimation.

    .. details::

        **Example**

        A PennyLane Operator can be extended for resource estimation by creating a new class that
        inherits from both the ``Operator`` and ``ResourceOperator``. Here is an example showing how to
        extend ``qml.QFT`` for resource estimation.

        .. code-block:: python

            import pennylane as qml
            from pennylane.labs.resource_estimation import CompressedResourceOp, ResourceOperator

            class ResourceQFT(qml.QFT, ResourceOperator):

                @staticmethod
                def _resource_decomp(num_wires) -> dict[CompressedResourceOp, int]:
                    gate_types = {}

                    hadamard = ResourceHadamard.resource_rep()
                    swap = ResourceSWAP.resource_rep()
                    ctrl_phase_shift = ResourceControlledPhaseShift.resource_rep()

                    gate_types[hadamard] = num_wires
                    gate_types[swap] = num_wires // 2
                    gate_types[ctrl_phase_shift] = num_wires*(num_wires - 1) // 2

                    return gate_types

                def resource_params(self, num_wires) -> dict:
                    return {"num_wires": num_wires}

                @classmethod
                def resource_rep(cls, num_wires) -> CompressedResourceOp:
                    params = {"num_wires": num_wires}
                    return CompressedResourceOp(cls, params)

        Which can be instantiated as a normal operation, but now contains the resources:

        .. code-block:: bash

            >>> op = ResourceQFT(range(3))
            >>> op.resources(**op.resource_params())
            {Hadamard(): 3, SWAP(): 1, ControlledPhaseShift(): 3}

    """

    @staticmethod
    @abstractmethod
    def _resource_decomp(*args, **kwargs) -> Dict[CompressedResourceOp, int]:
        """Returns a dictionary to be used for internal tracking of resources. This method is only to be used inside
        the methods of classes inheriting from ResourceOperator."""

    @classmethod
    def resources(cls, *args, **kwargs) -> Dict[CompressedResourceOp, int]:
        """Returns a dictionary containing the counts of each operator type used to
        compute the resources of the operator."""
        return cls._resource_decomp(*args, **kwargs)

    @classmethod
    def set_resources(cls, new_func: Callable) -> None:
        """Set a custom resource method."""
        cls.resources = new_func

    @abstractmethod
    def resource_params(self) -> dict:
        """Returns a dictionary containing the minimal information needed to
        compute a comparessed representation."""

    @classmethod
    @abstractmethod
    def resource_rep(cls, *args, **kwargs) -> CompressedResourceOp:
        """Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation."""

    def resource_rep_from_op(self) -> CompressedResourceOp:
        """Returns a compressed representation directly from the operator"""
        return self.__class__.resource_rep(**self.resource_params())
