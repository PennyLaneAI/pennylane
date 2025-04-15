# Copyright 2025 Xanadu Quantum Technologies Inc.

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

# pylint: disable=unused-argument


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
                def _resource_decomp(num_wires) -> Dict[CompressedResourceOp, int]:
                    gate_types = {}

                    hadamard = ResourceHadamard.resource_rep()
                    swap = ResourceSWAP.resource_rep()
                    ctrl_phase_shift = ResourceControlledPhaseShift.resource_rep()

                    gate_types[hadamard] = num_wires
                    gate_types[swap] = num_wires // 2
                    gate_types[ctrl_phase_shift] = num_wires*(num_wires - 1) // 2

                    return gate_types

                @property
                def resource_params(self, num_wires) -> dict:
                    return {"num_wires": num_wires}

                @classmethod
                def resource_rep(cls, num_wires) -> CompressedResourceOp:
                    params = {"num_wires": num_wires}
                    return CompressedResourceOp(cls, params)

        Which can be instantiated as a normal operation, but now contains the resources:

        .. code-block:: bash

            >>> op = ResourceQFT(range(3))
            >>> op.resources(**op.resource_params)
            {Hadamard: 3, SWAP: 1, ControlledPhaseShift: 3}

    """

    @staticmethod
    @abstractmethod
    def _resource_decomp(*args, **kwargs) -> Dict[CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources of the operator. The
        keys are the operators and the associated values are the counts."""

    @classmethod
    def resources(cls, *args, **kwargs) -> Dict[CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources of the operator. The
        keys are the operators and the associated values are the counts."""
        return cls._resource_decomp(*args, **kwargs)

    @classmethod
    def set_resources(cls, new_func: Callable) -> None:
        """Set a custom resource method."""
        cls.resources = new_func

    @property
    @abstractmethod
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to
        compute a compressed representation"""

    @classmethod
    @abstractmethod
    def resource_rep(cls, *args, **kwargs) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation."""

    def resource_rep_from_op(self) -> CompressedResourceOp:
        r"""Returns a compressed representation directly from the operator"""
        return self.__class__.resource_rep(**self.resource_params)

    @classmethod
    def adjoint_resource_decomp(cls, *args, **kwargs) -> Dict[CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources for the adjoint of the operator.

        Raises:
            ResourcesNotDefined: no resources implemented by default

        Returns:
            Dict[CompressedResourceOp, int]: The keys are the operators and the associated values are the counts.
        """
        raise ResourcesNotDefined

    @classmethod
    def controlled_resource_decomp(
        cls, num_ctrl_wires: int, num_ctrl_values: int, num_work_wires: int, *args, **kwargs
    ) -> Dict[CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources for a controlled version of the operator.

        Args:
            num_ctrl_wires (int): the number of qubits the operation is controlled on
            num_ctrl_values (int): the number of control qubits, that are controlled when in the :math:`|0\rangle` state
            num_work_wires (int): the number of additional qubits that can be used for decomposition

        Raises:
            ResourcesNotDefined: no resources implemented by default

        Returns:
            Dict[CompressedResourceOp, int]: The keys are the operators and the associated values are the counts.
        """

        raise ResourcesNotDefined

    @classmethod
    def pow_resource_decomp(cls, z: int, *args, **kwargs) -> Dict[CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources for an operator raised to a power.

        Args:
            z (int): the power that the operator is being raised to

        Raises:
            ResourcesNotDefined: no resources implemented by default

        Returns:
            Dict[CompressedResourceOp, int]: The keys are the operators and the associated values are the counts.
        """
        raise ResourcesNotDefined

    @classmethod
    def exp_resource_decomp(
        cls, scalar: complex, num_steps: int, *args, **kwargs
    ) -> Dict[CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources for the exponentiated operator.

        Args:
            scalar (complex): complex coefficient of the operator in the exponent
            num_steps (int): number of trotter steps to use when decomposing the expoentiated operator

        Raises:
            ResourcesNotDefined: no resources implemented by default

        Returns:
            Dict[CompressedResourceOp, int]: The keys are the operators and the associated values are the counts.
        """
        raise ResourcesNotDefined

    @classmethod
    def tracking_name(cls, *args, **kwargs) -> str:
        r"""Returns a name used to track the operator during resource estimation."""
        return cls.__name__.replace("Resource", "")

    def tracking_name_from_op(self) -> str:
        r"""Returns the tracking name built with the operator's parameters."""
        return self.__class__.tracking_name(**self.resource_params)


class ResourcesNotDefined(Exception):
    r"""Exception to be raised when a ``ResourceOperator`` does not implement _resource_decomp"""
