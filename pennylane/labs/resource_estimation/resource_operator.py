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
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Callable, Dict, List

from pennylane.queuing import QueuingManager
from pennylane.wires import Wires

# if TYPE_CHECKING:
#     from pennylane.labs.resource_estimation import CompressedResourceOp

# pylint: disable=unused-argument


class ResourceOperator(ABC):
    r"""Abstract class that defines the methods a PennyLane Operator
    must implement in order to be used for resource estimation.
    """

    num_wires = 0
    _queue_category = "_resource_op"

    def __init__(self, *args, wires=None, **kwargs) -> None:
        self.wires = None
        if wires:
            self.wires = Wires(wires)
            self.num_wires = len(self.wires)

        self.queue()
        super().__init__()

    def queue(self, context: QueuingManager = QueuingManager):
        """Append the operator to the Operator queue."""
        context.append(self)
        return self

    @property
    @abstractmethod
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to
        compute a compressed representation"""

    @classmethod
    @abstractmethod
    def resource_rep(cls, *args, **kwargs):
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation."""

    def resource_rep_from_op(self):
        r"""Returns a compressed representation directly from the operator"""
        return self.__class__.resource_rep(**self.resource_params)

    @classmethod
    @abstractmethod
    def default_resource_decomp(*args, **kwargs) -> List:
        r"""Returns a list of actions that define the resources of the operator."""

    @classmethod
    def resource_decomp(cls, *args, **kwargs) -> List:
        r"""Returns a list of actions that define the resources of the operator."""
        return cls.default_resource_decomp(*args, **kwargs)

    @classmethod
    def default_adjoint_resource_decomp(cls, *args, **kwargs) -> List:
        r"""Returns a list representing the resources for the adjoint of the operator."""
        raise ResourcesNotDefined

    @classmethod
    def adjoint_resource_decomp(cls, *args, **kwargs) -> List:
        r"""Returns a list of actions that define the resources of the operator."""
        return cls.default_adjoint_resource_decomp(*args, **kwargs)

    @classmethod
    def default_controlled_resource_decomp(
        cls, ctrl_num_ctrl_wires: int, ctrl_num_ctrl_values: int, *args, **kwargs
    ) -> List:
        r"""Returns a list representing the resources for a controlled version of the operator.

        Args:
            ctrl_num_ctrl_wires (int): the number of qubits the 
                operation is controlled on
            ctrl_num_ctrl_values (int): the number of control qubits, that are 
		        controlled when in the :math:`|0\rangle` state
        """
        raise ResourcesNotDefined

    @classmethod
    def controlled_resource_decomp(
		cls, ctrl_num_ctrl_wires: int, ctrl_num_ctrl_values: int, *args, **kwargs
	) -> List:
        r"""Returns a list representing the resources for a controlled version of the operator.

        Args:
            ctrl_num_ctrl_wires (int): the number of qubits the 
                operation is controlled on
            ctrl_num_ctrl_values (int): the number of control qubits, that are 
		        controlled when in the :math:`|0\rangle` state
        """
        return cls.default_controlled_resource_decomp(
            ctrl_num_ctrl_wires, ctrl_num_ctrl_values, *args, **kwargs
		)
    
    @classmethod
    def default_pow_resource_decomp(cls, pow_z: int, *args, **kwargs) -> List:
        r"""Returns a list representing the resources for an operator 
        raised to a power.

        Args:
            pow_z (int): the power that the operator is being raised to
        """
        raise ResourcesNotDefined

    @classmethod
    def pow_resource_decomp(cls, pow_z, *args, **kwargs) -> List:
        r"""Returns a list representing the resources for an operator 
        raised to a power.

        Args:
            pow_z (int): the power that the operator is being raised to
        """
        return cls.default_pow_resource_decomp(pow_z, *args, **kwargs)

    @classmethod
    def set_resources(cls, new_func: Callable, override_type: str = "base"):
        """Set a custom resource method."""
        if override_type == "base":
            cls.resource_decomp = new_func
        if override_type == "pow":
            cls.pow_resource_decomp = new_func
        if override_type == "adj":
            cls.adjoint_resource_decomp = new_func
        if override_type == "ctrl":
            cls.controlled_resource_decomp = new_func
        return

    def __repr__(self) -> str:
        str_rep = self.__class__.__name__ + "(" + str(self.resource_params) + ")"
        return str_rep

    # def __mul__(self, scalar: int):
    #     assert isinstance(scalar, int)
    #     gate_types = defaultdict(int, {self.resource_rep_from_op(): scalar})
    #     qubit_manager = QubitManager(0)
    #     qubit_manager._logic_qubit_counts = self.num_wires

    #     from pennylane.labs.resource_estimation.resource_container import Resources
    #     return Resources(qubit_manager, gate_types)
    
    # def __matmul__(self, scalar: int):
    #     assert isinstance(scalar, int)
    #     gate_types = defaultdict(int, {self.resource_rep_from_op(): scalar})
    #     qubit_manager = QubitManager(0)
    #     qubit_manager._logic_qubit_counts = scalar * self.num_wires

    #     from pennylane.labs.resource_estimation.resource_container import Resources
    #     return Resources(qubit_manager, gate_types)

    # __rmul__ = __mul__
    # __rmatmul__ = __matmul__

    @classmethod
    def tracking_name(cls, *args, **kwargs) -> str:
        r"""Returns a name used to track the operator during resource estimation."""
        return cls.__name__.replace("Resource", "")

    def tracking_name_from_op(self) -> str:
        r"""Returns the tracking name built with the operator's parameters."""
        return self.__class__.tracking_name(**self.resource_params)


class ResourcesNotDefined(Exception):
    r"""Exception to be raised when a ``ResourceOperator`` does not implement _resource_decomp"""
