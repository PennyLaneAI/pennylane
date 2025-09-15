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
r"""Resource operators for identity and global phase operations."""

from pennylane.exceptions import ResourcesUndefinedError

from pennylane.estimator.resource_operator import (
    CompressedResourceOp,
    GateCount,
    ResourceOperator,
)

# pylint: disable=arguments-differ


class Identity(ResourceOperator):
    r"""Resource class for the Identity gate.

    Args:
        wires (Iterable[Any], optional): wire label(s) that the identity acts on

    Resources:
        The Identity gate does not require any resources and thus it cannot be decomposed
        further. Requesting the resources of this gate returns an empty list.

    .. seealso:: The corresponding PennyLane operation :class:`~pennylane.Identity`.

    **Example**

    The resources for this operation can be requested using:

    >>> qml.estimator.Identity.resource_decomp()
    []
    """

    num_wires = 1

    def __init__(self, wires=None):
        """Initializes the ``Identity`` operator."""
        if wires is not None and not isinstance(wires, int):
            self.num_wires = len(wires)
        super().__init__(wires=wires)

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: Empty dictionary. The resources of this operation don't depend on any additional parameters.
        """
        return {}

    @classmethod
    def resource_rep(cls, **kwargs) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the operator that are needed to compute the resources."""
        return CompressedResourceOp(cls, cls.num_wires, {})

    @classmethod
    def resource_decomp(cls, **kwargs) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator. Each object represents a quantum gate
        and the number of times it occurs in the decomposition.

        Resources:
            The Identity gate does not require any resources and thus it cannot be decomposed
            further. Requesting the resources of this gate returns an empty list.

        Returns:
            list: empty list
        """
        return []

    @classmethod
    def adjoint_resource_decomp(cls) -> list[GateCount]:
        r"""Returns a list representing the resources for the adjoint of the operator.

        Resources:
            This operation is self-adjoint, so the resources of the adjoint operation are same as the base operation.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        return [GateCount(cls.resource_rep())]

    @classmethod
    def controlled_resource_decomp(
        cls,
        ctrl_num_ctrl_wires: int,
        ctrl_num_ctrl_values: int,
    ) -> list[GateCount]:
        r"""Returns a list representing the resources for a controlled version of the operator.

        Args:
            num_ctrl_wires (int): the number of qubits the operation is controlled on
            num_ctrl_values (int): The number of control qubits, that are triggered when in the :math:`|0\rangle` state.

        Resources:
            The Identity gate acts trivially when controlled. The resources of this operation are same as
            the original (un-controlled) operation.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        return [GateCount(cls.resource_rep())]

    @classmethod
    def pow_resource_decomp(cls, pow_z) -> list[GateCount]:
        r"""Returns a list representing the resources for an operator raised to a power.

        Args:
            pow_z (int): the power that the operator is being raised to

        Resources:
            The Identity gate acts trivially when raised to a power. The resources of this
            operation are same as the original operation.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        return [GateCount(cls.resource_rep())]


class GlobalPhase(ResourceOperator):
    r"""Resource class for the GlobalPhase gate.

    Args:
        wires (Iterable[Any], optional): the wires the operator acts on

    Resources:
        The GlobalPhase gate does not require any resources and thus it cannot be decomposed
        further. Requesting the resources of this gate returns an empty list.

    .. seealso:: The corresponding PennyLane operation :class:`~.pennylane.GlobalPhase`.

    **Example**

    The resources for this operation can be requested using:

    >>> qml.estimator.GlobalPhase.resource_decomp()
    []

    """

    num_wires = 1

    def __init__(self, wires=None):
        """Initializes the ``GlobalPhase`` operator."""
        if wires is not None and not isinstance(wires, int):
            self.num_wires = len(wires)
        super().__init__(wires=wires)

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: Empty dictionary. The resources of this operation don't depend on any additional parameters.
        """
        return {}

    @classmethod
    def resource_rep(cls, **kwargs) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the operator that are needed to compute the resources."""
        return CompressedResourceOp(cls, cls.num_wires, {})

    @classmethod
    def resource_decomp(cls, **kwargs) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator. Each object represents a quantum gate
        and the number of times it occurs in the decomposition.

        Resources:
            The GlobalPhase gate does not require any resources and thus it cannot be decomposed
            further. Requesting the resources of this gate returns an empty list.

        Returns:
            list: empty list
        """
        return []

    @classmethod
    def adjoint_resource_decomp(cls) -> list[GateCount]:
        r"""Returns a list representing the resources for the adjoint of the operator.

        Resources:
            The adjoint of GlobalPhase operator changes the sign of the phase, thus
            the resources of the adjoint operation are same as the original operation.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        return [GateCount(cls.resource_rep())]

    @classmethod
    def pow_resource_decomp(cls, pow_z) -> list[GateCount]:
        r"""Returns a list representing the resources for an operator raised to a power.

        Args:
            pow_z (int): the power that the operator is being raised to

        Resources:
            Taking arbitrary powers of a global phase produces a sum of global phases.
            The resources simplify to just one total global phase operator.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        return [GateCount(cls.resource_rep())]

    @classmethod
    def controlled_resource_decomp(
        cls,
        ctrl_num_ctrl_wires: int,
        ctrl_num_ctrl_values: int,
    ) -> list[GateCount]:
        r"""Returns a list representing the resources for a controlled version of the operator.

        Args:
            ctrl_num_ctrl_wires (int): the number of qubits the operation is controlled on
            ctrl_num_ctrl_values (int): The number of control qubits, that are controlled when
                in the :math:`|0\rangle` state.

        Raises:
            ResourcesUndefinedError: Controlled version of this gate is not defined.
        """
        raise ResourcesUndefinedError
