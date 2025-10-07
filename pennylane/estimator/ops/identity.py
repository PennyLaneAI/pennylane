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

import pennylane.estimator as qre
from pennylane.estimator.resource_operator import (
    CompressedResourceOp,
    GateCount,
    ResourceOperator,
    resource_rep,
)
from pennylane.wires import Wires

# pylint: disable=arguments-differ


class Identity(ResourceOperator):
    r"""Resource class for the Identity gate.

    Args:
        wires (Iterable[Any] | None): wire label(s) that the identity acts on

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
        if wires:
            self.num_wires = len(Wires(wires))
        else:
            self.num_wires = 1
        super().__init__(wires=wires)

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: Empty dictionary. The resources of this operation don't depend on any additional parameters.
        """
        return {}

    @classmethod
    def resource_rep(cls) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the operator that are needed to compute the resources."""
        return CompressedResourceOp(cls, cls.num_wires, {})

    @classmethod
    def resource_decomp(cls) -> list[GateCount]:
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
    def adjoint_resource_decomp(cls, target_resource_params: dict | None = None) -> list[GateCount]:
        r"""Returns a list representing the resources for the adjoint of the operator.

        Args:
            target_resource_params (dict | None): A dictionary containing the resource parameters
                of the target operator.

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
        num_ctrl_wires: int,
        num_zero_ctrl: int,
        target_resource_params: dict | None = None,
    ) -> list[GateCount]:
        r"""Returns a list representing the resources for a controlled version of the operator.

        Args:
            num_ctrl_wires (int): the number of qubits the operation is controlled on
            num_zero_ctrl (int): The number of control qubits, that are triggered when in the :math:`|0\rangle` state.
            target_resource_params (dict | None): A dictionary containing the resource parameters
                of the target operator.

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
    def pow_resource_decomp(
        cls, pow_z: int, target_resource_params: dict | None = None
    ) -> list[GateCount]:
        r"""Returns a list representing the resources for an operator raised to a power.

        Args:
            pow_z (int): the power that the operator is being raised to
            target_resource_params (dict | None): A dictionary containing the resource parameters
                of the target operator.

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
        wires (Iterable[Any] | None): the wires the operator acts on

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
        if wires:
            self.num_wires = len(Wires(wires))
        else:
            self.num_wires = 1
        super().__init__(wires=wires)

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: Empty dictionary. The resources of this operation don't depend on any additional parameters.
        """
        return {}

    @classmethod
    def resource_rep(cls) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the operator that are needed to compute the resources."""
        return CompressedResourceOp(cls, cls.num_wires, {})

    @classmethod
    def resource_decomp(cls) -> list[GateCount]:
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
    def adjoint_resource_decomp(cls, target_resource_params: dict | None = None) -> list[GateCount]:
        r"""Returns a list representing the resources for the adjoint of the operator.

        Args:
            target_resource_params (dict | None): A dictionary containing the resource parameters
                of the target operator.

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
    def pow_resource_decomp(
        cls, pow_z: int, target_resource_params: dict | None = None
    ) -> list[GateCount]:
        r"""Returns a list representing the resources for an operator raised to a power.

        Args:
            pow_z (int): the power that the operator is being raised to
            target_resource_params (dict | None): A dictionary containing the resource parameters
                of the target operator.

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
        num_ctrl_wires: int,
        num_zero_ctrl: int,
        target_resource_params: dict | None = None,
    ) -> list[GateCount]:
        r"""Returns a list representing the resources for a controlled version of the operator.

        Args:
            num_ctrl_wires (int): the number of qubits the operation is controlled on
            num_zero_ctrl (int): The number of control qubits that are controlled when
                in the :math:`|0\rangle` state.
            target_resource_params (dict | None): A dictionary containing the resource parameters
                of the target operator.

        Resources:
            The resources are generated from the fact that a global phase controlled on a
            single qubit is equivalent to a local phase shift on that control qubit.
            This idea can be generalized to a multi-qubit global phase by introducing one
            auxiliary qubit in a `zeroed` state which is reset at the end of the computation. In this
            case, we sandwich the phase shift operation with two multi-controlled ``X`` gates.

        Returns:
            list[`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        if num_ctrl_wires == 1:
            gate_types = [GateCount(resource_rep(qre.PhaseShift))]

            if num_zero_ctrl:
                gate_types.append(GateCount(resource_rep(qre.X), 2))

            return gate_types

        ps = resource_rep(qre.PhaseShift)
        mcx = resource_rep(
            qre.MultiControlledX,
            {
                "num_ctrl_wires": num_ctrl_wires,
                "num_zero_ctrl": num_zero_ctrl,
            },
        )

        return [GateCount(ps), GateCount(mcx, 2)]
