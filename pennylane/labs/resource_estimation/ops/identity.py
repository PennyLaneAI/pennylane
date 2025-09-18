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

from pennylane.labs import resource_estimation as plre
from pennylane.labs.resource_estimation.resource_operator import (
    CompressedResourceOp,
    GateCount,
    ResourceOperator,
    resource_rep,
)

# pylint: disable=arguments-differ,no-self-use


class ResourceIdentity(ResourceOperator):
    r"""Resource class for the Identity gate.

    Args:
        wires (Iterable[Any], optional): wire label(s) that the identity acts on

    Resources:
        The Identity gate is treated as a free gate and thus it cannot be decomposed
        further. Requesting the resources of this gate returns an empty list.

    .. seealso:: :class:`~.Identity`

    **Example**

    The resources for this operation are computed using:

    >>> plre.ResourceIdentity.resource_decomp()
    []
    """

    num_wires = 1

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
        the Operator that are needed to compute the resources."""
        return CompressedResourceOp(cls, cls.num_wires, {})

    @classmethod
    def resource_decomp(cls, **kwargs) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator. Each object represents a quantum gate
        and the number of times it occurs in the decomposition.

        Resources:
            The Identity gate is treated as a free gate and thus it cannot be decomposed
            further. Requesting the resources of this gate returns an empty list.

        Returns:
            list: empty list
        """
        return []

    @classmethod
    def adjoint_resource_decomp(cls) -> list[GateCount]:
        r"""Returns a list representing the resources for the adjoint of the operator.

        Resources:
            This operation is self-adjoint, so the resources of the adjoint operation is the base operation.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
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
            The Identity gate acts trivially when controlled. The resources of this operation are
            the original (un-controlled) operation.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
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
            operation are the original operation.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        return [GateCount(cls.resource_rep())]


class ResourceGlobalPhase(ResourceOperator):
    r"""Resource class for the GlobalPhase gate.

    Args:
        wires (Iterable[Any], optional): the wires the operator acts on

    Resources:
        The GlobalPhase gate is treated as a free gate and thus it cannot be decomposed
        further. Requesting the resources of this gate returns an empty list.

    .. seealso:: :class:`~.GlobalPhase`

    **Example**

    The resources for this operation are computed using:

    >>> plre.ResourceGlobalPhase.resource_decomp()
    []

    """

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
        the Operator that are needed to compute the resources."""
        return CompressedResourceOp(cls, cls.num_wires, {})

    @classmethod
    def resource_decomp(cls, **kwargs) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator. Each object represents a quantum gate
        and the number of times it occurs in the decomposition.

        Resources:
            The GlobalPhase gate is treated as a free gate and thus it cannot be decomposed
            further. Requesting the resources of this gate returns an empty list.

        Returns:
            list: empty list
        """
        return []

    @classmethod
    def adjoint_resource_decomp(cls) -> list[GateCount]:
        r"""Returns a list representing the resources for the adjoint of the operator.

        Resources:
            The adjoint of a global phase operator changes the sign of the phase, thus
            the resources of the adjoint operation is the original operation.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
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
            list[GateCount]: A list of GateCount objects, where each object
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

        Resources:
            The resources are generated from the fact that a global phase controlled on a
            single qubit is equivalent to a local phase shift on that control qubit.
            This idea can be generalized to a multi-qubit global phase by introducing one
            'clean' auxilliary qubit which gets reset at the end of the computation. In this
            case, we sandwich the phase shift operation with two multi-controlled X gates.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        if ctrl_num_ctrl_wires == 1:
            gate_types = [GateCount(resource_rep(plre.ResourcePhaseShift))]

            if ctrl_num_ctrl_values:
                gate_types.append(GateCount(resource_rep(plre.ResourceX), 2))

            return gate_types

        ps = resource_rep(plre.ResourcePhaseShift)
        mcx = resource_rep(
            plre.ResourceMultiControlledX,
            {
                "num_ctrl_wires": ctrl_num_ctrl_wires,
                "num_ctrl_values": ctrl_num_ctrl_values,
            },
        )

        return [GateCount(ps), GateCount(mcx, 2)]
