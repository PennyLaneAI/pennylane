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
r"""Resource operators for identity operations."""
from typing import Dict

import pennylane as qml
import pennylane.labs.resource_estimation as re

# pylint: disable=arguments-differ,no-self-use,too-many-ancestors


class ResourceIdentity(qml.Identity, re.ResourceOperator):
    r"""Resource class for the Identity gate.

    Args:
        wires (Iterable[Any] or Any): Wire label(s) that the identity acts on.
        id (str): custom label given to an operator instance,
            can be useful for some applications where the instance has to be identified.

    Resources:
        The Identity gate is treated as a free gate and thus it cannot be decomposed
        further. Requesting the resources of this gate returns an empty dictionary.

    .. seealso:: :class:`~.Identity`

    """

    @staticmethod
    def _resource_decomp(*args, **kwargs) -> Dict[re.CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources of the operator. The
        keys are the operators and the associated values are the counts.

        Resources:
            The Identity gate is treated as a free gate and thus it cannot be decomposed
            further. Requesting the resources of this gate returns an empty dictionary.

        Returns:
            dict: empty dictionary
        """
        return {}

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: Empty dictionary. The resources of this operation don't depend on any additional parameters.
        """
        return {}

    @classmethod
    def resource_rep(cls, **kwargs) -> re.CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation."""
        return re.CompressedResourceOp(cls, {})

    @classmethod
    def adjoint_resource_decomp(cls) -> Dict[re.CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources for the adjoint of the operator.

        Resources:
            This operation is self-adjoint, so the resources of the adjoint operation is also an empty dictionary.

        Returns:
            Dict[CompressedResourceOp, int]: The keys are the operators and the associated values are the counts.
        """
        return {cls.resource_rep(): 1}

    @classmethod
    def controlled_resource_decomp(
        cls, num_ctrl_wires, num_ctrl_values, num_work_wires
    ) -> Dict[re.CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources for a controlled version of the operator.

        Args:
            num_ctrl_wires (int): the number of qubits the operation is controlled on
            num_ctrl_values (int): The number of control qubits, that are triggered when in the :math:`|0\rangle` state.
            num_work_wires (int): the number of additional qubits that can be used for decomposition

        Resources:
            The Identity gate acts trivially when controlled. The resources of this operation are
            the original (un-controlled) operation.

        Returns:
            Dict[CompressedResourceOp, int]: The keys are the operators and the associated values are the counts.
        """
        return {cls.resource_rep(): 1}

    @classmethod
    def pow_resource_decomp(cls, z) -> Dict[re.CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources for an operator raised to a power.

        Args:
            z (int): the power that the operator is being raised to

        Resources:
            The Identity gate acts trivially when raised to a power. The resources of this
            operation are the original operation.

        Returns:
            Dict[CompressedResourceOp, int]: The keys are the operators and the associated values are the counts.
        """
        return {cls.resource_rep(): 1}


class ResourceGlobalPhase(qml.GlobalPhase, re.ResourceOperator):
    r"""Resource class for the GlobalPhase gate.

    Args:
        phi (TensorLike): the global phase
        wires (Iterable[Any] or Any): unused argument - the operator is applied to all wires
        id (str): custom label given to an operator instance,
            can be useful for some applications where the instance has to be identified.

    Resources:
        The GlobalPhase gate is treated as a free gate and thus it cannot be decomposed
        further. Requesting the resources of this gate returns an empty dictionary.

    .. seealso:: :class:`~.GlobalPhase`

    """

    @staticmethod
    def _resource_decomp(*args, **kwargs) -> Dict[re.CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources of the operator. The
        keys are the operators and the associated values are the counts.

        Resources:
            The GlobalPhase gate is treated as a free gate and thus it cannot be decomposed
            further. Requesting the resources of this gate returns an empty dictionary.

        Returns:
            dict: empty dictionary
        """
        return {}

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: Empty dictionary. The resources of this operation don't depend on any additional parameters.
        """
        return {}

    @classmethod
    def resource_rep(cls, **kwargs) -> re.CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation."""
        return re.CompressedResourceOp(cls, {})

    @staticmethod
    def adjoint_resource_decomp() -> Dict[re.CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources for the adjoint of the operator.

        Resources:
            The adjoint of a global phase operator changes the sign of the phase, thus
            the resources of the adjoint operation is the original operation.

        Returns:
            Dict[CompressedResourceOp, int]: The keys are the operators and the associated values are the counts.
        """
        return {re.ResourceGlobalPhase.resource_rep(): 1}

    @staticmethod
    def controlled_resource_decomp(
        num_ctrl_wires, num_ctrl_values, num_work_wires
    ) -> Dict[re.CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources for a controlled version of the operator.

        Args:
            num_ctrl_wires (int): the number of qubits the operation is controlled on
            num_ctrl_values (int): the number of control qubits, that are controlled when in the :math:`|0\rangle` state
            num_work_wires (int): the number of additional qubits that can be used for decomposition

        Resources:
            The resources are generated from the fact that a global phase controlled on a
            single qubit is equivalent to a local phase shift on that control qubit.

            This idea can be generalized to a multi-qubit global phase by introducing one
            'clean' auxilliary qubit which gets reset at the end of the computation. In this
            case, we sandwich the phase shift operation with two multi-controlled X gates.

        Returns:
            Dict[CompressedResourceOp, int]: The keys are the operators and the associated values are the counts.
        """
        if num_ctrl_wires == 1:
            gate_types = {re.ResourcePhaseShift.resource_rep(): 1}

            if num_ctrl_values:
                gate_types[re.ResourceX.resource_rep()] = 2

            return gate_types

        ps = re.ResourcePhaseShift.resource_rep()
        mcx = re.ResourceMultiControlledX.resource_rep(
            num_ctrl_wires=num_ctrl_wires,
            num_ctrl_values=num_ctrl_values,
            num_work_wires=num_work_wires,
        )

        return {ps: 1, mcx: 2}

    @staticmethod
    def pow_resource_decomp(z) -> Dict[re.CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources for an operator raised to a power.

        Args:
            z (int): the power that the operator is being raised to

        Resources:
            Taking arbitrary powers of a global phase produces a sum of global phases.
            The resources simplify to just one total global phase operator.

        Returns:
            Dict[CompressedResourceOp, int]: The keys are the operators and the associated values are the counts.
        """
        return {re.ResourceGlobalPhase.resource_rep(): 1}
