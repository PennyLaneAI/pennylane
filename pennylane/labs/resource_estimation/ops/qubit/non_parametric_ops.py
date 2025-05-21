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
r"""Resource operators for non parametric single qubit operations."""
from typing import Dict

import pennylane as qml
import pennylane.labs.resource_estimation as re
from pennylane.labs.resource_estimation.qubit_manager import GrabWires, FreeWires
from pennylane.labs.resource_estimation.resource_operator import (
    GateCount,
    ResourceOperator,
    CompressedResourceOp,
)

# pylint: disable=arguments-differ


class ResourceHadamard(ResourceOperator):
    r"""Resource class for the Hadamard gate.

    Args:
        wires (Sequence[int] or int): the wire the operation acts on

    Resources:
        The Hadamard gate is treated as a fundamental gate and thus it cannot be decomposed
        further. Requesting the resources of this gate raises a :code:`ResourcesNotDefined` error.

    .. seealso:: :class:`~.Hadamard`

    """

    num_wires = 1

    @staticmethod
    def _resource_decomp(**kwargs) -> Dict[CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources of the operator. The
        keys are the operators and the associated values are the counts.

        Resources:
            The Hadamard gate is treated as a fundamental gate and thus it cannot be decomposed
            further. Requesting the resources of this gate raises a :code:`ResourcesNotDefined` error.

        Raises:
            ResourcesNotDefined: This gate is fundamental, no further decomposition defined.
        """
        raise re.ResourcesNotDefined

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
        the Operator that are needed to compute a resource estimation."""
        return CompressedResourceOp(cls, {})

    @classmethod
    def adjoint_resource_decomp(cls) -> Dict[CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources for the adjoint of the operator.

        Resources:
            This operation is self-adjoint, so the resources of the adjoint operation results
            in the original operation.

        Returns:
            Dict[CompressedResourceOp, int]: The keys are the operators and the associated
                values are the counts.
        """
        return [GateCount(cls.resource_rep(), 1)]

    @staticmethod
    def controlled_resource_decomp(
        num_ctrl_wires, num_ctrl_values, num_work_wires
    ) -> Dict[CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources for a controlled version of the operator.

        Args:
            num_ctrl_wires (int): the number of qubits the operation is controlled on
            num_ctrl_values (int): the number of control qubits, that are controlled when in the :math:`|0\rangle` state
            num_work_wires (int): the number of additional qubits that can be used for decomposition

        Resources:
            For a single control wire, the cost is a single instance of :class:`~.ResourceCH`.
            Two additional :class:`~.ResourceX` gates are used to flip the control qubit if
            it is zero-controlled.

            In the case where multiple controlled wires are provided, the resources are derived from
            the following identities (as presented in this `blog post <https://quantumcomputing.stackexchange.com/questions/15734/how-to-construct-a-controlled-hadamard-gate-using-single-qubit-gates-and-control>`_):

            .. math::

                \begin{align}
                    \hat{H} &= \hat{R}_{y}(\frac{\pi}{4}) \cdot \hat{Z}  \cdot \hat{R}_{y}(\frac{-\pi}{4}), \\
                    \hat{Z} &= \hat{H} \cdot \hat{X}  \cdot \hat{H}.
                \end{align}

            Specifically, the resources are given by two :class:`~.ResourceRY` gates, two
            :class:`~.ResourceHadamard` gates and a :class:`~.ResourceX` gate. By replacing the
            :class:`~.ResourceX` gate with :class:`~.ResourceMultiControlledX` gate, we obtain a
            controlled-version of this identity.

        Returns:
            Dict[CompressedResourceOp, int]: The keys are the operators and the associated
                values are the counts.
        """
        if num_ctrl_wires == 1:
            gate_lst = [GateCount(re.ResourceCH.resource_rep(), 1)]

            if num_ctrl_values:
                gate_lst.append(GateCount(re.ResourceX.resource_rep(), 2))

            return gate_lst

        gate_lst = []

        ry = re.ResourceRY.resource_rep()
        h = re.ResourceHadamard.resource_rep()
        mcx = re.ResourceMultiControlledX.resource_rep(
            num_ctrl_wires=num_ctrl_wires,
            num_ctrl_values=num_ctrl_values,
            num_work_wires=num_work_wires,
        )

        gate_lst.append(GateCount(h, 2))
        gate_lst.append(GateCount(ry, 2))
        gate_lst.append(GateCount(mcx, 1))
        return gate_lst

    @classmethod
    def pow_resource_decomp(cls, z) -> Dict[CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources for an operator raised to a power.

        Args:
            z (int): the power that the operator is being raised to

        Resources:
            The Hadamard gate raised to even powers produces identity and raised
            to odd powers it produces itself.

        Returns:
            Dict[CompressedResourceOp, int]: The keys are the operators and the associated
                values are the counts.
        """
        if z % 2 == 0:
            return [GateCount(re.ResourceIdentity.resource_rep())]
        return [GateCount(cls.resource_rep())]


class ResourceS(ResourceOperator):
    r"""Resource class for the S-gate.

    Args:
        wires (Sequence[int] or int): the wire the operation acts on

    Resources:
        The S-gate decomposes into two T-gates.

    .. seealso:: :class:`~.S`

    **Example**

    The resources for this operation are computed using:

    >>> re.ResourceS.resources()
    {T: 2}
    """

    num_wires = 1

    @staticmethod
    def _resource_decomp(**kwargs) -> Dict[CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources of the operator. The
        keys are the operators and the associated values are the counts.

        Resources:
            The S-gate decomposes into two T-gates.
        """
        t = ResourceT.resource_rep()
        return [GateCount(t, 2)]

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
        the Operator that are needed to compute a resource estimation."""
        return CompressedResourceOp(cls, {})

    @classmethod
    def adjoint_resource_decomp(cls) -> Dict[CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources for the adjoint of the operator.

        Resources:
            The adjoint of the S-gate is equivalent to the S-gate raised to the third power.
            The resources are defined as three instances of the S-gate.

        Returns:
            Dict[CompressedResourceOp, int]: The keys are the operators and the associated
                values are the counts.
        """
        return [GateCount(cls.resource_rep(), 3)]

    @staticmethod
    def controlled_resource_decomp(num_ctrl_wires, num_ctrl_values, num_work_wires):
        r"""Returns a dictionary representing the resources for a controlled version of the operator.

        Args:
            num_ctrl_wires (int): the number of qubits the operation is controlled on
            num_ctrl_values (int): the number of control qubits, that are controlled when in the :math:`|0\rangle` state
            num_work_wires (int): the number of additional qubits that can be used for decomposition

        Resources:
            The S-gate is equivalent to the PhaseShift gate for some fixed phase. Given a single
            control wire, the cost is therefore a single instance of
            :class:`~.ResourceControlledPhaseShift`. Two additional :class:`~.ResourceX` gates are
            used to flip the control qubit if it is zero-controlled.

            In the case where multiple controlled wires are provided, we can collapse the control
            wires by introducing one 'clean' auxilliary qubit (which gets reset at the end).
            In this case the cost increases by two additional :class:`~.ResourceMultiControlledX` gates,
            as described in (Lemma 7.11) `Barenco et al. <https://arxiv.org/pdf/quant-ph/9503016>`_.

        Returns:
            Dict[CompressedResourceOp, int]: The keys are the operators and the associated
                values are the counts.
        """
        if num_ctrl_wires == 1:
            gate_lst = [GateCount(re.ResourceControlledPhaseShift.resource_rep(), 1)]

            if num_ctrl_values:
                gate_lst.append(GateCount(re.ResourceX.resource_rep(), 2))

            return gate_lst

        cs = re.ResourceControlledPhaseShift.resource_rep()
        mcx = re.ResourceMultiControlledX.resource_rep(
            num_ctrl_wires=num_ctrl_wires,
            num_ctrl_values=num_ctrl_values,
            num_work_wires=num_work_wires,
        )
        return [GateCount(cs, 1), GateCount(mcx, 2)]

    @classmethod
    def pow_resource_decomp(cls, z) -> Dict[CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources for an operator raised to a power.

        Args:
            z (int): the power that the operator is being raised to

        Resources:
            The S-gate, when raised to a power which is a multiple of four, produces identity.
            The cost of raising to an arbitrary integer power :math:`z` is given by
            :math:`z \mod 4` instances of the S-gate.

        Returns:
            Dict[CompressedResourceOp, int]: The keys are the operators and the associated
                values are the counts.
        """
        if (mod_4 := z % 4) == 0:
            return [GateCount(re.ResourceIdentity.resource_rep())]
        return [GateCount(cls.resource_rep(), mod_4)]


class ResourceSWAP(ResourceOperator):
    r"""Resource class for the SWAP gate.

    Args:
        wires (Sequence[int]): the wires the operation acts on

    Resources:
        The resources come from the following identity expressing SWAP as the product of
        three :class:`~.CNOT` gates:

        .. math::

            SWAP = \begin{bmatrix}
                        1 & 0 & 0 & 0 \\
                        0 & 0 & 1 & 0\\
                        0 & 1 & 0 & 0\\
                        0 & 0 & 0 & 1
                    \end{bmatrix}
            =  \begin{bmatrix}
                    1 & 0 & 0 & 0 \\
                    0 & 1 & 0 & 0\\
                    0 & 0 & 0 & 1\\
                    0 & 0 & 1 & 0
                \end{bmatrix}
                \begin{bmatrix}
                    1 & 0 & 0 & 0 \\
                    0 & 0 & 0 & 1\\
                    0 & 0 & 1 & 0\\
                    0 & 1 & 0 & 0
                \end{bmatrix}
                \begin{bmatrix}
                    1 & 0 & 0 & 0 \\
                    0 & 1 & 0 & 0\\
                    0 & 0 & 0 & 1\\
                    0 & 0 & 1 & 0
            \end{bmatrix}.

    .. seealso:: :class:`~.SWAP`

    **Example**

    The resources for this operation are computed using:

    >>> re.ResourceSWAP.resources()
    {CNOT: 3}
    """

    num_wires = 2

    @staticmethod
    def _resource_decomp(**kwargs) -> Dict[CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources of the operator. The
        keys are the operators and the associated values are the counts.

        Resources:
            The resources come from the following identity expressing SWAP as the product of
            three CNOT gates:

            .. math::

                SWAP = \begin{bmatrix}
                            1 & 0 & 0 & 0 \\
                            0 & 0 & 1 & 0\\
                            0 & 1 & 0 & 0\\
                            0 & 0 & 0 & 1
                        \end{bmatrix}
                =  \begin{bmatrix}
                        1 & 0 & 0 & 0 \\
                        0 & 1 & 0 & 0\\
                        0 & 0 & 0 & 1\\
                        0 & 0 & 1 & 0
                    \end{bmatrix}
                    \begin{bmatrix}
                        1 & 0 & 0 & 0 \\
                        0 & 0 & 0 & 1\\
                        0 & 0 & 1 & 0\\
                        0 & 1 & 0 & 0
                    \end{bmatrix}
                    \begin{bmatrix}
                        1 & 0 & 0 & 0 \\
                        0 & 1 & 0 & 0\\
                        0 & 0 & 0 & 1\\
                        0 & 0 & 1 & 0
                \end{bmatrix}.
        """
        cnot = re.ResourceCNOT.resource_rep()

        return [GateCount(cnot, 3)]

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
        the Operator that are needed to compute a resource estimation."""
        return CompressedResourceOp(cls, {})

    @classmethod
    def adjoint_resource_decomp(cls) -> Dict[CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources for the adjoint of the operator.

        Resources:
            This operation is self-adjoint, so the resources of the adjoint operation results
            in the original operation.

        Returns:
            Dict[CompressedResourceOp, int]: The keys are the operators and the associated
                values are the counts.
        """
        return [GateCount(cls.resource_rep())]

    @staticmethod
    def controlled_resource_decomp(
        num_ctrl_wires, num_ctrl_values, num_work_wires
    ) -> Dict[CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources for a controlled version of the operator.

        Args:
            num_ctrl_wires (int): the number of qubits the operation is controlled on
            num_ctrl_values (int): the number of control qubits, that are controlled when in the :math:`|0\rangle` state
            num_work_wires (int): the number of additional qubits that can be used for decomposition

        Resources:
            For a single control wire, the cost is a single instance of :class:`~.ResourceCSWAP`.
            Two additional :class:`~.ResourceX` gates are used to flip the control qubit if
            it is zero-controlled.

            In the case where multiple controlled wires are provided, the resources are given by
            two :class:`~.ResourceCNOT` gates and one :class:`~.ResourceMultiControlledX` gate. This
            is because of the symmetric resource decomposition of the SWAP gate. By controlling on
            the middle CNOT gate, we obtain the required controlled operation.

        Returns:
            Dict[CompressedResourceOp, int]: The keys are the operators and the associated
                values are the counts.
        """
        if num_ctrl_wires == 1:
            gate_types = [GateCount(re.ResourceCSWAP.resource_rep())]

            if num_ctrl_values:
                gate_types.append(GateCount(re.ResourceX.resource_rep(), 2))

            return gate_types

        cnot = re.ResourceCNOT.resource_rep()
        mcx = re.ResourceMultiControlledX.resource_rep(
            num_ctrl_wires=num_ctrl_wires,
            num_ctrl_values=num_ctrl_values,
            num_work_wires=num_work_wires,
        )
        return [GateCount(cnot, 2), GateCount(mcx)]

    @classmethod
    def pow_resource_decomp(cls, z) -> Dict[CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources for an operator raised to a power.

        Args:
            z (int): the power that the operator is being raised to

        Resources:
            The SWAP gate raised to even powers produces identity and raised
            to odd powers it produces itself.

        Returns:
            Dict[CompressedResourceOp, int]: The keys are the operators and the associated
                values are the counts.
        """
        if z % 2 == 0:
            return [GateCount(re.ResourceIdentity.resource_rep())]
        return [GateCount(cls.resource_rep())]


class ResourceT(ResourceOperator):
    r"""Resource class for the T-gate.

    Args:
        wires (Sequence[int] or int): the wire the operation acts on

    Resources:
        The T-gate is treated as a fundamental gate and thus it cannot be decomposed
        further. Requesting the resources of this gate raises a :code:`ResourcesNotDefined` error.

    .. seealso:: :class:`~.T`

    """

    num_wires = 1

    @staticmethod
    def _resource_decomp(**kwargs) -> Dict[CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources of the operator. The
        keys are the operators and the associated values are the counts.

        Resources:
            The T-gate is treated as a fundamental gate and thus it cannot be decomposed
            further. Requesting the resources of this gate raises a :code:`ResourcesNotDefined` error.

        Raises:
            ResourcesNotDefined: This gate is fundamental, no further decomposition defined.
        """
        raise re.ResourcesNotDefined

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
        the Operator that are needed to compute a resource estimation."""
        return CompressedResourceOp(cls, {})

    @classmethod
    def adjoint_resource_decomp(cls) -> Dict[CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources for the adjoint of the operator.

        Resources:
            The adjoint of the T-gate is equivalent to the T-gate raised to the 7th power.
            The resources are defined as seven instances of the T-gate.

        Returns:
            Dict[CompressedResourceOp, int]: The keys are the operators and the associated
                values are the counts.
        """

        return [GateCount(cls.resource_rep()), GateCount(ResourceX.resource_rep(), 2)]

    @staticmethod
    def controlled_resource_decomp(num_ctrl_wires, num_ctrl_values, num_work_wires):
        r"""Returns a dictionary representing the resources for a controlled version of the operator.

        Args:
            num_ctrl_wires (int): the number of qubits the operation is controlled on
            num_ctrl_values (int): the number of control qubits, that are controlled when in the :math:`|0\rangle` state
            num_work_wires (int): the number of additional qubits that can be used for decomposition

        Resources:
            The T-gate is equivalent to the PhaseShift gate for some fixed phase. Given a single
            control wire, the cost is therefore a single instance of
            :class:`~.ResourceControlledPhaseShift`. Two additional :class:`~.ResourceX` gates are
            used to flip the control qubit if it is zero-controlled.

            In the case where multiple controlled wires are provided, we can collapse the control
            wires by introducing one 'clean' auxilliary qubit (which gets reset at the end).
            In this case the cost increases by two additional :class:`~.ResourceMultiControlledX` gates,
            as described in (Lemma 7.11) `Barenco et al. <https://arxiv.org/pdf/quant-ph/9503016>`_.

        Returns:
            Dict[CompressedResourceOp, int]: The keys are the operators and the associated
                values are the counts.
        """
        if num_ctrl_wires == 1:
            gate_types = [GateCount(re.ResourceControlledPhaseShift.resource_rep())]

            if num_ctrl_values:
                gate_types.append(GateCount(re.ResourceX.resource_rep(), 2))

            return gate_types

        ct = re.ResourceControlledPhaseShift.resource_rep()
        mcx = re.ResourceMultiControlledX.resource_rep(
            num_ctrl_wires=num_ctrl_wires,
            num_ctrl_values=num_ctrl_values,
            num_work_wires=num_work_wires,
        )
        return [GateCount(ct), GateCount(mcx, 2)]

    @classmethod
    def pow_resource_decomp(cls, z) -> Dict[CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources for an operator raised to a power.

        Args:
            z (int): the power that the operator is being raised to

        Resources:
            The T-gate, when raised to a power which is a multiple of eight, produces identity.
            The cost of raising to an arbitrary integer power :math:`z` is given by
            :math:`z \mod 8` instances of the T-gate.

        Returns:
            Dict[CompressedResourceOp, int]: The keys are the operators and the associated
                values are the counts.
        """
        if (mod_8 := z % 8) == 0:
            return {re.ResourceIdentity.resource_rep(): 1}
        return {cls.resource_rep(): mod_8}


class ResourceX(ResourceOperator):
    r"""Resource class for the X-gate.

    Args:
        wires (Sequence[int] or int): the wire the operation acts on

    Resources:
        The X-gate can be decomposed according to the following identities:

        .. math::

            \begin{align}
                \hat{X} &= \hat{H} \cdot \hat{Z} \cdot \hat{H}, \\
                \hat{Z} &= \hat{S}^{2}.
            \end{align}

        Thus the resources for an X-gate are two :class:`~.ResourceS` gates and
        two :class:`~.ResourceHadamard` gates.

    .. seealso:: :class:`~.X`

    **Example**

    The resources for this operation are computed using:

    >>> re.ResourceX.resources()
    {S: 2, Hadamard: 2}
    """

    num_wires = 1

    @staticmethod
    def _resource_decomp(**kwargs) -> Dict[CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources of the operator. The
        keys are the operators and the associated values are the counts.

        Resources:
            The X-gate can be decomposed according to the following identities:

            .. math::

                \begin{align}
                    \hat{X} &= \hat{H} \cdot \hat{Z} \cdot \hat{H}, \\
                    \hat{Z} &= \hat{S}^{2}.
                \end{align}

            Thus the resources for an X-gate are two :class:`~.ResourceS` gates and
            two :class:`~.ResourceHadamard` gates.
        """
        s = re.ResourceS.resource_rep()
        h = re.ResourceHadamard.resource_rep()

        return [GateCount(h, 2), GateCount(s, 2)]

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
        the Operator that are needed to compute a resource estimation."""
        return CompressedResourceOp(cls, {})

    @classmethod
    def adjoint_resource_decomp(cls) -> Dict[CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources for the adjoint of the operator.

        Resources:
            This operation is self-adjoint, so the resources of the adjoint operation results
            in the original operation.

        Returns:
            Dict[CompressedResourceOp, int]: The keys are the operators and the associated
                values are the counts.
        """
        return [GateCount(cls.resource_rep())]

    @staticmethod
    def controlled_resource_decomp(num_ctrl_wires, num_ctrl_values, num_work_wires):
        r"""Returns a dictionary representing the resources for a controlled version of the operator.

        Args:
            num_ctrl_wires (int): the number of qubits the operation is controlled on
            num_ctrl_values (int): the number of control qubits, that are controlled when in the :math:`|0\rangle` state
            num_work_wires (int): the number of additional qubits that can be used for decomposition

        Resources:
            For one or two control wires, the cost is one of :class:`~.ResourceCNOT`
            or :class:`~.ResourceToffoli` respectively. Two additional :class:`~.ResourceX` gates
            per control qubit are used to flip the control qubits if they are zero-controlled.

            In the case where multiple controlled wires are provided, the cost is one general
            :class:`~.ResourceMultiControlledX` gate.

        Returns:
            Dict[CompressedResourceOp, int]: The keys are the operators and the associated
                values are the counts.
        """
        if num_ctrl_wires > 2:
            return [
                GateCount(
                    re.ResourceMultiControlledX.resource_rep(
                        num_ctrl_wires, num_ctrl_values, num_work_wires
                    )
                )
            ]

        gate_lst = []
        if num_ctrl_values:
            gate_lst.append(GateCount(re.ResourceX.resource_rep(), 2 * num_ctrl_values))

        if num_ctrl_wires == 1:
            gate_lst.append(GateCount(re.ResourceCNOT.resource_rep()))

        if num_ctrl_wires == 2:
            gate_lst.append(GateCount(re.ResourceToffoli.resource_rep()))

        return gate_lst

    @classmethod
    def pow_resource_decomp(cls, z) -> Dict[CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources for an operator raised to a power.

        Args:
            z (int): the power that the operator is being raised to

        Resources:
            The X-gate raised to even powers produces identity and raised
            to odd powers it produces itself.

        Returns:
            Dict[CompressedResourceOp, int]: The keys are the operators and the associated
                values are the counts.
        """
        if z % 2 == 0:
            return [GateCount(re.ResourceIdentity.resource_rep())]
        return [GateCount(cls.resource_rep())]


class ResourceY(ResourceOperator):
    r"""Resource class for the Y-gate.

    Args:
        wires (Sequence[int] or int): the wire the operation acts on

    Resources:
        The Y-gate can be decomposed according to the following identities:

        .. math::

            \begin{align}
                \hat{Y} &= \hat{S} \cdot \hat{X} \cdot \hat{S}^{\dagger}, \\
                \hat{X} &= \hat{H} \cdot \hat{Z} \cdot \hat{H}, \\
                \hat{Z} &= \hat{S}^{2}, \\
                \hat{S}^{\dagger} &= 3 \hat{S}.
            \end{align}

        Thus the resources for a Y-gate are six :class:`~.ResourceS` gates and
        two :class:`~.ResourceHadamard` gates.

    .. seealso:: :class:`~.Y`

    **Example**

    The resources for this operation are computed using:

    >>> re.ResourceY.resources()
    {S: 6, Hadamard: 2}
    """

    num_wires = 1

    @staticmethod
    def _resource_decomp(**kwargs) -> Dict[CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources of the operator. The
        keys are the operators and the associated values are the counts.

        Resources:
            The Y-gate can be decomposed according to the following identities:

            .. math::

                \begin{align}
                    \hat{Y} &= \hat{S} \cdot \hat{X} \cdot \hat{S}^{\dagger}, \\
                    \hat{X} &= \hat{H} \cdot \hat{Z} \cdot \hat{H}, \\
                    \hat{Z} &= \hat{S}^{2}, \\
                    \hat{S}^{\dagger} &= 3 \hat{S}.
                \end{align}

            Thus the resources for a Y-gate are six :class:`~.ResourceS` gates and
            two :class:`~.ResourceHadamard` gates.
        """
        s = re.ResourceS.resource_rep()
        h = re.ResourceHadamard.resource_rep()

        return [GateCount(s, 6), GateCount(h, 2)]

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
        the Operator that are needed to compute a resource estimation."""
        return CompressedResourceOp(cls, {})

    @classmethod
    def adjoint_resource_decomp(cls) -> Dict[CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources for the adjoint of the operator.

        Resources:
            This operation is self-adjoint, so the resources of the adjoint operation results
            in the original operation.

        Returns:
            Dict[CompressedResourceOp, int]: The keys are the operators and the associated
                values are the counts.
        """
        return [GateCount(cls.resource_rep())]

    @staticmethod
    def controlled_resource_decomp(
        num_ctrl_wires, num_ctrl_values, num_work_wires
    ) -> Dict[CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources for a controlled version of the operator.

        Args:
            num_ctrl_wires (int): the number of qubits the operation is controlled on
            num_ctrl_values (int): the number of control qubits, that are controlled when in the :math:`|0\rangle` state
            num_work_wires (int): the number of additional qubits that can be used for decomposition

        Resources:
            For a single control wire, the cost is a single instance of :class:`~.ResourceCY`.
            Two additional :class:`~.ResourceX` gates are used to flip the control qubit if
            it is zero-controlled.

            In the case where multiple controlled wires are provided, the resources are derived from
            the following identity:

                .. math:: \hat{Y} = \hat{S} \cdot \hat{X} \cdot \hat{S}^{\dagger}.

            Specifically, the resources are given by a :class:`~.ResourceX` gate conjugated with
            a pair of :class:`~.ResourceS` gates. By replacing the :class:`~.ResourceX` gate with a
            :class:`~.ResourceMultiControlledX` gate, we obtain a controlled-version of this identity.

        Returns:
            Dict[CompressedResourceOp, int]: The keys are the operators and the associated
                values are the counts.
        """
        if num_ctrl_wires == 1:
            gate_types = [GateCount(re.ResourceCY.resource_rep())]

            if num_ctrl_values:
                gate_types.append(GateCount(re.ResourceX.resource_rep(), 2))

            return gate_types

        gate_types = {}

        s = re.ResourceS.resource_rep()
        s_dagg = re.ResourceAdjoint.resource_rep(re.ResourceS, {})
        mcx = re.ResourceMultiControlledX.resource_rep(
            num_ctrl_wires=num_ctrl_wires,
            num_ctrl_values=num_ctrl_values,
            num_work_wires=num_work_wires,
        )

        gate_types[s] = 1
        gate_types[s_dagg] = 1
        gate_types[mcx] = 1
        return [GateCount(s), GateCount(s_dagg), GateCount(mcx)]

    @classmethod
    def pow_resource_decomp(cls, z) -> Dict[CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources for an operator raised to a power.

        Args:
            z (int): the power that the operator is being raised to

        Resources:
            The Y-gate raised to even powers produces identity and raised
            to odd powers it produces itself.

        Returns:
            Dict[CompressedResourceOp, int]: The keys are the operators and the associated
                values are the counts.
        """
        if z % 2 == 0:
            return [GateCount(re.ResourceIdentity.resource_rep())]
        return [GateCount(cls.resource_rep())]


class ResourceZ(ResourceOperator):
    r"""Resource class for the Z-gate.

    Args:
        wires (Sequence[int] or int): the wire the operation acts on

    Resources:
        The Z-gate can be decomposed according to the following identities:

        .. math:: \hat{Z} = \hat{S}^{2},

        thus the resources for a Z-gate are two :class:`~.ResourceS` gates.

    .. seealso:: :class:`~.Z`

    **Example**

    The resources for this operation are computed using:

    >>> re.ResourceZ.resources()
    {S: 2}
    """

    num_wires = 1

    @staticmethod
    def _resource_decomp(**kwargs) -> Dict[CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources of the operator. The
        keys are the operators and the associated values are the counts.

        Resources:
            The Z-gate can be decomposed according to the following identities:

            .. math:: \hat{Z} = \hat{S}^{2},

            thus the resources for a Z-gate are two :class:`~.ResourceS` gates.
        """
        s = re.ResourceS.resource_rep()
        return [GateCount(s, 2)]

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
        the Operator that are needed to compute a resource estimation."""
        return CompressedResourceOp(cls, {})

    @classmethod
    def adjoint_resource_decomp(cls) -> Dict[CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources for the adjoint of the operator.

        Resources:
            This operation is self-adjoint, so the resources of the adjoint operation results
            in the original operation.

        Returns:
            Dict[CompressedResourceOp, int]: The keys are the operators and the associated
                values are the counts.
        """
        return [GateCount(cls.resource_rep())]

    @staticmethod
    def controlled_resource_decomp(
        num_ctrl_wires, num_ctrl_values, num_work_wires
    ) -> Dict[CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources for a controlled version of the operator.

        Args:
            num_ctrl_wires (int): the number of qubits the operation is controlled on
            num_ctrl_values (int): the number of control qubits, that are controlled when in the :math:`|0\rangle` state
            num_work_wires (int): the number of additional qubits that can be used for decomposition

        Resources:
            For one or two control wires, the cost is one of :class:`~.ResourceCZ`
            or :class:`~.ResourceCCZ` respectively. Two additional :class:`~.ResourceX` gates
            per control qubit are used to flip the control qubits if they are zero-controlled.

            In the case where multiple controlled wires are provided, the resources are derived from
            the following identity:

            .. math:: \hat{Z} = \hat{H} \cdot \hat{X} \cdot \hat{H}.

            Specifically, the resources are given by a :class:`~.ResourceX` gate conjugated with
            a pair of :class:`~.ResourceHadamard` gates. By replacing the :class:`~.ResourceX` gate
            with a :class:`~.ResourceMultiControlledX` gate, we obtain a controlled-version of this
            identity.

        Returns:
            Dict[CompressedResourceOp, int]: The keys are the operators and the associated
                values are the counts.
        """
        if num_ctrl_wires > 2:
            h = re.ResourceHadamard.resource_rep()
            mcx = re.ResourceMultiControlledX.resource_rep(
                num_ctrl_wires=num_ctrl_wires,
                num_ctrl_values=num_ctrl_values,
                num_work_wires=num_work_wires,
            )
            return [GateCount(h, 2), GateCount(mcx)]

        gate_list = []
        if num_ctrl_wires == 1:
            gate_list.append(GateCount(re.ResourceCZ.resource_rep()))

        if num_ctrl_wires == 2:
            gate_list.append(GateCount(re.ResourceCCZ.resource_rep()))

        if num_ctrl_values:
            gate_list.append(GateCount(re.ResourceX.resource_rep(), 2 * num_ctrl_values))

        return gate_list

    @classmethod
    def pow_resource_decomp(cls, z) -> Dict[CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources for an operator raised to a power.

        Args:
            z (int): the power that the operator is being raised to

        Resources:
            The Z-gate raised to even powers produces identity and raised
            to odd powers it produces itself.

        Returns:
            Dict[CompressedResourceOp, int]: The keys are the operators and the associated
                values are the counts.
        """
        if z % 2 == 0:
            return [GateCount(re.ResourceIdentity.resource_rep())]
        return [GateCount(cls.resource_rep())]
