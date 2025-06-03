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
r"""Resource operators for non parametric single qubit operations."""
import pennylane.labs.resource_estimation as re
from pennylane.labs.resource_estimation.resource_operator import (
    CompressedResourceOp,
    GateCount,
    ResourceOperator,
    resource_rep,
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

    @classmethod
    def default_resource_decomp(cls, **kwargs) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator. The
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
        the Operator that are needed to compute the resources."""
        return CompressedResourceOp(cls, {})

    @classmethod
    def default_adjoint_resource_decomp(cls) -> list[GateCount]:
        r"""Returns a list representing the resources for the adjoint of the operator.

        Resources:
            This operation is self-adjoint, so the resources of the adjoint operation results
            in the original operation.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
                represents a specific quantum gate and the number of times it appears
                in the decomposition.
        """
        return [GateCount(cls.resource_rep(), 1)]

    @classmethod
    def default_controlled_resource_decomp(
        cls,
        ctrl_num_ctrl_wires: int,
        ctrl_num_ctrl_values: int,
    ) -> list[GateCount]:
        r"""Returns a list representing the resources for a controlled version of the operator.

        Args:
            ctrl_num_ctrl_wires (int): the number of qubits the operation is controlled on
            ctrl_num_ctrl_values (int): the number of control qubits, that are controlled when in the :math:`|0\rangle` state

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
            list[GateCount]: A list of GateCount objects, where each object
                represents a specific quantum gate and the number of times it appears
                in the decomposition.
        """
        if ctrl_num_ctrl_wires == 1:
            gate_lst = [GateCount(resource_rep(re.ResourceCH))]

            if ctrl_num_ctrl_values:
                gate_lst.append(GateCount(resource_rep(re.ResourceX), 2))

            return gate_lst

        gate_lst = []

        ry = resource_rep(re.ResourceRY)
        h = cls.resource_rep()
        mcx = resource_rep(
            re.ResourceMultiControlledX,
            {
                "num_ctrl_wires": ctrl_num_ctrl_wires,
                "num_ctrl_values": ctrl_num_ctrl_values,
            },
        )

        gate_lst.append(GateCount(h, 2))
        gate_lst.append(GateCount(ry, 2))
        gate_lst.append(GateCount(mcx))
        return gate_lst

    @classmethod
    def default_pow_resource_decomp(cls, pow_z) -> list[GateCount]:
        r"""Returns a list representing the resources for an operator raised to a power.

        Args:
            pow_z (int): the power that the operator is being raised to

        Resources:
            The Hadamard gate raised to even powers produces identity and raised
            to odd powers it produces itself.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
                represents a specific quantum gate and the number of times it appears
                in the decomposition.
        """
        if pow_z % 2 == 0:
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

    >>> re.ResourceS.resource_decomp()
    {T: 2}
    """

    num_wires = 1

    @classmethod
    def default_resource_decomp(cls, **kwargs) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator. The
        keys are the operators and the associated values are the counts.

        Resources:
            The S-gate decomposes into two T-gates.
        """
        t = resource_rep(ResourceT)
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
    def default_adjoint_resource_decomp(cls) -> list[GateCount]:
        r"""Returns a list representing the resources for the adjoint of the operator.

        Resources:
            The adjoint of the S-gate is equivalent to the Z.S.
            The resources are defined as one instance of Z-gate, and one instance of S-gate.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
                represents a specific quantum gate and the number of times it appears
                in the decomposition.
        """
        z = resource_rep(ResourceZ)
        return [GateCount(z), GateCount(cls.resource_rep())]

    @classmethod
    def default_controlled_resource_decomp(
        cls,
        ctrl_num_ctrl_wires,
        ctrl_num_ctrl_values,
    ):
        r"""Returns a list representing the resources for a controlled version of the operator.

        Args:
            ctrl_num_ctrl_wires (int): the number of qubits the operation is controlled on
            ctrl_num_ctrl_values (int): the number of control qubits, that are controlled when in the :math:`|0\rangle` state

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
            list[GateCount]: A list of GateCount objects, where each object
                represents a specific quantum gate and the number of times it appears
                in the decomposition.
        """
        if ctrl_num_ctrl_wires == 1:
            gate_lst = [GateCount(resource_rep(re.ResourceControlledPhaseShift))]

            if ctrl_num_ctrl_values:
                gate_lst.append(GateCount(resource_rep(re.ResourceX), 2))

            return gate_lst

        cs = resource_rep(re.ResourceControlledPhaseShift)
        mcx = resource_rep(
            re.ResourceMultiControlledX,
            {
                "num_ctrl_wires": ctrl_num_ctrl_wires,
                "num_ctrl_values": ctrl_num_ctrl_values,
            },
        )
        return [GateCount(cs, 1), GateCount(mcx, 2)]

    @classmethod
    def default_pow_resource_decomp(cls, pow_z) -> list[GateCount]:
        r"""Returns a list representing the resources for an operator raised to a power.

        Args:
            pow_z (int): the power that the operator is being raised to

        Resources:
            The S-gate, when raised to a power which is a multiple of four, produces identity.
            The cost of raising to an arbitrary integer power :math:`z`, when :math:`z \mod 4`
            is equal to one, means one instance of the S-gate.
            The cost of raising to an arbitrary integer power :math:`z`, when :math:`z \mod 4`
            is equal to two, means one instance of the Z-gate.
            The cost of raising to an arbitrary integer power :math:`z`, when :math:`z \mod 4`
            is equal to three, means one instance of the Z-gate and one instance of S-gate.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
                represents a specific quantum gate and the number of times it appears
                in the decomposition.
        """
        mod_4 = pow_z % 4
        if mod_4 == 0:
            return [GateCount(resource_rep(re.ResourceIdentity))]
        if mod_4 == 1:
            return [GateCount(cls.resource_rep())]
        if mod_4 == 2:
            return [GateCount(resource_rep(ResourceZ))]

        return [GateCount(resource_rep(ResourceZ)), GateCount(cls.resource_rep())]


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
    def default_resource_decomp(cls, **kwargs) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator. The
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
        cnot = resource_rep(re.ResourceCNOT)
        return [GateCount(cnot, 3)]

    @classmethod
    def default_adjoint_resource_decomp(cls) -> list[GateCount]:
        r"""Returns a list representing the resources for the adjoint of the operator.

        Resources:
            This operation is self-adjoint, so the resources of the adjoint operation results
            in the original operation.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
                represents a specific quantum gate and the number of times it appears
                in the decomposition.
        """
        return [GateCount(cls.resource_rep())]

    @classmethod
    def default_controlled_resource_decomp(
        cls, ctrl_num_ctrl_wires, ctrl_num_ctrl_values
    ) -> list[GateCount]:
        r"""Returns a list representing the resources for a controlled version of the operator.

        Args:
            ctrl_num_ctrl_wires (int): the number of qubits the operation is controlled on
            ctrl_num_ctrl_values (int): the number of control qubits, that are controlled when in the :math:`|0\rangle` state

        Resources:
            For a single control wire, the cost is a single instance of :class:`~.ResourceCSWAP`.
            Two additional :class:`~.ResourceX` gates are used to flip the control qubit if
            it is zero-controlled.

            In the case where multiple controlled wires are provided, the resources are given by
            two :class:`~.ResourceCNOT` gates and one :class:`~.ResourceMultiControlledX` gate. This
            is because of the symmetric resource decomposition of the SWAP gate. By controlling on
            the middle CNOT gate, we obtain the required controlled operation.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
                represents a specific quantum gate and the number of times it appears
                in the decomposition.
        """
        if ctrl_num_ctrl_wires == 1:
            gate_types = [GateCount(resource_rep(re.ResourceCSWAP))]

            if ctrl_num_ctrl_values:
                gate_types.append(GateCount(resource_rep(ResourceX), 2))

            return gate_types

        cnot = resource_rep(re.ResourceCNOT)
        mcx = resource_rep(
            re.ResourceMultiControlledX,
            {
                "num_ctrl_wires": ctrl_num_ctrl_wires,
                "num_ctrl_values": ctrl_num_ctrl_values,
            },
        )
        return [GateCount(cnot, 2), GateCount(mcx)]

    @classmethod
    def default_pow_resource_decomp(cls, pow_z) -> list[GateCount]:
        r"""Returns a list representing the resources for an operator raised to a power.

        Args:
            z (int): the power that the operator is being raised to

        Resources:
            The SWAP gate raised to even powers produces identity and raised
            to odd powers it produces itself.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
                represents a specific quantum gate and the number of times it appears
                in the decomposition.
        """
        if pow_z % 2 == 0:
            return [GateCount(resource_rep(re.ResourceIdentity))]
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

    @classmethod
    def default_resource_decomp(cls, **kwargs) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator. The
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
    def default_adjoint_resource_decomp(cls) -> list[GateCount]:
        r"""Returns a list representing the resources for the adjoint of the operator.

        Resources:
            The adjoint of the T-gate is equivalent to the T-gate raised to the 7th power.
            The resources are defined as seven instances of the T-gate.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
                represents a specific quantum gate and the number of times it appears
                in the decomposition.
        """
        z = resource_rep(ResourceZ)
        s = resource_rep(ResourceS)
        return [GateCount(cls.resource_rep()), GateCount(s), GateCount(z)]

    @classmethod
    def default_controlled_resource_decomp(cls, ctrl_num_ctrl_wires, ctrl_num_ctrl_values):
        r"""Returns a list representing the resources for a controlled version of the operator.

        Args:
            ctrl_num_ctrl_wires (int): the number of qubits the operation is controlled on
            ctrl_num_ctrl_values (int): the number of control qubits, that are controlled when in the :math:`|0\rangle` state

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
            list[GateCount]: A list of GateCount objects, where each object
                represents a specific quantum gate and the number of times it appears
                in the decomposition.
        """
        if ctrl_num_ctrl_wires == 1:
            gate_types = [GateCount(resource_rep(re.ResourceControlledPhaseShift))]

            if ctrl_num_ctrl_values:
                gate_types.append(GateCount(resource_rep(ResourceX), 2))

            return gate_types

        ct = resource_rep(re.ResourceControlledPhaseShift)
        mcx = resource_rep(
            re.ResourceMultiControlledX,
            {
                "num_ctrl_wires": ctrl_num_ctrl_wires,
                "num_ctrl_values": ctrl_num_ctrl_values,
            },
        )
        return [GateCount(ct), GateCount(mcx, 2)]

    @classmethod
    def default_pow_resource_decomp(cls, pow_z) -> list[GateCount]:
        r"""Returns a list representing the resources for an operator raised to a power.

        Args:
            pow_z (int): the power that the operator is being raised to

        Resources:
            The T-gate, when raised to a power which is a multiple of eight, produces identity.
            Consequently, for any integer power `z`, the effective quantum operation :math:`T^{z}` is equivalent
            to :math:`T^{z \pmod 8}`.

        The decomposition for :math:`T^{z}` (where :math:`z \pmod 8` is denoted as `z'`) is as follows:
            - If `z' = 0`: The operation is equivalent to the Identity gate (`I`).
            - If `z' = 1`: The operation is equivalent to the T-gate (`T`).
            - If `z' = 2`: The operation is equivalent to the S-gate (`S`).
            - If `z' = 3`: The operation is equivalent to a composition of an S-gate and a T-gate (:math:`S \cdot T`).
            - If `z' = 4` : The operation is equivalent to the Z-gate (`Z`).
            - If `z' = 5`: The operation is equivalent to a composition of a Z-gate and a T-gate (:math:`Z \cdot T`).
            - If `z' = 6`: The operation is equivalent to the S-adjoint gate (:math:`S^{\dagger}`).
            - If `z' = 7`: The operation is equivalent to the T-adjoint gate (:math:`T^{\dagger}`).

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
                represents a specific quantum gate and the number of times it appears
                in the decomposition.
        """
        if (mod_8 := pow_z % 8) == 0:
            return [GateCount(resource_rep(re.ResourceIdentity))]

        gate_lst = []
        if mod_8 >= 4:
            gate_lst.append(GateCount(resource_rep(ResourceZ)))
            mod_8 -= 4

        if mod_8 >= 2:
            gate_lst.append(GateCount(resource_rep(ResourceS)))
            mod_8 -= 2

        if mod_8 >= 1:
            gate_lst.append(GateCount(cls.resource_rep()))

        return gate_lst


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

    >>> re.ResourceX.resource_decomp()
    {S: 2, Hadamard: 2}
    """

    num_wires = 1

    @classmethod
    def default_resource_decomp(cls, **kwargs) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator. The
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
        s = resource_rep(ResourceS)
        h = resource_rep(ResourceHadamard)

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
    def default_adjoint_resource_decomp(cls) -> list[GateCount]:
        r"""Returns a list representing the resources for the adjoint of the operator.

        Resources:
            This operation is self-adjoint, so the resources of the adjoint operation results
            in the original operation.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
                represents a specific quantum gate and the number of times it appears
                in the decomposition.
        """
        return [GateCount(cls.resource_rep())]

    @classmethod
    def default_controlled_resource_decomp(
        cls,
        ctrl_num_ctrl_wires,
        ctrl_num_ctrl_values,
    ):
        r"""Returns a list representing the resources for a controlled version of the operator.

        Args:
            ctrl_num_ctrl_wires (int): the number of qubits the operation is controlled on
            ctrl_num_ctrl_values (int): the number of control qubits, that are controlled when in the :math:`|0\rangle` state

        Resources:
            For one or two control wires, the cost is one of :class:`~.ResourceCNOT`
            or :class:`~.ResourceToffoli` respectively. Two additional :class:`~.ResourceX` gates
            per control qubit are used to flip the control qubits if they are zero-controlled.

            In the case where multiple controlled wires are provided, the cost is one general
            :class:`~.ResourceMultiControlledX` gate.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
                represents a specific quantum gate and the number of times it appears
                in the decomposition.
        """
        if ctrl_num_ctrl_wires > 2:
            mcx = resource_rep(
                re.ResourceMultiControlledX,
                {
                    "num_ctrl_wires": ctrl_num_ctrl_wires,
                    "num_ctrl_values": ctrl_num_ctrl_values,
                },
            )

            return [GateCount(mcx)]

        gate_lst = []
        if ctrl_num_ctrl_values:
            gate_lst.append(GateCount(resource_rep(ResourceX), 2 * ctrl_num_ctrl_values))

        if ctrl_num_ctrl_wires == 0:
            gate_lst.append(GateCount(resource_rep(ResourceX)))
        elif ctrl_num_ctrl_wires == 1:
            gate_lst.append(GateCount(resource_rep(re.ResourceCNOT)))
        else:
            gate_lst.append(GateCount(resource_rep(re.ResourceToffoli)))

        return gate_lst

    @classmethod
    def default_pow_resource_decomp(cls, pow_z) -> list[GateCount]:
        r"""Returns a list representing the resources for an operator raised to a power.

        Args:
            pow_z (int): the power that the operator is being raised to

        Resources:
            The X-gate raised to even powers produces identity and raised
            to odd powers it produces itself.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
                represents a specific quantum gate and the number of times it appears
                in the decomposition.
        """
        if pow_z % 2 == 0:
            return [GateCount(resource_rep(re.ResourceIdentity))]
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

    >>> re.ResourceY.resource_decomp()
    {S: 6, Hadamard: 2}
    """

    num_wires = 1

    @classmethod
    def default_resource_decomp(cls, **kwargs) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator. The
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
        z = resource_rep(ResourceZ)
        s = resource_rep(ResourceS)
        s_adj = resource_rep(re.ResourceAdjoint, {"base_cmpr_op": s})
        h = resource_rep(ResourceHadamard)

        return [GateCount(s), GateCount(z), GateCount(s_adj), GateCount(h, 2)]

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
    def default_adjoint_resource_decomp(cls) -> list[GateCount]:
        r"""Returns a list representing the resources for the adjoint of the operator.

        Resources:
            This operation is self-adjoint, so the resources of the adjoint operation results
            in the original operation.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
                represents a specific quantum gate and the number of times it appears
                in the decomposition.
        """
        return [GateCount(cls.resource_rep())]

    @classmethod
    def default_controlled_resource_decomp(
        cls, ctrl_num_ctrl_wires, ctrl_num_ctrl_values
    ) -> list[GateCount]:
        r"""Returns a list representing the resources for a controlled version of the operator.

        Args:
            ctrl_num_ctrl_wires (int): the number of qubits the operation is controlled on
            ctrl_num_ctrl_values (int): the number of control qubits, that are controlled when in the :math:`|0\rangle` state

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
            list[GateCount]: A list of GateCount objects, where each object
                represents a specific quantum gate and the number of times it appears
                in the decomposition.
        """
        if ctrl_num_ctrl_wires == 1:
            gate_types = [GateCount(resource_rep(re.ResourceCY))]

            if ctrl_num_ctrl_values:
                gate_types.append(GateCount(resource_rep(ResourceX), 2))

            return gate_types

        s = resource_rep(ResourceS)
        s_dagg = resource_rep(re.ResourceAdjoint, {"base_cmpr_op": s})
        mcx = resource_rep(
            re.ResourceMultiControlledX,
            {
                "num_ctrl_wires": ctrl_num_ctrl_wires,
                "num_ctrl_values": ctrl_num_ctrl_values,
            },
        )
        return [GateCount(s), GateCount(s_dagg), GateCount(mcx)]

    @classmethod
    def default_pow_resource_decomp(cls, pow_z) -> list[GateCount]:
        r"""Returns a list representing the resources for an operator raised to a power.

        Args:
            pow_z (int): the power that the operator is being raised to

        Resources:
            The Y-gate raised to even powers produces identity and raised
            to odd powers it produces itself.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
                represents a specific quantum gate and the number of times it appears
                in the decomposition.
        """
        if pow_z % 2 == 0:
            return [GateCount(resource_rep(re.ResourceIdentity))]
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

    >>> re.ResourceZ.resource_decomp()
    {S: 2}
    """

    num_wires = 1

    @classmethod
    def default_resource_decomp(cls, **kwargs) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator. The
        keys are the operators and the associated values are the counts.

        Resources:
            The Z-gate can be decomposed according to the following identities:

            .. math:: \hat{Z} = \hat{S}^{2},

            thus the resources for a Z-gate are two :class:`~.ResourceS` gates.
        """
        s = resource_rep(ResourceS)
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
    def default_adjoint_resource_decomp(cls) -> list[GateCount]:
        r"""Returns a list representing the resources for the adjoint of the operator.

        Resources:
            This operation is self-adjoint, so the resources of the adjoint operation results
            in the original operation.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
                represents a specific quantum gate and the number of times it appears
                in the decomposition.
        """
        return [GateCount(cls.resource_rep())]

    @classmethod
    def default_controlled_resource_decomp(
        cls,
        ctrl_num_ctrl_wires,
        ctrl_num_ctrl_values,
    ) -> list[GateCount]:
        r"""Returns a list representing the resources for a controlled version of the operator.

        Args:
            ctrl_num_ctrl_wires (int): the number of qubits the operation is controlled on
            ctrl_num_ctrl_values (int): the number of control qubits, that are controlled when in the :math:`|0\rangle` state

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
            list[GateCount]: A list of GateCount objects, where each object
                represents a specific quantum gate and the number of times it appears
                in the decomposition.
        """
        if ctrl_num_ctrl_wires > 2:
            h = resource_rep(ResourceHadamard)
            mcx = resource_rep(
                re.ResourceMultiControlledX,
                {
                    "num_ctrl_wires": ctrl_num_ctrl_wires,
                    "num_ctrl_values": ctrl_num_ctrl_values,
                },
            )
            return [GateCount(h, 2), GateCount(mcx)]

        gate_list = []
        if ctrl_num_ctrl_wires == 1:
            gate_list.append(GateCount(resource_rep(re.ResourceCZ)))

        if ctrl_num_ctrl_wires == 2:
            gate_list.append(GateCount(resource_rep(re.ResourceCCZ)))

        if ctrl_num_ctrl_values:
            gate_list.append(GateCount(resource_rep(ResourceX), 2 * ctrl_num_ctrl_values))

        return gate_list

    @classmethod
    def default_pow_resource_decomp(cls, pow_z) -> list[GateCount]:
        r"""Returns a list representing the resources for an operator raised to a power.

        Args:
            pow_z (int): the power that the operator is being raised to

        Resources:
            The Z-gate raised to even powers produces identity and raised
            to odd powers it produces itself.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
                represents a specific quantum gate and the number of times it appears
                in the decomposition.
        """
        if pow_z % 2 == 0:
            return [GateCount(resource_rep(re.ResourceIdentity))]
        return [GateCount(cls.resource_rep())]
