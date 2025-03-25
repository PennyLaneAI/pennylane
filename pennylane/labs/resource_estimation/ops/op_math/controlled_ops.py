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
r"""Resource operators for controlled operations."""
from collections import defaultdict
from typing import Dict

import pennylane as qml
import pennylane.labs.resource_estimation as re

# pylint: disable=arguments-differ,too-many-ancestors,too-many-arguments,too-many-positional-arguments


class ResourceCH(qml.CH, re.ResourceOperator):
    r"""Resource class for the CH gate.

    Args:
        wires (Sequence[int]): the wires the operation acts on

    Resources:
        The resources are derived from the following identities (as presented in this
        `blog post <https://quantumcomputing.stackexchange.com/questions/15734/how-to-construct-a-controlled-hadamard-gate-using-single-qubit-gates-and-control>`_):

        .. math::

            \begin{align}
                \hat{H} &= \hat{R}_{y}(\frac{\pi}{4}) \cdot \hat{Z}  \cdot \hat{R}_{y}(\frac{-\pi}{4}), \\
                \hat{Z} &= \hat{H} \cdot \hat{X}  \cdot \hat{H}.
            \end{align}

        Specifically, the resources are defined as two :class:`~.ResourceRY`, two
        :class:`~.ResourceHadamard` and one :class:`~.ResourceCNOT` gates.

    .. seealso:: :class:`~.CH`

    **Example**

    The resources for this operation are computed using:

    >>> re.ResourceCH.resources()
    {Hadamard: 2, RY: 2, CNOT: 1}
    """

    @staticmethod
    def _resource_decomp(**kwargs) -> Dict[re.CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources of the operator. The
        keys are the operators and the associated values are the counts.

        Resources:
            The resources are derived from the following identities (as presented in this
            `blog post <https://quantumcomputing.stackexchange.com/questions/15734/how-to-construct-a-controlled-hadamard-gate-using-single-qubit-gates-and-control>`_):

            .. math::

                \begin{align}
                    \hat{H} &= \hat{R}_{y}(\frac{\pi}{4}) \cdot \hat{Z}  \cdot \hat{R}_{y}(\frac{-\pi}{4}), \\
                    \hat{Z} &= \hat{H} \cdot \hat{X}  \cdot \hat{H}.
                \end{align}

            Specifically, the resources are defined as two :class:`~.ResourceRY`, two
            :class:`~.ResourceHadamard` and one :class:`~.ResourceCNOT` gates.
        """
        gate_types = {}

        ry = re.ResourceRY.resource_rep()
        h = re.ResourceHadamard.resource_rep()
        cnot = re.ResourceCNOT.resource_rep()

        gate_types[h] = 2
        gate_types[ry] = 2
        gate_types[cnot] = 1

        return gate_types

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: Empty dictionary. The resources of this operation don't depend on any additional parameters.
        """
        return {}

    @classmethod
    def resource_rep(cls) -> re.CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation."""
        return re.CompressedResourceOp(cls, {})

    @classmethod
    def adjoint_resource_decomp(cls) -> Dict[re.CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources for the adjoint of the operator.

        Resources:
            This operation is self-adjoint, so the resources of the adjoint operation results
            in the original operation.

        Returns:
            Dict[CompressedResourceOp, int]: The keys are the operators and the associated
                values are the counts.
        """
        return {cls.resource_rep(): 1}

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
            The resources are expressed using the symbolic :class:`~.ResourceControlled`. The resources
            are computed according to the :code:`controlled_resource_decomp()` of the base
            :class:`~.ResourceHadamard` class.

        Returns:
            Dict[CompressedResourceOp, int]: The keys are the operators and the associated
                values are the counts.
        """
        return {
            re.ResourceControlled.resource_rep(
                re.ResourceHadamard, {}, num_ctrl_wires + 1, num_ctrl_values, num_work_wires
            ): 1
        }

    @classmethod
    def pow_resource_decomp(cls, z) -> Dict[re.CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources for an operator raised to a power.

        Args:
            z (int): the power that the operator is being raised to

        Resources:
            This operation is self-inverse, thus when raised to even integer powers acts like
            the identity operator and raised to odd powers it produces itself.

        Returns:
            Dict[CompressedResourceOp, int]: The keys are the operators and the associated
                values are the counts.
        """
        return {re.ResourceIdentity.resource_rep(): 1} if z % 2 == 0 else {cls.resource_rep(): 1}


class ResourceCY(qml.CY, re.ResourceOperator):
    r"""Resource class for the CY gate.

    Args:
        wires (Sequence[int]): the wires the operation acts on
        id (str): custom label given to an operator instance,
            can be useful for some applications where the instance has to be identified.

    Resources:
        The resources are derived from the following identity:

        .. math:: \hat{Y} = \hat{S} \cdot \hat{X} \cdot \hat{S}^{\dagger}.

        By replacing the :class:`~.ResourceX` gate with a :class:`~.ResourceCNOT` we
        obtain the controlled decomposition. Specifically, the resources are defined as a
        :class:`~.ResourceCNOT` gate conjugated by a pair of :class:`~.ResourceS` gates.

    .. seealso:: :class:`~.CY`

    **Example**

    The resources for this operation are computed using:

    >>> re.ResourceCY.resources()
    {CNOT: 1, S: 1, Adjoint(S): 1}
    """

    @staticmethod
    def _resource_decomp(**kwargs) -> Dict[re.CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources of the operator. The
        keys are the operators and the associated values are the counts.

        Resources:
            The resources are derived from the following identity:

            .. math:: \hat{Y} = \hat{S} \cdot \hat{X} \cdot \hat{S}^{\dagger}.

            By replacing the :class:`~.ResourceX` gate with a :class:`~.ResourceCNOT` we
            obtain the controlled decomposition. Specifically, the resources are defined as a
            :class:`~.ResourceCNOT` gate conjugated by a pair of :class:`~.ResourceS` gates.
        """
        gate_types = {}

        cnot = re.ResourceCNOT.resource_rep()
        s = re.ResourceS.resource_rep()
        s_dag = re.ResourceAdjoint.resource_rep(re.ResourceS, {})

        gate_types[cnot] = 1
        gate_types[s] = 1
        gate_types[s_dag] = 1

        return gate_types

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: Empty dictionary. The resources of this operation don't depend on any additional parameters.
        """
        return {}

    @classmethod
    def resource_rep(cls) -> re.CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation."""
        return re.CompressedResourceOp(cls, {})

    @classmethod
    def adjoint_resource_decomp(cls) -> Dict[re.CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources for the adjoint of the operator.

        Resources:
            This operation is self-adjoint, so the resources of the adjoint operation results
            in the original operation.

        Returns:
            Dict[CompressedResourceOp, int]: The keys are the operators and the associated
                values are the counts.
        """
        return {cls.resource_rep(): 1}

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
            The resources are expressed using the symbolic :class:`~.ResourceControlled`. The resources
            are computed according to the :code:`controlled_resource_decomp()` of the base
            :class:`~.ResourceY` class.

        Returns:
            Dict[CompressedResourceOp, int]: The keys are the operators and the associated
                values are the counts.
        """
        return {
            re.ResourceControlled.resource_rep(
                re.ResourceY, {}, num_ctrl_wires + 1, num_ctrl_values, num_work_wires
            ): 1
        }

    @classmethod
    def pow_resource_decomp(cls, z) -> Dict[re.CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources for an operator raised to a power.

        Args:
            z (int): the power that the operator is being raised to

        Resources:
            This operation is self-inverse, thus when raised to even integer powers acts like
            the identity operator and raised to odd powers it produces itself.

        Returns:
            Dict[CompressedResourceOp, int]: The keys are the operators and the associated
                values are the counts.
        """
        return {re.ResourceIdentity.resource_rep(): 1} if z % 2 == 0 else {cls.resource_rep(): 1}


class ResourceCZ(qml.CZ, re.ResourceOperator):
    r"""Resource class for the CZ gate.

    Args:
        wires (Sequence[int]): the wires the operation acts on

    Resources:
        The resources are derived from the following identity:

        .. math:: \hat{Z} = \hat{H} \cdot \hat{X} \cdot \hat{H}.

        By replacing the :class:`~.ResourceX` gate with a :class:`~.ResourceCNOT` we obtain
        the controlled decomposition. Specifically, the resources are defined as a
        :class:`~.ResourceCNOT` gate conjugated by a pair of :class:`~.ResourceHadamard` gates.

    .. seealso:: :class:`~.CZ`

    **Example**

    The resources for this operation are computed using:

    >>> re.ResourceCZ.resources()
    {CNOT: 1, Hadamard: 2}
    """

    @staticmethod
    def _resource_decomp(**kwargs) -> Dict[re.CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources of the operator. The
        keys are the operators and the associated values are the counts.

        Resources:
            The resources are derived from the following identity:

            .. math:: \hat{Z} = \hat{H} \cdot \hat{X} \cdot \hat{H}.

            By replacing the :class:`~.ResourceX` gate with a :class:`~.ResourceCNOT` we obtain
            the controlled decomposition. Specifically, the resources are defined as a
            :class:`~.ResourceCNOT` gate conjugated by a pair of :class:`~.ResourceHadamard` gates.
        """
        gate_types = {}

        cnot = re.ResourceCNOT.resource_rep()
        h = re.ResourceHadamard.resource_rep()

        gate_types[cnot] = 1
        gate_types[h] = 2

        return gate_types

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: Empty dictionary. The resources of this operation don't depend on any additional parameters.
        """
        return {}

    @classmethod
    def resource_rep(cls) -> re.CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation."""
        return re.CompressedResourceOp(cls, {})

    @classmethod
    def adjoint_resource_decomp(cls) -> Dict[re.CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources for the adjoint of the operator.

        Resources:
            This operation is self-adjoint, so the resources of the adjoint operation results
            in the original operation.

        Returns:
            Dict[CompressedResourceOp, int]: The keys are the operators and the associated
                values are the counts.
        """
        return {cls.resource_rep(): 1}

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
            The resources are expressed using the symbolic :class:`~.ResourceControlled`. The resources
            are computed according to the :code:`controlled_resource_decomp()` of the base
            :class:`~.ResourceZ` class.

        Returns:
            Dict[CompressedResourceOp, int]: The keys are the operators and the associated
                values are the counts.
        """
        if num_ctrl_wires == 1 and num_ctrl_values == 0 and num_work_wires == 0:
            return {re.ResourceCCZ.resource_rep(): 1}

        return {
            re.ResourceControlled.resource_rep(
                re.ResourceZ, {}, num_ctrl_wires + 1, num_ctrl_values, num_work_wires
            ): 1
        }

    @classmethod
    def pow_resource_decomp(cls, z) -> Dict[re.CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources for an operator raised to a power.

        Args:
            z (int): the power that the operator is being raised to

        Resources:
            This operation is self-inverse, thus when raised to even integer powers acts like
            the identity operator and raised to odd powers it produces itself.

        Returns:
            Dict[CompressedResourceOp, int]: The keys are the operators and the associated
                values are the counts.
        """
        return {re.ResourceIdentity.resource_rep(): 1} if z % 2 == 0 else {cls.resource_rep(): 1}


class ResourceCSWAP(qml.CSWAP, re.ResourceOperator):
    r"""Resource class for the CSWAP gate.

    Resources:
        The resources are taken from Figure 1d of `arXiv:2305.18128 <https://arxiv.org/pdf/2305.18128>`_.

        The circuit which applies the SWAP operation on wires (1, 2) and controlled on wire (0) is
        defined as:

        .. code-block:: bash

            0: ────╭●────┤
            1: ─╭X─├●─╭X─┤
            2: ─╰●─╰X─╰●─┤

    .. seealso:: :class:`~.CSWAP`

    """

    @staticmethod
    def _resource_decomp(**kwargs) -> Dict[re.CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources of the operator. The
        keys are the operators and the associated values are the counts.

        Resources:
            The resources are taken from Figure 1d of `arXiv:2305.18128 <https://arxiv.org/pdf/2305.18128>`_.

            The circuit which applies the SWAP operation on wires (1, 2) and controlled on wire (0) is
            defined as:

            .. code-block:: bash

                0: ────╭●────┤
                1: ─╭X─├●─╭X─┤
                2: ─╰●─╰X─╰●─┤
        """
        gate_types = {}

        tof = re.ResourceToffoli.resource_rep()
        cnot = re.ResourceCNOT.resource_rep()

        gate_types[tof] = 1
        gate_types[cnot] = 2

        return gate_types

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: Empty dictionary. The resources of this operation don't depend on any additional parameters.
        """
        return {}

    @classmethod
    def resource_rep(cls) -> re.CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation."""
        return re.CompressedResourceOp(cls, {})

    @classmethod
    def adjoint_resource_decomp(cls) -> Dict[re.CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources for the adjoint of the operator.

        Resources:
            This operation is self-adjoint, so the resources of the adjoint operation results
            in the original operation.

        Returns:
            Dict[CompressedResourceOp, int]: The keys are the operators and the associated
                values are the counts.
        """
        return {cls.resource_rep(): 1}

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
            The resources are expressed using the symbolic :class:`~.ResourceControlled`. The resources
            are computed according to the :code:`controlled_resource_decomp()` of the base
            :class:`~.ResourceSWAP` class.

        Returns:
            Dict[CompressedResourceOp, int]: The keys are the operators and the associated
                values are the counts.
        """
        return {
            re.ResourceControlled.resource_rep(
                re.ResourceSWAP, {}, num_ctrl_wires + 1, num_ctrl_values, num_work_wires
            ): 1
        }

    @classmethod
    def pow_resource_decomp(cls, z) -> Dict[re.CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources for an operator raised to a power.

        Args:
            z (int): the power that the operator is being raised to

        Resources:
            This operation is self-inverse, thus when raised to even integer powers acts like
            the identity operator and raised to odd powers it produces itself.

        Returns:
            Dict[CompressedResourceOp, int]: The keys are the operators and the associated
                values are the counts.
        """
        return {re.ResourceIdentity.resource_rep(): 1} if z % 2 == 0 else {cls.resource_rep(): 1}


class ResourceCCZ(qml.CCZ, re.ResourceOperator):
    r"""Resource class for the CCZ gate.

    Args:
        wires (Sequence[int]): the subsystem the gate acts on

    Resources:
        The resources are derived from the following identity:

        .. math:: \hat{Z} = \hat{H} \cdot \hat{X} \cdot \hat{H}.

        By replacing the :class:`~.ResourceX` gate with a :class:`~.ResourceToffoli` we obtain
        the controlled decomposition. Specifically, the resources are defined as a
        :class:`~.ResourceToffoli` gate conjugated by a pair of :class:`~.ResourceHadamard` gates.

    .. seealso:: :class:`~.CCZ`

    **Example**

    The resources for this operation are computed using:

    >>> re.ResourceCCZ.resources()
    {Toffoli: 1, Hadamard: 2}
    """

    @staticmethod
    def _resource_decomp(**kwargs) -> Dict[re.CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources of the operator. The
        keys are the operators and the associated values are the counts.

        Resources:
            The resources are derived from the following identity:

            .. math:: \hat{Z} = \hat{H} \cdot \hat{X} \cdot \hat{H}.

            By replacing the :class:`~.ResourceX` gate with a :class:`~.ResourceToffoli` we obtain
            the controlled decomposition. Specifically, the resources are defined as a
            :class:`~.ResourceToffoli` gate conjugated by a pair of :class:`~.ResourceHadamard` gates.
        """
        gate_types = {}

        toffoli = re.ResourceToffoli.resource_rep()
        h = re.ResourceHadamard.resource_rep()

        gate_types[toffoli] = 1
        gate_types[h] = 2

        return gate_types

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: Empty dictionary. The resources of this operation don't depend on any additional parameters.
        """
        return {}

    @classmethod
    def resource_rep(cls) -> re.CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation."""
        return re.CompressedResourceOp(cls, {})

    @classmethod
    def adjoint_resource_decomp(cls, **kwargs):
        r"""Returns a dictionary representing the resources for the adjoint of the operator.

        Resources:
            This operation is self-adjoint, so the resources of the adjoint operation results
            in the original operation.

        Returns:
            Dict[CompressedResourceOp, int]: The keys are the operators and the associated
                values are the counts.
        """
        return {cls.resource_rep(): 1}

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
            The resources are expressed using the symbolic :class:`~.ResourceControlled`. The resources
            are computed according to the :code:`controlled_resource_decomp()` of the base
            :class:`~.ResourceZ` class.

        Returns:
            Dict[CompressedResourceOp, int]: The keys are the operators and the associated
                values are the counts.
        """
        return {
            re.ResourceControlled.resource_rep(
                re.ResourceZ, {}, num_ctrl_wires + 2, num_ctrl_values, num_work_wires
            ): 1
        }

    @classmethod
    def pow_resource_decomp(cls, z) -> Dict[re.CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources for an operator raised to a power.

        Args:
            z (int): the power that the operator is being raised to

        Resources:
            This operation is self-inverse, thus when raised to even integer powers acts like
            the identity operator and raised to odd powers it produces itself.

        Returns:
            Dict[CompressedResourceOp, int]: The keys are the operators and the associated
                values are the counts.
        """
        return {re.ResourceIdentity.resource_rep(): 1} if z % 2 == 0 else {cls.resource_rep(): 1}


class ResourceCNOT(qml.CNOT, re.ResourceOperator):
    r"""Resource class for the CNOT gate.

    Args:
        wires (Sequence[int]): the wires the operation acts on

    Resources:
        The CNOT gate is treated as a fundamental gate and thus it cannot be decomposed
        further. Requesting the resources of this gate raises a :code:`ResourcesNotDefined` error.

    .. seealso:: :class:`~.CNOT`

    """

    @staticmethod
    def _resource_decomp(**kwargs) -> Dict[re.CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources of the operator. The
        keys are the operators and the associated values are the counts.

        Resources:
            The CNOT gate is treated as a fundamental gate and thus it cannot be decomposed
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
    def resource_rep(cls) -> re.CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation."""
        return re.CompressedResourceOp(cls, {})

    @classmethod
    def adjoint_resource_decomp(cls) -> Dict[re.CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources for the adjoint of the operator.

        Resources:
            This operation is self-adjoint, so the resources of the adjoint operation results
            in the original operation.

        Returns:
            Dict[CompressedResourceOp, int]: The keys are the operators and the associated
                values are the counts.
        """
        return {cls.resource_rep(): 1}

    @classmethod
    def controlled_resource_decomp(
        cls, num_ctrl_wires, num_ctrl_values, num_work_wires
    ) -> Dict[re.CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources for a controlled version of the operator.

        Args:
            num_ctrl_wires (int): the number of qubits the operation is controlled on
            num_ctrl_values (int): the number of control qubits, that are controlled when in the :math:`|0\rangle` state
            num_work_wires (int): the number of additional qubits that can be used for decomposition

        Resources:
            The resources are expressed as one general :class:`~.ResourceMultiControlledX` gate.

        Returns:
            Dict[CompressedResourceOp, int]: The keys are the operators and the associated
                values are the counts.
        """
        if num_ctrl_wires == 1 and num_ctrl_values == 0 and num_work_wires == 0:
            return {re.ResourceToffoli.resource_rep(): 1}

        return {
            re.ResourceMultiControlledX.resource_rep(
                num_ctrl_wires + 1, num_ctrl_values, num_work_wires
            ): 1
        }

    @classmethod
    def pow_resource_decomp(cls, z) -> Dict[re.CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources for an operator raised to a power.

        Args:
            z (int): the power that the operator is being raised to

        Resources:
            This operation is self-inverse, thus when raised to even integer powers acts like
            the identity operator and raised to odd powers it produces itself.

        Returns:
            Dict[CompressedResourceOp, int]: The keys are the operators and the associated
                values are the counts.
        """
        return {re.ResourceIdentity.resource_rep(): 1} if z % 2 == 0 else {cls.resource_rep(): 1}


class ResourceToffoli(qml.Toffoli, re.ResourceOperator):
    r"""Resource class for the Toffoli gate.

    Args:
        wires (Sequence[int]): the subsystem the gate acts on

    Resources:
        The resources are obtained from Figure 1 of `Jones 2012 <https://arxiv.org/pdf/1212.5069>`_.

        The circuit which applies the Toffoli gate on target wire 'target' with control wires
        ('c1', 'c2') is defined as:

        .. code-block:: bash

                c1: ─╭●────╭X──T†────────╭X────╭●───────────────╭●─┤
                c2: ─│──╭X─│──╭●───T†─╭●─│──╭X─│────────────────╰Z─┤
              aux1: ─╰X─│──│──╰X───T──╰X─│──│──╰X────────────────║─┤
              aux2: ──H─╰●─╰●──T─────────╰●─╰●──H──S─╭●──H──┤↗├──║─┤
            target: ─────────────────────────────────╰X──────║───║─┤
                                                             ╚═══╝

        Specifically, the resources are defined as nine :class:`~.ResourceCNOT` gates, three
        :class:`~.ResourceHadamard` gates, one :class:`~.ResourceCZ` gate, one :class:`~.ResourceS`
        gate, two :class:`~.ResourceT` gates and two adjoint :class:`~.ResourceT` gates.

    .. seealso:: :class:`~.Toffoli`

    **Example**

    The resources for this operation are computed using:

    >>> re.ResourceToffoli.resources()
    {CNOT: 9, Hadamard: 3, S: 1, CZ: 1, T: 2, Adjoint(T): 2}
    """

    @staticmethod
    def _resource_decomp(**kwargs) -> Dict[re.CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources of the operator. The
        keys are the operators and the associated values are the counts.

        Resources:
            The resources are obtained from Figure 1 of `Jones 2012 <https://arxiv.org/pdf/1212.5069>`_.

            The circuit which applies the Toffoli gate on target wire 'target' with control wires
            ('c1', 'c2') is defined as:

            .. code-block:: bash

                    c1: ─╭●────╭X──T†────────╭X────╭●───────────────╭●─┤
                    c2: ─│──╭X─│──╭●───T†─╭●─│──╭X─│────────────────╰Z─┤
                aux1: ─╰X─│──│──╰X───T──╰X─│──│──╰X────────────────║─┤
                aux2: ──H─╰●─╰●──T─────────╰●─╰●──H──S─╭●──H──┤↗├──║─┤
                target: ─────────────────────────────────╰X──────║───║─┤
                                                                ╚═══╝

            Specifically, the resources are defined as nine :class:`~.ResourceCNOT` gates, three
            :class:`~.ResourceHadamard` gates, one :class:`~.ResourceCZ` gate, one :class:`~.ResourceS`
            gate, two :class:`~.ResourceT` gates and two adjoint :class:`~.ResourceT` gates.
        """
        gate_types = {}

        cnot = re.ResourceCNOT.resource_rep()
        t = re.ResourceT.resource_rep()
        h = re.ResourceHadamard.resource_rep()
        s = re.ResourceS.resource_rep()
        cz = re.ResourceCZ.resource_rep()
        t_dag = re.ResourceAdjoint.resource_rep(re.ResourceT, {})

        gate_types[cnot] = 9
        gate_types[h] = 3
        gate_types[s] = 1
        gate_types[cz] = 1
        gate_types[t] = 2
        gate_types[t_dag] = 2

        return gate_types

    @staticmethod
    def textbook_resource_decomp() -> Dict[re.CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources of the operator. The
        keys are the operators and the associated values are the counts.

        Resources:
            The resources are taken from Figure 4.9 of `Nielsen, M. A., & Chuang, I. L. (2010) <https://www.cambridge.org/highereducation/books/quantum-computation-and-quantum-information/01E10196D0A682A6AEFFEA52D53BE9AE#overview>`_.

            The circuit is defined as:

            .. code-block:: bash

                0: ───────────╭●───────────╭●────╭●──T──╭●─┤
                1: ────╭●─────│─────╭●─────│───T─╰X──T†─╰X─┤
                2: ──H─╰X──T†─╰X──T─╰X──T†─╰X──T──H────────┤

            Specifically, the resources are defined as six :class:`~.ResourceCNOT` gates, two
            :class:`~.ResourceHadamard` gates, four :class:`~.ResourceT` gates and three adjoint
            :class:`~.ResourceT` gates.
        """
        gate_types = {}

        cnot = re.ResourceCNOT.resource_rep()
        t = re.ResourceT.resource_rep()
        h = re.ResourceHadamard.resource_rep()
        t_dag = re.ResourceAdjoint.resource_rep(re.ResourceT, {})

        gate_types[cnot] = 6
        gate_types[h] = 2
        gate_types[t] = 4
        gate_types[t_dag] = 3

        return gate_types

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: Empty dictionary. The resources of this operation don't depend on any additional parameters.
        """
        return {}

    @classmethod
    def resource_rep(cls) -> re.CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation."""
        return re.CompressedResourceOp(cls, {})

    @classmethod
    def adjoint_resource_decomp(cls) -> Dict[re.CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources for the adjoint of the operator.

        Resources:
            This operation is self-adjoint, so the resources of the adjoint operation results
            in the original operation.

        Returns:
            Dict[CompressedResourceOp, int]: The keys are the operators and the associated
                values are the counts.
        """
        return {cls.resource_rep(): 1}

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
            The resources are expressed as one general :class:`~.ResourceMultiControlledX` gate.

        Returns:
            Dict[CompressedResourceOp, int]: The keys are the operators and the associated
                values are the counts.
        """
        return {
            re.ResourceMultiControlledX.resource_rep(
                num_ctrl_wires + 2, num_ctrl_values, num_work_wires
            ): 1
        }

    @classmethod
    def pow_resource_decomp(cls, z) -> Dict[re.CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources for an operator raised to a power.

        Args:
            z (int): the power that the operator is being raised to

        Resources:
            This operation is self-inverse, thus when raised to even integer powers acts like
            the identity operator and raised to odd powers it produces itself.

        Returns:
            Dict[CompressedResourceOp, int]: The keys are the operators and the associated
                values are the counts.
        """
        return {re.ResourceIdentity.resource_rep(): 1} if z % 2 == 0 else {cls.resource_rep(): 1}


class ResourceMultiControlledX(qml.MultiControlledX, re.ResourceOperator):
    r"""Resource class for the MultiControlledX gate.

    Args:
        wires (Union[Wires, Sequence[int], or int]): control wire(s) followed by a single target wire (the last entry of ``wires``) where
            the operation acts on
        control_values (Union[bool, list[bool], int, list[int]]): The value(s) the control wire(s)
            should take. Integers other than 0 or 1 will be treated as ``int(bool(x))``.
        work_wires (Union[Wires, Sequence[int], or int]): optional work wires used to decompose
            the operation into a series of :class:`~.Toffoli` gates

    Resource Parameters:
        * num_ctrl_wires (int): the number of qubits the operation is controlled on
        * num_ctrl_values (int): the number of control qubits, that are controlled when in the :math:`|0\rangle` state
        * num_work_wires (int): the number of additional qubits that can be used for decomposition

    Resources:
        The resources are obtained from Table 3 of `Claudon, B., Zylberman, J., Feniou, C. et al.
        <https://www.nature.com/articles/s41467-024-50065-x>`_. Specifically, the
        resources are defined as the following rules:

        * If there is only one control qubit, treat the resources as a :class:`~.ResourceCNOT` gate.

        * If there are two control qubits, treat the resources as a :class:`~.ResourceToffoli` gate.

        * If there are three control qubits, the resources are two :class:`~.ResourceCNOT` gates and one :class:`~.ResourceToffoli` gate.

        * If there are more than three control qubits (:math:`n`), the resources are defined as :math:`36n - 111` :class:`~.ResourceCNOT` gates.

    .. seealso:: :class:`~.MultiControlledX`

    **Example**

    The resources for this operation are computed using:

    >>> re.ResourceMultiControlledX.resources(num_ctrl_wires=5, num_ctrl_values=2, num_work_wires=3)
    {X: 4, CNOT: 69}
    """

    @staticmethod
    def _resource_decomp(
        num_ctrl_wires,
        num_ctrl_values,
        num_work_wires,
        **kwargs,  # pylint: disable=unused-argument
    ) -> Dict[re.CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources of the operator. The
        keys are the operators and the associated values are the counts.

        Args:
            num_ctrl_wires (int): the number of qubits the operation is controlled on
            num_ctrl_values (int): the number of control qubits, that are controlled when in the :math:`|0\rangle` state
            num_work_wires (int): the number of additional qubits that can be used for decomposition

        Resources:
            The resources are obtained from Table 3 of `Claudon, B., Zylberman, J., Feniou, C. et al.
            <https://www.nature.com/articles/s41467-024-50065-x>`_. Specifically, the
            resources are defined as the following rules:

            * If there are no control qubits, treat the operation as a :class:`~.ResourceX` gate.

            * If there is only one control qubit, treat the resources as a :class:`~.ResourceCNOT` gate.

            * If there are two control qubits, treat the resources as a :class:`~.ResourceToffoli` gate.

            * If there are three control qubits, the resources are two :class:`~.ResourceCNOT` gates and
            one :class:`~.ResourceToffoli` gate.

            * If there are more than three control qubits (:math:`n`), the resources are defined as
            :math:`36n - 111` :class:`~.ResourceCNOT` gates.
        """
        gate_types = defaultdict(int)

        x = re.ResourceX.resource_rep()
        if num_ctrl_values:
            gate_types[x] = num_ctrl_values * 2

        if num_ctrl_wires == 0:
            gate_types[x] += 1
            return gate_types

        cnot = re.ResourceCNOT.resource_rep()
        if num_ctrl_wires == 1:
            gate_types[cnot] = 1
            return gate_types

        toffoli = re.ResourceToffoli.resource_rep()
        if num_ctrl_wires == 2:
            gate_types[toffoli] = 1
            return gate_types

        if num_ctrl_wires == 3:
            gate_types[cnot] = 2
            gate_types[toffoli] = 1
            return gate_types

        gate_types[cnot] = 36 * num_ctrl_wires - 111
        return gate_types

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * num_ctrl_wires (int): the number of qubits the operation is controlled on
                * num_ctrl_values (int): the number of control qubits, that are controlled when in the :math:`|0\rangle` state
                * num_work_wires (int): the number of additional qubits that can be used for decomposition
        """
        num_control = len(self.hyperparameters["control_wires"])
        num_work_wires = len(self.hyperparameters["work_wires"])

        num_control_values = len([val for val in self.hyperparameters["control_values"] if not val])

        return {
            "num_ctrl_wires": num_control,
            "num_ctrl_values": num_control_values,
            "num_work_wires": num_work_wires,
        }

    @classmethod
    def resource_rep(
        cls, num_ctrl_wires, num_ctrl_values, num_work_wires
    ) -> re.CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            num_ctrl_wires (int): the number of qubits the operation is controlled on
            num_ctrl_values (int): the number of control qubits, that are controlled when in the :math:`|0\rangle` state
            num_work_wires (int): the number of additional qubits that can be used for decomposition

        Returns:
            CompressedResourceOp: the operator in a compressed representation
        """
        return re.CompressedResourceOp(
            cls,
            {
                "num_ctrl_wires": num_ctrl_wires,
                "num_ctrl_values": num_ctrl_values,
                "num_work_wires": num_work_wires,
            },
        )

    @classmethod
    def adjoint_resource_decomp(
        cls, num_ctrl_wires, num_ctrl_values, num_work_wires
    ) -> Dict[re.CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources for the adjoint of the operator.

        Args:
            num_ctrl_wires (int): the number of qubits the operation is controlled on
            num_ctrl_values (int): the number of control qubits, that are controlled when in the :math:`|0\rangle` state
            num_work_wires (int): the number of additional qubits that can be used for decomposition

        Resources:
            This operation is self-adjoint, so the resources of the adjoint operation results
            in the original operation.

        Returns:
            Dict[CompressedResourceOp, int]: The keys are the operators and the associated
                values are the counts.
        """
        return {cls.resource_rep(num_ctrl_wires, num_ctrl_values, num_work_wires): 1}

    @classmethod
    def controlled_resource_decomp(
        cls,
        outer_num_ctrl_wires,
        outer_num_ctrl_values,
        outer_num_work_wires,
        num_ctrl_wires,
        num_ctrl_values,
        num_work_wires,
    ) -> Dict[re.CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources for a controlled version of the operator.

        Args:
            outer_num_ctrl_wires (int): The number of control qubits to further control the base
                controlled operation upon.
            outer_num_ctrl_values (int): The subset of those control qubits, which further control
                the base controlled operation, which are controlled when in the :math:`|0\rangle` state.
            outer_num_work_wires (int): the number of additional qubits that can be used in the
                decomposition for the further controlled, base control oepration.
            num_ctrl_wires (int): the number of control qubits of the operation
            num_ctrl_values (int): The subset of control qubits of the operation, that are controlled
                when in the :math:`|0\rangle` state.
            num_work_wires (int): The number of additional qubits that can be used for the
                decomposition of the operation.

        Resources:
            The resources are derived by combining the control qubits, control-values and
            work qubits into a single instance of :class:`~.ResourceMultiControlledX` gate, controlled
            on the whole set of control-qubits.

        Returns:
            Dict[CompressedResourceOp, int]: The keys are the operators and the associated
                values are the counts.
        """
        return cls.resources(
            outer_num_ctrl_wires + num_ctrl_wires,
            outer_num_ctrl_values + num_ctrl_values,
            outer_num_work_wires + num_work_wires,
        )

    @classmethod
    def pow_resource_decomp(
        cls, z, num_ctrl_wires, num_ctrl_values, num_work_wires
    ) -> Dict[re.CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources for an operator raised to a power.

        Args:
            z (int): the power that the operator is being raised to
            num_ctrl_wires (int): the number of qubits the operation is controlled on
            num_ctrl_values (int): the number of control qubits, that are controlled when in the :math:`|0\rangle` state
            num_work_wires (int): the number of additional qubits that can be used for decomposition

        Resources:
            This operation is self-inverse, thus when raised to even integer powers acts like
            the identity operator and raised to odd powers it produces itself.

        Returns:
            Dict[CompressedResourceOp, int]: The keys are the operators and the associated
                values are the counts.
        """
        return (
            {}
            if z % 2 == 0
            else {cls.resource_rep(num_ctrl_wires, num_ctrl_values, num_work_wires): 1}
        )


class ResourceCRX(qml.CRX, re.ResourceOperator):
    r"""Resource class for the CRX gate.

    Args:
        phi (float): rotation angle :math:`\phi`
        wires (Sequence[int]): the wire the operation acts on
        id (str or None): String representing the operation (optional)

    Resources:
        The resources are taken from Figure 1b of `Gheorghiu, V., Mosca, M. & Mukhopadhyay
        <https://arxiv.org/pdf/2110.10292>`_. In combination with the following identity:

        .. math:: \hat{RX} = \hat{H} \cdot \hat{RZ}  \cdot \hat{H},

        we can express the :code:`CRX` gate as a :code:`CRZ` gate conjugated by :code:`Hadamard`
        gates. The expression for controlled-RZ gates is used as defined in the reference above.
        Specifically, the resources are defined as two :class:`~.ResourceCNOT` gates, two
        :class:`~.ResourceHadamard` gates and two :class:`~.ResourceRZ` gates.

    .. seealso:: :class:`~.CRX`

    **Example**

    The resources for this operation are computed using:

    >>> re.ResourceCRX.resources()
    {CNOT: 2, RZ: 2, Hadamard: 2}
    """

    @staticmethod
    def _resource_decomp(**kwargs) -> Dict[re.CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources of the operator. The
        keys are the operators and the associated values are the counts.

        Resources:
            The resources are taken from Figure 1b of `Gheorghiu, V., Mosca, M. & Mukhopadhyay
            <https://arxiv.org/pdf/2110.10292>`_. In combination with the following identity:

            .. math:: \hat{RX} = \hat{H} \cdot \hat{RZ}  \cdot \hat{H},

            we can express the :code:`CRX` gate as a :code:`CRZ` gate conjugated by :code:`Hadamard`
            gates. The expression for controlled-RZ gates is used as defined in the reference above.
            Specifically, the resources are defined as two :class:`~.ResourceCNOT` gates, two
            :class:`~.ResourceHadamard` gates and two :class:`~.ResourceRZ` gates.
        """
        gate_types = {}

        h = re.ResourceHadamard.resource_rep()
        rz = re.ResourceRZ.resource_rep()
        cnot = re.ResourceCNOT.resource_rep()

        gate_types[cnot] = 2
        gate_types[rz] = 2
        gate_types[h] = 2

        return gate_types

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: Empty dictionary. The resources of this operation don't depend on any additional parameters.
        """
        return {}

    @classmethod
    def resource_rep(cls) -> re.CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation."""
        return re.CompressedResourceOp(cls, {})

    @classmethod
    def adjoint_resource_decomp(cls) -> Dict[re.CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources for the adjoint of the operator.

        Resources:
            The adjoint of a single qubit rotation changes the sign of the rotation angle,
            thus the resources of the adjoint operation result in the original operation.

        Returns:
            Dict[CompressedResourceOp, int]: The keys are the operators and the associated
                values are the counts.
        """
        return {cls.resource_rep(): 1}

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
            The resources are expressed using the symbolic :class:`~.ResourceControlled`. The resources
            are computed according to the :code:`controlled_resource_decomp()` of the base
            :class:`~.ResourceRX` class.

        Returns:
            Dict[CompressedResourceOp, int]: The keys are the operators and the associated
                values are the counts.
        """
        return {
            re.ResourceControlled.resource_rep(
                re.ResourceRX, {}, num_ctrl_wires + 1, num_ctrl_values, num_work_wires
            ): 1
        }

    @classmethod
    def pow_resource_decomp(cls, z) -> Dict[re.CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources for an operator raised to a power.

        Args:
            z (int): the power that the operator is being raised to

        Resources:
            Taking arbitrary powers of a single qubit rotation produces a sum of rotations.
            The resources simplify to just one total single qubit rotation.

        Returns:
            Dict[CompressedResourceOp, int]: The keys are the operators and the associated
                values are the counts.
        """
        return {cls.resource_rep(): 1}


class ResourceCRY(qml.CRY, re.ResourceOperator):
    r"""Resource class for the CRY gate.

    Args:
        phi (float): rotation angle :math:`\phi`
        wires (Sequence[int]): the wire the operation acts on
        id (str or None): String representing the operation (optional)

    Resources:
        The resources are taken from Figure 1b of `Gheorghiu, V., Mosca, M. & Mukhopadhyay
        <https://arxiv.org/pdf/2110.10292>`_. In combination with the following identity:

        .. math:: \hat{RY}(\theta) = \hat{X} \cdot \hat{RY}(- \theta) \cdot \hat{X}.

        By replacing the :class:`~.ResourceX` gates with :class:`~.ResourceCNOT` gates, we obtain a
        controlled-version of this identity. Thus we are able to constructively or destructively
        interfere the gates based on the value of the control qubit. Specifically, the resources are
        defined as two :class:`~.ResourceCNOT` gates and two :class:`~.ResourceRY` gates.

    .. seealso:: :class:`~.CRY`

    **Example**

    The resources for this operation are computed using:

    >>> re.ResourceCRY.resources()
    {CNOT: 2, RY: 2}
    """

    @staticmethod
    def _resource_decomp(**kwargs) -> Dict[re.CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources of the operator. The
        keys are the operators and the associated values are the counts.

        Resources:
            The resources are taken from Figure 1b of `Gheorghiu, V., Mosca, M. & Mukhopadhyay
            <https://arxiv.org/pdf/2110.10292>`_. In combination with the following identity:

            .. math:: \hat{RY}(\theta) = \hat{X} \cdot \hat{RY}(- \theta) \cdot \hat{X}.

            By replacing the :code:`X` gates with :code:`CNOT` gates, we obtain a controlled-version of this
            identity. Thus we are able to constructively or destructively interfere the gates based on the value
            of the control qubit. Specifically, the resources are defined as two :class:`~.ResourceCNOT` gates
            and two :class:`~.ResourceRY` gates.
        """
        gate_types = {}

        cnot = re.ResourceCNOT.resource_rep()
        ry = re.ResourceRY.resource_rep()

        gate_types[cnot] = 2
        gate_types[ry] = 2

        return gate_types

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: Empty dictionary. The resources of this operation don't depend on any additional parameters.
        """
        return {}

    @classmethod
    def resource_rep(cls) -> re.CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation."""
        return re.CompressedResourceOp(cls, {})

    @classmethod
    def adjoint_resource_decomp(cls) -> Dict[re.CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources for the adjoint of the operator.

        Resources:
            The adjoint of a single qubit rotation changes the sign of the rotation angle,
            thus the resources of the adjoint operation result in the original operation.

        Returns:
            Dict[CompressedResourceOp, int]: The keys are the operators and the associated
                values are the counts.
        """
        return {cls.resource_rep(): 1}

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
            The resources are expressed using the symbolic :class:`~.ResourceControlled`. The resources
            are computed according to the :code:`controlled_resource_decomp()` of the base
            :class:`~.ResourceRY` class.

        Returns:
            Dict[CompressedResourceOp, int]: The keys are the operators and the associated
                values are the counts.
        """
        return {
            re.ResourceControlled.resource_rep(
                re.ResourceRY, {}, num_ctrl_wires + 1, num_ctrl_values, num_work_wires
            ): 1
        }

    @classmethod
    def pow_resource_decomp(cls, z) -> Dict[re.CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources for an operator raised to a power.

        Args:
            z (int): the power that the operator is being raised to

        Resources:
            Taking arbitrary powers of a single qubit rotation produces a sum of rotations.
            The resources simplify to just one total single qubit rotation.

        Returns:
            Dict[CompressedResourceOp, int]: The keys are the operators and the associated
                values are the counts.
        """
        return {cls.resource_rep(): 1}


class ResourceCRZ(qml.CRZ, re.ResourceOperator):
    r"""Resource class for the CRZ gate.

    Args:
        phi (float): rotation angle :math:`\phi`
        wires (Sequence[int]): the wire the operation acts on
        id (str or None): String representing the operation (optional)

    Resources:
        The resources are taken from Figure 1b of `Gheorghiu, V., Mosca, M. & Mukhopadhyay
        <https://arxiv.org/pdf/2110.10292>`_. In combination with the following identity:

        .. math:: \hat{RZ}(\theta) = \hat{X} \cdot \hat{RZ}(- \theta) \cdot \hat{X}.

        By replacing the :code:`X` gates with :code:`CNOT` gates, we obtain a controlled-version of this
        identity. Thus we are able to constructively or destructively interfere the gates based on the value
        of the control qubit. Specifically, the resources are defined as two :class:`~.ResourceCNOT` gates
        and two :class:`~.ResourceRZ` gates.

    .. seealso:: :class:`~.CRZ`

    **Example**

    The resources for this operation are computed using:

    >>> re.ResourceCRZ.resources()
    {CNOT: 2, RZ: 2}
    """

    @staticmethod
    def _resource_decomp(**kwargs) -> Dict[re.CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources of the operator. The
        keys are the operators and the associated values are the counts.

        Resources:
            The resources are taken from Figure 1b of `Gheorghiu, V., Mosca, M. & Mukhopadhyay
            <https://arxiv.org/pdf/2110.10292>`_. In combination with the following identity:

            .. math:: \hat{RZ}(\theta) = \hat{X} \cdot \hat{RZ}(- \theta) \cdot \hat{X}.

            By replacing the :code:`X` gates with :code:`CNOT` gates, we obtain a controlled-version of this
            identity. Thus we are able to constructively or destructively interfere the gates based on the value
            of the control qubit. Specifically, the resources are defined as two :class:`~.ResourceCNOT` gates
            and two :class:`~.ResourceRZ` gates.
        """
        gate_types = {}

        cnot = re.ResourceCNOT.resource_rep()
        rz = re.ResourceRZ.resource_rep()

        gate_types[cnot] = 2
        gate_types[rz] = 2

        return gate_types

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: Empty dictionary. The resources of this operation don't depend on any additional parameters.
        """
        return {}

    @classmethod
    def resource_rep(cls) -> re.CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation."""
        return re.CompressedResourceOp(cls, {})

    @classmethod
    def adjoint_resource_decomp(cls) -> Dict[re.CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources for the adjoint of the operator.

        Resources:
            The adjoint of a single qubit rotation changes the sign of the rotation angle,
            thus the resources of the adjoint operation result in the original operation.

        Returns:
            Dict[CompressedResourceOp, int]: The keys are the operators and the associated
                values are the counts.
        """
        return {cls.resource_rep(): 1}

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
            The resources are expressed using the symbolic :class:`~.ResourceControlled`. The resources
            are computed according to the :code:`controlled_resource_decomp()` of the base
            :class:`~.ResourceRZ` class.

        Returns:
            Dict[CompressedResourceOp, int]: The keys are the operators and the associated
                values are the counts.
        """
        return {
            re.ResourceControlled.resource_rep(
                re.ResourceRZ, {}, num_ctrl_wires + 1, num_ctrl_values, num_work_wires
            ): 1
        }

    @classmethod
    def pow_resource_decomp(cls, z) -> Dict[re.CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources for an operator raised to a power.

        Args:
            z (int): the power that the operator is being raised to

        Resources:
            Taking arbitrary powers of a single qubit rotation produces a sum of rotations.
            The resources simplify to just one total single qubit rotation.

        Returns:
            Dict[CompressedResourceOp, int]: The keys are the operators and the associated
                values are the counts.
        """
        return {cls.resource_rep(): 1}


class ResourceCRot(qml.CRot, re.ResourceOperator):
    r"""Resource class for the CRot gate.

    Args:
        phi (float): rotation angle :math:`\phi`
        theta (float): rotation angle :math:`\theta`
        omega (float): rotation angle :math:`\omega`
        wires (Sequence[int]): the wire the operation acts on
        id (str or None): String representing the operation (optional)

    Resources:
        The resources are taken from Figure 1b of `Gheorghiu, V., Mosca, M. & Mukhopadhyay
        <https://arxiv.org/pdf/2110.10292>`_. In combination with the following identity:

        .. math::

            \begin{align}
                \hat{RZ}(\theta) = \hat{X} \cdot \hat{RZ}(- \theta) \cdot \hat{X}, \\
                \hat{RY}(\theta) = \hat{X} \cdot \hat{RY}(- \theta) \cdot \hat{X}.
            \end{align}

        This identity is applied along with some clever choices for the angle values to combine rotation;
        the final circuit takes the form:

        .. code-block:: bash

            ctrl: ─────╭●─────────╭●─────────┤
            trgt: ──RZ─╰X──RZ──RY─╰X──RY──RZ─┤

    .. seealso:: :class:`~.CRot`

    **Example**

    The resources for this operation are computed using:

    >>> re.ResourceCRot.resources()
    {CNOT: 2, RZ: 3, RY: 2}
    """

    @staticmethod
    def _resource_decomp(**kwargs) -> Dict[re.CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources of the operator. The
        keys are the operators and the associated values are the counts.

        Resources:
            The resources are taken from Figure 1b of `Gheorghiu, V., Mosca, M. & Mukhopadhyay
            <https://arxiv.org/pdf/2110.10292>`_. In combination with the following identity:

            .. math::

                \begin{align}
                    \hat{RZ}(\theta) = \hat{X} \cdot \hat{RZ}(- \theta) \cdot \hat{X}, \\
                    \hat{RY}(\theta) = \hat{X} \cdot \hat{RY}(- \theta) \cdot \hat{X}.
                \end{align}

            This identity is applied along with some clever choices for the angle values to combine rotation;
            the final circuit takes the form:

            .. code-block:: bash

                ctrl: ─────╭●─────────╭●─────────┤
                trgt: ──RZ─╰X──RZ──RY─╰X──RY──RZ─┤

        """
        gate_types = {}

        cnot = re.ResourceCNOT.resource_rep()
        rz = re.ResourceRZ.resource_rep()
        ry = re.ResourceRY.resource_rep()

        gate_types[cnot] = 2
        gate_types[rz] = 3
        gate_types[ry] = 2

        return gate_types

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: Empty dictionary. The resources of this operation don't depend on any additional parameters.
        """
        return {}

    @classmethod
    def resource_rep(cls) -> re.CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation."""
        return re.CompressedResourceOp(cls, {})

    @classmethod
    def adjoint_resource_decomp(cls) -> Dict[re.CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources for the adjoint of the operator.

        Resources:
            The adjoint of a general rotation flips the sign of the rotation angle,
            thus the resources of the adjoint operation result in the original operation.

        Returns:
            Dict[CompressedResourceOp, int]: The keys are the operators and the associated
                values are the counts.
        """
        return {cls.resource_rep(): 1}

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
            The resources are expressed using the symbolic :class:`~.ResourceControlled`. The resources
            are computed according to the :code:`controlled_resource_decomp()` of the base
            :class:`~.ResourceRot` class.

        Returns:
            Dict[CompressedResourceOp, int]: The keys are the operators and the associated
                values are the counts.
        """
        return {
            re.ResourceControlled.resource_rep(
                re.ResourceRot, {}, num_ctrl_wires + 1, num_ctrl_values, num_work_wires
            ): 1
        }

    @classmethod
    def pow_resource_decomp(cls, z) -> Dict[re.CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources for an operator raised to a power.

        Args:
            z (int): the power that the operator is being raised to

        Resources:
            Taking arbitrary powers of a general single qubit rotation produces a sum of rotations.
            The resources simplify to just one total single qubit rotation.

        Returns:
            Dict[CompressedResourceOp, int]: The keys are the operators and the associated
                values are the counts.
        """
        return {cls.resource_rep(): 1}


class ResourceControlledPhaseShift(qml.ControlledPhaseShift, re.ResourceOperator):
    r"""Resource class for the ControlledPhaseShift gate.

    Args:
        phi (float): rotation angle :math:`\phi`
        wires (Sequence[int]): the wire the operation acts on
        id (str or None): String representing the operation (optional)

    Resources:
        The resources are derived using the fact that a :class:`~.ResourcePhaseShift` gate is
        identical to the :class:`~.ResourceRZ` gate up to some global phase. Furthermore, a controlled
        global phase simplifies to a :class:`~.ResourcePhaseShift` gate. This gives rise to the
        following identity:

        .. math:: CR_\phi(\phi) = (R_\phi(\phi/2) \otimes I) \cdot CNOT \cdot (I \otimes R_\phi(-\phi/2)) \cdot CNOT \cdot (I \otimes R_\phi(\phi/2))

        Specifically, the resources are defined as two :class:`~.ResourceCNOT` gates and three
        :class:`~.ResourceRZ` gates.

    .. seealso:: :class:`~.ControlledPhaseShift`

    **Example**

    The resources for this operation are computed using:

    >>> re.ResourceControlledPhaseShift.resources()
    {CNOT: 2, RZ: 3}
    """

    @staticmethod
    def _resource_decomp(**kwargs) -> Dict[re.CompressedResourceOp, int]:
        gate_types = {}

        cnot = re.ResourceCNOT.resource_rep()
        rz = re.ResourceRZ.resource_rep()

        gate_types[cnot] = 2
        gate_types[rz] = 3

        return gate_types

    @property
    def resource_params(self):
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: Empty dictionary. The resources of this operation don't depend on any additional parameters.
        """
        return {}

    @classmethod
    def resource_rep(cls) -> re.CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation."""
        return re.CompressedResourceOp(cls, {})

    @classmethod
    def adjoint_resource_decomp(cls) -> Dict[re.CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources for the adjoint of the operator.

        Resources:
            The adjoint of a phase shift just flips the sign of the phase angle,
            thus the resources of the adjoint operation result in the original operation.

        Returns:
            Dict[CompressedResourceOp, int]: The keys are the operators and the associated
                values are the counts.
        """
        return {cls.resource_rep(): 1}

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
            The resources are expressed using the symbolic :class:`~.ResourceControlled`. The resources
            are computed according to the :code:`controlled_resource_decomp()` of the base
            :class:`~.ResourcePhaseShift` class.

        Returns:
            Dict[CompressedResourceOp, int]: The keys are the operators and the associated
                values are the counts.
        """
        return {
            re.ResourceControlled.resource_rep(
                re.ResourcePhaseShift, {}, num_ctrl_wires + 1, num_ctrl_values, num_work_wires
            ): 1
        }

    @classmethod
    def pow_resource_decomp(cls, z) -> Dict[re.CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources for an operator raised to a power.

        Args:
            z (int): the power that the operator is being raised to

        Resources:
            Taking arbitrary powers of a phase shift produces a sum of shifts.
            The resources simplify to just one total phase shift operator.

        Returns:
            Dict[CompressedResourceOp, int]: The keys are the operators and the associated
                values are the counts.
        """
        return {cls.resource_rep(): 1}
