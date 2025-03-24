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
r"""Resource operators for parametric single qubit operations."""
from typing import Dict

import numpy as np

import pennylane as qml
import pennylane.labs.resource_estimation as re

# pylint: disable=arguments-differ


def _rotation_resources(epsilon=10e-3):
    r"""An estimate on the number of T gates needed to implement a Pauli rotation.

    The expected T-count is taken from (the 'Simulation Results' section) `Eﬃcient
    Synthesis of Universal Repeat-Until-Success Circuits <https://arxiv.org/abs/1404.5320>`_.
    The cost is given as:

        .. math:: T_{count} = \ceil(1.149 * log_{2}(\frac{1}{\epsilon}) + 9.2)

    Args:
        epsilon (float): the acceptable error threshold for the approximation

    Returns:
        dict: the T-gate counts

    """
    gate_types = {}

    num_gates = round(1.149 * np.log2(1 / epsilon) + 9.2)
    t = re.ResourceT.resource_rep()
    gate_types[t] = num_gates

    return gate_types


class ResourcePhaseShift(qml.PhaseShift, re.ResourceOperator):
    r"""Resource class for the PhaseShift gate.

    Args:
        phi (float): rotation angle :math:`\phi`
        wires (Sequence[int] or int): the wire the operation acts on
        id (str or None): String representing the operation (optional)

    Resources:
        The phase shift gate is equivalent to a Z-rotation upto some global phase,
        as defined from the following identity:

        .. math:: R_\phi(\phi) = e^{i\phi/2}R_z(\phi) = \begin{bmatrix}
                    1 & 0 \\
                    0 & e^{i\phi}
                \end{bmatrix}.

    .. seealso:: :class:`~.PhaseShift`

    **Example**

    The resources for this operation are computed using:

    >>> re.ResourcePhaseShift.resources()
    {RZ: 1, GlobalPhase: 1}
    """

    @staticmethod
    def _resource_decomp(**kwargs) -> Dict[re.CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources of the operator. The
        keys are the operators and the associated values are the counts.

        Resources:
            The phase shift gate is equivalent to a Z-rotation upto some global phase,
            as defined from the following identity:

            .. math:: R_\phi(\phi) = e^{i\phi/2}R_z(\phi) = \begin{bmatrix}
                        1 & 0 \\
                        0 & e^{i\phi}
                    \end{bmatrix}.
        """
        gate_types = {}
        rz = re.ResourceRZ.resource_rep()
        global_phase = re.ResourceGlobalPhase.resource_rep()
        gate_types[rz] = 1
        gate_types[global_phase] = 1

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
            The adjoint of a phase shift operator just changes the sign of the phase, thus
            the resources of the adjoint operation results in the original operation.

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
            For a single control wire, the cost is a single instance of
            :class:`~.ResourceControlledPhaseShift`. Two additional :class:`~.ResourceX` gates are used
            to flip the control qubit if it is zero-controlled.

            In the case where multiple controlled wires are provided, we can collapse the control
            wires by introducing one 'clean' auxilliary qubit (which gets reset at the end).
            In this case the cost increases by two additional :class:`~.ResourceMultiControlledX` gates,
            as described in (lemma 7.11) `Elementary gates for quantum computation <https://arxiv.org/pdf/quant-ph/9503016>`_.

        Returns:
            Dict[CompressedResourceOp, int]: The keys are the operators and the associated
                values are the counts.
        """
        if num_ctrl_wires == 1:
            gate_types = {re.ResourceControlledPhaseShift.resource_rep(): 1}

            if num_ctrl_values:
                gate_types[re.ResourceX.resource_rep()] = 2

            return gate_types

        c_ps = re.ResourceControlledPhaseShift.resource_rep()
        mcx = re.ResourceMultiControlledX.resource_rep(
            num_ctrl_wires=num_ctrl_wires,
            num_ctrl_values=num_ctrl_values,
            num_work_wires=num_work_wires,
        )
        return {c_ps: 1, mcx: 2}

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
        if z == 0:
            return {re.ResourceIdentity.resource_rep(): 1}
        return {cls.resource_rep(): 1}


class ResourceRX(qml.RX, re.ResourceOperator):
    r"""Resource class for the RX gate.

    Args:
        phi (float): rotation angle :math:`\phi`
        wires (Sequence[int] or int): the wire the operation acts on
        id (str or None): String representing the operation (optional)

    Resources:
        A single qubit rotation gate can be approximately synthesised from Clifford and T gates. The
        resources are approximating the gate with a series of T gates. The expected T-count is taken
        from (the 'Simulation Results' section) `Eﬃcient Synthesis of Universal Repeat-Until-Success
        Circuits <https://arxiv.org/abs/1404.5320>`_. The cost is given as:

        .. math:: T_{count} = \ceil(1.149 * log_{2}(\frac{1}{\epsilon}) + 9.2)

    .. seealso:: :class:`~.RX`

    **Example**

    The resources for this operation are computed using:

    >>> re.get_resources(re.ResourceRX(1.23, 0))
    {'gate_types': defaultdict(<class 'int'>, {'T': 17}), 'num_gates': 17, 'num_wires': 1}

    The operation does not require any parameters directly, however, it will depend on the single
    qubit error threshold, which can be set using a config dictionary.

    >>> config = {"error_rx": 1e-3}
    >>> re.get_resources(re.ResourceRX(1.23, 0), config=config)
    {'gate_types': defaultdict(<class 'int'>, {'T': 21}), 'num_gates': 21, 'num_wires': 1}
    """

    @staticmethod
    def _resource_decomp(config, **kwargs) -> Dict[re.CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources of the operator. The
        keys are the operators and the associated values are the counts.

        Args:
            config (dict): a dictionary containing the error threshold

        Resources:
            A single qubit rotation gate can be approximately synthesised from Clifford and T gates. The
            resources are approximating the gate with a series of T gates. The expected T-count is taken
            from (the 'Simulation Results' section) `Eﬃcient Synthesis of Universal Repeat-Until-Success
            Circuits <https://arxiv.org/abs/1404.5320>`_. The cost is given as:

            .. math:: T_{count} = \ceil(1.149 * log_{2}(\frac{1}{\epsilon}) + 9.2)

        """
        return _rotation_resources(epsilon=config["error_rx"])

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
            For a single control wire, the cost is a single instance of :class:`~.ResourceCRX`.
            Two additional :class:`~.ResourceX` gates are used to flip the control qubit if
            it is zero-controlled.

            In the case where multiple controlled wires are provided, the resources are taken
            from Figure 1b of the paper `T-count and T-depth of any multi-qubit unitary
            <https://arxiv.org/pdf/2110.10292>`_. In combination with the following identity:

            .. math:: \hat{RX} = \hat{H} \cdot \hat{RZ}  \cdot \hat{H},

            we can express the :code:`CRX` gate as a :code:`CRZ` gate conjugated by :code:`Hadamard`
            gates. The expression for controlled-RZ gates is used as defined in the reference above.
            By replacing the :code:`X` gates with multi-controlled :code:`X` gates, we obtain a
            controlled-version of that identity.

        Returns:
            Dict[CompressedResourceOp, int]: The keys are the operators and the associated
                values are the counts.
        """
        if num_ctrl_wires == 1:
            gate_types = {re.ResourceCRX.resource_rep(): 1}

            if num_ctrl_values:
                gate_types[re.ResourceX.resource_rep()] = 2

            return gate_types

        gate_types = {}

        h = re.ResourceHadamard.resource_rep()
        rz = re.ResourceRZ.resource_rep()
        mcx = re.ResourceMultiControlledX.resource_rep(
            num_ctrl_wires=num_ctrl_wires,
            num_ctrl_values=num_ctrl_values,
            num_work_wires=num_work_wires,
        )

        gate_types[mcx] = 2
        gate_types[rz] = 2
        gate_types[h] = 2

        return gate_types

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
        if z == 0:
            return {re.ResourceIdentity.resource_rep(): 1}
        return {cls.resource_rep(): 1}


class ResourceRY(qml.RY, re.ResourceOperator):
    r"""Resource class for the RY gate.

    Args:
        phi (float): rotation angle :math:`\phi`
        wires (Sequence[int] or int): the wire the operation acts on
        id (str or None): String representing the operation (optional)

    Resources:
        A single qubit rotation gate can be approximately synthesised from Clifford and T gates. The
        resources are approximating the gate with a series of T gates. The expected T-count is taken
        from (the 'Simulation Results' section) `Eﬃcient Synthesis of Universal Repeat-Until-Success
        Circuits <https://arxiv.org/abs/1404.5320>`_. The cost is given as:

        .. math:: T_{count} = \ceil(1.149 * log_{2}(\frac{1}{\epsilon}) + 9.2)

    .. seealso:: :class:`~.RY`

    **Example**

    The resources for this operation are computed using:

    >>> re.get_resources(re.ResourceRY(1.23, 0))
    {'gate_types': defaultdict(<class 'int'>, {'T': 17}), 'num_gates': 17, 'num_wires': 1}

    The operation does not require any parameters directly, however, it will depend on the single
    qubit error threshold, which can be set using a config dictionary.

    >>> config = {"error_ry": 1e-3}
    >>> re.get_resources(re.ResourceRY(1.23, 0), config=config)
    {'gate_types': defaultdict(<class 'int'>, {'T': 21}), 'num_gates': 21, 'num_wires': 1}
    """

    @staticmethod
    def _resource_decomp(config, **kwargs) -> Dict[re.CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources of the operator. The
        keys are the operators and the associated values are the counts.

        Resources:
            A single qubit rotation gate can be approximately synthesised from Clifford and T gates. The
            resources are approximating the gate with a series of T gates. The expected T-count is taken
            from (the 'Simulation Results' section) `Eﬃcient Synthesis of Universal Repeat-Until-Success
            Circuits <https://arxiv.org/abs/1404.5320>`_. The cost is given as:

            .. math:: T_{count} = \ceil(1.149 * log_{2}(\frac{1}{\epsilon}) + 9.2)

        Args:
            config (dict): a dictionary containing the error threshold
        """
        return _rotation_resources(epsilon=config["error_ry"])

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
            For a single control wire, the cost is a single instance of :class:`~.ResourceCRY`.
            Two additional :class:`~.ResourceX` gates are used to flip the control qubit if
            it is zero-controlled.

            In the case where multiple controlled wires are provided, the resources are taken
            from Figure 1b of the paper `T-count and T-depth of any multi-qubit
            unitary <https://arxiv.org/pdf/2110.10292>`_. The resources are derived with the
            following identity:

            .. math:: \hat{RY}(\theta) = \hat{X} \cdot \hat{RY}(- \theta) \cdot \hat{X}.

            By replacing the :code:`X` gates with multi-controlled :code:`X` gates, we obtain a
            controlled-version of this identity. Thus we are able to constructively or destructively
            interfere the gates based on the value of the control qubits.

        Returns:
            Dict[CompressedResourceOp, int]: The keys are the operators and the associated
                values are the counts.
        """
        if num_ctrl_wires == 1:
            gate_types = {re.ResourceCRY.resource_rep(): 1}

            if num_ctrl_values:
                gate_types[re.ResourceX.resource_rep()] = 2

            return gate_types

        ry = re.ResourceRY.resource_rep()
        mcx = re.ResourceMultiControlledX.resource_rep(
            num_ctrl_wires=num_ctrl_wires,
            num_ctrl_values=num_ctrl_values,
            num_work_wires=num_work_wires,
        )

        return {ry: 2, mcx: 2}

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
        if z == 0:
            return {re.ResourceIdentity.resource_rep(): 1}
        return {cls.resource_rep(): 1}


class ResourceRZ(qml.RZ, re.ResourceOperator):
    r"""Resource class for the RZ gate.

    Args:
        phi (float): rotation angle :math:`\phi`
        wires (Sequence[int] or int): the wire the operation acts on
        id (str or None): String representing the operation (optional)

    Resources:
        A single qubit rotation gate can be approximately synthesised from Clifford and T gates. The
        resources are approximating the gate with a series of T gates. The expected T-count is taken
        from (the 'Simulation Results' section) `Eﬃcient Synthesis of Universal Repeat-Until-Success
        Circuits <https://arxiv.org/abs/1404.5320>`_. The cost is given as:

        .. math:: T_{count} = \ceil(1.149 * log_{2}(\frac{1}{\epsilon}) + 9.2)

    .. seealso:: :class:`~.RZ`

    **Example**

    The resources for this operation are computed using:

    >>> re.get_resources(re.ResourceRZ(1.23, 0))
    {'gate_types': defaultdict(<class 'int'>, {'T': 17}), 'num_gates': 17, 'num_wires': 1}

    The operation does not require any parameters directly, however, it will depend on the single
    qubit error threshold, which can be set using a config dictionary.

    >>> config = {"error_rz": 1e-3}
    >>> re.get_resources(re.ResourceRZ(1.23, 0), config=config)
    {'gate_types': defaultdict(<class 'int'>, {'T': 21}), 'num_gates': 21, 'num_wires': 1}s
    """

    @staticmethod
    def _resource_decomp(config, **kwargs) -> Dict[re.CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources of the operator. The
        keys are the operators and the associated values are the counts.

        Resources:
            A single qubit rotation gate can be approximately synthesised from Clifford and T gates. The
            resources are approximating the gate with a series of T gates. The expected T-count is taken
            from (the 'Simulation Results' section) `Eﬃcient Synthesis of Universal Repeat-Until-Success
            Circuits <https://arxiv.org/abs/1404.5320>`_. The cost is given as:

            .. math:: T_{count} = \ceil(1.149 * log_{2}(\frac{1}{\epsilon}) + 9.2)

        Args:
            config (dict): a dictionary containing the error threshold
        """
        return _rotation_resources(epsilon=config["error_rz"])

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
            For a single control wire, the cost is a single instance of :class:`~.ResourceCRY`.
            Two additional :class:`~.ResourceX` gates are used to flip the control qubit if
            it is zero-controlled.

            In the case where multiple controlled wires are provided, the resources are obtained
            from Figure 1b of the paper `T-count and T-depth of any multi-qubit unitary
            <https://arxiv.org/pdf/2110.10292>`_. They are derived from the following identity:

            .. math:: \hat{RZ}(\theta) = \hat{X} \cdot \hat{RZ}(- \theta) \cdot \hat{X}.

            By replacing the :code:`X` gates with multi-controlled :code:`X` gates, we obtain a
            controlled-version of this identity. Thus we are able to constructively or destructively
            interfere the gates based on the value of the control qubits.

        Returns:
            Dict[CompressedResourceOp, int]: The keys are the operators and the associated
                values are the counts.
        """
        if num_ctrl_wires == 1:
            gate_types = {re.ResourceCRZ.resource_rep(): 1}

            if num_ctrl_values:
                gate_types[re.ResourceX.resource_rep()] = 2

            return gate_types

        rz = re.ResourceRZ.resource_rep()
        mcx = re.ResourceMultiControlledX.resource_rep(
            num_ctrl_wires=num_ctrl_wires,
            num_ctrl_values=num_ctrl_values,
            num_work_wires=num_work_wires,
        )

        return {rz: 2, mcx: 2}

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
        if z == 0:
            return {re.ResourceIdentity.resource_rep(): 1}
        return {cls.resource_rep(): 1}


class ResourceRot(qml.Rot, re.ResourceOperator):
    r"""Resource class for the Rot-gate.

    Args:
        phi (float): rotation angle :math:`\phi`
        theta (float): rotation angle :math:`\theta`
        omega (float): rotation angle :math:`\omega`
        wires (Any, Wires): the wire the operation acts on
        id (str or None): String representing the operation (optional)

    Resources:
        The resources are obtained according to the definition of the :class:`Rot` gate:

        .. math:: \hat{R}(\omega, \theta, \phi) = \hat{RZ}(\omega) \cdot \hat{RY}(\theta) \cdot \hat{RZ}(\phi).

    .. seealso:: :class:`~.Rot`

    **Example**

    The resources for this operation are computed using:

    >>> re.ResourceRot.resources()
    {RY: 1, RZ: 2}
    """

    @staticmethod
    def _resource_decomp(**kwargs) -> Dict[re.CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources of the operator. The
        keys are the operators and the associated values are the counts.

        Resources:
            The resources are obtained according to the definition of the :class:`Rot` gate:

            .. math:: \hat{R}(\omega, \theta, \phi) = \hat{RZ}(\omega) \cdot \hat{RY}(\theta) \cdot \hat{RZ}(\phi).

        """
        ry = ResourceRY.resource_rep()
        rz = ResourceRZ.resource_rep()

        gate_types = {ry: 1, rz: 2}
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
            The adjoint of a general single qubit rotation changes the sign of the rotation angles,
            thus the resources of the adjoint operation results in the original operation.

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
            For a single control wire, the cost is a single instance of :class:`~.ResourceCRot`.
            Two additional :class:`~.ResourceX` gates are used to flip the control qubit if
            it is zero-controlled.

            In the case where multiple controlled wires are provided, the resources are derived
            from Figure 1b of the paper `T-count and T-depth of any multi-qubit unitary
            <https://arxiv.org/pdf/2110.10292>`_. The resources are derived with the following
            identities:

            .. math::

                \begin{align}
                    \hat{RZ}(\theta) = \hat{X} \cdot \hat{RZ}(- \theta) \cdot \hat{X}, \\
                    \hat{RY}(\theta) = \hat{X} \cdot \hat{RY}(- \theta) \cdot \hat{X}.
                \end{align}

            This identity is applied along with some clever choices for the angle values to combine
            rotations; the final circuit takes the form:

            .. code-block:: bash

                ctrl: ─────╭●─────────╭●─────────┤
                trgt: ──RZ─╰X──RZ──RY─╰X──RY──RZ─┤

            The :code:`CNOT` gates are replaced with multi-controlled X-gates to generalize to the
            multi-controlled case.

        Returns:
            Dict[CompressedResourceOp, int]: The keys are the operators and the associated
                values are the counts.
        """
        if num_ctrl_wires == 1:
            gate_types = {re.ResourceCRot.resource_rep(): 1}

            if num_ctrl_values:
                gate_types[re.ResourceX.resource_rep()] = 2

            return gate_types

        gate_types = {}

        rz = re.ResourceRZ.resource_rep()
        ry = re.ResourceRY.resource_rep()
        mcx = re.ResourceMultiControlledX.resource_rep(
            num_ctrl_wires=num_ctrl_wires,
            num_ctrl_values=num_ctrl_values,
            num_work_wires=num_work_wires,
        )

        gate_types[mcx] = 2
        gate_types[rz] = 3
        gate_types[ry] = 2

        return gate_types

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
        if z == 0:
            return {re.ResourceIdentity.resource_rep(): 1}
        return {cls.resource_rep(): 1}
