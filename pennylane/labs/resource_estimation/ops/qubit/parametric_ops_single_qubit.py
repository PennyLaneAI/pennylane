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
r"""Resource operators for parametric single qubit operations."""
from typing import Dict

import numpy as np

import pennylane as qml
import pennylane.labs.resource_estimation as re

# pylint: disable=arguments-differ


def _rotation_resources(epsilon=10e-3):
    """An estimate on the number of T gates needed to implement a Pauli rotation. The estimate is taken from https://arxiv.org/abs/1404.5320."""
    gate_types = {}

    num_gates = round(1.149 * np.log2(1 / epsilon) + 9.2)
    t = re.ResourceT.resource_rep()
    gate_types[t] = num_gates

    return gate_types


class ResourcePhaseShift(qml.PhaseShift, re.ResourceOperator):
    r"""
    Resource class for the PhaseShift gate.

    The resources are defined from the following identity:

    .. math:: R_\phi(\phi) = e^{i\phi/2}R_z(\phi) = \begin{bmatrix}
                1 & 0 \\
                0 & e^{i\phi}
            \end{bmatrix}.
    """

    @staticmethod
    def _resource_decomp(**kwargs) -> Dict[re.CompressedResourceOp, int]:
        gate_types = {}
        rz = re.ResourceRZ.resource_rep()
        global_phase = re.ResourceGlobalPhase.resource_rep()
        gate_types[rz] = 1
        gate_types[global_phase] = 1

        return gate_types

    def resource_params(self) -> dict:
        return {}

    @classmethod
    def resource_rep(cls) -> re.CompressedResourceOp:
        return re.CompressedResourceOp(cls, {})

    @classmethod
    def adjoint_resource_decomp(cls) -> Dict[re.CompressedResourceOp, int]:
        return {cls.resource_rep(): 1}

    @staticmethod
    def controlled_resource_decomp(
        num_ctrl_wires, num_ctrl_values, num_work_wires
    ) -> Dict[re.CompressedResourceOp, int]:
        r"""
        The resources for a multi-controlled phase shift gate are generated using
        the identity defined in (lemma 7.11) from https://arxiv.org/pdf/quant-ph/9503016.
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
        return {cls.resource_rep(): 1}


class ResourceRX(qml.RX, re.ResourceOperator):
    """Resource class for the RX gate.
    
    Resources:
        The resources are estimated by approximating the gate with a series of T gates.
        The estimate is taken from https://arxiv.org/abs/1404.5320.
    """

    @staticmethod
    def _resource_decomp(config, **kwargs) -> Dict[re.CompressedResourceOp, int]:
        return _rotation_resources(epsilon=config["error_rx"])

    def resource_params(self) -> dict:
        return {}

    @classmethod
    def resource_rep(cls) -> re.CompressedResourceOp:
        return re.CompressedResourceOp(cls, {})

    @classmethod
    def adjoint_resource_decomp(cls) -> Dict[re.CompressedResourceOp, int]:
        return {cls.resource_rep(): 1}

    @staticmethod
    def controlled_resource_decomp(
        num_ctrl_wires, num_ctrl_values, num_work_wires
    ) -> Dict[re.CompressedResourceOp, int]:
        r"""
        Resources:
            The resources are taken from (in figure 1b.) the paper `T-count and T-depth of any multi-qubit
            unitary <https://arxiv.org/pdf/2110.10292>`_. In combination with the following identity:

            .. math:: \hat{RX} = \hat{H} \cdot \hat{RZ}  \cdot \hat{H},

            we can express the :code:`CRX` gate as a :code:`CRZ` gate conjugated by :code:`Hadamard` gates.
            The expression for controlled-RZ gates is used as defined in the reference above. By replacing
            the :code:`X` gates with multi-controlled :code:`X` gates, we obtain a controlled-version
            of that identity.
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
        return {cls.resource_rep(): 1}


class ResourceRY(qml.RY, re.ResourceOperator):
    """Resource class for the RY gate.
    
    Resources:
        The resources are estimated by approximating the gate with a series of T gates.
        The estimate is taken from https://arxiv.org/abs/1404.5320.
    """

    @staticmethod
    def _resource_decomp(config, **kwargs) -> Dict[re.CompressedResourceOp, int]:
        return _rotation_resources(epsilon=config["error_ry"])

    def resource_params(self) -> dict:
        return {}

    @classmethod
    def resource_rep(cls) -> re.CompressedResourceOp:
        return re.CompressedResourceOp(cls, {})

    @classmethod
    def adjoint_resource_decomp(cls) -> Dict[re.CompressedResourceOp, int]:
        return {cls.resource_rep(): 1}

    @staticmethod
    def controlled_resource_decomp(
        num_ctrl_wires, num_ctrl_values, num_work_wires
    ) -> Dict[re.CompressedResourceOp, int]:
        r"""
        Resources:
        The resources are taken from (in figure 1b.) the paper `T-count and T-depth of any multi-qubit
        unitary <https://arxiv.org/pdf/2110.10292>`_. The resources are derived with the following identity:

        .. math:: \hat{RY}(\theta) = \hat{X} \cdot \hat{RY}(- \theta) \cdot \hat{X}.

        By replacing the :code:`X` gates with multi-controlled :code:`X` gates, we obtain a controlled-version
        of this identity. Thus we are able to constructively or destructively interfere the gates based on the
        value of the control qubits.
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
        return {cls.resource_rep(): 1}


class ResourceRZ(qml.RZ, re.ResourceOperator):
    r"""Resource class for the RZ gate.

    Resources:
        The resources are estimated by approximating the gate with a series of T gates.
        The estimate is taken from https://arxiv.org/abs/1404.5320.
    """

    @staticmethod
    def _resource_decomp(config, **kwargs) -> Dict[re.CompressedResourceOp, int]:
        return _rotation_resources(epsilon=config["error_rz"])

    def resource_params(self) -> dict:
        return {}

    @classmethod
    def resource_rep(cls) -> re.CompressedResourceOp:
        return re.CompressedResourceOp(cls, {})

    @classmethod
    def adjoint_resource_decomp(cls) -> Dict[re.CompressedResourceOp, int]:
        return {cls.resource_rep(): 1}

    @staticmethod
    def controlled_resource_decomp(
        num_ctrl_wires, num_ctrl_values, num_work_wires
    ) -> Dict[re.CompressedResourceOp, int]:
        r"""
        The resources are obtained from (in figure 1b.) the paper `T-count and T-depth of any multi-qubit
        unitary <https://arxiv.org/pdf/2110.10292>`_. They are derived from the following identity:

        .. math:: \hat{RZ}(\theta) = \hat{X} \cdot \hat{RZ}(- \theta) \cdot \hat{X}.

        By replacing the :code:`X` gates with multi-controlled :code:`X` gates, we obtain a controlled-version of
        this identity. Thus we are able to constructively or destructively interfere the gates based on the value
        of the control qubits.
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
        return {cls.resource_rep(): 1}


class ResourceRot(qml.Rot, re.ResourceOperator):
    """Resource class for the Rot gate."""

    @staticmethod
    def _resource_decomp(**kwargs) -> Dict[re.CompressedResourceOp, int]:
        ry = ResourceRY.resource_rep()
        rz = ResourceRZ.resource_rep()

        gate_types = {ry: 1, rz: 2}
        return gate_types

    def resource_params(self):
        return {}

    @classmethod
    def resource_rep(cls) -> re.CompressedResourceOp:
        return re.CompressedResourceOp(cls, {})

    @classmethod
    def adjoint_resource_decomp(cls) -> Dict[re.CompressedResourceOp, int]:
        return {cls.resource_rep(): 1}

    @staticmethod
    def controlled_resource_decomp(
        num_ctrl_wires, num_ctrl_values, num_work_wires
    ) -> Dict[re.CompressedResourceOp, int]:
        r"""
        Resources:
            The resources are derived from (in figure 1b.) the paper `T-count and T-depth of any multi-qubit
            unitary <https://arxiv.org/pdf/2110.10292>`_. The resources are derived with the following identities:

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
            
            The :code:`CNOT` gates are replaced with multi-controlled X gates to generalize to the multi-controlled case.

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
        return {cls.resource_rep(): 1}
