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

# pylint: disable=arguments-differ


class ResourceHadamard(qml.Hadamard, re.ResourceOperator):
    """Resource class for the Hadamard gate."""

    @staticmethod
    def _resource_decomp(**kwargs) -> Dict[re.CompressedResourceOp, int]:
        raise re.ResourcesNotDefined

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
        if num_ctrl_wires == 1:
            gate_types = {re.ResourceCH.resource_rep(): 1}

            if num_ctrl_values:
                gate_types[re.ResourceX.resource_rep()] = 2

            return gate_types

        ch = re.ResourceCH.resource_rep()
        mcx = re.ResourceMultiControlledX.resource_rep(
            num_ctrl_wires=num_ctrl_wires,
            num_ctrl_values=num_ctrl_values,
            num_work_wires=num_work_wires,
        )
        return {ch: 1, mcx: 2}

    @classmethod
    def pow_resource_decomp(cls, z) -> Dict[re.CompressedResourceOp, int]:
        if z % 2 == 0:
            return {}
        return {cls.resource_rep(): 1}


class ResourceS(qml.S, re.ResourceOperator):
    """Resource class for the S gate."""

    @staticmethod
    def _resource_decomp(**kwargs) -> Dict[re.CompressedResourceOp, int]:
        gate_types = {}
        t = ResourceT.resource_rep()
        gate_types[t] = 2

        return gate_types

    def resource_params(self) -> dict:
        return {}

    @classmethod
    def resource_rep(cls) -> re.CompressedResourceOp:
        return re.CompressedResourceOp(cls, {})

    @classmethod
    def adjoint_resource_decomp(cls) -> Dict[re.CompressedResourceOp, int]:
        return {cls.resource_rep(): 3}

    @staticmethod
    def controlled_resource_decomp(num_ctrl_wires, num_ctrl_values, num_work_wires):
        if num_ctrl_wires == 1:
            gate_types = {re.ResourceControlledPhaseShift.resource_rep(): 1}

            if num_ctrl_values:
                gate_types[re.ResourceX.resource_rep()] = 2

            return gate_types

        cs = re.ResourceControlledPhaseShift.resource_rep()
        mcx = re.ResourceMultiControlledX.resource_rep(
            num_ctrl_wires=num_ctrl_wires,
            num_ctrl_values=num_ctrl_values,
            num_work_wires=num_work_wires,
        )
        return {cs: 1, mcx: 2}

    @classmethod
    def pow_resource_decomp(cls, z) -> Dict[re.CompressedResourceOp, int]:
        if (mod_4 := z % 4) == 0:
            return {}
        return {cls.resource_rep(): mod_4}


class ResourceSWAP(qml.SWAP, re.ResourceOperator):
    r"""Resource class for the SWAP gate.

    Resources:
        The resources come from the following identity expressing SWAP as the product of three CNOT gates:

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

    @staticmethod
    def _resource_decomp(**kwargs) -> Dict[re.CompressedResourceOp, int]:
        gate_types = {}
        cnot = re.ResourceCNOT.resource_rep()
        gate_types[cnot] = 3

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
        if num_ctrl_wires == 1:
            gate_types = {re.ResourceCSWAP.resource_rep(): 1}

            if num_ctrl_values:
                gate_types[re.ResourceX.resource_rep()] = 2

            return gate_types

        cnot = re.ResourceCNOT.resource_rep()
        mcx = re.ResourceMultiControlledX.resource_rep(
            num_ctrl_wires=num_ctrl_wires,
            num_ctrl_values=num_ctrl_values,
            num_work_wires=num_work_wires,
        )
        return {cnot: 2, mcx: 1}

    @classmethod
    def pow_resource_decomp(cls, z) -> Dict[re.CompressedResourceOp, int]:
        if z % 2 == 0:
            return {}
        return {cls.resource_rep(): 1}


class ResourceT(qml.T, re.ResourceOperator):
    """Resource class for the T gate."""

    @staticmethod
    def _resource_decomp(**kwargs) -> Dict[re.CompressedResourceOp, int]:
        raise re.ResourcesNotDefined

    def resource_params(self) -> dict:
        return {}

    @classmethod
    def resource_rep(cls) -> re.CompressedResourceOp:
        return re.CompressedResourceOp(cls, {})

    @classmethod
    def adjoint_resource_decomp(cls) -> Dict[re.CompressedResourceOp, int]:
        """Resources obtained from the identity T^8 = I."""
        return {cls.resource_rep(): 7}

    @staticmethod
    def controlled_resource_decomp(num_ctrl_wires, num_ctrl_values, num_work_wires):
        if num_ctrl_wires == 1:
            gate_types = {re.ResourceControlledPhaseShift.resource_rep(): 1}

            if num_ctrl_values:
                gate_types[re.ResourceX.resource_rep()] = 2

            return gate_types

        ct = re.ResourceControlledPhaseShift.resource_rep()
        mcx = re.ResourceMultiControlledX.resource_rep(
            num_ctrl_wires=num_ctrl_wires,
            num_ctrl_values=num_ctrl_values,
            num_work_wires=num_work_wires,
        )
        return {ct: 1, mcx: 2}

    @classmethod
    def pow_resource_decomp(cls, z) -> Dict[re.CompressedResourceOp, int]:
        """Resources obtained from the identity T^8 = I."""
        if (mod_8 := z % 8) == 0:
            return {}
        return {cls.resource_rep(): mod_8}


class ResourceX(qml.X, re.ResourceOperator):
    """Resource class for the X gate."""

    @staticmethod
    def _resource_decomp(**kwargs) -> Dict[re.CompressedResourceOp, int]:
        s = re.ResourceS.resource_rep()
        h = re.ResourceHadamard.resource_rep()

        gate_types = {}
        gate_types[s] = 2
        gate_types[h] = 2

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
    def controlled_resource_decomp(num_ctrl_wires, num_ctrl_values, num_work_wires):
        if num_ctrl_wires > 2:
            return {
                re.ResourceMultiControlledX.resource_rep(
                    num_ctrl_wires, num_ctrl_values, num_work_wires
                ): 1
            }

        gate_types = {}
        if num_ctrl_values:
            gate_types[re.ResourceX.resource_rep()] = 2 * num_ctrl_values

        if num_ctrl_wires == 1:
            gate_types[re.ResourceCNOT.resource_rep()] = 1

        if num_ctrl_wires == 2:
            gate_types[re.ResourceToffoli.resource_rep()] = 1

        return gate_types

    @classmethod
    def pow_resource_decomp(cls, z) -> Dict[re.CompressedResourceOp, int]:
        if z % 2 == 0:
            return {}
        return {cls.resource_rep(): 1}


class ResourceY(qml.Y, re.ResourceOperator):
    """Resource class for the Y gate."""

    @staticmethod
    def _resource_decomp(**kwargs) -> Dict[re.CompressedResourceOp, int]:
        """
        The resources are defined using the identity: 

        .. math:: 

            \begin{align}
                \hat{Y} &= \hat{S} \cdot \hat{X} \cdot \hat{S}^{\dagger}, \\
                \hat{X} &= \hat{H} \cdot \hat{Z} \cdot \hat{H}, \\
                \hat{Z} &= \hat{S}^{2}, \\
                \hat{S}^{\dagger} &= 3 \hat{S}. 
            \end{align}

        """
        s = re.ResourceS.resource_rep()
        h = re.ResourceHadamard.resource_rep()

        gate_types = {}
        gate_types[s] = 6
        gate_types[h] = 2

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
        if num_ctrl_wires == 1:
            gate_types = {re.ResourceCY.resource_rep(): 1}

            if num_ctrl_values:
                gate_types[re.ResourceX.resource_rep()] = 2

            return gate_types

        cy = re.ResourceCY.resource_rep()
        mcx = re.ResourceMultiControlledX.resource_rep(
            num_ctrl_wires=num_ctrl_wires,
            num_ctrl_values=num_ctrl_values,
            num_work_wires=num_work_wires,
        )
        return {cy: 1, mcx: 2}

    @classmethod
    def pow_resource_decomp(cls, z) -> Dict[re.CompressedResourceOp, int]:
        if z % 2 == 0:
            return {}
        return {cls.resource_rep(): 1}


class ResourceZ(qml.Z, re.ResourceOperator):
    """Resource class for the Z gate."""

    @staticmethod
    def _resource_decomp(**kwargs) -> Dict[re.CompressedResourceOp, int]:
        s = re.ResourceS.resource_rep()

        gate_types = {}
        gate_types[s] = 2

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
        if num_ctrl_wires > 2:
            cz = re.ResourceCZ.resource_rep()
            mcx = re.ResourceMultiControlledX.resource_rep(
                num_ctrl_wires=num_ctrl_wires,
                num_ctrl_values=num_ctrl_values,
                num_work_wires=num_work_wires,
            )
            return {cz: 1, mcx: 2}

        gate_types = {}
        if num_ctrl_wires == 1:
            gate_types[re.ResourceCZ.resource_rep()] = 1

        if num_ctrl_wires == 2:
            gate_types[re.ResourceCCZ.resource_rep()] = 1

        if num_ctrl_values:
            gate_types[re.ResourceX.resource_rep()] = 2 * num_ctrl_values

        return gate_types

    @classmethod
    def pow_resource_decomp(cls, z) -> Dict[re.CompressedResourceOp, int]:
        if z % 2 == 0:
            return {}
        return {cls.resource_rep(): 1}
