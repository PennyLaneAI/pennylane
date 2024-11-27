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
    def adjoint_resource_decomp(cls, **kwargs) -> Dict[re.CompressedResourceOp, int]:
        return {cls.resource_rep(): 1}

    @staticmethod
    def controlled_resource_decomp(
        num_ctrl_wires, num_ctrl_values, num_work_wires, **kwargs
    ) -> Dict[re.CompressedResourceOp, int]:
        if num_ctrl_wires == 1 and num_ctrl_values == 1:
            return re.ResourceCH.resources(**kwargs)

        raise re.ResourcesNotDefined

    @classmethod
    def pow_resource_decomp(cls, z) -> Dict[re.CompressedResourceOp, int]:
        return {cls.resource_rep(): z % 2}


class ResourceS(qml.S, re.ResourceOperator):
    """Resource class for the S gate."""

    @staticmethod
    def _resource_decomp(**kwargs) -> Dict[re.CompressedResourceOp, int]:
        gate_types = {}
        t = ResourceT.resource_rep(**kwargs)
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

    @classmethod
    def pow_resource_decomp(cls, z) -> Dict[re.CompressedResourceOp, int]:
        return {cls.resource_rep(): z % 4}


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
    def adjoint_resource_decomp(cls, **kwargs) -> Dict[re.CompressedResourceOp, int]:
        return {cls.resource_rep(): 1}

    @staticmethod
    def controlled_resource_decomp(
        num_ctrl_wires, num_ctrl_values, num_work_wires, **kwargs
    ) -> Dict[re.CompressedResourceOp, int]:
        if num_ctrl_wires == 1 and num_ctrl_values == 1:
            return re.ResourceCSWAP.resources(**kwargs)

        raise re.ResourcesNotDefined

    @classmethod
    def pow_resource_decomp(cls, z, **kwargs) -> Dict[re.CompressedResourceOp, int]:
        return {cls.resource_rep(): z % 2}


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
    def adjoint_resource_decomp(cls, **kwargs) -> Dict[re.CompressedResourceOp, int]:
        """Resources obtained from the identity T^8 = I."""
        return {cls.resource_rep(): 7}

    @classmethod
    def pow_resource_decomp(cls, z, **kwargs) -> Dict[re.CompressedResourceOp, int]:
        """Resources obtained from the identity T^8 = I."""
        return {cls.resource_rep(): z % 8}


class ResourceX(qml.X, re.ResourceOperator):
    """Resource class for the X gate."""

    @staticmethod
    def _resource_decomp(**kwargs) -> Dict[re.CompressedResourceOp, int]:
        s = re.ResourceS.resource_rep(**kwargs)
        h = re.ResourceHadamard.resource_rep(**kwargs)

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
    def controlled_resource_decomp(num_ctrl_wires, num_ctrl_values, num_work_wires, **kwargs):
        if num_ctrl_wires == 1 and num_ctrl_values == 1:
            return re.ResourceCNOT.resources(**kwargs)
        if num_ctrl_wires == 2 and num_ctrl_values == 2:
            return re.ResourceToffoli.resources(**kwargs)

        return re.ResourceMultiControlledX.resources(
            num_ctrl_wires, num_ctrl_values, num_work_wires, **kwargs
        )

    @classmethod
    def pow_resource_decomp(cls, z) -> Dict[re.CompressedResourceOp, int]:
        return {cls.resource_rep(): z % 2}


class ResourceY(qml.Y, re.ResourceOperator):
    """Resource class for the Y gate."""

    @staticmethod
    def _resource_decomp(**kwargs) -> Dict[re.CompressedResourceOp, int]:
        s = re.ResourceS.resource_rep(**kwargs)
        h = re.ResourceHadamard.resource_rep(**kwargs)

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
        num_ctrl_wires, num_ctrl_values, num_work_wires, **kwargs
    ) -> Dict[re.CompressedResourceOp, int]:
        if num_ctrl_wires == 1 and num_ctrl_values == 1:
            return re.ResourceCY.resources(**kwargs)

        raise re.ResourcesNotDefined

    @classmethod
    def pow_resource_decomp(cls, z) -> Dict[re.CompressedResourceOp, int]:
        return {cls.resource_rep(): z % 2}


class ResourceZ(qml.Z, re.ResourceOperator):
    """Resource class for the Z gate."""

    @staticmethod
    def _resource_decomp(**kwargs) -> Dict[re.CompressedResourceOp, int]:
        s = re.ResourceS.resource_rep(**kwargs)

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
        num_ctrl_wires, num_ctrl_values, num_work_wires, **kwargs
    ) -> Dict[re.CompressedResourceOp, int]:
        if num_ctrl_wires == 1 and num_ctrl_values == 1:
            return re.ResourceCZ.resources(**kwargs)

        if num_ctrl_wires == 2 and num_ctrl_wires == 2:
            return re.ResourceCCZ.resources(**kwargs)

        raise re.ResourcesNotDefined

    @classmethod
    def pow_resource_decomp(cls, z) -> Dict[re.CompressedResourceOp, int]:
        return {cls.resource_rep(): z % 2}
