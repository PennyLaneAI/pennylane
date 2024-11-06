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
r"""Resource operators for controlled operations."""
from typing import Dict

import pennylane as qml
import pennylane.labs.resource_estimation as re

# pylint: disable=arguments-differ,too-many-ancestors


class ResourceCH(qml.CH, re.ResourceConstructor):
    r"""Resource class for CH gate.
    
    Resources:
        The resources are derived from the following identities:
        
        .. math:: 
            
            \begin{align}
                \hat{H} &= \hat{R}_{y}(\frac{\pi}{4}) \dot \hat{Z}  \dot \hat{R}_{y}(\frac{-\pi}{4}), \\
                \hat{Z} &= \hat{H} \dot \hat{X}  \dot \hat{H}
            \end{align}
        

        We can control on the Pauli-X gate to obtain our controlled Hadamard gate. 

    """

    # TODO: Reference this:
    # https://quantumcomputing.stackexchange.com/questions/15734/how-to-construct-a-controlled-hadamard-gate-using-single-qubit-gates-and-control

    @staticmethod
    def _resource_decomp(**kwargs) -> Dict[re.CompressedResourceOp, int]:
        gate_types = {}

        ry = re.ResourceRY.resource_rep()
        h = re.ResourceHadamard.resource_rep()
        cnot = re.ResourceCNOT.resource_rep()

        gate_types[h] = 2
        gate_types[ry] = 2
        gate_types[cnot] = 1

        return gate_types

    def resource_params(self) -> dict:
        return {}

    @classmethod
    def resource_rep(cls) -> re.CompressedResourceOp:
        return re.CompressedResourceOp(cls, {})


class ResourceCY(qml.CY, re.ResourceConstructor):
    """Resource class for CY gate.

    Resources:
        The resources are derived from the following identity:

        .. math:: \hat{Y} = \hat{S} \dot \hat{X} \dot \hat{S}^{\dagger}.

        We can control on the Pauli-X gate to obtain our controlled-Y gate.

    """

    @staticmethod
    def _resource_decomp(**kwargs) -> Dict[re.CompressedResourceOp, int]:
        gate_types = {}

        cnot = re.ResourceCNOT.resource_rep()
        s = re.ResourceS.resource_rep()

        gate_types[cnot] = 1
        gate_types[s] = 2  # Assuming S^dagg ~ S in cost!

        return gate_types

    def resource_params(self) -> dict:
        return {}

    @classmethod
    def resource_rep(cls) -> re.CompressedResourceOp:
        return re.CompressedResourceOp(cls, {})


class ResourceCZ(qml.CZ, re.ResourceConstructor):
    """Resource class for CZ

    Resources:
        The resources are derived from the following identity:

        .. math:: \hat{Z} = \hat{H} \dot \hat{X} \dot \hat{H}.

        We can control on the Pauli-X gate to obtain our controlled-Z gate.

    """

    @staticmethod
    def _resource_decomp(**kwargs) -> Dict[re.CompressedResourceOp, int]:
        gate_types = {}

        cnot = re.ResourceCNOT.resource_rep()
        h = re.ResourceHadamard.resource_rep()

        gate_types[cnot] = 1
        gate_types[h] = 2

        return gate_types

    def resource_params(self) -> dict:
        return {}

    @classmethod
    def resource_rep(cls) -> re.CompressedResourceOp:
        return re.CompressedResourceOp(cls, {})


class ResourceCSWAP(qml.CSWAP, re.ResourceConstructor):
    """Resource class for CSWAP

    Resources:
        The resources are taken from the paper "Shallow unitary decompositions
        of quantum Fredkin and Toffoli gates for connectivity-aware equivalent circuit averaging"
        (figure 1d).
    """  # TODO: Reference this https://arxiv.org/pdf/2305.18128

    @staticmethod
    def _resource_decomp(**kwargs) -> Dict[re.CompressedResourceOp, int]:
        gate_types = {}

        tof = re.ResourceToffoli.resource_rep()
        cnot = re.ResourceCNOT.resource_rep()

        gate_types[tof] = 1
        gate_types[cnot] = 2

        return gate_types

    def resource_params(self) -> dict:
        return {}

    @classmethod
    def resource_rep(cls) -> re.CompressedResourceOp:
        return re.CompressedResourceOp(cls, {})


class ResourceCCZ(qml.CCZ, re.ResourceConstructor):
    """Resource class for CCZ"""

    @staticmethod
    def _resource_decomp(**kwargs) -> Dict[re.CompressedResourceOp, int]:
        raise re.ResourcesNotDefined

    def resource_params(self) -> dict:
        return {}

    @classmethod
    def resource_rep(cls) -> re.CompressedResourceOp:
        return re.CompressedResourceOp(cls, {})


class ResourceCNOT(qml.CNOT, re.ResourceConstructor):
    """Resource class for CNOT"""

    @staticmethod
    def _resource_decomp(**kwargs) -> Dict[re.CompressedResourceOp, int]:
        raise re.ResourcesNotDefined

    def resource_params(self) -> dict:
        return {}

    @classmethod
    def resource_rep(cls) -> re.CompressedResourceOp:
        return re.CompressedResourceOp(cls, {})


class ResourceToffoli(qml.Toffoli, re.ResourceConstructor):
    """Resource class for Toffoli"""

    @staticmethod
    def _resource_decomp(**kwargs) -> Dict[re.CompressedResourceOp, int]:
        raise re.ResourcesNotDefined

    def resource_params(self) -> dict:
        return {}

    @classmethod
    def resource_rep(cls) -> re.CompressedResourceOp:
        return re.CompressedResourceOp(cls, {})


class ResourceMultiControlledX(qml.MultiControlledX, re.ResourceConstructor):
    """Resource class for MultiControlledX"""

    @staticmethod
    def _resource_decomp(**kwargs) -> Dict[re.CompressedResourceOp, int]:
        raise re.ResourcesNotDefined

    def resource_params(self) -> dict:
        return {}

    @classmethod
    def resource_rep(cls) -> re.CompressedResourceOp:
        return re.CompressedResourceOp(cls, {})


class ResourceCRX(qml.CRX, re.ResourceConstructor):
    """Resource class for CRX"""

    @staticmethod
    def _resource_decomp(**kwargs) -> Dict[re.CompressedResourceOp, int]:
        raise re.ResourcesNotDefined

    def resource_params(self) -> dict:
        return {}

    @classmethod
    def resource_rep(cls) -> re.CompressedResourceOp:
        return re.CompressedResourceOp(cls, {})


class ResourceCRY(qml.CRY, re.ResourceConstructor):
    """Resource class for CRY"""

    @staticmethod
    def _resource_decomp(**kwargs) -> Dict[re.CompressedResourceOp, int]:
        raise re.ResourcesNotDefined

    def resource_params(self) -> dict:
        return {}

    @classmethod
    def resource_rep(cls) -> re.CompressedResourceOp:
        return re.CompressedResourceOp(cls, {})


class ResourceCRZ(qml.CRZ, re.ResourceConstructor):
    """Resource class for CRZ"""

    @staticmethod
    def _resource_decomp(**kwargs) -> Dict[re.CompressedResourceOp, int]:
        raise re.ResourcesNotDefined

    def resource_params(self) -> dict:
        return {}

    @classmethod
    def resource_rep(cls) -> re.CompressedResourceOp:
        return re.CompressedResourceOp(cls, {})


class ResourceCRot(qml.CRot, re.ResourceConstructor):
    """Resource class for CRot"""

    @staticmethod
    def _resource_decomp(**kwargs) -> Dict[re.CompressedResourceOp, int]:
        raise re.ResourcesNotDefined

    def resource_params(self) -> dict:
        return {}

    @classmethod
    def resource_rep(cls) -> re.CompressedResourceOp:
        return re.CompressedResourceOp(cls, {})


class ResourceControlledPhaseShift(qml.ControlledPhaseShift, re.ResourceConstructor):
    """Resource class for ControlledPhaseShift"""

    @staticmethod
    def _resource_decomp(**kwargs) -> Dict[re.CompressedResourceOp, int]:
        gate_types = {}

        cnot = re.ResourceCNOT.resource_rep()
        rz = re.ResourceRZ.resource_rep()

        gate_types[cnot] = 2
        gate_types[rz] = 3

        return gate_types

    def resource_params(self):
        return {}

    @classmethod
    def resource_rep(cls) -> re.CompressedResourceOp:
        return re.CompressedResourceOp(cls, {})
