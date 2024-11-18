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


class ResourceControlledPhaseShift(qml.ControlledPhaseShift, re.ResourceOperator):
    r"""Resource class for ControlledPhaseShift
        
        Resources:
            The resources come from the following identity expressing Controlled Phase Shift
            as a product of Phase Shifts and CNOTs.


        .. math:: 

            CR_\phi(\phi) = \begin{bmatrix}
                    1 & 0 & 0 & 0 \\
                    0 & 1 & 0 & 0 \\
                    0 & 0 & 1 & 0 \\
                    0 & 0 & 0 & e^{i\phi}
                \end{bmatrix} =
            (R_\phi(\phi/2) \otimes I) \cdot CNOT \cdot (I \otimes R_\phi(-\phi/2)) \cdot CNOT \cdot (I \otimes R_\phi(\phi/2))    


    """

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


class ResourceCNOT(qml.CNOT, re.ResourceOperator):
    """Resource class for CNOT"""

    @staticmethod
    def _resource_decomp(**kwargs) -> Dict[re.CompressedResourceOp, int]:
        raise re.ResourcesNotDefined

    def resource_params(self) -> dict:
        return {}

    @classmethod
    def resource_rep(cls) -> re.CompressedResourceOp:
        return re.CompressedResourceOp(cls, {})
