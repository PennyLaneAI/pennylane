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
    """Resource class for the Identity gate."""

    @staticmethod
    def _resource_decomp(*args, **kwargs) -> Dict[re.CompressedResourceOp, int]:
        return {}

    def resource_params(self) -> dict:
        return {}

    @classmethod
    def resource_rep(cls, **kwargs) -> re.CompressedResourceOp:
        return re.CompressedResourceOp(cls, {})

    @classmethod
    def adjoint_resource_decomp(cls) -> Dict[re.CompressedResourceOp, int]:
        return {cls.resource_rep(): 1}

    @classmethod
    def controlled_resource_decomp(
        cls, num_ctrl_wires, num_ctrl_values, num_work_wires
    ) -> Dict[re.CompressedResourceOp, int]:
        return {cls.resource_rep(): 1}

    @classmethod
    def pow_resource_decomp(cls, z) -> Dict[re.CompressedResourceOp, int]:
        return {cls.resource_rep(): 1}


class ResourceGlobalPhase(qml.GlobalPhase, re.ResourceOperator):
    """Resource class for the GlobalPhase gate."""

    @staticmethod
    def _resource_decomp(*args, **kwargs) -> Dict[re.CompressedResourceOp, int]:
        return {}

    def resource_params(self) -> dict:
        return {}

    @classmethod
    def resource_rep(cls, **kwargs) -> re.CompressedResourceOp:
        return re.CompressedResourceOp(cls, {})

    @staticmethod
    def adjoint_resource_decomp() -> Dict[re.CompressedResourceOp, int]:
        """The adjoint of a global phase is itself another global phase"""
        return {re.ResourceGlobalPhase.resource_rep(): 1}

    @staticmethod
    def controlled_resource_decomp(
        num_ctrl_wires, num_ctrl_values, num_work_wires
    ) -> Dict[re.CompressedResourceOp, int]:
        """
        Resources:
            The resources are generated from the identity that a global phase
            controlled on a single qubit is equivalent to a local phase shift on that qubit.

            This idea can be generalized to a multi-qubit global phase by introducing one
            'clean' auxilliary qubit which gets reset at the end of the computation. In this
            case, we sandwich the phase shift operation with two multi-controlled X gates.
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
        """Taking arbitrary powers of a global phase produces another global phase"""
        return {re.ResourceGlobalPhase.resource_rep(): 1}
