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

    @staticmethod
    def adjoint_resource_decomp() -> Dict[re.CompressedResourceOp, int]:
        return {}

    @staticmethod
    def controlled_resource_decomp(num_ctrl_wires=1, num_ctrl_values=0, num_work_wires=0) -> Dict[re.CompressedResourceOp, int]:
        return {}

    @staticmethod
    def pow_resource_decomp(z) -> Dict[re.CompressedResourceOp, int]:
        return {}


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
        return {}

    @staticmethod
    def controlled_resource_decomp(num_ctrl_wires=1, num_ctrl_values=0, num_work_wires=0) -> Dict[re.CompressedResourceOp, int]:
        return {re.ResourcePhaseShift.resource_rep(): 1}

    @staticmethod
    def pow_resource_decomp(z) -> Dict[re.CompressedResourceOp, int]:
        return {}
