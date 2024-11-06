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
r"""Resource operator for the QFT template."""
from typing import Dict

import pennylane as qml
from pennylane.labs.resource_estimation import (
    CompressedResourceOp,
    ResourceOperator,
    ResourceControlledPhaseShift,
    ResourceHadamard,
    ResourceSWAP,
)

# pylint: disable=arguments-differ


class ResourceQFT(qml.QFT, ResourceOperator):
    """Resource class for QFT"""

    @staticmethod
    def _resource_decomp(num_wires) -> Dict[CompressedResourceOp, int]:
        if not isinstance(num_wires, int):
            raise TypeError("num_wires must be an int.")

        if num_wires < 1:
            raise ValueError("num_wires must be greater than 0.")

        gate_types = {}

        hadamard = ResourceHadamard.resource_rep()
        swap = ResourceSWAP.resource_rep()
        ctrl_phase_shift = ResourceControlledPhaseShift.resource_rep()

        gate_types[hadamard] = num_wires
        gate_types[swap] = num_wires // 2
        gate_types[ctrl_phase_shift] = num_wires * (num_wires - 1) // 2

        return gate_types

    def resource_params(self):
        return {"num_wires": len(self.wires)}

    @classmethod
    def resource_rep(cls, num_wires) -> CompressedResourceOp:
        params = {"num_wires": num_wires}
        return CompressedResourceOp(cls, params)
