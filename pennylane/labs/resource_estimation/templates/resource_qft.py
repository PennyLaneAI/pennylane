from typing import Dict

import pennylane as qml
from pennylane.labs.resource_estimation import (
    CompressedResourceOp,
    ResourceConstructor,
    ResourceControlledPhaseShift,
    ResourceHadamard,
    ResourceSWAP,
)

#pylint: disable=arguments-differ

class ResourceQFT(qml.QFT, ResourceConstructor):
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
