import numpy as np
from typing import Dict

import pennylane as qml
import pennylane.labs.resource_estimation as re

def _rotation_resources(epsilon=10e-3):
    gate_types = {}

    num_gates = round(1.149 * np.log2(1 / epsilon) + 9.2)
    t = re.ResourceT.resource_rep()
    gate_types[t] = num_gates

    return gate_types

class ResourceRX(qml.RX, re.ResourceConstructor):
    """Resource class for RX"""

    @staticmethod
    def _resource_decomp(epsilon=10e-3) -> Dict[re.CompressedResourceOp, int]:
        return _rotation_resources(epsilon=epsilon)

    @staticmethod
    def resource_rep(epsilon=10e-3) -> re.CompressedResourceOp:
        return re.CompressedResourceOp(qml.RX, {"epsilon": epsilon})

class ResourceRY(qml.RY, re.ResourceConstructor):
    """Resource class for RY"""

    @staticmethod
    def _resource_decomp(epsilon=10e-3) -> Dict[re.CompressedResourceOp, int]:
        return _rotation_resources(epsilon=epsilon)

    @staticmethod
    def resource_rep(epsilon=10e-3) -> re.CompressedResourceOp:
        return re.CompressedResourceOp(qml.RY, {"epsilon": epsilon})

class ResourceRZ(qml.RZ, re.ResourceConstructor):
    """Resource class for RZ"""

    @staticmethod
    def _resource_decomp(epsilon=10e-3) -> Dict[re.CompressedResourceOp, int]:
        return _rotation_resources(epsilon=epsilon)

    @staticmethod
    def resource_rep(epsilon=10e-3) -> re.CompressedResourceOp:
        return re.CompressedResourceOp(qml.RZ, {"epsilon": epsilon})

class ResourceRot(qml.Rot, re.ResourceConstructor):
    """Resource class for Rot"""

    @staticmethod
    def _resource_decomp() -> Dict[re.CompressedResourceOp, int]:
        rx = ResourceRX.resource_rep()
        ry = ResourceRY.resource_rep()
        rz = ResourceRZ.resource_rep()

        gate_types = {rx: 1, ry: 1, rz: 1}
        return gate_types

    @staticmethod
    def resource_rep() -> re.CompressedResourceOp:
        return re.CompressedResourceOp(qml.Rot, {})
