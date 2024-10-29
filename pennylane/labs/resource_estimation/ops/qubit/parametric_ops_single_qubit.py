import numpy as np
from typing import Dict

import pennylane as qml
from pennylane.labs.resource_estimation import CompressedResourceOp, ResourceConstructor

def _rotation_resources(epsilon=10e-3):
    gate_types = {}

    num_gates = round(1.149 * np.log2(1 / epsilon) + 9.2)
    t = CompressedResourceOp(qml.T, {})
    gate_types[t] = num_gates

    return gate_types

class ResourceRX(qml.RX, ResourceConstructor):
    """Resource class for RX"""

    @staticmethod
    def _resource_decomp(epsilon=10e-3) -> Dict[CompressedResourceOp, int]:
        return _rotation_resources(epsilon=epsilon)

    @staticmethod
    def compute_resource_rep(epsilon=10e-3) -> CompressedResourceOp:
        return CompressedResourceOp(qml.RX, {"epsilon": epsilon})

    def resource_rep(self, epsilon=10e-3) -> CompressedResourceOp:
        return ResourceRX.compute_resource_rep(epsilon=epsilon)

class ResourceRY(qml.RY, ResourceConstructor):
    """Resource class for RY"""

    @staticmethod
    def _resource_decomp(epsilon=10e-3) -> Dict[CompressedResourceOp, int]:
        return _rotation_resources(epsilon=epsilon)

    @staticmethod
    def compute_resource_rep(epsilon=10e-3) -> CompressedResourceOp:
        return CompressedResourceOp(qml.RY, {"epsilon": epsilon})

    def resource_rep(self, epsilon=10e-3) -> CompressedResourceOp:
        return ResourceRY.compute_resource_rep(epsilon=epsilon)

class ResourceRZ(qml.RZ, ResourceConstructor):
    """Resource class for RZ"""

    @staticmethod
    def _resource_decomp(epsilon=10e-3) -> Dict[CompressedResourceOp, int]:
        return _rotation_resources(epsilon=epsilon)

    @staticmethod
    def compute_resource_rep(epsilon=10e-3) -> CompressedResourceOp:
        return CompressedResourceOp(qml.RZ, {"epsilon": epsilon})

    def resource_rep(self, epsilon=10e-3) -> CompressedResourceOp:
        return ResourceRZ.compute_resource_rep(epsilon=epsilon)

class ResourceRot(qml.Rot, ResourceConstructor):
    """Resource class for Rot"""

    @staticmethod
    def _resource_decomp() -> Dict[CompressedResourceOp, int]:
        rx = ResourceRX.compute_resource_rep()
        ry = ResourceRY.compute_resource_rep()
        rz = ResourceRZ.compute_resource_rep()

        gate_types = {rx: 1, ry: 1, rz: 1}
        return gate_types

    @staticmethod
    def compute_resource_rep() -> CompressedResourceOp:
        return CompressedResourceOp(qml.Rot, {})

    def resource_rep(self) -> CompressedResourceOp:
        return ResourceRot.compute_resource_rep()
