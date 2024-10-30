from typing import Dict

import numpy as np

import pennylane as qml
from pennylane.labs.resource_estimation import CompressedResourceOp, ResourceConstructor


def _rotation_resources(epsilon=10e-3):
    gate_types = {}

    num_gates = round(1.149 * np.log2(1 / epsilon) + 9.2)
    t = CompressedResourceOp(qml.T, {})
    gate_types[t] = num_gates

    return gate_types


class ResourceRZ(qml.RZ, ResourceConstructor):
    """Resource class for RZ"""

    @staticmethod
    def _resource_decomp(epsilon=10e-3) -> Dict[CompressedResourceOp, int]:
        return _rotation_resources(epsilon=epsilon)

    def resource_rep(self, epsilon=10e-3) -> CompressedResourceOp:
        return CompressedResourceOp(qml.RZ, {"epsilon": epsilon})
