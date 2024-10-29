from typing import Dict

import pennylane as qml
from pennylane.labs.resource_estimation import CompressedResourceOp, ResourceConstructor, ResourcesNotDefined

class ResourceHadamard(qml.Hadamard, ResourceConstructor):
    """Resource class for Hadamard"""

    @staticmethod
    def _resource_decomp(*args, **kwargs) -> Dict[CompressedResourceOp, int]:
        raise ResourcesNotDefined

    def resource_rep(self) -> CompressedResourceOp:
        return CompressedResourceOp(qml.Hadamard, {})

class ResourceS(qml.S, ResourceConstructor):
    """Resource class for S"""

    @staticmethod
    def _resource_decomp(*args, **kwargs) -> Dict[CompressedResourceOp, int]:
        gate_types = {}
        t = ResourceT.compute_resource_rep()
        gate_types[t] = 2

        return gate_types

    @staticmethod
    def compute_resource_rep() -> CompressedResourceOp:
        return CompressedResourceOp(qml.S, {})

    def resource_rep(self) -> CompressedResourceOp:
        return ResourceS.compute_resource_rep()

class ResourceT(qml.T, ResourceConstructor):
    """Resource class for T"""

    @staticmethod
    def _resource_decomp(*args, **kwargs) -> Dict[CompressedResourceOp, int]:
        raise ResourcesNotDefined

    @staticmethod
    def compute_resource_rep() -> CompressedResourceOp:
        return CompressedResourceOp(qml.T, {})

    def resource_rep(self) -> CompressedResourceOp:
        return ResourceT.compute_resource_rep()
