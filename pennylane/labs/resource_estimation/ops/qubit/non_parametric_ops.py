from typing import Dict

import pennylane as qml
from pennylane.labs.resource_estimation import (
    CompressedResourceOp,
    ResourceConstructor,
    ResourcesNotDefined,
)


class ResourceHadamard(qml.Hadamard, ResourceConstructor):
    """Resource class for Hadamard"""

    @staticmethod
    def _resource_decomp(*args, **kwargs) -> Dict[CompressedResourceOp, int]:
        raise ResourcesNotDefined

    def resource_rep(self) -> CompressedResourceOp:
        return CompressedResourceOp(qml.Hadamard, {})


class ResourceT(qml.T, ResourceConstructor):
    """Resource class for T"""

    @staticmethod
    def _resource_decomp(*args, **kwargs) -> Dict[CompressedResourceOp, int]:
        raise ResourcesNotDefined

    def resource_rep(self) -> CompressedResourceOp:
        return CompressedResourceOp(qml.T, {})
