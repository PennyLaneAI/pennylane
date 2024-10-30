from typing import Dict

import pennylane as qml
<<<<<<< HEAD
from pennylane.labs.resource_estimation import (
    CompressedResourceOp,
    ResourceConstructor,
    ResourcesNotDefined,
)

=======
import pennylane.labs.resource_estimation as re
>>>>>>> resource_qft

class ResourceHadamard(qml.Hadamard, re.ResourceConstructor):
    """Resource class for Hadamard"""

    @staticmethod
    def _resource_decomp(*args, **kwargs) -> Dict[re.CompressedResourceOp, int]:
        raise re.ResourcesNotDefined

    def resource_params(self) -> dict:
        return {}

<<<<<<< HEAD

class ResourceT(qml.T, ResourceConstructor):
=======
    @staticmethod
    def resource_rep() -> re.CompressedResourceOp:
        return re.CompressedResourceOp(qml.Hadamard, {})

class ResourceSWAP(qml.SWAP, re.ResourceConstructor):
    """Resource class for SWAP"""

    @staticmethod
    def _resource_decomp(*args, **kwargs) -> Dict[re.CompressedResourceOp, int]:
        gate_types = {}
        cnot = re.ResourceCNOT.resource_rep()
        gate_types[cnot] = 3

        return gate_types

    def resource_params(self) -> dict:
        return {}

    @staticmethod
    def resource_rep() -> re.CompressedResourceOp:
        return re.CompressedResourceOp(qml.SWAP, {})

class ResourceT(qml.T, re.ResourceConstructor):
>>>>>>> resource_qft
    """Resource class for T"""

    @staticmethod
    def _resource_decomp(*args, **kwargs) -> Dict[re.CompressedResourceOp, int]:
        raise re.ResourcesNotDefined

    def resource_params(self) -> dict:
        return {}

    @staticmethod
    def resource_rep() -> re.CompressedResourceOp:
        return re.CompressedResourceOp(qml.T, {})
