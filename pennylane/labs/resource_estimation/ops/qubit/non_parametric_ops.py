from typing import Dict

import pennylane as qml
import pennylane.labs.resource_estimation as re


class ResourceHadamard(qml.Hadamard, re.ResourceConstructor):
    """Resource class for Hadamard"""

    @staticmethod
    def _resource_decomp(*args, **kwargs) -> Dict[re.CompressedResourceOp, int]:
        raise re.ResourcesNotDefined

    def resource_params(self) -> dict:
        return {}

    @classmethod
    def resource_rep(cls) -> re.CompressedResourceOp:
        return re.CompressedResourceOp(cls, {})


class ResourceS(qml.S, re.ResourceConstructor):
    """Resource class for S"""

    @staticmethod
    def _resource_decomp(*args, **kwargs) -> Dict[re.CompressedResourceOp, int]:
        gate_types = {}
        t = ResourceT.resource_rep()
        gate_types[t] = 2

        return gate_types

    def resource_params(self) -> dict:
        return {}

    @classmethod
    def resource_rep(cls) -> re.CompressedResourceOp:
        return re.CompressedResourceOp(cls, {})



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

    @classmethod
    def resource_rep(cls) -> re.CompressedResourceOp:
        return re.CompressedResourceOp(cls, {})


class ResourceT(qml.T, re.ResourceConstructor):
    """Resource class for T"""

    @staticmethod
    def _resource_decomp(*args, **kwargs) -> Dict[re.CompressedResourceOp, int]:
        raise re.ResourcesNotDefined

    def resource_params(self) -> dict:
        return {}

    @classmethod
    def resource_rep(cls) -> re.CompressedResourceOp:
        return re.CompressedResourceOp(cls, {})
