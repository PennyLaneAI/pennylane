from typing import Dict

import pennylane as qml
import pennylane.labs.resource_estimation as re

# pylint: disable=too-many-ancestors


class ResourceIdentity(qml.Identity, re.ResourceConstructor):
    """Resource class for Identity"""

    @staticmethod
    def _resource_decomp() -> Dict[re.CompressedResourceOp, int]:
        return {}

    def resource_params(self) -> dict:
        return {}

    @classmethod
    def resource_rep(cls) -> re.CompressedResourceOp:
        return re.CompressedResourceOp(cls, {})


class ResourceGlobalPhase(qml.GlobalPhase, re.ResourceConstructor):
    """Resource class for GlobalPhase"""

    @staticmethod
    def _resource_decomp() -> Dict[re.CompressedResourceOp, int]:
        return {}

    def resource_params(self) -> dict:
        return {}

    @classmethod
    def resource_rep(cls) -> re.CompressedResourceOp:
        return re.CompressedResourceOp(cls, {})
