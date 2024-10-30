from typing import Dict

import pennylane as qml
import pennylane.labs.resource_estimation as re

#pylint: disable=too-many-ancestors

class ResourceIdentity(qml.Identity, re.ResourceConstructor):

    @staticmethod
    def _resource_decomp() -> Dict[re.CompressedResourceOp, int]:
        return {}

    @staticmethod
    def resource_rep() -> re.CompressedResourceOp:
        return re.CompressedResourceOp(qml.Identity, {})
