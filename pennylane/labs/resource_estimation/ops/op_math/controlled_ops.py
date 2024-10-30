from typing import Dict

import pennylane as qml
from pennylane.labs.resource_estimation import (
    CompressedResourceOp,
    ResourceConstructor,
    ResourcesNotDefined,
)

# pylint: disable=too-many-ancestors


class ResourceControlledPhaseShift(qml.ControlledPhaseShift, ResourceConstructor):
    """Resource class for ControlledPhaseShift"""

    @staticmethod
    def _resource_decomp(*args, **kwargs) -> Dict[CompressedResourceOp, int]:
        gate_types = {}

        cnot = CompressedResourceOp(qml.CNOT, {})
        rz = CompressedResourceOp(qml.RZ, {})

        gate_types[cnot] = 2
        gate_types[rz] = 3

        return gate_types

    def resource_rep(self) -> CompressedResourceOp:
        return CompressedResourceOp(qml.ControlledPhaseShift, {})


class ResourceCNOT(qml.CNOT, ResourceConstructor):
    """Resource class for CNOT"""

    @staticmethod
    def _resource_decomp(*args, **kwargs) -> Dict[CompressedResourceOp, int]:
        raise ResourcesNotDefined

    def resource_rep(self) -> CompressedResourceOp:
        return CompressedResourceOp(qml.CNOT, {})
