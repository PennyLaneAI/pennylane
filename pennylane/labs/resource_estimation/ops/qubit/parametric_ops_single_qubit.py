from typing import Dict

import numpy as np

import pennylane as qml
import pennylane.labs.resource_estimation as re

#pylint: disable=arguments-differ


def _rotation_resources(epsilon=10e-3):
    gate_types = {}

    num_gates = round(1.149 * np.log2(1 / epsilon) + 9.2)
    t = re.ResourceT.resource_rep()
    gate_types[t] = num_gates

    return gate_types


class ResourcePhaseShift(qml.PhaseShift, re.ResourceConstructor):
    """Resource class for PhaseShift"""

    @staticmethod
    def _resource_decomp() -> Dict[re.CompressedResourceOp, int]:
        gate_types = {}
        rz = re.ResourceRZ.resource_rep()
        global_phase = re.ResourceGlobalPhase.resource_rep()
        gate_types[rz] = 1
        gate_types[global_phase] = 1

        return gate_types

    def resource_params(self) -> dict:
        return {}

    @classmethod
    def resource_rep(cls) -> re.CompressedResourceOp:
        return re.CompressedResourceOp(cls, {})


class ResourceRX(qml.RX, re.ResourceConstructor):
    """Resource class for RX"""

    @staticmethod
    def _resource_decomp(config) -> Dict[re.CompressedResourceOp, int]:
        return _rotation_resources(epsilon=config['error_rx'])

    def resource_params(self) -> dict:
        return {}

    @classmethod
    def resource_rep(cls) -> re.CompressedResourceOp:
        return re.CompressedResourceOp(cls, {})


class ResourceRY(qml.RY, re.ResourceConstructor):
    """Resource class for RY"""

    @staticmethod
    def _resource_decomp(config) -> Dict[re.CompressedResourceOp, int]:
        return _rotation_resources(epsilon=config['error_ry'])

    def resource_params(self) -> dict:
        return {}

    @classmethod
    def resource_rep(cls) -> re.CompressedResourceOp:
        return re.CompressedResourceOp(cls, {})


class ResourceRZ(qml.RZ, re.ResourceConstructor):
    """Resource class for RZ"""

    @staticmethod
    def _resource_decomp(config) -> Dict[re.CompressedResourceOp, int]:
        return _rotation_resources(epsilon=config['error_rz'])

    def resource_params(self) -> dict:
        return {}

    @classmethod
    def resource_rep(cls) -> re.CompressedResourceOp:
        return re.CompressedResourceOp(cls, {})


class ResourceRot(qml.Rot, re.ResourceConstructor):
    """Resource class for Rot"""

    @staticmethod
    def _resource_decomp() -> Dict[re.CompressedResourceOp, int]:
        rx = ResourceRX.resource_rep()
        ry = ResourceRY.resource_rep()
        rz = ResourceRZ.resource_rep()

        gate_types = {rx: 1, ry: 1, rz: 1}
        return gate_types

    def resource_params(self):
        return {}

    @classmethod
    def resource_rep(cls) -> re.CompressedResourceOp:
        return re.CompressedResourceOp(cls, {})
