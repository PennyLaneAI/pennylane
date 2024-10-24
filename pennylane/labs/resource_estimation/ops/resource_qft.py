import numpy as np
import pennylane as qml

from .. import ResourceConstructor, CompressedResourceOp, ResourcesNotDefined

#pylint: disable=too-many-ancestors,arguments-differ

class ResourceQFT(qml.QFT, ResourceConstructor):
    """Resource class for QFT"""

    @staticmethod
    def _resource_decomp(num_wires) -> dict:
        if not isinstance(num_wires, int):
            raise TypeError("num_wires must be an int.")

        if num_wires < 1:
            raise ValueError("num_wires must be greater than 0.")

        gate_types = {}

        hadamard = CompressedResourceOp(qml.Hadamard, {})
        swap = CompressedResourceOp(qml.SWAP, {})
        ctrl_phase_shift = CompressedResourceOp(qml.ControlledPhaseShift, {})

        gate_types[hadamard] = num_wires
        gate_types[swap] = num_wires // 2
        gate_types[ctrl_phase_shift] = num_wires*(num_wires - 1) // 2

        return gate_types

    def resource_rep(self) -> CompressedResourceOp:
        params = {"num_wires": len(self.wires)}
        return CompressedResourceOp(qml.QFT, params)


class ResourceControlledPhaseShift(qml.ControlledPhaseShift, ResourceConstructor):
    """Resource class for ControlledPhaseShift"""

    @staticmethod
    def _resource_decomp() -> dict:
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
    def _resource_decomp() -> dict:
        raise ResourcesNotDefined

    def resource_rep(self) -> CompressedResourceOp:
        return CompressedResourceOp(qml.CNOT, {})

class ResourceRZ(qml.RZ, ResourceConstructor):
    """Resource class for RZ"""

    @staticmethod
    def _resource_decomp(epsilon=10e-3) -> dict:
        gate_types = {}

        num_gates = round(1.149 * np.log2(1 / epsilon) + 9.2)
        t = CompressedResourceOp(qml.T, {})
        gate_types[t] = num_gates

        return gate_types

    def resource_rep(self) -> CompressedResourceOp:
        return CompressedResourceOp(qml.RZ, {})


class ResourceT(qml.T, ResourceConstructor):
    """Resource class for T"""

    @staticmethod
    def _resource_decomp() -> dict:
        raise ResourcesNotDefined

    def resource_rep(self) -> CompressedResourceOp:
        return CompressedResourceOp(qml.T, {})
