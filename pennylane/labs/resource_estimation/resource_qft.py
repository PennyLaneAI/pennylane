import pennylane as qml
from pennylane.labs.resource_estimation import ResourceConstructor, CompressedResourceOp

#pylint: disable=too-many-ancestors,arguments-differ

class ResourceQFT(qml.QFT, ResourceConstructor):
    """Resource class for QFT"""

    @staticmethod
    def compute_resources(num_wires) -> dict:
        gate_types = {}

        hadamard = CompressedResourceOp(qml.Hadamard, ())
        swap = CompressedResourceOp(qml.SWAP, ())
        ctrl_phase_shift = CompressedResourceOp(qml.ControlledPhaseShift, ())

        gate_types[hadamard] = num_wires
        gate_types[swap] = num_wires // 2
        gate_types[ctrl_phase_shift] = num_wires*(num_wires - 1) // 2

        return gate_types

    def resource_rep(self) -> CompressedResourceOp:
        params = (('num_wires', len(self.wires)),)
        return CompressedResourceOp(qml.QFT, params)


class ResourceControlledPhaseShift(qml.ControlledPhaseShift, ResourceConstructor):
    """Resource class for ControlledPhaseShift"""

    @staticmethod
    def compute_resources() -> dict:
        gate_types = {}

        cnot = CompressedResourceOp(qml.CNOT, ())
        rz = CompressedResourceOp(qml.RZ, ())

        gate_types[cnot] = 2
        gate_types[rz] = 3

        return gate_types

    def resource_rep(self) -> CompressedResourceOp:
        return CompressedResourceOp(qml.ControlledPhaseShift, ())

class ResourceCNOT(qml.CNOT, ResourceConstructor):
    """Resource class for CNOT"""

class ResourceRZ(qml.RZ, ResourceConstructor):
    """Resource class for RZ"""
