# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This submodule contains the adapter class for Qualtran-PennyLane interoperability.
"""
# pylint:disable=

import pennylane as qml
from qualtran import Bloq, CompositeBloq, Soquet, LeftDangle, Register
from pennylane.operation import Operation
from pennylane.wires import Wires, WiresLike


BLOQ_TO_OP_MAP = {
    "XGate": qml.X,
    "YGate": qml.Y,
    "ZGate": qml.Z,
    "Hadamard": qml.Hadamard,
    "CNOT": qml.CNOT,
    "CZ": qml.CZ,
    "TwoBitSwap": qml.SWAP,
    "Rx": qml.RX,
    "Ry": qml.RY,
    "Rz": qml.RZ,
    "Identity": qml.I,
    "TwoBitCSwap": qml.CSWAP,
    "GlobalPhase": qml.GlobalPhase,
    "Toffoli": qml.Toffoli,
    "SGate": qml.S,
    "TGate": qml.T,
    "CYGate": qml.CY,
    "CHadamard": qml.CH,
}

def get_named_registers(registers):
    """Returns a `qml.registers` object with the juices"""

    temp_register_dict = {}
    for reg in registers:
        temp_register_dict[reg.name] = reg.bitsize

    return qml.registers(temp_register_dict)


class FromBloq(Operation):
    r"""
    A shim for using bloqs as a PennyLane operation.

    Args:
        bloq: The bloq to wrap.
    """

    def __init__(self, bloq: Bloq, wires: WiresLike):
        self._hyperparameters = {
            "bloq": bloq
        }
        super().__init__(wires=wires, id=None)

    def compute_decomposition(self, wires, **kwargs):
        b = self._hyperparameters["bloq"]
        if type(b).__name__ in BLOQ_TO_OP_MAP:
            pl_op = BLOQ_TO_OP_MAP[type(b).__name__]
            params = []
            if hasattr(b, "angle"):
                params.append(b.angle)
            
            return [pl_op(*params, wires=wires)]
        
        if isinstance(b, CompositeBloq):
            temp_registers = get_named_registers(b.signature.lefts())
            qvar_to_qreg = {
                Soquet(LeftDangle, reg.name, idx): temp_registers[reg.name]
                for reg in b.signature.lefts()
                for idx in reg.all_idxs()
            }

            for binst, pred_cxns, succ_cxns, in b.flatten().iter_bloqnections():
                for pred in pred_cxns:
                    qvar_to_qreg[pred.right] = Wires(qvar_to_qreg[pred.left])

        return [qml.X(1)]
