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
from qualtran import Bloq, CompositeBloq, Soquet, LeftDangle, Side
from pennylane.operation import Operation
from pennylane.wires import Wires, WiresLike

import numpy as np


def get_named_registers(registers):
    """Returns a `qml.registers` object with the appropriate Wires"""

    temp_register_dict = {}
    for reg in registers:
        temp_register_dict[reg.name] = reg.bitsize

    return qml.registers(temp_register_dict)


def bloq_to_op(bloq, wires):
    BLOQ_TO_OP_MAP = {
        "XGate": qml.PauliX,
        "YGate": qml.PauliY,
        "ZGate": qml.PauliZ,
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

    if isinstance(wires, dict):
        total_wires = []
        for ws in wires.values():
            for w in list(ws.flatten()):
                total_wires.append(w)

        wires = Wires(total_wires)

    if type(bloq).__name__ in BLOQ_TO_OP_MAP:
        pl_op = BLOQ_TO_OP_MAP[type(bloq).__name__]
        params = []
        if hasattr(bloq, "angle"):
            params.append(bloq.angle)

        return pl_op(*params, wires=wires)
    return None


class FromBloq(Operation):
    r"""
    A shim for using bloqs as a PennyLane operation.

    Args:
        bloq: The bloq to wrap.
    """

    def __init__(self, bloq: Bloq, wires: WiresLike):
        self._hyperparameters = {"bloq": bloq}
        super().__init__(wires=wires, id=None)

    def compute_decomposition(self, wires, **kwargs):
        ops = []
        bloq = self._hyperparameters["bloq"]

        if isinstance(bloq, CompositeBloq):
            temp_registers = get_named_registers(bloq.signature.lefts())
            qvar_to_qreg = {
                Soquet(LeftDangle, idx=idx, reg=reg): list(temp_registers[reg.name])
                for reg in bloq.signature.lefts()
                for idx in reg.all_idxs()
            }

            for (
                binst,
                pred_cxns,
                succ_cxns,
            ) in bloq.iter_bloqnections():
                in_quregs = {
                    reg.name: np.empty((*reg.shape, reg.bitsize), dtype=object).flatten()
                    for reg in binst.bloq.signature.lefts()
                }
                for pred in pred_cxns:
                    soq = pred.right
                    # assert soq in qvar_to_qreg, f"{soq=} should exist in {qvar_to_qreg=}."
                    qvar_to_qreg[soq] = qvar_to_qreg[pred.left]
                    in_quregs[soq.reg.name][soq.idx] = qvar_to_qreg[soq]
                    # if soq.reg.side == Side.LEFT:
                    #     del qvar_to_qreg[soq]
                op = bloq_to_op(binst.bloq, in_quregs)
                if op:
                    ops.append(op)
                for succ in succ_cxns:
                    soq = succ.left
                    if len(in_quregs) == 0 and soq.reg.side == Side.RIGHT:
                        total_elements = np.prod(soq.reg.shape) * soq.reg.bitsize
                        ascending_vals = np.arange(
                            len(qvar_to_qreg), total_elements + len(qvar_to_qreg), dtype=object
                        )
                        in_quregs[soq.reg.name] = ascending_vals.reshape(
                            (*soq.reg.shape, soq.reg.bitsize)
                        )
                    if succ.left.reg.side == Side.RIGHT:
                        qvar_to_qreg[soq] = in_quregs[soq.reg.name][soq.idx]
        else:
            op = bloq_to_op(bloq, wires)
            ops.append(op)

        return ops
