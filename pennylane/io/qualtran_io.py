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
from qualtran import (
    Bloq,
    CompositeBloq,
    Soquet,
    LeftDangle,
    Side,
    DecomposeNotImplementedError,
    DecomposeTypeError,
)
from pennylane.operation import Operation
from pennylane.wires import WiresLike

import numpy as np


def get_named_registers(registers):
    """Returns a `qml.registers` object associated with the named registers in the bloq"""

    temp_register_dict = {}
    for reg in registers:
        temp_register_dict[reg.name] = reg.bitsize

    return qml.registers(temp_register_dict)


class FromBloq(Operation):
    r"""
    A shim for using bloqs as a PennyLane operation.

    Args:
        bloq: the bloq to wrap
        wires: the wires to act on
    """

    def __init__(self, bloq: Bloq, wires: WiresLike):
        self._hyperparameters = {"bloq": bloq}
        super().__init__(wires=wires, id=None)

    def __repr__(self):
        return f'FromBloq({self._hyperparameters["bloq"]})'

    def compute_decomposition(self, wires, **kwargs):  # pylint: disable=arguments-differ
        ops = []
        bloq = self._hyperparameters["bloq"]

        try:
            cbloq = bloq.decompose_bloq() if not isinstance(bloq, CompositeBloq) else bloq
            temp_registers = get_named_registers(cbloq.signature.lefts())
            soq_to_wires = {
                Soquet(LeftDangle, idx=idx, reg=reg): list(temp_registers[reg.name])
                for reg in cbloq.signature.lefts()
                for idx in reg.all_idxs()
            }

            for binst, pred_cxns, succ_cxns in cbloq.iter_bloqnections():
                # TODO: Rename this variable to something more intuitive
                in_quregs = {
                    reg.name: np.empty((*reg.shape, reg.bitsize), dtype=object).flatten()
                    for reg in binst.bloq.signature.lefts()
                }
                out_quregs = {
                    reg.name: np.empty((*reg.shape, reg.bitsize), dtype=object).flatten()
                    for reg in binst.bloq.signature.rights()
                }

                for pred in pred_cxns:
                    soq = pred.right
                    soq_to_wires[soq] = soq_to_wires[pred.left]
                    in_quregs[soq.reg.name][soq.idx] = soq_to_wires[soq]

                total_wires = [w for ws in in_quregs.values() for w in list(ws.flatten())]
                if len(total_wires) == 0:  # if bloq decomposes to allocate + subbloqs
                    total_wires = [-1]  # dummy value
                op = binst.bloq.as_pl_op(total_wires)
                if op:
                    ops.append(op)
                for succ in succ_cxns:
                    soq = succ.left
                    if soq.reg.side == Side.RIGHT:
                        # If in_quregs is not equal to out_quregs, we insert key, value pair where the key is
                        # the register name, and the value is the list of wires associated with it
                        if len(in_quregs) != len(out_quregs) and soq.reg.side == Side.RIGHT:
                            total_elements = np.prod(soq.reg.shape) * soq.reg.bitsize
                            ascending_vals = np.arange(
                                len(soq_to_wires), total_elements + len(soq_to_wires), dtype=object
                            )
                            in_quregs[soq.reg.name] = ascending_vals.reshape(
                                (*soq.reg.shape, soq.reg.bitsize)
                            )
                        soq_to_wires[soq] = in_quregs[soq.reg.name][soq.idx]
        except (DecomposeNotImplementedError, DecomposeTypeError):
            pass

        return ops

    def compute_matrix(*params, **kwargs):  # pylint: disable=unused-argument
        bloq = params[0]._hyperparameters["bloq"]

        return bloq.tensor_contract()
