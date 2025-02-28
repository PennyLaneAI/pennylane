# Copyright 2018-2025 Xanadu Quantum Technologies Inc.

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
_has_qualtran = True

import numpy as np
import pennylane as qml

from pennylane.operation import Operation
from pennylane.wires import WiresLike
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qualtran import Bloq

try:
    import qualtran  # pylint: disable=unused-import
    from qualtran import (
        Bloq,
        CompositeBloq,
        Soquet,
        LeftDangle,
        Side,
        DecomposeNotImplementedError,
        DecomposeTypeError,
    )
except (ModuleNotFoundError, ImportError) as import_error:  # pragma: no cover
    _has_qualtran = False


def get_bloq_registers_info(bloq):
    """Returns a `qml.registers` object associated with all named and unnamed registers and wires
    in the bloq.

    Args:
        bloq: the bloq to get the registers info of

    Returns:
        dict: A dictionary that has all the named and un-named registers with default wire
        ordering.

    **Example**

    Given a qualtran bloq:

    from qualtran.bloqs.basic_gates import Swap

    >>> qml.get_bloq_registers_info(Swap(3))
    {'x': Wires([0, 1, 2]), 'y': Wires([3, 4, 5])}
    """
    if not _has_qualtran:
        raise ImportError(
            "Qualtran is required for get_bloq_registers_info. Please install it with `pip install qualtran`"
        )
    cbloq = bloq.decompose_bloq() if not isinstance(bloq, CompositeBloq) else bloq

    temp_register_dict = {reg.name: reg.bitsize for reg in cbloq.signature.rights()}

    return qml.registers(temp_register_dict)


def _get_named_registers(registers):
    """Returns a `qml.registers` object associated with the named registers in the bloq"""

    temp_register_dict = {reg.name: reg.bitsize for reg in registers}

    return qml.registers(temp_register_dict)


class FromBloq(Operation):
    r"""
    A shim for using bloqs as a PennyLane operation.

    Args:
        bloq: the bloq to wrap
        wires: the wires to act on

    **Example**

    Given a qualtran bloq:

    from qualtran.bloqs.basic_gates import CNOT

    >>> qualtran_toffoli = qml.FromBloq(CNOT(), [0, 1])
    >>> qualtran_toffoli.matrix()
    array([[1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j]])

    A simple example showcasing how to use `qml.FromBloq` inside a device:

    .. code-block::

        dev = qml.device("default.qubit")

        @qml.qnode(dev)
        def circuit():
            qml.FromBloq(XGate(), [0])
            return qml.expval(qml.Z(wires=[0]))

    >>> circuit()
    -1.0
    """

    def __init__(self, bloq: Bloq, wires: WiresLike):
        assert isinstance(bloq, Bloq)
        self._hyperparameters = {"bloq": bloq}
        super().__init__(wires=wires, id=None)

    def __repr__(self):  # pylint: disable=protected-access
        return f'FromBloq({self._hyperparameters["bloq"]}, wires={self.wires})'

    def compute_decomposition(
        self, wires, **kwargs
    ):  # pylint: disable=arguments-differ, unused-argument
        ops = []
        bloq = self._hyperparameters["bloq"]

        try:
            # Bloqs need to be decomposed in order to access the connections
            cbloq = bloq.decompose_bloq() if not isinstance(bloq, CompositeBloq) else bloq
            temp_registers = _get_named_registers(cbloq.signature.lefts())
            soq_to_wires = {
                Soquet(LeftDangle, idx=idx, reg=reg): list(temp_registers[reg.name])
                for reg in cbloq.signature.lefts()
                for idx in reg.all_idxs()
            }

            for binst, pred_cxns, succ_cxns in cbloq.iter_bloqnections():
                in_quregs = {
                    reg.name: np.empty((*reg.shape, reg.bitsize), dtype=object).flatten()
                    for reg in binst.bloq.signature.lefts()
                }
                # The out_quregs inform us of the total # of wires in the circuit to account for
                # wires that are split or allocated in the cbloq
                out_quregs = {
                    reg.name: np.empty((*reg.shape, reg.bitsize), dtype=object).flatten()
                    for reg in binst.bloq.signature.rights()
                }

                # This is to track the number of wires defined at the LeftDangle stage
                # so if we need to add more wires, we know what index to start at
                soq_to_wires_len = 0
                if len(soq_to_wires.values()) > 0:
                    try:
                        soq_to_wires_len = list(soq_to_wires.values())[-1][-1] + 1
                    except (IndexError, TypeError):
                        soq_to_wires_len = list(soq_to_wires.values())[-1] + 1

                for pred in pred_cxns:
                    soq = pred.right
                    soq_to_wires[soq] = soq_to_wires[pred.left]
                    in_quregs[soq.reg.name][soq.idx] = soq_to_wires[soq]

                for succ in succ_cxns:
                    soq = succ.left
                    if soq.reg.side == Side.RIGHT:
                        # When in_quregs != out_quregs, it means that there are wires unaccounted
                        # for. We account for these wires and update soq_to_wires and in_quregs
                        # accordingly.
                        if len(in_quregs) != len(out_quregs):
                            total_elements = np.prod(soq.reg.shape) * soq.reg.bitsize
                            ascending_vals = np.arange(
                                soq_to_wires_len,
                                total_elements + soq_to_wires_len,
                                dtype=object,
                            )
                            in_quregs[soq.reg.name] = ascending_vals.reshape(
                                (*soq.reg.shape, soq.reg.bitsize)
                            )
                        soq_to_wires[soq] = in_quregs[soq.reg.name][soq.idx]

                total_wires = [w for ws in in_quregs.values() for w in list(ws.flatten())]

                mapped_wires = [wires[idx] for idx in total_wires]
                op = binst.bloq.as_pl_op(mapped_wires)

                if op:
                    ops.append(op)
        except (DecomposeNotImplementedError, DecomposeTypeError):
            pass

        return ops

    @property
    def has_matrix(
        self,
    ) -> bool:  # pylint: disable=invalid-overridden-method, protected-access, arguments-renamed
        r"""Return if the bloq has a valid matrix representation."""
        bloq = self._hyperparameters["bloq"]
        matrix = bloq.tensor_contract()
        return matrix.shape == (2 ** len(self.wires), 2 ** len(self.wires))

    def compute_matrix(
        *params, **kwargs
    ):  # pylint: disable=unused-argument, no-self-argument, no-method-argument, protected-access
        bloq = params[0]._hyperparameters["bloq"]
        matrix = bloq.tensor_contract()
        return matrix
