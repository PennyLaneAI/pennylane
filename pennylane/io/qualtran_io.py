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
import numpy as np

import pennylane as qml
from pennylane.operation import Operation
from pennylane.wires import WiresLike

try:
    from qualtran import (
        Bloq,
        CompositeBloq,
        DecomposeNotImplementedError,
        DecomposeTypeError,
        LeftDangle,
        Side,
        Soquet,
    )
except (ModuleNotFoundError, ImportError) as import_error:
    pass


def get_bloq_registers_info(bloq):
    """Returns a `qml.registers` object associated with all named and unnamed registers and wires
    in the bloq.

    Args:
        bloq (Bloq): the bloq to get the registers info of

    Returns:
        dict: A dictionary that has all the named and un-named registers with default wire
        ordering.

    Raises:
        TypeError: bloq must be an instance of :code:`~.Bloq`.

    **Example**

    Given a qualtran bloq:

    >>> from qualtran.bloqs.basic_gates import Swap

    >>> qml.get_bloq_registers_info(Swap(3))
    {'x': Wires([0, 1, 2]), 'y': Wires([3, 4, 5])}
    """
    if not isinstance(bloq, Bloq):
        raise TypeError(f"bloq must be an instance of {Bloq}.")
    cbloq = bloq.decompose_bloq() if not isinstance(bloq, CompositeBloq) else bloq

    temp_register_dict = {reg.name: reg.bitsize for reg in cbloq.signature.rights()}

    return qml.registers(temp_register_dict)


def _get_named_registers(registers):
    """Returns a `qml.registers` object associated with the named registers in the bloq"""

    temp_register_dict = {reg.name: reg.bitsize for reg in registers}

    return qml.registers(temp_register_dict)


class FromBloq(Operation):
    r"""
    An adapter for using `Qualtran bloqs <https://qualtran.readthedocs.io/en/latest/bloqs/index.html#bloqs-library>`_
    as a PennyLane :class:`~.Operation`.

    Args:
        bloq (qualtran.Bloq): the bloq to wrap
        wires (WiresLike): the wires to act on

    Raises:
        TypeError: bloq must be an instance of :code:`~.Bloq`.

    **Example**

    Given a qualtran bloq:

    >>> from qualtran.bloqs.basic_gates import CNOT

    >>> qualtran_cnot = qml.FromBloq(CNOT(), [0, 1])
    >>> qualtran_cnot.matrix()
    array([[1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j]])

    A simple example showcasing how to use `qml.FromBloq` inside a device:

    .. code-block::

        from qualtran.bloqs.phase_estimation import RectangularWindowState, TextbookQPE
        from qualtran.bloqs.chemistry.trotter.ising import IsingXUnitary, IsingZZUnitary
        from qualtran.bloqs.chemistry.trotter.trotterized_unitary import TrotterizedUnitary

        nsites = 5
        j_zz = 2
        gamma_x = 0.1
        dt = 0.01
        indices = (0, 1, 0)
        coeffs = (0.5 * gamma_x, j_zz, 0.5 * gamma_x)
        zz_bloq = IsingZZUnitary(nsites=nsites, angle=2 * dt * j_zz)
        x_bloq = IsingXUnitary(nsites=nsites, angle=0.5 * 2 * dt * gamma_x)
        trott_unitary = TrotterizedUnitary(
            bloqs=(x_bloq, zz_bloq), indices=indices, coeffs=coeffs, timestep=dt
        )
        textbook_qpe = TextbookQPE(trott_unitary, RectangularWindowState(3))

        dev = qml.device("default.qubit")
        @qml.qnode(dev)
        def circuit():
            qml.FromBloq(textbook_qpe, wires=list(range(8)))
            return qml.state()

    """

    def __init__(self, bloq, wires: WiresLike):
        if not isinstance(bloq, Bloq):
            raise TypeError(f"bloq must be an instance of {Bloq}.")
        self._hyperparameters = {"bloq": bloq}
        super().__init__(wires=wires, id=None)

    def __repr__(self):
        return f'FromBloq({self.hyperparameters["bloq"]}, wires={self.wires})'

    @staticmethod
    def compute_decomposition(
        *params,
        wires=None,
        **hyperparameters,
    ):  # pylint: disable=arguments-differ, unused-argument
        ops = []
        bloq = hyperparameters["bloq"]

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
                    soq_to_wires_len = list(soq_to_wires.values())[-1]
                    if not isinstance(soq_to_wires_len, int):
                        soq_to_wires_len = list(soq_to_wires.values())[-1][-1]
                    soq_to_wires_len += 1

                for pred in pred_cxns:
                    soq = pred.right
                    soq_to_wires[soq] = soq_to_wires[pred.left]
                    if isinstance(soq_to_wires[soq], list) and len(soq_to_wires[soq]) == 1:
                        soq_to_wires[soq] = soq_to_wires[soq][0]
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

                total_wires = [w for ws in in_quregs.values() for w in list(ws.ravel())]
                mapped_wires = [wires[idx] for idx in total_wires]
                op = binst.bloq.as_pl_op(mapped_wires)

                if op:
                    ops.append(op)
        except (DecomposeNotImplementedError, DecomposeTypeError):
            pass

        return ops

    # pylint: disable=invalid-overridden-method, arguments-renamed
    @property
    def has_matrix(self) -> bool:
        r"""Return if the bloq has a valid matrix representation."""
        bloq = self.hyperparameters["bloq"]
        matrix = bloq.tensor_contract()
        return matrix.shape == (2 ** len(self.wires), 2 ** len(self.wires))

    def compute_matrix(*params, **hyperparams):  # pylint: disable=no-method-argument
        bloq = hyperparams["bloq"]
        matrix = bloq.tensor_contract()
        return matrix
