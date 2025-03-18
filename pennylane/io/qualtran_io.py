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
import numpy as np

import pennylane as qml
from pennylane.operation import DecompositionUndefinedError, MatrixUndefinedError, Operation
from pennylane.wires import WiresLike

try:
    import qualtran as qt
except (ModuleNotFoundError, ImportError) as import_error:
    pass


def get_bloq_registers_info(bloq):
    """Returns a ``qml.registers`` object associated with all named and unnamed registers and wires
    in the bloq.

    Args:
        bloq (Bloq): an initialized Qualtran bloq to be wrapped as a PennyLane operator

    Returns:
        dict: A dictionary that has all the named and un-named registers with default wire
        ordering.

    Raises:
        TypeError: bloq must be an instance of ``Bloq``.

    **Example**

    Given a qualtran bloq:

    >>> from qualtran.bloqs.phase_estimation import RectangularWindowState, TextbookQPE
    >>> textbook_qpe_small = TextbookQPE(ZPowGate(exponent=2 * 0.234), RectangularWindowState(3))
    >>> qml.get_bloq_registers_info(textbook_qpe_small)
    {'qpe_reg': Wires([0, 1, 2]), 'q': Wires([3])}
    """
    if not isinstance(bloq, qt.Bloq):
        raise TypeError(f"bloq must be an instance of {qt.Bloq}.")

    wire_register_dict = {reg.name: reg.bitsize for reg in bloq.signature.rights()}

    return qml.registers(wire_register_dict)


def _get_named_registers(registers):
    """Returns a ``qml.registers`` object associated with the named registers in the bloq"""

    temp_register_dict = {reg.name: reg.total_bits() for reg in registers}

    return qml.registers(temp_register_dict)


class FromBloq(Operation):
    r"""
    An adapter for using a `Qualtran bloq <https://qualtran.readthedocs.io/en/latest/bloqs/index.html#bloqs-library>`_
    as a PennyLane :class:`~.Operation`.

    .. note::
        This class requires the latest version of qualtran. We recommend installing the main
        branch via ``pip``:

        .. code-block:: console

            pip install git+https://github.com/quantumlib/Qualtran.git

    Args:
        bloq (qualtran.Bloq): an initialized Qualtran bloq to be wrapped as a PennyLane operator
        wires (WiresLike): The wires the operator acts on. The number of wires required can be determined by using the
            signature of the ``Bloq``, ``bloq.signature.n_qubits()``.

    Raises:
        TypeError: bloq must be an instance of ``Bloq``.

    **Example**

    Given a qualtran bloq:

    >>> from qualtran.bloqs.basic_gates import CNOT
    >>> qualtran_cnot = qml.FromBloq(CNOT(), wires=[0, 1])
    >>> qualtran_cnot.matrix()
    array([[1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j]])

    A simple example showcasing how to use ``qml.FromBloq`` inside a device:

    .. code-block::

        from qualtran.bloqs.phase_estimation import RectangularWindowState, TextbookQPE
        from qualtran.bloqs.chemistry.trotter.ising import IsingXUnitary, IsingZZUnitary
        from qualtran.bloqs.chemistry.trotter.trotterized_unitary import TrotterizedUnitary

        # Parameters for the TrotterizedUnitary
        nsites = 5
        j_zz, gamma_x = 2, 0.1
        zz_bloq = IsingZZUnitary(nsites=nsites, angle=0.02 * j_zz)
        x_bloq = IsingXUnitary(nsites=nsites, angle=0.01 * gamma_x)
        trott_unitary = TrotterizedUnitary(
            bloqs=(x_bloq, zz_bloq),  timestep=0.01,
            indices=(0, 1, 0), coeffs=(0.5 * gamma_x, j_zz, 0.5 * gamma_x)
        )

        # Instantiate the TextbookQPE and pass in our unitary
        textbook_qpe = TextbookQPE(trott_unitary, RectangularWindowState(3))

        # Execute on device
        dev = qml.device("default.qubit")
        @qml.qnode(dev)
        def circuit():
            qml.FromBloq(textbook_qpe, wires=range(textbook_qpe.signature.n_qubits()))
            return qml.probs(wires=[5, 6, 7])

    >>> circuit()
    array([0.94855734, 0.01291602, 0.00431044, 0.00267219, 0.00237645,
    0.00290337, 0.00526717, 0.02099701])

    .. details::
        :title: Usage Details

        The decomposition of a bloq wrapped in ``qml.FromBloq`` may use more wires than expected.
        For example, when we wrap Qualtran's ``CZPowGate``, we get

        >>> from qualtran.bloqs.basic_gates import CZPowGate
        >>> qml.FromBloq(CZPowGate(0.468, eps=1e-11), wires=[0, 1]).decomposition()
        [FromBloq(And, wires=Wires([0, 1, 'alloc_free2'])),
        FromBloq(Z**0.468, wires=Wires(['alloc_free2'])),
        FromBloq(And†, wires=Wires([0, 1, 'alloc_free2']))]

        This behaviour results from the decomposition of ``CZPowGate`` as defined in qualtran,
        which allocates and frees a wire all in the same ``bloq``. In this situation,
        PennyLane automatically allocates this wire under the hood, and that additional wire is
        named ``alloc_free_{idx}``. The indexing starts at the length of the wires defined in the
        signature, which in the case of ``CZPowGate`` is :math:`2`. Due to the current
        limitations of PennyLane, these wires cannot be accessed manually or mapped.

    """

    def __init__(self, bloq, wires: WiresLike):
        if not isinstance(bloq, qt.Bloq):
            raise TypeError(f"bloq must be an instance of {qt.Bloq}.")
        self._hyperparameters = {"bloq": bloq}
        super().__init__(wires=wires, id=None)

    def __repr__(self):
        return f'FromBloq({self.hyperparameters["bloq"]}, wires={self.wires})'

    @staticmethod
    def compute_decomposition(
        *params,
        wires=None,
        **hyperparameters,
    ):  # pylint: disable=arguments-differ, unused-argument, too-many-branches, too-many-statements
        ops = []
        bloq = hyperparameters["bloq"]

        if len(wires) != bloq.signature.n_qubits():
            raise ValueError(
                f"The length of wires must match the signature of {qt.Bloq}. Please provide a list of wires of length {bloq.signature.n_qubits()}"
            )

        try:
            # Bloqs need to be decomposed in order to access the connections
            cbloq = bloq.decompose_bloq() if not isinstance(bloq, qt.CompositeBloq) else bloq
            temp_registers = _get_named_registers(cbloq.signature.lefts())
            soq_to_wires = {
                qt.Soquet(qt.LeftDangle, idx=idx, reg=reg): (
                    list(temp_registers[reg.name])[idx[0]]
                    if len(idx) == 1
                    else list(temp_registers[reg.name])
                )
                for reg in cbloq.signature.lefts()
                for idx in reg.all_idxs()
            }

            # This is to track the number of wires defined at the LeftDangle stage
            # so if we need to add more wires, we know what index to start at
            soq_to_wires_len = 0
            if len(soq_to_wires.values()) > 0:
                soq_to_wires_len = list(soq_to_wires.values())[-1]
                if not isinstance(soq_to_wires_len, int):
                    soq_to_wires_len = list(soq_to_wires.values())[-1][-1]
                soq_to_wires_len += 1

            for binst, pred_cxns, succ_cxns in cbloq.iter_bloqnections():
                if isinstance(binst.bloq, qt.bloqs.bookkeeping.Partition):
                    in_quregs = {}
                    for succ in succ_cxns:
                        soq = succ.left
                        if soq.reg.side == qt.Side.RIGHT and not soq.reg.name in in_quregs:
                            soq_to_wires_len -= np.prod(soq.reg.shape) * soq.reg.bitsize

                    for succ in succ_cxns:
                        soq = succ.left
                        if soq.reg.side == qt.Side.RIGHT and not soq.reg.name in in_quregs:
                            total_elements = np.prod(soq.reg.shape) * soq.reg.bitsize
                            ascending_vals = np.arange(
                                soq_to_wires_len,
                                soq_to_wires_len + total_elements,
                                dtype=object,
                            )
                            soq_to_wires_len += total_elements
                            in_quregs[soq.reg.name] = ascending_vals.reshape(
                                (*soq.reg.shape, soq.reg.bitsize)
                            )
                        soq_to_wires[soq] = in_quregs[soq.reg.name][soq.idx]
                    continue

                in_quregs = {
                    reg.name: np.empty((*reg.shape, reg.bitsize), dtype=object)
                    for reg in binst.bloq.signature.lefts()
                }
                # The out_quregs inform us of the total # of wires in the circuit to account for
                # wires that are split or allocated in the cbloq
                out_quregs = {
                    reg.name: np.empty((*reg.shape, reg.bitsize), dtype=object)
                    for reg in binst.bloq.signature.rights()
                }

                for pred in pred_cxns:
                    soq = pred.right
                    soq_to_wires[soq] = soq_to_wires[pred.left]
                    if isinstance(soq_to_wires[soq], list) and len(soq_to_wires[soq]) == 1:
                        soq_to_wires[soq] = soq_to_wires[soq][0]
                    in_quregs[soq.reg.name][soq.idx] = np.squeeze(soq_to_wires[soq])

                for succ in succ_cxns:
                    soq = succ.left
                    if soq.reg.side == qt.Side.RIGHT:
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
                            soq_to_wires_len += total_elements
                            in_quregs[soq.reg.name] = ascending_vals.reshape(
                                (*soq.reg.shape, soq.reg.bitsize)
                            )
                        soq_to_wires[soq] = in_quregs[soq.reg.name][soq.idx]

                total_wires = [int(w) for ws in in_quregs.values() for w in list(ws.ravel())]
                mapped_wires = [wires[idx] for idx in total_wires if idx < len(wires)]
                ghost_wires = [f"alloc_free_{val}" for val in total_wires if val >= len(wires)]
                op = binst.bloq.as_pl_op(mapped_wires + ghost_wires)

                if op:
                    ops.append(op)
        except (qt.DecomposeNotImplementedError, qt.DecomposeTypeError):
            pass

        if len(ops) == 0:
            raise DecompositionUndefinedError

        return ops

    # pylint: disable=invalid-overridden-method, arguments-renamed
    @property
    def has_matrix(self) -> bool:
        r"""Return if the bloq has a valid matrix representation."""
        bloq = self.hyperparameters["bloq"]
        matrix = bloq.tensor_contract()
        return matrix.shape == (2 ** len(self.wires), 2 ** len(self.wires))

    def compute_matrix(
        *params, **hyperparams
    ):  # pylint: disable=no-method-argument, no-self-argument
        bloq = hyperparams["bloq"]
        matrix = bloq.tensor_contract()

        if matrix.shape != (2 ** len(params[0].wires), 2 ** len(params[0].wires)):
            raise MatrixUndefinedError

        return matrix
