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
from collections import defaultdict
from functools import cached_property, lru_cache, singledispatch
from typing import TYPE_CHECKING, Dict, List

import numpy as np

import pennylane as qml
from pennylane.operation import DecompositionUndefinedError, MatrixUndefinedError, Operation
from pennylane.wires import WiresLike

try:
    import qualtran as qt
    from attrs import frozen
except (ModuleNotFoundError, ImportError) as import_error:
    qualtran = None

if TYPE_CHECKING:
    from qualtran import BloqBuilder


# pylint: disable=unused-argument
@lru_cache
def _get_to_pl_op():
    @singledispatch
    def _to_pl_op(bloq, wires):
        return FromBloq(bloq=bloq, wires=wires)

    @_to_pl_op.register
    def _(bloq: qt.bloqs.basic_gates.CNOT, wires):
        return qml.CNOT(wires=wires)

    @_to_pl_op.register
    def _(bloq: qt.bloqs.basic_gates.GlobalPhase, wires):
        return qml.GlobalPhase(bloq.exponent * np.pi, wires)

    @_to_pl_op.register
    def _(bloq: qt.bloqs.basic_gates.Hadamard, wires):
        return qml.Hadamard(wires)

    @_to_pl_op.register
    def _(bloq: qt.bloqs.basic_gates.Identity, wires):
        return qml.Identity(wires)

    @_to_pl_op.register
    def _(bloq: qt.bloqs.basic_gates.Rx, wires):
        return qml.RX(bloq.angle, wires)

    @_to_pl_op.register
    def _(bloq: qt.bloqs.basic_gates.Ry, wires):
        return qml.RY(bloq.angle, wires)

    @_to_pl_op.register
    def _(bloq: qt.bloqs.basic_gates.Rz, wires):
        return qml.RZ(bloq.angle, wires)

    @_to_pl_op.register
    def _(bloq: qt.bloqs.basic_gates.SGate, wires):
        return qml.adjoint(qml.S(wires)) if bloq.is_adjoint else qml.S(wires)

    @_to_pl_op.register
    def _(bloq: qt.bloqs.basic_gates.TwoBitSwap, wires):
        return qml.SWAP(wires)

    @_to_pl_op.register
    def _(bloq: qt.bloqs.basic_gates.TwoBitCSwap, wires):
        return qml.CSWAP(wires)

    @_to_pl_op.register
    def _(bloq: qt.bloqs.basic_gates.TGate, wires):
        return qml.adjoint(qml.T(wires)) if bloq.is_adjoint else qml.T(wires)

    @_to_pl_op.register
    def _(bloq: qt.bloqs.basic_gates.Toffoli, wires):
        return qml.Toffoli(wires)

    @_to_pl_op.register
    def _(bloq: qt.bloqs.basic_gates.XGate, wires):
        return qml.X(wires)

    @_to_pl_op.register
    def _(bloq: qt.bloqs.basic_gates.YGate, wires):
        return qml.Y(wires)

    @_to_pl_op.register
    def _(bloq: qt.bloqs.basic_gates.CYGate, wires):
        return qml.CY(wires)

    @_to_pl_op.register
    def _(bloq: qt.bloqs.basic_gates.ZGate, wires):
        return qml.Z(wires)

    @_to_pl_op.register
    def _(bloq: qt.bloqs.basic_gates.CZ, wires):
        return qml.CZ(wires)

    @_to_pl_op.register(qt.bloqs.bookkeeping.Allocate)
    @_to_pl_op.register(qt.bloqs.bookkeeping.Cast)
    @_to_pl_op.register(qt.bloqs.bookkeeping.Free)
    @_to_pl_op.register(qt.bloqs.bookkeeping.Join)
    @_to_pl_op.register(qt.bloqs.bookkeeping.Partition)
    @_to_pl_op.register(qt.bloqs.bookkeeping.Split)
    def _(bloq, wires):
        return None

    return _to_pl_op


def bloq_registers(bloq):
    """Reads a `Qualtran Bloq <https://qualtran.readthedocs.io/en/latest/bloqs/index.html#bloqs-library>`_
    signature and returns a dictionary mapping the Bloq's register names to :class:`~.Wires`.

    .. note::
        This function requires the latest version of Qualtran. We recommend installing the main
        branch via ``pip``:

        .. code-block:: console

            pip install qualtran

    The keys of the ``qml.registers`` dictionary are the register names in the Qualtran Bloq. The
    values are :class:`~.Wires` objects with a length equal to the bitsize of its respective
    register. The wires are indexed in ascending order, starting from 0.

    This function makes it easy to access the wires that a Bloq acts on and use them to precisely
    control how gates connect.

    Args:
        bloq (Bloq): an initialized Qualtran ``Bloq`` to be wrapped as a PennyLane operator

    Returns:
        dict: A dictionary built with information from the Bloq's signature. The dictionary keys
        are strings that come from the names of the Bloq's registers. The values are :class:`~.Wires`
        objects that are determined by the bitsizes of those same registers.

    Raises:
        TypeError: bloq must be an instance of ``Bloq``.

    **Example**

    This example shows how to find the estimation wires of a textbook Quantum Phase Estimation Bloq.

    >>> from qualtran.bloqs.phase_estimation import RectangularWindowState, TextbookQPE
    >>> from qualtran.bloqs.basic_gates import ZPowGate
    >>> textbook_qpe_small = TextbookQPE(ZPowGate(exponent=2 * 0.234), RectangularWindowState(3))
    >>> qml.bloq_registers(textbook_qpe_small)
    {'q': Wires([0]), 'qpe_reg': Wires([1, 2, 3])}
    """

    if not isinstance(bloq, qt.Bloq):
        raise TypeError(f"bloq must be an instance of {qt.Bloq}.")

    wire_register_dict = defaultdict()

    for reg in bloq.signature.lefts():
        wire_register_dict[reg.name] = reg.bitsize

    for reg in bloq.signature.rights():
        wire_register_dict[reg.name] = reg.bitsize

    return qml.registers(wire_register_dict)


def _get_named_registers(registers):
    """Returns a ``qml.registers`` object associated with the named registers in the bloq"""

    temp_register_dict = {reg.name: reg.total_bits() for reg in registers}

    return qml.registers(temp_register_dict)


def _preprocess_bloq(bloq):
    """Processes a bloq's information to prepare for decomposition"""

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

    return cbloq, soq_to_wires, soq_to_wires_len


class FromBloq(Operation):
    r"""
    An adapter for using a `Qualtran Bloq <https://qualtran.readthedocs.io/en/latest/bloqs/index.html#bloqs-library>`_
    as a PennyLane :class:`~.Operation`.

    .. note::
        This class requires the latest version of Qualtran. We recommend installing the main
        branch via ``pip``:

        .. code-block:: console

            pip install qualtran

    Args:
        bloq (qualtran.Bloq): an initialized Qualtran ``Bloq`` to be wrapped as a PennyLane operator
        wires (WiresLike): The wires the operator acts on. The number of wires can be determined by using the
            signature of the ``Bloq`` using ``bloq.signature.n_qubits()``.

    Raises:
        TypeError: bloq must be an instance of ``Bloq``.

    **Example**

    This example shows how to use ``qml.FromBloq``:

    >>> from qualtran.bloqs.basic_gates import CNOT
    >>> qualtran_cnot = qml.FromBloq(CNOT(), wires=[0, 1])
    >>> qualtran_cnot.matrix()
    array([[1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j]])

    This example shows how to use ``qml.FromBloq`` inside a device:

    >>> from qualtran.bloqs.basic_gates import CNOT
    >>> dev = qml.device("default.qubit") # Execute on device
    >>> @qml.qnode(dev)
    ... def circuit():
    ...     qml.FromBloq(CNOT(), wires=[0, 1])
    ...     return qml.state()
    >>> circuit()
    array([1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j])

    .. details::
        :title: Advanced Example

        This example shows how to use ``qml.FromBloq`` to implement a textbook Quantum Phase Estimation Bloq inside a device:

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

            # Instantiate the TextbookQPE and pass in the unitary
            textbook_qpe = TextbookQPE(trott_unitary, RectangularWindowState(3))

            # Execute on device
            dev = qml.device("default.qubit")
            @qml.qnode(dev)
            def circuit():
                qml.FromBloq(textbook_qpe, wires=range(textbook_qpe.signature.n_qubits()))
                return qml.probs(wires=[5, 6, 7])

            circuit()

    .. details::
        :title: Usage Details

        The decomposition of a ``Bloq`` wrapped in ``qml.FromBloq`` may use more wires than expected.
        For example, when we wrap Qualtran's ``CZPowGate``, we get

        >>> from qualtran.bloqs.basic_gates import CZPowGate
        >>> qml.FromBloq(CZPowGate(0.468, eps=1e-11), wires=[0, 1]).decomposition()
        [FromBloq(And, wires=Wires([0, 1, 'alloc_free_2'])),
        FromBloq(Z**0.468, wires=Wires(['alloc_free_2'])),
        FromBloq(And†, wires=Wires([0, 1, 'alloc_free_2']))]

        This behaviour results from the decomposition of ``CZPowGate`` as defined in Qualtran,
        which allocates and frees a wire in the same ``bloq``. In this situation,
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
    def compute_decomposition(wires, bloq):  # pylint: disable=arguments-differ, too-many-branches
        ops = []

        if len(wires) != bloq.signature.n_qubits():
            raise ValueError(
                f"The length of wires must match the signature of {qt.Bloq}. Please provide a list of wires of length {bloq.signature.n_qubits()}"
            )

        try:
            cbloq, soq_to_wires, soq_to_wires_len = _preprocess_bloq(bloq)

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
                op = _get_to_pl_op()(binst.bloq, mapped_wires + ghost_wires)
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
        r"""Return if the ``Bloq`` has a valid matrix representation."""
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


def split_qubits(registers, qubits):  # type: ignore[type-var]
    """Splits the flat list of qubits into a dictionary of appropriately shaped qubit arrays."""

    qubit_regs = {}
    base = 0
    for reg in registers:
        qubit_regs[reg.name] = np.array(qubits[base : base + reg.total_bits()]).reshape(
            reg.shape + (reg.bitsize,)
        )
        base += reg.total_bits()
    return qubit_regs


def _ensure_in_reg_exists(
    bb: BloqBuilder,
    in_reg: qt.cirq_interop._cirq_to_bloq._QReg,
    qreg_to_qvar: Dict[qt.cirq_interop._cirq_to_bloq._QReg, qt.Soquet],
) -> None:
    """Takes care of qubit allocations, split and joins to ensure `qreg_to_qvar[in_reg]` exists."""
    all_mapped_qubits = {q for qreg in qreg_to_qvar for q in qreg.qubits}
    qubits_to_allocate = [q for q in in_reg.qubits if q not in all_mapped_qubits]
    if qubits_to_allocate:
        n_alloc = len(qubits_to_allocate)
        qreg_to_qvar[
            qt.cirq_interop._cirq_to_bloq._QReg(
                qubits_to_allocate, dtype=qt.QBit() if n_alloc == 1 else qt.QAny(n_alloc)
            )
        ] = bb.allocate(n_alloc)

    if in_reg in qreg_to_qvar:
        # This is the easy case when no split / joins are needed.
        return

    # a. Split all registers containing at-least one qubit corresponding to `in_reg`.
    in_reg_qubits = set(in_reg.qubits)

    new_qreg_to_qvar: Dict[qt.cirq_interop._cirq_to_bloq._QReg, qt.Soquet] = {}
    for qreg, soq in qreg_to_qvar.items():
        if len(qreg.qubits) > 1 and any(q in qreg.qubits for q in in_reg_qubits):
            new_qreg_to_qvar |= {
                qt.cirq_interop._cirq_to_bloq._QReg(q, qt.QBit()): s
                for q, s in zip(qreg.qubits, bb.split(soq=soq))
            }
        else:
            new_qreg_to_qvar[qreg] = soq
    qreg_to_qvar.clear()

    # b. Join all 1-bit registers, corresponding to individual qubits, that make up `in_reg`.
    soqs_to_join = {}
    for qreg, soq in new_qreg_to_qvar.items():
        if len(in_reg_qubits) > 1 and qreg.qubits and qreg.qubits[0] in in_reg_qubits:
            assert len(qreg.qubits) == 1, "Individual qubits should have been split by now."
            # Cast single bit registers to QBit to preserve signature of later join.
            if not isinstance(qreg.dtype, qt.QBit):
                soqs_to_join[qreg.qubits[0]] = bb.add(qt.bloqs.bookkeeping.Cast(qreg.dtype, qt.QBit()), reg=soq)
            else:
                soqs_to_join[qreg.qubits[0]] = soq
        elif len(in_reg_qubits) == 1 and qreg.qubits and qreg.qubits[0] in in_reg_qubits:
            # Cast single QBit registers to the appropriate single-bit register dtype.
            err_msg = (
                "Found non-QBit type register which shouldn't happen: "
                f"{soq.reg.name} {soq.reg.dtype}"
            )
            assert isinstance(soq.reg.dtype, qt.QBit), err_msg
            if not isinstance(in_reg.dtype, qt.QBit):
                qreg_to_qvar[in_reg] = bb.add(qt.bloqs.bookkeeping.Cast(qt.QBit(), in_reg.dtype), reg=soq)
            else:
                qreg_to_qvar[qreg] = soq
        else:
            qreg_to_qvar[qreg] = soq
    if soqs_to_join:
        # A split is not necessarily matched with a join of the same size so we
        # need to strip the data type of the parent split before assigning the correct bitsize.
        qreg_to_qvar[in_reg] = bb.join(
            np.array([soqs_to_join[q] for q in in_reg.qubits]), dtype=in_reg.dtype
        )


def _gather_input_soqs(
    bb: BloqBuilder, op_quregs, qreg_to_qvar  # type: ignore[type-var]
):  # type: ignore[type-var]
    qvars_in = {}  # type: ignore[type-var]
    for reg_name, quregs in op_quregs.items():
        flat_soqs: List[qt.Soquet] = []
        for qureg in quregs.flatten():
            _ensure_in_reg_exists(bb, qureg, qreg_to_qvar)
            flat_soqs.append(qreg_to_qvar[qureg])
        qvars_in[reg_name] = np.array(flat_soqs).reshape(quregs.shape)
    return qvars_in


@frozen
class ToBloq(qt.Bloq):
    r"""
    Adapter class to convert PennyLane operators into Qualtran Bloqs
    """

    op: Operation

    @cached_property
    def signature(self) -> "qt.Signature":
        num_wires = len(self.op.wires)
        return qt.Signature([qt.Register("qubits", qt.QBit(), shape=num_wires)])

    def decompose_bloq(self, **kwargs):
        try:
            ops = self.op.decomposition()

            signature = self.signature
            all_wires = list(self.op.wires)
            in_quregs = out_quregs = {"qubits": np.array(all_wires).reshape(len(all_wires), 1)}

            in_quregs = {
                k: np.apply_along_axis(qt.cirq_interop._cirq_to_bloq._QReg, -1, *(v, signature.get_left(k).dtype))  # type: ignore
                for k, v in in_quregs.items()
            }

            out_quregs = {
                k: np.apply_along_axis(qt.cirq_interop._cirq_to_bloq._QReg, -1, *(v, signature.get_right(k).dtype))  # type: ignore
                for k, v in out_quregs.items()
            }
            bb, initial_soqs = qt.BloqBuilder.from_signature(signature, add_registers_allowed=False)

            # 1. Compute qreg_to_qvar for input qubits in the LEFT signature.
            qreg_to_qvar = {}
            for reg in signature.lefts():
                if reg.name not in in_quregs:
                    raise ValueError(
                        f"Register {reg.name} from signature must be present in in_quregs."
                    )
                soqs = initial_soqs[reg.name]
                if isinstance(soqs, qt.Soquet):
                    soqs = np.array(soqs)
                if in_quregs[reg.name].shape != soqs.shape:
                    raise ValueError(
                        f"Shape {in_quregs[reg.name].shape} of qubit register "
                        f"{reg.name} should be {soqs.shape}."
                    )
                qreg_to_qvar |= zip(in_quregs[reg.name].flatten(), soqs.flatten())

            # 2. Add each operation to the composite Bloq.
            for op in ops:
                bloq = ToBloq(op)
                if bloq.signature == qt.Signature([]):
                    bb.add(bloq)
                    continue

                reg_dtypes = [r.dtype for r in bloq.signature]
                # 3.1 Find input / output registers.
                all_op_quregs = {
                    k: np.apply_along_axis(qt.cirq_interop._cirq_to_bloq._QReg, -1, *(v, reg_dtypes[i]))  # type: ignore
                    for i, (k, v) in enumerate(split_qubits(bloq.signature, op.wires).items())
                }

                in_op_quregs = {reg.name: all_op_quregs[reg.name] for reg in bloq.signature.lefts()}

                # 3.2 Find input Soquets, by potentially allocating new Bloq registers corresponding to
                # input Cirq `in_quregs` and updating the `qreg_to_qvar` mapping.
                qvars_in = _gather_input_soqs(bb, in_op_quregs, qreg_to_qvar)

                # 3.3 Add Bloq to the `CompositeBloq` compute graph and get corresponding output Soquets.
                qvars_out = bb.add_d(bloq, **qvars_in)

                # 3.4 Update `qreg_to_qvar` mapping using output soquets `qvars_out`.
                for reg in bloq.signature:
                    # all_op_quregs should exist for both LEFT & RIGHT registers.
                    assert reg.name in all_op_quregs
                    quregs = all_op_quregs[reg.name]
                    if reg.side == qt.Side.LEFT:
                        # This register got de-allocated, update the `qreg_to_qvar` mapping.
                        for q in quregs.flatten():
                            _ = qreg_to_qvar.pop(q)
                    else:
                        assert quregs.shape == np.array(qvars_out[reg.name]).shape
                        qreg_to_qvar |= zip(
                            quregs.flatten(), np.array(qvars_out[reg.name]).flatten()
                        )

            # 4. Combine Soquets to match the right signature.
            final_soqs_dict = _gather_input_soqs(
                bb, {reg.name: out_quregs[reg.name] for reg in signature.rights()}, qreg_to_qvar
            )
            final_soqs_set = set(soq for soqs in final_soqs_dict.values() for soq in soqs.flatten())
            # 5. Free all dangling Soquets which are not part of the final soquets set.
            for qvar in qreg_to_qvar.values():
                if qvar not in final_soqs_set:
                    bb.free(qvar)

            cbloq = bb.finalize(**final_soqs_dict)
            return cbloq
        except DecompositionUndefinedError:
            raise qt.DecomposeNotImplementedError

    def __str__(self):
        return "PL" + self.op.name
