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

# TODO: Remove when PL supports pylint==3.3.6 (it is considered a useless-suppression) [sc-91362]
# pylint: disable=unused-argument

from collections import defaultdict
from functools import cached_property, singledispatch
from typing import Dict, List, Tuple

import numpy as np

import pennylane.measurements as qmeas
import pennylane.ops as qops
import pennylane.templates as qtemps
from pennylane.operation import (
    DecompositionUndefinedError,
    MatrixUndefinedError,
    Operation,
    Operator,
)
from pennylane.registers import registers
from pennylane.tape import make_qscript
from pennylane.wires import WiresLike
from pennylane.workflow import construct_tape
from pennylane.workflow.qnode import QNode

try:
    import cirq
    import qualtran as qt
    from qualtran import Bloq
    from qualtran._infra.gate_with_registers import split_qubits
    from qualtran.bloqs import basic_gates as qt_gates

    qualtran = True
except (ModuleNotFoundError, ImportError) as import_error:
    qualtran = False

    Bloq = object


@singledispatch
def _get_op_call_graph(op):
    # TODO: Integrate with resource operators and the new decomposition pipelines
    """Return call graphs for given PennyLane Operator"""
    return None


@_get_op_call_graph.register
def _(op: qtemps.subroutines.qpe.QuantumPhaseEstimation):
    return {
        qt_gates.Hadamard(): len(op.estimation_wires),
        _map_to_bloq(op.hyperparameters["unitary"]).controlled(): (2 ** len(op.estimation_wires))
        - 1,
        _map_to_bloq((qtemps.QFT(wires=op.estimation_wires)), map_ops=False).adjoint(): 1,
    }


@singledispatch
def _map_to_bloq(op, map_ops=True, custom_mapping=None,  **kwargs):
    """Map PennyLane operators to Qualtran Bloqs. Operators with direct equivalents are directly
    mapped to their Qualtran equivalent even if ``map_ops`` is set to ``False``. Other operators are
    given a smart default mapping. When given a ``custom_mapping``, the custom mapping is used."""
    if not isinstance(op, Operator):
        return ToBloq(op, map_ops=map_ops, **kwargs)

    if custom_mapping is not None:
        return custom_mapping[op]

    return ToBloq(op, map_ops=map_ops, **kwargs)


# pylint: disable=import-outside-toplevel
@_map_to_bloq.register
def _(
    op: qtemps.subroutines.qpe.QuantumPhaseEstimation,
    map_ops=True,
    custom_mapping=None,
    **kwargs,
):
    from qualtran.bloqs.phase_estimation import RectangularWindowState
    from qualtran.bloqs.phase_estimation.text_book_qpe import TextbookQPE

    if not map_ops:
        return ToBloq(op, **kwargs)

    if custom_mapping is not None:
        return custom_mapping[op]

    return TextbookQPE(
        unitary=_map_to_bloq(op.hyperparameters["unitary"]),
        ctrl_state_prep=RectangularWindowState(len(op.hyperparameters["estimation_wires"])),
    )


@_map_to_bloq.register
def _(op: qops.GlobalPhase, **kwargs):
    return qt_gates.GlobalPhase(exponent=op.data[0] / np.pi)


@_map_to_bloq.register
def _(op: qops.Hadamard, **kwargs):
    return qt_gates.Hadamard()


@_map_to_bloq.register
def _(op: qops.Identity, **kwargs):
    return qt_gates.Identity()


@_map_to_bloq.register
def _(op: qops.RX, **kwargs):
    return qt_gates.Rx(angle=float(op.data[0]))


@_map_to_bloq.register
def _(op: qops.RY, **kwargs):
    return qt_gates.Ry(angle=float(op.data[0]))


@_map_to_bloq.register
def _(op: qops.RZ, **kwargs):
    return qt_gates.Rz(angle=float(op.data[0]))


@_map_to_bloq.register
def _(op: qops.S, **kwargs):
    return qt_gates.SGate()


@_map_to_bloq.register
def _(op: qops.SWAP, **kwargs):
    return qt_gates.TwoBitSwap()


@_map_to_bloq.register
def _(op: qops.CSWAP, **kwargs):
    return qt_gates.TwoBitCSwap()


@_map_to_bloq.register
def _(op: qops.T, **kwargs):
    return qt_gates.TGate()


@_map_to_bloq.register
def _(op: qops.X, **kwargs):
    return qt_gates.XGate()


@_map_to_bloq.register
def _(op: qops.Y, **kwargs):
    return qt_gates.YGate()


@_map_to_bloq.register
def _(op: qops.CY, **kwargs):
    return qt_gates.CYGate()


@_map_to_bloq.register
def _(op: qops.Z, **kwargs):
    return qt_gates.ZGate()


@_map_to_bloq.register
def _(op: qops.CZ, **kwargs):
    return qt_gates.CZ()


@_map_to_bloq.register
def _(op: qops.Adjoint, custom_mapping=None, map_ops=True, **kwargs):
    return _map_to_bloq(op.base, custom_mapping=custom_mapping, map_ops=map_ops, **kwargs).adjoint()


@_map_to_bloq.register
def _(op: qops.Controlled, custom_mapping=None, map_ops=True, **kwargs):
    if isinstance(op, qops.Toffoli):
        return qt_gates.Toffoli()

    return _map_to_bloq(
        op.base, custom_mapping=custom_mapping, map_ops=map_ops, **kwargs
    ).controlled()


@_map_to_bloq.register
def _(op: qmeas.MeasurementProcess, **kwargs):
    return None


def _get_to_pl_op():
    @singledispatch
    def _to_pl_op(bloq, wires):
        return FromBloq(bloq=bloq, wires=wires)

    @_to_pl_op.register
    def _(bloq: qt.bloqs.basic_gates.CNOT, wires):
        return qops.CNOT(wires=wires)

    @_to_pl_op.register
    def _(bloq: qt.bloqs.basic_gates.GlobalPhase, wires):
        return qops.GlobalPhase(bloq.exponent * np.pi, wires)

    @_to_pl_op.register
    def _(bloq: qt.bloqs.basic_gates.Hadamard, wires):
        return qops.Hadamard(wires)

    @_to_pl_op.register
    def _(bloq: qt.bloqs.basic_gates.Identity, wires):
        return qops.Identity(wires)

    @_to_pl_op.register
    def _(bloq: qt.bloqs.basic_gates.Rx, wires):
        return qops.RX(bloq.angle, wires)

    @_to_pl_op.register
    def _(bloq: qt.bloqs.basic_gates.Ry, wires):
        return qops.RY(bloq.angle, wires)

    @_to_pl_op.register
    def _(bloq: qt.bloqs.basic_gates.Rz, wires):
        return qops.RZ(bloq.angle, wires)

    @_to_pl_op.register
    def _(bloq: qt.bloqs.basic_gates.SGate, wires):
        return qops.adjoint(qops.S(wires)) if bloq.is_adjoint else qops.S(wires)

    @_to_pl_op.register
    def _(bloq: qt.bloqs.basic_gates.TwoBitSwap, wires):
        return qops.SWAP(wires)

    @_to_pl_op.register
    def _(bloq: qt.bloqs.basic_gates.TwoBitCSwap, wires):
        return qops.CSWAP(wires)

    @_to_pl_op.register
    def _(bloq: qt.bloqs.basic_gates.TGate, wires):
        return qops.adjoint(qops.T(wires)) if bloq.is_adjoint else qops.T(wires)

    @_to_pl_op.register
    def _(bloq: qt.bloqs.basic_gates.Toffoli, wires):
        return qops.Toffoli(wires)

    @_to_pl_op.register
    def _(bloq: qt.bloqs.basic_gates.XGate, wires):
        return qops.X(wires)

    @_to_pl_op.register
    def _(bloq: qt.bloqs.basic_gates.YGate, wires):
        return qops.Y(wires)

    @_to_pl_op.register
    def _(bloq: qt.bloqs.basic_gates.CYGate, wires):
        return qops.CY(wires)

    @_to_pl_op.register
    def _(bloq: qt.bloqs.basic_gates.ZGate, wires):
        return qops.Z(wires)

    @_to_pl_op.register
    def _(bloq: qt.bloqs.basic_gates.CZ, wires):
        return qops.CZ(wires)

    @_to_pl_op.register(qt.bloqs.bookkeeping.Allocate)
    @_to_pl_op.register(qt.bloqs.bookkeeping.Cast)
    @_to_pl_op.register(qt.bloqs.bookkeeping.Free)
    @_to_pl_op.register(qt.bloqs.bookkeeping.Join)
    @_to_pl_op.register(qt.bloqs.bookkeeping.Partition)
    @_to_pl_op.register(qt.bloqs.bookkeeping.Split)
    def _(bloq, wires):
        return None

    return _to_pl_op


def bloq_registers(bloq: "qt.Bloq"):
    """Reads a `Qualtran Bloq <https://qualtran.readthedocs.io/en/latest/bloqs/index.html#bloqs-library>`_
    signature and returns a dictionary mapping the Bloq's register names to :class:`~.Wires`.

    .. note::
        This function requires the latest version of Qualtran. We recommend installing the latest
        release via ``pip``:

        .. code-block:: console

            pip install qualtran

    The keys of the returned dictionary are the register names in the Qualtran Bloq. The
    values are :class:`~.Wires` objects with a length equal to the bitsize of its respective
    register. The wires are indexed in ascending order, starting from 0.

    This function makes it easy to access the wires that a Bloq acts on and use them to precisely
    control how gates connect.

    Args:
        bloq (Bloq): an initialized Qualtran ``Bloq`` to be wrapped as a PennyLane operator

    Returns:
        dict: A dictionary mapping the names of the Bloq's registers to :class:`~.Wires`
            objects with the same lengths as the bitsizes of their respective registers.

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

    return registers(wire_register_dict)


def _get_named_registers(regs):
    """Returns a ``qml.registers`` object associated with the named registers in the bloq"""

    temp_register_dict = {reg.name: reg.total_bits() for reg in regs}

    return registers(temp_register_dict)


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
    r"""An adapter for using a `Qualtran Bloq <https://qualtran.readthedocs.io/en/latest/bloqs/index.html#bloqs-library>`__
    as a PennyLane :class:`~.Operation`.

    .. note::
        This class requires the latest version of Qualtran. We recommend installing the latest
        release via ``pip``:

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

    # TODO: Remove when PL supports pylint==3.3.6 (it is considered a useless-suppression) [sc-91362]
    # pylint: disable=no-method-argument
    def compute_matrix(*params, **hyperparams):  # pylint: disable=no-self-argument
        bloq = hyperparams["bloq"]
        matrix = bloq.tensor_contract()

        if matrix.shape != (2 ** len(params[0].wires), 2 ** len(params[0].wires)):
            raise MatrixUndefinedError

        return matrix


class _QReg:
    """Used as a container for qubits that form a `Register` of a given bitsize. This is a modified
    version of `_QReg <https://github.com/quantumlib/Qualtran/blob/main/qualtran/cirq_interop/_cirq_to_bloq.py>`_
    found in Qualtran as well.

    Each instance of `_QReg` would correspond to a `Soquet` in Bloqs and represents an opaque collection
    of qubits that together form a quantum register.
    """

    def __init__(self, qubits: Tuple["cirq.Qid", ...], dtype: "qt.QDType"):
        if isinstance(qubits, cirq.Qid):
            self.qubits = (qubits,)
        else:
            self.qubits = tuple(qubits)

        self.dtype = dtype
        self._initialized = True

    def __setattr__(self, name, value):
        """Makes the instance immutable after initialization."""
        if getattr(self, "_initialized", False):
            raise AttributeError(
                f"Cannot set attribute '{name}'. Instances of _QReg are immutable."
            )
        super().__setattr__(name, value)

    def __repr__(self) -> str:
        """Provides a developer-friendly string representation."""
        return f"_QReg(qubits={self.qubits!r}, dtype={self.dtype!r})"

    # Override the __eq__ and __hash__ functions to handle single qubit registers
    # that are functionally the same but have different dtypes.
    def __eq__(self, other) -> bool:
        if not isinstance(other, _QReg):
            return False
        return self.qubits == other.qubits

    def __hash__(self):
        return hash(self.qubits)


def _ensure_in_reg_exists(
    bb: "qt.BloqBuilder",
    in_reg: "_QReg",
    qreg_to_qvar: Dict["_QReg", "qt.Soquet"],
) -> None:
    """Modified function from the Qualtran-Cirq interop module to ensure `qreg_to_qvar[in_reg]`
    exists. If `in_reg` is not found in `qreg_to_qvar`, that means that the input qubit register
    is a multi-qubit register. All in_regs should be single qubit registers, so this would be a
    bug, and an AssertionError is raised. To capture control flow, multi-qubit registers will be
    allowed, and we will remove the AssertionError and use Split and Join operations as needed.

    Args:
        bb (qt.BloqBuilder): an instance of a Qualtran BloqBuilder
        in_reg (_QReg): a container for qubits that form a Register of a given bitsize
        qreg_to_qvar (Dict[_QReg, qt.Soquet]): a dictionary of _QRegs that corresponds to Soquets

    Raises:
        AssertionError: `in_reg` was not found in `qreg_to_qvar`, meaning there exists multi-qubit
            registers that we do not support at the moment
    """

    all_mapped_qubits = {q for qreg in qreg_to_qvar for q in qreg.qubits}
    qubits_to_allocate = [q for q in in_reg.qubits if q not in all_mapped_qubits]
    if qubits_to_allocate:
        n_alloc = len(qubits_to_allocate)
        qreg_to_qvar[
            _QReg(qubits_to_allocate, dtype=qt.QBit() if n_alloc == 1 else qt.QAny(n_alloc))
        ] = bb.allocate(n_alloc)

    # if in_reg not in qreg_to_qvar: splits & joins needed, which shouldn't be the case
    assert in_reg in qreg_to_qvar, f"Input register {in_reg} not found, suggesting a bug"


def _gather_input_soqs(bb: "qt.BloqBuilder", op_quregs, qreg_to_qvar):
    """Modified function from Qualtran-Cirq interop module that collects input Soquets.

    Args:
        bb (qt.BloqBuilder): an instance of a Qualtran BloqBuilder
        op_quregs (Dict[str, _QRegs]): a dict of register names that corresponds to _QRegs
        qreg_to_qvar (Dict[str, qt.Soquet]): a dict of register names that corresponds to input Soquets

    Returns:
        dict: in_reg was not found in qreg_to_qvar
    """
    qvars_in = {}
    for reg_name, quregs in op_quregs.items():
        flat_soqs: List[qt.Soquet] = []
        for qureg in quregs.flatten():
            _ensure_in_reg_exists(bb, qureg, qreg_to_qvar)
            flat_soqs.append(qreg_to_qvar[qureg])
        qvars_in[reg_name] = np.array(flat_soqs).reshape(quregs.shape)
    return qvars_in


class ToBloq(Bloq):  # pylint:disable=useless-object-inheritance (Inherit qt.Bloq optionally)
    r"""
    An adapter to convert a PennyLane :class:`~.QNode`, ``Qfunc, or :class:`~.Operation` to a
    `Qualtran Bloq <https://qualtran.readthedocs.io/en/latest/bloqs/index.html#bloqs-library>`__.

    .. note::
        This class requires the latest version of Qualtran. We recommend installing the latest
        release via ``pip``:

        .. code-block:: console

            pip install qualtran

    Args:
        op (QNode |Qfunc | Operation): a PennyLane ``QNode``, ``Qfunc``, or operator to be wrapped
            as a Qualtran Bloq.
        map_ops (bool): Whether to map operations to a Qualtran Bloq. Operations are wrapped
            as a ``ToBloq`` when False. Default is True.
        custom_mapping (dict): Dictionary to specify a mapping between a PennyLane operator and a
            Qualtran Bloq. A default mapping is used if not defined.

    Raises:
        TypeError: operator must be an instance of :class:`~.Operation`.

    .. seealso:: :func:`~.to_bloq`

    **Example**

    This example shows how to use ``qml.ToBloq``:

    >>> from qualtran.resource_counting.generalizers import generalize_rotation_angle
    >>> op = qml.QuantumPhaseEstimation(
    ...     qml.RX(0.2, wires=[0]), estimation_wires=[1, 2]
    ... )
    >>> op_as_bloq = qml.ToBloq(op)
    >>> graph, sigma = op_as_bloq.call_graph(generalize_rotation_angle)
    >>> sigma
    {Hadamard(): 4,
    Controlled(subbloq=Rx(angle=0.2, eps=1e-11), ctrl_spec=CtrlSpec(qdtypes=(QBit(),), cvs=(array(1),))): 3,
    TwoBitSwap(): 1,
    CNOT(): 2,
    ZPowGate(exponent=\phi, eps=5e-12): 2,
    ZPowGate(exponent=\phi, eps=1e-11): 1}
    """

    def __init__(self, op, map_ops=False, **kwargs):
        if not qualtran:
            raise ImportError(
                "Optional dependency 'qualtran' is required "
                "for ToBloq functionality but is not installed. Try `pip install qualtran`."
            )

        if not isinstance(op, Operator) and not isinstance(op, QNode) and not callable(op):
            raise TypeError(
                f"Input must be either an instance of {Operator}, {QNode} or a quantum function."
            )

        self.op = op
        self.map_ops = map_ops
        self._kwargs = kwargs
        super().__init__()

    @cached_property
    def signature(self) -> "qt.Signature":
        """Compute and return Qualtran signature for given op or QNode."""
        if isinstance(self.op, QNode):
            self.op.name = "QNode"
            num_wires = len(construct_tape(self.op)(**self._kwargs).wires)
        elif isinstance(self.op, Operator):
            num_wires = len(self.op.wires)
        else:
            num_wires = len(make_qscript(self.op)(**self._kwargs).wires)
        return qt.Signature([qt.Register("qubits", qt.QBit(), shape=num_wires)])

    def decompose_bloq(self):  # pylint:disable=too-many-branches
        """Decompose the bloq using the op's decomposition or the tape of the QNode"""
        try:
            if isinstance(self.op, QNode):
                tape = construct_tape(self.op)(**self._kwargs)
                ops = tape.circuit
                all_wires = list(tape.wires)
            elif isinstance(self.op, Operator):
                ops = self.op.decomposition()
                all_wires = list(self.op.wires)
            else:
                tape = make_qscript(self.op)(**self._kwargs)
                ops = tape.operations
                all_wires = list(tape.wires)

            signature = self.signature
            in_quregs = out_quregs = {"qubits": np.array(all_wires).reshape(len(all_wires), 1)}

            in_key = list(in_quregs.keys())[0]
            out_key = list(out_quregs.keys())[0]

            in_quregs = {
                in_key: np.apply_along_axis(
                    _QReg, -1, in_quregs[in_key], signature.get_left(in_key).dtype
                )
            }
            out_quregs = {
                out_key: np.apply_along_axis(
                    _QReg, -1, out_quregs[out_key], signature.get_right(out_key).dtype
                )
            }

            bb, initial_soqs = qt.BloqBuilder.from_signature(signature, add_registers_allowed=False)

            # `signature.lefts()` can be thought of as input qubits. For our purposes LEFT and
            # RIGHT signatures will in most cases match since there are no allocated & freed
            # qubits. Here, qreg_to_qvar is a map between a register and a Soquet. This serves
            # as the foundation to wire up the rest of the bloqs.
            qreg_to_qvar = {}
            for reg in signature.lefts():
                assert reg.name in in_quregs
                soqs = initial_soqs[reg.name]
                assert in_quregs[reg.name].shape == soqs.shape
                qreg_to_qvar |= zip(in_quregs[reg.name].flatten(), soqs.flatten())

            # Add each operation to the composite Bloq.
            for op in ops:
                bloq = _map_to_bloq(op, map_ops=self.map_ops)
                if bloq is None:
                    continue

                if bloq.signature == qt.Signature([]):
                    bb.add(bloq)
                    continue

                reg_dtypes = [r.dtype for r in bloq.signature]
                # Find input / output registers.
                all_op_quregs = {
                    k: np.apply_along_axis(_QReg, -1, *(v, reg_dtypes[i]))  # type: ignore
                    for i, (k, v) in enumerate(split_qubits(bloq.signature, op.wires).items())
                }

                in_op_quregs = {reg.name: all_op_quregs[reg.name] for reg in bloq.signature.lefts()}
                # Find input Soquets, by potentially allocating new Bloq registers corresponding to
                # input `in_quregs` and updating the `qreg_to_qvar` mapping.
                qvars_in = _gather_input_soqs(bb, in_op_quregs, qreg_to_qvar)

                # Add Bloq to the `CompositeBloq` compute graph and get corresponding output Soquets.
                qvars_out = bb.add_d(bloq, **qvars_in)

                # Update `qreg_to_qvar` mapping using output soquets `qvars_out`.
                for reg in bloq.signature:
                    # all_op_quregs should exist for both LEFT & RIGHT registers.
                    assert reg.name in all_op_quregs
                    quregs = all_op_quregs[reg.name]
                    if reg.side != qt.Side.LEFT:
                        assert quregs.shape == np.array(qvars_out[reg.name]).shape
                        qreg_to_qvar |= zip(
                            quregs.flatten(), np.array(qvars_out[reg.name]).flatten()
                        )

            # Combine Soquets to match the right signature.
            final_soqs_dict = _gather_input_soqs(
                bb,
                {reg.name: out_quregs[reg.name] for reg in signature.rights()},
                qreg_to_qvar,
            )
            final_soqs_set = set(soq for soqs in final_soqs_dict.values() for soq in soqs.flatten())
            # Free all dangling Soquets which are not part of the final soquets set.
            for qvar in qreg_to_qvar.values():
                if qvar not in final_soqs_set:
                    bb.free(qvar)

            cbloq = bb.finalize(**final_soqs_dict)
            return cbloq
        except DecompositionUndefinedError as undefined_decomposition:
            raise qt.DecomposeNotImplementedError from undefined_decomposition

    def build_call_graph(self, ssa):
        """Build Qualtran call graph with defined call graph if available, otherwise build
        said call graph with the decomposition"""
        call_graph = _get_op_call_graph(self.op)
        if call_graph:
            return call_graph

        return self.decompose_bloq().build_call_graph(ssa)

    def __repr__(self):
        if isinstance(self.op, QNode):
            return "ToBloq(QNode)"
        if isinstance(self.op, Operation):
            return f"ToBloq({self.op.name})"
        return "ToBloq(Qfunc)"

    def __eq__(self, other):
        if type(other) is type(self):
            return self.op == other.op
        return False

    def __hash__(self):
        return hash(self.op)

    def __str__(self):
        if hasattr(self.op, "name"):
            return f"PL{self.op.name}"
        return "PLQfunc"


def to_bloq(circuit, map_ops: bool = True, custom_mapping: dict = None, **kwargs):
    """
    Converts a PennyLane circuit or :class:`~.Operation` to the corresponding `Qualtran Bloq <https://qualtran.readthedocs.io/en/latest/bloqs/index.html#bloqs-library>`__.

    .. note::
        This class requires the latest version of Qualtran. We recommend installing the latest
        release via ``pip``:

        .. code-block:: console

            pip install qualtran

    Args:
        circuit (QNode |Qfunc | Operation): a PennyLane ``QNode``, ``Qfunc``, or operator to be wrapped
            as a Qualtran Bloq.
        map_ops (bool): Whether to map operations to a Qualtran Bloq. Operations are wrapped
            as a ``ToBloq`` when False. Default is True.
        custom_mapping (dict): Dictionary to specify a mapping between a PennyLane operator and a
            Qualtran Bloq. A default mapping is used if not defined.

    Returns:
        Bloq: The Qualtran Bloq that corresponds to the given circuit or :class:`~.Operation` and
            options.

    .. seealso:: :class:`~.ToBloq`

    **Example**

    This example shows how to use ``qml.to_bloq``:

    >>> from qualtran.resource_counting.generalizers import generalize_rotation_angle
    >>> op = qml.QuantumPhaseEstimation(
    ...     qml.RX(0.2, wires=[0]), estimation_wires=[1, 2]
    ... )
    >>> op_as_bloq = qml.to_bloq(op)
    >>> graph, sigma = op_as_bloq.call_graph(generalize_rotation_angle)
    >>> sigma
    {Allocate(dtype=QFxp(bitsize=2, num_frac=2, signed=False), dirty=False): 1,
    Hadamard(): 4,
    Controlled(subbloq=Rx(angle=0.2, eps=1e-11), ctrl_spec=CtrlSpec(qdtypes=(QBit(),), cvs=(array(1),))): 3,
    And(cv1=1, cv2=1, uncompute=True): 1,
    And(cv1=1, cv2=1, uncompute=False): 1,
    ZPowGate(exponent=\\phi, eps=1e-10): 1,
    TwoBitSwap(): 1}

    .. details::
        :title: Usage Details

        Some PennyLane operators don't have a direct equivalent in Qualtran. For example, in Qualtran, there
        are many varieties of Quantum Phase Estimation. When ``qml.to_bloq`` is called on
        :class:`~pennylane.QuantumPhaseEstimation`, a smart default is chosen.

        >>> qml.to_bloq(qml.QuantumPhaseEstimation(
        ...     unitary=qml.RX(0.1, wires=0), estimation_wires=range(1, 5)
        ... ))
        TextbookQPE(unitary=Rx(angle=0.1, eps=1e-11), ctrl_state_prep=RectangularWindowState(bitsize=4), qft_inv=Adjoint(subbloq=QFTTextBook(bitsize=4, with_reverse=True)))

        Note that the chosen Qualtran Bloq may not be an exact equivalent. If an exact
        equivalent is needed, we recommend setting ``map_ops`` to False.
        This will wrap the input PennyLane operator as a Qualtran Bloq, enabling Qualtran functions
        such as ``decompose_bloq`` or ``call_graph``, but maintaining the PennyLane decomposition definition of the operator.

        >>> qml.to_bloq(qml.QuantumPhaseEstimation(
        ...     unitary=qml.RX(0.1, wires=0), estimation_wires=range(1, 5)
        ... ), map_ops=False)
        ToBloq(QuantumPhaseEstimation)


        Alternatively, users can provide a custom mapping that maps a PennyLane operator to a
        specific Qualtran Bloq.

        >>> from qualtran.bloqs.phase_estimation import TextbookQPE
        >>> from qualtran.bloqs.phase_estimation.lp_resource_state import LPResourceState
        >>> op = qml.QuantumPhaseEstimation(
        ...         unitary=qml.RX(0.1, wires=0), estimation_wires=range(1, 5)
        ...     )
        >>> custom_mapping = {
        ...     op : TextbookQPE(
        ...         unitary=qml.to_bloq(qml.RX(0.1, wires=0)),
        ...         ctrl_state_prep=LPResourceState(4),
        ...     )
        ... }
        >>> qml.to_bloq(op, custom_mapping=custom_mapping)
        TextbookQPE(unitary=Rx(angle=0.1, eps=1e-11), ctrl_state_prep=LPResourceState(bitsize=4), qft_inv=Adjoint(subbloq=QFTTextBook(bitsize=4, with_reverse=True)))

    """

    if not qualtran:
        raise ImportError(
            "The `to_bloq` function requires Qualtran to be installed. You can install"
            "qualtran via: pip install qualtran."
        )

    if map_ops and custom_mapping:
        return _map_to_bloq(circuit, map_ops=True, custom_mapping=custom_mapping, **kwargs)

    return _map_to_bloq(circuit, map_ops=map_ops, **kwargs)
