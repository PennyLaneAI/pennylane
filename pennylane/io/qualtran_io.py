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
from functools import cached_property, lru_cache, singledispatch
from typing import TYPE_CHECKING, Dict, List

import numpy as np

import pennylane.ops as qops
import pennylane.templates as qtemps
from pennylane.operation import DecompositionUndefinedError, MatrixUndefinedError, Operation
from pennylane.registers import registers
from pennylane.wires import WiresLike

try:
    import qualtran as qt

    qualtran = True
except (ModuleNotFoundError, ImportError) as import_error:
    qualtran = False

if TYPE_CHECKING:
    from qualtran.cirq_interop._bloq_to_cirq import _QReg


# pylint: disable=import-outside-toplevel, unused-argument
@lru_cache
def _get_op_call_graph():
    @singledispatch
    def _op_call_graph(op):
        return None

    @_op_call_graph.register
    def _(op: qtemps.subroutines.qpe.QuantumPhaseEstimation):
        from qualtran.bloqs.basic_gates import Hadamard

        return {
            Hadamard(): len(op.estimation_wires),
            _map_to_bloq()(op.hyperparameters["unitary"]).controlled(): (
                2 ** len(op.estimation_wires)
            )
            - 1,
            _map_to_bloq()((qtemps.QFT(wires=op.estimation_wires))).adjoint(): 1,
        }

    @_op_call_graph.register
    def _(op: qtemps.subroutines.TrotterizedQfunc):
        import pennylane as qml

        n = op.hyperparameters["n"]
        order = op.hyperparameters["order"]
        k = order // 2
        qfunc = op.hyperparameters["qfunc"]
        qfunc_args = op.parameters[1:]

        with qml.QueuingManager.stop_recording():
            with qml.queuing.AnnotatedQueue() as q:
                base_hyper_params = ("n", "order", "qfunc", "reverse")

                qfunc_args = op.parameters
                qfunc_kwargs = {
                    k: v for k, v in op.hyperparameters.items() if not k in base_hyper_params
                }

                qfunc = op.hyperparameters["qfunc"]
                qfunc(*qfunc_args, wires=op.wires, **qfunc_kwargs)

        call_graph = defaultdict(int, {})
        num_gates = 2 * n * (5 ** (k - 1))
        for op in q.queue:
            call_graph[_map_to_bloq()(op)] += num_gates

        return call_graph

    @_op_call_graph.register
    def _(op: qtemps.state_preparations.Superposition):
        from qualtran.bloqs.basic_gates import CNOT

        import pennylane as qml
        from pennylane.templates.state_preparations.superposition import _assign_states

        gate_types = {}
        wires = op.wires
        coeffs = op.coeffs
        bases = op.hyperparameters["bases"]
        num_basis_states = len(bases)
        size_basis_state = len(bases[0])  # assuming they are all the same size

        dic_state = dict(zip(bases, coeffs))
        perms = _assign_states(bases)
        new_dic_state = {perms[key]: dic_state[key] for key in dic_state if key in perms}

        sorted_coefficients = [
            value
            for key, value in sorted(
                new_dic_state.items(), key=lambda item: int("".join(map(str, item[0])), 2)
            )
        ]
        msp = qml.StatePrep(
            qml.math.stack(sorted_coefficients),
            wires=wires[-int(qml.math.ceil(qml.math.log2(len(coeffs)))) :],
            pad_with=0,
        )
        gate_types[_map_to_bloq()(msp)] = 1

        cnot = CNOT()
        num_zero_ctrls = size_basis_state // 2
        control_values = [1] * num_zero_ctrls + [0] * (size_basis_state - num_zero_ctrls)

        multi_x = _map_to_bloq()(
            qml.MultiControlledX(wires=range(size_basis_state + 1), control_values=control_values)
        )

        basis_size = 2**size_basis_state
        prob_matching_basis_states = num_basis_states / basis_size
        num_permutes = round(num_basis_states * (1 - prob_matching_basis_states))
        if num_permutes:
            gate_types[cnot] = num_permutes * (
                size_basis_state // 2
            )  # average number of bits to flip
            gate_types[multi_x] = 2 * num_permutes  # for compute and uncompute

        return gate_types

    @_op_call_graph.register
    def _(op: qtemps.state_preparations.QROMStatePreparation):
        import pennylane as qml

        gate_types = defaultdict(int, {})
        state_vector = op.state_vector
        positive_and_real = True

        for c in state_vector:
            if c.imag != 0 or c.real < 0:
                positive_and_real = False
                break

        num_state_qubits = int(qml.math.log2(len(op.state_vector)))
        precision_wires = op.hyperparameters["precision_wires"]
        input_wires = op.hyperparameters["input_wires"]
        work_wires = op.hyperparameters["work_wires"]
        num_precision_wires = len(precision_wires)

        for i in range(num_state_qubits):
            num_bit_flips = 2 ** (i - 1)
            if i == 0:
                num_bit_flips = 0

            zero_string = "0" * num_precision_wires
            one_string = "0" * (num_precision_wires - 1) + "1" if num_precision_wires > 0 else ""
            bitstrings = [zero_string for _ in range(num_bit_flips)] + [
                one_string for _ in range(num_bit_flips)
            ]

            if len(bitstrings) == 0:
                bitstrings = ["0" * (num_precision_wires - 1) + "1"]

            qrom_op = qml.QROM(
                bitstrings=bitstrings,
                target_wires=precision_wires,
                control_wires=input_wires[:i],
                work_wires=work_wires,
                clean=False,
            )
            gate_types[_map_to_bloq()(qrom_op)] += 1
            gate_types[_map_to_bloq()(qml.adjoint(qrom_op))] += 1

        gate_types[_map_to_bloq()(qml.CRY(0, wires=[0, 1]))] = (
            num_precision_wires * num_state_qubits
        )

        if not positive_and_real:
            num_bit_flips = 2 ** (num_state_qubits - 1)
            if i == 0:
                num_bit_flips = 0

            zero_string = "0" * num_precision_wires
            one_string = "0" * (num_precision_wires - 1) + "1" if num_precision_wires > 0 else ""
            bitstrings = [zero_string for _ in range(num_bit_flips)] + [
                one_string for _ in range(num_bit_flips)
            ]

            if len(bitstrings) == 0:
                bitstrings = ["0" * (num_precision_wires - 1) + "1"]

            qrom_op = qml.QROM(
                bitstrings=bitstrings,
                target_wires=precision_wires,
                control_wires=input_wires,
                work_wires=work_wires,
                clean=False,
            )

            gate_types[_map_to_bloq()(qrom_op)] += 1
            gate_types[_map_to_bloq()(qml.adjoint(qrom_op))] += 1
            gate_types[
                _map_to_bloq()(
                    qml.ctrl(
                        qml.GlobalPhase((2 * np.pi), wires=input_wires[0]),
                        control=0,
                    )
                )
            ] = num_precision_wires

        return gate_types

    @_op_call_graph.register
    def _(op: qops.BasisState):
        from qualtran.bloqs.basic_gates import XGate

        gate_types = {}
        gate_types[XGate()] = sum(op.parameters[0])

        return gate_types

    @_op_call_graph.register
    def _(op: qtemps.subroutines.QROM):
        from qualtran.bloqs.basic_gates import CNOT, Hadamard, TwoBitCSwap, XGate

        import pennylane as qml

        gate_types = defaultdict(int)
        bitstrings = op.hyperparameters["bitstrings"]
        num_bitstrings = len(bitstrings)

        num_bit_flips = 0
        for bit_string in bitstrings:
            num_bit_flips += bit_string.count("1")

        num_work_wires = len(op.hyperparameters["work_wires"])
        size_bitstring = len(op.hyperparameters["target_wires"])
        num_control_wires = len(op.hyperparameters["control_wires"])
        clean = op.hyperparameters["clean"]

        if num_control_wires == 0:
            gate_types[XGate()] = num_bit_flips
            return gate_types

        cnot = CNOT()
        hadamard = Hadamard()
        num_parallel_computations = (num_work_wires + size_bitstring) // size_bitstring
        # num_parallel_computations = min(num_parallel_computations, num_bitstrings)

        square_fact = qml.math.floor(
            qml.math.sqrt(num_bitstrings)
        )  # use a square scheme for rows and cloumns
        num_parallel_computations = min(num_parallel_computations, square_fact)

        num_swap_wires = qml.math.floor(qml.math.log2(num_parallel_computations))
        num_select_wires = qml.math.ceil(
            qml.math.log2(qml.math.ceil(num_bitstrings / (2**num_swap_wires)))
        )
        assert num_swap_wires + num_select_wires <= num_control_wires

        swap_work_wires = (int(2**num_swap_wires) - 1) * size_bitstring
        free_work_wires = num_work_wires - swap_work_wires

        swap_clean_prefactor = 1
        select_clean_prefactor = 1

        if clean:
            gate_types[hadamard] = 2 * size_bitstring
            swap_clean_prefactor = 4
            select_clean_prefactor = 2

        # SELECT cost:
        gate_types[cnot] = num_bit_flips  # each unitary in the select is just a CNOT

        num_select_wires = int(num_select_wires)
        multi_x = _map_to_bloq()(
            qml.MultiControlledX(
                wires=range(num_select_wires + 1),
                control_values=[True] * num_select_wires,
                work_wires=range(num_select_wires + 1, num_select_wires + 1 + free_work_wires),
            )
        )

        num_total_ctrl_possibilities = 2**num_select_wires
        gate_types[multi_x] = select_clean_prefactor * (
            2 * num_total_ctrl_possibilities  # two applications targetting the aux qubit
        )
        num_zero_controls = (2 * num_total_ctrl_possibilities * num_select_wires) // 2
        gate_types[XGate()] = select_clean_prefactor * (
            num_zero_controls * 2  # conjugate 0 controls on the multi-qubit x gates from above
        )
        # SWAP cost:
        ctrl_swap = TwoBitCSwap()
        gate_types[ctrl_swap] = swap_clean_prefactor * ((2**num_swap_wires) - 1) * size_bitstring

        return gate_types

    @_op_call_graph.register
    def _(op: qtemps.subroutines.QFT):
        from qualtran.bloqs.basic_gates import Hadamard, TwoBitSwap

        import pennylane as qml

        gate_types = {}
        num_wires = len(op.wires)
        gate_types[Hadamard()] = num_wires
        gate_types[_map_to_bloq()(qml.ControlledPhaseShift(1, [0, 1]))] = (
            num_wires * (num_wires - 1) // 2
        )
        gate_types[TwoBitSwap()] = num_wires // 2
        return gate_types

    @_op_call_graph.register
    def _(op: qtemps.subroutines.QSVT):
        gate_types = defaultdict(int, {})
        UA = op.hyperparameters["UA"]
        projectors = op.hyperparameters["projectors"]
        num_projectors = len(projectors)

        for _, op in enumerate(projectors[:-1]):
            gate_types[_map_to_bloq()(op)] += 1

        gate_types[_map_to_bloq()(UA)] += num_projectors // 2
        gate_types[_map_to_bloq()(UA).adjoint()] += (num_projectors - 1) // 2
        gate_types[_map_to_bloq()(projectors[-1])] += 1

        return gate_types

    @_op_call_graph.register
    def _(op: qtemps.subroutines.ModExp):
        from qualtran.bloqs.basic_gates import CNOT

        import pennylane as qml

        mod = op.hyperparameters["mod"]
        num_work_wires = len(op.hyperparameters["work_wires"])
        num_x_wires = len(op.hyperparameters["x_wires"])

        mult_resources = {}
        if mod == 2**num_x_wires:
            num_aux_wires = num_x_wires
            num_aux_swap = num_x_wires
        else:
            num_aux_wires = num_work_wires - 1
            num_aux_swap = num_aux_wires - 1

        qft = _map_to_bloq()(qml.QFT(wires=range(num_aux_wires)))
        qft_dag = qft.adjoint()

        sequence = _map_to_bloq()(
            qml.ControlledSequence(
                qml.PhaseAdder(k=3, x_wires=range(1, num_x_wires + 1)), control=[0]
            )
        )

        sequence_dag = sequence.adjoint()

        cnot = CNOT()

        mult_resources = {}
        mult_resources[qft] = 2
        mult_resources[qft_dag] = 2
        mult_resources[sequence] = 1
        mult_resources[sequence_dag] = 1
        mult_resources[cnot] = min(num_x_wires, num_aux_swap)

        gate_types = {}
        for comp_rep, _ in mult_resources.items():
            new_rep = comp_rep.controlled()
            # cancel out QFTs from consecutive Multipliers
            if hasattr(comp_rep, "op"):
                if comp_rep.op.name in ("QFT"):
                    gate_types[new_rep] = 1
            elif hasattr(comp_rep, "subbloq"):
                if comp_rep.subbloq.op.name in ("QFT"):
                    gate_types[new_rep] = 1
            else:
                gate_types[new_rep] = mult_resources[comp_rep] * ((2**num_x_wires) - 1)

        return gate_types

    return _op_call_graph


# pylint: disable=import-outside-toplevel, unused-argument
@lru_cache
def _map_to_bloq():
    @singledispatch
    def _to_qt_bloq(op, **kwargs):
        return ToBloq(op, **kwargs)

    @_to_qt_bloq.register
    def _(op: qops.Adjoint):
        return _map_to_bloq()(op.base).adjoint()

    @_to_qt_bloq.register
    def _(op: qops.Controlled):
        if isinstance(op, qops.MultiControlledX):
            return ToBloq(op)
        return _map_to_bloq()(op.base).controlled()

    @_to_qt_bloq.register
    def _(op: qtemps.subroutines.qpe.QuantumPhaseEstimation, **kwargs):
        from qualtran.bloqs.phase_estimation import RectangularWindowState
        from qualtran.bloqs.phase_estimation.text_book_qpe import TextbookQPE

        if "custom_mapping" in kwargs:
            return kwargs["custom_mapping"][op]

        return TextbookQPE(
            unitary=_map_to_bloq()(op.hyperparameters["unitary"]),
            ctrl_state_prep=RectangularWindowState(len(op.hyperparameters["estimation_wires"])),
        )

    @_to_qt_bloq.register
    def _(op: qops.GlobalPhase):
        from qualtran.bloqs.basic_gates import GlobalPhase

        return GlobalPhase(exponent=op.data[0] / np.pi)

    @_to_qt_bloq.register
    def _(op: qops.CNOT):
        from qualtran.bloqs.basic_gates import CNOT

        return CNOT()

    @_to_qt_bloq.register
    def _(op: qops.Hadamard):
        from qualtran.bloqs.basic_gates import Hadamard

        return Hadamard()

    @_to_qt_bloq.register
    def _(op: qops.Identity):
        from qualtran.bloqs.basic_gates import Identity

        return Identity()

    @_to_qt_bloq.register
    def _(op: qops.RX):
        from qualtran.bloqs.basic_gates import Rx

        return Rx(angle=float(op.data[0]))

    @_to_qt_bloq.register
    def _(op: qops.RY):
        from qualtran.bloqs.basic_gates import Ry

        return Ry(angle=float(op.data[0]))

    @_to_qt_bloq.register
    def _(op: qops.RZ):
        from qualtran.bloqs.basic_gates import Rz

        return Rz(angle=float(op.data[0]))

    @_to_qt_bloq.register
    def _(op: qops.S):
        from qualtran.bloqs.basic_gates import SGate

        return SGate()

    @_to_qt_bloq.register
    def _(op: qops.SWAP):
        from qualtran.bloqs.basic_gates import TwoBitSwap

        return TwoBitSwap()

    @_to_qt_bloq.register
    def _(op: qops.CSWAP):
        from qualtran.bloqs.basic_gates import TwoBitCSwap

        return TwoBitCSwap()

    @_to_qt_bloq.register
    def _(op: qops.T):
        from qualtran.bloqs.basic_gates import TGate

        return TGate()

    @_to_qt_bloq.register
    def _(op: qops.Toffoli):
        from qualtran.bloqs.basic_gates import Toffoli

        return Toffoli()

    @_to_qt_bloq.register
    def _(op: qops.Toffoli):
        from qualtran.bloqs.basic_gates import Toffoli

        return Toffoli()

    @_to_qt_bloq.register
    def _(op: qops.X):
        from qualtran.bloqs.basic_gates import XGate

        return XGate()

    @_to_qt_bloq.register
    def _(op: qops.Y):
        from qualtran.bloqs.basic_gates import YGate

        return YGate()

    @_to_qt_bloq.register
    def _(op: qops.CY):
        from qualtran.bloqs.basic_gates import CYGate

        return CYGate()

    @_to_qt_bloq.register
    def _(op: qops.Z):
        from qualtran.bloqs.basic_gates import ZGate

        return ZGate()

    @_to_qt_bloq.register
    def _(op: qops.CZ):
        from qualtran.bloqs.basic_gates import CZ

        return CZ()

    return _to_qt_bloq


@lru_cache
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
        This function requires the latest version of Qualtran. We recommend installing the main
        branch via ``pip``:

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

    # TODO: Remove when PL supports pylint==3.3.6 (it is considered a useless-suppression) [sc-91362]
    # pylint: disable=no-method-argument
    def compute_matrix(*params, **hyperparams):  # pylint: disable=no-self-argument
        bloq = hyperparams["bloq"]
        matrix = bloq.tensor_contract()

        if matrix.shape != (2 ** len(params[0].wires), 2 ** len(params[0].wires)):
            raise MatrixUndefinedError

        return matrix


def _split_qubits(registers, qubits):  # pylint: disable=redefined-outer-name
    """Function from the Qualtran-Cirq interop module that splits the flat list of qubits into
    a dictionary of appropriately shaped qubit arrays."""

    qubit_regs = {}
    base = 0
    for reg in registers:
        qubit_regs[reg.name] = np.array(qubits[base : base + reg.total_bits()]).reshape(
            reg.shape + (reg.bitsize,)
        )
        base += reg.total_bits()
    return qubit_regs


def _ensure_in_reg_exists(  # pylint: disable=too-many-branches
    bb: "qt.BloqBuilder",
    in_reg: "_QReg",
    qreg_to_qvar: Dict["_QReg", "qt.Soquet"],
) -> None:
    """Modified function from the Qualtran-Cirq interop module that takes care of qubit allocations,
    split and joins to ensure `qreg_to_qvar[in_reg]` exists."""
    from qualtran.cirq_interop._cirq_to_bloq import _QReg

    all_mapped_qubits = {q for qreg in qreg_to_qvar for q in qreg.qubits}
    qubits_to_allocate = [q for q in in_reg.qubits if q not in all_mapped_qubits]
    if qubits_to_allocate:
        n_alloc = len(qubits_to_allocate)
        qreg_to_qvar[
            _QReg(qubits_to_allocate, dtype=qt.QBit() if n_alloc == 1 else qt.QAny(n_alloc))
        ] = bb.allocate(n_alloc)

    if in_reg in qreg_to_qvar:
        # This is the easy case when no split / joins are needed.
        return

    # a. Split all registers containing at-least one qubit corresponding to `in_reg`.
    in_reg_qubits = set(in_reg.qubits)

    new_qreg_to_qvar: Dict[_QReg, qt.Soquet] = {}
    for qreg, soq in qreg_to_qvar.items():
        if len(qreg.qubits) > 1 and any(q in qreg.qubits for q in in_reg_qubits):
            new_qreg_to_qvar |= {
                _QReg(q, qt.QBit()): s for q, s in zip(qreg.qubits, bb.split(soq=soq))
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
                soqs_to_join[qreg.qubits[0]] = bb.add(
                    qt.bloqs.bookkeeping.Cast(qreg.dtype, qt.QBit()), reg=soq
                )
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
                qreg_to_qvar[in_reg] = bb.add(
                    qt.bloqs.bookkeeping.Cast(qt.QBit(), in_reg.dtype), reg=soq
                )
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


def _gather_input_soqs(bb: "qt.BloqBuilder", op_quregs, qreg_to_qvar):
    """Modified function from Qualtran-Cirq interop module that collects input Soquets."""
    qvars_in = {}
    for reg_name, quregs in op_quregs.items():
        flat_soqs: List[qt.Soquet] = []
        for qureg in quregs.flatten():
            _ensure_in_reg_exists(bb, qureg, qreg_to_qvar)
            flat_soqs.append(qreg_to_qvar[qureg])
        qvars_in[reg_name] = np.array(flat_soqs).reshape(quregs.shape)
    return qvars_in


def _inherit_from_bloq(cls):  # pylint: disable=too-many-statements
    """Decorator for ToBloq to inherit from Qualtran's bloq only when qualtran is available."""
    if qualtran:

        class ToBloq(qt.Bloq):  # pylint: disable=redefined-outer-name
            r"""
            Adapter class to convert PennyLane operators into Qualtran Bloqs
            """

            def __init__(self, op, **kwargs):
                from pennylane.workflow.qnode import QNode

                if not isinstance(op, Operation) and not isinstance(op, QNode):
                    raise TypeError(f"Input must be either an instance of {Operation} or {QNode}.")

                self.op = op
                self._kwargs = kwargs
                super().__init__()

            @cached_property
            def signature(self) -> "qt.Signature":
                """Compute and return Qualtran signature for given op or QNode."""
                from pennylane.workflow import construct_tape
                from pennylane.workflow.qnode import QNode

                if isinstance(self.op, QNode):
                    self.op.name = "QNode"
                    num_wires = len(construct_tape(self.op)(**self._kwargs).wires)
                else:
                    num_wires = len(self.op.wires)
                return qt.Signature([qt.Register("qubits", qt.QBit(), shape=num_wires)])

            def decompose_bloq(self):  # pylint:disable=too-many-branches
                """Decompose the bloq using the op's decomposition or the tape of the QNode"""
                from qualtran.cirq_interop._cirq_to_bloq import _QReg

                from pennylane.workflow import construct_tape
                from pennylane.workflow.qnode import QNode

                try:
                    if isinstance(self.op, QNode):
                        tape = construct_tape(self.op)(**self._kwargs)
                        ops = tape.circuit
                        all_wires = list(tape.wires)
                    else:
                        ops = self.op.decomposition()
                        all_wires = list(self.op.wires)

                    signature = self.signature
                    in_quregs = out_quregs = {
                        "qubits": np.array(all_wires).reshape(len(all_wires), 1)
                    }

                    in_quregs = {
                        k: np.apply_along_axis(_QReg, -1, *(v, signature.get_left(k).dtype))  # type: ignore
                        for k, v in in_quregs.items()
                    }

                    out_quregs = {
                        k: np.apply_along_axis(_QReg, -1, *(v, signature.get_right(k).dtype))  # type: ignore
                        for k, v in out_quregs.items()
                    }
                    bb, initial_soqs = qt.BloqBuilder.from_signature(
                        signature, add_registers_allowed=False
                    )

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
                        bloq = _map_to_bloq()(op)
                        if bloq.signature == qt.Signature([]):
                            bb.add(bloq)
                            continue

                        reg_dtypes = [r.dtype for r in bloq.signature]
                        # 3.1 Find input / output registers.
                        all_op_quregs = {
                            k: np.apply_along_axis(_QReg, -1, *(v, reg_dtypes[i]))  # type: ignore
                            for i, (k, v) in enumerate(
                                _split_qubits(bloq.signature, op.wires).items()
                            )
                        }

                        in_op_quregs = {
                            reg.name: all_op_quregs[reg.name] for reg in bloq.signature.lefts()
                        }

                        # 3.2 Find input Soquets, by potentially allocating new Bloq registers corresponding to
                        # input `in_quregs` and updating the `qreg_to_qvar` mapping.
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
                        bb,
                        {reg.name: out_quregs[reg.name] for reg in signature.rights()},
                        qreg_to_qvar,
                    )
                    final_soqs_set = set(
                        soq for soqs in final_soqs_dict.values() for soq in soqs.flatten()
                    )
                    # 5. Free all dangling Soquets which are not part of the final soquets set.
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
                call_graph = _get_op_call_graph()(self.op)
                if call_graph:
                    return call_graph

                return self.decompose_bloq().build_call_graph(ssa)

            def __repr__(self):
                if isinstance(self.op, Operation):
                    return f"ToBloq({self.op.name})"
                return "ToBloq(QNode)"

            def __eq__(self, other):
                if type(other) is type(self):
                    return self.op == other.op
                return False

            def __hash__(self):
                return hash(self.op)

            def __str__(self):
                return "PL" + self.op.name

        return ToBloq
    return cls


@_inherit_from_bloq
class ToBloq:
    r"""
    An adapter for using a PennyLane :class:`~.Operation` as a
    `Qualtran Bloq <https://qualtran.readthedocs.io/en/latest/bloqs/index.html#bloqs-library>`_.

    .. note::
        This class requires the latest version of Qualtran. We recommend installing the main
        branch via ``pip``:

        .. code-block:: console

            pip install qualtran

    Args:
        op (Operation): an initialized PennyLane operator to be wrapped as a Qualtran ``Bloq``.

    Raises:
        TypeError: operator must be an instance of :class:`~.Operation`.

    **Example**

    This example shows how to use ``qml.ToBloq``:

    >>> wrapped_op = qml.ToBloq(qml.CNOT([0, 1]))
    >>> wrapped_op.tensor_contract()
    array([[1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j]])
    """

    _dependency_missing = True
    _error_message = (
        "Optional dependency 'qualtran' is required "
        "for ToBloq functionality but is not installed."
    )

    # Prevent instantiation if the dependency is missing
    def __init__(self, *args, **kwargs):
        raise ImportError(self._error_message)

    def __getattr__(self, name):
        raise ImportError(self._error_message)

    def __call__(self, *args, **kwargs):
        raise ImportError(self._error_message)


def to_bloq(circuit, map_ops: bool = True, custom_mapping: dict = None, **kwargs):
    """
    Converts the given circuit or :class:`~.Operation` and returns the appropriate `Qualtran Bloq <https://qualtran.readthedocs.io/en/latest/bloqs/index.html#bloqs-library>`_.

    .. note::
        This class requires the latest version of Qualtran. We recommend installing the main
        branch via ``pip``:

        .. code-block:: console

            pip install qualtran

    Args:
        circuit (QNode | Operation): a QNode or an initialized PennyLane operator to be wrapped as a Qualtran Bloq.
        map_ops (bool): Whether or not if the operations are mapped to a Qualtran Bloq or wrapped
            as a `ToBloq`. Default is True.
        custom_mapping (dict): Dictionary to specify a mapping between a PennyLane operator and a
            Qualtran Bloq. A default mapping is used if not defined.

    Returns:
        Bloq: The Qualtran Bloq that corresponds to the given circuit or :class:`~.Operation` and
            options.

    **Example**

    This example shows how to use ``qml.to_bloq``:

    >>> qt_bloq = qml.to_bloq(qml.CNOT([0, 1]))
    >>> qt_bloq
    CNOT()

    .. details::
        :title: Usage Details

        Not every operator has a direct equivalent in Qualtran. For example, in Qualtran, there
        exists many varieties of Quantum Phase Estimation. When ``qml.to_bloq`` is called on
        Quantum Phase Estimation, a smart default is chosen.

        >>> qml.to_bloq(qml.QuantumPhaseEstimation(
        ...     unitary=qml.RX(0.1, wires=0), estimation_wires=range(1, 5)
        ... ))
        TextbookQPE(unitary=Rx(angle=0.1, eps=1e-11), ctrl_state_prep=RectangularWindowState(bitsize=4), qft_inv=Adjoint(subbloq=QFTTextBook(bitsize=4, with_reverse=True)))

        Note that the chosen Qualtran Bloq may not be an exact equivalent. If you want an exact
        equivalent, we recommend setting ``map_ops`` to False.

        >>> qml.to_bloq(qml.QuantumPhaseEstimation(
        ...     unitary=qml.RX(0.1, wires=0), estimation_wires=range(1, 5)
        ... ), map_ops=False)
        ToBloq(QuantumPhaseEstimation)

        This will wrap the input PennyLane operator as a Bloq, allowing you to use the functions
        you would expect a Bloq to have, such as ``decompose_bloq`` or ``call_graph``.

        Alternatively, you can provide a custom mapping that maps a PennyLane operator to a
        specific Qualtran Bloq.

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

    if map_ops:
        if custom_mapping:
            return _map_to_bloq()(circuit, custom_mapping=custom_mapping, **kwargs)
        return _map_to_bloq()(circuit, **kwargs)

    return ToBloq(circuit, **kwargs)
