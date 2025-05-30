# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

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
This submodule defines functions to decompose controlled operations
"""

from typing import Literal

import numpy as np

import pennylane as qml
from pennylane.operation import Operation, Operator
from pennylane.wires import Wires, WiresLike

from .decompositions.controlled_decompositions import ctrl_decomp_bisect, ctrl_decomp_zyz


def _is_single_qubit_special_unitary(op):
    mat = op.matrix()
    det = mat[0, 0] * mat[1, 1] - mat[0, 1] * mat[1, 0]
    return qml.math.allclose(det, 1)


def decompose_mcx(
    control_wires, target_wire, work_wires, work_wire_type: Literal["clean", "dirty"] = "clean"
):
    """Decomposes the multi-controlled PauliX"""

    n_ctrl_wires, n_work_wires = len(control_wires), len(work_wires)
    if n_ctrl_wires == 1:
        return [qml.CNOT(wires=control_wires + Wires(target_wire))]
    if n_ctrl_wires == 2:
        return qml.Toffoli.compute_decomposition(wires=control_wires + Wires(target_wire))

    if n_work_wires >= n_ctrl_wires - 2:
        # Lemma 7.2 of `Barenco et al. (1995) <https://arxiv.org/abs/quant-ph/9503016>`_
        return _decompose_mcx_with_many_workers(control_wires, target_wire, work_wires)
    if n_work_wires >= 2:
        return _decompose_mcx_with_two_workers(
            control_wires, target_wire, work_wires[0:2], work_wire_type
        )
    if n_work_wires == 1:
        return _decompose_mcx_with_one_worker_kg24(
            control_wires, target_wire, work_wires[0], work_wire_type
        )

    # Lemma 7.5
    with qml.QueuingManager.stop_recording():
        op = qml.X(target_wire)
    return _decompose_multicontrolled_unitary(op, control_wires)


def _decompose_multicontrolled_unitary(op, control_wires):
    """Decomposes general multi controlled unitary with no work wires
    Follows approach from Lemma 7.5 combined with 7.3 and 7.2 of
    https://arxiv.org/abs/quant-ph/9503016.

    We are assuming this decomposition is used only in the general cases
    """
    if not op.has_matrix or len(op.wires) != 1:
        raise ValueError(
            "The target operation must be a single-qubit operation with a matrix representation"
        )

    target_wire = op.wires
    if len(control_wires) == 0:
        return [op]
    if len(control_wires) == 1:
        return ctrl_decomp_zyz(op, control_wires)
    if _is_single_qubit_special_unitary(op):
        return ctrl_decomp_bisect(op, control_wires)
    # use recursive decomposition of general gate
    return _decompose_recursive(op, 1.0, control_wires, target_wire, Wires([]))


def _decompose_recursive(op, power, control_wires, target_wire, work_wires):
    """Decompose multicontrolled operator recursively using lemma 7.5
    Number of gates in decomposition are: O(len(control_wires)^2)
    """
    if len(control_wires) == 1:
        with qml.QueuingManager.stop_recording():
            powered_op = qml.pow(op, power, lazy=True)
        return ctrl_decomp_zyz(powered_op, control_wires)

    with qml.QueuingManager.stop_recording():
        cnots = decompose_mcx(
            control_wires=control_wires[:-1],
            target_wire=control_wires[-1],
            work_wires=work_wires + target_wire,
            work_wire_type="dirty",
        )
    with qml.QueuingManager.stop_recording():
        powered_op = qml.pow(op, 0.5 * power, lazy=True)
        powered_op_adj = qml.adjoint(powered_op, lazy=True)

    if qml.QueuingManager.recording():
        decomposition = [
            *ctrl_decomp_zyz(powered_op, control_wires[-1]),
            *(qml.apply(o) for o in cnots),
            *ctrl_decomp_zyz(powered_op_adj, control_wires[-1]),
            *(qml.apply(o) for o in cnots),
            *_decompose_recursive(
                op, 0.5 * power, control_wires[:-1], target_wire, control_wires[-1] + work_wires
            ),
        ]
    else:
        decomposition = [
            *ctrl_decomp_zyz(powered_op, control_wires[-1]),
            *cnots,
            *ctrl_decomp_zyz(powered_op_adj, control_wires[-1]),
            *cnots,
            *_decompose_recursive(
                op, 0.5 * power, control_wires[:-1], target_wire, control_wires[-1] + work_wires
            ),
        ]
    return decomposition


def _decompose_mcx_with_many_workers(control_wires, target_wire, work_wires):
    """Decomposes the multi-controlled PauliX gate using the approach in Lemma 7.2 of
    https://arxiv.org/abs/quant-ph/9503016, which requires a suitably large register of
    work wires"""
    num_work_wires_needed = len(control_wires) - 2
    work_wires = work_wires[:num_work_wires_needed]

    work_wires_reversed = list(reversed(work_wires))
    control_wires_reversed = list(reversed(control_wires))

    gates = []

    for i in range(len(work_wires)):
        ctrl1 = control_wires_reversed[i]
        ctrl2 = work_wires_reversed[i]
        t = target_wire if i == 0 else work_wires_reversed[i - 1]
        gates.append(qml.Toffoli(wires=[ctrl1, ctrl2, t]))

    gates.append(qml.Toffoli(wires=[*control_wires[:2], work_wires[0]]))

    for i in reversed(range(len(work_wires))):
        ctrl1 = control_wires_reversed[i]
        ctrl2 = work_wires_reversed[i]
        t = target_wire if i == 0 else work_wires_reversed[i - 1]
        gates.append(qml.Toffoli(wires=[ctrl1, ctrl2, t]))

    for i in range(len(work_wires) - 1):
        ctrl1 = control_wires_reversed[i + 1]
        ctrl2 = work_wires_reversed[i + 1]
        t = work_wires_reversed[i]
        gates.append(qml.Toffoli(wires=[ctrl1, ctrl2, t]))

    gates.append(qml.Toffoli(wires=[*control_wires[:2], work_wires[0]]))

    for i in reversed(range(len(work_wires) - 1)):
        ctrl1 = control_wires_reversed[i + 1]
        ctrl2 = work_wires_reversed[i + 1]
        t = work_wires_reversed[i]
        gates.append(qml.Toffoli(wires=[ctrl1, ctrl2, t]))

    return gates


def _decompose_mcx_with_one_worker_b95(control_wires, target_wire, work_wire):
    """
    Decomposes the multi-controlled PauliX gate using the approach in Lemma 7.3 of
    `Barenco et al. (1995) <https://arxiv.org/abs/quant-ph/9503016>`_, which requires
    a single work wire. This approach requires O(16k) CX gates, where k is the number of control wires.
    """
    tot_wires = len(control_wires) + 2
    partition = int(np.ceil(tot_wires / 2))

    first_part = control_wires[:partition]
    second_part = control_wires[partition:]

    gates = [
        qml.ctrl(qml.X(work_wire), control=first_part, work_wires=second_part + target_wire),
        qml.ctrl(qml.X(target_wire), control=second_part + work_wire, work_wires=first_part),
        qml.ctrl(qml.X(work_wire), control=first_part, work_wires=second_part + target_wire),
        qml.ctrl(qml.X(target_wire), control=second_part + work_wire, work_wires=first_part),
    ]

    return gates


def _linear_depth_ladder_ops(wires: WiresLike) -> tuple[list[Operator], int]:
    r"""
    Helper function to create linear-depth ladder operations used in Khattar and Gidney's MCX synthesis.
    In particular, this implements Step-1 and Step-2 on Fig. 3 of [1] except for the first and last
    CCX gates.

    Preconditions:
        - The number of wires must be greater than 2.

    Args:
        wires (Wires): Wires to apply the ladder operations on.

    Returns:
        tuple[list[Operator], int]: Linear-depth ladder circuit and the index of control qubit to
        apply the final CCX gate.

    References:
        1. Khattar and Gidney, Rise of conditionally clean ancillae for optimizing quantum circuits
        `arXiv:2407.17966 <https://arxiv.org/abs/2407.17966>`__
    """

    n = len(wires)
    assert n > 2, "n_ctrls > 2 to use MCX ladder. Otherwise, use CCX"

    gates = []
    # up-ladder
    for i in range(1, n - 2, 2):
        gates.append(qml.Toffoli(wires=[wires[i + 1], wires[i + 2], wires[i]]))
        gates.append(qml.PauliX(wires=wires[i]))

    # down-ladder
    if n % 2 == 0:
        ctrl_1, ctrl_2, target = n - 3, n - 5, n - 6
    else:
        ctrl_1, ctrl_2, target = n - 1, n - 4, n - 5

    if target >= 0:
        gates.append(qml.Toffoli(wires=[wires[ctrl_1], wires[ctrl_2], wires[target]]))
        gates.append(qml.PauliX(wires=wires[target]))

    for i in range(target, 1, -2):
        gates.append(qml.Toffoli(wires=[wires[i], wires[i - 1], wires[i - 2]]))
        gates.append(qml.PauliX(wires=wires[i - 2]))

    final_ctrl = max(0, 5 - n)

    return gates, final_ctrl


def _decompose_mcx_with_one_worker_kg24(
    control_wires: WiresLike,
    target_wire: int,
    work_wire: int,
    work_wire_type: Literal["clean", "dirty"] = "clean",
) -> list[Operator]:
    r"""
    Synthesise a multi-controlled X gate with :math:`k` controls using :math:`1` ancillary qubit. It
    produces a circuit with :math:`2k-3` Toffoli gates and depth :math:`O(k)` if the ancilla is clean
    and :math:`4k-3` Toffoli gates and depth :math:`O(k)` if the ancilla is dirty as described in
    Sec. 5.1 of [1].

    Args:
        control_wires (Wires): the control wires
        target_wire (int): the target wire
        work_wires (Wires): the work wires used to decompose the gate
        work_wire_type (string): If "dirty", perform un-computation. Default is "clean".

    Returns:
        list[Operator]: the synthesized quantum circuit

    References:
        1. Khattar and Gidney, Rise of conditionally clean ancillae for optimizing quantum circuits
        `arXiv:2407.17966 <https://arxiv.org/abs/2407.17966>`__
    """

    gates = []
    gates.append(qml.Toffoli(wires=[control_wires[0], control_wires[1], work_wire]))
    ladder_ops, final_ctrl = _linear_depth_ladder_ops(control_wires)
    gates += ladder_ops
    gates.append(qml.Toffoli(wires=[work_wire, control_wires[final_ctrl], target_wire]))
    gates += ladder_ops[::-1]
    gates.append(qml.Toffoli(wires=[control_wires[0], control_wires[1], work_wire]))

    if work_wire_type == "dirty":
        # perform toggle-detection if ancilla is dirty
        gates += ladder_ops
        gates.append(qml.Toffoli(wires=[work_wire, control_wires[final_ctrl], target_wire]))
        gates += ladder_ops[::-1]

    return gates


def _n_parallel_ccx_x(
    control_wires_x: WiresLike, control_wires_y: WiresLike, target_wires: WiresLike
) -> list[Operation]:
    r"""
    Construct a quantum circuit for creating n-condionally clean ancillae using 3n qubits. This
    implements Fig. 4a of [1]. Each wire is of the same size :math:`n`.

    Args:
        control_wires_x (Wires): The control wires for register 1.
        control_wires_y (Wires): The control wires for register 2.
        target_wires (Wires): The wires for target register.

    Returns:
        list[Operation]: The quantum circuit for creating n-condionally clean ancillae.

    References:
        1. Khattar and Gidney, Rise of conditionally clean ancillae for optimizing quantum circuits
        `arXiv:2407.17966 <https://arxiv.org/abs/2407.17966>`__
    """

    assert (
        len(control_wires_x) == len(control_wires_y) == len(target_wires)
    ), "The number of wires must be the same for x, y, and target."

    gates = []
    for i, ctrl_x in enumerate(control_wires_x):
        gates.append(qml.X(wires=target_wires[i]))
        gates.append(qml.Toffoli(wires=[ctrl_x, control_wires_y[i], target_wires[i]]))

    return gates


def _build_logn_depth_ccx_ladder(
    work_wire: int, control_wires: WiresLike
) -> tuple[list[Operator], list[Operator]]:
    r"""
    Helper function to build a log-depth ladder compose of CCX and X gates as shown in Fig. 4b of [1].

    Args:
        work_wire (int): The work wire.
        control_wires (list[Wire]): The control wires.

    Returns:
        tuple[list[Operator], WiresLike: log-depth ladder circuit of cond. clean ancillae and
        control_wires to apply the linear-depth MCX gate on.

    References:
        1. Khattar and Gidney, Rise of conditionally clean ancillae for optimizing quantum circuits
        `arXiv:2407.17966 <https://arxiv.org/abs/2407.17966>`__
    """

    gates = []
    anc = [work_wire]
    final_ctrls = []

    while len(control_wires) > 1:
        next_batch_len = min(len(anc) + 1, len(control_wires))
        control_wires, nxt_batch = control_wires[next_batch_len:], control_wires[:next_batch_len]
        new_anc = []
        while len(nxt_batch) > 1:
            ccx_n = len(nxt_batch) // 2
            st = int(len(nxt_batch) % 2)
            ccx_x, ccx_y, ccx_t = (
                nxt_batch[st : st + ccx_n],
                nxt_batch[st + ccx_n :],
                anc[-ccx_n:],
            )
            assert len(ccx_x) == len(ccx_y) == len(ccx_t) == ccx_n >= 1
            if ccx_t != [work_wire]:
                gates += _n_parallel_ccx_x(ccx_x, ccx_y, ccx_t)
            else:
                gates.append(qml.Toffoli(wires=[ccx_x[0], ccx_y[0], ccx_t[0]]))
            new_anc += nxt_batch[st:]  # newly created cond. clean ancilla
            nxt_batch = ccx_t + nxt_batch[:st]
            anc = anc[:-ccx_n]

        anc = sorted(anc + new_anc)
        final_ctrls += nxt_batch

    final_ctrls += control_wires
    final_ctrls = sorted(final_ctrls)
    final_ctrls.remove(work_wire)  #                        # exclude ancilla
    return gates, final_ctrls


def _decompose_mcx_with_two_workers(
    control_wires: WiresLike,
    target_wire: int,
    work_wires: WiresLike,
    work_wire_type: Literal["clean", "dirty"] = "clean",
) -> list[Operator]:
    r"""
    Synthesise a multi-controlled X gate with :math:`k` controls using :math:`2` ancillary qubits.
    It produces a circuit with :math:`2k-3` Toffoli gates and depth :math:`O(\log(k))` if using
    clean ancillae, and :math:`4k-8` Toffoli gates and depth :math:`O(\log(k))` if using dirty
    ancillae as described in Sec. 5 of [1].

    Args:
        control_wires (Wires): The control wires.
        target_wire (int): The target wire.
        work_wires (Wires): The work wires.
        work_wire_type (string): If "dirty" perform uncomputation after we're done. Default is "clean".

    Returns:
        list[Operator]: The synthesized quantum circuit.

    References:
        1. Khattar and Gidney, Rise of conditionally clean ancillae for optimizing quantum circuits
        `arXiv:2407.17966 <https://arxiv.org/abs/2407.17966>`__
    """

    if len(work_wires) < 2:
        raise ValueError("At least 2 work wires are needed for this decomposition.")

    gates = []
    ladder_ops, final_ctrls = _build_logn_depth_ccx_ladder(work_wires[0], control_wires)
    gates += ladder_ops
    if len(final_ctrls) == 1:  # Already a toffoli
        gates.append(qml.Toffoli(wires=[work_wires[0], final_ctrls[0], target_wire]))
    else:
        mid_mcx = _decompose_mcx_with_one_worker_kg24(
            work_wires[0:1] + final_ctrls, target_wire, work_wires[1], work_wire_type="clean"
        )
        gates += mid_mcx
    gates += ladder_ops[::-1]

    if work_wire_type == "dirty":
        # perform toggle-detection if ancilla is dirty
        gates += ladder_ops[1:]
        if len(final_ctrls) == 1:
            gates.append(qml.Toffoli(wires=[work_wires[0], final_ctrls[0], target_wire]))
        else:
            gates += mid_mcx
        gates += ladder_ops[1:][::-1]

    return gates
