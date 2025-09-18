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
from pennylane.operation import Operator
from pennylane.wires import Wires, WiresLike

from .decompositions.controlled_decompositions import (
    _mcx_many_workers,
    _mcx_one_worker,
    _mcx_two_workers,
    ctrl_decomp_bisect,
    ctrl_decomp_zyz,
)


def _is_single_qubit_special_unitary(op):
    mat = op.matrix()
    det = mat[0, 0] * mat[1, 1] - mat[0, 1] * mat[1, 0]
    return qml.math.allclose(det, 1)


def decompose_mcx(
    control_wires,
    target_wire,
    work_wires,
    work_wire_type: Literal["zeroed", "borrowed"] = "borrowed",
):
    """Decomposes the multi-controlled PauliX"""

    n_ctrl_wires, n_work_wires = len(control_wires), len(work_wires)
    if n_ctrl_wires == 1:
        return [qml.CNOT(wires=control_wires + Wires(target_wire))]
    if n_ctrl_wires == 2:
        return qml.Toffoli.compute_decomposition(wires=control_wires + Wires(target_wire))

    if n_work_wires >= n_ctrl_wires - 2:
        # Lemma 7.2 of `Barenco et al. (1995) <https://arxiv.org/abs/quant-ph/9503016>`_
        return _decompose_mcx_with_many_workers_old(
            control_wires, target_wire, work_wires, work_wire_type
        )
    if n_work_wires >= 2:
        return _decompose_mcx_with_two_workers_old(
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
            work_wire_type="borrowed",
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


def _decompose_mcx_with_many_workers_old(control_wires, target_wire, work_wires, work_wire_type):
    """Decomposes the multi-controlled PauliX gate using the approach in Lemma 7.2 of
    https://arxiv.org/abs/quant-ph/9503016, which requires a suitably large register of
    work wires"""

    with qml.queuing.AnnotatedQueue() as q:
        wires = list(control_wires) + [target_wire]
        _mcx_many_workers(wires=wires, work_wires=work_wires, work_wire_type=work_wire_type)

    if qml.QueuingManager.recording():
        for op in q.queue:  # pragma: no cover
            qml.apply(op)

    return q.queue


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


def _decompose_mcx_with_one_worker_kg24(
    control_wires: WiresLike,
    target_wire: int,
    work_wire: int,
    work_wire_type: Literal["zeroed", "borrowed"] = "borrowed",
) -> list[Operator]:
    r"""
    Synthesise a multi-controlled X gate with :math:`k` controls using :math:`1` auxiliary qubit. It
    produces a circuit with :math:`2k-3` Toffoli gates and depth :math:`O(k)` if the auxiliary is zeroed
    and :math:`4k-3` Toffoli gates and depth :math:`O(k)` if the auxiliary is borrowed as described in
    Sec. 5.1 of [1].

    Args:
        control_wires (Wires): the control wires
        target_wire (int): the target wire
        work_wires (Wires): the work wires used to decompose the gate
        work_wire_type (string): If "borrowed", perform un-computation. Default is "borrowed".

    Returns:
        list[Operator]: the synthesized quantum circuit

    References:
        1. Khattar and Gidney, Rise of conditionally clean ancillae for optimizing quantum circuits
        `arXiv:2407.17966 <https://arxiv.org/abs/2407.17966>`__
    """

    with qml.queuing.AnnotatedQueue() as q:
        wires = list(control_wires) + [target_wire]
        _mcx_one_worker(wires=wires, work_wires=[work_wire], work_wire_type=work_wire_type)

    if qml.QueuingManager.recording():
        for op in q.queue:  # pragma: no cover
            qml.apply(op)

    return q.queue


def _decompose_mcx_with_two_workers_old(
    control_wires: WiresLike,
    target_wire: int,
    work_wires: WiresLike,
    work_wire_type: Literal["zeroed", "borrowed"] = "borrowed",
) -> list[Operator]:
    r"""
    Synthesise a multi-controlled X gate with :math:`k` controls using :math:`2` auxiliary qubits.
    It produces a circuit with :math:`2k-3` Toffoli gates and depth :math:`O(\log(k))` if using
    zeroed auxiliary qubits, and :math:`4k-8` Toffoli gates and depth :math:`O(\log(k))` if using borrowed
    auxiliary qubits as described in Sec. 5 of [1].

    Args:
        control_wires (Wires): The control wires.
        target_wire (int): The target wire.
        work_wires (Wires): The work wires.
        work_wire_type (string): If "borrowed" perform uncomputation after we're done. Default is "borrowed".

    Returns:
        list[Operator]: The synthesized quantum circuit.

    References:
        1. Khattar and Gidney, Rise of conditionally clean ancillae for optimizing quantum circuits
        `arXiv:2407.17966 <https://arxiv.org/abs/2407.17966>`__
    """

    if len(work_wires) < 2:
        raise ValueError("At least 2 work wires are needed for this decomposition.")

    with qml.queuing.AnnotatedQueue() as q:
        wires = list(control_wires) + [target_wire]
        _mcx_two_workers(wires=wires, work_wires=work_wires, work_wire_type=work_wire_type)

    if qml.QueuingManager.recording():
        for op in q.queue:  # pragma: no cover
            qml.apply(op)

    return q.queue
