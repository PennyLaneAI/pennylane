# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""
Contains the ``rot_to_phase_gradient`` transform.
"""
# pylint: disable=too-many-branches
import numpy as np

import pennylane as qp
from pennylane.operation import Operator
from pennylane.queuing import QueuingManager
from pennylane.transforms.rz_phase_gradient import _rz_phase_gradient
from pennylane.wires import Wires


def ladder(wires):
    """ladder operator"""
    if len(wires) == 1:
        return qp.I(wires)
    return qp.prod(*[qp.CNOT(wires) for wires in zip(wires[1:], wires[:-1])])


# pylint: disable=too-many-arguments
def _select_pauli_rot_phase_gradient(
    phis: np.ndarray,
    rot_axis: str,
    control_wires: Wires,
    target_wire: Wires,
    angle_wires: Wires,
    phase_grad_wires: Wires,
    work_wires: Wires,
) -> Operator:
    """Function that transforms the SelectPauliRot gate to the phase gradient circuit
    The precision is implicitly defined by the length of ``angle_wires``
    """

    precision = len(angle_wires)
    binary_int = qp.math.binary_decimals(phis, precision, unit=4 * np.pi)

    ops = [
        qp.QROM(
            binary_int, control_wires, angle_wires, work_wires=work_wires[: len(control_wires) - 1]
        )
    ] + [qp.ctrl(qp.X(wire), control=target_wire, control_values=[0]) for wire in phase_grad_wires]
    # The uncomputation does not need any adjoints because both QROM and C(X) are self-adjoint.
    adj_ops = ops[::-1]

    pg_op = qp.change_op_basis(
        qp.prod(*ops[::-1]),
        qp.SemiAdder(angle_wires, phase_grad_wires, work_wires=work_wires[: len(angle_wires) - 1]),
        qp.prod(*adj_ops[::-1]),
    )

    match rot_axis:
        case "X":
            comp = uncomp = qp.Hadamard(target_wire)
            pg_op = qp.change_op_basis(comp, pg_op, uncomp)
        case "Y":
            comp = qp.Hadamard(target_wire) @ qp.adjoint(qp.S(target_wire))
            uncomp = qp.S(target_wire) @ qp.Hadamard(target_wire)
            pg_op = qp.change_op_basis(comp, pg_op, uncomp)

    return pg_op


def _pauli_rot_phase_gradient(op, **other_wires):
    wires = op.wires
    phi = op.parameters[0]
    if isinstance(op, (qp.IsingXX, qp.IsingYY, qp.IsingZZ)):
        with QueuingManager.stop_recording():
            pauli_word = op.name[-2:]
            op = qp.PauliRot(phi, pauli_word=pauli_word, wires=wires)

    # collect diagonalizing gates of each wire
    # this turns any rotation to MultiRZ
    diagonalizing_gates = []
    for sub_op in op.decomposition():
        if isinstance(sub_op, qp.MultiRZ):
            break
        diagonalizing_gates.append(sub_op)

    diagonalizing_gate = ladder(wires) @ qp.prod(*diagonalizing_gates[::-1])
    diagonalizing_gate_inv = qp.prod(*diagonalizing_gates) @ ladder(wires)

    pg_op = _rz_phase_gradient(phi, wires[:1], **other_wires)
    new_op = qp.change_op_basis(diagonalizing_gate, pg_op, diagonalizing_gate_inv)

    return new_op, phi / 2  # op to be appended, global phase
