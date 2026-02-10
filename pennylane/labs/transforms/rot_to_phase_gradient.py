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
r"""
Contains the ``select_pauli_rot_phase_gradient`` transform.
"""
import numpy as np

import pennylane as qp
from pennylane.operation import Operator
from pennylane.queuing import QueuingManager
from pennylane.tape import QuantumScript, QuantumScriptBatch
from pennylane.transforms import transform

# from pennylane.transforms.rz_phase_gradient import _rz_phase_gradient
from pennylane.typing import PostprocessingFn
from pennylane.wires import Wires


def _binary_repr_int(phi, precision):
    """
    Reasoning for +1e-10 term:
    due to the division by pi, we obtain 14.999.. instead of 15 for, e.g., (1, 1, 1, 1) pi
    at the same time, we want to floor off any additional floats when converting to the desired precision,
    e.g. representing (1, 1, 1, 1) with only 3 digits we want to obtain (1, 1, 1)
    so overall we floor but make sure we add a little term to not accidentally write 14 when the result is 14.999..
    """
    phi = phi % (4 * np.pi)
    phi_round = np.round(2**precision * phi / (2 * np.pi))
    return bin(int(np.floor(phi_round / 2 + 1e-10)) + 2 * 2**precision)[-precision:]


def _binary_repr_int2(phi, precision):
    phi = phi % (2 * np.pi)
    phi_round = np.round(2**precision * phi / (2 * np.pi))
    return int(bin(int(np.floor(phi_round + 1e-10)))[-precision:], 2)


@QueuingManager.stop_recording()
def _rz_phase_gradient(
    phi: float, wire: Wires, angle_wires: Wires, phase_grad_wires: Wires, work_wires: Wires
) -> Operator:
    """Function that transforms the RZ gate to the phase gradient circuit
    The precision is implicitly defined by the length of ``angle_wires``
    Note that the global phases are collected and added as one big global phase in the main function
    """
    # variation of pennylane.transforms.rz_phase_gradient._rz_phase_gradient without the need to collect global phases, as done in

    precision = len(angle_wires)
    # BasisEmbedding can handle integer inputs, no need to actually translate to binary
    binary_int = int(_binary_repr_int(phi * 2, precision), base=2)

    compute_op = qp.ctrl(qp.BasisEmbedding(features=binary_int, wires=angle_wires), control=wire)
    target_op = qp.SemiAdder(angle_wires, phase_grad_wires, work_wires)

    return qp.change_op_basis(compute_op, target_op, compute_op)


# pylint: disable=too-many-arguments
def _select_pauli_rot_phase_gradient(
    phis: list,
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
    binary_int = [_binary_repr_int(phi, precision) for phi in phis]

    ops = [
        qp.QROM(
            binary_int, control_wires, angle_wires, work_wires=work_wires[len(angle_wires) - 1 :]
        )
    ] + [qp.ctrl(qp.X(wire), control=target_wire, control_values=[0]) for wire in phase_grad_wires]

    return qp.change_op_basis(
        qp.prod(*ops[::-1]),
        qp.SemiAdder(angle_wires, phase_grad_wires, work_wires[: len(angle_wires) - 1]),
    )


@transform
def rot_to_phase_gradient(
    tape: QuantumScript, angle_wires: Wires, phase_grad_wires: Wires, work_wires: Wires
) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    r"""Quantum function transform to discretize all rotation gates using the `phase gradient decomposition <https://pennylane.ai/compilation/phase-gradient/>`__.

    For this routine to work, the provided ``phase_grad_wires`` need to hold the phase gradient
    state :math:`|\nabla_Z\rangle = \frac{1}{\sqrt{2^n}} \sum_{m=0}^{2^n-1} e^{-2 \pi i \frac{m}{2^n}} |m\rangle`.
    Because this state is not modified and can be re-used at a later stage, the transform does not prepare it but
    rather assumes it has been prepared on those wires at an earlier stage. Look at the example below to see how
    this state can be prepared.

    Note that this operator contains :class:`~.SemiAdder` that typically uses additional ``work_wires`` for the semi-in-place addition
    :math:`\text{SemiAdder}|x\rangle_\text{ang} |y\rangle_\text{phg} = |x\rangle_\text{ang} |x + y\rangle_\text{phg}`.


    Args:
        tape (QNode or QuantumTape or Callable): A quantum circuit containing :class:`~.SelectPauliRot` operators.
        angle_wires (Wires): The qubits that conditionally load the angle :math:`\phi` of
            the :class:`~.SelectPauliRot` gate in binary as a multiple of :math:`2\pi`.
            The length of the ``angle_wires`` , i.e. :math:`b`, implicitly determines the precision
            with which the angle is represented.
            E.g., :math:`(1 \cdot 2^{-1} + 0 \cdot 2^{-2} + 1 \cdot 2^{-3}) 2\pi` is represented by three bits as ``101``.
        phase_grad_wires (Wires): Qubits with the catalytic phase gradient state prepared on them.
            Needs to be at least :math:`b` wires and will only use the first :math:`b`.
        work_wires (Wires): Additional work wires to realize the :class:`~.SemiAdder` and :class:`~.QROM`.
            Needs to be at least :math:`b-1` wires.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[QuantumTape], function]: The transformed circuit as described in :func:`qp.transform <pennylane.transform>`.

    **Example**

    .. code-block:: python

        from pennylane.labs.transforms import select_pauli_rot_phase_gradient
        from functools import partial
        import numpy as np

        precision = 4
        wire = "targ"
        angle_wires = [f"ang_{i}" for i in range(precision)]
        phase_grad_wires = [f"phg_{i}" for i in range(precision)]
        work_wires = [f"work_{i}" for i in range(precision - 1)]

        def phase_gradient(wires):
            # prepare phase gradient state
            for i, w in enumerate(wires):
                qp.H(w)
                qp.PhaseShift(-np.pi / 2**i, w)

        @partial(
            select_pauli_rot_phase_gradient,
            angle_wires=angle_wires,
            phase_grad_wires=phase_grad_wires,
            work_wires=work_wires,
        )
        @qp.qnode(qp.device("default.qubit"))
        def select_pauli_rot_circ(phis, control_wires, target_wire):
            phase_gradient(phase_grad_wires)  # prepare phase gradient state

            for wire in control_wires:
                qp.Hadamard(wire)

            qp.SelectPauliRot(phis, control_wires, target_wire, rot_axis="X")

            return qp.probs(target_wire)

        phis = [
            (1 / 2 + 1 / 4 + 1 / 8) * 2 * np.pi,
            (1 / 2 + 1 / 4 + 0 / 8) * 2 * np.pi,
            (1 / 2 + 0 / 4 + 1 / 8) * 2 * np.pi,
            (0 / 2 + 1 / 4 + 1 / 8) * 2 * np.pi,
        ]

    >>> print(select_pauli_rot_circ(phis, control_wires=[0, 1], target_wire=wire))
    [0.41161165 0.58838835]
    """

    if len(phase_grad_wires) < len(angle_wires):
        raise ValueError(
            f"phase_grad_wires needs to be at least as large as angle_wires. Got {len(phase_grad_wires)} phase_grad_wires, which is fewer than the {len(angle_wires)} angle_wires."
        )

    if len(work_wires) < len(angle_wires) - 1:
        raise ValueError(
            f"work_wires needs to be at least as large as angle_wires - 1. Got {len(work_wires)} work_wires, which is fewer than the {len(angle_wires) - 1}."
        )

    operations = []
    global_phases = []

    for op in tape.operations:
        if isinstance(op, qp.SelectPauliRot):
            control_wires = op.wires[:-1]
            target_wire = op.wires[-1]
            rot_axis = op.hyperparameters["rot_axis"]

            angles = op.parameters[0]

            pg_op = _select_pauli_rot_phase_gradient(
                angles,
                control_wires=control_wires,
                target_wire=target_wire,
                angle_wires=angle_wires,
                phase_grad_wires=phase_grad_wires,
                work_wires=work_wires,
            )

            match rot_axis:
                case "X":
                    operations.append(
                        qp.change_op_basis(
                            qp.Hadamard(target_wire),
                            pg_op,
                            qp.Hadamard(target_wire),
                        )
                    )
                case "Y":
                    operations.append(
                        qp.change_op_basis(
                            qp.Hadamard(target_wire) @ qp.adjoint(qp.S(target_wire)),
                            pg_op,
                            qp.S(target_wire) @ qp.Hadamard(target_wire),
                        )
                    )
                case "Z":
                    operations.append(pg_op)

        elif isinstance(op, qp.RZ) or (isinstance(op, qp.MultiRZ) and len(op.wires) == 1):
            wire = op.wires
            phi = op.parameters[0]
            global_phases.append(phi / 2)

            operations.append(
                _rz_phase_gradient(
                    phi,
                    wire,
                    angle_wires=angle_wires,
                    phase_grad_wires=phase_grad_wires,
                    work_wires=work_wires,
                )
            )

        elif isinstance(op, (qp.RX, qp.RY)):
            wire = op.wires
            phi = op.parameters[0]
            global_phases.append(phi / 2)

            diagonalizing_gate = (
                qp.Hadamard(wire)
                if isinstance(op, qp.RX)
                else qp.Hadamard(wire) @ qp.adjoint(qp.S(wire))
            )

            operations.append(
                qp.change_op_basis(
                    diagonalizing_gate,
                    _rz_phase_gradient(
                        phi,
                        wire,
                        angle_wires=angle_wires,
                        phase_grad_wires=phase_grad_wires,
                        work_wires=work_wires,
                    ),
                )
            )

        elif isinstance(op, qp.MultiRZ) and len(op.wires) > 1:
            wires = op.wires
            phi = op.parameters[0]
            global_phases.append(phi / 2)

            diagonalizing_gate = qp.prod(
                *[qp.CNOT(wires=(w0, w1)) for w0, w1 in zip(wires[~0:0:-1], wires[~1::-1])][::-1]
            )

            print("phi in rot_to_phase..:", phi)
            operations.append(
                qp.change_op_basis(
                    diagonalizing_gate,
                    _rz_phase_gradient(
                        phi,
                        wires[:1],
                        angle_wires=angle_wires,
                        phase_grad_wires=phase_grad_wires,
                        work_wires=work_wires,
                    ),
                )
            )

        else:
            operations.append(op)

    operations.append(qp.GlobalPhase(sum(global_phases)))
    new_tape = tape.copy(operations=operations)

    def null_postprocessing(results):
        """A postprocessing function returned by a transform that only converts the batch of results
        into a result for a single ``QuantumTape``.
        """
        return results[0]

    return [new_tape], null_postprocessing
