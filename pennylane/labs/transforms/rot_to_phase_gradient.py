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
from pennylane.tape import QuantumScript, QuantumScriptBatch
from pennylane.transforms import transform
from pennylane.transforms.rz_phase_gradient import _rz_phase_gradient
from pennylane.typing import PostprocessingFn
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

    Note that the discretized circuit contains :class:`~.SemiAdder`, which typically uses additional ``work_wires`` for the semi-in-place addition
    :math:`\text{SemiAdder}|x\rangle_\text{ang} |y\rangle_\text{phg} = |x\rangle_\text{ang} |x + y\rangle_\text{phg}`.

    Supported gates include:
    * :class:`~RX`
    * :class:`~RY`
    * :class:`~RZ`
    * :class:`~PhaseShift`
    * :class:`~PauliRot`
    * :class:`~SelectPauliRot`
    * :class:`~IsingXX`
    * :class:`~IsingYY`
    * :class:`~IsingZZ`

    Args:
        tape (QNode or QuantumTape or Callable): A quantum circuit containing rotation gates. See above
            for a list of supported rotation gates.
        angle_wires (Wires): The qubits that conditionally load the angle :math:`\phi` of
            the rotation gate in binary as a multiple of :math:`4\pi`.
            The length of the ``angle_wires`` , denoted :math:`b`, implicitly determines the precision
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

        from pennylane.labs.transforms import rot_to_phase_gradient
        from functools import partial
        import numpy as np

        precision = 4
        wire = "targ"
        angle_wires = [f"ang_{i}" for i in range(precision)]
        phase_grad_wires = [f"phg_{i}" for i in range(precision)]
        work_wires = [f"work_{i}" for i in range(precision - 1)]

        def prepare_phase_gradient(wires):
            # Preparing the phase gradient state needs to happen separately of the transform
            n = len(wires)
            basis = np.eye(2**n)
            B = 2**n
            wave_function = np.exp(-1j * 2 * np.pi * np.arange(B) / B) / np.sqrt(B)
            return qp.StatePrep(wave_function, wires)

        @partial(
            rot_to_phase_gradient,
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
            "phase_grad_wires needs to be at least as large as angle_wires. Got "
            f"{len(phase_grad_wires)} phase_grad_wires, which is fewer than the "
            f"{len(angle_wires)} angle_wires."
        )

    if len(work_wires) < len(angle_wires) - 1:
        raise ValueError(
            "work_wires needs to be at least as large as angle_wires, minus 1. "
            f"Got {len(work_wires)} work_wires and {len(angle_wires)} angle_wires."
        )

    operations = []
    global_phases = []

    kwargs = {
        "angle_wires": angle_wires,
        "phase_grad_wires": phase_grad_wires,
        "work_wires": work_wires,
    }
    for op in tape.operations:
        if isinstance(op, qp.SelectPauliRot):
            control_wires = op.wires[:-1]
            target_wire = op.wires[-1:]

            pg_op = _select_pauli_rot_phase_gradient(
                op.data[0],
                op.hyperparameters["rot_axis"],
                control_wires=control_wires,
                target_wire=target_wire,
                **kwargs,
            )

            operations.append(pg_op)

        elif isinstance(op, (qp.RX, qp.RY, qp.RZ, qp.PhaseShift)) or (
            isinstance(op, qp.MultiRZ) and len(op.wires) == 1
        ):
            target_wire = op.wires[:1]
            phi = op.parameters[0]

            if not isinstance(op, qp.PhaseShift):
                global_phases.append(phi / 2)

            pg_op = _rz_phase_gradient(phi, target_wire, **kwargs)

            match type(op):
                case qp.RX:
                    comp = uncomp = qp.Hadamard(target_wire)
                    operations.append(qp.change_op_basis(comp, pg_op, uncomp))
                case qp.RY:
                    comp = qp.Hadamard(target_wire) @ qp.adjoint(qp.S(target_wire))
                    uncomp = qp.S(target_wire) @ qp.Hadamard(target_wire)
                    operations.append(qp.change_op_basis(comp, pg_op, uncomp))
                case qp.MultiRZ if len(op.wires) > 1:
                    operations.append(qp.change_op_basis(ladder(op.wires), pg_op))
                case _:
                    operations.append(pg_op)

        elif isinstance(op, (qp.PauliRot, qp.IsingXX, qp.IsingYY, qp.IsingZZ)):
            new_op, global_phase = _pauli_rot_phase_gradient(op, **kwargs)
            operations.append(new_op)
            global_phases.append(global_phase)

        elif isinstance(op, qp.IsingXY):
            raise TypeError("IsingXY currently not supported by rot_to_phase_gradient transform")

        else:
            operations.append(op)

    if len(global_phases) > 0:
        operations.append(qp.GlobalPhase(sum(global_phases)))
    new_tape = tape.copy(operations=operations)

    def null_postprocessing(results):
        """A postprocessing function returned by a transform that only converts the batch of results
        into a result for a single ``QuantumTape``.
        """
        return results[0]

    return (new_tape,), null_postprocessing
