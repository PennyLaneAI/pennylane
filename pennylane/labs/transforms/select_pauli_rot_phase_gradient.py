import numpy as np

import pennylane as qml
from pennylane.operation import Operator
from pennylane.queuing import QueuingManager
from pennylane.tape import QuantumScript, QuantumScriptBatch
from pennylane.transforms import transform
from pennylane.typing import PostprocessingFn
from pennylane.wires import Wires


def _binary_repr_int(phi, precision):
    # Reasoning for +1e-10 term:
    # due to the division by pi, we obtain 14.999.. instead of 15 for, e.g., (1, 1, 1, 1) pi
    # at the same time, we want to floor off any additional floats when converting to the desired precision,
    # e.g. representing (1, 1, 1, 1) with only 3 digits we want to obtain (1, 1, 1)
    # so overall we floor but make sure we add a little term to not accidentally write 14 when the result is 14.999..
    return bin(int(np.floor(2**precision * phi / (4 * np.pi) + 1e-10)) + 2 * 2**precision)[
        -precision:
    ]


# @QueuingManager.stop_recording()
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

    ops = []
    ops.append(
        qml.QROM(
            binary_int, control_wires, angle_wires, work_wires=work_wires[len(angle_wires) - 1 :]
        )
    )

    for wire in phase_grad_wires:
        ops.append(qml.ctrl(qml.X(wire), control=target_wire, control_values=[0]))

    ops.append(qml.SemiAdder(angle_wires, phase_grad_wires, work_wires[: len(angle_wires) - 1]))

    for wire in phase_grad_wires:
        ops.append(qml.ctrl(qml.X(wire), control=target_wire, control_values=[0]))

    ops.append(
        qml.adjoint(qml.QROM)(
            binary_int, control_wires, angle_wires, work_wires=work_wires[len(angle_wires) - 1 :]
        )
    )

    return ops


@transform
def select_pauli_rot_phase_gradient(
    tape: QuantumScript, angle_wires: Wires, phase_grad_wires: Wires, work_wires: Wires
) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    r"""Quantum function transform to decompose all instances of :class:`~.SelectPauliRot` gates into additions
    using a phase gradient resource state.

    For this routine to work, the provided ``phase_grad_wires`` need to hold a phase gradient
    state :math:`|\nabla Z\rangle = \frac{1}{\sqrt{2^n}} \sum_{m=0}^{2^n-1} e^{2 \pi i \frac{m}{2^n}} |m\rangle`.
    Because this state is not modified and can be re-used at a later stage, the transform does not prepare it but
    rather assumes it has been prepared on those wires at an earlier stage.

    Note that this operator contains :class:`~.SemiAdder` that requires additional ``work_wires`` for the semi-in-place addition
    :math:`\text{SemiAdder}|x\rangle_\text{ang} |y\rangle_\text{phg} = |x\rangle_\text{ang} |x + y\rangle_\text{phg}`.


    Args:
        tape (QNode or QuantumTape or Callable): A quantum circuit containing :class:`~.RZ` gates.
        angle_wires (Wires): The qubits that conditionally load the angle :math:`\phi` of
            the :class:`~.SelectPauliRot` gate in binary as a multiple of :math:`2\pi`.
            The length of the ``angle_wires`` implicitly determines the precision
            with which the angle is represented.
            E.g., :math:`(2^{-1} + 2^{-2} + 2^{-3}) 2\pi` is exactly represented by three bits as ``111``.
        phase_grad_wires (Wires): The catalyst qubits with a phase gradient state prepared on them.
            Needs to be at least the length of ``angle_wires`` and will only
            use the first ``len(angle_wires)``.
        work_wires (Wires): Additional work wires to realize the :class:`~.SemiAdder` and :class:`~.QROM`.
            Needs to be at least ``b-1`` wires, where ``b=len(phase_grad_wires)`` is the precision of the angle :math:`\phi`.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[QuantumTape], function]: The transformed circuit as described in :func:`qml.transform <pennylane.transform>`.

    """

    if len(phase_grad_wires) < len(angle_wires):
        raise ValueError(
            f"phase_grad_wires needs to be at least as large as angle_wires. Got {len(phase_grad_wires)} phase_grad_wires, which is fewer than the {len(angle_wires)} angle wires."
        )

    operations = []
    for op in tape.operations:
        if isinstance(op, qml.SelectPauliRot):
            control_wires = op.wires[:-1]
            target_wire = op.wires[-1]
            rot_axis = op.hyperparameters["rot_axis"]

            angles = op.parameters[0]

            match rot_axis:
                case "X":
                    operations.append(
                        qml.change_op_basis(
                            qml.Hadamard(target_wire),
                            qml.ops.prod(_select_pauli_rot_phase_gradient)(
                                angles,
                                control_wires=control_wires,
                                target_wire=target_wire,
                                angle_wires=angle_wires,
                                phase_grad_wires=phase_grad_wires,
                                work_wires=work_wires,
                            ),
                            qml.Hadamard(target_wire),
                        )
                    )
                case "Y":
                    operations.append(
                        qml.change_op_basis(
                            qml.Hadamard(target_wire) @ qml.adjoint(qml.S(target_wire)),
                            qml.ops.prod(_select_pauli_rot_phase_gradient)(
                                angles,
                                control_wires=control_wires,
                                target_wire=target_wire,
                                angle_wires=angle_wires,
                                phase_grad_wires=phase_grad_wires,
                                work_wires=work_wires,
                            ),
                            qml.S(target_wire) @ qml.Hadamard(target_wire),
                        )
                    )
                case "Z":
                    operations.append(
                        qml.prod(_select_pauli_rot_phase_gradient)(
                            angles,
                            control_wires=control_wires,
                            target_wire=target_wire,
                            angle_wires=angle_wires,
                            phase_grad_wires=phase_grad_wires,
                            work_wires=work_wires,
                        )
                    ),

        else:
            operations.append(op)
    print(operations)
    new_tape = tape.copy(operations=operations)

    def null_postprocessing(results):
        """A postprocesing function returned by a transform that only converts the batch of results
        into a result for a single ``QuantumTape``.
        """
        return results[0]

    return [new_tape], null_postprocessing
