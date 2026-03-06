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
# TODO: make function neat, e.g. using dispatching
import numpy as np

import pennylane as qp
from pennylane.operation import Operator
from pennylane.queuing import QueuingManager
from pennylane.tape import QuantumScript, QuantumScriptBatch
from pennylane.transforms import transform
from pennylane.typing import PostprocessingFn
from pennylane.wires import Wires


def binary_repr_int(phi, precision):
    r"""
    Binary representation of ``phi`` to the closest precision

    The function is relying on ``np.round`` to do the heavy lifting to correctly handling the midpoint "round to even"

    Parameters:
        phi (float): number to be represented in binary
        precision (int): number of digits to keep

    **Example**

    We round the binary representation of :math:`(0.11011) 4 \pi`, which simply yields :math:`(0.11) 4 \pi` from rounding down.

    >>> from pennylane.labs.transforms.rot_to_phase_gradient import binary_repr_int
    >>> precision = 2
    >>> phi = (1 / 2 + 1 / 4 + 0 / 8 + 1 / 16 + 1 / 32) * 4 * np.pi
    >>> binary_repr_int(phi, precision)
    array([1, 1])

    When we pass the midpoint of the cut off decimals, we round up. In particular, for :math:`(0.1011) 4 \pi`, we round to :math:`(0.11) 4 \pi`:

    >>> phi = (1 / 2 + 0 / 4 + 1 / 8 + 1/16) * 4 * np.pi
    >>> binary_repr_int(phi, precision)
    array([1, 1])

    Note that we ignore the positive decimals. E.g., because :math:`(0.1111) 4 \pi` rounds to :math:`(1.0000) 4 \pi`, we obtain ``[0, 0, 0, 0]``:

    >>> phi = (1 / 2 + 1 / 4 + 1 / 8 + 1/16) * 4 * np.pi
    >>> binary_repr_int(phi, precision)
    array([0, 0])

    .. details::
        :title: Tie to even rule

        The most non-trivial case is when we are exactly at the midpoint, i.e. the truncated bits are :math:`100`.
        In this case, the so-called ties to even rule kicks in. This is automatically handled by numpy under the hood.
        For example, take :math:`(0.10100) 4 \pi = 0.625 \cdot 4 \pi`. We can either round down to :math:`(0.10) 4 \pi = 0.5 \cdot 4 \pi`, or round up to :math:`(0.11) 4 \pi = 0.75 \cdot 4 \pi`, but it is a tie because both numbers
        are equally close to :math:`0.625 \cdot 4 \pi`. In this case we use the so-called tie to even rule, which rounds to the closest even number, which in this case is up to :math:`(0.11) 4 \pi = 0.75 \cdot 4 \pi`.

        >>> phi = (1 / 2 + 0 / 4 + 1 / 8 + 0/16 + 1/32) * 4 * np.pi
        >>> binary_repr_int(phi, precision)
        array([1, 1])


    """
    phi = qp.math.mod(phi, 4 * np.pi)
    phi_round = qp.math.round(2**precision * phi / 4 / np.pi)
    return qp.math.int_to_binary(phi_round.astype(int), precision)


def fanout(wires):
    """Fanout operator"""
    if len(wires) == 1:
        return qp.I(wires)
    return qp.prod(*[qp.CNOT(wires) for wires in zip(wires[~0:0:-1], wires[~1::-1])][::-1])


@QueuingManager.stop_recording()
def _rz_phase_gradient(
    phi: float, wire: Wires, angle_wires: Wires, phase_grad_wires: Wires, work_wires: Wires
) -> Operator:
    """Function that transforms the RZ gate to the phase gradient circuit
    The precision is implicitly defined by the length of ``angle_wires``
    Note that the global phases are collected and added as one big global phase in the main function
    """
    # variation of pennylane.transforms.rz_phase_gradient._rz_phase_gradient
    # adapted to the slightly different binary_repr_int from above

    precision = len(angle_wires)
    # BasisEmbedding can handle integer inputs, no need to actually translate to binary
    binary_int = 2 ** np.arange(precision - 1, -1, -1) @ binary_repr_int(phi * 2, precision)

    compute_op = qp.ctrl(qp.BasisEmbedding(features=binary_int, wires=angle_wires), control=wire)
    target_op = qp.SemiAdder(angle_wires, phase_grad_wires, work_wires)

    return qp.change_op_basis(compute_op, target_op, compute_op)


# pylint: disable=too-many-arguments
def _select_pauli_rot_phase_gradient(
    phis: np.ndarray,
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
    binary_int = binary_repr_int(phis, precision)

    ops = [
        qp.QROM(
            binary_int, control_wires, angle_wires, work_wires=work_wires[: len(control_wires) - 1]
        )
    ] + [qp.ctrl(qp.X(wire), control=target_wire, control_values=[0]) for wire in phase_grad_wires]
    # The uncomputation does not need any adjoints because both QROM and C(X) are self-adjoint.
    adj_ops = ops[::-1]

    return qp.change_op_basis(
        qp.prod(*ops[::-1]),
        qp.SemiAdder(angle_wires, phase_grad_wires, work_wires=work_wires[: len(angle_wires) - 1]),
        qp.prod(*adj_ops[::-1]),
    )


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
    diagonalizing_gate = fanout(wires) @ qp.prod(*diagonalizing_gates[::-1])

    pg_op = _rz_phase_gradient(phi, wires[:1], **other_wires)
    new_op = qp.change_op_basis(diagonalizing_gate, pg_op)

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
                control_wires=control_wires,
                target_wire=target_wire,
                **kwargs,
            )

            match op.hyperparameters["rot_axis"]:
                case "X":
                    comp = uncomp = qp.Hadamard(target_wire)
                    operations.append(qp.change_op_basis(comp, pg_op, uncomp))
                case "Y":
                    comp = qp.Hadamard(target_wire) @ qp.adjoint(qp.S(target_wire))
                    uncomp = qp.S(target_wire) @ qp.Hadamard(target_wire)
                    operations.append(qp.change_op_basis(comp, pg_op, uncomp))
                case "Z":
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
                    operations.append(qp.change_op_basis(fanout(op.wires), pg_op))
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
