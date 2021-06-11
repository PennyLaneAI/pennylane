# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Transforms for optimizing quantum circuits."""

from pennylane import numpy as np

from pennylane.wires import Wires
from pennylane.transforms import qfunc_transform
from pennylane.operation import DiagonalOperation
from pennylane.ops.qubit import Rot

from pennylane.math import allclose, isclose

from .optimization_utils import fuse_rot


def _find_next_gate(wires, op_list):
    """Given a list of operations, finds the next operation that acts on at least one of
    the same set of wires, if present.

    Args:
        wires (Wires): A set of wires acted on by a quantum operation.
        op_list (list[Operation]): A list of operations that are implemented after the
            operation that acts on ``wires``.

    Returns:
        int or None: The index, in `op_list`, of the earliest gate that uses one or more
            of the same wires, or None if no such gate is present.
    """
    next_gate_idx = None

    for op_idx, op in enumerate(op_list):
        if len(Wires.shared_wires([wires, op.wires])) > 0:
            next_gate_idx = op_idx
            break

    return next_gate_idx


@qfunc_transform
def cancel_inverses(tape):
    """Quantum function transform to remove any operations that are applied next to their
    (self-)inverse.

    Args:
        tape (.QuantumTape): A quantum tape.

    **Example**

    Consider the following quantum function:

    .. code-block:: python

        def qfunc(x, y, z):
            qml.Hadamard(wires=0)
            qml.Hadamard(wires=1)
            qml.Hadamard(wires=0)
            qml.RX(x, wires=2)
            qml.RY(y, wires=1)
            qml.PauliX(wires=1)
            qml.RZ(z, wires=0)
            qml.RX(y, wires=2)
            qml.CNOT(wires=[0, 2])
            qml.PauliX(wires=1)
            return qml.expval(qml.PauliZ(0))

    The circuit before optimization:

    >>> dev = qml.device('default.qubit', wires=3)
    >>> qnode = qml.QNode(qfunc, dev)
    >>> print(qml.draw(qnode)(1, 2, 3))
    0: ──H──────H──────RZ(3)─────╭C──┤ ⟨Z⟩
    1: ──H──────RY(2)──X──────X──│───┤
    2: ──RX(1)──RX(2)────────────╰X──┤

    We can see that there are two adjacent Hadamards on the first qubit that
    should cancel each other out. Similarly, there are two Pauli-X gates on the
    second qubit that should cancel. We can obtain a simplified circuit by running
    the ``cancel_inverses`` transform:

    >>> optimized_qfunc = cancel_inverses(qfunc)
    >>> optimized_qnode = qml.QNode(optimized_qfunc, dev)
    >>> print(qml.draw(optimized_qnode)(1, 2, 3))
    0: ──RZ(3)─────────╭C──┤ ⟨Z⟩
    1: ──H──────RY(2)──│───┤
    2: ──RX(1)──RX(2)──╰X──┤
    """
    # Make a working copy of the list to traverse
    list_copy = tape.operations.copy()

    while len(list_copy) > 0:
        current_gate = list_copy[0]

        # Normally queue any gates that are not their own inverse
        if not current_gate.is_self_inverse:
            current_gate.queue()
            list_copy.pop(0)
            continue

        # If a gate does has a self-inverse, find the next gate that acts on the same wires
        next_gate_idx = _find_next_gate(current_gate.wires, list_copy[1:])

        # If no such gate is found queue the operation and move on
        if next_gate_idx is None:
            current_gate.queue()
            list_copy.pop(0)
            continue

        # Otherwise, get the next gate
        next_gate = list_copy[next_gate_idx + 1]

        # If next gate is the same (self inverse), we can potentially remove it
        # This implicitly ensures that the number of wires for the gates is also the same
        if current_gate.name == next_gate.name:
            # If the wires are the same, then we can safely remove
            if current_gate.wires == next_gate.wires:
                list_copy.pop(next_gate_idx + 1)
            # If wires are not equal, there are two things that can happen
            else:
                # There is not full overlap in the wires
                if len(Wires.shared_wires([current_gate.wires, next_gate.wires])) != len(
                    current_gate.wires
                ):
                    current_gate.queue()
                # There is full overlap, but the wires are in a different order
                else:
                    # If the wires are in a different order, only gates that are "symmetric"
                    # over the wires (e.g., CZ), can be cancelled.
                    if current_gate.is_symmetric_over_wires:
                        list_copy.pop(next_gate_idx + 1)
                    else:
                        current_gate.queue()
        # Otherwise, queue and move on to the next item
        else:
            current_gate.queue()

        # Remove this gate from the working list
        list_copy.pop(0)

    # Queue the measurements normally
    for m in tape.measurements:
        m.queue()


@qfunc_transform
def merge_rotations(tape):
    """Quantum function transform to combine rotation gates of the same type
    that act sequentially.

    If the combination of two rotation produces an angle that is close to 0,
    neither gate will be applied.

    Args:
        tape (.QuantumTape): A quantum tape.

    **Example**

    Consider the following quantum function.

    .. code-block:: python

        def qfunc(x, y, z):
            qml.RX(x, wires=0)
            qml.RX(y, wires=0)
            qml.CNOT(wires=[1, 2])
            qml.RY(y, wires=1)
            qml.Hadamard(wires=2)
            qml.CRZ(z, wires=[2, 0])
            qml.RY(-y, wires=1)
            return qml.expval(qml.PauliZ(0))

    The circuit before optimization:

    >>> dev = qml.device('default.qubit', wires=3)
    >>> qnode = qml.QNode(qfunc, dev)
    >>> print(qml.draw(qnode)(1, 2, 3))
    0: ───RX(1)──RX(2)──────────╭RZ(3)──┤ ⟨Z⟩
    1: ──╭C──────RY(2)──RY(-2)──│───────┤
    2: ──╰X──────H──────────────╰C──────┤

    By inspection, we can combine the two ``RX`` rotations on the first qubit.
    On the second qubit, we have a cumulative angle of 0, and the gates will cancel.

    >>> optimized_qfunc = merge_rotations(qfunc)
    >>> optimized_qnode = qml.QNode(optimized_qfunc, dev)
    >>> print(qml.draw(optimized_qnode)(1, 2, 3))
    0: ───RX(3)─────╭RZ(3)──┤ ⟨Z⟩
    1: ──╭C─────────│───────┤
    2: ──╰X──────H──╰C──────┤
    """
    # Make a working copy of the list to traverse
    list_copy = tape.operations.copy()

    while len(list_copy) > 0:
        current_gate = list_copy[0]

        # Normally queue any non-rotation gates
        if not current_gate.is_composable_rotation:
            current_gate.queue()
            list_copy.pop(0)
            continue

        # Otherwise, find the next gate that acts on the same wires
        next_gate_idx = _find_next_gate(current_gate.wires, list_copy[1:])

        # If no such gate is found (either there simply is none, or there are other gates
        # "in the way", queue the operation and move on
        if next_gate_idx is None:
            current_gate.queue()
            list_copy.pop(0)
            continue

        # Cumulative rotation angle(s)
        cumulative_angles = current_gate.parameters

        # As long as there is a valid next gate, check if we can merge the angles
        while next_gate_idx is not None:
            # Get the next gate
            next_gate = list_copy[next_gate_idx + 1]

            # If next gate is of the same type, we can merge the angles
            if current_gate.name == next_gate.name and current_gate.wires == next_gate.wires:
                list_copy.pop(next_gate_idx + 1)

                # The Rot gate must be treated separately
                if current_gate.name == "Rot":
                    cumulative_angles = fuse_rot(cumulative_angles, next_gate.parameters)
                # Other, single-parameter rotation gates just have the angle summed
                else:
                    cumulative_angles[0] = cumulative_angles[0] + next_gate.parameters[0]
            # If it is not, we need to stop
            else:
                break

            # If we did merge, look now at the next gate
            next_gate_idx = _find_next_gate(current_gate.wires, list_copy[1:])

        # If the cumulative angle is not close to 0, apply the cumulative gate
        if not allclose(np.array(cumulative_angles), np.zeros(len(cumulative_angles))):
            type(current_gate)(*cumulative_angles, wires=current_gate.wires).queue()

        # Remove the first gate gate from the working list
        list_copy.pop(0)

    # Queue the measurements normally
    for m in tape.measurements:
        m.queue()


@qfunc_transform
def single_qubit_fusion(tape):
    """Quantum function transform to fuse together groups of single-qubit
    operations into the general single-qubit unitary form.

    Args:
        tape (.QuantumTape): A quantum tape.

    **Example**

    Consider the following quantum function.

    .. code-block:: python

        def qfunc(r1, r2):
            qml.Hadamard(wires=0)
            qml.Rot(*r1, wires=0)
            qml.Rot(*r2, wires=0)
            qml.RZ(r1[0], wires=0)
            qml.RZ(r2[0], wires=0)
            return qml.expval(qml.PauliX(0))

    The circuit before optimization:

    >>> dev = qml.device('default.qubit', wires=1)
    >>> qnode = qml.QNode(qfunc, dev)
    >>> print(qml.draw(qnode)([0.1, 0.2, 0.3], [0.4, 0.5, 0.6]))
    0: ──H──Rot(0.1, 0.2, 0.3)──Rot(0.4, 0.5, 0.6)──RZ(0.1)──RZ(0.4)──┤ ⟨X⟩

    Full single-qubit gate fusion allows us to collapse this entire sequence into a
    single ``qml.Rot`` rotation gate.

    >>> optimized_qfunc = qml.compile(pipeline=[single_qubit_fusion], num_passes=3)(qfunc)
    >>> optimized_qnode = qml.QNode(optimized_qfunc, dev)
    >>> print(qml.draw(optimized_qnode)([0.1, 0.2, 0.3], [0.4, 0.5, 0.6]))
    0: ──Rot(3.57, 2.09, 2.05)──┤ ⟨X⟩

    """
    # Make a working copy of the list to traverse
    list_copy = tape.operations.copy()

    while len(list_copy) > 0:
        current_gate = list_copy[0]

        # Normally queue any multi-qubit gates
        if current_gate.num_wires > 1:
            current_gate.queue()
            list_copy.pop(0)
            continue

        # Find the next gate that acts on the same wires
        next_gate_idx = _find_next_gate(current_gate.wires, list_copy[1:])

        # If no such gate is found (either there simply is none, or there are other gates
        # "in the way", queue the operation and move on
        if next_gate_idx is None:
            current_gate.queue()
            list_copy.pop(0)
            continue

        # Set up a cumulative Rot starting from the angles of the initial gate
        cumulative_angles = current_gate.as_rot_angles()

        # Loop as long as a valid next gate exists
        while next_gate_idx is not None:
            # Get the next gate
            next_gate = list_copy[next_gate_idx + 1]

            # If next gate is on the same qubit, we can fuse them
            if current_gate.wires == next_gate.wires:
                list_copy.pop(next_gate_idx + 1)

                # Merge the angles
                next_gate_angles = next_gate.as_rot_angles()
                cumulative_angles = fuse_rot(cumulative_angles, next_gate_angles)
            else:
                break

            next_gate_idx = _find_next_gate(current_gate.wires, list_copy[1:])

        # If the cumulative angle is not close to 0, apply the gate
        if not allclose(cumulative_angles, np.zeros(3)):
            Rot(*cumulative_angles, wires=current_gate.wires).queue()

        # Remove the starting gate from the list
        list_copy.pop(0)

    # Queue the measurements normally
    for m in tape.measurements:
        m.queue()


@qfunc_transform
def diag_behind_controls(tape):
    """Quantum function transform to push controlled gates past diagonal gates that occur
    afterwards on the control qubit.

    Args:
        tape (.QuantumTape): A quantum tape.

    **Example**

    Consider the following quantum function.

    .. code-block:: python

        def qfunc_with_many_rots():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.T(wires=0)
            return qml.expval(qml.PauliX(0))

    In this circuit, the ``PauliZ`` and the ``CNOT`` commute; this means
    that the they can swap places, and then the ``PauliZ`` can be fused with the
    ``Hadamard`` gate if desired:

    >>> optimized_qfunc = qml.compile(pipeline=[diag_behind_controls, single_qubit_fusion])(qfunc)
    >>> optimized_qnode = qml.QNode(optimized_qfunc, dev)
    >>> print(qml.draw(optimized_qnode)())
    0: ──Rot(3.14, 1.57, 0.785)──╭C──┤ ⟨X⟩
    1: ──────────────────────────╰X──┤
    """
    # Make a working copy of the list to traverse
    list_copy = tape.operations.copy()

    while len(list_copy) > 0:
        current_gate = list_copy[0]

        # Consider only two-qubit gates
        if current_gate.num_wires != 2:
            current_gate.queue()
            list_copy.pop(0)
            continue

        # Find the next gate that uses the control wire
        next_gate_idx = _find_next_gate(Wires(current_gate.wires[0]), list_copy[1:])

        # If no such gate is found (either there simply is none, or there are other gates
        # "in the way", queue the operation and move on
        if next_gate_idx is None:
            current_gate.queue()
            list_copy.pop(0)
            continue

        # Loop as long as a valid next gate exists
        while next_gate_idx is not None:
            # Get the next gate
            next_gate = list_copy[next_gate_idx + 1]

            # If it is not a single-qubit gate, we can't push it through, so stop.
            if len(next_gate.wires) != 1:
                break

            # Valid next gates must be diagonal, and must share the *control* wire
            if next_gate.wires[0] == current_gate.wires[0]:
                if isinstance(next_gate, DiagonalOperation):
                    list_copy.pop(next_gate_idx + 1)
                    next_gate.queue()
                # It could also be that the gate is a Rot but still diagonal if
                # the angle of the RY gate in the middle is 0
                elif isinstance(next_gate, Rot):
                    if isclose(next_gate.parameters[1], 0.0):
                        list_copy.pop(next_gate_idx+1)
                        next_gate.queue()
                else:
                    break

            next_gate_idx = _find_next_gate(Wires(current_gate.wires[0]), list_copy[1:])

        # After we have found all possible diagonal gates to push through the control,
        # we must still apply the original gate
        current_gate.queue()
        list_copy.pop(0)

    # Queue the measurements normally
    for m in tape.measurements:
        m.queue()
