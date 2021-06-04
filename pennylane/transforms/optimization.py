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

from pennylane.wires import Wires
from pennylane.transforms import qfunc_transform

# This is a set of gates that are their own inverse; rather than keeping a list,
# this is possibly something that could be kept as a property of the Operations
self_inverse_symmetric = ["Hadamard", "CNOT", "CX", "CY", "CZ", "PauliX", "PauliY", "PauliZ"]
self_inverse_asymmetric = ["Hadamard", "CZ", "PauliX", "PauliY", "PauliZ"]

# Similarly, would be nice to mark this property within the operation
mergeable_rotations = ["RX", "RY", "RZ", "CRX", "CRY", "CRZ"]


def _find_next_gate(wires, op_list):
    """Given a list of operations, finds the next operation that acts on the same
    set of wires, in the same order, if one is present.

    Args:
        wires (Wires): A set of wires acted on by a quantum operation
        op_list (list[Operation]): A list of operations that are implemented after the
            operation that acts on wires.

    Returns:
        int or None: The index, in `op_list`, of the earliest gate that uses the same
            set of wires, or None if no such gate is present (e.g., the target wires are used
            in other gates that "block" potential gate adjacency)
    """
    next_gate_idx = None

    for op_idx in range(len(op_list)):
        # If there are no unique wires, then the wires are the same and we're done
        if len(Wires.unique_wires([wires, op_list[op_idx].wires])) == 0:
            next_gate_idx = op_idx
            break

        # If the sets of wires are not identical, check if any wires are
        # shared; if they are not (i.e., the next operation is on disjoint qubits),
        # then we can still keep looking
        elif len(Wires.shared_wires([wires, op_list[op_idx].wires])) == 0:
            op_idx += 1
            continue

        # If there are some shared wires, this gate is separated from where its
        # inverse might be, so we must stop
        else:
            break

    return next_gate_idx


@qfunc_transform
def cancel_inverses(tape):
    """Quantum function transform to remove any operations that are applied next to their
    (self) inverse.

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
    should cancel each other out. Similarly, there are two Pauli X gates on the
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
        if current_gate.name not in self_inverse_symmetric:
            current_gate.queue()
            list_copy.pop(0)
            continue

        # If a gate does has a self-inverse, find the next gate that acts on the same wires
        next_gate_idx = _find_next_gate(current_gate.wires, list_copy[1:])

        # If no such gate is found (either there simply is none, or there are other gates
        # "in the way", queue the operation and move on
        if next_gate_idx is None:
            current_gate.queue()
            list_copy.pop(0)
            continue

        # Otherwise, get the next gate
        next_gate = list_copy[next_gate_idx + 1]

        # If next gate is the same (self inverse), we can potentially remove it
        if current_gate.name == next_gate.name:
            # If the wires are the same, then we can safely remove
            if current_gate.wires == next_gate.wires:
                list_copy.pop(next_gate_idx + 1)
            # If wires are not equal, need to check if the inverse is asymmetric;
            # if it is not, then we can't cancel and have to queue it
            else:
                if current_gate.name in self_inverse_asymmetric:
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

        def (x, y, z):
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
        if current_gate.name not in mergeable_rotations:
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

        # Otherwise, get the next gate
        next_gate = list_copy[next_gate_idx + 1]

        # If next gate is the same, we can potentially remove it
        if current_gate.name == next_gate.name and current_gate.wires == next_gate.wires:
            list_copy.pop(next_gate_idx + 1)

            combined_angle = current_gate.parameters[0] + next_gate.parameters[0]

            # If the cumulative angle is close to 0, don't apply anything
            if np.isclose(combined_angle, 0):
                continue
            # Otherwise, apply the gate; we need qml.apply for this I think
            # qml.apply(current_gate_type, angle, wires)
            else:
                continue
        else:
            current_gate.queue()

        # Remove this gate from the working list
        list_copy.pop(0)

    # Queue the measurements normally
    for m in tape.measurements:
        m.queue()
