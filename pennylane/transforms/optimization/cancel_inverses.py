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
"""Transform for cancelling adjacent inverse gates in quantum circuits."""

from pennylane import apply
from pennylane.wires import Wires
from pennylane.transforms import qfunc_transform

from .optimization_utils import _find_next_gate


@qfunc_transform
def cancel_inverses(tape):
    """Quantum function transform to remove any operations that are applied next to their
    (self-)inverse.

    Args:
        qfunc (function): A quantum function.

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
            apply(current_gate)
            list_copy.pop(0)
            continue

        # If a gate does has a self-inverse, find the next gate that acts on the same wires
        next_gate_idx = _find_next_gate(current_gate.wires, list_copy[1:])

        # If no such gate is found queue the operation and move on
        if next_gate_idx is None:
            apply(current_gate)
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
                    apply(current_gate)
                # There is full overlap, but the wires are in a different order
                else:
                    # If the wires are in a different order, only gates that are "symmetric"
                    # over the wires (e.g., CZ), can be cancelled.
                    if current_gate.is_symmetric_over_wires:
                        list_copy.pop(next_gate_idx + 1)
                    else:
                        apply(current_gate)
        # Otherwise, queue and move on to the next item
        else:
            apply(current_gate)

        # Remove this gate from the working list
        list_copy.pop(0)

    # Queue the measurements normally
    for m in tape.measurements:
        apply(m)
