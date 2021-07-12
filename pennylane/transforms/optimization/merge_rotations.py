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
"""Transform for merging adjacent rotations of the same type in a quantum circuit."""

from pennylane import apply
from pennylane.transforms import qfunc_transform
from pennylane.math import allclose, stack, cast_like, zeros

from .optimization_utils import _find_next_gate, _fuse_rot_angles


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
            apply(current_gate)
            list_copy.pop(0)
            continue

        # Otherwise, find the next gate that acts on the same wires
        next_gate_idx = _find_next_gate(current_gate.wires, list_copy[1:])

        # If no such gate is found (either there simply is none, or there are other gates
        # "in the way", queue the operation and move on
        if next_gate_idx is None:
            apply(current_gate)
            list_copy.pop(0)
            continue

        # We need to use stack to get this to work and be differentiable in all interfaces
        cumulative_angles = stack(current_gate.parameters)

        # As long as there is a valid next gate, check if we can merge the angles
        while next_gate_idx is not None:
            # Get the next gate
            next_gate = list_copy[next_gate_idx + 1]

            # If next gate is of the same type, we can merge the angles
            if current_gate.name == next_gate.name and current_gate.wires == next_gate.wires:
                list_copy.pop(next_gate_idx + 1)

                # The Rot gate must be treated separately
                if current_gate.name == "Rot":
                    cumulative_angles = _fuse_rot_angles(
                        cumulative_angles, cast_like(stack(next_gate.parameters), cumulative_angles)
                    )
                # Other, single-parameter rotation gates just have the angle summed
                else:
                    cumulative_angles = cumulative_angles + cast_like(
                        stack(next_gate.parameters), cumulative_angles
                    )
            # If it is not, we need to stop
            else:
                break

            # If we did merge, look now at the next gate
            next_gate_idx = _find_next_gate(current_gate.wires, list_copy[1:])

        if not allclose(cumulative_angles, zeros(len(cumulative_angles))):
            current_gate.__class__(*cumulative_angles, wires=current_gate.wires)

        # Remove the first gate gate from the working list
        list_copy.pop(0)

    # Queue the measurements normally
    for m in tape.measurements:
        apply(m)
