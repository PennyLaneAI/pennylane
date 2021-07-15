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
from pennylane import apply
from pennylane.transforms import qfunc_transform
from pennylane.ops.qubit import Rot
from pennylane.math import allclose, cast_like, stack, zeros

from .optimization_utils import find_next_gate, fuse_rot_angles


@qfunc_transform
def single_qubit_fusion(tape, tol=1e-8):
    """Quantum function transform to fuse together groups of single-qubit
    operations into the general single-qubit unitary form (~.Rot).

    Args:
        qfunc (function): A quantum function.
        tol (float): A tolerance for which to apply a rotation after fusion.
            If all the angles of rotation are smaller than this amount, no
            ``Rot`` gate will be applied.

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

    >>> optimized_qfunc = single_qubit_fusion()(qfunc)
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
            apply(current_gate)
            list_copy.pop(0)
            continue

        # Find the next gate that acts on the same wires
        next_gate_idx = find_next_gate(current_gate.wires, list_copy[1:])

        # If no such gate is found (either there simply is none, or there are other gates
        # "in the way", queue the operation and move on
        if next_gate_idx is None:
            apply(current_gate)
            list_copy.pop(0)
            continue

        # Set up a cumulative Rot starting from the angles of the initial gate
        cumulative_angles = stack(current_gate.as_rot_angles())

        # Loop as long as a valid next gate exists
        while next_gate_idx is not None:
            # Get the next gate
            next_gate = list_copy[next_gate_idx + 1]

            # If next gate is on the same qubit, we can fuse them
            if current_gate.wires == next_gate.wires:
                list_copy.pop(next_gate_idx + 1)

                # Merge the angles
                next_gate_angles = next_gate.as_rot_angles()
                cumulative_angles = fuse_rot_angles(
                    cumulative_angles, cast_like(stack(next_gate_angles), cumulative_angles)
                )
            else:
                break

            next_gate_idx = find_next_gate(current_gate.wires, list_copy[1:])

        # Only apply if the cumulative angle is not close to 0
        if not allclose(cumulative_angles, zeros(3), atol=tol):
            Rot(*cumulative_angles, wires=current_gate.wires)

        # Remove the starting gate from the list
        list_copy.pop(0)

    # Queue the measurements normally
    for m in tape.measurements:
        apply(m)
