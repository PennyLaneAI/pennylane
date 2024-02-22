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
# pylint: disable=too-many-branches
from typing import Sequence, Callable

from pennylane.tape import QuantumTape
from pennylane.transforms import transform
from pennylane.math import allclose, stack, cast_like, zeros, is_abstract, get_interface
from pennylane.queuing import QueuingManager

from pennylane.ops.qubit.attributes import composable_rotations
from pennylane.ops.op_math import Adjoint
from .optimization_utils import find_next_gate, fuse_rot_angles


@transform
def merge_rotations(
    tape: QuantumTape, atol=1e-8, include_gates=None
) -> (Sequence[QuantumTape], Callable):
    r"""Quantum transform to combine rotation gates of the same type that act sequentially.

    If the combination of two rotation produces an angle that is close to 0,
    neither gate will be applied.

    Args:
        tape (QNode or QuantumTape or Callable): A quantum circuit.
        atol (float): After fusion of gates, if the fused angle :math:`\theta` is such that
            :math:`|\theta|\leq \text{atol}`, no rotation gate will be applied.
        include_gates (None or list[str]): A list of specific operations to merge. If
            set to ``None`` (default), all operations in the
            `~.pennylane.ops.qubit.attributes.composable_rotations` attribute will be merged. Otherwise,
            only the operations whose names match those in the list will undergo merging.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[QuantumTape], function]: The transformed circuit as described in :func:`qml.transform <pennylane.transform>`.

    **Example**

    >>> dev = qml.device('default.qubit', wires=3)

    You can apply the transform directly on :class:`QNode`

    .. code-block:: python

        @merge_rotations
        @qml.qnode(device=dev)
        def circuit(x, y, z):
            qml.RX(x, wires=0)
            qml.RX(y, wires=0)
            qml.CNOT(wires=[1, 2])
            qml.RY(y, wires=1)
            qml.Hadamard(wires=2)
            qml.CRZ(z, wires=[2, 0])
            qml.RY(-y, wires=1)
            return qml.expval(qml.Z(0))

    >>> circuit(0.1, 0.2, 0.3)
    0.9553364891256055

    .. details::
        :title: Usage Details

        You can also apply it on quantum function.

        .. code-block:: python

            def qfunc(x, y, z):
                qml.RX(x, wires=0)
                qml.RX(y, wires=0)
                qml.CNOT(wires=[1, 2])
                qml.RY(y, wires=1)
                qml.Hadamard(wires=2)
                qml.CRZ(z, wires=[2, 0])
                qml.RY(-y, wires=1)
                return qml.expval(qml.Z(0))

        The circuit before optimization:

        >>> qnode = qml.QNode(qfunc, dev)
        >>> print(qml.draw(qnode)(1, 2, 3))
        0: ──RX(1.00)──RX(2.00)─╭RZ(3.00)────────────┤  <Z>
        1: ─╭●─────────RY(2.00)─│──────────RY(-2.00)─┤
        2: ─╰X─────────H────────╰●───────────────────┤

        By inspection, we can combine the two ``RX`` rotations on the first qubit.
        On the second qubit, we have a cumulative angle of 0, and the gates will cancel.

        >>> optimized_qfunc = merge_rotations()(qfunc)
        >>> optimized_qnode = qml.QNode(optimized_qfunc, dev)
        >>> print(qml.draw(optimized_qnode)(1, 2, 3))
        0: ──RX(3.00)────╭RZ(3.00)─┤  <Z>
        1: ─╭●───────────│─────────┤
        2: ─╰X─────────H─╰●────────┤

        It is also possible to explicitly specify which rotations ``merge_rotations`` should
        be merged using the ``include_gates`` argument. For example, if in the above
        circuit we wanted only to merge the "RX" gates, we could do so as follows:

        >>> optimized_qfunc = merge_rotations(include_gates=["RX"])(qfunc)
        >>> optimized_qnode = qml.QNode(optimized_qfunc, dev)
        >>> print(qml.draw(optimized_qnode)(1, 2, 3))
        0: ──RX(3.00)───────────╭RZ(3.00)────────────┤  <Z>
        1: ─╭●─────────RY(2.00)─│──────────RY(-2.00)─┤
        2: ─╰X─────────H────────╰●───────────────────┤

    """
    # Expand away adjoint ops
    expanded_tape = tape.expand(stop_at=lambda obj: not isinstance(obj, Adjoint))
    list_copy = expanded_tape.operations
    new_operations = []
    while len(list_copy) > 0:
        current_gate = list_copy[0]

        # If a specific list of operations is specified, check and see if our
        # op is in it, then try to merge. If not, queue and move on.
        if include_gates is not None:
            if current_gate.name not in include_gates:
                new_operations.append(current_gate)
                list_copy.pop(0)
                continue

        # Check if the rotation is composable; if it is not, move on.
        if not current_gate in composable_rotations:
            new_operations.append(current_gate)
            list_copy.pop(0)
            continue

        # Find the next gate that acts on the same wires
        next_gate_idx = find_next_gate(current_gate.wires, list_copy[1:])

        # If no such gate is found (either there simply is none, or there are other gates
        # "in the way", queue the operation and move on
        if next_gate_idx is None:
            new_operations.append(current_gate)
            list_copy.pop(0)
            continue

        # We need to use stack to get this to work and be differentiable in all interfaces
        cumulative_angles = stack(current_gate.parameters)
        interface = get_interface(cumulative_angles)
        # As long as there is a valid next gate, check if we can merge the angles
        while next_gate_idx is not None:
            # Get the next gate
            next_gate = list_copy[next_gate_idx + 1]

            # If next gate is of the same type, we can merge the angles
            if current_gate.name == next_gate.name and current_gate.wires == next_gate.wires:
                list_copy.pop(next_gate_idx + 1)
                # The Rot gate must be treated separately
                if current_gate.name == "Rot":
                    if is_abstract(cumulative_angles):
                        # jax-jit does not support cast_like
                        cumulative_angles = cumulative_angles + stack(next_gate.parameters)
                    else:
                        cumulative_angles = fuse_rot_angles(
                            cumulative_angles,
                            cast_like(
                                stack(next_gate.parameters, like=interface), cumulative_angles
                            ),
                        )
                # Other, single-parameter rotation gates just have the angle summed
                else:
                    if is_abstract(cumulative_angles):
                        # jax-jit does not support cast_like
                        cumulative_angles = cumulative_angles + stack(next_gate.parameters)
                    else:
                        cumulative_angles = cumulative_angles + cast_like(
                            stack(next_gate.parameters, like=interface), cumulative_angles
                        )
            # If it is not, we need to stop
            else:
                break

            # If we did merge, look now at the next gate
            next_gate_idx = find_next_gate(current_gate.wires, list_copy[1:])

        # If we are tracing/jitting, don't perform any conditional checks and
        # apply the operation regardless of the angles. Otherwise, only apply if
        # the rotation angle is non-trivial.
        if is_abstract(cumulative_angles):
            with QueuingManager.stop_recording():
                new_operations.append(
                    current_gate.__class__(*cumulative_angles, wires=current_gate.wires)
                )
        else:
            if not allclose(cumulative_angles, zeros(len(cumulative_angles)), atol=atol, rtol=0):
                with QueuingManager.stop_recording():
                    new_operations.append(
                        current_gate.__class__(*cumulative_angles, wires=current_gate.wires)
                    )

        # Remove the first gate from the working list
        list_copy.pop(0)

    new_tape = type(tape)(new_operations, tape.measurements, shots=tape.shots)

    def null_postprocessing(results):
        """A postprocesing function returned by a transform that only converts the batch of results
        into a result for a single ``QuantumTape``.
        """
        return results[0]

    return [new_tape], null_postprocessing
