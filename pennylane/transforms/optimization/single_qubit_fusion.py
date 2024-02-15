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
"""Transform for fusing sequences of single-qubit gates."""
# pylint: disable=too-many-branches
from typing import Sequence, Callable

from pennylane.tape import QuantumTape
from pennylane.transforms import transform
from pennylane.ops.qubit import Rot
from pennylane.math import allclose, stack, is_abstract
from pennylane.queuing import QueuingManager

from .optimization_utils import find_next_gate, fuse_rot_angles


@transform
def single_qubit_fusion(
    tape: QuantumTape, atol=1e-8, exclude_gates=None
) -> (Sequence[QuantumTape], Callable):
    r"""Quantum function transform to fuse together groups of single-qubit
    operations into a general single-qubit unitary operation (:class:`~.Rot`).

    Fusion is performed only between gates that implement the property
    ``single_qubit_rot_angles``. Any sequence of two or more single-qubit gates
    (on the same qubit) with that property defined will be fused into one ``Rot``.

    Args:
        tape (QNode or QuantumTape or Callable): A quantum circuit.
        atol (float): An absolute tolerance for which to apply a rotation after
            fusion. After fusion of gates, if the fused angles :math:`\theta` are such that
            :math:`|\theta|\leq \text{atol}`, no rotation gate will be applied.
        exclude_gates (None or list[str]): A list of gates that should be excluded
            from full fusion. If set to ``None``, all single-qubit gates that can
            be fused will be fused.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[QuantumTape], function]: The transformed circuit as described in :func:`qml.transform <pennylane.transform>`.

    **Example**

    >>> dev = qml.device('default.qubit', wires=1)

    You can apply the transform directly on :class:`QNode`:

    .. code-block:: python

        @single_qubit_fusion
        @qml.qnode(device=dev)
        def qfunc(r1, r2):
            qml.Hadamard(wires=0)
            qml.Rot(*r1, wires=0)
            qml.Rot(*r2, wires=0)
            qml.RZ(r1[0], wires=0)
            qml.RZ(r2[0], wires=0)
            return qml.expval(qml.X(0))

    The single qubit gates are fused before execution.

    .. details::
        :title: Usage Details

        Consider the following quantum function.

        .. code-block:: python

            def qfunc(r1, r2):
                qml.Hadamard(wires=0)
                qml.Rot(*r1, wires=0)
                qml.Rot(*r2, wires=0)
                qml.RZ(r1[0], wires=0)
                qml.RZ(r2[0], wires=0)
                return qml.expval(qml.X(0))

        The circuit before optimization:

        >>> qnode = qml.QNode(qfunc, dev)
        >>> print(qml.draw(qnode)([0.1, 0.2, 0.3], [0.4, 0.5, 0.6]))
        0: ──H──Rot(0.1, 0.2, 0.3)──Rot(0.4, 0.5, 0.6)──RZ(0.1)──RZ(0.4)──┤ ⟨X⟩

        Full single-qubit gate fusion allows us to collapse this entire sequence into a
        single ``qml.Rot`` rotation gate.

        >>> optimized_qfunc = single_qubit_fusion(qfunc)
        >>> optimized_qnode = qml.QNode(optimized_qfunc, dev)
        >>> print(qml.draw(optimized_qnode)([0.1, 0.2, 0.3], [0.4, 0.5, 0.6]))
        0: ──Rot(3.57, 2.09, 2.05)──┤ ⟨X⟩

    """
    # Make a working copy of the list to traverse
    list_copy = tape.operations.copy()
    new_operations = []
    while len(list_copy) > 0:
        current_gate = list_copy[0]

        # If the gate should be excluded, queue it and move on regardless
        # of fusion potential
        if exclude_gates is not None:
            if current_gate.name in exclude_gates:
                new_operations.append(current_gate)
                list_copy.pop(0)
                continue

        # Look for single_qubit_rot_angles; if not available, queue and move on.
        # If available, grab the angles and try to fuse.
        try:
            cumulative_angles = stack(current_gate.single_qubit_rot_angles())
        except (NotImplementedError, AttributeError):
            new_operations.append(current_gate)
            list_copy.pop(0)
            continue

        # Find the next gate that acts on the same wires
        next_gate_idx = find_next_gate(current_gate.wires, list_copy[1:])

        if next_gate_idx is None:
            new_operations.append(current_gate)
            list_copy.pop(0)
            continue

        # Before entering the loop, we check to make sure the next gate is not in the
        # exclusion list. If it is, we should apply the original gate as-is, and not the
        # Rot version (example in test test_single_qubit_fusion_exclude_gates).
        if exclude_gates is not None:
            next_gate = list_copy[next_gate_idx + 1]
            if next_gate.name in exclude_gates:
                new_operations.append(current_gate)
                list_copy.pop(0)
                continue

        # Loop as long as a valid next gate exists
        while next_gate_idx is not None:
            next_gate = list_copy[next_gate_idx + 1]

            # Check first if the next gate is in the exclusion list
            if exclude_gates is not None:
                if next_gate.name in exclude_gates:
                    break

            # Try to extract the angles; since the Rot angles are implemented
            # solely for single-qubit gates, and we used find_next_gate to obtain
            # the gate in question, only valid single-qubit gates on the same
            # wire as the current gate will be fused.
            try:
                next_gate_angles = stack(next_gate.single_qubit_rot_angles())
            except (NotImplementedError, AttributeError):
                break

            cumulative_angles = fuse_rot_angles(cumulative_angles, stack(next_gate_angles))

            list_copy.pop(next_gate_idx + 1)
            next_gate_idx = find_next_gate(current_gate.wires, list_copy[1:])

        # If we are tracing/jitting, don't perform any conditional checks and
        # apply the rotation regardless of the angles.
        if is_abstract(cumulative_angles):
            with QueuingManager.stop_recording():
                new_operations.append(Rot(*cumulative_angles, wires=current_gate.wires))
        # If not tracing, check whether all angles are 0 (or equivalently, if the RY
        # angle is close to 0, and so is the sum of the RZ angles
        else:
            if not allclose(
                stack([cumulative_angles[0] + cumulative_angles[2], cumulative_angles[1]]),
                [0.0, 0.0],
                atol=atol,
                rtol=0,
            ):
                with QueuingManager.stop_recording():
                    new_operations.append(Rot(*cumulative_angles, wires=current_gate.wires))

        # Remove the starting gate from the list
        list_copy.pop(0)

    new_tape = type(tape)(new_operations, tape.measurements, shots=tape.shots)

    def null_postprocessing(results):
        """A postprocesing function returned by a transform that only converts the batch of results
        into a result for a single ``QuantumTape``.
        """
        return results[0]

    return [new_tape], null_postprocessing
