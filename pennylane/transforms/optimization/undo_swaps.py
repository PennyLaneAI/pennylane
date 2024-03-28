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

"""Transform that eliminates the swap operators by reordering the wires."""
# pylint: disable=too-many-branches
from typing import Sequence, Callable

from pennylane.transforms import transform

from pennylane.tape import QuantumTape


def null_postprocessing(results):
    """A postprocesing function returned by a transform that only converts the batch of results
    into a result for a single ``QuantumTape``.
    """
    return results[0]


@transform
def undo_swaps(tape: QuantumTape) -> (Sequence[QuantumTape], Callable):
    """Quantum function transform to remove SWAP gates by running from right
    to left through the circuit changing the position of the qubits accordingly.

    Args:
        tape (QNode or QuantumTape or Callable): A quantum circuit.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[QuantumTape], function]: The transformed circuit as described in :func:`qml.transform <pennylane.transform>`.

    **Example**

    >>> dev = qml.device('default.qubit', wires=3)

    You can apply the transform directly on a :class:`QNode`

    .. code-block:: python

        @undo_swaps
        @qml.qnode(device=dev)
        def circuit():
            qml.Hadamard(wires=0)
            qml.X(1)
            qml.SWAP(wires=[0,1])
            qml.SWAP(wires=[0,2])
            qml.Y(0)
            return qml.expval(qml.Z(0))

    The SWAP gates are removed before execution.

    .. details::
        :title: Usage Details

        Consider the following quantum function:

        .. code-block:: python

            def qfunc():
                qml.Hadamard(wires=0)
                qml.X(1)
                qml.SWAP(wires=[0,1])
                qml.SWAP(wires=[0,2])
                qml.Y(0)
                return qml.expval(qml.Z(0))

        The circuit before optimization:

        >>> dev = qml.device('default.qubit', wires=3)
        >>> qnode = qml.QNode(qfunc, dev)
        >>> print(qml.draw(qnode)())
            0: ──H──╭SWAP──╭SWAP──Y──┤ ⟨Z⟩
            1: ──X──╰SWAP──│─────────┤
            2: ────────────╰SWAP─────┤


        We can remove the SWAP gates by running the ``undo_swap`` transform:

        >>> optimized_qfunc = undo_swaps(qfunc)
        >>> optimized_qnode = qml.QNode(optimized_qfunc, dev)
        >>> print(qml.draw(optimized_qnode)())
            0: ──Y──┤ ⟨Z⟩
            1: ──H──┤
            2: ──X──┤

    """

    wire_map = {wire: wire for wire in tape.wires}
    gates = []

    for current_gate in reversed(tape.operations):
        if current_gate.name == "SWAP":
            swap_wires_0, swap_wires_1 = current_gate.wires
            wire_map[swap_wires_0], wire_map[swap_wires_1] = (
                wire_map[swap_wires_1],
                wire_map[swap_wires_0],
            )
        else:
            gates.append(current_gate.map_wires(wire_map))

    gates.reverse()

    new_tape = type(tape)(
        gates, tape.measurements, shots=tape.shots, trainable_params=tape.trainable_params
    )

    return [new_tape], null_postprocessing
