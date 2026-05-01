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

from pennylane.tape import QuantumScript, QuantumScriptBatch
from pennylane.transforms import transform
from pennylane.typing import PostprocessingFn


def null_postprocessing(results):
    """A postprocessing function returned by a transform that only converts the batch of results
    into a result for a single ``QuantumTape``.
    """
    return results[0]


@transform
def undo_swaps(tape: QuantumScript) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    """Quantum function transform to remove SWAP gates by running from right to left through the
    circuit changing the position of the qubits accordingly.

    Args:
        tape (QNode or QuantumTape or Callable): A quantum circuit.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[QuantumTape], function]: The transformed circuit as described in :func:`qp.transform <pennylane.transform>`.

    **Example**

    You can apply the transform directly on a :class:`QNode`.

    .. code-block:: python

        import pennylane as qp

        dev = qp.device('default.qubit', wires=3)

        @qp.transforms.undo_swaps
        @qp.qnode(device=dev)
        def circuit():
            qp.Hadamard(wires=0)
            qp.X(1)
            qp.SWAP(wires=[0,1])
            qp.SWAP(wires=[0,2])
            qp.Y(0)
            return qp.expval(qp.Z(0))

    >>> print(qp.draw(circuit)())
    0: ──Y─┤  <Z>
    1: ──H─┤
    2: ──X─┤

    The SWAP gates are removed before execution.

    .. details::
        :title: Usage Details

        Consider the following quantum function:

        .. code-block:: python

            def qfunc():
                qp.Hadamard(wires=0)
                qp.X(1)
                qp.SWAP(wires=[0,1])
                qp.SWAP(wires=[0,2])
                qp.Y(0)
                return qp.expval(qp.Z(0))

        The circuit before optimization:

        >>> dev = qp.device('default.qubit', wires=3)
        >>> qnode = qp.QNode(qfunc, dev)
        >>> print(qp.draw(qnode)())
        0: ──H─╭SWAP─╭SWAP──Y─┤  <Z>
        1: ──X─╰SWAP─│────────┤
        2: ──────────╰SWAP────┤

        We can remove the SWAP gates by running the ``undo_swap`` transform, where the wires
        involved in the SWAP gates are interchanged:

        >>> optimized_qnode = undo_swaps(qnode)
        >>> print(qp.draw(optimized_qnode)())
        0: ──Y─┤ <Z>
        1: ──H─┤
        2: ──X─┤

        Gates are iterated through from right to left, where non-SWAP gates are ignored. The first
        gate is a ``Y`` gate, which is left to act on wire ``0``. Next, the right-most SWAP gate
        acting on wires ``(0, 2)`` is removed, and the wires are manually swapped; wire ``2`` now
        becomes wire ``0``, and vice versa. Next, the SWAP gate acting on wires ``(0, 1)`` is
        removed and the wires are interchanged. Altogether, this affects the wire labels as follows,
        where the operations to the left of both SWAP gates have their wire labels changed
        accordingly:

        - wire ``0`` :math:`\rightarrow` ``2`` :math:`\rightarrow` ``1``. This moves the ``H`` gate from wire ``0`` to wire ``1``.
        - wire ``2`` :math:`\rightarrow` ``0``.
        - wire ``1`` :math:`\rightarrow` ``2``. This moves the ``X`` gate from wire ``1`` to wire ``2``.

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

    new_tape = tape.copy(operations=gates)
    new_tape.trainable_params = tape.trainable_params

    return [new_tape], null_postprocessing
