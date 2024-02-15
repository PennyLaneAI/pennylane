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
"""Transform for removing the Barrier gate from quantum circuits."""
# pylint: disable=too-many-branches
from typing import Sequence, Callable

from pennylane.tape import QuantumTape
from pennylane.transforms import transform


@transform
def remove_barrier(tape: QuantumTape) -> (Sequence[QuantumTape], Callable):
    """Quantum transform to remove Barrier gates.

    Args:
        tape (QNode or QuantumTape or Callable): A quantum circuit.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[.QuantumTape], function]: The transformed circuit as described in :func:`qml.transform <pennylane.transform>`.

    **Example**

    The transform can be applied on :class:`QNode` directly.

    .. code-block:: python

        @remove_barrier
        @qml.qnode(device=dev)
        def circuit(x, y):
            qml.Hadamard(wires=0)
            qml.Hadamard(wires=1)
            qml.Barrier(wires=[0,1])
            qml.X(0)
            return qml.expval(qml.Z(0))

    The barrier is then removed before execution.

    .. details::
        :title: Usage Details

        Consider the following quantum function:

        .. code-block:: python

            def qfunc(x, y):
                qml.Hadamard(wires=0)
                qml.Hadamard(wires=1)
                qml.Barrier(wires=[0,1])
                qml.X(0)
                return qml.expval(qml.Z(0))

        The circuit before optimization:

        >>> dev = qml.device('default.qubit', wires=2)
        >>> qnode = qml.QNode(qfunc, dev)
        >>> print(qml.draw(qnode)(1, 2))
            0: ──H──╭||──X──┤ ⟨Z⟩
            1: ──H──╰||─────┤


        We can remove the Barrier by running the ``remove_barrier`` transform:

        >>> optimized_qfunc = remove_barrier(qfunc)
        >>> optimized_qnode = qml.QNode(optimized_qfunc, dev)
        >>> print(qml.draw(optimized_qnode)(1, 2))
           0: ──H──X──┤ ⟨Z⟩
           1: ──H─────┤

    """
    operations = filter(lambda op: op.name != "Barrier", tape.operations)
    new_tape = type(tape)(operations, tape.measurements, shots=tape.shots)

    def null_postprocessing(results):
        """A postprocesing function returned by a transform that only converts the batch of results
        into a result for a single ``QuantumTape``.
        """
        return results[0]  # pragma: no cover

    return [new_tape], null_postprocessing
