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


from pennylane.tape import QuantumScript, QuantumScriptBatch
from pennylane.transforms import transform
from pennylane.typing import PostprocessingFn


@transform
def remove_barrier(tape: QuantumScript) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    """Quantum transform to remove Barrier gates.

    Args:
        tape (QNode or QuantumTape or Callable): A quantum circuit.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[.QuantumTape], function]: The transformed circuit as described in :func:`qp.transform <pennylane.transform>`.

    **Example**

    The transform can be applied on :class:`QNode` directly.

    .. code-block:: python

        @remove_barrier
        @qp.qnode(qp.device('default.qubit'))
        def circuit(x, y):
            qp.Hadamard(wires=0)
            qp.Hadamard(wires=1)
            qp.Barrier(wires=[0,1])
            qp.X(0)
            return qp.expval(qp.Z(0))

    The barrier is then removed before execution.

    .. details::
        :title: Usage Details

        Consider the following quantum function:

        .. code-block:: python

            def qfunc(x, y):
                qp.Hadamard(wires=0)
                qp.Hadamard(wires=1)
                qp.Barrier(wires=[0,1])
                qp.X(0)
                return qp.expval(qp.Z(0))

        The circuit before optimization:

        >>> dev = qp.device('default.qubit')
        >>> qnode = qp.QNode(qfunc, dev)
        >>> print(qp.draw(qnode)(1, 2))
        0: ──H─╭||──X─┤  <Z>
        1: ──H─╰||────┤


        We can remove the Barrier by running the ``remove_barrier`` transform:

        >>> optimized_qfunc = remove_barrier(qfunc)
        >>> optimized_qnode = qp.QNode(optimized_qfunc, dev)
        >>> print(qp.draw(optimized_qnode)(1, 2))
        0: ──H──X─┤  <Z>
        1: ──H────┤

    """
    operations = filter(lambda op: op.name != "Barrier", tape.operations)
    new_tape = tape.copy(operations=operations)

    def null_postprocessing(results):
        """A postprocessing function returned by a transform that only converts the batch of results
        into a result for a single ``QuantumTape``.
        """
        return results[0]  # pragma: no cover

    return [new_tape], null_postprocessing
