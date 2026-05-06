# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Provides a transform to combine all ``qp.GlobalPhase`` gates in a circuit into a single one applied at the end.
"""

from functools import partial

import pennylane as qp
from pennylane.tape import QuantumScript, QuantumScriptBatch
from pennylane.transforms import transform
from pennylane.typing import PostprocessingFn


def _combine_global_phases_setup_inputs():
    return (), {}


@partial(
    transform, pass_name="combine-global-phases", setup_inputs=_combine_global_phases_setup_inputs
)
def combine_global_phases(tape: QuantumScript) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    """Combine all ``qp.GlobalPhase`` gates into a single ``qp.GlobalPhase`` operation.

    This transform returns a new circuit where all ``qp.GlobalPhase`` gates in the original circuit (if exists)
    are removed, and a new ``qp.GlobalPhase`` is added at the end of the list of operations with its phase
    being a total global phase computed as the algebraic sum of all global phases in the original circuit.

    Args:
        tape (QNode or QuantumScript or Callable): the input circuit to be transformed.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[QuantumScript], function]:
        the transformed circuit as described in :func:`qp.transform <pennylane.transform>`.

    **Example**

    Suppose we want to combine all the global phase gates in a given quantum circuit.
    The ``combine_global_phases`` transform can be used to do this as follows:

    .. code-block:: python

        dev = qp.device("default.qubit", wires=3)

        @qp.transforms.combine_global_phases
        @qp.qnode(dev)
        def circuit():
            qp.GlobalPhase(0.3, wires=0)
            qp.PauliY(wires=0)
            qp.Hadamard(wires=1)
            qp.CNOT(wires=(1,2))
            qp.GlobalPhase(0.46, wires=2)
            return qp.expval(qp.X(0) @ qp.Z(1))

    To check the result, let's print out the circuit:

    >>> print(qp.draw(circuit)())
    0: ──Y────╭GlobalPhase(0.76)─┤ ╭<X@Z>
    1: ──H─╭●─├GlobalPhase(0.76)─┤ ╰<X@Z>
    2: ────╰X─╰GlobalPhase(0.76)─┤

    .. details::
        :title: Usage with qjit

        When used with ``qjit``, the ``combine_global_phases`` compilation pass will merge
        operations surrounding control flow together, while those within the control flow are merged
        together separately (i.e., no formal loop-boundary optimizations).

        Consider the following example:

        .. code-block:: python

            import pennylane as qp

            n = 3
            dev = qp.device('null.qubit', wires=n)

            @qp.qjit(keep_intermediate=True, capture=True)
            @qp.transforms.combine_global_phases
            @qp.qnode(dev)
            def circuit():
                qp.GlobalPhase(0.1, wires = 2)
                qp.X(n-1)
                qp.GlobalPhase(0.1, wires = 1)
                qp.H(n-2)

                @qp.for_loop(0, 2)
                def loop(i):
                    qp.GlobalPhase(0.1967, wires=i)
                    qp.GlobalPhase(0.7691, wires=i)

                loop()

                qp.GlobalPhase(0.1, wires=0)
                qp.GlobalPhase(0.1, wires=0)

                return qp.expval(qp.Z(0))

        The two ``GlobalPhase`` operations within the ``for_loop`` context will be merged together.
        However, they will not be merged together with the ``GlobalPhase`` operations that occur
        before and after the ``for_loop``.

        This behaviour is shown in the image below, where the application of
        ``combine_global_phases`` results in two ``GlobalPhase`` instances (one inside of a
        ``for_loop`` and the other from the ``GlobalPhase`` instances outside of the ``for_loop``).

        >>> print(qp.draw_graph(circuit)()) # doctest: +SKIP

        .. figure:: ../../_static/catalyst-combine-global-phases-example.png
            :align: left

    """

    has_global_phase = False
    phi = 0
    operations = []
    for op in tape.operations:
        if isinstance(op, qp.GlobalPhase):
            has_global_phase = True
            phi += op.parameters[0]
        else:
            operations.append(op)

    if has_global_phase:
        with qp.QueuingManager.stop_recording():
            operations.append(qp.GlobalPhase(phi=phi))

    new_tape = tape.copy(operations=operations)

    def null_postprocessing(results):
        """A postprocessing function returned by a transform that only converts the batch of results
        into a result for a single ``QuantumScript``.
        """
        return results[0]

    return (new_tape,), null_postprocessing
