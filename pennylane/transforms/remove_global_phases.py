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
Provides a transform to remove all ``qml.GlobalPhase`` gates in a circuit.
"""

import pennylane as qml
from pennylane.measurements import StateMP
from pennylane.tape import QuantumScript, QuantumScriptBatch
from pennylane.transforms import transform
from pennylane.typing import PostprocessingFn


@transform
def remove_global_phases(tape: QuantumScript) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    """Remove all ``qml.GlobalPhase`` gates present in a quantum circuit.

    This transform returns a new circuit where all ``qml.GlobalPhase`` gates in the original circuit (if exists)
    are removed.

    Args:
        tape (QNode or QuantumScript or Callable): the input circuit to be transformed.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[QuantumScript], function]:
        the transformed circuit as described in :func:`qml.transform <pennylane.transform>`.

    **Example**

    Suppose we want to remove all of the global phase gates in a given quantum circuit.
    The ``remove_global_phases`` transform can be used to do this as follows:

    .. code-block:: python3

        dev = qml.device("default.qubit", wires=3)

        @qml.transforms.remove_global_phases
        @qml.qnode(dev)
        def circuit():
            qml.GlobalPhase(0.3, wires=0)
            qml.PauliY(wires=0)
            qml.Hadamard(wires=1)
            qml.GlobalPhase(0.4, wires=0)
            qml.CNOT(wires=(1,2))
            qml.GlobalPhase(0.5, wires=2)
            return qml.expval(qml.X(0) @ qml.Z(1))

    To check the result, let's print out the circuit:

    >>> print(qml.draw(circuit)())
    0: ──Y────┤ ╭<X@Z>
    1: ──H─╭●─┤ ╰<X@Z>
    2: ────╰X─┤
    """

    if any(isinstance(mp, StateMP) for mp in tape.measurements):
        raise qml.QuantumFunctionError(
            "The quantum circuit cannot contain a state measurement. Removing GlobalPhase operators in this case can cause errors."
        )

    operations = filter(lambda op: op.name != "GlobalPhase", tape.operations)
    new_tape = tape.copy(operations=operations)

    def null_postprocessing(results):
        """A postprocesing function returned by a transform that only converts the batch of results
        into a result for a single ``QuantumScript``.
        """
        return results[0]

    return (new_tape,), null_postprocessing
