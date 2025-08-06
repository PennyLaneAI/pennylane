# Copyright 2018-2025 Xanadu Quantum Technologies Inc.

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
TODO
"""

import pennylane as qml
from pennylane.tape import QuantumScript, QuantumScriptBatch
from pennylane.transforms import transform
from pennylane.typing import PostprocessingFn

from .converter import from_zx

has_pyzx = True
try:
    import pyzx
except ModuleNotFoundError:
    has_pyzx = False


@transform
def todd(tape: QuantumScript) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    """
    TODO

    Args:
        tape (QNode or QuantumScript or Callable): the input circuit to be transformed.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[QuantumScript], function]:
        the transformed circuit as described in :func:`qml.transform <pennylane.transform>`.

    Raises:
        ModuleNotFoundError: if the required ``pyzx`` package is not installed.

    **Example:**

    .. code-block:: python3

        import pennylane as qml
        import pennylane.transforms.zx as zx

        dev = qml.device("default.qubit")

        @zx.todd
        @qml.qnode(dev)
        def circuit():
            qml.T(0)
            qml.CNOT([0, 1])
            qml.S(0)
            qml.T(0)
            qml.T(1)
            qml.CNOT([0, 2])
            qml.T(1)
            return qml.state()

    ```pycon
    >>> print(qml.draw(circuit)())
    0: ──S†─╭Z─╭●─╭●─┤  State
    1: ──S──╰●─│──╰X─┤  State
    2: ────────╰X────┤  State
    ```
    """

    if not has_pyzx:  # pragma: no cover
        raise ModuleNotFoundError(
            "The `pyzx` package is required. You can install it by `pip install pyzx`."
        )

    qasm2_no_meas = qml.to_openqasm(tape, measure_all=False)

    pyzx_circ = pyzx.Circuit.from_qasm(qasm2_no_meas)
    pyzx_circ = pyzx.phase_block_optimize(pyzx_circ, pre_optimize=False)

    qscript = from_zx(pyzx_circ.to_graph())

    new_tape = tape.copy(operations=qscript.operations)

    def null_postprocessing(results):
        """A postprocesing function returned by a transform that only converts the batch of results
        into a result for a single ``QuantumScript``.
        """
        return results[0]

    return (new_tape,), null_postprocessing
