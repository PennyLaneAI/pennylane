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
This module contains a transform to apply the
`basic_optimization <https://pyzx.readthedocs.io/en/latest/api.html#pyzx.optimize.basic_optimization>`__
pass (available through the external `pyzx <https://pyzx.readthedocs.io/en/latest/index.html>`__ package)
to a PennyLane phase-polynomial circuit with Hadamard gates.
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
def push_hadamards(tape: QuantumScript) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    """
    Apply the `pyzx.basic_optimization <https://pyzx.readthedocs.io/en/latest/api.html#pyzx.optimize.basic_optimization>`__
    pass to a PennyLane phase-polynomial circuit with :class:`~Hadamard` gates.

    The transform optimizes the circuit using a strategy that involves delayed placement of gates so that more
    matches for gate cancellations are found.
    Specifically, it pushes Hadamard gates as far as possible to the side to create few big phase-polynomial
    blocks and improve the effectiveness of phase-polynomial optimization techniques.

    Args:
        tape (QNode or QuantumScript or Callable): the input circuit to be transformed.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[QuantumScript], function]:
        the transformed circuit as described in :func:`qml.transform <pennylane.transform>`.

    Raises:
        ModuleNotFoundError: if the required ``pyzx`` package is not installed.
    """

    if not has_pyzx:  # pragma: no cover
        raise ModuleNotFoundError(
            "The `pyzx` package is required. You can install it by `pip install pyzx`."
        )

    qasm2_no_meas = qml.to_openqasm(tape, measure_all=False)

    pyzx_circ = pyzx.Circuit.from_qasm(qasm2_no_meas)
    pyzx_circ = pyzx.basic_optimization(pyzx_circ.to_basic_gates())

    qscript = from_zx(pyzx_circ.to_graph())

    new_tape = tape.copy(operations=qscript.operations)

    def null_postprocessing(results):
        """A postprocesing function returned by a transform that only converts the batch of results
        into a result for a single ``QuantumScript``.
        """
        return results[0]

    return (new_tape,), null_postprocessing
