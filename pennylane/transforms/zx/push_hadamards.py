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
to a PennyLane phase-polynomial + Hadamard circuit.
"""

from pennylane.tape import QuantumScript, QuantumScriptBatch
from pennylane.transforms import transform
from pennylane.typing import PostprocessingFn

from .converter import from_zx, to_zx

try:
    import pyzx

    has_pyzx = True
except ModuleNotFoundError:
    has_pyzx = False


@transform
def push_hadamards(tape: QuantumScript) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    """
    Pushes Hadamard gates as far as possible to one side to cancel them and reduce the number of large phase-polynomial blocks,
    improving the effectiveness of phase-polynomial optimization techniques.

    This transform optimizes circuits composed of phase-polynomial blocks and Hadamard gates.
    This strategy works by commuting Hadamard gates through the circuit.
    To preserve the overall unitary, this process relies on commutation rules that can transform the gates a
    Hadamard moves past. For instance, pushing a Hadamard through a CNOT gate will convert the latter into a
    CZ gate. Consequently, the final optimized circuit may have, in some cases, a significantly different
    internal gate structure.

    The transform also applies some basic simplification rules to phase-polynomial blocks themselves to merge phase
    gates together when possible (e.g. T^4 = S^2 = Z).

    The implementation is based on the
    `pyzx.basic_optimization <https://pyzx.readthedocs.io/en/latest/api.html#pyzx.optimize.basic_optimization>`__ pass.

    Args:
        tape (QNode or QuantumScript or Callable): the input circuit to be transformed.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[QuantumScript], function]:
        the transformed circuit as described in :func:`qml.transform <pennylane.transform>`.

    Raises:
        ModuleNotFoundError: if the required ``pyzx`` package is not installed.
        TypeError: if the input quantum circuit is not a phase-polynomial + Hadamard circuit.

    **Example:**

    .. code-block:: python3

        import pennylane as qml
        import pennylane.transforms.zx as zx

        dev = qml.device("default.qubit")

        @zx.push_hadamards
        @qml.qnode(dev)
        def circuit():
            qml.T(0)
            qml.Hadamard(0)
            qml.Hadamard(0)
            qml.T(1)
            qml.Hadamard(1)
            qml.CNOT([1, 2])
            qml.Hadamard(1)
            qml.Hadamard(2)
            return qml.state()


    .. code-block:: pycon

        >>> print(qml.draw(circuit)())
        0: ──T────┤  State
        1: ──T─╭X─┤  State
        2: ──H─╰●─┤  State

    """

    if not has_pyzx:  # pragma: no cover
        raise ModuleNotFoundError(
            "The `pyzx` package is required. You can install it with `pip install pyzx`."
        )

    pyzx_graph = to_zx(tape)

    pyzx_circ = pyzx.Circuit.from_graph(pyzx_graph)

    try:
        pyzx_circ = pyzx.basic_optimization(pyzx_circ.to_basic_gates())

    except TypeError as e:

        raise TypeError(
            "The input quantum circuit must be a phase-polynomial + Hadamard circuit. "
            "RX and RY rotation gates are not supported."
        ) from e

    qscript = from_zx(pyzx_circ.to_graph())

    new_tape = tape.copy(operations=qscript.operations)

    def null_postprocessing(results):
        """A postprocesing function returned by a transform that only converts the batch of results
        into a result for a single ``QuantumScript``.
        """
        return results[0]

    return (new_tape,), null_postprocessing
