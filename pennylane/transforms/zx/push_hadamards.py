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
This module contains a transform ``push_hadamards`` to apply the
`basic_optimization <https://pyzx.readthedocs.io/en/latest/api.html#pyzx.optimize.basic_optimization>`__
pass (available through the external `pyzx <https://pyzx.readthedocs.io/en/latest/index.html>`__ package)
to a PennyLane phase-polynomial + Hadamard circuit.
"""

from pennylane.tape import QuantumScript, QuantumScriptBatch
from pennylane.transforms import transform
from pennylane.typing import PostprocessingFn

from .converter import from_zx, to_zx
from .helper import _needs_pyzx


@_needs_pyzx
@transform
def push_hadamards(tape: QuantumScript) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    """
    Push Hadamard gates as far as possible to one side to cancel them and create fewer larger
    `phase-polynomial <https://pennylane.ai/compilation/phase-polynomial-intermediate-representation>`__
    blocks, improving the effectiveness of phase-polynomial optimization techniques.

    This transform optimizes circuits composed of phase-polynomial blocks and Hadamard gates.
    This strategy works by commuting Hadamard gates through the circuit.
    To preserve the overall unitary, this process relies on commutation rules that can transform the gates a
    Hadamard moves past. For instance, pushing a Hadamard through a CNOT gate will convert the latter into a
    CZ gate. Consequently, the final optimized circuit may have, in some cases, a significantly different
    internal gate structure.

    The transform also applies some basic simplification rules to phase-polynomial blocks themselves to merge phase
    gates together when possible (e.g. T^4 = S^2 = Z).

    The implementation is based on the
    `pyzx.basic_optimization <https://pyzx.readthedocs.io/en/latest/api.html#pyzx.optimize.basic_optimization>`__ pass, using
    `ZX calculus <https://pennylane.ai/compilation/zx-calculus-intermediate-representation>`__
    under the hood.
    It often is paired with :func:`~.transforms.zx.todd` into the combined optimization
    pass :func:`~.transforms.zx.optimize_t_count`.

    Args:
        tape (QNode or QuantumScript or Callable): the input circuit to be transformed.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[QuantumScript], function]:
        the transformed circuit as described in :func:`qml.transform <pennylane.transform>`.

    Raises:
        ModuleNotFoundError: if the required ``pyzx`` package is not installed.
        TypeError: if the input quantum circuit is not a phase-polynomial + Hadamard circuit.

    **Example:**

    .. code-block:: python

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

    >>> print(qml.draw(circuit)())
    0: ──T────┤  State
    1: ──T─╭X─┤  State
    2: ──H─╰●─┤  State

    """
    # pylint: disable=import-outside-toplevel
    import pyzx

    pyzx_graph = to_zx(tape)
    pyzx_circ = pyzx.Circuit.from_graph(pyzx_graph)

    try:
        pyzx_circ = pyzx.basic_optimization(pyzx_circ.to_basic_gates())

    except TypeError:

        raise TypeError(
            "The input quantum circuit must be a phase-polynomial + Hadamard circuit. "
            "RX and RY rotation gates are not supported."
        ) from None

    qscript = from_zx(pyzx_circ.to_graph())
    new_tape = tape.copy(operations=qscript.operations)

    def null_postprocessing(results):
        """A postprocesing function returned by a transform that only converts the batch of results
        into a result for a single ``QuantumScript``.
        """
        return results[0]

    return (new_tape,), null_postprocessing
