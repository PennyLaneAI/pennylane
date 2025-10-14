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
This module contains a transform ``optimize_t_count`` to apply the
`full_optimize <https://pyzx.readthedocs.io/en/latest/api.html#pyzx.optimize.full_optimize>`__
pass (available through the external `pyzx <https://pyzx.readthedocs.io/en/latest/index.html>`__ package)
to a PennyLane Clifford + T circuit.
"""

from pennylane.tape import QuantumScript, QuantumScriptBatch
from pennylane.transforms import transform
from pennylane.typing import PostprocessingFn

from .converter import from_zx, to_zx
from .helper import _needs_pyzx


@_needs_pyzx
@transform
def optimize_t_count(tape: QuantumScript) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    """Reduce the number of T gates in a `Clifford + T circuit <https://pennylane.ai/compilation/clifford-t-gate-set>`__
    by using basic commutation and cancellation rules, combined with a dedicated
    `phase-polynomial <https://pennylane.ai/compilation/phase-polynomial-intermediate-representation>`__
    optimization strategy based on the
    `Third Order Duplicate and Destroy (TODD) <https://arxiv.org/abs/1712.01557>`__ algorithm.

    This transform applies a sequence of passes for T-count optimization to the given Clifford + T circuit.
    First, some ZX-based commutation and cancellation rules are applied to simplify the circuit.
    Then, the circuit is cut into phase-polynomial blocks and the TODD algorithm is used to optimize each of these phase polynomials.
    For circuits with many qubits and T gates, this transform may exhibit long run-times.

    .. note::

        The transformed output circuit is equivalent to the input up to a global phase.

    The implementation is based on the
    `pyzx.full_optimize <https://pyzx.readthedocs.io/en/latest/api.html#pyzx.optimize.full_optimize>`__ pass, using
    `ZX calculus <https://pennylane.ai/compilation/zx-calculus-intermediate-representation>`__
    under the hood. It combines :func:`~.transforms.zx.push_hadamards` and
    :func:`~.transforms.zx.todd` into a holistic method.

    Args:
        tape (QNode or QuantumScript or Callable): the input circuit to be transformed.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[QuantumScript], function]:
        the transformed circuit as described in :func:`qml.transform <pennylane.transform>`.

    Raises:
        ModuleNotFoundError: if the required ``pyzx`` package is not installed.
        TypeError: if the input quantum circuit is not a Clifford + T circuit.


    **Example:**

    .. code-block:: python

        import pennylane.transforms.zx as zx

        dev = qml.device("default.qubit")

        @zx.optimize_t_count
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

    >>> print(qml.draw(circuit)())
    0: ──Z─╭●────╭●─┤  State
    1: ────╰X──S─│──┤  State
    2: ──────────╰X─┤  State

    """
    # pylint: disable=import-outside-toplevel
    import pyzx

    pyzx_graph = to_zx(tape)
    pyzx_circ = pyzx.Circuit.from_graph(pyzx_graph)

    try:
        pyzx_circ = pyzx.full_optimize(pyzx_circ)
    except TypeError as e:
        raise TypeError(
            "The input circuit must be a Clifford + T circuit. Consider using `qml.clifford_t_decomposition` first."
        ) from e

    qscript = from_zx(pyzx_circ.to_graph())
    new_tape = tape.copy(operations=qscript.operations)

    def null_postprocessing(results):
        """A postprocesing function returned by a transform that only converts the batch of results
        into a result for a single ``QuantumScript``.
        """
        return results[0]

    return (new_tape,), null_postprocessing
