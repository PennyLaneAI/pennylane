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
This module contains a transform ``reduce_non_clifford`` to apply the
`full_reduce <https://pyzx.readthedocs.io/en/latest/api.html#pyzx.simplify.full_reduce>`__ simplification
pipeline (available through the external `pyzx <https://pyzx.readthedocs.io/en/latest/index.html>`__ package)
to a PennyLane circuit.
"""

from pennylane.tape import QuantumScript, QuantumScriptBatch
from pennylane.transforms import transform
from pennylane.typing import PostprocessingFn

from .converter import from_zx, to_zx
from .helper import _needs_pyzx


@_needs_pyzx
@transform
def reduce_non_clifford(tape: QuantumScript) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    """Reduce the number of non-Clifford gates by applying a combination of phase
    gadgetization strategies and Clifford gate simplification rules.

    This transform performs the following simplification/optimization steps, using
    `ZX calculus <https://pennylane.ai/compilation/zx-calculus-intermediate-representation>`__
    under the hood:

    - Apply the `full_reduce <https://pyzx.readthedocs.io/en/latest/api.html#pyzx.simplify.full_reduce>`__
      simplification pipeline to the ``pyzx`` graph representation (see :func:`~.to_zx`) of the given input circuit.

    - Use the `extract_circuit <https://pyzx.readthedocs.io/en/latest/api.html#pyzx.extract.extract_circuit>`__
      function to extract the equivalent sequence of gates and build a new optimized circuit.

    - Apply the `basic_optimization <https://pyzx.readthedocs.io/en/latest/api.html#pyzx.optimize.basic_optimization>`__ pass implemented in :func:`~.transforms.zx.push_hadamards`.
      to further optimize the
      `phase-polynomial <https://pennylane.ai/compilation/phase-polynomial-intermediate-representation>`__
      blocks in the circuit.

    This pipeline does not run the Third Order Duplicate and Destroy (TODD) algorithm
    implemented in :func:`~.transforms.zx.todd` and thus is not restricted to
    Clifford + T circuits.

    .. note::

        The transformed output circuit is equivalent to the input up to a global phase.

    Args:
        tape (QNode or QuantumScript or Callable): the input circuit to be transformed.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[QuantumScript], function]:
        the transformed circuit as described in :func:`qml.transform <pennylane.transform>`.

    Raises:
        ModuleNotFoundError: if the required ``pyzx`` package is not installed.

    **Example:**

    .. code-block:: python

        import pennylane.transforms.zx as zx

        dev = qml.device("default.qubit")

        @zx.reduce_non_clifford
        @qml.qnode(dev)
        def circuit(x, y):
            qml.T(0)
            qml.Hadamard(0)
            qml.Hadamard(0)
            qml.CNOT([0, 1])
            qml.T(0)
            qml.RX(x, 1)
            qml.RX(y, 1)
            return qml.state()

    >>> print(qml.draw(circuit)(3.2, -2.2))
    0: ──S─╭●─────────────────┤  State
    1: ────╰X──H──RZ(1.00)──H─┤  State


    .. note::

        This transform is designed to minimize non-Clifford phase gates (e.g. ``T``, ``RZ``),
        and is not as effective at reducing the number of two-qubit gates (e.g. ``CNOT``).
        For example, you might see a substantial increase in CNOT gates when optimizing a circuit composed primarily of Toffoli gates.
        Conversely, it tends to perform quite well on Trotterized chemistry circuits.

    For more details about ZX calculus-based simplification of quantum circuits, see the following papers:

    - Ross Duncan, Aleks Kissinger, Simon Perdrix, John van de Wetering (2019), "Graph-theoretic Simplification of Quantum Circuits with the ZX-calculus", `arXiv:1902.03178 <https://arxiv.org/abs/1902.03178>`__;

    - Aleks Kissinger, John van de Wetering (2020), "Reducing T-count with the ZX-calculus", `arXiv:1903.10477 <https://arxiv.org/abs/1903.10477>`__.

    """
    # pylint: disable=import-outside-toplevel
    import pyzx

    zx_graph = to_zx(tape)

    pyzx.hsimplify.from_hypergraph_form(zx_graph)
    pyzx.full_reduce(zx_graph)

    zx_circ = pyzx.extract_circuit(zx_graph)
    zx_circ = pyzx.basic_optimization(zx_circ.to_basic_gates())

    qscript = from_zx(zx_circ.to_graph())
    new_tape = tape.copy(operations=qscript.operations)

    def null_postprocessing(results):
        """A postprocesing function returned by a transform that only converts the batch of results
        into a result for a single ``QuantumScript``.
        """
        return results[0]

    return (new_tape,), null_postprocessing
