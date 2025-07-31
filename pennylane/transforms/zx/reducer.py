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
This module provides transforms to reduce a circuit applying simplification rules based on the ZX calculus.
"""

from pennylane.tape import QuantumScript, QuantumScriptBatch
from pennylane.transforms import transform
from pennylane.typing import PostprocessingFn

from .converter import from_zx, to_zx

has_pyzx = True
try:
    import pyzx
except ModuleNotFoundError:
    has_pyzx = False


@transform
def zx_full_reduce(tape: QuantumScript) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    """Reduce an arbitrary circuit applying the full ZX-based pipeline for T-gate optimization,
    available through the external `pyzx <https://pyzx.readthedocs.io/en/latest/index.html>`__ package.

    This transform performs a graph-theoretic circuit simplification based on ZX calculus rules.
    It works as follows:

        - convert the quantum circuit into the corresponding ``pyzx`` graph;

        - apply the `full_reduce <https://pyzx.readthedocs.io/en/latest/api.html#pyzx.simplify.full_reduce>`__
          optimization pass to the ``pyzx`` graph;

        - use `extract_circuit <https://pyzx.readthedocs.io/en/latest/api.html#pyzx.extract.extract_circuit>`__ and
          apply the `basic_optimization <https://pyzx.readthedocs.io/en/latest/api.html#pyzx.optimize.basic_optimization>`__ pass
          to the extracted ``pyzx`` circuit;

        - build a new simplified ``pyzx`` graph and convert it back to its quantum circuit representation.

    Args:
        tape (QNode or QuantumScript or Callable): the input circuit to be transformed.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[QuantumScript], function]:
        the transformed circuit as described in :func:`qml.transform <pennylane.transform>`.

    Raises:
        ModuleNotFoundError: if the required ``pyzx`` Python package is not installed.

    **Example:**

    Consider the following simple circuit:

    .. code-block:: python3

        import pennylane as qml
        from pennylane.transforms import zx_full_reduce

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(x, y):
            qml.T(wires=0)
            qml.Hadamard(wires=0)
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0,1])
            qml.T(wires=0)
            qml.RX(x, wires=1)
            qml.RX(y, wires=1)
            return qml.state()

    To apply this ZX-based optimization pass you can do:

    >>> new_circuit = zx_full_reduce(circuit)
    >>> print(qml.draw(new_circuit)(3.2, -2.2))
    0: ──S─╭●─────────────────┤  State
    1: ────╰X──H──RZ(1.00)──H─┤  State

    .. note::

        This transform is designed to minimize T-count, and is not as effective at reducing the
        number of two-qubit gates, such as CNOTs. However, its performance varies significantly
        depending on the type of circuit. For example, you might see a substantial increase in CNOT
        gates when optimizing a circuit composed primarily of Toffoli gates. Conversely, it tends
        to perform much better on Trotterized chemistry circuits.

    For more details about ZX calculus-based simplification of quantum circuits, see the following papers:

        - Ross Duncan, Aleks Kissinger, Simon Perdrix, John van de Wetering (2019), "Graph-theoretic Simplification of Quantum Circuits with the ZX-calculus", <https://arxiv.org/abs/1902.03178>;

        - Aleks Kissinger, John van de Wetering (2020), "Reducing T-count with the ZX-calculus", <https://arxiv.org/abs/1903.10477>.

    For the list of ZX calculus-based simplification rules implemented in ``pyzx``, see the
    `online documentation <https://pyzx.readthedocs.io/en/latest/api.html#list-of-simplifications>`__.
    """

    if not has_pyzx:  # pragma: no cover
        raise ModuleNotFoundError(
            "The `pyzx` package is required. You can install it by `pip install pyzx`."
        )

    zx_graph = to_zx(tape)

    pyzx.hsimplify.from_hypergraph_form(zx_graph)
    pyzx.simplify.full_reduce(zx_graph)
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
