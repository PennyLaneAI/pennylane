# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Optimization pass ``full_reduce`` from pyzx using ZX calculus."""

import pennylane as qml
from pennylane.tape import QuantumScript, QuantumScriptBatch
from pennylane.transforms import transform
from pennylane.typing import PostprocessingFn

from .util import _tape2pyzx

has_zx = True
try:
    import pyzx as zx
    from pyzx.graph.base import BaseGraph

except ImportError:
    has_zx = False


def null_postprocessing(results):
    """A postprocesing function returned by a transform that only converts the batch of results
    into a result for a single ``QuantumTape``.
    """
    return results[0]


@transform
def full_reduce(tape: QuantumScript) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    r"""

    ZX-based T gate reduction on an arbitrary PennyLane circuit.

    This implements the full `pipeline for T gate optimizations in pyzx <https://pyzx.readthedocs.io/en/latest/simplify.html>`__.

    This pipeline performs, in that order

    * `full_reduce <https://pyzx.readthedocs.io/en/latest/api.html#pyzx.simplify.full_reduce>`__
    * `normalize <https://pyzx.readthedocs.io/en/latest/api.html#pyzx.graph.base.BaseGraph.normalize>`__
    * `extract_circuit <https://pyzx.readthedocs.io/en/latest/api.html#pyzx.extract.extract_circuit>`__

    In particular, this pipeline does not apply :func:`~todd` and thus is not restricted to `(Clifford + T) <https://pennylane.ai/compilation/clifford-t-gate-set>`__ circuits.
    The latter two functions are simply to retrieve the circuit from the ZX graph.

    Args:
        tape (QNode or QuantumTape or Callable): Input PennyLane circuit.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[QuantumTape], function]: T gate optimized PennyLane circuit. See :func:`qml.transform <pennylane.transform>` for the different output formats depending on the input type.

    .. seealso:: :func:`~full_optimize`

    **Example**

    Let us optimize a circuit with :class:`~T` as well as :class:`~RZ` gates.

    .. code-block:: python

        from pennylane.labs.zxopt import full_reduce

        circ = qml.tape.QuantumScript([
            qml.CNOT((0, 1)),
            qml.T(0),
            qml.CNOT((3, 2)),
            qml.T(1),
            qml.CNOT((1, 2)),
            qml.T(2),
            qml.RZ(0.5, 1),
            qml.CNOT((1, 2)),
            qml.T(1),
            qml.CNOT((3, 2)),
            qml.T(0),
            qml.CNOT((0, 1)),
        ], [])

        print(f"Circuit before:")
        print(qml.drawer.tape_text(circ, wire_order=range(4)))

        (new_circ,), _ = full_reduce(circ)
        print(f"Circuit after full_reduce:")
        print(qml.drawer.tape_text(new_circ, wire_order=range(4)))


    .. code-block::

        Circuit before:
        0: ─╭●──T──T───────────╭●─┤
        1: ─╰X──T─╭●──RZ─╭●──T─╰X─┤
        2: ─╭X────╰X──T──╰X─╭X────┤
        3: ─╰●──────────────╰●────┤

        Circuit after full_reduce:
        0: ──S─╭●──────────────╭●─┤
        1: ────╰X──RZ─╭●────╭●─╰X─┤
        2: ─╭●────────│─────│──╭●─┤
        3: ─╰X────────╰X──T─╰X─╰X─┤

    The original circuit has five :class:`~T` gates which are reduced to just one.
    """
    if not has_zx:  # pragma: no cover
        raise ImportError(
            "The package pyzx is required by full_reduce. "
            "You can install it with pip install pyzx"
        )  # pragma: no cover

    pyzx_circ = _tape2pyzx(tape)

    if not isinstance(pyzx_circ, BaseGraph):
        g = pyzx_circ.to_graph()
    else:
        g = pyzx_circ

    zx.hsimplify.from_hypergraph_form(g)

    # simplify the Graph in-place, and show the rewrite steps taken.
    zx.full_reduce(g)
    g.normalize()  # Makes the graph more suitable for displaying

    c_opt = zx.extract_circuit(g.copy())

    c_opt2 = c_opt.to_basic_gates()
    c_opt2 = zx.basic_optimization(c_opt2)

    pl_circ = qml.transforms.from_zx(c_opt2.to_graph())
    return [pl_circ], null_postprocessing
