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
"""Optimization pass ``full_optimize`` from pyzx using ZX calculus."""
import warnings

import pyzx as zx

import pennylane as qml

from .zx_conversion import _tape2pyzx


def full_optimize(tape):
    r"""

    Full optimization pipeline applying TODD and ZX-based T gate reduction to a PennyLane `(Clifford + T) <https://pennylane.ai/compilation/clifford-t-gate-set>`__ circuit.

    This function applies `zx.full_optimize <https://pyzx.readthedocs.io/en/latest/api.html#pyzx.optimize.full_optimize>`__ and is basically a combination of :func:`~todd` and :func:`~full_reduce`.

    When there are continuous rotation gates such as :class:`~RZ`, we suggest to use :func:`~full_reduce`.

    Args:
        tape (qml.tape.QuantumScript): Input PennyLane circuit. This circuit has to be in the `(Clifford + T) <https://pennylane.ai/compilation/clifford-t-gate-set>`__ gate set.

    Returns:
        qml.tape.QuantumScript: T gate optimized PennyLane circuit.

    .. seealso:: :func:`~full_reduce`, :func:`~todd`

    **Example**

    Let us optimize a circuit with :class:`~T` gates.

    .. code-block:: python

        from pennylane.labs.zxopt import full_optimize

        circ = qml.tape.QuantumScript([
            qml.CNOT((0, 1)),
            qml.T(0),
            qml.CNOT((3, 2)),
            qml.T(1),
            qml.CNOT((1, 2)),
            qml.T(2),
            qml.CNOT((1, 2)),
            qml.T(1),
            qml.CNOT((3, 2)),
            qml.T(0),
            qml.CNOT((0, 1)),
        ], [])

        print(f"Circuit before:")
        print(qml.drawer.tape_text(circ, wire_order=range(4)))

        new_circ = full_optimize(circ)
        print(f"Circuit after full_optimize:")
        print(qml.drawer.tape_text(new_circ, wire_order=range(4)))

    .. code-block::

        Circuit before:
        0: ─╭●──T──T──────────╭●─┤
        1: ─╰X──T─╭●────╭●──T─╰X─┤
        2: ─╭X────╰X──T─╰X─╭X────┤
        3: ─╰●─────────────╰●────┤

        Circuit after full_optimize:
        0: ──S─╭X──S†─╭X─╭X──Z──T─╭X─╭X─╭X─┤
        1: ────╰●─────│──│────────╰●─│──│──┤
        2: ──Z────────╰●─│───────────╰●─│──┤
        3: ──Z───────────╰●─────────────╰●─┤

    The original five T gates are reduced to just one.

    """
    try:
        pyzx_circ = _tape2pyzx(tape)

        pyzx_circ = zx.full_optimize(pyzx_circ)

    except TypeError:
        warnings.warn(
            "Input circuit is not in the (Clifford + T) basis, will attempt to decompose using qml.clifford_t_decomposition."
        )
        (tape,), _ = qml.clifford_t_decomposition(tape)
        pyzx_circ = _tape2pyzx(tape)

        pyzx_circ = zx.full_optimize(pyzx_circ)

    pl_circ = qml.transforms.from_zx(pyzx_circ.to_graph())

    return pl_circ
