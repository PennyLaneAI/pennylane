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
"""Parity matrix representation"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

import pennylane as qml

if TYPE_CHECKING:
    from collections.abc import Callable

    from pennylane.tape import QuantumScript
    from pennylane.workflow import QNode


def parity_matrix(circ: QuantumScript | QNode | Callable, wire_order: Sequence = None):
    r"""Compute the `parity matrix intermediate representation <https://pennylane.ai/compilation/parity-matrix-intermediate-representation>`__ of a CNOT circuit.

    Args:
        circ (QNode or QuantumTape or Callable): Quantum circuit containing only CNOT gates.
        wire_order (Sequence): Wire order indicating how rows and columns should be ordered. If ``None`` is provided, we take the wires of the input circuit (``tape.wires``).

    Returns:
        np.ndarray: :math:`n \times n` Parity matrix for :math:`n` qubits.

    **Example**

    .. code-block:: python

        import pennylane as qml
        from pennylane.labs.intermediate_reps import parity_matrix

        tape = qml.tape.QuantumScript([
            qml.CNOT((3, 2)),
            qml.CNOT((0, 2)),
            qml.CNOT((2, 1)),
            qml.CNOT((3, 2)),
            qml.CNOT((3, 0)),
            qml.CNOT((0, 2)),
        ], [])

        P = parity_matrix(tape, wire_order=range(4))

    >>> P
    array([[1, 0, 0, 1],
           [1, 1, 1, 1],
           [0, 0, 1, 1],
           [0, 0, 0, 1]])

    The corresponding circuit is the following, with output values of the qubits denoted at the right end.

    .. code-block::

        x_0: ────╭●───────╭X─╭●─┤  x_0 ⊕ x_3
        x_1: ────│──╭X────│──│──┤  x_0 ⊕ x_1 ⊕ x_2 ⊕ x_3
        x_2: ─╭X─╰X─╰●─╭X─│──╰X─┤  x_2 ⊕ x_3
        x_3: ─╰●───────╰●─╰●────┤  x_3

    For more details, see the `compilation page <https://pennylane.ai/compilation/parity-matrix-intermediate-representation>`__ on the parity matrix intermediate representation.

    """
    if callable(circ):
        circ = qml.tape.make_qscript(circ)()

    wires = circ.wires

    if wire_order is None:
        wire_order = wires

    if not qml.wires.Wires(wire_order).contains_wires(wires):
        raise qml.wires.WireError(
            f"The provided wire_order {wire_order} does not contain all wires of the circuit {wires}"
        )

    if any(op.name != "CNOT" for op in circ.operations):
        raise TypeError(
            f"parity_matrix requires all input circuits to consist solely of CNOT gates. Received circuit with the following gates: {circ.operations}"
        )

    wire_map = {wire: idx for idx, wire in enumerate(wire_order)}

    P = np.eye(len(wire_order), dtype=int)
    for op in circ.operations:

        control, target = op.wires
        P[wire_map[target]] += P[wire_map[control]]

    return P % 2
