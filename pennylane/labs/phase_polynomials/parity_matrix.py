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

import numpy as np


def parity_matrix(circ):
    r"""
    :doc:`Parity matrix intermediate representation <compilation/parity-matrix-intermediate-representation>` of a CNOT circuit

    Args:
        circ (qml.tape.QuantumScript):

    Returns:
        np.ndarray: :math:`n \times n` Parity matrix

    **Example**

    .. code-block:: python

        circ = qml.tape.QuantumScript([
            qml.CNOT((3, 2)),
            qml.CNOT((0, 2)),
            qml.CNOT((2, 1)),
            qml.CNOT((3, 2)),
            qml.CNOT((3, 0)),
            qml.CNOT((0, 2)),
        ], [])

        P = parity_matrix(circ)

    >>> P
    array([[1., 0., 0., 1.],
           [1., 1., 1., 1.],
           [0., 0., 1., 1.],
           [0., 0., 0., 1.]])

    The corresponding circuit is the following.

    .. code-block::

        x_0: ────╭●───────╭X─╭●─┤  x_0 ⊕ x_3
        x_1: ────│──╭X────│──│──┤  x_0 ⊕ x_1 ⊕ x_2 ⊕ x_3
        x_2: ─╭X─╰X─╰●─╭X─│──╰X─┤  x_2 ⊕ x_3
        x_3: ─╰●───────╰●─╰●────┤  x_3

    """
    wires = circ.wires
    P = np.eye(len(wires))
    for op in circ.operations:
        control, target = op.wires
        P[target] = P[target] + P[control]

    return P % 2
