# Copyright 2018 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""
QML Embeddings
==============

**Module name:** :mod:`pennylane.embeddings`

.. currentmodule:: pennylane.embeddings

This module provides quantum circuit architectures that can serve as an embedding of inputs
(represented by the gate parameters) into a quantum state, according to Schuld & Killoran 2019
:cite:`schuld2019`.

Provided embeddings
-------------------

.. autosummary::

    cosine_encoding
    amplitude_encoding
    basis_encoding
    angle_encoding
    squeezing_encoding

Code details
^^^^^^^^^^^^
"""
#pylint: disable-msg=too-many-branches,too-many-arguments,protected-access
from pennylane import RX, RY, RZ


def cosine_encoding(features, n_qubits, rotation='X'):
    """
    Rotates each qubit of the circuit by subsequent entries of x.

    The `rotation` parameter defines whether an x-rotation (`rotation = 'X'`), a y- rotation (`rotation = 'Y'`),
    or a z-rotation (`rotation = 'Z'`) is used. For `rotation = 'XYZ', the circuit iterates throug `x` and uses the
    first `n_qubits` entries in x-rotations, the next `n_qubit` entries in y-rotations, and the last `n_qubit` entries
    in z-rotations.

    If there are fewer entries in `x` than rotations available, the circuit does not apply the remaining rotation gates.
    If there are fewer rotations than entries in `x`, the circuit will not use the remaining entries.

    Args:
        features (array): Input array of shape (N, ), where N is the number of input features to embed
        n_qubits (int): Number of qubits in the circuit
        rotation (str): Mode of operation

    """
    if rotation == 'XYZ':
        n_ops = min(len(features), 3*n_qubits)

        for op in range(n_ops):
            if op < n_qubits:
                RX(features[op], wires=op)
            elif op < 2*n_qubits:
                RY(features[op], wires=op)
            else:
                RZ(features[op], wires=op)
    else:
        n_ops = min(len(features), n_qubits)

        if rotation == 'X':
            for op in range(n_ops):
                RX(features[op], wires=op)

        elif rotation == 'Y':
            for op in range(n_ops):
                RY(features[op], wires=op)

        elif rotation == 'Z':
            for op in range(n_ops):
                RZ(features[op], wires=op)


def amplitude_encoding(features, n_qubits, execution='hack'):
    """
    Prepares a quantum state whose amplitude vector is given by `features`.

    `features` is a 1-d vector of unit length and with 2**n_qubits entries.

    .. note::
        At the moment only the execution mode 'hack' is implemented, where

    Args:
        features (array): Input array of shape (N, ), where N is the number of input features to embed
        n_qubits (int): Number of qubits in the circuit
        rotation (str): Mode of operation

    """
