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

r"""
This module contains Pauli Tracking functions.
"""

from typing import List, Tuple

import pennylane as qml
from pennylane import numpy as np

_ENCODE_XZ_OPS = {
    qml.I: (0, 0),
    qml.X: (1, 0),
    qml.Y: (1, 1),
    qml.Z: (0, 1),
}

_paulis = frozenset({qml.X, qml.Y, qml.Z, qml.I})


def pauli_encode_xz(op: qml.operation.Operator) -> Tuple[np.uint8, np.uint8]:
    """
    Encode a `Pauli` operator to its `xz` representation up to a global phase, i.e., :math:`encode_{xz}(Pauli)=(x,z)=X^xZ^z)`, where
    :math:`x` is the exponent of the :class:`~pennylane.X` and :math:`z` is the exponent of
    the :class:`~pennylane.Z`, meaning :math:`encode_{xz}(I) = (0, 0)`, :math:`encode_{xz}(X) = (1, 0)`,
    :math:`encode_{xz}(Y) = (1, 1)` and :math:`encode_{xz}(Z) = (0, 1)`.

    Args:
        op (qml.operation.Operator): A Pauli operator.

    Return:
        A tuple of xz encoding data, :math:`x` is the exponent of the :class:`~pennylane.X`, :math:`z` is the exponent of
    the :class:`~pennylane.Z`.
    """

    if op in _paulis:
        return _ENCODE_XZ_OPS[op]
    raise NotImplementedError(f"{op.name} gate does not support xz encoding.")


def pauli_prod_to_xz(ops: List[qml.operation.Operator]) -> Tuple[np.uint8, np.uint8]:
    """
    Get the result of a product of list of Pauli operators. The result is encoded with `xz` representation.

    Args:
        ops (List[qml.operation.Operator]): A list of Pauli operators with the same target wire.

    Return:
        A tuple of `xz` encoding data, :math:`x` is the exponent of the :class:`~pennylane.X`, :math:`z` is the exponent of
    the :class:`~pennylane.Z`.
    """

    if len(ops) == 0:
        raise ValueError("Please ensure that a valid list of operators are passed to the method.")

    res = pauli_encode_xz(ops.pop())

    while len(ops) > 0:
        res = np.bitwise_xor(pauli_encode_xz(ops.pop()), res)

    return res
