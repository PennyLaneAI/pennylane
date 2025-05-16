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

import itertools
from typing import List, Tuple

import pennylane as qml
from pennylane import numpy as np
from pennylane.operation import Operator

_ENCODE_XZ_OPS = {
    qml.I: (0, 0),
    qml.X: (1, 0),
    qml.Y: (1, 1),
    qml.Z: (0, 1),
}

_DECODE_XZ = {
    (0, 0): qml.I,
    (1, 0): qml.X,
    (1, 1): qml.Y,
    (0, 1): qml.Z,
}

_CLIFFORD_TABLEAU = {
    qml.H: [[qml.Z, qml.X]],
    qml.S: [[qml.Y, qml.Z]],
    qml.CNOT: [[qml.X, qml.Z, qml.I, qml.Z], [qml.X, qml.I, qml.X, qml.Z]],
}

_paulis = (qml.X, qml.Y, qml.Z, qml.I)


def pauli_encode_xz(op: Operator) -> Tuple[np.uint8, np.uint8]:
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


def xz_decode_pauli(x: np.uint8, z: np.uint8):
    """
    Decode a x, z to a Pauli operator.

    Args:
        x (np.uint8) : Exponent of :class:`~pennylane.X` in the Pauli record.
        z (np.uint8) : Exponent of :class:`~pennylane.Z` in the Pauli record.

    Return:
        A Pauli operator.
    """
    if x in [0, 1] and z in [0, 1]:
        return _DECODE_XZ[(x, z)]
    raise ValueError("x and z should either 0 or 1.")


def pauli_prod_to_xz(ops: List[Operator]) -> Operator:
    """
    Get the result of a product of list of Pauli operators. The result is encoded with `xz` representation.

    Args:
        ops (List[qml.operation.Operator]): A list of Pauli operators with the same target wire.

    Return:
        A Pauli operator.
    """

    if len(ops) == 0:
        raise ValueError("Please ensure that a valid list of operators are passed to the method.")

    res_x, res_z = pauli_encode_xz(ops.pop())

    while len(ops) > 0:
        x, z = pauli_encode_xz(ops.pop())
        res_x ^= x
        res_z ^= z
    return xz_decode_pauli(res_x, res_z)


def apply_clifford_op(clifford_op: Operator, paulis: list[Operator]):
    """Conjugate a xz encoded ops to a new xz encoded ops with a given
    Clifford op.

        Args:
            clifford_op (qml.operation.Operator): A Clifford operator class. Supported operators are: :class:`qml.S`, :class:`qml.H`, :class:`qml.CNOT`.
            paulis (List): A list of Pauli operator

        Return:
            A list of Pauli operators that clifford_op conjugates the paulis to.
    """
    if clifford_op not in _CLIFFORD_TABLEAU:
        raise NotImplementedError("Only qml.H, qml.S and qml.CNOT are supported.")

    if not all(pauli in _paulis for pauli in paulis):
        raise ValueError("Please ensure the operator passed in are Paulis")

    if clifford_op.num_wires != len(paulis):
        raise ValueError(
            "Please ensure the number of Paulis matches the number of wires of the Clifford gate."
        )

    if all(pauli == qml.I for pauli in paulis):
        return paulis

    xz = [pauli_encode_xz(pauli) for pauli in paulis]
    xz = tuple(itertools.chain.from_iterable(xz))

    # A Clifford gate conjugate non-Identify Pauli ops to a new Pauli ops
    new_ops = []
    nonzero_indices = []
    for idx, element in enumerate(xz):
        if element == 1:
            nonzero_indices.append(idx)

    # Get Paulis prod for each target wire
    for table_row in _CLIFFORD_TABLEAU[clifford_op]:
        ps = []
        for idx in nonzero_indices:
            ps.append(table_row[idx])
        new_ops.append(pauli_prod_to_xz(ps))
    return new_ops
