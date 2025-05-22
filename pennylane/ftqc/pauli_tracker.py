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

import numpy as np

from pennylane import CNOT, H, I, S, X, Y, Z
from pennylane.operation import Operator

_OPS_TO_XZ = {
    I: (0, 0),
    X: (1, 0),
    Y: (1, 1),
    Z: (0, 1),
}

_XZ_TO_OPS = {
    (0, 0): I,
    (1, 0): X,
    (1, 1): Y,
    (0, 1): Z,
}

_PAULIS = (X, Y, Z, I)

_CLIFFORD_TABLEAU = {
    H: [[Z, X]],
    S: [[Y, Z]],
    CNOT: [[X, Z, I, Z], [X, I, X, Z]],
}


def pauli_to_xz(op: Operator) -> Tuple[np.uint8, np.uint8]:
    r"""
    Convert a `Pauli` operator to its `xz` representation up to a global phase, i.e., :math:`encode_{xz}(Pauli)=(x,z)=X^xZ^z`, where
    :math:`x` is the exponent of the :class:`~pennylane.X` and :math:`z` is the exponent of
    the :class:`~pennylane.Z`, meaning :math:`encode_{xz}(I) = (0, 0)`, :math:`encode_{xz}(X) = (1, 0)`,
    :math:`encode_{xz}(Y) = (1, 1)` and :math:`encode_{xz}(Z) = (0, 1)`.

    Args:
        op (qml.operation.Operator): A Pauli operator.

    Return:
        A tuple of xz encoding data, :math:`x` is the exponent of the :class:`~pennylane.X`, :math:`z` is the exponent of
        the :class:`~pennylane.Z`.

    **Example:**
        The following example shows how the Pauli to XZ works.

        .. code-block:: python3

            from pennylane.ftqc.pauli_tracker import pauli_to_xz
            from pennylane import I
            >>> pauli_to_xz(I(0))
            (0, 0)

        A xz tuple representation is returned for a given Pauli operator.
    """

    if isinstance(op, _PAULIS):
        return _OPS_TO_XZ[type(op)]

    if op in _PAULIS:
        return _OPS_TO_XZ[op]

    raise NotImplementedError(f"{type(op)} gate does not support xz encoding.")


def xz_to_pauli(x: np.uint8, z: np.uint8) -> Operator:
    """
    Convert x, z to a Pauli operator.

    Args:
        x (np.uint8) : Exponent of :class:`~pennylane.X` in the Pauli record.
        z (np.uint8) : Exponent of :class:`~pennylane.Z` in the Pauli record.

    Return:
        A Pauli operator.

    **Example:**
        The following example shows how the XZ to Pauli works.

        .. code-block:: python3

            from pennylane.ftqc.pauli_tracker import xz_to_pauli
            >>> xz_to_pauli(0, 0)(wires=0)
            I(0)

        A Pauli operator is returned for a given xz tuple.
    """
    if x in [0, 1] and z in [0, 1]:
        return _XZ_TO_OPS[(x, z)]
    raise ValueError("x and z should either 0 or 1.")


def pauli_prod(ops: List[Operator]) -> Tuple[np.uint8, np.uint8]:
    """
    Get the result of a product of a list of Pauli operators. The result is a new Pauli operator up to a global phase.

    Args:
        ops (List[qml.operation.Operator]): A list of Pauli operators with the same target wire.

    Return:
        A xz tuple representing a new Pauli operator.

    **Example:**
        The following example shows how the `pauli_prod` works.

        .. code-block:: python3

            from pennylane.ftqc.pauli_tracker import pauli_prod
            from pennylane import I, X, Y, Z
            >>> pauli_prod([I(0),X(0),Y(0),Z(0)])
            (0, 0)

        A Pauli operator is returned for a list of Pauli operator up to a global phase.
    """

    if len(ops) == 0:
        raise ValueError("Please ensure that a valid list of operators are passed to the method.")
    res_x, res_z = pauli_to_xz(ops.pop())

    while len(ops) > 0:
        x, z = pauli_to_xz(ops.pop())
        res_x ^= x
        res_z ^= z

    return (res_x, res_z)


def apply_clifford_op(
    clifford_op: Operator, xz: List[Tuple[np.uint8, np.uint8]]
) -> List[Tuple[np.uint8, np.uint8]]:
    """Commuting a list of Pauli ops to a list of new Pauli ops with a given Clifford op.

    Args:
        clifford_op (Operator): A Clifford operator class. Supported operators are: :class:`qml.S`, :class:`qml.H`, :class:`qml.CNOT`.
        xz (list(tuple)): A list of xz tuples which map to Pauli operators

    Return:
        A list of new xz tuples that the clifford_op commute the xz to.

    **Example:**
        The following example shows how the `pauli_prod` works.

        .. code-block:: python3

            from pennylane.ftqc.pauli_tracker import apply_clifford_op
            from pennylane import I, CNOT
            >>> apply_clifford_op(CNOT(wires=[0,1]), [(0, 0), (0, 0)])
            [(0, 0), (0, 0)]

        A list of xz representation of Pauli operators is returned.
    """
    if type(clifford_op) not in _CLIFFORD_TABLEAU:
        raise NotImplementedError("Only qml.H, qml.S and qml.CNOT are supported.")

    if len(xz) != clifford_op.num_wires:
        raise ValueError(
            "Please ensure that the length of xz matches the number of wires of the clifford_op."
        )

    if not all(len(element) == 2 for element in xz):
        raise ValueError(
            "Please ensure there are 2 elements instead of in each tuple in the xz list."
        )

    xz_flatten = tuple(itertools.chain.from_iterable(xz))

    if not all(element in [0, 1] for element in xz_flatten):
        raise ValueError("Please ensure xz are either 0 or 1.")

    if all(element == (0, 0) for element in xz):
        return xz

    # A Clifford gate conjugate non-Identify Pauli ops to a new Pauli ops
    new_xz = []
    nonzero_indices = [idx for idx, element in enumerate(xz_flatten) if element == 1]

    # Get Paulis prod for each target wire
    for table_row in _CLIFFORD_TABLEAU[type(clifford_op)]:
        ps = [table_row[idx] for idx in nonzero_indices]
        new_xz.append(pauli_prod(ps))
    return new_xz
