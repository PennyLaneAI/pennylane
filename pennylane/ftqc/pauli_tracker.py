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


def pauli_to_xz(op: Operator) -> Tuple[int, int]:
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


def xz_to_pauli(x: int, z: int) -> Operator:
    """
    Convert x, z to a Pauli operator class.

    Args:
        x (int) : Exponent of :class:`~pennylane.X` in the Pauli record.
        z (int) : Exponent of :class:`~pennylane.Z` in the Pauli record.

    Return:
        A Pauli operator class.

    **Example:**
        The following example shows how the XZ to Pauli works.

        .. code-block:: python3

            from pennylane.ftqc.pauli_tracker import xz_to_pauli
            >>> xz_to_pauli(0, 0)(wires=0)
            I(0)

        A Pauli operator class is returned for a given xz tuple.
    """
    if x in [0, 1] and z in [0, 1]:
        return _XZ_TO_OPS[(x, z)]
    raise ValueError("x and z should either 0 or 1.")


def pauli_prod(ops: List[Operator]) -> Tuple[int, int]:
    r"""
    Get the result of a product of a list of Pauli operators. The result is a new Pauli operator up to a global phase.
    Mathematically, this function returns :math:`\prod_{i=0}^{n} ops[i]`.

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

        The result is a new Pauli operator in the xz-encoding representation.
    """
    if len(ops) == 0:
        raise ValueError("Please ensure that a valid list of operators are passed to the method.")
    res_x, res_z = pauli_to_xz(ops[0])

    for i in range(1, len(ops)):
        x, z = pauli_to_xz(ops[i])
        res_x ^= x
        res_z ^= z

    return (res_x, res_z)


def _commute_h(x: int, z: int):
    r"""
    Commute/move a Pauli represented by xz through :class:`~pennylane.H`.

    Args:
        x(int): Exponent of PauliX in the xz representation of a Pauli.
        z(int): Exponent of PauliZ in the xz representation of a Pauli.

    Return:
        A list of a tuple of xz representing a new Pauli operation that the :class:`~pennylane.H` commutes to.
    """
    return [(z, x)]


def _commute_s(x: int, z: int):
    r"""
    Commute/move a Pauli represented by xz through :class:`~pennylane.S`.

    Args:
        x(int): Exponent of PauliX in the xz representation of a Pauli.
        z(int): Exponent of PauliZ in the xz representation of a Pauli.

    Return:
        A list of a tuple of xz representing a new Pauli operation that the :class:`~pennylane.S` commutes to.
    """
    return [(x, x ^ z)]


def _commute_cnot(xc: int, zc: int, xt: int, zt: int):
    r"""
    Commute/move a Pauli represented by xz through :class:`~pennylane.CNOT`.

    Args:
        xc(int): Exponent of PauliX in the xz representation of a Pauli at the control wire.
        zc(int): Exponent of PauliZ in the xz representation of a Pauli at the control wire.
        xt(int): Exponent of PauliX in the xz representation of a Pauli at the target wire.
        zt(int): Exponent of PauliZ in the xz representation of a Pauli at the target wire.

    Return:
        A list of xz tuples representing new Paulis operation that the :class:`~pennylane.cnot` commutes to.
    """
    return [(xc, zc ^ zt), (xc ^ xt, zt)]


def commute_clifford_op(clifford_op: Operator, xz: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    r"""Gets the list of xz-encoded bits representing the list of input Pauli ops after being commuted through the given Clifford op.
    Mathematically, this function applies the following equation: :math:`new\_xz \cdot clifford\_op = clifford\_op \cdot xz`
    up to a global phase to move the :math:`xz` through the :math:`clifford\_op` and returns the :math:`new\_xz`. Note that :math:`xz` and
    :math:`new\_xz` represent a list of Pauli operations.

    Args:
        clifford_op (Operator): A Clifford operator class. Supported operators are: :class:`qml.S`, :class:`qml.H`, :class:`qml.CNOT`.
        xz (list(tuple)): A list of xz tuples which map to Pauli operators

    Return:
        A list of new xz tuples that the clifford_op commute the xz to.

    **Example:**
        The following example shows how the `commute_clifford_op` works.

        .. code-block:: python3

            from pennylane.ftqc.pauli_tracker import commute_clifford_op
            from pennylane import I, CNOT
            >>> commute_clifford_op(CNOT(wires=[0,1]), [(1, 1), (1, 0)])
            [(1, 1), (0, 0)]

        A list of Pauli operators in the xz representation is returned.
    """
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

    if isinstance(clifford_op, S):
        _x, _z = xz[0]
        return _commute_s(_x, _z)

    if isinstance(clifford_op, H):
        _x, _z = xz[0]
        return _commute_h(_x, _z)

    if isinstance(clifford_op, CNOT):
        _xc, _zc = xz[0]
        _xt, _zt = xz[1]
        return _commute_cnot(_xc, _zc, _xt, _zt)

    raise NotImplementedError("Only qml.H, qml.S and qml.CNOT are supported.")
