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

_PAULIS = frozenset({X, Y, Z, I})

_CLIFFORD_TABLEAU = {
    H: [[Z, X]],
    S: [[Y, Z]],
    CNOT: [[X, Z, I, Z], [X, I, X, Z]],
}


def pauli_to_xz(op: Operator) -> Tuple[np.uint8, np.uint8]:
    r"""
    Convert a `Pauli` operator to its `xz` representation up to a global phase, i.e., :math:`encode_{xz}(Pauli)=(x,z)=X^xZ^z)`, where
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

        A xz tuple representation is return for a given Pauli operator.
    """

    if type(op) in _PAULIS:
        return _OPS_TO_XZ[type(op)]
    raise NotImplementedError(f"{op.name} gate does not support xz encoding.")


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
            >>> xz_to_pauli(0, 0)
            <class 'pennylane.ops.identity.Identity'>

        A Pauli operator is returned for a given xz tuple.
    """
    if x in [0, 1] and z in [0, 1]:
        return _XZ_TO_OPS[(x, z)]
    raise ValueError("x and z should either 0 or 1.")


def pauli_prod(ops: List[Operator]) -> Operator:
    """
    Get the result of a product of list of Pauli operators. The result is a new Pauli operator up to global phase.

    Args:
        ops (List[qml.operation.Operator]): A list of Pauli operators with the same target wire.

    Return:
        A new Pauli operator.

    **Example:**
        The following example shows how the `pauli_prod` works.

        .. code-block:: python3
            from pennylane.ftqc.pauli_tracker import pauli_prod
            from pennylane import I, X, Y, Z
            >>> pauli_prod([I(0),X(0),Y(0),Z(0)])
            I(0)

        A Pauli operator is returned for a list of Pauli operator up to global phase.
    """

    if len(ops) == 0:
        raise ValueError("Please ensure that a valid list of operators are passed to the method.")
    op0 = ops.pop()
    res_x, res_z = pauli_to_xz(op0)
    op0_wire = op0.wires

    while len(ops) > 0:
        op = ops.pop()
        wire = op.wires
        if wire != op0_wire:
            raise ValueError("All operators should target at the same wire.")
        x, z = pauli_to_xz(op)
        res_x ^= x
        res_z ^= z

    return xz_to_pauli(res_x, res_z)(wires=op0_wire)


def apply_clifford_op(clifford_op: Operator, paulis: list[Operator]):
    """Conjugate a xz encoded ops to a new xz encoded ops with a given Clifford op.
       
        Args:
            clifford_op (qml.operation.Operator): A Clifford operator class. Supported operators are: :class:`qml.S`, :class:`qml.H`, :class:`qml.CNOT`.
            paulis (List): A list of Pauli operator
        
        Return:
            A list of Pauli operators that clifford_op conjugates the paulis to.
        
        **Example:**
            The following example shows how the `pauli_prod` works.

            .. code-block:: python3
                from pennylane.ftqc.pauli_tracker import apply_clifford_op
                from pennylane import I, CNOT
                >>> apply_clifford_op(CNOT(wires=[0,1]), [I(0), I(1)])
                [I(0), I(1)]

            A list of Pauli operator is returned up to global phase.
    """
    if type(clifford_op) not in _CLIFFORD_TABLEAU:
        raise NotImplementedError("Only qml.H, qml.S and qml.CNOT are supported.")

    if not all(type(pauli) in _PAULIS for pauli in paulis):
        raise ValueError("Please ensure the operator passed in are Paulis.")

    pauli_wires_set = {pauli.wires[0] for pauli in paulis}

    clifford_op_wires_set = {wire for wire in clifford_op.wires}

    if len(paulis) != len(pauli_wires_set):
        raise ValueError("Please ensure each Pauli target at a different wire.")

    if pauli_wires_set != clifford_op_wires_set:
        raise ValueError("Please the target wires of Clifford op match those of Paulis.")

    if all(type(pauli) == I for pauli in paulis):
        return paulis

    wire_map = {}

    for clifford_wire in clifford_op.wires:
        for idx in range(len(paulis)):
            if clifford_wire == paulis[idx].wires[0]:
                wire_map[clifford_wire] = idx

    xz = [pauli_to_xz(paulis[wire_map[clifford_wire]]) for clifford_wire in clifford_op.wires]
    xz = tuple(itertools.chain.from_iterable(xz))

    # A Clifford gate conjugate non-Identify Pauli ops to a new Pauli ops
    new_ops = []
    nonzero_indices = []
    for idx, element in enumerate(xz):
        if element == 1:
            nonzero_indices.append(idx)

    # Get Paulis prod for each target wire
    for wire_idx, table_row in enumerate(_CLIFFORD_TABLEAU[type(clifford_op)]):
        wire = clifford_op.wires[wire_idx]
        ps = []
        for idx in nonzero_indices:
            ps.append(table_row[idx](wires=wire))
        new_ops.append(pauli_prod(ps))
    return new_ops
