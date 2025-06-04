# Copyright 2018-2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Decompositions for Matsumoto-Amano normal forms in SO(3) matrices."""

from copy import deepcopy
from functools import lru_cache

from rings import SO3Matrix, SU2Matrix, ZOmega

import pennylane as qml


@lru_cache
def _clifford_group_to_SO3():
    """Return a dictionary mapping Clifford group elements to their corresponding SO(3) matrices."""
    clifford_elems = {
        qml.I(0): SU2Matrix(ZOmega(d=1), ZOmega(), ZOmega(), ZOmega(d=1)),
        qml.H(0): -SU2Matrix(ZOmega(b=1), ZOmega(b=1), ZOmega(b=1), ZOmega(b=-1), k=1),
        qml.S(0): SU2Matrix(ZOmega(a=-1), ZOmega(), ZOmega(), ZOmega(c=1)),
        qml.X(0): SU2Matrix(ZOmega(), ZOmega(b=-1), ZOmega(b=-1), ZOmega()),
        qml.Y(0): SU2Matrix(ZOmega(), ZOmega(d=-1), ZOmega(d=1), ZOmega()),
        qml.Z(0): SU2Matrix(ZOmega(b=-1), ZOmega(), ZOmega(), ZOmega(b=1)),
        qml.adjoint(qml.S(0)): SU2Matrix(ZOmega(c=-1), ZOmega(), ZOmega(), ZOmega(a=1)),
        qml.H(0) @ qml.S(0): SU2Matrix(ZOmega(c=-1), ZOmega(a=-1), ZOmega(c=-1), ZOmega(a=1), k=1),
        qml.H(0) @ qml.Z(0): SU2Matrix(ZOmega(d=1), ZOmega(d=-1), ZOmega(d=1), ZOmega(d=1), k=1),
        qml.H(0)
        @ qml.adjoint(qml.S(0)): SU2Matrix(
            ZOmega(a=-1), ZOmega(c=-1), ZOmega(a=-1), ZOmega(c=1), k=1
        ),
        qml.S(0) @ qml.H(0): SU2Matrix(ZOmega(c=-1), ZOmega(c=-1), ZOmega(a=-1), ZOmega(a=1), k=1),
        qml.S(0) @ qml.X(0): SU2Matrix(ZOmega(), ZOmega(c=-1), ZOmega(a=-1), ZOmega()),
        qml.S(0) @ qml.Y(0): SU2Matrix(ZOmega(), ZOmega(a=1), ZOmega(c=1), ZOmega()),
        qml.Z(0) @ qml.H(0): SU2Matrix(ZOmega(d=1), ZOmega(d=1), ZOmega(d=-1), ZOmega(d=1), k=1),
        qml.adjoint(qml.S(0))
        @ qml.H(0): SU2Matrix(ZOmega(a=-1), ZOmega(a=-1), ZOmega(c=-1), ZOmega(c=1), k=1),
        qml.S(0)
        @ qml.H(0)
        @ qml.S(0): SU2Matrix(ZOmega(d=1), ZOmega(b=1), ZOmega(b=1), ZOmega(d=1), k=1),
        qml.S(0)
        @ qml.H(0)
        @ qml.Z(0): SU2Matrix(ZOmega(a=-1), ZOmega(a=1), ZOmega(c=1), ZOmega(c=1), k=1),
        qml.S(0)
        @ qml.H(0)
        @ qml.adjoint(qml.S(0)): SU2Matrix(
            ZOmega(b=-1), ZOmega(d=-1), ZOmega(d=1), ZOmega(b=1), k=1
        ),
        qml.Z(0)
        @ qml.H(0)
        @ qml.S(0): SU2Matrix(ZOmega(a=-1), ZOmega(c=1), ZOmega(a=1), ZOmega(c=1), k=1),
        qml.Z(0)
        @ qml.H(0)
        @ qml.Z(0): SU2Matrix(ZOmega(b=-1), ZOmega(b=1), ZOmega(b=1), ZOmega(b=1), k=1),
        qml.Z(0)
        @ qml.H(0)
        @ qml.adjoint(qml.S(0)): SU2Matrix(
            ZOmega(c=-1), ZOmega(a=1), ZOmega(c=1), ZOmega(a=1), k=1
        ),
        qml.adjoint(qml.S(0))
        @ qml.H(0)
        @ qml.S(0): SU2Matrix(ZOmega(b=-1), ZOmega(d=1), ZOmega(d=-1), ZOmega(b=1), k=1),
        qml.adjoint(qml.S(0))
        @ qml.H(0)
        @ qml.Z(0): SU2Matrix(ZOmega(c=-1), ZOmega(c=1), ZOmega(a=1), ZOmega(a=1), k=1),
        qml.adjoint(qml.S(0))
        @ qml.H(0)
        @ qml.adjoint(qml.S(0)): SU2Matrix(
            ZOmega(d=1), ZOmega(b=-1), ZOmega(b=-1), ZOmega(d=1), k=1
        ),
    }
    return {gate: SO3Matrix(su2) for gate, su2 in clifford_elems.items()}


@lru_cache
def _parity_transforms():
    """Return a dictionary mapping parity transformations to their corresponding SO(3) matrices."""
    transform_ops = {
        "C": (
            SO3Matrix(SU2Matrix(ZOmega(d=1), ZOmega(), ZOmega(), ZOmega(d=1))),
            SO3Matrix(SU2Matrix(ZOmega(d=1), ZOmega(), ZOmega(), ZOmega(d=1))),
            qml.I(0),
        ),
        "T": (
            SO3Matrix(SU2Matrix(ZOmega(d=1), ZOmega(), ZOmega(), ZOmega(c=1))),
            SO3Matrix(SU2Matrix(ZOmega(d=1), ZOmega(), ZOmega(), ZOmega(a=-1))),
            qml.T(0),
        ),
        "HT": (
            SO3Matrix(SU2Matrix(ZOmega(d=1), ZOmega(c=1), ZOmega(d=1), ZOmega(c=-1), k=1)),
            SO3Matrix(SU2Matrix(ZOmega(d=1), ZOmega(d=1), ZOmega(a=-1), ZOmega(a=1), k=1)),
            qml.H(0) @ qml.T(0),
        ),
        "SHT": (
            SO3Matrix(SU2Matrix(ZOmega(d=1), ZOmega(c=1), ZOmega(b=1), ZOmega(a=-1), k=1)),
            SO3Matrix(SU2Matrix(ZOmega(d=1), ZOmega(b=-1), ZOmega(a=-1), ZOmega(c=1), k=1)),
            qml.S(0) @ qml.H(0) @ qml.T(0),
        ),
    }

    return {
        tuple(so3_1.parity_vec): (so3_2, gate) for (so3_1, so3_2, gate) in transform_ops.values()
    }


def ma_normal_form(op: SO3Matrix):
    """Decompose an SO(3) matrix into Matsumoto-Amano normal forms.

    Args:
        op (SO3Matrix): The SO(3) matrix to decompose.

    Returns:
        Tuple[qml.operation.Operator]: The decomposition of the SO(3) matrix into Matsumoto-Amano normal forms.
    """
    parity_transforms = _parity_transforms()
    clifford_elements = _clifford_group_to_SO3()

    so3_op = deepcopy(op)

    decomposition = qml.I(0)
    while (parity_vec := tuple(so3_op.parity_vec)) != (1, 1, 1):
        so3_val, op_gate = parity_transforms[parity_vec]
        print(parity_vec, op_gate)
        decomposition = op_gate @ decomposition
        so3_op = so3_val @ so3_op

    for clifford_gate, clifford_so3 in clifford_elements.items():
        if clifford_so3 == so3_op:
            decomposition = clifford_gate @ decomposition

    return decomposition[:-1]
