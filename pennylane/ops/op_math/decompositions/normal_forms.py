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

import pennylane as qml
from pennylane.ops.op_math.decompositions.rings import DyadicMatrix, SO3Matrix, ZOmega


@lru_cache
def _clifford_group_to_SO3() -> dict:
    """Return a dictionary mapping Clifford group elements to their corresponding SO(3) matrices."""
    I, X, Y, Z = qml.I(0), qml.X(0), qml.Y(0), qml.Z(0)
    H, S, Sdg = qml.H(0), qml.S(0), qml.adjoint(qml.S(0))
    # These are the Clifford group elements with :math:`\{−1, 0, 1\}` as their matrix entries.
    clifford_elems = {
        I: DyadicMatrix(ZOmega(d=1), ZOmega(), ZOmega(), ZOmega(d=1)),
        H: -DyadicMatrix(ZOmega(b=1), ZOmega(b=1), ZOmega(b=1), ZOmega(b=-1), k=1),
        S: DyadicMatrix(ZOmega(a=-1), ZOmega(), ZOmega(), ZOmega(c=1)),
        X: DyadicMatrix(ZOmega(), ZOmega(b=-1), ZOmega(b=-1), ZOmega()),
        Y: DyadicMatrix(ZOmega(), ZOmega(d=-1), ZOmega(d=1), ZOmega()),
        Z: DyadicMatrix(ZOmega(b=-1), ZOmega(), ZOmega(), ZOmega(b=1)),
        Sdg: DyadicMatrix(ZOmega(c=-1), ZOmega(), ZOmega(), ZOmega(a=1)),
        H @ S: DyadicMatrix(ZOmega(c=-1), ZOmega(a=-1), ZOmega(c=-1), ZOmega(a=1), k=1),
        H @ Z: DyadicMatrix(ZOmega(d=1), ZOmega(d=-1), ZOmega(d=1), ZOmega(d=1), k=1),
        H @ Sdg: DyadicMatrix(ZOmega(a=-1), ZOmega(c=-1), ZOmega(a=-1), ZOmega(c=1), k=1),
        S @ H: DyadicMatrix(ZOmega(c=-1), ZOmega(c=-1), ZOmega(a=-1), ZOmega(a=1), k=1),
        S @ X: DyadicMatrix(ZOmega(), ZOmega(c=-1), ZOmega(a=-1), ZOmega()),
        S @ Y: DyadicMatrix(ZOmega(), ZOmega(a=1), ZOmega(c=1), ZOmega()),
        Z @ H: DyadicMatrix(ZOmega(d=1), ZOmega(d=1), ZOmega(d=-1), ZOmega(d=1), k=1),
        Sdg @ H: DyadicMatrix(ZOmega(a=-1), ZOmega(a=-1), ZOmega(c=-1), ZOmega(c=1), k=1),
        S @ H @ S: DyadicMatrix(ZOmega(d=1), ZOmega(b=1), ZOmega(b=1), ZOmega(d=1), k=1),
        S @ H @ Z: DyadicMatrix(ZOmega(a=-1), ZOmega(a=1), ZOmega(c=1), ZOmega(c=1), k=1),
        S @ H @ Sdg: DyadicMatrix(ZOmega(b=-1), ZOmega(d=-1), ZOmega(d=1), ZOmega(b=1), k=1),
        Z @ H @ S: DyadicMatrix(ZOmega(a=-1), ZOmega(c=1), ZOmega(a=1), ZOmega(c=1), k=1),
        Z @ H @ Z: DyadicMatrix(ZOmega(b=-1), ZOmega(b=1), ZOmega(b=1), ZOmega(b=1), k=1),
        Z @ H @ Sdg: DyadicMatrix(ZOmega(c=-1), ZOmega(a=1), ZOmega(c=1), ZOmega(a=1), k=1),
        Sdg @ H @ S: DyadicMatrix(ZOmega(b=-1), ZOmega(d=1), ZOmega(d=-1), ZOmega(b=1), k=1),
        Sdg @ H @ Z: DyadicMatrix(ZOmega(c=-1), ZOmega(c=1), ZOmega(a=1), ZOmega(a=1), k=1),
        Sdg @ H @ Sdg: DyadicMatrix(ZOmega(d=1), ZOmega(b=-1), ZOmega(b=-1), ZOmega(d=1), k=1),
    }
    return {gate: SO3Matrix(su2) for gate, su2 in clifford_elems.items()}


@lru_cache
def _parity_transforms() -> dict:
    """Returns information required to perform the parity transforms used in the MAnormal form.

    Returns:
        dict: A dictionary mapping parity vectors to tuples containing:
            - The SO(3) matrix representation of its inverse.
            - The PennyLane gate representation of the transformation.
    """
    # The following dictionary maps the keys in the Matsumoto-Amano normal form to their:
    # 1. SO(3) matrix representation (useful for obtaining the parity vector)
    # 2. SO(3) matrix representation of their inverse (useful of reversing the operation)
    # 3. PennyLane gate representation (useful for capturing normal form output)
    transform_ops = {
        "C": (
            SO3Matrix(DyadicMatrix(ZOmega(d=1), ZOmega(), ZOmega(), ZOmega(d=1))),
            SO3Matrix(DyadicMatrix(ZOmega(d=1), ZOmega(), ZOmega(), ZOmega(d=1))),
            qml.I(0),
        ),  # Identity is used as a placeholder to represent arbitrary Clifford group element.
        "T": (
            SO3Matrix(DyadicMatrix(ZOmega(d=1), ZOmega(), ZOmega(), ZOmega(c=1))),
            SO3Matrix(DyadicMatrix(ZOmega(d=1), ZOmega(), ZOmega(), ZOmega(a=-1))),
            qml.T(0),
        ),
        "HT": (
            SO3Matrix(DyadicMatrix(ZOmega(d=1), ZOmega(c=1), ZOmega(d=1), ZOmega(c=-1), k=1)),
            SO3Matrix(DyadicMatrix(ZOmega(d=1), ZOmega(d=1), ZOmega(a=-1), ZOmega(a=1), k=1)),
            qml.H(0) @ qml.T(0),
        ),
        "SHT": (
            SO3Matrix(DyadicMatrix(ZOmega(d=1), ZOmega(c=1), ZOmega(b=1), ZOmega(a=-1), k=1)),
            SO3Matrix(DyadicMatrix(ZOmega(d=1), ZOmega(b=-1), ZOmega(a=-1), ZOmega(c=1), k=1)),
            qml.S(0) @ qml.H(0) @ qml.T(0),
        ),
    }
    return {
        tuple(so3_1.parity_vec): (so3_2, gate) for (so3_1, so3_2, gate) in transform_ops.values()
    }


def _ma_normal_form(
    op: SO3Matrix, compressed=False
) -> tuple[qml.operation.Operator] | tuple[int, tuple[int, ...], int]:
    r"""Decompose an SO(3) matrix into Matsumoto-Amano normal form.

    A Matsumoto-Amano normal form - :math:`(T | \epsilon) (HT | SHT)^* \mathcal{C}`, consists of a rightmost
    Clifford operator, followed by any number of syllables of the form ``HT`` or ``SHT``, followed by an
    optional syllable ``T`` [`arXiv:1312.6584 <https://arxiv.org/abs/1312.6584>`_\ ].

    Args:
        op (SO3Matrix): The SO(3) matrix to decompose.
        compressed (bool): If True, the output will be a single operator that is the product of all gates in the decomposition.
            If False, the output will be a tuple containing information about the decomposition in terms of bits and indices.

    Returns:
        Tuple[qml.operation.Operator] | tuple[int, tuple[int, ...], int]: The decomposition of the SO(3) matrix into Matsumoto-Amano normal forms.
    """
    parity_transforms = _parity_transforms()
    clifford_elements = _clifford_group_to_SO3()

    so3_op = deepcopy(op)

    # The following use lemmas from arXiv:1312.6584.
    decomposition = []
    while (parity_vec := tuple(so3_op.parity_vec)) != (1, 1, 1):  # Fig. 1 and Lemma 6.3
        so3_val, op_gate = parity_transforms[parity_vec]  # Lemma 4.10
        so3_op = so3_val @ so3_op  # Lemma 6.4
        decomposition.append(op_gate)

    cl_index = -1
    for clifford_index, (clifford_gate, clifford_so3) in enumerate(clifford_elements.items()):
        if clifford_so3 == so3_op:
            decomposition.append(clifford_gate)
            cl_index = clifford_index
            break

    if not compressed:
        return qml.prod(*decomposition)

    t_bit = 0
    if decomposition[0] == qml.T(0):  # T gate is the first operation
        t_bit = 1
        decomposition.pop(0)

    c_bit = 0
    if cl_index != -1:  # If a Clifford gate was found
        c_bit = cl_index
        decomposition.pop(-1)

    rep_bits = [0] * len(decomposition)
    for i, d_op in enumerate(decomposition):
        if d_op == parity_transforms[(2, 0, 2)][1]:  # S @ H @ T
            rep_bits[i] = 1

    return (t_bit, tuple(rep_bits), c_bit)
