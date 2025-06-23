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
"""Decompositions for Matsumoto-Amano normal forms in SO(3) matrices."""

from copy import deepcopy
from functools import lru_cache

import pennylane as qml
from pennylane.ops.op_math.decompositions.rings import DyadicMatrix, SO3Matrix, ZOmega


@lru_cache
def _clifford_group_to_SO3() -> dict:
    r"""Return a dictionary mapping Clifford group elements to their corresponding SO(3) matrices
    and global phase scaled by :math:`\pi^{-1}`."""
    I, X, Y, Z = qml.I(0), qml.X(0), qml.Y(0), qml.Z(0)
    H, S, Sd = qml.H(0), qml.S(0), qml.adjoint(qml.S(0))
    # These are the Clifford group elements with :math:`\{−1, 0, 1\}` as their matrix entries.
    clifford_elems = {
        (I,): (DyadicMatrix(ZOmega(d=1), ZOmega(), ZOmega(), ZOmega(d=1)), 0.0),
        (H,): (-DyadicMatrix(ZOmega(b=1), ZOmega(b=1), ZOmega(b=1), ZOmega(b=-1), k=1), 0.5),
        (S,): (DyadicMatrix(ZOmega(a=-1), ZOmega(), ZOmega(), ZOmega(c=1)), 0.25),
        (X,): (DyadicMatrix(ZOmega(), ZOmega(b=-1), ZOmega(b=-1), ZOmega()), 0.5),
        (Y,): (DyadicMatrix(ZOmega(), ZOmega(d=-1), ZOmega(d=1), ZOmega()), 0.5),
        (Z,): (DyadicMatrix(ZOmega(b=-1), ZOmega(), ZOmega(), ZOmega(b=1)), 0.5),
        (Sd,): (DyadicMatrix(ZOmega(c=-1), ZOmega(), ZOmega(), ZOmega(a=1)), 0.75),
        (H, S): (DyadicMatrix(ZOmega(c=-1), ZOmega(a=-1), ZOmega(c=-1), ZOmega(a=1), k=1), 0.75),
        (H, Z): (DyadicMatrix(ZOmega(d=1), ZOmega(d=-1), ZOmega(d=1), ZOmega(d=1), k=1), 0.0),
        (H, Sd): (DyadicMatrix(ZOmega(a=-1), ZOmega(c=-1), ZOmega(a=-1), ZOmega(c=1), k=1), 0.25),
        (S, H): (DyadicMatrix(ZOmega(c=-1), ZOmega(c=-1), ZOmega(a=-1), ZOmega(a=1), k=1), 0.75),
        (S, X): (DyadicMatrix(ZOmega(), ZOmega(c=-1), ZOmega(a=-1), ZOmega()), 0.75),
        (S, Y): (DyadicMatrix(ZOmega(), ZOmega(a=1), ZOmega(c=1), ZOmega()), 0.75),
        (Z, H): (DyadicMatrix(ZOmega(d=1), ZOmega(d=1), ZOmega(d=-1), ZOmega(d=1), k=1), 0.0),
        (Sd, H): (DyadicMatrix(ZOmega(a=-1), ZOmega(a=-1), ZOmega(c=-1), ZOmega(c=1), k=1), 0.25),
        (S, H, S): (DyadicMatrix(ZOmega(d=1), ZOmega(b=1), ZOmega(b=1), ZOmega(d=1), k=1), 0.0),
        (S, H, Z): (DyadicMatrix(ZOmega(a=-1), ZOmega(a=1), ZOmega(c=1), ZOmega(c=1), k=1), 0.25),
        (S, H, Sd): (DyadicMatrix(ZOmega(b=-1), ZOmega(d=-1), ZOmega(d=1), ZOmega(b=1), k=1), 0.5),
        (Z, H, S): (DyadicMatrix(ZOmega(a=-1), ZOmega(c=1), ZOmega(a=1), ZOmega(c=1), k=1), 0.25),
        (Z, H, Z): (DyadicMatrix(ZOmega(b=-1), ZOmega(b=1), ZOmega(b=1), ZOmega(b=1), k=1), 0.5),
        (Z, H, Sd): (DyadicMatrix(ZOmega(c=-1), ZOmega(a=1), ZOmega(c=1), ZOmega(a=1), k=1), 0.75),
        (Sd, H, S): (DyadicMatrix(ZOmega(b=-1), ZOmega(d=1), ZOmega(d=-1), ZOmega(b=1), k=1), 0.5),
        (Sd, H, Z): (DyadicMatrix(ZOmega(c=-1), ZOmega(c=1), ZOmega(a=1), ZOmega(a=1), k=1), 0.75),
        (Sd, H, Sd): (DyadicMatrix(ZOmega(d=1), ZOmega(b=-1), ZOmega(b=-1), ZOmega(d=1), k=1), 0.0),
    }
    return {gate: (SO3Matrix(su2), phase) for gate, (su2, phase) in clifford_elems.items()}


@lru_cache
def _parity_transforms() -> dict:
    r"""Returns information required to perform the parity transforms used in the _ma_normal_form.

    Returns:
        dict: A dictionary mapping parity vectors to tuples containing:
            - The SO(3) matrix representation of its inverse.
            - The PennyLane gate representation of the transformation.
            - The global phase scaled by :math:`\pi^{-1}` of the transformation.
    """
    # The following dictionary maps the keys in the Matsumoto-Amano normal form to their:
    # 1. SO(3) matrix representation (useful for obtaining the parity vector)
    # 2. SO(3) matrix representation of their inverse (useful of reversing the operation)
    # 3. PennyLane gate representation (useful for capturing normal form output)
    transform_ops = {
        "C": (
            SO3Matrix(DyadicMatrix(ZOmega(d=1), ZOmega(), ZOmega(), ZOmega(d=1))),
            SO3Matrix(DyadicMatrix(ZOmega(d=1), ZOmega(), ZOmega(), ZOmega(d=1))),
            (qml.I(0),),
            0.0,
        ),  # Identity is used as a placeholder to represent arbitrary Clifford group element.
        "T": (
            SO3Matrix(DyadicMatrix(ZOmega(d=1), ZOmega(), ZOmega(), ZOmega(c=1))),
            SO3Matrix(DyadicMatrix(ZOmega(d=1), ZOmega(), ZOmega(), ZOmega(a=-1))),
            (qml.T(0),),
            0.125,
        ),
        "HT": (
            SO3Matrix(DyadicMatrix(ZOmega(d=1), ZOmega(c=1), ZOmega(d=1), ZOmega(c=-1), k=1)),
            SO3Matrix(DyadicMatrix(ZOmega(d=1), ZOmega(d=1), ZOmega(a=-1), ZOmega(a=1), k=1)),
            (qml.H(0), qml.T(0)),
            0.625,
        ),
        "SHT": (
            SO3Matrix(DyadicMatrix(ZOmega(d=1), ZOmega(c=1), ZOmega(b=1), ZOmega(a=-1), k=1)),
            SO3Matrix(DyadicMatrix(ZOmega(d=1), ZOmega(b=-1), ZOmega(a=-1), ZOmega(c=1), k=1)),
            (qml.S(0), qml.H(0), qml.T(0)),
            0.875,
        ),
    }
    return {
        tuple(so3_1.parity_vec): (so3_2, gate, phase)
        for (so3_1, so3_2, gate, phase) in transform_ops.values()
    }


def _ma_normal_form(
    op: SO3Matrix, compressed=False
) -> tuple[qml.operation.Operator, float] | tuple[int, tuple[int, ...], int, float]:
    r"""Decompose an SO(3) matrix into Matsumoto-Amano normal form.

    A Matsumoto-Amano normal form - :math:`(T | \epsilon) (HT | SHT)^* \mathcal{C}`, consists of a rightmost
    Clifford operator, followed by any number of syllables of the form ``HT`` or ``SHT``, followed by an
    optional syllable ``T`` [`arXiv:1312.6584 <https://arxiv.org/abs/1312.6584>`_\ ].

    Args:
        op (SO3Matrix): The SO(3) matrix to decompose.
        compressed (bool): If ``True``, the output will be a tuple containing information about the decomposition
            in terms of bits and indices. If ``False``, the output will be a list of all gates in the decomposition.
            Default is ``False``.

    Returns:
        Tuple[qml.operation.Operator, float] | tuple[int, tuple[int, ...], int, float]: The decomposition of the SO(3) matrix into Matsumoto-Amano normal forms and acquired global phase.
    """
    parity_transforms = _parity_transforms()
    clifford_elements = _clifford_group_to_SO3()

    so3_op = deepcopy(op)

    # The following use lemmas from arXiv:1312.6584.
    decomposition, rep_bits, g_phase = [], [], 0.0
    while (parity_vec := tuple(so3_op.parity_vec)) != (1, 1, 1):  # Fig. 1 and Lemma 6.3
        so3_val, op_gate, op_phase = parity_transforms[parity_vec]  # Lemma 4.10
        so3_op = so3_val @ so3_op  # Lemma 6.4
        rep_bits.append(1 if parity_vec == (2, 0, 2) else 0)
        decomposition.extend(op_gate)
        g_phase += op_phase

    cl_index = -1
    for clifford_index, (clifford_gate, clifford_data) in enumerate(clifford_elements.items()):
        if clifford_data[0] == so3_op:
            decomposition.extend(clifford_gate)
            cl_index = clifford_index
            g_phase += clifford_data[1]
            break

    if not compressed:
        return decomposition, g_phase

    t_bit = int(decomposition[0] == qml.T(0))
    c_bit = max(0, cl_index)

    return (t_bit, tuple(rep_bits[t_bit:]), c_bit, g_phase)
