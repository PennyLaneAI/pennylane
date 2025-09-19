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
from math import pi as PI

import pennylane as qml
from pennylane.ops.op_math.decompositions.rings import _SQRT2, DyadicMatrix, SO3Matrix, ZOmega

is_jax = True
try:
    import jax.numpy as jnp
except (ModuleNotFoundError, ImportError):  # pragma: no cover
    is_jax = False


@lru_cache
def _clifford_keys_unwired() -> list:
    """Returns a list of standard Clifford gate sequences (without wires).

    Returns:
        list[tuple[~pennylane.operation.Operation]]: Clifford gate sequences.
    """
    # fmt: off
    I, X, Y, Z = qml.I, qml.X, qml.Y, qml.Z
    H, S, Sd = qml.H, qml.S, qml.adjoint(qml.S)

    return [
        (I,), (H,), (S,), (X,), (Y,), (Z,), (Sd,),
        (H, S), (H, Z), (H, Sd), (S, H),
        (S, X), (S, Y), (Z, H), (Sd, H),
        (S, H, S), (S, H, Z), (S, H, Sd),
        (Z, H, S), (Z, H, Z), (Z, H, Sd),
        (Sd, H, S), (Sd, H, Z), (Sd, H, Sd),
    ]


@lru_cache
def _clifford_group_to_SO3() -> dict:
    """Maps each single-qubit Clifford gate sequence to its corresponding SO(3) matrix.

    Uses `clifford_keys_unwired` for gate sequence definitions.
    """
    # Get gate sequences from shared source
    gate_sequences = _clifford_keys_unwired()

    # Apply wire=0 to the gate sequences
    gate_sequences = [tuple(g(0) for g in seq) for seq in gate_sequences]

    # Corresponding SU(2) DyadicMatrix representations
    # These are the Clifford group elements with :math:`\{−1, 0, 1\}` as their matrix entries.
    su2_matrices = [
        DyadicMatrix(ZOmega(d=1), ZOmega(), ZOmega(), ZOmega(d=1)),
        -DyadicMatrix(ZOmega(b=1), ZOmega(b=1), ZOmega(b=1), ZOmega(b=-1), k=1),
        DyadicMatrix(ZOmega(a=-1), ZOmega(), ZOmega(), ZOmega(c=1)),
        DyadicMatrix(ZOmega(), ZOmega(b=-1), ZOmega(b=-1), ZOmega()),
        DyadicMatrix(ZOmega(), ZOmega(d=-1), ZOmega(d=1), ZOmega()),
        DyadicMatrix(ZOmega(b=-1), ZOmega(), ZOmega(), ZOmega(b=1)),
        DyadicMatrix(ZOmega(c=-1), ZOmega(), ZOmega(), ZOmega(a=1)),
        DyadicMatrix(ZOmega(c=-1), ZOmega(a=-1), ZOmega(c=-1), ZOmega(a=1), k=1),
        DyadicMatrix(ZOmega(d=1), ZOmega(d=-1), ZOmega(d=1), ZOmega(d=1), k=1),
        DyadicMatrix(ZOmega(a=-1), ZOmega(c=-1), ZOmega(a=-1), ZOmega(c=1), k=1),
        DyadicMatrix(ZOmega(c=-1), ZOmega(c=-1), ZOmega(a=-1), ZOmega(a=1), k=1),
        DyadicMatrix(ZOmega(), ZOmega(c=-1), ZOmega(a=-1), ZOmega()),
        DyadicMatrix(ZOmega(), ZOmega(a=1), ZOmega(c=1), ZOmega()),
        DyadicMatrix(ZOmega(d=1), ZOmega(d=1), ZOmega(d=-1), ZOmega(d=1), k=1),
        DyadicMatrix(ZOmega(a=-1), ZOmega(a=-1), ZOmega(c=-1), ZOmega(c=1), k=1),
        DyadicMatrix(ZOmega(d=1), ZOmega(b=1), ZOmega(b=1), ZOmega(d=1), k=1),
        DyadicMatrix(ZOmega(a=-1), ZOmega(a=1), ZOmega(c=1), ZOmega(c=1), k=1),
        DyadicMatrix(ZOmega(b=-1), ZOmega(d=-1), ZOmega(d=1), ZOmega(b=1), k=1),
        DyadicMatrix(ZOmega(a=-1), ZOmega(c=1), ZOmega(a=1), ZOmega(c=1), k=1),
        DyadicMatrix(ZOmega(b=-1), ZOmega(b=1), ZOmega(b=1), ZOmega(b=1), k=1),
        DyadicMatrix(ZOmega(c=-1), ZOmega(a=1), ZOmega(c=1), ZOmega(a=1), k=1),
        DyadicMatrix(ZOmega(b=-1), ZOmega(d=1), ZOmega(d=-1), ZOmega(b=1), k=1),
        DyadicMatrix(ZOmega(c=-1), ZOmega(c=1), ZOmega(a=1), ZOmega(a=1), k=1),
        DyadicMatrix(ZOmega(d=1), ZOmega(b=-1), ZOmega(b=-1), ZOmega(d=1), k=1),
    ]

    # Zip the sequences with SO(3) matrices
    return {gate: SO3Matrix(su2) for gate, su2 in zip(gate_sequences, su2_matrices)}


def _clifford_gates_to_SU2() -> dict:
    r"""Returns a dictionary mapping single-qubit Clifford group elements to their corresponding SU(2) matrices
    and global phase scaled by :math:`\pi^{-1}`."""
    I, X, Y, Z = qml.I(0), qml.X(0), qml.Y(0), qml.Z(0)
    H, S, Sd = qml.H(0), qml.S(0), qml.adjoint(qml.S(0))
    return {
        I: (DyadicMatrix(ZOmega(d=1), ZOmega(), ZOmega(), ZOmega(d=1)), 0.0),
        H: (-DyadicMatrix(ZOmega(b=1), ZOmega(b=1), ZOmega(b=1), ZOmega(b=-1), k=1), 0.5),
        S: (DyadicMatrix(ZOmega(a=-1), ZOmega(), ZOmega(), ZOmega(c=1)), 0.25),
        X: (DyadicMatrix(ZOmega(), ZOmega(b=-1), ZOmega(b=-1), ZOmega()), 0.5),
        Y: (DyadicMatrix(ZOmega(), ZOmega(d=-1), ZOmega(d=1), ZOmega()), 0.5),
        Z: (DyadicMatrix(ZOmega(b=-1), ZOmega(), ZOmega(), ZOmega(b=1)), 0.5),
        Sd: (DyadicMatrix(ZOmega(c=-1), ZOmega(), ZOmega(), ZOmega(a=1)), 0.75),
    }


@lru_cache
def _parity_transforms() -> dict:
    r"""Returns information required to perform the parity transforms used in the _ma_normal_form.

    Returns:
        dict: A dictionary mapping parity vectors to tuples containing:
            - The SO(3) matrix representation of its inverse.
            - The PennyLane gate representation of the transformation.
            - The global phase scaled by :math:`\pi^{-1}` of the transformation.

    Developer Notes:

        - The following dictionary maps the keys in the Matsumoto-Amano normal form to their:
            1. SO(3) matrix representation (useful for obtaining the parity vector)
            2. SO(3) matrix representation of their inverse (useful of reversing the operation)
            3. PennyLane gate representation (useful for capturing normal form output)
            4. Global phase difference from PL operators scaled by :math:`\pi^{-1}` (read below).

        - We use the exact matrix representation of the T-gate instead of its SU(2) representation,
          as representing :math:`\exp(\pm i\pi/8)` in the `ZOmega` ring seems non trivial. We take
          this into account when computing the global phase in (4), where we compute the global
          phase by checking the difference between the PL-gates implementation of the keys from
          the `DyadicMatrix` implementation.
    """
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
            0.0,
        ),
        "HT": (
            SO3Matrix(DyadicMatrix(ZOmega(d=1), ZOmega(c=1), ZOmega(d=1), ZOmega(c=-1), k=1)),
            SO3Matrix(DyadicMatrix(ZOmega(d=1), ZOmega(d=1), ZOmega(a=-1), ZOmega(a=1), k=1)),
            (qml.H(0), qml.T(0)),
            1.5,
        ),
        "SHT": (
            SO3Matrix(DyadicMatrix(ZOmega(d=1), ZOmega(c=1), ZOmega(b=1), ZOmega(a=-1), k=1)),
            SO3Matrix(DyadicMatrix(ZOmega(d=1), ZOmega(b=-1), ZOmega(a=-1), ZOmega(c=1), k=1)),
            (qml.S(0), qml.H(0), qml.T(0)),
            1.25,
        ),
    }
    return {
        tuple(so3_1.parity_vec): (so3_2, gate, phase)
        for (so3_1, so3_2, gate, phase) in transform_ops.values()
    }


def _ma_normal_form(op: SO3Matrix, compressed=False, upper_bounded_size=None):
    r"""Decompose an SO(3) matrix into Matsumoto-Amano normal form.

    A Matsumoto-Amano normal form - :math:`(T | \epsilon) (HT | SHT)^* \mathcal{C}`, consists of a rightmost
    Clifford operator, followed by any number of syllables of the form ``HT`` or ``SHT``, followed by an
    optional syllable ``T`` [`arXiv:1312.6584 <https://arxiv.org/abs/1312.6584>`_\ ].

    Args:
        op (SO3Matrix): The SO(3) matrix to decompose.
        compressed (bool): If ``True``, the output will be a tuple containing information about the decomposition
            in terms of bits and indices. If ``False``, the output will be a list of all gates in the decomposition.
            Default is ``False``.
        upper_bounded_size (int): The maximum number of syllables to return. Default is ``None``.
            Since JAX arrays are static, we need to specify the maximum size of the output array.

    Returns:
        Tuple[qml.operation.Operator, float] | tuple[tuple[int, tuple[int, ...], int], float]: The decomposition of the SO(3) matrix into Matsumoto-Amano normal forms and acquired global phase.
    """
    parity_transforms = _parity_transforms()
    clifford_so3s = _clifford_group_to_SO3()
    clifford_su2s = _clifford_gates_to_SU2()

    so3_op = deepcopy(op)

    # The following use lemmas from arXiv:1312.6584.
    a, c, k = ZOmega(d=1), ZOmega(), 0  # Useful for global phase tracking.
    decomposition, rep_bits, g_phase = [], [], 0.0
    while (parity_vec := tuple(so3_op.parity_vec)) != (1, 1, 1):  # Fig. 1 and Lemma 6.3
        so3_val, op_gate, op_phase = parity_transforms[parity_vec]  # Lemma 4.10
        so3_op = so3_val @ so3_op  # Lemma 6.4
        decomposition.extend(op_gate)
        g_phase += op_phase

        if parity_vec == (2, 2, 0):  # T
            c = c * ZOmega(c=1)
            rep_bits.append(0)
        elif parity_vec == (0, 2, 2):  # HT
            a, c, k = ZOmega(b=-1) * (a + c), ZOmega(a=-1) * (a - c), k + 1
            rep_bits.append(0)
        else:  # SHT
            ic = ZOmega(b=1) * c
            a, c, k = ZOmega(c=-1) * (a + ic), ZOmega(b=-1) * (a - ic), k + 1
            rep_bits.append(1)

    cl_index = -1
    for clifford_index, (clifford_ops, clifford_so3) in enumerate(clifford_so3s.items()):
        if clifford_so3 == so3_op:
            cl_index = clifford_index
            for clf_op in clifford_ops:
                decomposition.append(clf_op)
                su2, gp = clifford_su2s[clf_op]
                a, c, k = su2.a * a + su2.b * c, su2.c * a + su2.d * c, k + su2.k
                g_phase -= gp
            break

    # Extract the global phase from the decomposition from the
    # tracked elements (`a` and `c`) of the Dyadic matrix.
    su2mat = op.matrix
    g_angle = -qml.math.angle(complex(su2mat.a) / complex(a) * _SQRT2 ** (k - su2mat.k))
    g_phase = g_angle / PI - g_phase

    if not compressed:
        return decomposition, g_phase

    if not is_jax:
        raise ImportError(
            "QJIT mode requires JAX. Please install it with `pip install jax jaxlib`."
        )  # pragma: no cover

    t_bit = jnp.int32(int(decomposition[0] == qml.T(0)))
    c_bit = jnp.int32(max(0, cl_index))
    syllable_sequence = jnp.array(rep_bits[t_bit:], dtype=jnp.int32)

    # If the upper_bounded_size is specified, we need to pad the syllable_sequence with -1s.
    if upper_bounded_size is not None and compressed:
        size = upper_bounded_size
        # If the upper_bounded_size is smaller than actual size, raise an error.
        # This is not supposed to happen, if it does,
        # check upper_bounded_size calculation in rs_decomposition function.
        if size < syllable_sequence.shape[0]:
            raise ValueError(
                f"The upper_bounded_size is smaller than the actual size of the syllable sequence. "
                f"Upper bounded size: {size}, Actual size: {syllable_sequence.shape[0]}."
            )  # pragma: no cover
        syllable_sequence = syllable_sequence[:size]
        pad_len = size - syllable_sequence.shape[0]
        if pad_len > 0:
            syllable_sequence = jnp.pad(syllable_sequence, (0, pad_len), constant_values=-1)

    return ((t_bit, syllable_sequence, c_bit), g_phase)
