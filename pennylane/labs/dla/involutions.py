# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Cartan involutions"""
# pylint: disable=missing-function-docstring
from typing import Union

import numpy as np

from pennylane import Y, Z
from pennylane.operation import Operator
from pennylane.pauli import PauliSentence


def khaneja_glaser_involution(op: Union[PauliSentence, Operator], wire=None):
    r"""Khaneja-Glaser involution

    .. warning:: This involution currently only works with Pauli words, either presented as PennyLane operators or :class:`~PauliSentence` instances.

    Args:
        op (PauliSentence): Input operator
        wire (int): Qubit wire on which to perform KG decomposition

    Returns:
        bool: Accordingly to whether ``op`` should go to the even or odd subspace of the decomposition

    .. seealso:: :func:`~cartan_decomposition`

    **Example**

    Let us perform a full recursive Khaneja-Glaser decomposition of :math:`\mathfrak{g} = \mathfrak{su}(8)`, i.e. the Lie algebra of all Pauli words on 3 qubits.

    >>> g = list(qml.pauli.pauli_group(3)) # su(8)
    >>> g = [_.pauli_rep for _ in g]
    >>> g = g[1:] # remove identity

    We perform the first iteration on the first qubit. We use :func:`~cartan_decomposition`.

    >>> from functools import partial
    >>> k0, m0 = cartan_decomposition(g, partial(khaneja_glaser_involution, wire=0))
    >>> print(f"First iteration: {len(k0)}, {len(m0)}")
    First iteration: 31, 32
    >>> assert qml.labs.dla.check_cartan_decomp(k0, m0) # check Cartan commutation relations of subspaces

    We continue this recursive process on the :math:`\mathfrak{k}` subalgebra with the other two wires.

    >>> k1, m1 = cartan_decomposition(k0, partial(khaneja_glaser_involution, wire=1))
    >>> assert check_cartan_decomp(k1, m1)
    >>> print(f"Second iteration: {len(k1)}, {len(m1)}")
    Second iteration: 15, 16

    >>> k2, m2 = cartan_decomposition(k1, partial(khaneja_glaser_involution, wire=2))
    >>> assert check_cartan_decomp(k2, m2)
    >>> print(f"Third iteration: {len(k2)}, {len(m2)}")
    Third iteration: 7, 8
    """
    if wire is None:
        raise ValueError(
            "please specify the ``wire`` for the Khaneja-Glaser involution via functools.partial(khaneja_glaser_involution, wire=wire)"
        )
    if isinstance(op, Operator):
        op = op.pauli_rep

    assert len(op) == 1  # no PauliSentences allowed atm
    [pw] = op  # get PauliWord
    return pw[wire] in ["I", "Z"]


# Canonical involutions
# see https://arxiv.org/pdf/2406.04418 appendix C

# matrices


def int_log2(x):
    return int(np.round(np.log2(x)))


def J(n, wire=None):
    """This is the standard choice for the symplectic transformation operator.
    For an :math:`N`-qubit system (:math:`n=2^N`), it equals :math:`iY_0`."""
    N = int_log2(n)
    if 2**N == n:
        if wire is None:
            wire = 0
        return 1j * Y(wire).matrix(wire_order=range(N + 1))
    if wire is not None:
        raise ValueError("The wire argument is only supported for n=2**N for some N." "")
    return np.block([[np.zeros((n, n)), np.eye(n)], [-np.eye(n), np.zeros((n, n))]])


def Ipq(p, q, wire=None):
    """This is the canonical transformation operator for AIII and BDI Cartan
    decompositions. For an :math:`N`-qubit system (:math:`n=2^N`) and
    :math:`p=q=n/2`, it equals :math:`-Z_0`."""
    # If p = q and they are a power of two, use Pauli representation
    if p == q and 2 ** int_log2(p) == p:
        if wire is None:
            wire = 0
        return -1 * Z(wire).matrix(wire_order=range(int_log2(p) + 1))
    if wire is not None:
        raise ValueError("The wire argument is only supported for p=q=2**N for some N." "")
    return np.block([[-np.eye(p), np.zeros((p, q))], [np.zeros((q, p)), np.eye(q)]])


def Kpq(p, q, wire=None):
    """This is the canonical transformation operator for CII Cartan
    decompositions. For an :math:`N`-qubit system (:math:`n=2^N`) and
    :math:`p=q=n/2`, it equals :math:`-Z_1`."""
    # If p = q and they are a power of two, use Pauli representation
    if p == q and 2 ** int_log2(p) == p:
        if wire is None:
            wire = 1
        return -1 * Z(wire).matrix(wire_order=range(int_log2(p) + 1))
    if wire is not None:
        raise ValueError("The wire argument is only supported for p=q=2**N for some N." "")
    KKm = np.block(
        [
            [-np.eye(p), np.zeros((p, p)), np.zeros((p, p)), np.zeros((p, p))],
            [np.zeros((q, q)), np.eye(q), np.zeros((q, q)), np.zeros((q, q))],
            [np.zeros((p, p)), np.zeros((p, p)), -np.eye(p), np.zeros((p, p))],
            [np.zeros((q, q)), np.zeros((q, q)), np.zeros((q, q)), np.eye(q)],
        ]
    )
    return KKm


def AI(op):
    """Involution for AI Cartan decomposition.
    Note that we work with Hermitian matrices internally, so that we need to multiply by
    ``1j`` to obtain a skew-Hermitian matrix, before applying the involution itself.
    """
    op = 1j * op
    return np.allclose(op, op.conj())


def AII(op, wire=None):
    """Involution for AII Cartan decomposition.
    Note that we work with Hermitian matrices internally, so that we need to multiply by
    ``1j`` to obtain a skew-Hermitian matrix, before applying the involution itself.
    """
    op = 1j * op
    JJ = J(op.shape[-1] // 2, wire=wire)
    return np.allclose(op, JJ @ op.conj() @ JJ.T)


def AIII(op, p=None, q=None, wire=None):
    """Involution for AIII Cartan decomposition.
    Note that we work with Hermitian matrices internally, so that we need to multiply by
    ``1j`` to obtain a skew-Hermitian matrix, before applying the involution itself.
    """
    op = 1j * op
    if p is None or q is None:
        raise ValueError(
            "please specify p and q for the involution via functools.partial(AIII, p=p, q=q)"
        )
    IIm = Ipq(p, q, wire=wire)
    return np.allclose(op, IIm @ op @ IIm)


def BDI(op, p=None, q=None, wire=None):
    """Involution for BDI Cartan decomposition.
    Note that we work with Hermitian matrices internally, so that we need to multiply by
    ``1j`` to obtain a skew-Hermitian matrix, before applying the involution itself.
    """
    return AIII(op, p, q, wire)


def CI(op):
    """Involution for CI Cartan decomposition.
    Note that we work with Hermitian matrices internally, so that we need to multiply by
    ``1j`` to obtain a skew-Hermitian matrix, before applying the involution itself.
    """
    return AI(op)


def CII(op, p=None, q=None, wire=None):
    """Involution for CII Cartan decomposition.
    Note that we work with Hermitian matrices internally, so that we need to multiply by
    ``1j`` to obtain a skew-Hermitian matrix, before applying the involution itself.
    """
    op = 1j * op
    if p is None or q is None:
        raise ValueError(
            "please specify p and q for the involution via functools.partial(CII, p=p, q=q)"
        )
    KKm = Kpq(p, q, wire)
    return np.allclose(op, KKm @ op @ KKm)


def DIII(op, wire=None):
    """Involution for DIII Cartan decomposition.
    Note that we work with Hermitian matrices internally, so that we need to multiply by
    ``1j`` to obtain a skew-Hermitian matrix, before applying the involution itself.
    """
    op = 1j * op
    JJ = J(op.shape[-1] // 2, wire=wire)
    return np.allclose(op, JJ @ op @ JJ.T)


def ClassB(op, wire=None):
    return DIII(op, wire)
