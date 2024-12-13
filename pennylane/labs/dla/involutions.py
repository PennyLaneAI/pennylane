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
from functools import singledispatch

# pylint: disable=missing-function-docstring
from typing import Optional, Union

import numpy as np

import pennylane as qml
from pennylane import Y, Z, matrix
from pennylane.operation import Operator
from pennylane.pauli import PauliSentence


def khaneja_glaser_involution(op: Union[np.ndarray, PauliSentence, Operator], wire: int = None):
    r"""Khaneja-Glaser involution, which is a type-:func:`~AIII` Cartan involution with p=q.

    Args:
        op (Union[np.ndarray, PauliSentence, Operator]): Input operator
        wire (int): Qubit wire on which to perform Khaneja-Glaser decomposition

    Returns:
        bool: Whether ``op`` should go to the even or odd subspace of the decomposition

    .. seealso:: :func:`~cartan_decomp`

    See `Khaneja and Glaser (2000) <https://arxiv.org/abs/quant-ph/0010100>`__ for reference.

    **Example**

    Let us first perform a single Khaneja-Glaser decomposition of
    :math:`\mathfrak{g} = \mathfrak{su}(8)`, i.e. the Lie algebra of all Pauli words on 3 qubits.

    >>> g = list(qml.pauli.pauli_group(3)) # u(8)
    >>> g = g[1:] # remove identity
    >>> g = [op.pauli_rep for op in g]


    We perform the first iteration on the first qubit. We use :func:`~cartan_decomp`.

    >>> from functools import partial
    >>> from pennylane.labs.dla import cartan_decomp, khaneja_glaser_involution
    >>> k0, m0 = cartan_decomp(g, partial(khaneja_glaser_involution, wire=0))
    >>> print(f"First iteration, AIII: {len(k0)}, {len(m0)}")
    First iteration, AIII: 31, 32
    >>> assert qml.labs.dla.check_cartan_decomp(k0, m0) # check Cartan commutation relations

    We see that we split the 63-dimensional algebra :math:`\mathfrak{su}(8)` into a 31-dimensional
    and a 32-dimensional subspace, and :func:`~check_cartan_decomp` verified that they satisfy
    the commutation relations of a Cartan decomposition.

    To expand on this example, we continue process recursively on the :math:`\mathfrak{k}_0`
    subalgebra. Before we can apply the Khaneja-Glaser (type AIII) decomposition a second time,
    we need to a) remove the :math:`\mathfrak{u}(1)` center from
    :math:`\mathfrak{k}_0=\mathfrak{su}(4)\oplus\mathfrak{su}(4)\oplus\mathfrak{u}(1)` to make it
    semisimple and b) perform a different Cartan decomposition, which is sometimes termed class B.

    >>> center_k0 = qml.center(k0, pauli=True) # Compute center of k0
    >>> k0_semi = [op for op in k0 if op not in center_k0] # Remove center from k0
    >>> print(f"Removed operators {center_k0}, new length: {len(k0_semi)}")
    Removed operators [1.0 * Z(0)], new length: 30

    >>> from pennylane.labs.dla import ClassB
    >>> k1, m1 = cartan_decomp(k0_semi, partial(ClassB, wire=0))
    >>> assert qml.labs.dla.check_cartan_decomp(k1, m1)
    >>> print(f"First iteration, class B: {len(k1)}, {len(m1)}")
    First iteration, class B: 15, 15

    Now we arrived at the subalgebra :math:`\mathfrak{k}_1=\mathfrak{su}(4)` and can perform the
    next iteration of the recursive decomposition.

    >>> k2, m2 = cartan_decomp(k1, partial(khaneja_glaser_involution, wire=1))
    >>> assert qml.labs.dla.check_cartan_decomp(k2, m2)
    >>> print(f"Second iteration, AIII: {len(k2)}, {len(m2)}")
    Second iteration, AIII: 7, 8

    We can follow up on this by removing the center from :math:`\mathfrak{k}_1` again, and
    applying a final class B decomposition:

    >>> center_k2 = qml.center(k2, pauli=True)
    >>> k2_semi = [op for op in k2 if op not in center_k2]
    >>> print(f"Removed operators {center_k2}, new length: {len(k2_semi)}")
    Removed operators [1.0 * Z(1)], new length: 6

    >>> k3, m3 = cartan_decomp(k2_semi, partial(ClassB, wire=1))
    >>> assert qml.labs.dla.check_cartan_decomp(k3, m3)
    >>> print(f"Second iteration, class B: {len(k3)}, {len(m3)}")
    Second iteration, class B: 3, 3

    This concludes our example of the Khaneja-Glaser recursive decomposition, as we arrived at
    easy to implement single-qubit operations.
    """
    if wire is None:
        raise ValueError(
            "Please specify the wire for the Khaneja-Glaser involution via "
            "functools.partial(khaneja_glaser_involution, wire=wire)"
        )
    if isinstance(op, np.ndarray):
        p = q = op.shape[0] // 2
    elif isinstance(op, (PauliSentence, Operator)):
        p = q = 2 ** (len(op.wires) - 1)
    else:
        raise ValueError(f"Can't infer p and q from operator of type {type(op)}.")
    return AIII(op, p, q, wire)


# Canonical involutions
# see https://arxiv.org/abs/2406.04418 appendix C

# matrices


def int_log2(x):
    """Compute the integer closest to log_2(x)."""
    return int(np.round(np.log2(x)))


def is_qubit_case(p, q):
    """Return whether p and q are the same and a power of 2."""
    return p == q and 2 ** int_log2(p) == p


def J(n, wire=None):
    """This is the standard choice for the symplectic transformation operator.
    For an :math:`N`-qubit system (:math:`n=2^N`), it equals :math:`Y_0`."""
    N = int_log2(n)
    if 2**N == n:
        if wire is None:
            wire = 0
        return Y(wire).matrix(wire_order=range(N + 1))
    if wire is not None:
        raise ValueError("The wire argument is only supported for n=2**N for some integer N.")
    zeros = np.zeros((n, n))
    eye = np.eye(n)
    return np.block([[zeros, -1j * eye], [1j * eye, zeros]])


def Ipq(p, q, wire=None):
    """This is the canonical transformation operator for AIII and BDI Cartan
    decompositions. For an :math:`N`-qubit system (:math:`n=2^N`) and
    :math:`p=q=n/2`, it equals :math:`Z_0`."""
    # If p = q and they are a power of two, use Pauli representation
    if is_qubit_case(p, q):
        if wire is None:
            wire = 0
        return Z(wire).matrix(wire_order=range(int_log2(p) + 1))
    if wire is not None:
        raise ValueError("The wire argument is only supported for p=q=2**N for some integer N.")
    return np.block([[np.eye(p), np.zeros((p, q))], [np.zeros((q, p)), -np.eye(q)]])


def Kpq(p, q, wire=None):
    """This is the canonical transformation operator for CII Cartan
    decompositions. For an :math:`N`-qubit system (:math:`n=2^N`) and
    :math:`p=q=n/2`, it equals :math:`Z_1`."""
    # If p = q and they are a power of two, use Pauli representation
    if is_qubit_case(p, q):
        if wire is None:
            wire = 1
        return Z(wire).matrix(wire_order=range(int_log2(p) + 1))
    if wire is not None:
        raise ValueError("The wire argument is only supported for p=q=2**N for some integer N.")
    zeros = np.zeros((p + q, p + q))
    d = np.diag(np.concatenate([np.ones(p), -np.ones(q)]))
    KKm = np.block([[d, zeros], [zeros, d]])
    return KKm


def AI(op: Union[np.ndarray, PauliSentence, Operator]) -> bool:
    r"""Canonical Cartan decomposition of type AI, given by :math:`\theta: x \mapsto x^\ast`.

    .. note:: Note that we work with Hermitian
        operators internally, so that the input will be multiplied by :math:`i` before
        evaluating the involution.

    Args:
        op (Union[np.ndarray, PauliSentence, Operator]): Operator on which the involution is
        evaluated and for which the parity under the involution is returned.

    Returns:
        bool: Whether or not the input operator (times :math:`i`) is in the eigenspace of the
        involution :math:`\theta` with eigenvalue :math:`+1`.
    """
    return _AI(op)


@singledispatch
def _AI(op):  # pylint:disable=unused-argument
    r"""Default implementation of the canonical form of the AI involution
    :math:`\theta: x \mapsto x^\ast`.
    """
    raise NotImplementedError(f"Involution not implemented for operator {op} of type {type(op)}")


@_AI.register(np.ndarray)
def _AI_matrix(op: np.ndarray) -> bool:
    r"""Matrix implementation of the canonical form of the AI involution
    :math:`\theta: x \mapsto x^\ast`.
    """
    op = op * 1j
    return np.allclose(op, op.conj())


@_AI.register(PauliSentence)
def _AI_ps(op: PauliSentence) -> bool:
    r"""PauliSentence implementation of the canonical form of the AI involution
    :math:`\theta: x \mapsto x^\ast`.
    """
    parity = []
    for pw in op:
        result = sum(el == "Y" for el in pw.values())
        parity.append(bool(result % 2))

    # only makes sense if parity is the same for all terms
    assert all(parity) or not any(parity)
    return parity[0]


@_AI.register(Operator)
def _AI_op(op: Operator) -> bool:
    r"""Operator implementation of the canonical form of the AI involution
    :math:`\theta: x \mapsto x^\ast`.
    """
    return _AI_ps(op.pauli_rep)


def AII(op: Union[np.ndarray, PauliSentence, Operator], wire: Optional[int] = None) -> bool:
    r"""Canonical Cartan decomposition of type AII, given by :math:`\theta: x \mapsto Y_0 x^\ast Y_0`.

    .. note:: Note that we work with Hermitian
        operators internally, so that the input will be multiplied by :math:`i` before
        evaluating the involution.

    Args:
        op (Union[np.ndarray, PauliSentence, Operator]): Operator on which the involution is
            evaluated and for which the parity under the involution is returned.
        wire (int): The wire on which the Pauli-:math:`Y` operator acts to implement the
            involution. Will default to ``0`` if ``None``.

    Returns:
        bool: Whether or not the input operator (times :math:`i`) is in the eigenspace of the
        involution :math:`\theta` with eigenvalue :math:`+1`.
    """
    return _AII(op, wire)


@singledispatch
def _AII(op, wire=None):  # pylint:disable=unused-argument
    r"""Default implementation of the canonical form of the AII involution
    :math:`\theta: x \mapsto Y_0 x^\ast Y_0`.
    """
    raise NotImplementedError(f"Involution not implemented for operator {op} of type {type(op)}")


@_AII.register(np.ndarray)
def _AII_matrix(op: np.ndarray, wire: Optional[int] = None) -> bool:
    r"""Matrix implementation of the canonical form of the AII involution
    :math:`\theta: x \mapsto Y_0 x^\ast Y_0`.
    """
    op = op * 1j

    y = J(op.shape[-1] // 2, wire=wire)
    return np.allclose(op, y @ op.conj() @ y)


@_AII.register(PauliSentence)
def _AII_ps(op: PauliSentence, wire: Optional[int] = None) -> bool:
    r"""PauliSentence implementation of the canonical form of the AII involution
    :math:`\theta: x \mapsto Y_0 x^\ast Y_0`.
    """
    if wire is None:
        wire = 0
    parity = []
    for pw in op:
        result = sum(el == "Y" for el in pw.values()) + (pw.get(wire, "I") in "XZ")
        parity.append(bool(result % 2))

    # only makes sense if parity is the same for all terms
    assert all(parity) or not any(parity)
    return parity[0]


@_AII.register(Operator)
def _AII_op(op: Operator, wire: Optional[int] = None) -> bool:
    r"""Operator implementation of the canonical form of the AII involution
    :math:`\theta: x \mapsto Y_0 x^\ast Y_0`.
    """
    return _AII_ps(op.pauli_rep, wire)


def AIII(
    op: Union[np.ndarray, PauliSentence, Operator],
    p: int = None,
    q: int = None,
    wire: Optional[int] = None,
) -> bool:
    r"""Canonical Cartan decomposition of type AIII, given by :math:`\theta: x \mapsto I_{p,q} x I_{p,q}`.

    The matrix :math:`I_{p,q}` is given by

    .. math::
        I_{p,q}=\text{diag}(\underset{p \text{times}}{\underbrace{1, \dots 1}},
        \underset{q \text{times}}{\underbrace{-1, \dots -1}}).

    For :math:`p=q=2^N` for some integer :math:`N`, we have :math:`I_{p,q}=Z_0`.

    .. note:: Note that we work with Hermitian operators internally, so that the input will be
        multiplied by :math:`i` before evaluating the involution.

    Args:
        op (Union[np.ndarray, PauliSentence, Operator]): Operator on which the involution is
            evaluated and for which the parity under the involution is returned.
        p (int): Dimension of first subspace.
        q (int): Dimension of second subspace.
        wire (int): The wire on which the Pauli-:math:`Z` operator acts to implement the
            involution. Will default to ``0`` if ``None``.

    Returns:
        bool: Whether or not the input operator (times :math:`i`) is in the eigenspace of the
        involution :math:`\theta` with eigenvalue :math:`+1`.
    """
    if p is None or q is None:
        raise ValueError(
            "please specify p and q for the involution via functools.partial(AIII, p=p, q=q)"
        )
    return _AIII(op, p, q, wire)


@singledispatch
def _AIII(op, p=None, q=None, wire=None):  # pylint:disable=unused-argument
    r"""Default implementation of the canonical form of the AIII involution
    :math:`\theta: x \mapsto I_{p,q} x I_{p,q}`.
    """
    raise NotImplementedError(f"Involution not implemented for operator {op} of type {type(op)}")


@_AIII.register(np.ndarray)
def _AIII_matrix(op: np.ndarray, p: int = None, q: int = None, wire: Optional[int] = None) -> bool:
    r"""Matrix implementation of the canonical form of the AIII involution
    :math:`\theta: x \mapsto I_{p,q} x I_{p,q}`.
    """
    op = op * 1j

    z = Ipq(p, q, wire=wire)
    return np.allclose(op, z @ op @ z)


@_AIII.register(PauliSentence)
def _AIII_ps(op: PauliSentence, p: int = None, q: int = None, wire: Optional[int] = None) -> bool:
    r"""PauliSentence implementation of the canonical form of the AIII involution
    :math:`\theta: x \mapsto I_{p,q} x I_{p,q}`.
    """
    if is_qubit_case(p, q):

        if wire is None:
            wire = 0
        parity = [pw.get(wire, "I") in "IZ" for pw in op]

        # only makes sense if parity is the same for all terms
        assert all(parity) or not any(parity)
        return parity[0]

    # If it is not a qubit case, use the matrix representation
    return _AIII_matrix(matrix(op, wire_order=sorted(op.wires)), p, q, wire)


@_AIII.register(Operator)
def _AIII_op(op: Operator, p: int = None, q: int = None, wire: Optional[int] = None) -> bool:
    r"""Operator implementation of the canonical form of the AIII involution
    :math:`\theta: x \mapsto I_{p,q} x I_{p,q}`.
    """
    return _AIII_ps(op.pauli_rep, p, q, wire)


def BDI(
    op: Union[np.ndarray, PauliSentence, Operator],
    p: int = None,
    q: int = None,
    wire: Optional[int] = None,
) -> bool:
    r"""Canonical Cartan decomposition of type BDI, given by :math:`\theta: x \mapsto I_{p,q} x I_{p,q}`.

    The matrix :math:`I_{p,q}` is given by

    .. math::
        I_{p,q}=\text{diag}(\underset{p \text{times}}{\underbrace{1, \dots 1}},
        \underset{q \text{times}}{\underbrace{-1, \dots -1}}).

    For :math:`p=q=2^N` for some integer :math:`N`, we have :math:`I_{p,q}=Z_0`.

    .. note:: Note that we work with Hermitian operators internally, so that the input will be
        multiplied by :math:`i` before evaluating the involution.


    Args:
        op (Union[np.ndarray, PauliSentence, Operator]): Operator on which the involution is
            evaluated and for which the parity under the involution is returned.
        p (int): Dimension of first subspace.
        q (int): Dimension of second subspace.
        wire (int): The wire on which the Pauli-:math:`Z` operator acts to implement the
            involution. Will default to ``0`` if ``None``.

    Returns:
        bool: Whether or not the input operator (times :math:`i`) is in the eigenspace of the
        involution :math:`\theta` with eigenvalue :math:`+1`.
    """
    return AIII(op, p, q, wire)


def CI(op: Union[np.ndarray, PauliSentence, Operator]) -> bool:
    r"""Canonical Cartan decomposition of type CI, given by :math:`\theta: x \mapsto x^\ast`.

    .. note:: Note that we work with Hermitian
        operators internally, so that the input will be multiplied by :math:`i` before
        evaluating the involution.

    Args:
        op (Union[np.ndarray, PauliSentence, Operator]): Operator on which the involution is
        evaluated and for which the parity under the involution is returned.

    Returns:
        bool: Whether or not the input operator (times :math:`i`) is in the eigenspace of the
        involution :math:`\theta` with eigenvalue :math:`+1`.
    """
    return AI(op)


def CII(
    op: Union[np.ndarray, PauliSentence, Operator],
    p: int = None,
    q: int = None,
    wire: Optional[int] = None,
) -> bool:
    r"""Canonical Cartan decomposition of type CII, given by :math:`\theta: x \mapsto K_{p,q} x K_{p,q}`.

    The matrix :math:`K_{p,q}` is given by

    .. math::

        K_{p,q}=\text{diag}(
        \underset{p \text{times}}{\underbrace{1, \dots 1}},
        \underset{q \text{times}}{\underbrace{-1, \dots -1}},
        \underset{p \text{times}}{\underbrace{1, \dots 1}},
        \underset{q \text{times}}{\underbrace{-1, \dots -1}},
        ).

    For :math:`p=q=2^N` for some integer :math:`N`, we have :math:`K_{p,q}=Z_1`.

    .. note:: Note that we work with Hermitian operators internally, so that the input will be
        multiplied by :math:`i` before evaluating the involution.

    Args:
        op (Union[np.ndarray, PauliSentence, Operator]): Operator on which the involution is
            evaluated and for which the parity under the involution is returned.
        p (int): Dimension of first subspace.
        q (int): Dimension of second subspace.
        wire (int): The wire on which the Pauli-:math:`Z` operator acts to implement the
            involution. Will default to ``1`` if ``None``.

    Returns:
        bool: Whether or not the input operator (times :math:`i`) is in the eigenspace of the
        involution :math:`\theta` with eigenvalue :math:`+1`.
    """
    if p is None or q is None:
        raise ValueError(
            "please specify p and q for the involution via functools.partial(CII, p=p, q=q)"
        )
    return _CII(op, p, q, wire)


@singledispatch
def _CII(op, p=None, q=None, wire=None):  # pylint:disable=unused-argument
    r"""Default implementation of the canonical form of the CII involution
    :math:`\theta: x \mapsto K_{p,q} x K_{p,q}`.
    """
    raise NotImplementedError(f"Involution not implemented for operator {op} of type {type(op)}")


@_CII.register(np.ndarray)
def _CII_matrix(op: np.ndarray, p: int = None, q: int = None, wire: Optional[int] = None) -> bool:
    r"""Matrix implementation of the canonical form of the CII involution
    :math:`\theta: x \mapsto K_{p,q} x K_{p,q}`.
    """
    op = op * 1j

    z = Kpq(p, q, wire=wire)
    return np.allclose(op, z @ op @ z)


@_CII.register(PauliSentence)
def _CII_ps(op: PauliSentence, p: int = None, q: int = None, wire: Optional[int] = None) -> bool:
    r"""PauliSentence implementation of the canonical form of the CII involution
    :math:`\theta: x \mapsto K_{p,q} x K_{p,q}`.
    """
    if is_qubit_case(p, q):

        if wire is None:
            wire = 1
        parity = [pw.get(wire, "I") in "IZ" for pw in op]

        # only makes sense if parity is the same for all terms
        assert all(parity) or not any(parity)
        return parity[0]

    # If it is not a qubit case, use the matrix representation
    return _CII(matrix(op, wire_order=sorted(op.wires)), p, q, wire)


@_CII.register(Operator)
def _CII_op(op: Operator, p: int = None, q: int = None, wire: Optional[int] = None) -> bool:
    r"""Operator implementation of the canonical form of the CII involution
    :math:`\theta: x \mapsto K_{p,q} x K_{p,q}`.
    """
    return _CII_ps(op.pauli_rep, p, q, wire)


def DIII(op: Union[np.ndarray, PauliSentence, Operator], wire: Optional[int] = None) -> bool:
    r"""Canonical Cartan decomposition of type DIII, given by :math:`\theta: x \mapsto Y_0 x Y_0`.

    Args:
        op (Union[np.ndarray, PauliSentence, Operator]): Operator on which the involution is
            evaluated and for which the parity under the involution is returned.
        wire (int): The wire on which the Pauli-:math:`Y` operator acts to implement the
            involution. Will default to ``0`` if ``None``.

    Returns:
        bool: Whether or not the input operator (times :math:`i`) is in the eigenspace of the
        involution :math:`\theta` with eigenvalue :math:`+1`.
    """
    return _DIII(op, wire)


@singledispatch
def _DIII(op, wire=None):  # pylint:disable=unused-argument
    r"""Default implementation of the canonical form of the DIII involution
    :math:`\theta: x \mapsto Y_0 x Y_0`.
    """
    raise NotImplementedError(f"Involution not implemented for operator {op} of type {type(op)}")


@_DIII.register(np.ndarray)
def _DIII_matrix(op: np.ndarray, wire: Optional[int] = None) -> bool:
    r"""Matrix implementation of the canonical form of the DIII involution
    :math:`\theta: x \mapsto Y_0 x Y_0`.
    """
    y = J(op.shape[-1] // 2, wire=wire)
    return np.allclose(op, y @ op @ y)


@_DIII.register(PauliSentence)
def _DIII_ps(op: PauliSentence, wire: Optional[int] = None) -> bool:
    r"""PauliSentence implementation of the canonical form of the DIII involution
    :math:`\theta: x \mapsto Y_0 x Y_0`.
    """
    if wire is None:
        wire = 0
    parity = [pw.get(wire, "I") in "IY" for pw in op]

    # only makes sense if parity is the same for all terms
    assert all(parity) or not any(parity)
    return parity[0]


@_DIII.register(Operator)
def _DIII_op(op: Operator, wire: Optional[int] = None) -> bool:
    r"""Operator implementation of the canonical form of the DIII involution
    :math:`\theta: x \mapsto Y_0 x Y_0`.
    """
    return _DIII_ps(op.pauli_rep, wire)


def ClassB(op: Union[np.ndarray, PauliSentence, Operator], wire: Optional[int] = None) -> bool:
    r"""Canonical Cartan decomposition of class B, given by :math:`\theta: x \mapsto Y_0 x Y_0`.

    Args:
        op (Union[np.ndarray, PauliSentence, Operator]): Operator on which the involution is
            evaluated and for which the parity under the involution is returned.
        wire (int): The wire on which the Pauli-:math:`Y` operator acts to implement the
            involution. Will default to ``0`` if ``None``.

    Returns:
        bool: Whether or not the input operator (times :math:`i`) is in the eigenspace of the
        involution :math:`\theta` with eigenvalue :math:`+1`.
    """
    return DIII(op, wire)


# dispatch to different input types
def even_odd_involution(op: Union[PauliSentence, np.ndarray, Operator]) -> bool:
    r"""The Even-Odd involution.

    This is defined in `quant-ph/0701193 <https://arxiv.org/abs/quant-ph/0701193>`__.
    For Pauli words and sentences, it comes down to counting non-trivial Paulis in Pauli words.

    Args:
        op ( Union[PauliSentence, np.ndarray, Operator]): Input operator

    Returns:
        bool: Boolean output ``True`` or ``False`` for odd (:math:`\mathfrak{k}`) and even parity subspace (:math:`\mathfrak{m}`), respectively

    .. seealso:: :func:`~cartan_decomp`

    **Example**

    >>> from pennylane import X, Y, Z
    >>> from pennylane.labs.dla import even_odd_involution
    >>> ops = [X(0), X(0) @ Y(1), X(0) @ Y(1) @ Z(2)]
    >>> [even_odd_involution(op) for op in ops]
    [True, False, True]

    Operators with an odd number of non-identity Paulis yield ``1``, whereas even ones yield ``0``.

    The function also works with dense matrix representations.

    >>> ops_m = [qml.matrix(op, wire_order=range(3)) for op in ops]
    >>> [even_odd_involution(op_m) for op_m in ops_m]
    [True, False, True]

    """
    return _even_odd_involution(op)


@singledispatch
def _even_odd_involution(op):  # pylint:disable=unused-argument, missing-function-docstring
    return NotImplementedError(f"Involution not defined for operator {op} of type {type(op)}")


@_even_odd_involution.register(PauliSentence)
def _even_odd_involution_ps(op: PauliSentence):
    # Generalization to sums of Paulis: check each term and assert they all have the same parity
    parity = []
    for pw in op.keys():
        parity.append(len(pw) % 2)

    # only makes sense if parity is the same for all terms, e.g. Heisenberg model
    assert all(
        parity[0] == p for p in parity
    ), f"The Even-Odd involution is not well-defined for operator {op} as individual terms have different parity"
    return bool(parity[0])


@_even_odd_involution.register(np.ndarray)
def _even_odd_involution_matrix(op: np.ndarray):
    """see Table CI in https://arxiv.org/abs/2406.04418"""
    n = int(np.round(np.log2(op.shape[-1])))
    YYY = qml.prod(*[Y(i) for i in range(n)])
    YYY = qml.matrix(YYY, range(n))

    transformed = YYY @ op.conj() @ YYY
    if np.allclose(transformed, op):
        return False
    if np.allclose(transformed, -op):
        return True
    raise ValueError(f"The Even-Odd involution is not well-defined for operator {op}.")


@_even_odd_involution.register(Operator)
def _even_odd_involution_op(op: Operator):
    """use pauli representation"""
    return _even_odd_involution_ps(op.pauli_rep)


# dispatch to different input types
def concurrence_involution(op: Union[PauliSentence, np.ndarray, Operator]) -> bool:
    r"""The Concurrence Canonical Decomposition :math:`\Theta(g) = -g^T` as a Cartan involution function.

    This is defined in `quant-ph/0701193 <https://arxiv.org/abs/quant-ph/0701193>`__.
    For Pauli words and sentences, it comes down to counting Pauli-Y operators.

    Args:
        op ( Union[PauliSentence, np.ndarray, Operator]): Input operator

    Returns:
        bool: Boolean output ``True`` or ``False`` for odd (:math:`\mathfrak{k}`) and even parity subspace (:math:`\mathfrak{m}`), respectively

    .. seealso:: :func:`~cartan_decomp`

    **Example**

    >>> from pennylane import X, Y, Z
    >>> from pennylane.labs.dla import concurrence_involution
    >>> ops = [X(0), X(0) @ Y(1), X(0) @ Y(1) @ Z(2), Y(0) @ Y(2)]
    >>> [concurrence_involution(op) for op in ops]
    [False, True, True, False]

    Operators with an odd number of ``Y`` operators yield ``1``, whereas even ones yield ``0``.

    The function also works with dense matrix representations.

    >>> ops_m = [qml.matrix(op, wire_order=range(3)) for op in ops]
    >>> [even_odd_involution(op_m) for op_m in ops_m]
    [False, True, True, False]

    """
    return _concurrence_involution(op)


@singledispatch
def _concurrence_involution(op):
    return NotImplementedError(f"Involution not defined for operator {op} of type {type(op)}")


@_concurrence_involution.register(PauliSentence)
def _concurrence_involution_pauli(op: PauliSentence):
    # Generalization to sums of Paulis: check each term and assert they all have the same parity
    parity = []
    for pw in op.keys():
        result = sum(1 if el == "Y" else 0 for el in pw.values())
        parity.append(result % 2)

    # only makes sense if parity is the same for all terms, e.g. Heisenberg model
    assert all(
        parity[0] == p for p in parity
    ), f"The concurrence canonical decomposition is not well-defined for operator {op} as individual terms have different parity"
    return bool(parity[0])


@_concurrence_involution.register(Operator)
def _concurrence_involution_operator(op: Operator):
    return _concurrence_involution_matrix(op.matrix())


@_concurrence_involution.register(np.ndarray)
def _concurrence_involution_matrix(op: np.ndarray):
    if np.allclose(op, -op.T):
        return True
    if np.allclose(op, op.T):
        return False
    raise ValueError(
        f"The concurrence canonical decomposition is not well-defined for operator {op}"
    )
