# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Contains transforms and helpers functions for decomposing arbitrary two-qubit
unitary operations into elementary gates.
"""
import pennylane as qml
from pennylane import math

from .single_qubit_unitary import zyz_decomposition

# This gate E is called the "magic basis". It can be used to convert between
# SO(4) and SU(2) x SU(2). For A in SO(4), E A E^\dag is in SU(2) x SU(2).
E = np.array([[1, 1j, 0, 0], [0, 0, 1j, 1], [0, 0, 1j, -1], [1, -1j, 0, 0]]) / np.sqrt(2)

Et = qml.math.T(E)
Edag = qml.math.conj(Et)


def _perm_matrix_from_sequence(seq):
    """Construct a permutation matrix based on a provided permutation of integers.

    This is used in the two-qubit unitary decomposition to permute a set of
    simultaneous eigenvalues/vectors into the same order.
    """
    mat = qml.math.zeros((4, 4))

    for row_idx in range(4):
        mat[row_idx, seq[row_idx]] = 1

    return mat


def _convert_to_su4(U):
    r"""Check unitarity of a 4x4 matrix and convert it to :math:`SU(4)` if possible.

    Args:
        U (array[complex]): A matrix, presumed to be :math:`4 \times 4` and unitary.

    Returns:
        array[complex]: A :math:`4 \times 4` matrix in :math:`SU(4)` that is
        equivalent to U up to a global phase.
    """
    # Check unitarity
    if not math.allclose(math.dot(U, math.T(math.conj(U))), math.eye(4), atol=1e-7):
        raise ValueError("Operator must be unitary.")

    # Compute the determinant
    det = math.linalg.det(U)

    # Convert to SU(4) if it's not close to 1
    if not math.allclose(det, 1.0):
        exp_angle = -1j * math.cast_like(math.angle(det), 1j) / 4
        U = math.cast_like(U, exp_angle) * math.exp(exp_angle)

    return U


def _select_rotation_angles(U):
    r"""Choose the rotation angles of RZ, RY in the two-qubit decomposition.
    They are chosen as per Proposition V.1 in quant-ph/0308033 and are based
    on the phases of the eigenvalues of :math:`E^\dagger gamma(U) E`, where

    .. math::

        \gamma(U) = U (Y \otimes Y) U^T (Y \otimes Y)

    where :math:`Y` is the Pauli :math:`y`, or equivalently,

    .. math::

        \gamma(U) = U E E^T U^T E E^T

    since :math:`EE^T = Y \otimes Y`.
    """

    gammaU = qml.math.linalg.multi_dot([Edag, U, E, Et, qml.math.T(U), E, Et, E])
    evs = qml.math.linalg.eigvals(gammaU)
    x, y, z = qml.math.angle(evs[0]), qml.math.angle(evs[1]), qml.math.angle(evs[2])

    # The rotation angles can be computed as follows (any three eigenvalues can be used)
    alpha = (x + y) / 2
    beta = (x + z) / 2
    delta = (y + z) / 2

    return alpha, beta, delta


def _su2su2_to_tensor_products(U):
    """Given a matrix U = A \otimes B in SU(2) x SU(2), extract the two SU(2)
    operations A and B.

    This process has been described in detail in the Appendix of Coffey & Deiotte
    https://link.springer.com/article/10.1007/s11128-009-0156-3
    """

    return A, B


def _extract_su2su2_prefactors(U, V):
    """U, V are SU(4) matrices for which there exists A, B, C, D such that
    (C \otimes D) V (A \otimes B) = U. The problem is to find A, B, C, D in SU(2)
    in an analytic and fully differentiable manner.

    This decomposition is possible when U and V are in the same double coset of
    SU(4), meaning that there exists G, H in SO(4) s.t. G V H = U. This is
    guaranteed by how the eigenvalues of U were used to construct V.
    """

    # A lot of the work here happens in the magic basis. Essentially, we
    # don't look explicitly at U = G V H, but rather at
    #     E^\dagger U E = G E^\dagger V E H
    # so that we can recover
    #     U = (E G E^\dagger) V (E H E^\dagger) = (A \otimes B) V (C \otimes D).
    # There is some math in the paper explaining how when we define U in this way,
    # we can simultaneously diagonalize functions of U and V to ensure they are
    # in the same coset and recover the decomposition.

    u = qml.math.linalg.multi_dot([Edag, U, E])
    v = qml.math.linalg.multi_dot([Edag, V, E])

    uuT = qml.math.dot(u, qml.math.T(u))
    vvT = qml.math.dot(v, qml.math.T(v))

    # First, we find a matrix p in SO(4) s.t. p^T u u^T p is diagonal.
    ev_p, p = qml.math.linalg.eig(uuT)
    p_diag = ev_p.copy()

    # We also do this for v, i.e., find q in SO(4) s.t. q^T v v^T q is diagonal.
    ev_q, q = qml.math.linalg.eig(vvT)
    q_diag = ev_q.copy()

    # In order for the decompositions to be equal, we have to permute their
    # eigenvalues. Do so by sorting the eigenvalues in both p and q, then
    # reshuffling them to match.
    p_diag.real[np.abs(p_diag.real) < 1e-8] = 0
    p_diag.imag[np.abs(p_diag.imag) < 1e-8] = 0
    q_diag.real[np.abs(q_diag.real) < 1e-8] = 0
    q_diag.imag[np.abs(q_diag.imag) < 1e-8] = 0

    p_order = qml.math.argsort(p_diag)
    q_order = qml.math.argsort(q_diag)

    new_p_eigvals = p_diag[[p_order]]
    new_q_eigvals = q_diag[[q_order]]

    # Given the new order, get the permutation matrices needed to reshuffle the
    # columns of p and q.
    p_perm = _perm_matrix_from_sequence(p_order)
    q_perm = _perm_matrix_from_sequence(q_order)

    #  After the shuffling below, we will have that p^T u u^T p = q^T v v^T q.
    p = qml.math.linalg.multi_dot([p, qml.math.T(p_perm)])
    q = qml.math.linalg.multi_dot([q, qml.math.T(q_perm)])

    # This means there exist p, q in SO(4) such that p^T u u^T p = q^T v v^T q.
    # Then (v^\dag q p^T u)(v^\dag q p^T U)^T = I.
    # So we can set G = p q^T, H = v^\dag q p^T u to obtain G v H = u.
    G = qml.math.dot(p, qml.math.T(q))
    H = qml.math.linalg.multi_dot([qml.math.conj(qml.math.T(v)), q, qml.math.T(p), u])

    # These are still in SO(4) though - we want to convert things into SU(2) x SU(2)
    # so use the entangler. Since u = E^\dagger U E and v = E^\dagger V E where U, V
    # are the target matrices, we can reshuffle as in the
    #     U = (E G E^\dagger) V (E H E^\dagger) = (A \otimes B) V (C \otimes D)
    # where A, B, C, D are in SU(2) x SU(2).
    AB = qml.math.linalg.multi_dot([E, G, Edag])
    CD = qml.math.linalg.multi_dot([E, H, Edag])

    # Now, we just need to extract the constituent tensor products.
    A, B = _su2su2_to_tensor_products(AB)
    C, D = _su2su2_to_tensor_products(CD)

    # Return the four single-qubit operations.
    return A, B, C, D


def two_qubit_decomposition(U, wires):
    r"""Recover the decomposition of a two-qubit matrix :math:`U` in terms of
    elementary operations.

    The work of https://arxiv.org/abs/quant-ph/0308033 presents a fixed-form
    decomposition of :math:`U` in terms of single-qubit gates and CNOTs. Multiple such
    decompositions are possible (by choosing two of {``RX``, ``RY``, ``RZ``}),
    here we choose the ``RY``, ``RZ`` case (fig. 2 in the above) to match with
    the default decomposition of the single-qubit ``Rot`` operations as ``RZ RY
    RZ``. The form of the decomposition is:

     0: -C--X--RZ(d)--C---------X--A-|
     1: -D--C--RY(b)--X--RY(a)--C--B-|

    where :math:`A, B, C, D` are :math:`SU(2)` gates.

    Args:
        U (tensor): A 4 x 4 unitary matrix.
        wires (Union[Wires, Sequence[int] or int]): The wire on which to apply the operation.

    Returns:
        list[qml.Operation]: A list of operations that represent the decomposition
        of the matrix U.
    """

    # First, we note that this method works only for SU(4) gates, meaning that
    # we need to compute rescale the matrix by its determinant.
    U = _convert_to_su4(U)

    # Next, we can choose the angles of the RZ / RY rotations.
    # See documentation within the function used below.
    alpha, beta, delta = _select_rotation_angles(U)
    print(f"Angles are {alpha}, {beta}, {delta}\n")

    # This gives us the full interior portion of the decomposition
    interior_decomp = [
        qml.CNOT(wires=[wires[1], wires[0]]),
        qml.RZ(delta, wires=wires[0]),
        qml.RY(beta, wires=wires[1]),
        qml.CNOT(wires=[wires[0], wires[1]]),
        qml.RY(alpha, wires=wires[1]),
        qml.CNOT(wires=[wires[1], wires[0]]),
    ]

    # We need the matrix representation of this interior part, V, in order to decompose
    # U = (A \otimes B) V (C \otimes D)
    CNOT10 = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])
    CNOT01 = qml.CNOT(wires=[0, 1]).matrix
    RZd = qml.RZ(delta, wires=0).matrix
    RYb = qml.RY(beta, wires=0).matrix
    RYa = qml.RY(alpha, wires=0).matrix

    V = qml.math.linalg.multi_dot(
        [CNOT10, np.kron(np.eye(2), RYa), CNOT01, np.kron(RZd, RYb), CNOT10]
    )

    # Now we need to find the four SU(2) operations A, B, C, D
    A, B, C, D = _extract_su2su2_prefactors(U, V)

    # Since this gives us their unitary form, we need to decompose them as well.
    A_ops = zyz_decomposition(A, wires[0])
    B_ops = zyz_decomposition(B, wires[1])
    C_ops = zyz_decomposition(C, wires[0])
    D_ops = zyz_decomposition(D, wires[1])

    # Return the full decomposition
    return C_ops + D_ops + interior_decomp + A_ops + B_ops
