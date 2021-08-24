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
from pennylane import numpy as np
from pennylane import math

from .single_qubit_unitary import zyz_decomposition

# This gate E is called the "magic basis". It can be used to convert between
# SO(4) and SU(2) x SU(2). For A in SO(4), E A E^\dag is in SU(2) x SU(2).
E = np.array([[1, 1j, 0, 0], [0, 0, 1j, 1], [0, 0, 1j, -1], [1, -1j, 0, 0]]) / np.sqrt(2)

Et = qml.math.T(E)
Edag = qml.math.conj(Et)

odd_perms = [
    [0, 1, 2, 3],
    [0, 2, 3, 1],
    [0, 3, 1, 2],
    [1, 0, 3, 2],
    [1, 2, 0, 3],
    [1, 3, 2, 0],
    [2, 1, 3, 0],
    [2, 0, 1, 3],
    [2, 3, 0, 1],
    [3, 0, 2, 1],
    [3, 2, 1, 0],
    [3, 1, 0, 2]
]


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
    if not qml.math.allclose(det, 1.0):
        exp_angle = -1j * math.cast_like(math.angle(det), 1j) / 4
        U = math.cast_like(U, exp_angle) * math.exp(exp_angle)
        
    return U


def _select_rotation_angles(U):
    r"""Choose the rotation angles of RZ, RY in the two-qubit decomposition.
    They are chosen as per Proposition V.1 in quant-ph/0308033 and are based
    on the phases of the eigenvalues of :math:`E^\dagger \gamma(U) E`, where

    .. math::

        \gamma(U) = U (Y \otimes Y) U^T (Y \otimes Y),

    and :math:`Y` is the Pauli :math:`Y` operation. Equivalently,

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

    if not qml.math.isclose(qml.math.linalg.det(U), 1.0):
        print(f"Original U = {U}\n")
        #U = U * np.exp(1j * np.pi/4)
        #U = qml.math.dot(U, qml.math.T(U))
        #print(f"New U = {U}\n")
        
        #print(f"New determinant is {qml.math.linalg.det(U)}")
        #print(f"Is it unitary {np.round(qml.math.dot(U, qml.math.T(qml.math.conj(U))), decimals=4)}\n")

    # First, write A = [[a1, a2], [-a2*, a1*]], which we can do for any SU(2) element.
    # Then, A \otimes B = [[a1 B, a2 B], [-a2*B, a1*B]] = [[C1, C2], [C3, C4]]
    # where the Ci are 2x2 matrices.
    C1 = U[0:2, 0:2]
    C2 = U[0:2, 2:4]
    C3 = U[2:4, 0:2]
    C4 = U[2:4, 2:4]

    # From the definition of A \otimes B, C1 C4^\dag = a1^2 I
    C14 = qml.math.dot(C1, qml.math.conj(qml.math.T(C4)))
    a1 = qml.math.sqrt(C14[0, 0])

    C12 = qml.math.dot(C1, qml.math.conj(qml.math.T(C2)))

    if qml.math.isclose(a1, 0.0, atol=1e-6):
        # If the a1 we got was close to 0, try extracting it from elsewhere
        # C2 C3^\dag = -a2^2 I
        C23 = -qml.math.dot(C2, qml.math.conj(qml.math.T(C3)))

        a2 = qml.math.sqrt(-1 * C23[0, 0])
        a1 = C12[0, 0] / qml.math.conj(a2)
    else:
        a2 = qml.math.conj(C12[0, 0] / a1)
    
    # Construct A
    A = qml.math.stack([[a1, a2], [-qml.math.conj(a2), qml.math.conj(a1)]])

    # Next, extract B. Can do from any of the C, just need to be careful in
    # case one of the elements of A is 0. 
    if not qml.math.isclose(A[0, 0], 0.0, atol=1e-10):
        B = C1 / A[0, 0]
    else:
        B = C2 / A[0, 1]

    return A, B


def _extract_su2su2_prefactors(U, V):
    """U, V are SU(4) matrices for which there exists A, B, C, D such that
    (C \otimes D) V (A \otimes B) = U. The problem is to find A, B, C, D in SU(2)
    in an analytic and fully differentiable manner.

    This decomposition is possible when U and V are in the same double coset of
    SU(4), meaning that there exists G, H in SO(4) s.t. G V H = U. This is
    guaranteed by how the eigenvalues of U were used to construct V.
    """

    det_V = qml.math.linalg.det(V)
    V = _convert_to_su4(V)

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

    print(f"Initial value of v = {v}")

    print(f"det v = {qml.math.linalg.det(qml.math.conj(qml.math.T(v)))}")
    
    uuT = qml.math.dot(u, qml.math.T(u))
    vvT = qml.math.dot(v, qml.math.T(v))

    # First, we find a matrix p in SO(4) s.t. p^T u u^T p is diagonal.
    ev_p, p = qml.math.linalg.eig(uuT)
    p_diag = ev_p.copy()

    # We also do this for v, i.e., find q in SO(4) s.t. q^T v v^T q is diagonal.
    ev_q, q = qml.math.linalg.eig(vvT)
    q_diag = ev_q.copy()

    print(f"p in O(4): {qml.math.allclose(qml.math.dot(p, qml.math.T(p)), np.eye(4))}")
    print(f"p det = {qml.math.linalg.det(p)}\n")
    print(f"q in O(4): {qml.math.allclose(qml.math.dot(q, qml.math.T(q)), np.eye(4))}")
    print(f"q det = {qml.math.linalg.det(q)}\n")
    
    p_in_so4 = qml.math.isclose(qml.math.linalg.det(p), 1.0)
    q_in_so4 = qml.math.isclose(qml.math.linalg.det(q), 1.0)

    p_product = qml.math.linalg.multi_dot([qml.math.T(p), uuT, p])
    q_product = qml.math.linalg.multi_dot([qml.math.T(q), vvT, q])
    
    print("Initial diagonalization results")
    print(f"pT uuT p = {np.round(p_product, decimals=4)}\n")
    print(f"qT vvT q = {np.round(q_product, decimals=4)}\n")
    

    # If determinants are not 1, need to negate one of the columns. This ensures
    # things are in SO(4), and simply negates one of the eigenvalues.
    if not p_in_so4:
       p[:, -1] = -p[:, -1]

    if not q_in_so4:
       q[:, -1] = -q[:, -1]

    print(f"p in O(4): {qml.math.allclose(qml.math.dot(p, qml.math.T(p)), np.eye(4))}")
    print(f"p det = {qml.math.linalg.det(p)}\n")
    print(f"q in O(4): {qml.math.allclose(qml.math.dot(q, qml.math.T(q)), np.eye(4))}")
    print(f"q det = {qml.math.linalg.det(q)}\n")

    p_product = qml.math.linalg.multi_dot([qml.math.T(p), uuT, p])
    q_product = qml.math.linalg.multi_dot([qml.math.T(q), vvT, q])
    
    print("Final diagonalization results")
    print(f"pT uuT p = {np.round(p_product, decimals=4)}\n")
    print(f"qT vvT q = {np.round(q_product, decimals=4)}\n")


    # Reorder the columns of p so that the eigenvalues are in the same order
    # as those of q.

    print(p_diag)
    print(q_diag)

    print(f"\nInitial q = \n{np.round(q, decimals=4)}\n")

    column_phases = [1, 1, 1, 1]
    new_q_order = []

    for idx, eigval in enumerate(p_diag):
        for phase in [1, -1, 1j, -1j]:
            are_close = [qml.math.isclose(x, phase * eigval) for x in q_diag]

            if any(are_close):
                where = qml.math.argmax(are_close)
                column_phases[where] = phase
                new_q_order.append(where)
                break

    print("\nAfter re-ordering")
    print(p_diag)
    print(q_diag[[new_q_order]] / column_phases)

    # Given the new order, get the permutation matrices needed to reshuffle the
    # columns of p and q.
    q_perm = _perm_matrix_from_sequence(new_q_order)

    print(q_perm)
    print()
    #  After the shuffling below, we will have that p^T u u^T p = q^T v v^T q.
    q = qml.math.linalg.multi_dot([q, qml.math.T(q_perm)])

    p_product = qml.math.linalg.multi_dot([qml.math.T(p), uuT, p])
    q_product = qml.math.linalg.multi_dot([qml.math.T(q), vvT, q])

    # p_product and q_product should be the same, potentially up to a phase
    # if there is a phase, let's absorb it into the matrix v.
    phase = q_product[0, 0] / p_product[0, 0]

    if not qml.math.isclose(phase, 1.0):
        print(f"Extracted an extra phase of {phase}")
        v = v / qml.math.sqrt(phase)
        vvT = vvT / phase
        print(f"New value of v is {v}")
        
    p_product = qml.math.linalg.multi_dot([qml.math.T(p), uuT, p])
    q_product = qml.math.linalg.multi_dot([qml.math.T(q), vvT, q])

    print("After phase extra, diagonal operations are now")
    print(f"pT uuT p = {np.round(p_product, decimals=4)}\n")
    print(f"qT vvT q = {np.round(q_product, decimals=4)}\n")     
        
    print(f"Final p = \n{np.round(p, decimals=4)}\n")
    print(f"p in O(4): {qml.math.allclose(qml.math.dot(p, qml.math.T(p)), np.eye(4))}")
    print(f"p det = {qml.math.linalg.det(p)}\n")
    
    print(f"Final q = \n{np.round(q, decimals=4)}\n")
    print(f"q in O(4): {qml.math.allclose(qml.math.dot(q, qml.math.T(q)), np.eye(4))}")
    print(f"q det = {qml.math.linalg.det(q)}\n")

    
    p_in_so4 = qml.math.isclose(qml.math.linalg.det(p), 1.0)
    q_in_so4 = qml.math.isclose(qml.math.linalg.det(q), 1.0)

    
    if not p_in_so4:
        print("Adjusting p to be in SO(4)")
        p[:, -1] = -p[:, -1]
 
    if not q_in_so4:
        print("Adjusting q to be in SO(4)")
        q[:, -1] = -q[:, -1]

    print(f"Final p = \n{np.round(p, decimals=4)}\n")
    print(f"p in O(4): {qml.math.allclose(qml.math.dot(p, qml.math.T(p)), np.eye(4))}")
    print(f"p det = {qml.math.linalg.det(p)}\n")
    
    print(f"Final q = \n{np.round(q, decimals=4)}\n")
    print(f"q in O(4): {qml.math.allclose(qml.math.dot(q, qml.math.T(q)), np.eye(4))}")
    print(f"q det = {qml.math.linalg.det(q)}\n")

    p_product = qml.math.linalg.multi_dot([qml.math.T(p), uuT, p])
    q_product = qml.math.linalg.multi_dot([qml.math.T(q), vvT, q])

    print("After making things SO(4) again,")
    print(f"pT uuT p = {np.round(p_product, decimals=4)}\n")
    print(f"qT vvT q = {np.round(q_product, decimals=4)}\n")     
    
    # This means there exist p, q in SO(4) such that p^T u u^T p = q^T v v^T q.
    # Then (v^\dag q p^T u)(v^\dag q p^T U)^T = I.
    # So we can set G = p q^T, H = v^\dag q p^T u to obtain G v H = u.
    G = qml.math.dot(p, qml.math.T(q))

    print(f"det v = {qml.math.linalg.det(qml.math.conj(qml.math.T(v)))}")
    print(f"det q = {qml.math.linalg.det(q)}")
    print(f"det p = {qml.math.linalg.det(p)}")
    print(f"det u = {qml.math.linalg.det(u)}")
    
    H = qml.math.sqrt(phase) * qml.math.linalg.multi_dot([qml.math.conj(qml.math.T(v)), q, qml.math.T(p), u])

    #exp_angle = -1j * math.cast_like(math.angle(det_V), 1j) / 4
    #H = math.cast_like(H, exp_angle) / math.exp(exp_angle)
    
    # Remove global phase to ensure determinant 1 (SO(4))
    #GGT = qml.math.dot(G, qml.math.T(G))
    #G = G * qml.math.sqrt(GGT[0, 0])

    HHT = qml.math.dot(H, qml.math.T(H))
    #H = H / HHT[0, 0]
    
    #print(f"GGT = {np.round(qml.math.dot(G, qml.math.T(G)), decimals=4)}\n")
    print(f"HHT = {np.round(HHT, decimals=4)}\n")
    print(f"G in O(4): {qml.math.allclose(qml.math.dot(G, qml.math.T(G)), np.eye(4))}")
    print(f"G det = {qml.math.linalg.det(G)}\n")
    print(f"H in O(4): {qml.math.allclose(qml.math.dot(H, qml.math.T(H)), np.eye(4))}")
    print(f"H det = {qml.math.linalg.det(H)}\n")

    # These are still in SO(4) though - we want to convert things into SU(2) x SU(2)
    # so use the entangler. Since u = E^\dagger U E and v = E^\dagger V E where U, V
    # are the target matrices, we can reshuffle as in the
    #     U = (E G E^\dagger) V (E H E^\dagger) = (A \otimes B) V (C \otimes D)
    # where A, B, C, D are in SU(2) x SU(2).
    AB = qml.math.linalg.multi_dot([E, G, Edag])
    CD = qml.math.linalg.multi_dot([E, H, Edag])
    
    print(f"Original U = {U}\n")
    print(f"AB V CD = {qml.math.linalg.multi_dot([AB, V, CD])}\n")

    print(f"AB = {np.round(AB, decimals=4)}\n")
    print(f"CD = {np.round(CD, decimals=4)}\n")
    
    # Now, we just need to extract the constituent tensor products.
    A, B = _su2su2_to_tensor_products(AB)
    C, D = _su2su2_to_tensor_products(CD)

    #print(f"A in the decomp func = {A}\n")
    #print(f"B in the decomp func = {B}\n")
    #print(f"C in the decomp func = {C}\n")
    #print(f"D in the decomp func = {D}\n")
    
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

    right_part = qml.math.kron(C_ops[0].matrix, D_ops[0].matrix)
    left_part = qml.math.kron(A_ops[0].matrix, B_ops[0].matrix)

    recovered_U = qml.math.linalg.multi_dot([left_part, V, right_part])
    # Return the full decomposition
    return C_ops + D_ops + interior_decomp + A_ops + B_ops
