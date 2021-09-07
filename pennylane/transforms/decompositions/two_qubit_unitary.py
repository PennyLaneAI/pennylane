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
Et = E.T
Edag = Et.conj()

# Helpful to have static copies of these since they are needed in a few places.
CNOT01 = qml.CNOT(wires=[0, 1]).matrix
CNOT10 = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])
SWAP = qml.SWAP(wires=[0, 1]).matrix

LAST_COL_NEG = np.diag([1, 1, 1, -1])  # used to negate the last column of a matrix


def _convert_to_su4(U):
    r"""Check unitarity of a 4x4 matrix and convert it to :math:`SU(4)` if the determinant is not 1.

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
        U = math.cast_like(U, det) * math.exp(exp_angle)

    return U


def _compute_num_cnots(U):
    r"""Compute the number of CNOTs required to implement U. This is based on
    the trace of

    .. math::

        \gamma(U) = (E^\dag U E) (E^\dag U E)^T.
    """
    u = math.dot(Edag, math.dot(U, E))
    gammaU = math.dot(u, math.T(u))

    trace = qml.math.trace(gammaU)

    # For the case with 3 CNOTs, the trace is a non-zero complex number
    # with both real and imaginary parts.
    num_cnots = 3

    # Case: 0 CNOTs (tensor product), the trace is +/- 4
    if qml.math.allclose(trace, 4) or qml.math.allclose(trace, -4):
        num_cnots = 0
    # Case: 1 CNOT, the trace is 0
    elif qml.math.allclose(trace, 0.0):
        num_cnots = 1
    # Case: 2 CNOTs, the trace has only a real part
    elif qml.math.allclose(qml.math.imag(trace), 0):
        num_cnots = 2

    return num_cnots


def _su2su2_to_tensor_products(U):
    r"""Given a matrix :math:`U = A \otimes B` in SU(2) x SU(2), extract the two SU(2)
    operations A and B.

    This process has been described in detail in the Appendix of Coffey & Deiotte
    https://link.springer.com/article/10.1007/s11128-009-0156-3
    """

    # First, write A = [[a1, a2], [-a2*, a1*]], which we can do for any SU(2) element.
    # Then, A \otimes B = [[a1 B, a2 B], [-a2*B, a1*B]] = [[C1, C2], [C3, C4]]
    # where the Ci are 2x2 matrices.
    C1 = U[0:2, 0:2]
    C2 = U[0:2, 2:4]
    C3 = U[2:4, 0:2]
    C4 = U[2:4, 2:4]

    # From the definition of A \otimes B, C1 C4^\dag = a1^2 I, so we can extract a1
    C14 = math.dot(C1, math.conj(math.T(C4)))
    a1 = math.sqrt(C14[0, 0])

    # Similarly, -C2 C3^\dag = a2^2 I, so we can extract a2
    C23 = math.dot(C2, math.conj(math.T(C3)))
    a2 = math.sqrt(-C23[0, 0])

    # This gets us a1, a2 up to a sign. To resolve the sign, ensure that
    # C1 C2^dag = a1 a2* I
    C12 = math.dot(C1, math.conj(math.T(C2)))

    if not math.allclose(a1 * np.conj(a2), C12[0, 0]):
        a2 *= -1

    # Construct A
    A = math.stack([[a1, a2], [-math.conj(a2), math.conj(a1)]])

    # Next, extract B. Can do from any of the C, just need to be careful in
    # case one of the elements of A is 0.
    if not math.allclose(A[0, 0], 0.0, atol=1e-8):
        B = C1 / A[0, 0]
    else:
        B = C2 / A[0, 1]

    return math.convert_like(A, U), math.convert_like(B, U)


def _extract_su2su2_prefactors(U, V):
    r"""U, V are SU(4) matrices for which there exists A, B, C, D such that
    (A \otimes B) V (C \otimes D) = U. The problem is to find A, B, C, D in SU(2)
    in an analytic and fully differentiable manner.

    This decomposition is possible when U and V are in the same double coset of
    SU(4), meaning there exists G, H in SO(4) s.t. G (Edag V E) H = (Edag U
    E). This is guaranteed here by how V was constructed using the
    _select_rotation_angles method. Then, we can use the fact that E SO(4) Edag
    gives us something in SU(2) x SU(2) to give A, B, C, D.
    """

    # A lot of the work here happens in the magic basis. Essentially, we
    # don't look explicitly at some U = G V H, but rather at
    #     E^\dagger U E = G E^\dagger V E H
    # so that we can recover
    #     U = (E G E^\dagger) V (E H E^\dagger) = (A \otimes B) V (C \otimes D).
    # There is some math in the paper explaining how when we define U in this way,
    # we can simultaneously diagonalize functions of U and V to ensure they are
    # in the same coset and recover the decomposition.

    u = math.dot(math.cast_like(Edag, U), math.dot(U, math.cast_like(E, U)))
    v = math.dot(math.cast_like(Edag, V), math.dot(V, math.cast_like(E, V)))

    uuT = math.dot(u, math.T(u))
    vvT = math.dot(v, math.T(v))

    # First, we find a matrix p (hopefully) in SO(4) s.t. p^T u u^T p is diagonal.
    # Since uuT is complex and symmetric, both its real / imag parts share a set
    # of real-valued eigenvectors.
    ev_p, p = math.linalg.eig(uuT)

    # We also do this for v, i.e., find q (hopefully) in SO(4) s.t. q^T v v^T q is diagonal.
    ev_q, q = math.linalg.eig(vvT)

    # If determinant of p is not 1, it is in O(4) but not SO(4), and has
    # determinant -1. We can transform it to SO(4) by simply negating one
    # of the columns.
    if not math.allclose(math.linalg.det(p), 1.0):
        p = math.dot(p, math.cast_like(LAST_COL_NEG, p))

    new_q_order = []

    for _, eigval in enumerate(ev_p):
        are_close = [math.allclose(x, eigval) for x in ev_q]

        if any(are_close):
            new_q_order.append(math.argmax(are_close))

    # Reshuffle the columns.
    q_perm = np.identity(4)[:, np.array(new_q_order)]
    q = math.dot(q, math.cast_like(q_perm, q))

    # Depending on the sign of the permutation, it may be that q is in O(4) but
    # not SO(4). Again we can fix this by simply negating a column.
    q_in_so4 = math.allclose(math.linalg.det(q), 1.0)
    if not q_in_so4:
        q = math.dot(q, LAST_COL_NEG)

    # Now, we should have p, q in SO(4) such that p^T u u^T p = q^T v v^T q.
    # Then (v^\dag q p^T u)(v^\dag q p^T u)^T = I.
    # So we can set G = p q^T, H = v^\dag q p^T u to obtain G v H = u.
    G = math.dot(p, math.T(q))
    H = math.dot(math.conj(math.T(v)), math.dot(math.T(G), u))

    # These are still in SO(4) though - we want to convert things into SU(2) x SU(2)
    # so use the entangler. Since u = E^\dagger U E and v = E^\dagger V E where U, V
    # are the target matrices, we can reshuffle as in the docstring above,
    #     U = (E G E^\dagger) V (E H E^\dagger) = (A \otimes B) V (C \otimes D)
    # where A, B, C, D are in SU(2) x SU(2).
    AB = math.dot(E, math.dot(G, Edag))
    CD = math.dot(E, math.dot(H, Edag))

    # Now, we just need to extract the constituent tensor products.
    A, B = _su2su2_to_tensor_products(AB)
    C, D = _su2su2_to_tensor_products(CD)

    return A, B, C, D


def _decomposition_0_cnots(U, wires):
    r"""If there are no CNOTs, this is just a tensor product of two single-qubit gates.
    We can perform that decomposition directly:
     -U- = -A-
     -U- = -B-
    """
    A, B = _su2su2_to_tensor_products(U)
    A_ops = zyz_decomposition(A, wires[0])
    B_ops = zyz_decomposition(B, wires[1])
    return A_ops + B_ops


def _decomposition_1_cnot(U, wires):
    r"""If 1 CNOT is required, we can write the circuit as
     -U- = -A--C--C-
     -U- = -B--X--D-
    Note that the direction of the CNOT is irrelevant, because we could always
    flip it by applying Hadamards on both qubits on each side, and then absorb
    the Hadamards into A/B/C/D.
    """
    A, B = _su2su2_to_tensor_products(U)
    A_ops = zyz_decomposition(A, wires[0])
    B_ops = zyz_decomposition(B, wires[1])
    return A_ops + B_ops


def _decomposition_2_cnots(U, wires):
    r"""If 2 CNOTs are required, we can write the circuit as
     -U- = -A--X--RZ(d)--X--C-
     -U- = -B--C--RX(p)--C--D-
    We need to find the angles for the Z and X rotations such that the inner
    part has the same spectrum as U, and then we can recover A, B, C, D.
    """
    # Compute the rotation angles
    u = math.dot(Edag, math.dot(U, E))
    gammaU = math.dot(u, math.T(u))
    evs, _ = math.linalg.eig(gammaU)

    # These choices are based on Proposition 5.1 of
    # https://web.eecs.umich.edu/~imarkov/pubs/conf/spie04-2qubits.pdf

    # Need to find the angle that is not the conjugate of this one
    x = math.angle(evs[0])
    y = math.angle(evs[1])

    # If it was the conjugate, grab a different eigenvalue.
    if math.allclose(x, -y):
        y = math.angle(evs[2])

    delta = (x + y) / 2
    phi = (x - y) / 2

    # This is the "interior" part of the decomposition
    interior_decomp = [
        qml.CNOT(wires=[wires[1], wires[0]]),
        qml.RZ(delta, wires=wires[0]),
        qml.RX(phi, wires=wires[1]),
        qml.CNOT(wires=[wires[1], wires[0]]),
    ]

    RZd = qml.RZ(delta, wires=0).matrix
    RXp = qml.RX(phi, wires=0).matrix

    # We need the matrix representation of this interior part, V, in order to
    # decompose U = (A \otimes B) V (C \otimes D)
    V = qml.math.dot(CNOT10, qml.math.dot(qml.math.kron(RZd, RXp), CNOT10))

    # Now we find the A, B, C, D in SU(2), and return the decomposition
    A, B, C, D = _extract_su2su2_prefactors(U, V)

    A_ops = zyz_decomposition(A, wires[0])
    B_ops = zyz_decomposition(B, wires[1])
    C_ops = zyz_decomposition(C, wires[0])
    D_ops = zyz_decomposition(D, wires[1])

    return C_ops + D_ops + interior_decomp + A_ops + B_ops


def _decomposition_3_cnots(U, wires):
    r"""The most general form of this decomposition is U = (A \otimes B) V (C \otimes D),
    where V is as depicted in the circuit below:
     -U- = -C--X--RZ(d)--C---------X--A-
     -U- = -D--C--RY(b)--X--RY(a)--C--B-
    """

    # First we add a SWAP as per v1 of 0308033, which helps with some
    # rearranging of gates in the decomposition (it will cancel out the fact
    # that we need to add a SWAP to fix the determinant in another part later).
    swap_U = np.exp(1j * np.pi / 4) * math.dot(math.cast_like(SWAP, U), U)

    # Choose the rotation angles of RZ, RY in the two-qubit decomposition.
    # They are chosen as per Proposition V.1 in quant-ph/0308033 and are based
    # on the phases of the eigenvalues of :math:`E^\dagger \gamma(U) E`, where
    #    \gamma(U) = (E^\dag U E) (E^\dag U E)^T.
    # The rotation angles can be computed as follows (any three eigenvalues can be used)
    u = math.dot(Edag, math.dot(swap_U, E))
    gammaU = math.dot(u, math.T(u))
    evs, _ = math.linalg.eig(gammaU)

    x, y, z = math.angle(evs[0]), math.angle(evs[1]), math.angle(evs[2])

    # Compute functions of the eigenvalues; there are different options in v1
    # vs. v3 of the paper, I'm not entirely sure why. This is the version from v3.
    alpha = (x + y) / 2
    beta = (x + z) / 2
    delta = (z + y) / 2

    # This is the interior portion of the decomposition circuit
    interior_decomp = [
        qml.CNOT(wires=[wires[1], wires[0]]),
        qml.RZ(delta, wires=wires[0]),
        qml.RY(beta, wires=wires[1]),
        qml.CNOT(wires=[wires[0], wires[1]]),
        qml.RY(alpha, wires=wires[1]),
        qml.CNOT(wires=[wires[1], wires[0]]),
    ]

    # We need the matrix representation of this interior part, V, in order to
    # decompose U = (A \otimes B) V (C \otimes D)
    #
    # Looking at the decomposition above, V has determinant -1 (because there
    # are 3 CNOTs, each with determinant -1). The relationship between U and V
    # requires that both are in SU(4), so we add a SWAP after to V. We will see
    # how this gets fixed later.
    #
    # -V- = -X--RZ(d)--C---------X--SWAP-
    # -V- = -C--RY(b)--X--RY(a)--C--SWAP-

    RZd = qml.RZ(math.cast_like(delta, 1j), wires=0).matrix
    RYb = qml.RY(beta, wires=0).matrix
    RYa = qml.RY(alpha, wires=0).matrix

    V_mats = [
        SWAP,
        CNOT10,
        math.kron(math.eye(2), RYa),
        CNOT01,
        math.kron(RZd, RYb),
        CNOT10,
    ]

    V = math.convert_like(math.eye(4), U)

    for mat in V_mats:
        V = math.dot(V, math.cast_like(mat, U))

    # Now we need to find the four SU(2) operations A, B, C, D
    A, B, C, D = _extract_su2su2_prefactors(swap_U, V)

    # At this point, we have the following:
    # -U-SWAP- = --C--X-RZ(d)-C-------X-SWAP--A
    # -U-SWAP- = --D--C-RZ(b)-X-RY(a)-C-SWAP--B
    #
    # Using the relationship that SWAP(A \otimes B) SWAP = B \otimes A,
    # -U-SWAP- = --C--X-RZ(d)-C-------X--B--SWAP-
    # -U-SWAP- = --D--C-RZ(b)-X-RY(a)-C--A--SWAP-
    #
    # Now the SWAPs cancel, giving us the desired decomposition
    # (up to a global phase).
    # -U- = --C--X-RZ(d)-C-------X--B--
    # -U- = --D--C-RZ(b)-X-RY(a)-C--A--

    A_ops = zyz_decomposition(A, wires[1])
    B_ops = zyz_decomposition(B, wires[0])
    C_ops = zyz_decomposition(C, wires[0])
    D_ops = zyz_decomposition(D, wires[1])

    # Return the full decomposition
    return C_ops + D_ops + interior_decomp + A_ops + B_ops


def two_qubit_decomposition(U, wires):
    r"""Recover the decomposition of a two-qubit matrix :math:`U` in terms of
    elementary operations.

    The work of `Shende, Markov, and Bullock (2003)
    <https://arxiv.org/abs/quant-ph/0308033>`__ presents a fixed-form
    decomposition of :math:`U` in terms of single-qubit gates and
    CNOTs. Multiple such decompositions are possible (by choosing two of ``{RX,
    RY, RZ}``). Here we choose the ``RY``, ``RZ`` case (fig. 2 in the above) to
    match with the default decomposition of the single-qubit ``Rot`` operations
    as ``RZ RY RZ``. The most general form of the decomposition is:

    .. figure:: ../../_static/two_qubit_decomposition.svg
        :align: center
        :width: 100%
        :target: javascript:void(0);

    where :math:`A, B, C, D` are :math:`SU(2)` gates.

    However, it may also be the case that the circuit can be implemented using
    fewer than 3 CNOT gates; this condition is checked for, and simpler decompositions
    are applied where possible.

    Args:
        U (tensor): A 4 x 4 unitary matrix.
        wires (Union[Wires, Sequence[int] or int]): The wires on which to apply the operation.

    Returns:
        list[Operation]: A list of operations that represent the decomposition
        of the matrix U.

    """
    # First, we note that this method works only for SU(4) gates, meaning that
    # we need to rescale the matrix by its determinant.
    U = _convert_to_su4(U)

    # The next thing we will do is compute the number of CNOTs needed, as this affects
    # the form of the decomposition.
    num_cnots = _compute_num_cnots(U)

    if num_cnots == 0:
        decomp = _decomposition_0_cnots(U, wires)
    elif num_cnots == 1:
        decomp = _decomposition_1_cnot(U, wires)
    elif num_cnots == 2:
        decomp = _decomposition_2_cnots(U, wires)
    else:
        decomp = _decomposition_3_cnots(U, wires)

    return decomp
