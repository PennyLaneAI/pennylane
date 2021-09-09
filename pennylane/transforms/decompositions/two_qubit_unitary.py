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
import warnings

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

        \gamma(U) = (E^\dag U E) (E^\dag U E)^T,

    and follows the arguments of this paper: https://arxiv.org/abs/quant-ph/0308045.
    """
    u = math.dot(Edag, math.dot(U, E))
    gammaU = math.dot(u, math.T(u))

    trace = math.trace(gammaU)

    # For the case with 3 CNOTs, the trace is a non-zero complex number
    # with both real and imaginary parts.
    num_cnots = 3

    # Case: 0 CNOTs (tensor product), the trace is +/- 4
    if math.allclose(trace, 4) or math.allclose(trace, -4):
        num_cnots = 0
    # Case: 1 CNOT, the trace is 0
    elif math.allclose(trace, 0.0):
        num_cnots = 1
    # Case: 2 CNOTs, the trace has only a real part
    elif math.allclose(math.imag(trace), 0):
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
    if not math.allclose(A[0, 0], 0.0, atol=1e-6):
        B = C1 / math.cast_like(A[0, 0], 1j)
    else:
        B = C2 / math.cast_like(A[0, 1], 1j)

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

    u = math.dot(math.cast_like(Edag, V), math.dot(U, math.cast_like(E, V)))
    v = math.dot(math.cast_like(Edag, V), math.dot(V, math.cast_like(E, V)))

    uuT = math.dot(u, math.T(u))
    vvT = math.dot(v, math.T(v))

    # First, we find a matrix p (hopefully) in SO(4) s.t. p^T u u^T p is diagonal.
    # Since uuT is complex and symmetric, both its real / imag parts share a set
    # of real-valued eigenvectors.
    if math.get_interface(u) == "tensorflow":
        ev_p, p = math.linalg.eig(math.real(uuT))
    else:
        ev_p, p = math.linalg.eig(uuT)

    # We also do this for v, i.e., find q (hopefully) in SO(4) s.t. q^T v v^T q is diagonal.
    if math.get_interface(u) == "tensorflow":
        ev_q, q = math.linalg.eig(math.real(vvT))
    else:
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
        q = math.dot(q, math.cast_like(LAST_COL_NEG, q))

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
    AB = math.dot(math.cast_like(E, V), math.dot(G, math.cast_like(Edag, V)))
    CD = math.dot(math.cast_like(E, V), math.dot(H, math.cast_like(Edag, V)))

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


def _decomposition_3_cnots(U, wires):
    r"""The most general form of this decomposition is U = (A \otimes B) V (C \otimes D),
    where V is as depicted in the circuit below:
     -U- = -C--X--RZ(d)--C---------X--A-
     -U- = -D--C--RY(b)--X--RY(a)--C--B-
    """

    # First we add a SWAP as per v1 of arXiv:0308033, which helps with some
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

    # To get consistent results in all interfaces, sort the eigenvalues first
    angles = qml.math.sort([math.angle(ev) for ev in evs])
    x, y, z = angles[0], angles[1], angles[2]

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

    RZd = qml.RZ(math.cast_like(delta, 1j), wires=wires[0]).matrix
    RYb = qml.RY(beta, wires=wires[0]).matrix
    RYa = qml.RY(alpha, wires=wires[0]).matrix

    V_mats = [CNOT10, math.kron(RZd, RYb), CNOT01, math.kron(math.eye(2), RYa), CNOT10, SWAP]

    V = math.convert_like(math.eye(4), U)

    for mat in V_mats:
        V = math.dot(math.cast_like(mat, U), V)

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
    r"""Recover the decomposition of a two-qubit unitary :math:`U` in terms of
    elementary operations.

    The work of `Shende, Markov, and Bullock (2003)
    <https://arxiv.org/abs/quant-ph/0308033>`__ presents a fixed-form
    decomposition of :math:`U` in terms of single-qubit gates and
    CNOTs. Multiple such decompositions are possible (by choosing two of ``{RX,
    RY, RZ}``). Here we choose the ``RY``, ``RZ`` case (fig. 2 in the above) to
    match with the default decomposition of the single-qubit :class:`~.Rot` operations
    as ``RZ RY RZ``. The most general form of the decomposition is:

    .. figure:: ../../_static/two_qubit_decomposition.svg
        :align: center
        :width: 70%
        :target: javascript:void(0);


    where :math:`A, B, C, D` are :math:`SU(2)` gates.

    .. note::

        Currently, the decomposition is only implemented for two cases: when the
        provided unitary can be expressed with no CNOTs (i.e., it is a tensor
        product of two single-qubit operations), or when it can be expressed
        using exactly 3, as in the graphic above. It generally works when the
        provided unitary is sampled at random from :math:`U(4)`. The case of 1-
        and 2-CNOTs will be implemented at a later time.

    This decomposition can be applied automatically to all valid two-qubit
    :class:`~.QubitUnitary` operations by applying the :func:`~pennylane.transforms.unitary_to_rot`
    transform.

    Args:
        U (tensor): A 4 x 4 unitary matrix.
        wires (Union[Wires, Sequence[int] or int]): The wires on which to apply the operation.

    Returns:
        list[Operation]: A list of operations that represent the decomposition
        of the matrix U.

    **Example**

    Suppose we create a random element of :math:`U(4)`, and would like to decompose it
    into elementary gates in our circuit.

    >>> from scipy.stats import unitary_group
    >>> U = unitary_group.rvs(4)
    >>> U
    array([[-0.29113625+0.56393527j,  0.39546712-0.14193837j,
             0.04637428+0.01311566j, -0.62006741+0.18403743j],
           [-0.45479211+0.25978444j, -0.52737418-0.5549423j ,
            -0.23429057+0.10728103j,  0.16061807-0.21769762j],
           [-0.4501231 +0.04065613j, -0.25558662+0.38209554j,
            -0.04143479-0.56598134j,  0.12983673+0.49548507j],
           [ 0.23899902+0.24800931j,  0.03374589-0.15784319j,
             0.24898226-0.73975147j,  0.0269508 -0.49534518j]])

    We can compute its decompositon like so:

    >>> decomp = qml.transforms.two_qubit_decomposition(np.array(U), wires=[0, 1])
    >>> decomp
    [Rot(tensor(-1.69488788, requires_grad=True), tensor(1.06701916, requires_grad=True), tensor(0.41190893, requires_grad=True), wires=[0]),
     Rot(tensor(1.57705621, requires_grad=True), tensor(2.42621204, requires_grad=True), tensor(2.57842249, requires_grad=True), wires=[1]),
     CNOT(wires=[1, 0]),
     RZ(0.4503059654281863, wires=[0]),
     RY(-0.8872497960867665, wires=[1]),
     CNOT(wires=[0, 1]),
     RY(-1.6472464849278514, wires=[1]),
     CNOT(wires=[1, 0]),
     Rot(tensor(2.93239686, requires_grad=True), tensor(1.8725019, requires_grad=True), tensor(0.0418203, requires_grad=True), wires=[1]),
     Rot(tensor(-3.78673588, requires_grad=True), tensor(2.03936812, requires_grad=True), tensor(-2.46956972, requires_grad=True), wires=[0])]

    """
    # First, we note that this method works only for SU(4) gates, meaning that
    # we need to rescale the matrix by its determinant.
    U = _convert_to_su4(U)

    # The next thing we will do is compute the number of CNOTs needed, as this affects
    # the form of the decomposition.
    num_cnots = _compute_num_cnots(U)

    if num_cnots == 0:
        decomp = _decomposition_0_cnots(U, wires)
    elif num_cnots == 3:
        decomp = _decomposition_3_cnots(U, wires)
    else:
        decomp = [qml.QubitUnitary(U, wires=wires)]

        warnings.warn(
            "Decomposition for numerically-supplied 2-qubit unitaries requiring "
            "1 or 2 CNOTs is not currently supported. Your unitary matrix\n"
            f"U = {U}\nwhich requires {num_cnots} CNOT(s) will not be decomposed.",
            UserWarning,
        )

    return decomp
