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
import numpy as np

import pennylane as qml
from pennylane import math

from .single_qubit_unitary import one_qubit_decomposition


###################################################################################
# Developer notes:
#
# I was not able to get this transform to be fully differentiable for unitary
# matrices that were constructed within a QNode, based on the QNode's input
# arguments. I would argue this is a fairly limited use case, but it would still
# be nice to have this eventually. Each interface fails for different reasons.
#
# - In Autograd, we obtain the AttributeError
#       'ArrayBox' object has no attribute 'conjugate'
#   for the 0-CNOT case when the zyz_decomposition function is called. In the
#   other cases, it cannot autodifferentiate through the linalg.eigvals function.
# - In Torch, it is not currently possible to autodiff through linalg.det for
#   complex values.
# - In Tensorflow, it sometimes works in limited cases (0, sometimes 1 CNOT), but
#   for others it fails without output making it hard to pinpoint the cause.
# - In JAX, we receive the TypeError:
#       Can't differentiate w.r.t. type <class 'jaxlib.xla_extension.Array'>
#
###################################################################################


# This gate E is called the "magic basis". It can be used to convert between
# SO(4) and SU(2) x SU(2). For A in SO(4), E A E^\dag is in SU(2) x SU(2).
E = np.array([[1, 1j, 0, 0], [0, 0, 1j, 1], [0, 0, 1j, -1], [1, -1j, 0, 0]]) / np.sqrt(2)
Edag = E.conj().T

# Helpful to have static copies of these since they are needed in a few places.
CNOT01 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
CNOT10 = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])
SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])

# S \otimes SX
S_SX = np.array(
    [
        [0.5 + 0.5j, 0.5 - 0.5j, 0.0 + 0.0j, 0.0 + 0.0j],
        [0.5 - 0.5j, 0.5 + 0.5j, 0.0 + 0.0j, 0.0 + 0.0j],
        [0.0 + 0.0j, 0.0 + 0.0j, -0.5 + 0.5j, 0.5 + 0.5j],
        [0.0 + 0.0j, 0.0 + 0.0j, 0.5 + 0.5j, -0.5 + 0.5j],
    ]
)

# Any two-qubit operation can be decomposed into single-qubit operations and
# at most 3 CNOTs. The number of CNOTs needed affects the form of the decomposition,
# so we separate the code into four cases.


# This constant matrix is used in the 1-CNOT decomposition
v_one_cnot = np.array(
    [
        [0.5, 0.5j, 0.5j, -0.5],
        [-0.5j, 0.5, -0.5, -0.5j],
        [-0.5j, -0.5, 0.5, -0.5j],
        [0.5, -0.5j, -0.5j, -0.5],
    ]
)

# This q is properly in SO(4) and is used in the 1-CNOT decomposition
q_one_cnot = (1 / np.sqrt(2)) * np.array(
    [[-1, 0, -1, 0], [0, 1, 0, 1], [0, 1, 0, -1], [1, 0, -1, 0]]
)


def _convert_to_su4(U):
    r"""Convert a 4x4 matrix to :math:`SU(4)`.

    Args:
        U (array[complex]): A matrix, presumed to be :math:`4 \times 4` and unitary.

    Returns:
        array[complex]: A :math:`4 \times 4` matrix in :math:`SU(4)` that is
        equivalent to U up to a global phase.
    """
    # Compute the determinant
    det = math.linalg.det(U)

    exp_angle = -1j * math.cast_like(math.angle(det), 1j) / 4
    return math.cast_like(U, det) * math.exp(exp_angle)


def _compute_num_cnots(U):
    r"""Compute the number of CNOTs required to implement a U in SU(4). This is based on
    the trace of

    .. math::

        \gamma(U) = (E^\dag U E) (E^\dag U E)^T,

    and follows the arguments of this paper: https://arxiv.org/abs/quant-ph/0308045.
    """
    u = math.dot(Edag, math.dot(U, E))
    gammaU = math.dot(u, math.T(u))
    trace = math.trace(gammaU)

    # Case: 0 CNOTs (tensor product), the trace is +/- 4
    # We need a tolerance of around 1e-7 here in order to work with the case where U
    # is specified with 8 decimal places.
    if math.allclose(trace, 4, atol=1e-7) or math.allclose(trace, -4, atol=1e-7):
        return 0

    # To distinguish between 1/2 CNOT cases, we need to look at the eigenvalues
    evs = math.linalg.eigvals(gammaU)

    sorted_evs = math.sort(math.imag(evs))

    # Case: 1 CNOT, the trace is 0, and the eigenvalues of gammaU are [-1j, -1j, 1j, 1j]
    # Checking the eigenvalues is needed because of some special 2-CNOT cases that yield
    # a trace 0.
    if math.allclose(trace, 0j, atol=1e-7) and math.allclose(sorted_evs, [-1, -1, 1, 1]):
        return 1

    # Case: 2 CNOTs, the trace has only a real part (or is 0)
    if math.allclose(math.imag(trace), 0.0, atol=1e-7):
        return 2

    # For the case with 3 CNOTs, the trace is a non-zero complex number
    # with both real and imaginary parts.
    return 3


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
    a1 = math.sqrt(math.cast_like(C14[0, 0], 1j))

    # Similarly, -C2 C3^\dag = a2^2 I, so we can extract a2
    C23 = math.dot(C2, math.conj(math.T(C3)))
    a2 = math.sqrt(-math.cast_like(C23[0, 0], 1j))

    # This gets us a1, a2 up to a sign. To resolve the sign, ensure that
    # C1 C2^dag = a1 a2* I
    C12 = math.dot(C1, math.conj(math.T(C2)))

    if not math.is_abstract(C12):
        if not math.allclose(a1 * math.conj(a2), C12[0, 0]):
            a2 *= -1
    else:
        sign_is_correct = math.allclose(a1 * math.conj(a2), C12[0, 0])
        sign = (-1) ** (sign_is_correct + 1)  # True + 1 = 2, False + 1 = 1
        a2 *= sign

    # Construct A
    A = math.stack([math.stack([a1, a2]), math.stack([-math.conj(a2), math.conj(a1)])])

    # Next, extract B. Can do from any of the C, just need to be careful in
    # case one of the elements of A is 0.
    # We use B1 unless division by 0 would cause all elements to be inf.
    use_B2 = math.allclose(A[0, 0], 0.0, atol=1e-6)
    if not math.is_abstract(A):
        B = C2 / math.cast_like(A[0, 1], 1j) if use_B2 else C1 / math.cast_like(A[0, 0], 1j)
    elif qml.math.get_interface(A) == "jax":
        B = qml.math.cond(
            use_B2,
            lambda x: C2 / math.cast_like(A[0, 1], 1j),
            lambda x: C1 / math.cast_like(A[0, 0], 1j),
            [0],  # arbitrary value for x
        )

    return math.convert_like(A, U), math.convert_like(B, U)


def _extract_su2su2_prefactors(U, V):
    r"""This function is used for the case of 2 CNOTs and 3 CNOTs. It does something
    similar as the 1-CNOT case, but there is no special form for one of the
    SO(4) operations.

    Suppose U, V are SU(4) matrices for which there exists A, B, C, D such that
    (A \otimes B) V (C \otimes D) = U. The problem is to find A, B, C, D in SU(2)
    in an analytic and fully differentiable manner.

    This decomposition is possible when U and V are in the same double coset of
    SU(4), meaning there exists G, H in SO(4) s.t. G (Edag V E) H = (Edag U
    E). This is guaranteed here by how V was constructed in both the
    _decomposition_2_cnots and _decomposition_3_cnots methods.

    Then, we can use the fact that E SO(4) Edag gives us something in SU(2) x
    SU(2) to give A, B, C, D.
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

    # Get the p and q in SO(4) that diagonalize uuT and vvT respectively (and
    # their eigenvalues). We are looking for a simultaneous diagonalization,
    # which we know exists because of how U and V were constructed. Furthermore,
    # The way we will do this is by noting that, since uuT/vvT are complex and
    # symmetric, so both their real and imaginary parts share a set of
    # real-valued eigenvectors, which are also eigenvectors of uuT/vvT
    # themselves. So we can use eigh, which orders the eigenvectors, and so we
    # are guaranteed that the p and q returned will be "in the same order".
    _, p = math.linalg.eigh(math.real(uuT) + math.imag(uuT))
    _, q = math.linalg.eigh(math.real(vvT) + math.imag(vvT))

    # If determinant of p/q is not 1, it is in O(4) but not SO(4), and has determinant
    # We can transform it to SO(4) by simply negating one of the columns.
    p = math.dot(p, math.diag([1, 1, 1, math.sign(math.linalg.det(p))]))
    q = math.dot(q, math.diag([1, 1, 1, math.sign(math.linalg.det(q))]))

    # Now, we should have p, q in SO(4) such that p^T u u^T p = q^T v v^T q.
    # Then (v^\dag q p^T u)(v^\dag q p^T u)^T = I.
    # So we can set G = p q^T, H = v^\dag q p^T u to obtain G v H = u.
    G = math.dot(math.cast_like(p, 1j), math.T(q))
    H = math.dot(math.conj(math.T(v)), math.dot(math.T(G), u))

    # These are still in SO(4) though - we want to convert things into SU(2) x SU(2)
    # so use the entangler. Since u = E^\dagger U E and v = E^\dagger V E where U, V
    # are the target matrices, we can reshuffle as in the docstring above,
    #     U = (E G E^\dagger) V (E H E^\dagger) = (A \otimes B) V (C \otimes D)
    # where A, B, C, D are in SU(2) x SU(2).
    AB = math.dot(math.cast_like(E, G), math.dot(G, math.cast_like(Edag, G)))
    CD = math.dot(math.cast_like(E, H), math.dot(H, math.cast_like(Edag, H)))

    # Now, we just need to extract the constituent tensor products.
    A, B = _su2su2_to_tensor_products(AB)
    C, D = _su2su2_to_tensor_products(CD)

    return A, B, C, D


def _decomposition_0_cnots(U, wires):
    r"""If there are no CNOTs, this is just a tensor product of two single-qubit gates.
    We can perform that decomposition directly:
     -╭U- = -A-
     -╰U- = -B-
    """
    A, B = _su2su2_to_tensor_products(U)
    A_ops = one_qubit_decomposition(A, wires[0])
    B_ops = one_qubit_decomposition(B, wires[1])
    return A_ops + B_ops


def _decomposition_1_cnot(U, wires):
    r"""If there is just one CNOT, we can write the circuit in the form
     -╭U- = -C--╭C--A-
     -╰U- = -D--╰X--B-

    To do this decomposition, first we find G, H in SO(4) such that
        G (Edag V E) H = (Edag U E)

    where V depends on the central CNOT gate, and both U, V are in SU(4). This
    is done following the methods in https://arxiv.org/abs/quant-ph/0308045.

    Once we find G and H, we can use the fact that E SO(4) Edag gives us
    something in SU(2) x SU(2) to give A, B, C, D.
    """
    # We will actually find a decomposition for the following circuit instead
    # of the original U
    # -╭U-╭SWAP- = -C--╭C-╭SWAP--B-
    # -╰U-╰SWAP- = -D--╰X-╰SWAP--A-
    # This ensures that the internal part of the decomposition has determinant 1.
    swap_U = np.exp(1j * np.pi / 4) * math.dot(math.cast_like(SWAP, U), U)

    # First let's compute gamma(u). For the one-CNOT case, uuT is always real.
    u = math.dot(math.cast_like(Edag, U), math.dot(swap_U, math.cast_like(E, U)))
    uuT = math.dot(u, math.T(u))

    # Since uuT is real, we can use eigh of its real part. eigh also orders the
    # eigenvalues in ascending order.
    _, p = math.linalg.eigh(qml.math.real(uuT))

    # Fix the determinant if necessary so that p is in SO(4)
    p = math.dot(p, math.diag([1, 1, 1, math.sign(math.linalg.det(p))]))

    # Now, we must find q such that p uu^T p^T = q vv^T q^T.
    # For this case, our V = SWAP CNOT01 is constant. Thus, we can compute v,
    # vvT, and its eigenvalues and eigenvectors directly. These matrices are stored
    # above as the constants v_one_cnot and q_one_cnot.

    # Once we have p and q properly in SO(4), we compute G and H in SO(4) such
    # that U = G V H
    G = math.dot(p, q_one_cnot.T)
    H = math.dot(math.conj(math.T(v_one_cnot)), math.dot(math.T(G), u))

    # We now use the magic basis to convert G, H to SU(2) x SU(2)
    AB = math.dot(E, math.dot(G, Edag))
    CD = math.dot(E, math.dot(H, Edag))

    # Extract the tensor prodcts to SU(2) x SU(2)
    A, B = _su2su2_to_tensor_products(AB)
    C, D = _su2su2_to_tensor_products(CD)

    # Recover the operators in the decomposition; note that because of the
    # initial SWAP, we exchange the order of A and B
    A_ops = one_qubit_decomposition(A, wires[1])
    B_ops = one_qubit_decomposition(B, wires[0])
    C_ops = one_qubit_decomposition(C, wires[0])
    D_ops = one_qubit_decomposition(D, wires[1])

    return C_ops + D_ops + [qml.CNOT(wires=wires)] + A_ops + B_ops


def _decomposition_2_cnots(U, wires):
    r"""If 2 CNOTs are required, we can write the circuit as
     -╭U- = -A--╭X--RZ(d)--╭X--C-
     -╰U- = -B--╰C--RX(p)--╰C--D-
    We need to find the angles for the Z and X rotations such that the inner
    part has the same spectrum as U, and then we can recover A, B, C, D.
    """
    # Compute the rotation angles

    u = math.dot(Edag, math.dot(U, E))
    gammaU = math.dot(u, math.T(u))
    evs, _ = math.linalg.eig(gammaU)

    # These choices are based on Proposition III.3 of
    # https://arxiv.org/abs/quant-ph/0308045
    # There is, however, a special case where the circuit has the form
    # -╭U- = -A--╭C--╭X--C-
    # -╰U- = -B--╰X--╰C--D-
    #
    # or some variant of this, where the two CNOTs are adjacent.
    #
    # What happens here is that the set of evs is -1, -1, 1, 1 and we can write
    # -╭U- = -A--╭X--SZ--╭X--C-
    # -╰U- = -B--╰C--SX--╰C--D-
    # where SZ and SX are square roots of Z and X respectively. (This
    # decomposition comes from using Hadamards to flip the direction of the
    # first CNOT, and then decomposing them and merging single-qubit gates.) For
    # some reason this case is not handled properly with the full algorithm, so
    # we treat it separately.

    sorted_evs = math.sort(math.real(evs))

    if math.allclose(sorted_evs, [-1, -1, 1, 1]):
        interior_decomp = [
            qml.CNOT(wires=[wires[1], wires[0]]),
            qml.S(wires=wires[0]),
            qml.SX(wires=wires[1]),
            qml.CNOT(wires=[wires[1], wires[0]]),
        ]

        # S \otimes SX
        inner_matrix = S_SX
    else:
        # For the non-special case, the eigenvalues come in conjugate pairs.
        # We need to find two non-conjugate eigenvalues to extract the angles.

        x = math.angle(evs[0])
        y = math.angle(evs[1])

        # If it was the conjugate, grab a different eigenvalue.
        if math.allclose(x, -y):
            y = math.angle(evs[2])

        delta = (x + y) / 2
        phi = (x - y) / 2
        interior_decomp = [
            qml.CNOT(wires=[wires[1], wires[0]]),
            qml.RZ(delta, wires=wires[0]),
            qml.RX(phi, wires=wires[1]),
            qml.CNOT(wires=[wires[1], wires[0]]),
        ]

        # need to perturb x by 5 precision to avoid a discontinuity at a special case.
        # see https://github.com/PennyLaneAI/pennylane/issues/5308
        precision = qml.math.finfo(delta.dtype).eps
        RZd = qml.RZ(math.cast_like(delta + 5 * precision, 1j), wires=0).matrix()
        RXp = qml.RX(phi, wires=0).matrix()
        inner_matrix = math.kron(RZd, RXp)

    # We need the matrix representation of this interior part, V, in order to
    # decompose U = (A \otimes B) V (C \otimes D)
    V = math.dot(math.cast_like(CNOT10, U), math.dot(inner_matrix, math.cast_like(CNOT10, U)))

    # Now we find the A, B, C, D in SU(2), and return the decomposition
    A, B, C, D = _extract_su2su2_prefactors(U, V)

    A_ops = one_qubit_decomposition(A, wires[0])
    B_ops = one_qubit_decomposition(B, wires[1])
    C_ops = one_qubit_decomposition(C, wires[0])
    D_ops = one_qubit_decomposition(D, wires[1])

    return C_ops + D_ops + interior_decomp + A_ops + B_ops


def _decomposition_3_cnots(U, wires):
    r"""The most general form of this decomposition is U = (A \otimes B) V (C \otimes D),
    where V is as depicted in the circuit below:
     -╭U- = -C--╭X--RZ(d)--╭C---------╭X--A-
     -╰U- = -D--╰C--RY(b)--╰X--RY(a)--╰C--B-
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

    angles = [math.angle(ev) for ev in evs]

    # We will sort the angles so that results are consistent across interfaces.
    # This step is skipped when using JAX-JIT, because it cannot sort without making the
    # magnitude of the angles concrete. This does not impact the validity of the resulting
    # decomposition, but may result in a different decompositions for jitting vs not.
    if not qml.math.is_abstract(U):
        angles = math.sort(angles)

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
        qml.CNOT(wires=wires),
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
    # -╭V- = -╭X--RZ(d)--╭C---------╭X--╭SWAP-
    # -╰V- = -╰C--RY(b)--╰X--RY(a)--╰C--╰SWAP-

    RZd = qml.RZ(math.cast_like(delta, 1j), wires=wires[0]).matrix()
    RYb = qml.RY(beta, wires=wires[0]).matrix()
    RYa = qml.RY(alpha, wires=wires[0]).matrix()

    V_mats = [CNOT10, math.kron(RZd, RYb), CNOT01, math.kron(math.eye(2), RYa), CNOT10, SWAP]

    V = math.convert_like(math.eye(4), U)

    for mat in V_mats:
        V = math.dot(math.cast_like(mat, U), V)

    # Now we need to find the four SU(2) operations A, B, C, D
    A, B, C, D = _extract_su2su2_prefactors(swap_U, V)

    # At this point, we have the following:
    # -╭U-╭SWAP- = --C--╭X-RZ(d)-╭C-------╭X-╭SWAP--A
    # -╰U-╰SWAP- = --D--╰C-RZ(b)-╰X-RY(a)-╰C-╰SWAP--B
    #
    # Using the relationship that SWAP(A \otimes B) SWAP = B \otimes A,
    # -╭U-╭SWAP- = --C--╭X-RZ(d)-╭C-------╭X--B--╭SWAP-
    # -╰U-╰SWAP- = --D--╰C-RZ(b)-╰X-RY(a)-╰C--A--╰SWAP-
    #
    # Now the SWAPs cancel, giving us the desired decomposition
    # (up to a global phase).
    # -╭U- = --C--╭X-RZ(d)-╭C-------╭X--B--
    # -╰U- = --D--╰C-RZ(b)-╰X-RY(a)-╰C--A--

    A_ops = one_qubit_decomposition(A, wires[1])
    B_ops = one_qubit_decomposition(B, wires[0])
    C_ops = one_qubit_decomposition(C, wires[0])
    D_ops = one_qubit_decomposition(D, wires[1])

    # Return the full decomposition
    return C_ops + D_ops + interior_decomp + A_ops + B_ops


def two_qubit_decomposition(U, wires):
    r"""Decompose a two-qubit unitary :math:`U` in terms of elementary operations.

    It is known that an arbitrary two-qubit operation can be implemented using a
    maximum of 3 CNOTs. This transform first determines the required number of
    CNOTs, then decomposes the operator into a circuit with a fixed form.  These
    decompositions are based a number of works by Shende, Markov, and Bullock
    `(1) <https://arxiv.org/abs/quant-ph/0308033>`__, `(2)
    <https://arxiv.org/abs/quant-ph/0308045v3>`__, `(3)
    <https://web.eecs.umich.edu/~imarkov/pubs/conf/spie04-2qubits.pdf>`__,
    though we note that many alternative decompositions are possible.

    For the 3-CNOT case, we recover the following circuit, which is Figure 2 in
    reference (1) above:

    .. figure:: ../../_static/two_qubit_decomposition_3_cnots.svg
        :align: center
        :width: 70%
        :target: javascript:void(0);

    where :math:`A, B, C, D` are :math:`SU(2)` operations, and the rotation angles are
    computed based on features of the input unitary :math:`U`.

    For the 2-CNOT case, the decomposition is

    .. figure:: ../../_static/two_qubit_decomposition_2_cnots.svg
        :align: center
        :width: 50%
        :target: javascript:void(0);

    For 1 CNOT, we have a CNOT surrounded by one :math:`SU(2)` per wire on each
    side.  The special case of no CNOTs simply returns a tensor product of two
    :math:`SU(2)` operations.

    This decomposition can be applied automatically to all two-qubit
    :class:`~.QubitUnitary` operations using the
    :func:`~pennylane.transforms.unitary_to_rot` transform.

    .. warning::

        This decomposition will not be differentiable in the ``unitary_to_rot``
        transform if the matrix being decomposed depends on parameters with
        respect to which we would like to take the gradient.  See the
        documentation of :func:`~pennylane.transforms.unitary_to_rot` for
        explicit examples of the differentiable and non-differentiable cases.

    Args:
        U (tensor): A :math:`4 \times 4` unitary matrix.
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

    >>> decomp = qml.ops.two_qubit_decomposition(np.array(U), wires=[0, 1])
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
    if not qml.math.is_abstract(U):
        num_cnots = _compute_num_cnots(U)

    with qml.QueuingManager.stop_recording():
        if qml.math.is_abstract(U):
            decomp = _decomposition_3_cnots(U, wires)
        elif num_cnots == 0:
            decomp = _decomposition_0_cnots(U, wires)
        elif num_cnots == 1:
            decomp = _decomposition_1_cnot(U, wires)
        elif num_cnots == 2:
            decomp = _decomposition_2_cnots(U, wires)
        else:
            decomp = _decomposition_3_cnots(U, wires)

    # If there is an active tape, queue the decomposition so that expand works
    current_tape = qml.queuing.QueuingManager.active_context()

    if current_tape is not None:
        for op in decomp:  # pragma: no cover
            qml.apply(op, context=current_tape)

    return decomp
