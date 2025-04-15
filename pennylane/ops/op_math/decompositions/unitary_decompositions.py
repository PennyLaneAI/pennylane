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

"""This module defines decomposition functions for unitary matrices."""

import numpy as np

import pennylane as qml
from pennylane import math, register_resources
from pennylane.decomposition.decomposition_rule import DecompositionRule
from pennylane.decomposition.resources import resource_rep
from pennylane.decomposition.utils import DecompositionNotApplicable
from pennylane.math.decomposition import (
    xyx_rotation_angles,
    xzx_rotation_angles,
    zxz_rotation_angles,
    zyz_rotation_angles,
)


class OneQubitUnitaryDecomposition(DecompositionRule):
    """Wrapper around naive one-qubit decomposition rules that adds a global phase.

    Args:
        su2_rule (callable): A function that implements the naive decomposition rule which
            assumes that the unitary is SU(2)
        su2_resource (callable): A function that returns the resources required by the naive
            decomposition rule, without the GlobalPhase.

    """

    def __init__(self, su2_rule, su2_resource):
        self._naive_rule = su2_rule
        self._naive_resources = su2_resource
        super().__init__(self._get_impl(), self._get_resource_fn())

    def _get_impl(self):
        """The implementation of the decomposition rule."""

        def _impl(U, wires, **__):
            U, global_phase = math.convert_to_su2(U, return_global_phase=True)
            self._naive_rule(U, wires=wires)
            qml.GlobalPhase(-global_phase)

        return _impl

    def _get_resource_fn(self):
        """The resource function of the decomposition rule."""

        def _resource_fn(num_wires):
            if num_wires != 1:
                raise DecompositionNotApplicable
            return self._naive_resources() | {qml.GlobalPhase: 1}

        return _resource_fn


def _su2_rot_resource():
    return {qml.Rot: 1}


def _su2_rot_decomp(U, wires, **__):
    phi, theta, omega = zyz_rotation_angles(U)
    qml.Rot(phi, theta, omega, wires=wires[0])


rot_decomposition = OneQubitUnitaryDecomposition(_su2_rot_decomp, _su2_rot_resource)


def _su2_zyz_resource():
    return {qml.RZ: 2, qml.RY: 1}


def _su2_zyz_decomp(U, wires, **__):
    phi, theta, omega = zyz_rotation_angles(U)
    qml.RZ(phi, wires=wires[0])
    qml.RY(theta, wires=wires[0])
    qml.RZ(omega, wires=wires[0])


zyz_decomposition = OneQubitUnitaryDecomposition(_su2_zyz_decomp, _su2_zyz_resource)


def _su2_xyx_resource():
    return {qml.RX: 2, qml.RY: 1}


def _su2_xyx_decomp(U, wires, **__):
    """Decomposes a QubitUnitary into a sequence of XYX rotations."""
    phi, theta, omega = xyx_rotation_angles(U)
    qml.RX(phi, wires=wires[0])
    qml.RY(theta, wires=wires[0])
    qml.RX(omega, wires=wires[0])


xyx_decomposition = OneQubitUnitaryDecomposition(_su2_xyx_decomp, _su2_xyx_resource)


def _su2_xzx_resource():
    return {qml.RX: 2, qml.RZ: 1}


def _su2_xzx_decomp(U, wires, **__):
    phi, theta, omega = xzx_rotation_angles(U)
    qml.RX(phi, wires=wires[0])
    qml.RZ(theta, wires=wires[0])
    qml.RX(omega, wires=wires[0])


xzx_decomposition = OneQubitUnitaryDecomposition(_su2_xzx_decomp, _su2_xzx_resource)


def _su2_zxz_resource():
    return {qml.RZ: 2, qml.RX: 1}


def _su2_zxz_decomp(U, wires, **__):
    phi, theta, omega = zxz_rotation_angles(U)
    qml.RZ(phi, wires=wires[0])
    qml.RX(theta, wires=wires[0])
    qml.RZ(omega, wires=wires[0])


zxz_decomposition = OneQubitUnitaryDecomposition(_su2_zxz_decomp, _su2_zxz_resource)

# This gate E is called the "magic basis". It can be used to convert between
# SO(4) and SU(2) x SU(2). For A in SO(4), E A E^\dag is in SU(2) x SU(2).
E = np.array([[1, 1j, 0, 0], [0, 0, 1j, 1], [0, 0, 1j, -1], [1, -1j, 0, 0]]) / np.sqrt(2)
E_dag = E.conj().T

CNOT01 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
CNOT10 = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])
SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])


def _compute_num_cnots(U):
    r"""Compute the number of CNOTs required to implement a U in SU(4).
    This is based on the trace of

    .. math::

        \gamma(U) = (E^\dag U E) (E^\dag U E)^T,

    and follows the arguments of this paper: https://arxiv.org/abs/quant-ph/0308045.

    """

    U = math.dot(E_dag, math.dot(U, E))
    gamma = math.dot(U, U.T)
    trace = math.trace(gamma)
    g2 = math.dot(gamma, gamma)
    id4 = math.eye(4)

    # We need a tolerance of around 1e-7 here to accommodate U specified with 8 decimal places.
    return qml.cond(
        # Case: 0 CNOTs (tensor product), the trace is +/- 4
        math.allclose(math.abs(trace), 4, atol=1e-7),
        lambda: 0,
        # Case: 3 CNOTs, the trace is a non-zero complex number with both real and imaginary parts.
        lambda: 3,
        elifs=[
            # Case: 1 CNOT, the trace is 0, and the eigenvalues of gammaU are [-1j, -1j, 1j, 1j]
            (math.allclose(trace, 0.0, atol=1e-7) & math.allclose(g2 + id4, 0.0), lambda: 1),
            # Case: 2 CNOTs, the trace has only a real part (or is 0)
            (math.allclose(math.imag(trace), 0.0, atol=1e-7), lambda: 2),
        ],
    )()


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
    u = math.dot(math.cast_like(E_dag, V), math.dot(U, math.cast_like(E, V)))
    v = math.dot(math.cast_like(E_dag, V), math.dot(V, math.cast_like(E, V)))

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
    AB = math.dot(math.cast_like(E, G), math.dot(G, math.cast_like(E_dag, G)))
    CD = math.dot(math.cast_like(E, H), math.dot(H, math.cast_like(E_dag, H)))

    # Now, we just need to extract the constituent tensor products.
    A, B = math.decomposition.su2su2_to_tensor_products(AB)
    C, D = math.decomposition.su2su2_to_tensor_products(CD)

    return A, B, C, D


def _decompose_0_cnots(U, wires):
    r"""If there are no CNOTs, this is just a tensor product of two single-qubit gates.
    We can perform that decomposition directly:
     -╭U- = -A-
     -╰U- = -B-
    """
    A, B = math.decomposition.su2su2_to_tensor_products(U)
    A, phaseA = math.convert_to_su2(A, return_global_phase=True)
    B, phaseB = math.convert_to_su2(B, return_global_phase=True)
    qml.QubitUnitary(A, wires=wires[0])
    qml.QubitUnitary(B, wires=wires[1])
    return phaseA + phaseB


def _decompose_1_cnot(U, wires):
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
    u = math.dot(math.cast_like(E_dag, U), math.dot(swap_U, math.cast_like(E, U)))
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

    # Once we have p and q properly in SO(4), compute G and H in SO(4) such that U = G V H
    V = np.array(
        [
            [0.5, 0.5j, 0.5j, -0.5],
            [-0.5j, 0.5, -0.5, -0.5j],
            [-0.5j, -0.5, 0.5, -0.5j],
            [0.5, -0.5j, -0.5j, -0.5],
        ]
    )
    # This Q is properly in SO(4)
    Q = (1 / np.sqrt(2)) * np.array(
        [
            [-1, 0, -1, 0],
            [0, 1, 0, 1],
            [0, 1, 0, -1],
            [1, 0, -1, 0],
        ]
    )

    G = math.dot(p, Q.T)
    H = math.dot(math.conj(math.T(V)), math.dot(math.T(G), u))

    # We now use the magic basis to convert G, H to SU(2) x SU(2)
    AB = math.dot(E, math.dot(G, E_dag))
    CD = math.dot(E, math.dot(H, E_dag))

    # Extract the tensor prodcts to SU(2) x SU(2)
    A, B = math.decomposition.su2su2_to_tensor_products(AB)
    C, D = math.decomposition.su2su2_to_tensor_products(CD)

    # Recover the operators in the decomposition; note that because of the
    # initial SWAP, we exchange the order of A and B
    qml.QubitUnitary(C, wires=wires[0])
    qml.QubitUnitary(D, wires=wires[1])
    qml.CNOT(wires=wires)
    qml.QubitUnitary(A, wires=wires[1])
    qml.QubitUnitary(B, wires=wires[0])

    return -np.pi / 4


def _decompose_2_cnots(U, wires):
    r"""If 2 CNOTs are required, we can write the circuit as
     -╭U- = -A--╭X--RZ(d)--╭X--C-
     -╰U- = -B--╰C--RX(p)--╰C--D-
    We need to find the angles for the Z and X rotations such that the inner
    part has the same spectrum as U, and then we can recover A, B, C, D.
    """

    # Compute the rotation angles
    u = math.dot(E_dag, math.dot(U, E))
    gammaU = math.dot(u, math.T(u))
    evs = math.linalg.eigvals(gammaU)

    # These choices are based on Proposition III.3 of
    # https://arxiv.org/abs/quant-ph/0308045

    x = math.angle(evs[0])
    y = math.angle(evs[1])

    # If it was the conjugate, grab a different eigenvalue.
    y = math.cond(math.allclose(x, -y), lambda: math.angle(evs[2]), lambda: y, ())

    delta = (x + y) / 2
    phi = (x - y) / 2

    # need to perturb x by 5 precision to avoid a discontinuity at a special case.
    # see https://github.com/PennyLaneAI/pennylane/issues/5308
    precision = qml.math.finfo(delta.dtype).eps
    RZd = qml.RZ.compute_matrix(math.cast_like(delta + 5 * precision, 1j))
    RXp = qml.RX.compute_matrix(phi)
    inner_u = math.kron(RZd, RXp)

    # We need the matrix representation of this interior part, V, in order to
    # decompose U = (A \otimes B) V (C \otimes D)
    V = math.dot(math.cast_like(CNOT10, U), math.dot(inner_u, math.cast_like(CNOT10, U)))

    # Now we find the A, B, C, D in SU(2), and return the decomposition
    A, B, C, D = _extract_su2su2_prefactors(U, V)

    qml.QubitUnitary(C, wires[0])
    qml.QubitUnitary(D, wires[1])
    qml.CNOT(wires=[wires[1], wires[0]])
    qml.RZ(delta, wires=wires[0])
    qml.RX(phi, wires=wires[1])
    qml.CNOT(wires=[wires[1], wires[0]])
    qml.QubitUnitary(A, wires[0])
    qml.QubitUnitary(B, wires[1])

    return 0.0


def _decompose_3_cnots(U, wires):
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
    u = math.dot(E_dag, math.dot(swap_U, E))
    gammaU = math.dot(u, math.T(u))
    evs, _ = math.linalg.eig(gammaU)

    x = math.angle(evs[0])
    y = math.angle(evs[1])
    z = math.angle(evs[2])

    # Compute functions of the eigenvalues; there are different options in v1
    # vs. v3 of the paper, I'm not entirely sure why. This is the version from v3.
    alpha = (x + y) / 2
    beta = (x + z) / 2
    delta = (z + y) / 2

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

    RZd = qml.RZ.compute_matrix(math.cast_like(delta, 1j))
    RYb = qml.RY.compute_matrix(beta)
    RYa = qml.RY.compute_matrix(alpha)

    V_mats = [
        CNOT10,
        math.kron(RZd, RYb),
        CNOT01,
        math.kron(math.eye(2), RYa),
        CNOT10,
        SWAP,
    ]
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

    qml.QubitUnitary(C, wires[0])
    qml.QubitUnitary(D, wires[1])
    qml.CNOT(wires=[wires[1], wires[0]])
    qml.RZ(delta, wires=wires[0])
    qml.RY(beta, wires=wires[1])
    qml.CNOT(wires=wires)
    qml.RY(alpha, wires=wires[1])
    qml.CNOT(wires=[wires[1], wires[0]])
    qml.QubitUnitary(A, wires[1])
    qml.QubitUnitary(B, wires[0])

    return -np.pi / 4


def _two_qubit_resource(num_wires):
    """A worst-case over-estimate for the resources of two-qubit unitary decomposition."""
    if num_wires != 2:
        raise DecompositionNotApplicable
    # Assume the 3-CNOT case.
    return {
        resource_rep(qml.QubitUnitary, num_wires=1): 4,
        qml.CNOT: 3,
        qml.RZ: 1,
        qml.RY: 2,
        # The three CNOT case does not involve an RX, but an RX must be accounted
        # for in case the two CNOT case is chosen at runtime.
        qml.RX: 0,
        qml.GlobalPhase: 1,
    }


@register_resources(_two_qubit_resource)
def two_qubit_decomp_rule(U, wires, **__):
    """The decomposition rule for a two-qubit unitary."""

    U, initial_phase = math.convert_to_su4(U, return_global_phase=True)
    num_cnots = _compute_num_cnots(U)
    additional_phase = qml.cond(
        num_cnots == 0,
        _decompose_0_cnots,
        _decompose_3_cnots,
        elifs=[
            (num_cnots == 1, _decompose_1_cnot),
            (num_cnots == 2, _decompose_2_cnots),
        ],
    )(U, wires=wires)
    total_phase = initial_phase + additional_phase
    qml.GlobalPhase(-total_phase)
