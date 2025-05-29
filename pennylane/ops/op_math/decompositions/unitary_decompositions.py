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

import warnings

import numpy as np
from scipy import sparse
from scipy.linalg import cossin

from pennylane import capture, compiler, math, ops, queuing, templates
from pennylane.decomposition.decomposition_rule import register_condition, register_resources
from pennylane.decomposition.resources import resource_rep
from pennylane.math.decomposition import (
    xyx_rotation_angles,
    xzx_rotation_angles,
    zxz_rotation_angles,
    zyz_rotation_angles,
)
from pennylane.operation import DecompositionUndefinedError
from pennylane.wires import Wires


def one_qubit_decomposition(U, wire, rotations="ZYZ", return_global_phase=False):
    r"""Decompose a one-qubit unitary :math:`U` in terms of elementary operations.

    Any one qubit unitary operation can be implemented up to a global phase by composing
    RX, RY, and RZ gates. Currently supported values for ``rotations`` are "rot", "ZYZ",
    "XYX", "XZX", and "ZXZ".

    Args:
        U (tensor): A :math:`2 \times 2` unitary matrix.
        wire (Union[Wires, Sequence[int] or int]): The wire on which to apply the operation.
        rotations (str): A string defining the sequence of rotations to decompose :math:`U` into.
        return_global_phase (bool): Whether to return the global phase as a ``qml.GlobalPhase(-alpha)``
            as the last element of the returned list of operations.

    Returns:
        list[Operation]: A list of gates which when applied in the order of appearance in the list
            is equivalent to the unitary :math:`U` up to a global phase. If ``return_global_phase=True``,
            the global phase is returned as the last element of the list.

    **Example**

    >>> U = np.array([[1, 1], [1, -1]]) / np.sqrt(2)  # Hadamard
    >>> qml.ops.one_qubit_decomposition(U, 0, rotations='ZYZ', return_global_phase=True)
    [RZ(3.1415926535897927, wires=[0]),
     RY(1.5707963267948963, wires=[0]),
     RZ(0.0, wires=[0]),
     GlobalPhase(-1.5707963267948966, wires=[])]
    >>> qml.ops.one_qubit_decomposition(U, 0, rotations='XZX', return_global_phase=True)
    [RX(1.5707963267948966, wires=[0]),
     RZ(1.5707963267948968, wires=[0]),
     RX(1.5707963267948966, wires=[0]),
     GlobalPhase(-1.5707963267948966, wires=[])]
    """

    supported_rotations = {
        "rot": _su2_rot_decomp,
        "ZYZ": _su2_zyz_decomp,
        "XYX": _su2_xyx_decomp,
        "XZX": _su2_xzx_decomp,
        "ZXZ": _su2_zxz_decomp,
    }

    if rotations not in supported_rotations:
        raise ValueError(
            f"Value {rotations} passed to rotations is either invalid or currently unsupported."
        )

    # It's fine to convert to dense here because the matrix is 2x2, and the decomposition
    # only consists of single-qubit rotation gates with a scalar rotation angle.
    if sparse.issparse(U):
        U = U.todense()

    U, global_phase = math.convert_to_su2(U, return_global_phase=True)
    with queuing.AnnotatedQueue() as q:
        supported_rotations[rotations](U, wires=Wires(wire))
        if return_global_phase:
            ops.GlobalPhase(-global_phase)

    # If there is an active queuing context, queue the decomposition so that expand works
    current_queue = queuing.QueuingManager.active_context()
    if current_queue is not None:
        for op in q.queue:  # pragma: no cover
            queuing.apply(op, context=current_queue)

    return q.queue


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
    [QubitUnitary(array([[ 0.02867704+0.82548843j,  0.5568274 -0.08769111j],
           [-0.5568274 -0.08769111j,  0.02867704-0.82548843j]]), wires=[0]),
    QubitUnitary(array([[ 0.32799033-0.78598401j,  0.40660725+0.33063881j],
           [-0.40660725+0.33063881j,  0.32799033+0.78598401j]]), wires=[1]),
    CNOT(wires=[1, 0]),
    RZ(0.259291854677022, wires=[0]),
    RY(-0.05808874413267284, wires=[1]),
    CNOT(wires=[0, 1]),
    RY(-1.6742322786950354, wires=[1]),
    CNOT(wires=[1, 0]),
    QubitUnitary(array([[ 0.91031205-0.21930866j,  0.20674186-0.28371375j],
           [-0.20674186-0.28371375j,  0.91031205+0.21930866j]]), wires=[1]),
    QubitUnitary(array([[-0.81886788-0.02979899j,  0.53279787-0.21140919j],
           [-0.53279787-0.21140919j, -0.81886788+0.02979899j]]), wires=[0]),
    GlobalPhase(0.1180587403699308, wires=[])]

    """

    if math.requires_grad(U):
        warnings.warn(
            "The two-qubit decomposition may not be differentiable when the input "
            "unitary depends on trainable parameters.",
            RuntimeWarning,
            stacklevel=2,
        )

    if sparse.issparse(U):
        raise DecompositionUndefinedError(
            "two_qubit_decomposition does not accept sparse matrices."
        )

    with queuing.AnnotatedQueue() as q:

        U, global_phase = math.convert_to_su4(U, return_global_phase=True)

        if _is_jax_jit(U):
            # Always use the 3-CNOT case when in jax.jit, because it is not compatible
            # with conditional logic. However, we want to still take advantage of the
            # more efficient decompositions in a qjit or program capture context.
            global_phase += _decompose_3_cnots(U, wires, global_phase)
        else:
            num_cnots = _compute_num_cnots(U)
            global_phase += ops.cond(
                num_cnots == 0,
                _decompose_0_cnots,
                _decompose_3_cnots,
                elifs=[
                    (num_cnots == 1, _decompose_1_cnot),
                    (num_cnots == 2, _decompose_2_cnots),
                ],
            )(U, wires, global_phase)

        if _is_jax_jit(U) or not math.allclose(global_phase, 0):
            ops.GlobalPhase(-global_phase)

    # If there is an active queuing context, queue the decomposition so that expand works
    current_queue = queuing.QueuingManager.active_context()
    if current_queue is not None:
        for op in q.queue:  # pragma: no cover
            queuing.apply(op, context=current_queue)

    return q.queue


def multi_qubit_decomposition(U, wires):
    r"""Decompose a multi-qubit unitary :math:`U` in terms of elementary operations.

    The n-qubit unitary :math:`U`, with :math:`n > 1`, is decomposed into four (:math:`n-1`)-qubit
    unitaries (:class:`~.QubitUnitary`) and three multiplexers (:class:`~.SelectPauliRot`)
    using the cosine-sine decomposition.
    This implementation is based on `arXiv:quant-ph/0504100 <https://arxiv.org/pdf/quant-ph/0504100>`__.

    Args:
        U (tensor): A :math:`2^n \times 2^n` unitary matrix with :math:`n > 1`.
        wires (Union[Wires, Sequence[int] or int]): The wires on which to apply the operation.

    Returns:
        list[Operation]: A list of operations that represent the decomposition
        of the matrix U.

    **Example**

    .. code-block:: pycon

        >>> matrix_target = qml.matrix(qml.QFT([0,1,2]))
        >>> ops = qml.ops.multi_qubit_decomposition(matrix_target, [0,1,2])
        >>> matrix_decomposition = qml.matrix(qml.prod(*ops[::-1]), wire_order = [0,1,2])
        >>> print([op.name for op in ops])
        ['QubitUnitary', 'SelectPauliRot', 'QubitUnitary', 'SelectPauliRot', 'QubitUnitary', 'SelectPauliRot', 'QubitUnitary']
        >>> print(np.allclose(matrix_decomposition, matrix_target))
        True
    """

    with queuing.AnnotatedQueue() as q:
        multi_qubit_decomp_rule(U, wires)

    # If there is an active queuing context, queue the decomposition so that expand works
    current_queue = queuing.QueuingManager.active_context()
    if current_queue is not None:
        for op in q.queue:  # pragma: no cover
            queuing.apply(op, context=current_queue)

    return q.queue


#######################
# Decomposition Rules #
#######################


def make_one_qubit_unitary_decomposition(su2_rule, su2_resource):
    """Wrapper around a naive one-qubit decomposition rule that adds a global phase."""

    def _resource_fn(num_wires):  # pylint: disable=unused-argument
        return su2_resource() | {ops.GlobalPhase: 1}

    @register_condition(lambda num_wires: num_wires == 1)
    @register_resources(_resource_fn)
    def _impl(U, wires, **__):
        if sparse.issparse(U):
            U = U.todense()
        U, global_phase = math.convert_to_su2(U, return_global_phase=True)
        su2_rule(U, wires=wires)
        ops.cond(math.logical_not(math.allclose(global_phase, 0)), _global_phase)(global_phase)

    return _impl


def _su2_rot_resource():
    return {
        ops.Rot: 1,
        ops.RZ: 1,  # RZ is produced in a special case, which has to be accounted for.
    }


def _su2_rot_decomp(U, wires, **__):
    phi, theta, omega = zyz_rotation_angles(U)
    ops.cond(
        math.allclose(U[..., 0, 1], 0.0),
        lambda: ops.RZ(2 * math.angle(U[..., 1, 1]) % (4 * np.pi), wires=wires[0]),
        lambda: ops.Rot(phi, theta, omega, wires=wires[0]),
    )()


def _su2_zyz_resource():
    return {ops.RZ: 2, ops.RY: 1}


def _su2_zyz_decomp(U, wires, **__):
    phi, theta, omega = zyz_rotation_angles(U)
    ops.RZ(phi, wires=wires[0])
    ops.RY(theta, wires=wires[0])
    ops.RZ(omega, wires=wires[0])


def _su2_xyx_resource():
    return {ops.RX: 2, ops.RY: 1}


def _su2_xyx_decomp(U, wires, **__):
    """Decomposes a QubitUnitary into a sequence of XYX rotations."""
    phi, theta, omega = xyx_rotation_angles(U)
    ops.RX(phi, wires=wires[0])
    ops.RY(theta, wires=wires[0])
    ops.RX(omega, wires=wires[0])


def _su2_xzx_resource():
    return {ops.RX: 2, ops.RZ: 1}


def _su2_xzx_decomp(U, wires, **__):
    phi, theta, omega = xzx_rotation_angles(U)
    ops.RX(phi, wires=wires[0])
    ops.RZ(theta, wires=wires[0])
    ops.RX(omega, wires=wires[0])


def _su2_zxz_resource():
    return {ops.RZ: 2, ops.RX: 1}


def _su2_zxz_decomp(U, wires, **__):
    phi, theta, omega = zxz_rotation_angles(U)
    ops.RZ(phi, wires=wires[0])
    ops.RX(theta, wires=wires[0])
    ops.RZ(omega, wires=wires[0])


rot_decomp_rule = make_one_qubit_unitary_decomposition(_su2_rot_decomp, _su2_rot_resource)
zyz_decomp_rule = make_one_qubit_unitary_decomposition(_su2_zyz_decomp, _su2_zyz_resource)
xyx_decomp_rule = make_one_qubit_unitary_decomposition(_su2_xyx_decomp, _su2_xyx_resource)
xzx_decomp_rule = make_one_qubit_unitary_decomposition(_su2_xzx_decomp, _su2_xzx_resource)
zxz_decomp_rule = make_one_qubit_unitary_decomposition(_su2_zxz_decomp, _su2_zxz_resource)


def _two_qubit_resource(**_):
    """A worst-case over-estimate for the resources of two-qubit unitary decomposition."""
    return {
        resource_rep(ops.QubitUnitary, num_wires=1): 4,
        ops.CNOT: 3,
        ops.RZ: 1,
        ops.RY: 2,
        # The three-CNOT case does not involve an RX, but an RX must be accounted
        # for in case the two-CNOT case is chosen at runtime.
        ops.RX: 1,
        ops.GlobalPhase: 1,
    }


@register_condition(lambda num_wires: num_wires == 2)
@register_resources(_two_qubit_resource)
def two_qubit_decomp_rule(U, wires, **__):
    """The decomposition rule for a two-qubit unitary."""

    U, initial_phase = math.convert_to_su4(U, return_global_phase=True)
    num_cnots = _compute_num_cnots(U)
    additional_phase = ops.cond(
        num_cnots == 0,
        _decompose_0_cnots,
        _decompose_3_cnots,
        elifs=[
            (num_cnots == 1, _decompose_1_cnot),
            (num_cnots == 2, _decompose_2_cnots),
        ],
    )(U, wires, initial_phase)
    total_phase = initial_phase + additional_phase
    ops.cond(math.logical_not(math.allclose(total_phase, 0)), _global_phase)(total_phase)


def _multi_qubit_decomp_resource(num_wires):
    return {
        resource_rep(ops.QubitUnitary, num_wires=num_wires - 1): 4,
        resource_rep(templates.SelectPauliRot, num_wires=num_wires, rot_axis="Z"): 2,
        resource_rep(templates.SelectPauliRot, num_wires=num_wires, rot_axis="Y"): 1,
    }


@register_condition(lambda num_wires: num_wires > 2)
@register_resources(_multi_qubit_decomp_resource)
def multi_qubit_decomp_rule(U, wires, **__):
    """The decomposition rule for a multi-qubit unitary."""

    # Combining the two equalities in Fig. 14 [https://arxiv.org/pdf/quant-ph/0504100], we can express
    # a n-qubit unitary U with four (n-1)-qubit unitaries and three multiplexed rotations ( via `qml.SelectPauliRot`)
    p = 2 ** (len(wires) - 1)

    (u1, u2), theta, (v1_dagg, v2_dagg) = _cossin_decomposition(U, p)

    v11_dagg, diag_v, v12_dagg = _compute_udv(v1_dagg, v2_dagg)
    u11, diag_u, u12 = _compute_udv(u1, u2)

    ops.QubitUnitary(v12_dagg, wires=wires[1:])

    templates.SelectPauliRot(
        -2 * math.angle(diag_v),
        target_wire=wires[0],
        control_wires=wires[1:],
        rot_axis="Z",
    )

    ops.QubitUnitary(v11_dagg, wires=wires[1:])

    templates.SelectPauliRot(2 * theta, target_wire=wires[0], control_wires=wires[1:], rot_axis="Y")

    ops.QubitUnitary(u12, wires=wires[1:])

    templates.SelectPauliRot(
        -2 * math.angle(diag_u),
        target_wire=wires[0],
        control_wires=wires[1:],
        rot_axis="Z",
    )

    ops.QubitUnitary(u11, wires=wires[1:])


####################
# Helper Functions #
####################


###################################################################################
# Developer notes:
#
# I was not able to get two-qubit decompositions to be fully differentiable for
# unitary matrices that were constructed within a QNode, based on the QNode's input
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
E_dag = E.conj().T

# Helpful to have static copies of these since they are needed in a few places.
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
    gamma = math.dot(U, math.T(U))
    trace = math.trace(gamma)
    g2 = math.dot(gamma, gamma)
    id4 = math.eye(4, like=g2)

    # We need a tolerance of around 1e-7 here to accommodate U specified with 8 decimal places.
    return ops.cond(
        # Case: 0 CNOTs (tensor product), the trace is +/- 4
        math.allclose(trace, 4, atol=1e-7) | math.allclose(trace, -4, atol=1e-7),
        lambda: 0,
        # Case: 3 CNOTs, the trace is a non-zero complex number with both real and imaginary parts.
        lambda: 3,
        elifs=[
            # Case: 1 CNOT, the trace is 0, and the eigenvalues of gammaU are [-1j, -1j, 1j, 1j]
            (
                math.allclose(trace, 0.0, atol=1e-7) & math.allclose(g2 + id4, 0.0, atol=1e-7),
                lambda: 1,
            ),
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
    # TODO: no you don't, the eigenvectors returned from math.linalg.eigh are not guaranteed to
    #       be in any particular order, especially when there is degeneracy. This means that p
    #       and q are not necessarily "in the same order" as claimed above. This may sometimes
    #       lead to incorrect results (see https://github.com/PennyLaneAI/pennylane/issues/5308)
    #       The current solution is to add a small perturbation to U and V to break the potential
    #       degeneracy. We should probably find a better algorithm at some point. [sc-89460]

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


def _decompose_0_cnots(U, wires, initial_phase):
    r"""If there are no CNOTs, this is just a tensor product of two single-qubit gates.
    We can perform that decomposition directly:
     -╭U- = -A-
     -╰U- = -B-
    """
    A, B = math.decomposition.su2su2_to_tensor_products(U)
    A, phaseA = math.convert_to_su2(A, return_global_phase=True)
    B, phaseB = math.convert_to_su2(B, return_global_phase=True)
    ops.QubitUnitary(A, wires=wires[0])
    ops.QubitUnitary(B, wires=wires[1])
    return math.cast_like(phaseA + phaseB, initial_phase)


def _decompose_1_cnot(U, wires, initial_phase):
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
    # !Note: future review on the eigh usage and risky eigvec order is needed: [sc-89460]
    _, p = math.linalg.eigh(math.real(uuT))

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

    G = math.dot(p, math.T(Q))
    H = math.dot(math.conj(math.T(V)), math.dot(math.T(G), u))

    # We now use the magic basis to convert G, H to SU(2) x SU(2)
    AB = math.dot(E, math.dot(G, E_dag))
    CD = math.dot(E, math.dot(H, E_dag))

    # Extract the tensor prodcts to SU(2) x SU(2)
    A, B = math.decomposition.su2su2_to_tensor_products(AB)
    C, D = math.decomposition.su2su2_to_tensor_products(CD)

    # Recover the operators in the decomposition; note that because of the
    # initial SWAP, we exchange the order of A and B
    ops.QubitUnitary(C, wires=wires[0])
    ops.QubitUnitary(D, wires=wires[1])
    ops.CNOT(wires=wires)
    ops.QubitUnitary(A, wires=wires[1])
    ops.QubitUnitary(B, wires=wires[0])

    return math.cast_like(-np.pi / 4, initial_phase)


def _decompose_2_cnots(U, wires, initial_phase):
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

    # If it was the conjugate, grab a different eigenvalue.
    y = math.cond(
        math.allclose(evs[0], math.conj(evs[1])),
        lambda: math.angle(evs[2]),
        lambda: math.angle(evs[1]),
        (),
    )

    delta = (x + y) / 2
    phi = (x - y) / 2

    # need to perturb x by 5 precision to avoid a discontinuity at a special case.
    # see https://github.com/PennyLaneAI/pennylane/issues/5308
    precision = math.finfo(delta.dtype).eps
    RZd = ops.RZ.compute_matrix(math.cast_like(delta + 5 * precision, 1j))
    RXp = ops.RX.compute_matrix(math.cast_like(phi + 5 * precision, 1j))
    inner_u = math.kron(RZd, RXp)

    # We need the matrix representation of this interior part, V, in order to
    # decompose U = (A \otimes B) V (C \otimes D)
    V = math.dot(math.cast_like(CNOT10, U), math.dot(inner_u, math.cast_like(CNOT10, U)))

    # Now we find the A, B, C, D in SU(2), and return the decomposition
    A, B, C, D = _extract_su2su2_prefactors(U, V)

    ops.QubitUnitary(C, wires[0])
    ops.QubitUnitary(D, wires[1])
    ops.CNOT(wires=[wires[1], wires[0]])
    ops.RZ(delta, wires=wires[0])
    ops.RX(phi, wires=wires[1])
    ops.CNOT(wires=[wires[1], wires[0]])
    ops.QubitUnitary(A, wires[0])
    ops.QubitUnitary(B, wires[1])

    return math.cast_like(0, initial_phase)


def _decompose_3_cnots(U, wires, initial_phase):
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
    # !Note: [sc-89460]
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

    EPS = math.finfo(delta.dtype).eps
    RZd = ops.RZ.compute_matrix(math.cast_like(delta + 5 * EPS, 1j))
    RYb = ops.RY.compute_matrix(beta + 5 * EPS)
    RYa = ops.RY.compute_matrix(alpha)

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

    ops.QubitUnitary(C, wires[0])
    ops.QubitUnitary(D, wires[1])
    ops.CNOT(wires=[wires[1], wires[0]])
    ops.RZ(delta, wires=wires[0])
    ops.RY(beta, wires=wires[1])
    ops.CNOT(wires=wires)
    ops.RY(alpha, wires=wires[1])
    ops.CNOT(wires=[wires[1], wires[0]])
    ops.QubitUnitary(A, wires[1])
    ops.QubitUnitary(B, wires[0])

    return math.cast_like(-np.pi / 4, initial_phase)


def _compute_udv(a, b):
    r"""Given the matrices `a` and `b`, calculates the matrices `u`, `d` and `v`
    of Eq. 36 in [arXiv-quant-ph:0504100](https://arxiv.org/pdf/quant-ph/0504100):

    .. math::

        a = u d v \\
        b = u d^{\dagger} v.
    """

    # Calculates u and d diagonalizing ab^dagger (Eq.39)
    ab_dagger = a @ math.conj(b.T)
    d_square, u = math.linalg.eig(ab_dagger)
    u, _ = math.linalg.qr(u)

    # complex square root of eigenvalues
    d = math.exp(1j * math.angle(d_square) / 2)

    # Calculates v using Eq.40
    v = math.conj(math.diag(d).T) @ math.conj(u.T) @ a

    return u, d, v


def _cossin_decomposition(U, p):

    # pylint: disable=import-outside-toplevel
    if math.get_interface(U) == "jax":
        # Wrap scipy's cossin function with pure_callback to make the decomposition compatible with jit

        import jax

        def scipy_cossin_callback(U_flat, p):
            dim = int(np.sqrt(U_flat.size))
            U_np = U_flat.reshape((dim, dim))
            (u1, u2), theta, (v1_dagg, v2_dagg) = cossin(U_np, p=p, q=p, separate=True)
            return u1, u2, theta, v1_dagg, v2_dagg

        def cossin_decomposition(U, p):
            dtype = U.dtype
            U_flat = U.reshape(-1)

            def callback(U_flat):
                return tuple(
                    arr.astype(dtype) for arr in scipy_cossin_callback(np.asarray(U_flat), p)
                )

            u1, u2, theta, v1_dagg, v2_dagg = jax.pure_callback(
                callback,
                result_shape_dtypes=(
                    jax.ShapeDtypeStruct((p, p), dtype),
                    jax.ShapeDtypeStruct((p, p), dtype),
                    jax.ShapeDtypeStruct((p,), dtype),
                    jax.ShapeDtypeStruct((p, p), dtype),
                    jax.ShapeDtypeStruct((p, p), dtype),
                ),
                U_flat=U_flat,
            )

            return (u1, u2), theta, (v1_dagg, v2_dagg)

    else:

        def cossin_decomposition(U, p):
            return cossin(U, p=p, q=p, separate=True)

    return cossin_decomposition(U, p)


def _global_phase(phase):
    ops.GlobalPhase(-phase)


def _is_jax_jit(U):
    """Assume jax-jit if U is abstract and not in a capture or qjit context."""
    return math.is_abstract(U) and not (capture.enabled() or compiler.active())
