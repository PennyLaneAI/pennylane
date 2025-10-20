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
from pennylane.exceptions import DecompositionUndefinedError
from pennylane.math.decomposition import (
    xyx_rotation_angles,
    xzx_rotation_angles,
    zxz_rotation_angles,
    zyz_rotation_angles,
)
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

    >>> from pprint import pprint
    >>> U = np.array([[1, 1], [1, -1]]) / np.sqrt(2)  # Hadamard
    >>> decomp = qml.ops.one_qubit_decomposition(U, 0, rotations='ZYZ', return_global_phase=True)
    >>> pprint(decomp)
    [RZ(np.float64(3.14159...), wires=[0]),
     RY(np.float64(1.57079...), wires=[0]),
     RZ(np.float64(0.0), wires=[0]),
     GlobalPhase(np.float64(-1.57079...), wires=[])]
    >>> decomp = qml.ops.one_qubit_decomposition(U, 0, rotations='XZX', return_global_phase=True)
    >>> pprint(decomp)
    [RX(np.float64(1.57079...), wires=[0]),
     RZ(np.float64(1.57079...), wires=[0]),
     RX(np.float64(1.57079...), wires=[0]),
     GlobalPhase(np.float64(-1.57079...), wires=[])]
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
    if queuing.QueuingManager.recording():
        for op in q.queue:  # pragma: no cover
            queuing.apply(op)

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

    For the 2-CNOT case, the decomposition is currently not supported and will
    instead produce a 3-CNOT circuit like above.

    For a single CNOT, we have a CNOT surrounded by one :math:`SU(2)` per wire on each
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
    >>> U = unitary_group.rvs(4, random_state=42)

    We can compute its decompositon like so:

    >>> from pprint import pprint
    >>> decomp = qml.ops.two_qubit_decomposition(np.array(U), wires=[0, 1])
    >>> pprint(decomp) # doctest: +SKIP
    [QubitUnitary(array([[ 0.35935497-0.35945703j, -0.81150079+0.28830732j],
           [ 0.81150079+0.28830732j,  0.35935497+0.35945703j]]), wires=[0]),
     QubitUnitary(array([[ 0.73465919-0.15696895j,  0.51629531-0.41118825j],
           [-0.51629531-0.41118825j,  0.73465919+0.15696895j]]), wires=[1]),
     CNOT(wires=[1, 0]),
     RZ(np.float64(0.028408953417448358), wires=[0]),
     RY(np.float64(0.6226823676455966), wires=[1]),
     CNOT(wires=[0, 1]),
     RY(np.float64(-0.7259987841675299), wires=[1]),
     CNOT(wires=[1, 0]),
     QubitUnitary(array([[ 0.85429569-0.34743933j,  0.14569083+0.35810469j],
           [-0.14569083+0.35810469j,  0.85429569+0.34743933j]]), wires=[0]),
     QubitUnitary(array([[-0.30052527-0.4826478j ,  0.74833925-0.34164898j],
           [-0.74833925-0.34164898j, -0.30052527+0.4826478j ]]), wires=[1]),
     GlobalPhase(np.float64(0.07394316416802127), wires=[])]

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
            # Use the 3-CNOT case for num_cnots=2 as well because we do not have a reliably
            # correct implementation of the 2-CNOT case right now.
            global_phase += ops.cond(
                num_cnots == 0,
                _decompose_0_cnots,
                _decompose_3_cnots,
                elifs=[(num_cnots == 1, _decompose_1_cnot)],
            )(U, wires, global_phase)

        if _is_jax_jit(U) or not math.allclose(global_phase, 0):
            ops.GlobalPhase(-global_phase)

    # If there is an active queuing context, queue the decomposition so that expand works
    if queuing.QueuingManager.recording():
        for op in q.queue:  # pragma: no cover
            queuing.apply(op)

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
    if queuing.QueuingManager.recording():
        for op in q.queue:  # pragma: no cover
            queuing.apply(op)

    return q.queue


#######################
# Decomposition Rules #
#######################


def make_one_qubit_unitary_decomposition(su2_rule, su2_resource):
    """Wrapper around a naive one-qubit decomposition rule that adds a global phase."""

    def _resource_fn(num_wires):  # pylint: disable=unused-argument
        return su2_resource() | {ops.GlobalPhase: 1}

    # Resources are not exact because the global phase or rotations might be skipped
    @register_condition(lambda num_wires: num_wires == 1)
    @register_resources(_resource_fn, exact=False)
    def _impl(U, wires, **__):
        if sparse.issparse(U):
            U = U.todense()
        U, global_phase = math.convert_to_su2(U, return_global_phase=True)
        su2_rule(U, wires=wires)
        ops.cond(math.logical_not(math.allclose(global_phase, 0)), ops.GlobalPhase)(-global_phase)

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
        ops.GlobalPhase: 1,
    }


@register_condition(lambda num_wires: num_wires == 2)
@register_resources(_two_qubit_resource, exact=False)
def two_qubit_decomp_rule(U, wires, **__):
    """The decomposition rule for a two-qubit unitary."""

    U, initial_phase = math.convert_to_su4(U, return_global_phase=True)
    num_cnots = _compute_num_cnots(U)
    # Use the 3-CNOT case for num_cnots=2 as well because we do not have a reliably
    # correct implementation of the 2-CNOT case right now.
    additional_phase = ops.cond(
        num_cnots == 0,
        _decompose_0_cnots,
        _decompose_3_cnots,
        elifs=[(num_cnots == 1, _decompose_1_cnot)],
    )(U, wires, initial_phase)
    total_phase = initial_phase + additional_phase
    ops.cond(math.logical_not(math.allclose(total_phase, 0)), ops.GlobalPhase)(-total_phase)


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
SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
S_0 = np.kron(np.diag([1j, 1]), np.eye(2))
S_0_dag = S_0.conj().T


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


def _multidot(*matrices):
    mat = matrices[0]
    for m in matrices[1:]:
        mat = math.dot(mat, m)
    return mat


def _real_imag_split_eigh(A, factor):
    _, basis = math.linalg.eigh(math.real(A) / factor + factor * math.imag(A))
    eigvals = _multidot(math.transpose(basis), A, basis)
    return eigvals, basis


def _ai_kak(U):
    """Compute a type-AI Cartan decomposition of a unitary ``U`` in the standard basis/representation.
    This is done in the following steps (see App.E of https://arxiv.org/abs/2503.19014, case AI):
    - compute a real-valued eigenbasis O_1 (orthogonal matrix) and the eigenvalues d^2 of Δ = U U^T,
    - make O_1 *special* orthogonal by adjusting the sign of its first column,
    - take the square root of d^2 to obtain d,
    - compute O_2 = d* O_1^T U,
    - adjust the sign of the first row of O_2 so that O_2 is special orthogonal, and multiply
      the first entry of d with the same sign.
    """

    # Delta is symmetric (Delta^T=Delta) and unitary (because U is) but not real-valued.
    Delta = math.dot(U, math.transpose(U))
    # Denote the real and imaginary parts of Delta as R and I, respectively.
    # Delta^T=Delta ==> R^T=R and I^T=I.  (1)
    # Delta unitary ==> (R+iI)(R^T-iI^T)=id ==> RR^T+II^T=id and IR^T-RI^T = 0 (2)
    # Combine (1) and second equation in (2): 0=IR-RI=[I,R] ==> I and R share an eigenbasis
    # ==> I, R and Delta=R+iI share an eigenbasis.
    #
    # We need to make sure that for degenerate real or imaginary parts
    # we actually find an eigenbasis that also is one for Delta. In a first step, we make this
    # likely by weighting the real and imaginary part by 1/pi and pi. In a second step,
    # we check whether this basis diagonalized Delta, and recompute with a new weighting (by 10)
    # if it did not.
    d_squared, o1 = _real_imag_split_eigh(Delta, np.pi)
    d_squared, o1 = math.cond(
        math.allclose(math.diag(math.diag(d_squared)), d_squared, atol=1e-7),
        lambda: (d_squared, o1),
        lambda: _real_imag_split_eigh(Delta, 10.0),
        (),
    )

    # This implements o1[:, 0] *= det(o1) to ensure det(o1) = 1 afterwards
    # No need to modify the eigenvalues or d because this change will be absorbed in o2
    o1 = math.transpose(math.set_index(math.transpose(o1), 0, math.linalg.det(o1) * o1[:, 0]))

    d = math.diag(math.sqrt(math.diag(d_squared)))
    o2 = _multidot(math.conj(d), math.transpose(o1), U)

    # Instead of guaranteeing the correct determinant while taking the square root,
    # we correct it after the fact
    # This implements o2[0] *= det(o2) to ensure det(o2) = 1 afterwards
    # Here we need to also adapt d because o1 already is fixed, so it can not absorb det(o2)
    det_o2 = math.linalg.det(o2)
    o2 = math.set_index(o2, 0, det_o2 * o2[0])
    d = math.set_index(d, 0, det_o2 * d[0])

    return o1, d, o2


def _extract_abde(A):
    """Extract the parameters for the central part of a 3-CNOT circuit as well
    as a global phase. See documentation of _decompose_3_cnots for details.

    The math.cond calls are necessary in order to decide from which matrix entries to extract
    the angles.
    The input matrix is expected to be of the form (called C1 in _decompose_3_cnots)

    exp(-i d') cos(a')           0                  0          -exp(-i d') sin(a')
              0         exp(i e') sin(b')   exp(i e') cos(b')            0
              0         exp(i e') cos(b')  -exp(i e') sin(b')            0
    exp(-i d') sin(a')           0                  0           exp(-i d') cos(a')

    and this function will return

    a = a' + b'
    b = a' - b'
    d = d' + e'
    e = (d' - e') / 2.

    The performed computation steps are the following:
    1. Compute a' from A_{00} and A_{30}
        a. If cos(d')sin(a')≠0 or cos(d')cos(a')≠0, use
           atan2(cos(d')sin(a'), cos(d')cos(a')) = atan2(sin(a'), cos(a')) = a'
        b. If cos(d')sin(a')=cos(d')cos(a')=0, use
           atan2(sin(d')sin(a'), sin(d')cos(a')) = atan2(sin(a'), cos(a')) = a'
        Note that if cos(d')sin(a')=cos(d')cos(a')=0, we know that cos(d')=0 and sin(d')≠0
    2. Compute b' from A_{11} and A_{21}
        a. If cos(e')sin(b')≠0 or cos(e')cos(b')≠0, use
           atan2(cos(e')sin(b'), cos(e')cos(b')) = atan2(sin(b'), cos(b')) = b'
        b. If cos(e')sin(b')=cos(e')cos(b')=0, use
           atan2(sin(e')sin(b'), sin(e')cos(b')) = atan2(sin(b'), cos(b')) = b'
        Note that if cos(e')sin(b')=cos(e')cos(b')=0, we know that cos(e')=0 and sin(e')≠0
    3. Compute a = a' + b' and b = a' - b'
    4. Compute exp(-id') from A_{00} or A_{30}
        a. If A_{00}≠0, use it and compute exp(-i d')cos(a') / cos(a')
        b. If A_{00}=0, us A_{30} and compute exp(-i d')sin(a') / sin(a')
    5. Compute exp(ie') from A_{11} or A_{21}
        a. If A_{21}≠0, use it and compute exp(i e')cos(b') / cos(b')
        b. If A_{21}=0, us A_{11} and compute exp(i e')sin(b') / sin(b')
    6. Compute d = d' + e' as arg(exp(ie') exp(-id')*)
    7. Compute e = (d' - e')/2 as -arg(exp(ie') exp(-id/2))
    """
    a_plus_b_half = math.cond(
        math.allclose(math.real(A[::3, 0]), math.zeros_like(math.real(A[::3, 0]))),
        lambda: math.arctan2(math.imag(A[3, 0]), math.imag(A[0, 0])),
        lambda: math.arctan2(math.real(A[3, 0]), math.real(A[0, 0])),
        (),
    )
    a_minus_b_half = math.cond(
        math.allclose(math.real(A[1:3, 1]), math.zeros_like(math.real(A[1:3, 1]))),
        lambda: math.arctan2(math.imag(A[1, 1]), math.imag(A[2, 1])),
        lambda: math.arctan2(math.real(A[1, 1]), math.real(A[2, 1])),
        (),
    )
    a = a_plus_b_half + a_minus_b_half
    b = a_plus_b_half - a_minus_b_half

    apb_frac = math.cond(
        math.isclose(A[0, 0], math.zeros_like(A[0, 0])),
        lambda: A[3, 0] / math.cast_like(math.sin(a_plus_b_half), A[3, 0]),
        lambda: A[0, 0] / math.cast_like(math.cos(a_plus_b_half), A[0, 0]),
        (),
    )
    amb_frac = math.cond(
        math.isclose(A[2, 1], math.zeros_like(A[2, 1])),
        lambda: A[1, 1] / math.cast_like(math.sin(a_minus_b_half), A[1, 1]),
        lambda: A[2, 1] / math.cast_like(math.cos(a_minus_b_half), A[2, 1]),
        (),
    )

    d = math.angle(amb_frac * math.conj(apb_frac))
    e = -math.angle(amb_frac * math.exp(-1j * math.cast_like(d / 2, 1j)))
    return a, b, d, e


def _central_circuit(a, b, d, wires):
    """Central part of 3-CNOT circuit."""
    ops.CNOT([wires[1], wires[0]])
    ops.RZ(d, wires[0])
    ops.RY(b, wires[1])
    ops.CNOT(wires)
    ops.RY(a, wires[1])
    ops.CNOT([wires[1], wires[0]])


def _decompose_3_cnots(U, wires, initial_phase):
    """Decompose a unitary 4x4 matrix into a 3-CNOT circuit.

    From a mathematical perspective, this decomposition mainly is from one matrix into
    a product of three matrices, which is an instance of a Cartan, or KAK, decomposition.
    The Cartan decomposition is of type AI, decomposing a (special) unitary matrix into
    two special orthogonal matrices and a matrix from some representation of U(1)^r, with r=4
    for unitary input (used here) and r=3 for special unitary input.
    See e.g. App. E, paragraph on AI in https://arxiv.org/abs/2503.19014 for details.
    This Cartan decomposition is implemented in _ai_kak. Here we take care
    of translating the input and output matrices of _ai_kak into the right representations
    of SO(4) and U(1)^4.
    The representations that we want are given by the fixed circuit structure that we are after:
    An arbitrary special unitary single-qubit operation on each qubit necessarily is from
    SU(2) x SU(2), which is isomorphic to SO(4) (a so-called accidental or exceptional
    isomorphism). The central circuit part

    0: ─╭X──RZ(d)─╭●────────╭X──GlobalPhase(e)─┤
    1: ─╰●──RY(b)─╰X──RY(a)─╰●──GlobalPhase(e)─┤

    forms a representation of U(1)^4, given by the matrix (call it C1)

    exp(-i d') cos(a')           0                  0          -exp(-i d') sin(a')
              0         exp(i e') sin(b')   exp(i e') cos(b')            0
              0         exp(i e') cos(b')  -exp(i e') sin(b')            0
    exp(-i d') sin(a')           0                  0           exp(-i d') cos(a')

    where
    a' = a/2 + b/2
    b' = a/2 - b/2
    d' = d/2 + e
    e' = d/2 - e.

    Now, as is used throughout the two-qubit decompositions in this file, the transformation
    between the canonical representation of SO(4) (real matrices with OO^T = 1 and determinant 1)
    and the representation as single-qubit unitaries on both qubits is given by the so-called
    magic basis E. That is, for O in the canonical representation, E O E† is of the form A⊗B with
    A and B 2x2 special unitary matrices.
    Simultaneously, E transforms diagonal unitary matrices into matrices of the form

      exp(i t) cos(r)         0                  0         i exp(i t) sin(r)
              0         exp(i u) cos(s)  i exp(i u) sin(s)         0
              0       i exp(i u) sin(s)    exp(i u) cos(s)         0
    i exp(i t) sin(r)         0                  0           exp(i t) cos(r)

    generated by the "Cartan coordinate" generators X⊗X, Y⊗Y, Z⊗Z (and I⊗I). Call this matrix C2.

    Finally, note that C2 can be transformed into C1 via the following static gates:

    C1 = SWAP S† C2 S.

    (S is just `qml.S`)
    Now, we "just" need to combine all of these basis changes with the type-AI Cartan
    decomposition (_ai_kak) and a function that extracts the parameters a, b, d, e from a matrix
    of the form C1 (_extract_abde). For this, let's compute (not necessarily obvious to come up
    with but easy to verify).

    V := S SWAP U S†    (bookkeeping un-transformation)
    W := E† V E         (magic basis un-rotation)
    W =: K_1 A_K K_2    (computed by _ai_kak; K_i∈SO(4), A_K diagonal)

    L_1 := E K_1 E†             (L_1 is of form A⊗B)
    L_2 := E K_2 E†             (L_2 is of form A⊗B)
    A_L := E A_K E†             (A_L is of form C2)

    M_1 := SWAP S† L_1 S SWAP   (M_1 is still of the form A⊗B, with new A, B)
    M_2 := S† L_2 S             (M_2 is still of the form A⊗B, with new A, B)
    A_M := SWAP S† A_L S        (A_M is of the form C1)

    Now we can extract a, b, d, e from A_M because it is of form C1, implemented by the
    central circuit part. Also, we can decompose M_1 and M_2 into two single-qubit
    unitaries via _decompose_0_cnots.
    To verify correctness, compute

    M_1 A_M M_2
    = (SWAP S† L_1 S SWAP) (SWAP S A_L S†) (S† L_2 S)
    = (SWAP S† L_1 A_L L_2 S)
    = (SWAP S† (E K_1 E†) (E A_K E†) (E K_2 E†) S)
    = (SWAP S† E (K_1 A_K K_2) E† S)
    = (SWAP S† E W E† S)
    = (SWAP S† E (E† V E) E† S)
    = (SWAP S† (S SWAP U S†) S)
    = U

    So we actually implemented U!
    """
    W = _multidot(E_dag, S_0_dag, SWAP, U, S_0, E)
    K_1, A_K, K_2 = _ai_kak(W)

    L_1 = _multidot(E, K_1, E_dag)
    A_L = _multidot(E, A_K, E_dag)
    L_2 = _multidot(E, K_2, E_dag)

    M_1 = _multidot(SWAP, S_0, L_1, S_0_dag, SWAP)
    M_2 = _multidot(S_0, L_2, S_0_dag)
    A_M = _multidot(SWAP, S_0, A_L, S_0_dag)

    a, b, d, e = _extract_abde(A_M)

    _decompose_0_cnots(M_2, wires, 0.0)
    _central_circuit(a, b, d, wires)
    _decompose_0_cnots(M_1, wires, 0.0)
    # global phases here are zero because we are guaranteed that M_1 and M_2 have unit determinant

    # Return the global phase obtained from A_M. It will be combined with initial_phase
    # in `two_qubit_decomposition`
    return math.cast_like(-e, initial_phase)


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


def _is_jax_jit(U):
    """Assume jax-jit if U is abstract and not in a capture or qjit context."""
    return math.is_abstract(U) and not (capture.enabled() or compiler.active())
