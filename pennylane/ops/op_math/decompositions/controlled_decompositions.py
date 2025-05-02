# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This submodule defines functions to decompose controlled operations."""

from pennylane import math, ops, queuing
from pennylane.decomposition import (
    DecompositionNotApplicable,
    adjoint_resource_rep,
    register_resources,
    resource_rep,
)
from pennylane.operation import Operator
from pennylane.wires import Wires


def ctrl_decomp_bisect(target_operation: Operator, control_wires: Wires):
    """Decompose the controlled version of a target single-qubit operation

    Not backpropagation compatible (as currently implemented). Use only with numpy.

    Automatically selects the best algorithm based on the matrix (uses specialized more efficient
    algorithms if the matrix has a certain form, otherwise falls back to the general algorithm).
    These algorithms are defined in section 3.1 and 3.2 of
    `Vale et al. (2023) <https://arxiv.org/abs/2302.06377>`_.

    .. warning:: This method will add a global phase for target operations that do not
        belong to the SU(2) group.

    Args:
        target_operation (~.operation.Operator): the target operation to decompose
        control_wires (~.wires.Wires): the control wires of the operation

    Returns:
        list[Operation]: the decomposed operations

    Raises:
        ValueError: if ``target_operation`` is not a single-qubit operation

    **Example:**

    >>> op = qml.T(0) # uses OD algorithm
    >>> print(qml.draw(ctrl_decomp_bisect, wire_order=(0,1,2,3,4,5), show_matrices=False)(op, (1,2,3,4,5)))
    0: ─╭X──U(M0)─╭X──U(M0)†─╭X──U(M0)─╭X──U(M0)†─┤
    1: ─├●────────│──────────├●────────│──────────┤
    2: ─├●────────│──────────├●────────│──────────┤
    3: ─╰●────────│──────────╰●────────│──────────┤
    4: ───────────├●───────────────────├●─────────┤
    5: ───────────╰●───────────────────╰●─────────┤
    >>> op = qml.QubitUnitary([[0,1j],[1j,0]], 0) # uses MD algorithm
    >>> print(qml.draw(ctrl_decomp_bisect, wire_order=(0,1,2,3,4,5), show_matrices=False)(op, (1,2,3,4,5)))
    0: ──H─╭X──U(M0)─╭X──U(M0)†─╭X──U(M0)─╭X──U(M0)†──H─┤
    1: ────├●────────│──────────├●────────│─────────────┤
    2: ────├●────────│──────────├●────────│─────────────┤
    3: ────╰●────────│──────────╰●────────│─────────────┤
    4: ──────────────├●───────────────────├●────────────┤
    5: ──────────────╰●───────────────────╰●────────────┤
    >>> op = qml.Hadamard(0) # uses general algorithm
    >>> print(qml.draw(ctrl_decomp_bisect, wire_order=(0,1,2,3,4,5), show_matrices=False)(op, (1,2,3,4,5)))
    0: ──U(M0)─╭X──U(M1)†──U(M2)─╭X──U(M2)†─╭X──U(M2)─╭X──U(M2)†─╭X──U(M1)─╭X──U(M0)─┤
    1: ────────│─────────────────│──────────├●────────│──────────├●────────│─────────┤
    2: ────────│─────────────────│──────────├●────────│──────────├●────────│─────────┤
    3: ────────│─────────────────│──────────╰●────────│──────────╰●────────│─────────┤
    4: ────────├●────────────────├●───────────────────├●───────────────────├●────────┤
    5: ────────╰●────────────────╰●───────────────────╰●───────────────────╰●────────┤

    """
    if len(target_operation.wires) > 1:
        raise ValueError(
            "The target operation must be a single-qubit operation, instead "
            f"got {target_operation}."
        )

    with queuing.AnnotatedQueue() as q:
        ctrl_decomp_bisect_rule(target_operation.matrix(), control_wires + target_operation.wires)

    # If there is an active queuing context, queue the decomposition so that expand works
    current_queue = queuing.QueuingManager.active_context()
    if current_queue is not None:
        for op in q.queue:  # pragma: no cover
            queuing.apply(op, context=current_queue)

    return q.queue


def _ctrl_decomp_bisect_resources(num_target_wires, num_control_wires, **__):

    # This decomposition rule is only applicable when the target is a single-qubit unitary.
    if num_target_wires > 1:
        raise DecompositionNotApplicable

    # This decomposition rule is not helpful when there's only a single control wire.
    if num_control_wires < 2:
        raise DecompositionNotApplicable

    len_k1 = (num_control_wires + 1) // 2
    len_k2 = num_control_wires - len_k1
    # this is a general overestimate based on the resource requirement of the general case.
    return {
        resource_rep(ops.QubitUnitary, num_wires=num_target_wires): 4,
        adjoint_resource_rep(ops.QubitUnitary, {"num_wires": num_target_wires}): 4,
        resource_rep(ops.MultiControlledX, num_control_wires=len_k2, num_work_wires=len_k1): 4,
        resource_rep(ops.MultiControlledX, num_control_wires=len_k1, num_work_wires=len_k2): 2,
        # we only need Hadamard for the md case, but it still needs to be accounted for.
        ops.Hadamard: 2,
        ops.GlobalPhase: 1,
    }


@register_resources(_ctrl_decomp_bisect_resources)
def ctrl_decomp_bisect_rule(U, wires, **__):
    """The decomposition rule for ControlledQubitUnitary."""
    U, phase = math.convert_to_su2(U, return_global_phase=True)
    imag_U = math.imag(U)
    ops.cond(
        math.allclose(imag_U[1, 0], 0) & math.allclose(imag_U[0, 1], 0),
        # Real off-diagonal specialized algorithm - 16n+O(1) CNOTs
        _ctrl_decomp_bisect_od,
        # General algorithm - 20n+O(1) CNOTs
        _ctrl_decomp_bisect_general,
        elifs=[
            (
                # Real main-diagonal specialized algorithm - 16n+O(1) CNOTs
                math.allclose(imag_U[0, 0], 0) & math.allclose(imag_U[1, 1], 0),
                _ctrl_decomp_bisect_md,
            )
        ],
    )(U, wires)
    ops.cond(~math.allclose(phase, 0), lambda: ops.GlobalPhase(-phase))


def _ctrl_decomp_bisect_general(U, wires):
    """Decompose the controlled version of a target single-qubit operation

    This function decomposes a controlled single-qubit target operation using the
    decomposition defined in section 3.2 of
    `Vale et al. (2023) <https://arxiv.org/abs/2302.06377>`_.

    Args:
        U (tensor): the target operation to decompose
        wires (WiresLike): the wires of the operation (control wires followed by the target wire)

    """

    x_matrix = ops.X.compute_matrix()
    h_matrix = ops.Hadamard.compute_matrix()
    alternate_h_matrix = x_matrix @ h_matrix @ x_matrix

    d, q = math.linalg.eig(U)
    d = math.diag(d)
    q = _convert_to_real_diagonal(q)
    b = _bisect_compute_b(q)
    c1 = math.matmul(b, alternate_h_matrix)
    c2t = math.matmul(b, h_matrix)

    mid = len(wires) // 2  # for odd n, make control_k1 bigger
    control_k1 = wires[:mid]
    control_k2 = wires[mid:-1]

    # The component
    ops.QubitUnitary(c2t, wires[-1])
    ops.ctrl(ops.X(wires[-1]), control=control_k2, work_wires=control_k1)
    ops.adjoint(ops.QubitUnitary(c1, wires[-1]))

    # Cancel the two identity controlled X gates
    _ctrl_decomp_bisect_od(d, wires, skip_initial_cx=True)

    # Adjoint of the component
    ops.ctrl(ops.X(wires[-1]), control=control_k1, work_wires=control_k2)
    ops.QubitUnitary(c1, wires[-1])
    ops.ctrl(ops.X(wires[-1]), control=control_k2, work_wires=control_k1)
    ops.adjoint(ops.QubitUnitary(c2t, wires[-1]))


def _ctrl_decomp_bisect_od(U, wires, skip_initial_cx=False):
    """Decompose the controlled version of a target single-qubit operation

    This function decomposes a controlled single-qubit target operation using the
    decomposition defined in section 3.1, Theorem 1 of
    `Vale et al. (2023) <https://arxiv.org/abs/2302.06377>`_.

    The target operation's matrix must have a real off-diagonal for this specialized method to work.

    Args:
        U (tensor): the target operation to decompose
        wires (WiresLike): the wires of the operation (control wires followed by the target wire)

    """
    a = _bisect_compute_a(U)

    mid = len(wires) // 2  # for odd n, make control_k1 bigger
    control_k1 = wires[:mid]
    control_k2 = wires[mid:-1]

    if not skip_initial_cx:
        ops.ctrl(ops.X(wires[-1]), control=control_k1, work_wires=control_k2)

    ops.QubitUnitary(a, wires[-1])
    ops.ctrl(ops.X(wires[-1]), control=control_k2, work_wires=control_k1)
    ops.adjoint(ops.QubitUnitary(a, wires[-1]))
    ops.ctrl(ops.X(wires[-1]), control=control_k1, work_wires=control_k2)
    ops.QubitUnitary(a, wires[-1])
    ops.ctrl(ops.X(wires[-1]), control=control_k2, work_wires=control_k1)
    ops.adjoint(ops.QubitUnitary(a, wires[-1]))


def _ctrl_decomp_bisect_md(U, wires):
    """Decompose the controlled version of a target single-qubit operation

    This function decomposes a controlled single-qubit target operation using the
    decomposition defined in section 3.1, Theorem 2 of
    `Vale et al. (2023) <https://arxiv.org/abs/2302.06377>`_.

    The target operation's matrix must have a real main-diagonal for this specialized method to work.

    Args:
        U (tensor): the target operation to decompose
        wires (WiresLike): the wires of the operation (control wires followed by the target wire)

    """
    h_matrix = ops.Hadamard.compute_matrix()
    mod_u = math.matmul(math.matmul(h_matrix, U), h_matrix)

    ops.H(wires[-1])
    _ctrl_decomp_bisect_od(mod_u, wires)
    ops.H(wires[-1])


def _convert_to_real_diagonal(q):
    """
    Change the phases of Q so the main diagonal is real, and return the modified Q.
    """
    exp_angles = math.angle(math.diag(q))
    return q * math.reshape(math.exp(-1j * exp_angles), (1, 2))


def _param_su2(ar, ai, br, bi):
    """
    Create a matrix in the SU(2) form from complex parameters a, b.
    The resulting matrix is not guaranteed to be in SU(2), unless |a|^2 + |b|^2 = 1.
    """
    return math.array([[ar + 1j * ai, -br + 1j * bi], [br + 1j * bi, ar + 1j * -ai]])


def _bisect_compute_a(u):
    """
    Given the U matrix, compute the A matrix such that
    At x A x At x A x = U
    where At is the adjoint of A
    and x is the Pauli X matrix.
    """
    x = math.real(u[0, 1])
    z = u[1, 1]
    zr = math.real(z)
    zi = math.imag(z)

    def _compute_a():
        ar = math.sqrt((math.sqrt((zr + 1) / 2) + 1) / 2)
        mul = 1 / (2 * math.sqrt((zr + 1) * (math.sqrt((zr + 1) / 2) + 1)))
        ai = zi * mul
        br = x * mul
        bi = 0
        return _param_su2(ar, ai, br, bi)

    return math.cond(
        math.allclose(zr, -1), lambda: math.array([[1, -1], [1, 1]]) * 2**-0.5, _compute_a, ()
    )


def _bisect_compute_b(u):
    """
    Given the U matrix, compute the B matrix such that
    H Bt x B x H = U
    where Bt is the adjoint of B,
    H is the Hadamard matrix,
    and x is the Pauli X matrix.
    """
    w = math.real(u[0, 0])
    s = math.real(u[1, 0])
    t = math.imag(u[1, 0])

    b = math.cond(
        math.allclose(s, 0),
        lambda: 0,
        lambda: math.cond(
            math.allclose(t, 0),
            lambda: (1 / 2 - w / 2) * math.sqrt(2 * w + 2) / s,
            lambda: math.sqrt(2) * s * math.sqrt((1 - w) / (s**2 + t**2)) * math.abs(t) / (2 * t),
            (),
        ),
        (),
    )

    c = math.cond(
        math.allclose(s, 0),
        lambda: math.cond(
            math.allclose(t, 0),
            lambda: math.cond(w < 0, lambda: 0, lambda: math.sqrt(w), ()),
            lambda: math.sqrt(2 - 2 * w) * (-w / 2 - 1 / 2) / t,
            (),
        ),
        lambda: math.cond(
            math.allclose(t, 0),
            lambda: math.sqrt(2 * w + 2) / 2,
            lambda: math.sqrt(2)
            * math.sqrt((1 - w) / (s**2 + t**2))
            * (w + 1)
            * math.abs(t)
            / (2 * t),
            (),
        ),
        (),
    )

    d = math.cond(
        math.allclose(s, 0),
        lambda: math.cond(
            math.allclose(t, 0),
            lambda: math.cond(w < 0, lambda: math.sqrt(-w), lambda: 0, ()),
            lambda: math.sqrt(2 - 2 * w) / 2,
            (),
        ),
        lambda: math.cond(
            math.allclose(t, 0),
            lambda: 0,
            lambda: -math.sqrt(2) * math.sqrt((1 - w) / (s**2 + t**2)) * math.abs(t) / 2,
            (),
        ),
        (),
    )

    return _param_su2(c, d, b, 0)
