# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Decomposition rules for controlled qubit unitary."""

from functools import partial
from typing import Literal

from scipy import sparse

from pennylane import capture, compiler, control_flow, math, ops
from pennylane.core import queuing
from pennylane.core.apply import apply
from pennylane.core.operator import Operation, Operator, abstractify
from pennylane.decomposition import add_decomps, register_condition, register_resources
from pennylane.decomposition.decomposition_rule import _get_registered_decomps
from pennylane.ops.functions import single_qubit_zyz_angles
from pennylane.ops.op_math.controlled_ops import ControlledQubitUnitary
from pennylane.typing import Bool, Complex, Float, Wire
from pennylane.wires import Wires, concatenate_wires


def ctrl_decomp_bisect(target_operation: Operator, control_wires: Wires):
    """Decompose the controlled version of a target single-qubit operation

    Not backpropagation compatible (as currently implemented). Use only with numpy.

    Automatically selects the best algorithm based on the matrix (uses specialized more efficient
    algorithms if the matrix has a certain form, otherwise falls back to the general algorithm).
    These algorithms are defined in sections 3.1 and 3.2 of
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

    >>> from pennylane.decomposition_rules import ctrl_decomp_bisect
    >>> op = qp.T(0) # uses OD algorithm
    >>> print(qp.draw(ctrl_decomp_bisect, wire_order=(0,1,2,3,4,5), show_matrices=False)(op, (1,2,3,4,5)))
    0: ─╭X──U(M0)─╭X──U(M0)†─╭X──U(M0)─╭X──U(M0)†─╭GlobalPhase(-0.39)─┤
    1: ─├●────────│──────────├●────────│──────────├●──────────────────┤
    2: ─├●────────│──────────├●────────│──────────├●──────────────────┤
    3: ─╰●────────│──────────╰●────────│──────────├●──────────────────┤
    4: ───────────├●───────────────────├●─────────├●──────────────────┤
    5: ───────────╰●───────────────────╰●─────────╰●──────────────────┤
    >>> op = qp.QubitUnitary([[0,1j],[1j,0]], 0) # uses MD algorithm
    >>> print(qp.draw(ctrl_decomp_bisect, wire_order=(0,1,2,3,4,5), show_matrices=False)(op, (1,2,3,4,5)))
    0: ──H─╭X──U(M0)─╭X──U(M0)†─╭X──U(M0)─╭X──U(M0)†──H─┤
    1: ────├●────────│──────────├●────────│─────────────┤
    2: ────├●────────│──────────├●────────│─────────────┤
    3: ────╰●────────│──────────╰●────────│─────────────┤
    4: ──────────────├●───────────────────├●────────────┤
    5: ──────────────╰●───────────────────╰●────────────┤
    >>> op = qp.Hadamard(0) # uses general algorithm
    >>> print(qp.draw(ctrl_decomp_bisect, wire_order=(0,1,2,3,4,5), show_matrices=False)(op, (1,2,3,4,5)))
    0: ──U(M0)─╭X──U(M1)†──U(M2)─╭X──U(M2)†─╭X──U(M2)─╭X──U(M2)†─╭X──U(M1)─╭X──U(M0)† ···
    1: ────────│─────────────────│──────────├●────────│──────────├●────────│───────── ···
    2: ────────│─────────────────│──────────├●────────│──────────├●────────│───────── ···
    3: ────────│─────────────────│──────────╰●────────│──────────╰●────────│───────── ···
    4: ────────├●────────────────├●───────────────────├●───────────────────├●──────── ···
    5: ────────╰●────────────────╰●───────────────────╰●───────────────────╰●──────── ···
    <BLANKLINE>
    0: ··· ─╭GlobalPhase(-1.57)─┤
    1: ··· ─├●──────────────────┤
    2: ··· ─├●──────────────────┤
    3: ··· ─├●──────────────────┤
    4: ··· ─├●──────────────────┤
    5: ··· ─╰●──────────────────┤

    """
    if len(target_operation.wires) > 1:
        raise ValueError(
            "The target operation must be a single-qubit operation, instead "
            f"got {target_operation}."
        )

    with queuing.AnnotatedQueue() as q:
        controlled_unitary_bisect(
            target_operation.matrix(),
            control_wires + target_operation.wires,
            control_values=[True] * len(control_wires),
        )

    # If there is an active queuing context, queue the decomposition so that expand works
    if queuing.QueuingManager.recording():
        for op in q.queue:  # pragma: no cover
            apply(op)

    return q.queue

def ctrl_decomp_zyz(
    target_operation: Operator,
    control_wires: Wires,
    work_wires: Wires | None = None,
    work_wire_type: str | None = "borrowed",
) -> list[Operation]:
    """Decompose the controlled version of a target single-qubit operation

    This function decomposes both single and multiple controlled single-qubit
    target operations using the decomposition defined in Lemma 4.3 and Lemma 5.1
    for single ``controlled_wires``, and Lemma 7.9 for multiple ``controlled_wires``
    from `Barenco et al. (1995) <https://arxiv.org/abs/quant-ph/9503016>`_.

    Args:
        target_operation (~.operation.Operator): the target operation or matrix to decompose
        control_wires (~.wires.Wires): the control wires of the operation.
        work_wires (~.wires.Wires): the work wires available for this decomposition
        work_wire_type (str): the type of work wires, either "zeroed" or "borrowed".

    Returns:
        list[Operation]: the decomposed operations

    Raises:
        ValueError: if ``target_operation`` is not a single-qubit operation

    **Example**

    We can create a controlled operation using ``qp.ctrl``, or by creating the
    decomposed controlled version using ``qp.decomposition_rules.ctrl_decomp_zyz``.

    .. code-block:: python

        dev = qp.device("default.qubit", wires=2)

        @qp.qnode(dev)
        def expected_circuit(op):
            qp.Hadamard(wires=0)
            qp.ctrl(op, [0])
            return qp.probs()

        @qp.qnode(dev)
        def decomp_circuit(op):
            qp.Hadamard(wires=0)
            qp.decomposition_rules.ctrl_decomp_zyz(op, [0])
            return qp.probs()

    Measurements on both circuits will give us the same results:

    >>> op = qp.RX(0.123, wires=1)
    >>> expected_circuit(op)
    array([0.5       , 0.        , 0.498..., 0.001...])

    >>> decomp_circuit(op)
    array([0.5       , 0.        , 0.498..., 0.001...])

    """
    if len(target_operation.wires) != 1:
        raise ValueError(
            "The target operation must be a single-qubit operation, instead "
            f"got {target_operation.__class__.__name__}."
        )

    control_wires = Wires(control_wires)
    target_wire = target_operation.wires

    *rot_angles, global_phase = single_qubit_zyz_angles(target_operation)
    with queuing.AnnotatedQueue() as q:
        all_wires = control_wires + target_wire
        if len(control_wires) > 1:
            _multi_control_zyz(
                *rot_angles, wires=all_wires, work_wires=work_wires, work_wire_type=work_wire_type
            )
        else:
            _single_control_zyz(*rot_angles, wires=all_wires)
        ops.cond(_not_zero(global_phase), _ctrl_global_phase)(
            global_phase,
            control_wires,
            work_wires,
            work_wire_type,
        )

    # If there is an active queuing context, queue the decomposition so that expand works
    if queuing.QueuingManager.recording():
        for op in q.queue:  # pragma: no cover
            apply(op)

    return q.queue


def _ctrl_decomp_bisect_condition(U, wires, *_, **__):
    num_target_wires = int(math.log2(math.shape(U)[-1]))
    num_control_wires = len(wires) - num_target_wires
    # This decomposition rule is only applicable when the target is a single-qubit unitary.
    # Also, it is not helpful when there's only a single control wire.
    return num_target_wires == 1 and num_control_wires > 1


# pylint: disable-next=unused-argument
def _controlled_unitary_bisect_resources(U, wires, control_values, *_, **__):

    num_ctrl_wires = len(wires) - 1
    len_k1 = (num_ctrl_wires + 1) // 2
    len_k2 = num_ctrl_wires - len_k1

    # this is a heuristic overestimate based on the resource requirement of the general case.
    mcx_k1 = ops.MultiControlledX(Wire[len_k1 + 1], work_wires=Wire[len_k2])
    mcx_k2 = ops.MultiControlledX(Wire[len_k2 + 1], work_wires=Wire[len_k1])
    mcx_reps = {mcx_k1: 6} if len_k1 == len_k2 else {mcx_k2: 4, mcx_k1: 2}
    return {
        ops.QubitUnitary(Complex[2, 2], wires=Wire[1]): 4,
        ops.adjoint(ops.QubitUnitary(Complex[2, 2], wires=Wire[1])): 4,
        # we only need Hadamard for the main diagonal case (see _ctrl_decomp_bisect_md),
        # but it still needs to be accounted for.
        ops.Hadamard: 2,
        ops.ctrl(ops.GlobalPhase(Float), Wire[num_ctrl_wires], work_wires=Wire[1]): 1,
        ops.X: num_ctrl_wires,  # assume half the control values are 0s
    } | mcx_reps


# Resources are not exact because rotations might be skipped for zero angles
@register_condition(_ctrl_decomp_bisect_condition)
@register_resources(_controlled_unitary_bisect_resources, exact=False, name="bisect")
def controlled_unitary_bisect(U, wires, control_values, *_, **__):
    """The decomposition rule for ControlledQubitUnitary from
    `Vale et al. (2023) <https://arxiv.org/abs/2302.06377>`_."""

    if compiler.active() or capture.enabled():
        wires = math.array(wires, like="jax")
        control_values = math.array(control_values, like="jax")

    @control_flow.for_loop(len(control_values))
    def _x_flips(i):
        ops.cond(math.logical_not(control_values[i]), ops.X)(wires[i])

    su2_U, phase = math.convert_to_su2(U)
    imag_U = math.imag(su2_U)

    _x_flips()  # pylint: disable=no-value-for-parameter
    real_main_diagonal = math.allclose(imag_U[0, 0], 0) & math.allclose(imag_U[1, 1], 0)
    ops.cond(
        math.allclose(imag_U[1, 0], 0) & math.allclose(imag_U[0, 1], 0),
        # Real off-diagonal specialized algorithm - 16n+O(1) CNOTs
        _ctrl_decomp_bisect_od,
        # General algorithm - 20n+O(1) CNOTs
        _ctrl_decomp_bisect_general,
        # Real main-diagonal specialized algorithm - 16n+O(1) CNOTs
        elifs=[(real_main_diagonal, _ctrl_decomp_bisect_md)],
    )(su2_U, wires)
    fn = partial(_ctrl_global_phase, work_wire_type="borrowed")
    ops.cond(_not_zero(phase), fn)(phase, wires[:-1], wires[-1])
    _x_flips()  # pylint: disable=no-value-for-parameter


def _single_ctrl_decomp_zyz_condition(U, wires, *_, **__):
    num_target_wires = int(math.log2(math.shape(U)[-1]))
    num_control_wires = len(wires) - num_target_wires
    return num_target_wires == 1 and num_control_wires == 1


# pylint: disable-next=unused-argument
def _controlled_unitary_single_control_zyz_resources(U, wires, **_):
    return {
        ops.RZ: 3,
        ops.RY: 2,
        ops.CNOT: 2,
        ops.ctrl(ops.GlobalPhase(Float), Wire[1]): 1,
        ops.X: len(wires) - 1,
    }


# Resources are not exact because rotations might be skipped for zero angles
@register_condition(_single_ctrl_decomp_zyz_condition)
@register_resources(
    _controlled_unitary_single_control_zyz_resources, exact=False, name="single_control_zyz"
)
# pylint: disable-next=unused-argument
def controlled_unitary_single_control_zyz(U, wires, control_values, work_wires, **_):
    """The decomposition rule for ControlledQubitUnitary from Lemma 5.1 of
    https://arxiv.org/pdf/quant-ph/9503016"""

    if capture.enabled() or compiler.active():
        wires = math.array(wires, like="jax")
        control_values = math.array(control_values, like="jax")

    @control_flow.for_loop(len(control_values))
    def _x_flips(i):
        ops.cond(math.logical_not(control_values[i]), ops.X)(wires[i])

    phi, theta, omega, phase = math.decomposition.zyz_rotation_angles(U)

    _x_flips()  # pylint: disable=no-value-for-parameter
    _single_control_zyz(phi, theta, omega, wires=wires)
    ops.cond(_not_zero(phase), _ctrl_global_phase)(phase, wires[:-1])
    _x_flips()  # pylint: disable=no-value-for-parameter


def _multi_ctrl_decomp_zyz_condition(U, wires, *_, **__):
    num_target_wires = int(math.log2(math.shape(U)[-1]))
    num_control_wires = len(wires) - num_target_wires
    return num_target_wires == 1 and num_control_wires > 1


def _controlled_unitary_multi_control_zyz_resources(  # pylint: disable=unused-argument
    U, wires, control_values, work_wires, work_wire_type, **_
):
    num_ctrl_wires = len(wires) - 1
    num_work_wires = len(work_wires)
    return {
        ops.CRZ: 3,
        ops.CRY: 2,
        ops.MultiControlledX(
            Wire[num_ctrl_wires],
            work_wires=Wire[num_work_wires],
            work_wire_type=work_wire_type,
        ): 2,
        ops.ctrl(ops.GlobalPhase(Float), Wire[num_ctrl_wires], work_wires=Wire[1]): 1,
        ops.X: num_ctrl_wires,
    }


@register_condition(_multi_ctrl_decomp_zyz_condition)
# Resources are not exact because rotations might be skipped for zero angle(s)
@register_resources(
    _controlled_unitary_multi_control_zyz_resources, exact=False, name="multi_control_zyz"
)
def controlled_unitary_multi_control_zyz(
    U, wires, control_values, work_wires, work_wire_type, **_
):
    """The decomposition rule for ControlledQubitUnitary from Lemma 7.9 of
    https://arxiv.org/pdf/quant-ph/9503016"""

    if compiler.active() or capture.enabled():
        wires = math.array(wires, like="jax")
        control_values = math.array(control_values, like="jax")

    @control_flow.for_loop(len(control_values))
    def _x_flips(i):
        ops.cond(math.logical_not(control_values[i]), ops.X)(wires[i])

    phi, theta, omega, phase = math.decomposition.zyz_rotation_angles(U)

    _x_flips()  # pylint: disable=no-value-for-parameter
    _multi_control_zyz(
        phi,
        theta,
        omega,
        wires=wires,
        work_wires=work_wires,
        work_wire_type=work_wire_type,
    )
    fn = partial(_ctrl_global_phase, work_wire_type="borrowed")
    ops.cond(_not_zero(phase), fn)(phase, wires[:-1], wires[-1])
    _x_flips()  # pylint: disable=no-value-for-parameter


def _controlled_unitary_two_qubit_resources(  # pylint: disable=unused-argument
    U, wires, control_values, work_wires, work_wire_type, **_
):
    base_rule = _get_registered_decomps("QubitUnitary")["two_qubit"]

    num_target_wires = int(math.log2(math.shape(U)[-1]))
    num_control_wires = len(wires) - num_target_wires
    num_work_wires = len(work_wires)
    base_resources = base_rule.compute_resources(num_wires=num_target_wires)
    gate_counts = {
        ops.ctrl(
            base_op_rep,
            Wire[num_control_wires],
            work_wires=Wire[num_work_wires],
            work_wire_type=work_wire_type,
        ): count
        for base_op_rep, count in base_resources.gate_counts.items()
    }
    # The impl applies X gates in pairs to flip any zero control values. With abstract inputs the
    # concrete control values (and thus the exact number of X gates) are unknown, so we declare a
    # heuristic of half the control wires using zero as their control values. This equates to
    # X_count = (num_control_wires // 2) * 2 ~= num_control_wires. This keeps X in the estimated
    # gate set (required even for exact=False rules) while the actual count is allowed to be lower.
    gate_counts[ops.X] = num_control_wires
    return gate_counts


# Resources are not exact because rotations might be skipped for zero angle(s)
@register_condition(lambda U, *_, **__: math.ceil_log2(math.shape(U)[0]) == 2)
@register_resources(_controlled_unitary_two_qubit_resources, exact=False, name="two_qubit")
def controlled_unitary_two_qubit(U, wires, control_values, work_wires, work_wire_type, **_):
    """A controlled two-qubit unitary is decomposed by applying ctrl to the base decomposition."""

    # Fetch the built-in rule even when QubitUnitary has exclusions or a fixed rule.
    # This preserves the direct dependency without importing the rule library from ops.
    base_rule = _get_registered_decomps("QubitUnitary")["two_qubit"]

    if compiler.active() or capture.enabled():
        wires = math.array(wires, like="jax")
        control_values = math.array(control_values, like="jax")

    @control_flow.for_loop(len(control_values))
    def _x_flips(i):
        ops.cond(math.logical_not(control_values[i]), ops.X)(wires[i])

    _x_flips()  # pylint: disable=no-value-for-parameter
    ops.ctrl(
        base_rule._impl,  # pylint: disable=protected-access
        control=wires[:-2],
        work_wires=work_wires,
        work_wire_type=work_wire_type,
    )(U, wires=wires[-2:])
    _x_flips()  # pylint: disable=no-value-for-parameter


def _not_zero(x):
    return math.logical_not(math.allclose(x, 0))


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
    ctrl_k1 = wires[:mid]
    ctrl_k2 = wires[mid:-1]

    # The component
    ops.QubitUnitary(c2t, wires[-1])
    mcx_wires_k2 = concatenate_wires(ctrl_k2, wires[-1:])
    ops.MultiControlledX(mcx_wires_k2, work_wires=ctrl_k1, work_wire_type="borrowed")
    ops.adjoint(ops.QubitUnitary(c1, wires[-1]))

    # Cancel the two identity controlled X gates
    _ctrl_decomp_bisect_od(d, wires, skip_initial_cx=True)

    # Adjoint of the component
    mcx_wires_k1 = concatenate_wires(ctrl_k1, wires[-1:])
    ops.MultiControlledX(mcx_wires_k1, work_wires=ctrl_k2, work_wire_type="borrowed")
    ops.QubitUnitary(c1, wires[-1])
    ops.MultiControlledX(mcx_wires_k2, work_wires=ctrl_k1, work_wire_type="borrowed")
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
    ctrl_k1 = wires[:mid]
    ctrl_k2 = wires[mid:-1]
    mcx_wires_k1 = concatenate_wires(ctrl_k1, wires[-1:])
    mcx_wires_k2 = concatenate_wires(ctrl_k2, wires[-1:])

    if not skip_initial_cx:
        ops.MultiControlledX(mcx_wires_k1, work_wires=ctrl_k2, work_wire_type="borrowed")

    ops.QubitUnitary(a, wires[-1])
    ops.MultiControlledX(mcx_wires_k2, work_wires=ctrl_k1, work_wire_type="borrowed")
    ops.adjoint(ops.QubitUnitary(a, wires[-1]))
    ops.MultiControlledX(mcx_wires_k1, work_wires=ctrl_k2, work_wire_type="borrowed")
    ops.QubitUnitary(a, wires[-1])
    ops.MultiControlledX(mcx_wires_k2, work_wires=ctrl_k1, work_wire_type="borrowed")
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


def _param_su2(ar, ai, br, bi=0):
    """
    Create a matrix in the SU(2) form from complex parameters a, b.
    The resulting matrix is not guaranteed to be in SU(2), unless |a|^2 + |b|^2 = 1.
    """
    return math.stack(
        [math.stack([ar + 1j * ai, -br + 1j * bi]), math.stack([br + 1j * bi, ar + 1j * -ai])]
    )


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
        math.allclose(zr, -1),
        lambda: math.array([[1 + 0j, -1 + 0j], [1 + 0j, 1 + 0j]]) * 2**-0.5,
        _compute_a,
        (),
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
        lambda: 0.0,
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
            lambda: math.cond(w < 0, lambda: 0.0, lambda: math.sqrt(w), ()),
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
            lambda: math.cond(w < 0, lambda: math.sqrt(-w), lambda: 0.0, ()),
            lambda: math.sqrt(2 - 2 * w) / 2,
            (),
        ),
        lambda: math.cond(
            math.allclose(t, 0),
            lambda: 0.0,
            lambda: -math.sqrt(2) * math.sqrt((1 - w) / (s**2 + t**2)) * math.abs(t) / 2,
            (),
        ),
        (),
    )

    return _param_su2(c, d, b, 0)


def _single_control_zyz(phi, theta, omega, wires):
    """Implements Lemma 5.1 from https://arxiv.org/pdf/quant-ph/9503016"""

    # Operator A
    ops.cond(_not_zero(phi), ops.RZ)(phi, wires=wires[-1])
    ops.cond(_not_zero(theta), ops.RY)(theta / 2, wires=wires[-1])

    ops.CNOT(wires)

    # Operator B
    ops.cond(_not_zero(theta), ops.RY)(-theta / 2, wires=wires[-1])
    ops.cond(_not_zero(phi + omega), ops.RZ)(-(phi + omega) / 2, wires=wires[-1])

    ops.CNOT(wires)

    # Operator C
    ops.cond(_not_zero(omega - phi), ops.RZ)((omega - phi) / 2, wires=wires[-1])


# pylint: disable-next=too-many-arguments
def _multi_control_zyz(phi, theta, omega, wires, work_wires, work_wire_type):
    """Implements Lemma 7.9 from https://arxiv.org/pdf/quant-ph/9503016"""

    # Operator A
    ops.cond(_not_zero(phi), ops.CRZ)(phi, wires=wires[-2:])
    ops.cond(_not_zero(theta), ops.CRY)(theta / 2, wires=wires[-2:])

    ops.MultiControlledX(
        concatenate_wires(wires[:-2], wires[-1:]),
        work_wires=work_wires,
        work_wire_type=work_wire_type,
    )

    # Operator B
    ops.cond(_not_zero(theta), ops.CRY)(-theta / 2, wires=wires[-2:])
    ops.cond(_not_zero(phi + omega), ops.CRZ)(-(phi + omega) / 2, wires=wires[-2:])

    ops.MultiControlledX(
        concatenate_wires(wires[:-2], wires[-1:]),
        work_wires=work_wires,
        work_wire_type=work_wire_type,
    )

    # Operator C
    ops.cond(_not_zero(omega - phi), ops.CRZ)((omega - phi) / 2, wires=wires[-2:])


def _ctrl_global_phase(
    phase,
    control_wires,
    work_wires=None,
    work_wire_type: Literal["zeroed", "borrowed"] = "borrowed",
):
    ops.ctrl(
        ops.GlobalPhase(-phase),
        control=control_wires,
        work_wires=work_wires,
        work_wire_type=work_wire_type,
    )


# pylint: disable-next=unused-argument
def _to_general_c_qu_standard_resources(U, wires, control_values, work_wires, work_wire_type, **_):
    num_target_wires = int(math.log2(math.shape(U)[-1]))
    num_control_wires = len(wires) - num_target_wires
    num_work_wires = len(work_wires)
    u_size = 2**num_target_wires
    return {
        ops.ControlledOp2(
            ops.QubitUnitary(Complex[u_size, u_size], wires=Wire[num_target_wires]),
            control_wires=Wire[num_control_wires],
            control_values=Bool[num_control_wires],
            work_wires=Wire[num_work_wires],
            work_wire_type=work_wire_type,
        ): 1
    }


@register_condition(lambda U, *_, **__: math.ceil_log2(math.shape(U)[0]) > 2)
@register_resources(_to_general_c_qu_standard_resources, name="to_controlled_qubit_unitary")
def to_general_c_qu_standard(U, wires, control_values, work_wires, work_wire_type, **_):
    """Convert a ControlledQubitUnitary to a general Controlled(QubitUnitary) so that
    the graph finds the general decomposition rule of applying control to the decomposition
    of the base QubitUnitary."""
    num_target_wires = len(wires) - len(control_values)
    ops.ControlledOp2(
        ops.QubitUnitary(U, wires=wires[-num_target_wires:]),
        control_wires=wires[:-num_target_wires],
        control_values=control_values,
        work_wires=work_wires,
        work_wire_type=work_wire_type,
    )


add_decomps(
    ControlledQubitUnitary,
    controlled_unitary_bisect,
    controlled_unitary_single_control_zyz,
    controlled_unitary_multi_control_zyz,
    controlled_unitary_two_qubit,
    to_general_c_qu_standard,
)


def _conj_transpose(U):
    return U.conjugate().transpose() if sparse.issparse(U) else math.moveaxis(math.conj(U), -2, -1)


@register_resources(lambda base: {abstractify(base): 1}, name="conjugate_transpose")
def adjoint_controlled_qubit_unitary_standard(base):
    """Decompose ``Adjoint(ControlledQubitUnitary)`` using the ``conjugate_transpose`` strategy."""
    ControlledQubitUnitary(
        _conj_transpose(base.U),
        wires=base.wires,
        control_values=base.control_values,
        work_wires=base.work_wires,
        work_wire_type=base.work_wire_type,
    )


add_decomps("Adjoint(ControlledQubitUnitary)", adjoint_controlled_qubit_unitary_standard)
