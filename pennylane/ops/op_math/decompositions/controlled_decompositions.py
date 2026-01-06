# Copyright 2018-2025 Xanadu Quantum Technologies Inc.

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


from typing import Literal

import numpy as np

import pennylane as qml
from pennylane import allocation, control_flow, math, ops, queuing
from pennylane.decomposition import (
    adjoint_resource_rep,
    controlled_resource_rep,
    register_condition,
    register_resources,
    resource_rep,
)
from pennylane.decomposition.symbolic_decomposition import flip_zero_control
from pennylane.operation import Operation, Operator
from pennylane.ops.op_math.decompositions.unitary_decompositions import two_qubit_decomp_rule
from pennylane.wires import Wires


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

    >>> from pennylane.ops.op_math import ctrl_decomp_bisect
    >>> op = qml.T(0) # uses OD algorithm
    >>> print(qml.draw(ctrl_decomp_bisect, wire_order=(0,1,2,3,4,5), show_matrices=False)(op, (1,2,3,4,5)))
    0: ─╭X──U(M0)─╭X──U(M0)†─╭X──U(M0)─╭X──U(M0)†─╭GlobalPhase(-0.39)─┤
    1: ─├●────────│──────────├●────────│──────────├●──────────────────┤
    2: ─├●────────│──────────├●────────│──────────├●──────────────────┤
    3: ─╰●────────│──────────╰●────────│──────────├●──────────────────┤
    4: ───────────├●───────────────────├●─────────├●──────────────────┤
    5: ───────────╰●───────────────────╰●─────────╰●──────────────────┤
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
        ctrl_decomp_bisect_rule(target_operation.matrix(), control_wires + target_operation.wires)

    # If there is an active queuing context, queue the decomposition so that expand works
    if queuing.QueuingManager.recording():
        for op in q.queue:  # pragma: no cover
            queuing.apply(op)

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

    We can create a controlled operation using ``qml.ctrl``, or by creating the
    decomposed controlled version using ``qml.ctrl_decomp_zyz``.

    .. code-block:: python

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def expected_circuit(op):
            qml.Hadamard(wires=0)
            qml.ctrl(op, [0])
            return qml.probs()

        @qml.qnode(dev)
        def decomp_circuit(op):
            qml.Hadamard(wires=0)
            qml.ops.ctrl_decomp_zyz(op, [0])
            return qml.probs()

    Measurements on both circuits will give us the same results:

    >>> op = qml.RX(0.123, wires=1)
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

    if isinstance(target_operation, Operation):
        try:
            rot_angles = target_operation.single_qubit_rot_angles()
            _, global_phase = math.convert_to_su2(
                ops.functions.matrix(target_operation), return_global_phase=True
            )
        except NotImplementedError:
            *rot_angles, global_phase = math.decomposition.zyz_rotation_angles(
                ops.functions.matrix(target_operation), return_global_phase=True
            )
    else:
        *rot_angles, global_phase = math.decomposition.zyz_rotation_angles(
            ops.functions.matrix(target_operation), return_global_phase=True
        )

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
            queuing.apply(op)

    return q.queue


#######################
# Decomposition Rules #
#######################


def _ctrl_decomp_bisect_condition(num_target_wires, num_control_wires, **__):
    # This decomposition rule is only applicable when the target is a single-qubit unitary.
    # Also, it is not helpful when there's only a single control wire.
    return num_target_wires == 1 and num_control_wires > 1


def _ctrl_decomp_bisect_resources(num_target_wires, num_control_wires, **__):

    len_k1 = (num_control_wires + 1) // 2
    len_k2 = num_control_wires - len_k1
    # this is a general overestimate based on the resource requirement of the general case.
    if len_k1 == len_k2:
        return {
            resource_rep(ops.QubitUnitary, num_wires=num_target_wires): 4,
            adjoint_resource_rep(ops.QubitUnitary, {"num_wires": num_target_wires}): 4,
            controlled_resource_rep(
                ops.X,
                {},
                num_control_wires=len_k2,
                num_work_wires=len_k1,
                work_wire_type="borrowed",
            ): 6,
            # we only need Hadamard for the main diagonal case (see _ctrl_decomp_bisect_md), but it still needs to be accounted for.
            ops.Hadamard: 2,
            controlled_resource_rep(
                ops.GlobalPhase,
                {},
                num_control_wires=num_control_wires,
                num_work_wires=1,
                work_wire_type="borrowed",
            ): 1,
        }
    return {
        resource_rep(ops.QubitUnitary, num_wires=num_target_wires): 4,
        adjoint_resource_rep(ops.QubitUnitary, {"num_wires": num_target_wires}): 4,
        controlled_resource_rep(
            ops.X,
            {},
            num_control_wires=len_k2,
            num_work_wires=len_k1,
            work_wire_type="borrowed",
        ): 4,
        controlled_resource_rep(
            ops.X,
            {},
            num_control_wires=len_k1,
            num_work_wires=len_k2,
            work_wire_type="borrowed",
        ): 2,
        # we only need Hadamard for the main diagonal case (see _ctrl_decomp_bisect_md), but it still needs to be accounted for.
        ops.Hadamard: 2,
        controlled_resource_rep(
            ops.GlobalPhase,
            {},
            num_control_wires=num_control_wires,
            num_work_wires=1,
            work_wire_type="borrowed",
        ): 1,
    }


# Resources are not exact because rotations might be skipped for zero angles
@register_condition(_ctrl_decomp_bisect_condition)
@register_resources(_ctrl_decomp_bisect_resources, exact=False)
def ctrl_decomp_bisect_rule(U, wires, **__):
    """The decomposition rule for ControlledQubitUnitary from
    `Vale et al. (2023) <https://arxiv.org/abs/2302.06377>`_."""
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
    ops.cond(_not_zero(phase), _ctrl_global_phase)(phase, wires[:-1], wires[-1], "borrowed")


def _single_ctrl_decomp_zyz_condition(num_target_wires, num_control_wires, **__):
    return num_target_wires == 1 and num_control_wires == 1


def _single_ctrl_decomp_zyz_resources(**__):
    return {
        ops.RZ: 3,
        ops.RY: 2,
        ops.CNOT: 2,
        controlled_resource_rep(ops.GlobalPhase, {}, num_control_wires=1): 1,
    }


# Resources are not exact because rotations might be skipped for zero angles
@register_condition(_single_ctrl_decomp_zyz_condition)
@register_resources(_single_ctrl_decomp_zyz_resources, exact=False)
def single_ctrl_decomp_zyz_rule(U, wires, **__):
    """The decomposition rule for ControlledQubitUnitary from Lemma 5.1 of
    https://arxiv.org/pdf/quant-ph/9503016"""

    phi, theta, omega, phase = math.decomposition.zyz_rotation_angles(U, return_global_phase=True)
    _single_control_zyz(phi, theta, omega, wires=wires)
    ops.cond(_not_zero(phase), _ctrl_global_phase)(phase, wires[:-1])


def _multi_ctrl_decomp_zyz_condition(num_target_wires, num_control_wires, **__):
    return num_target_wires == 1 and num_control_wires > 1


def _multi_ctrl_decomp_zyz_resources(num_control_wires, num_work_wires, work_wire_type, **__):
    return {
        ops.CRZ: 3,
        ops.CRY: 2,
        controlled_resource_rep(
            ops.X,
            {},
            num_control_wires=num_control_wires - 1,
            num_work_wires=num_work_wires,
            work_wire_type=work_wire_type,
        ): 2,
        controlled_resource_rep(
            ops.GlobalPhase,
            {},
            num_control_wires=num_control_wires,
            num_work_wires=1,
            work_wire_type="borrowed",
        ): 1,
    }


# Resources are not exact because rotations might be skipped for zero angle(s)
@register_condition(_multi_ctrl_decomp_zyz_condition)
@register_resources(_multi_ctrl_decomp_zyz_resources, exact=False)
def multi_control_decomp_zyz_rule(U, wires, work_wires, work_wire_type, **__):
    """The decomposition rule for ControlledQubitUnitary from Lemma 7.9 of
    https://arxiv.org/pdf/quant-ph/9503016"""

    phi, theta, omega, phase = math.decomposition.zyz_rotation_angles(U, return_global_phase=True)
    _multi_control_zyz(
        phi,
        theta,
        omega,
        wires=wires,
        work_wires=work_wires,
        work_wire_type=work_wire_type,
    )
    ops.cond(_not_zero(phase), _ctrl_global_phase)(phase, wires[:-1], wires[-1], "borrowed")


def _controlled_two_qubit_unitary_resource(
    num_target_wires,
    num_control_wires,
    num_zero_control_values,
    num_work_wires,
    work_wire_type,
    **__,
):
    base_resources = two_qubit_decomp_rule.compute_resources(num_wires=num_target_wires)
    gate_counts = {
        controlled_resource_rep(
            base_class=base_op_rep.op_type,
            base_params=base_op_rep.params,
            num_control_wires=num_control_wires,
            num_zero_control_values=0,
            num_work_wires=num_work_wires,
            work_wire_type=work_wire_type,
        ): count
        for base_op_rep, count in base_resources.gate_counts.items()
    }
    gate_counts[ops.X] = num_zero_control_values * 2
    return gate_counts


# Resources are not exact because rotations might be skipped for zero angle(s)
@register_condition(lambda num_target_wires, **_: num_target_wires == 2)
@register_resources(_controlled_two_qubit_unitary_resource, exact=False)
def controlled_two_qubit_unitary_rule(U, wires, control_values, work_wires, work_wire_type, **__):
    """A controlled two-qubit unitary is decomposed by applying ctrl to the base decomposition."""
    zero_control_wires = [w for w, val in zip(wires[:-2], control_values) if not val]
    for w in zero_control_wires:
        ops.PauliX(w)
    ops.ctrl(
        two_qubit_decomp_rule._impl,  # pylint: disable=protected-access
        control=wires[:-2],
        work_wires=work_wires,
        work_wire_type=work_wire_type,
    )(U, wires=wires[-2:])
    for w in zero_control_wires:
        ops.PauliX(w)


def _mcx_many_workers_condition(num_control_wires, num_work_wires, **__):
    return num_control_wires > 2 and num_work_wires >= num_control_wires - 2


def _mcx_many_workers_resource(num_control_wires, work_wire_type, **__):

    if work_wire_type == "borrowed":
        return {ops.Toffoli: 4 * (num_control_wires - 2)}
    return {
        qml.TemporaryAND: num_control_wires - 2,
        adjoint_resource_rep(qml.TemporaryAND): num_control_wires - 2,
        ops.Toffoli: 1,
    }


# pylint: disable=no-value-for-parameter
@register_condition(_mcx_many_workers_condition)
@register_resources(_mcx_many_workers_resource)
def _mcx_many_workers(wires, work_wires, work_wire_type, **__):
    """Decomposes the multi-controlled PauliX gate using the approach in Lemma 7.2 of
    https://arxiv.org/abs/quant-ph/9503016, which requires a suitably large register of
    work wires"""
    target_wire, control_wires = wires[-1], wires[:-1]
    work_wires = work_wires[: len(control_wires) - 2]

    if work_wire_type == "borrowed":
        up_gate = down_gate = ops.Toffoli
    else:
        down_gate = qml.TemporaryAND
        up_gate = ops.adjoint(qml.TemporaryAND)

    @control_flow.for_loop(1, len(work_wires), 1)
    def loop_up(i):
        up_gate(wires=[control_wires[i], work_wires[i], work_wires[i - 1]])

    @control_flow.for_loop(len(work_wires) - 1, 0, -1)
    def loop_down(i):
        down_gate(wires=[control_wires[i], work_wires[i], work_wires[i - 1]])

    if work_wire_type == "borrowed":
        ops.Toffoli(wires=[control_wires[0], work_wires[0], target_wire])
        loop_up()

    down_gate(wires=[control_wires[-1], control_wires[-2], work_wires[-1]])
    loop_down()
    ops.Toffoli(wires=[control_wires[0], work_wires[0], target_wire])
    loop_up()
    up_gate(wires=[control_wires[-1], control_wires[-2], work_wires[-1]])

    if work_wire_type == "borrowed":
        loop_down()


decompose_mcx_many_workers_explicit = flip_zero_control(_mcx_many_workers)


@register_condition(lambda num_work_wires, **_: not num_work_wires)
@register_condition(lambda num_control_wires, **_: num_control_wires > 2)
@register_resources(
    lambda num_control_wires, **_: _mcx_many_workers_resource(num_control_wires, "zeroed"),
    work_wires=lambda num_control_wires, **_: {"zeroed": num_control_wires - 2},
)
def _mcx_many_zeroed_workers(wires, **kwargs):
    num_control_wires = len(wires) - 1
    num_work_wires = num_control_wires - 2
    with allocation.allocate(num_work_wires, state="zero", restored=True) as work_wires:
        kwargs.update({"work_wires": work_wires, "work_wire_type": "zeroed"})
        _mcx_many_workers(wires, **kwargs)


decompose_mcx_many_zeroed_workers = flip_zero_control(_mcx_many_zeroed_workers)


@register_condition(lambda num_work_wires, **_: not num_work_wires)
@register_condition(lambda num_control_wires, **_: num_control_wires > 2)
@register_resources(
    lambda num_control_wires, **_: _mcx_many_workers_resource(num_control_wires, "borrowed"),
    work_wires=lambda num_control_wires, **_: {"borrowed": num_control_wires - 2},
)
def _mcx_many_borrowed_workers(wires, **kwargs):
    num_control_wires = len(wires) - 1
    num_work_wires = num_control_wires - 2
    with allocation.allocate(num_work_wires, state="any", restored=True) as work_wires:
        kwargs.update({"work_wires": work_wires, "work_wire_type": "borrowed"})
        _mcx_many_workers(wires, **kwargs)


decompose_mcx_many_borrowed_workers = flip_zero_control(_mcx_many_borrowed_workers)


def _mcx_two_workers_condition(num_control_wires, num_work_wires, **__):
    return num_control_wires > 2 and (
        num_work_wires >= 2 or (num_work_wires == 1 and num_control_wires < 6)
    )


def _mcx_two_workers_resource(num_control_wires, work_wire_type, **__):

    is_small_mcx = num_control_wires < 6

    if work_wire_type == "zeroed":
        n_ccx = 2 * num_control_wires - 3
        n_temporary_ccx_pairs = 2 - is_small_mcx
        return {
            ops.Toffoli: n_ccx - 2 * n_temporary_ccx_pairs,
            ops.X: n_ccx - 3 if is_small_mcx else n_ccx - 5,
            qml.TemporaryAND: n_temporary_ccx_pairs,
            adjoint_resource_rep(qml.TemporaryAND): n_temporary_ccx_pairs,
        }
    # Otherwise, we assume the work wires are borrowed
    n_ccx = 4 * num_control_wires - 8
    return {ops.Toffoli: n_ccx, ops.X: n_ccx - 4 if is_small_mcx else n_ccx - 8}


@register_condition(_mcx_two_workers_condition)
@register_resources(_mcx_two_workers_resource)
def _mcx_two_workers(wires, work_wires, work_wire_type, **__):
    r"""
    Synthesise a multi-controlled X gate with :math:`k` controls using :math:`2` auxiliary qubits.
    It produces a circuit with :math:`2k-3` Toffoli gates and depth :math:`O(\log(k))` if using
    zeroed auxiliary qubits, and :math:`4k-8` Toffoli gates and depth :math:`O(\log(k))` if using borrowed
    auxiliary qubits as described in Sec. 5 of [1].

    References:
        1. Khattar and Gidney, Rise of conditionally clean ancillae for optimizing quantum circuits
        `arXiv:2407.17966 <https://arxiv.org/abs/2407.17966>`__

    """
    # Unpack work wires for readability. There might just be one of them if it is a "small" MCX
    # (less than 6 controls)
    work0, *work1 = work_wires
    # First use the work wire to prepare the first two control wires as conditionally clean.
    left_elbow = ops.Toffoli if work_wire_type == "borrowed" else qml.TemporaryAND
    left_elbow([wires[0], wires[1], work0])

    middle_ctrl_indices = _build_log_n_depth_ccx_ladder(wires[:-1])

    # Apply the MCX in the middle. This is just a single Toffoli without work wires for "small" MCX
    if len(middle_ctrl_indices) == 1:
        ops.Toffoli([work0, wires[middle_ctrl_indices[0]], wires[-1]])
    else:
        middle_wires = [wires[i] for i in middle_ctrl_indices]
        # No toggle detection needed for the inner MCX decomposition, even for borrowed work wires
        _mcx_one_worker(
            [work0] + middle_wires + wires[-1:],
            work1,
            work_wire_type=work_wire_type,
            _skip_toggle_detection=True,
        )

    # Uncompute the first ladder
    ops.adjoint(_build_log_n_depth_ccx_ladder, lazy=False)(wires[:-1])

    right_elbow = ops.Toffoli if work_wire_type == "borrowed" else qml.adjoint(qml.TemporaryAND)
    right_elbow([wires[0], wires[1], work0])

    if work_wire_type == "borrowed":
        # Perform toggle-detection if the work wire is borrowed
        middle_ctrl_indices = _build_log_n_depth_ccx_ladder(wires[:-1])
        if len(middle_ctrl_indices) == 1:
            ops.Toffoli([work0, wires[middle_ctrl_indices[0]], wires[-1]])
        else:
            middle_wires = [wires[i] for i in middle_ctrl_indices]
            _mcx_one_worker(
                [work0] + middle_wires + wires[-1:],
                work1,
                work_wire_type=work_wire_type,
                _skip_toggle_detection=True,
            )

        ops.adjoint(_build_log_n_depth_ccx_ladder, lazy=False)(wires[:-1])


decompose_mcx_two_workers_explicit = flip_zero_control(_mcx_two_workers)


@register_condition(lambda num_work_wires, **_: not num_work_wires)
@register_condition(lambda num_control_wires, **_: num_control_wires > 2)
@register_resources(
    lambda num_control_wires, **_: _mcx_two_workers_resource(num_control_wires, "zeroed"),
    work_wires=lambda num_control_wires, **_: {"zeroed": 1 + (num_control_wires >= 6)},
)
def _mcx_two_zeroed_workers(wires, **kwargs):
    is_small_mcx = (len(wires) - 1) < 6
    with allocation.allocate(2 - is_small_mcx, state="zero", restored=True) as work_wires:
        kwargs.update({"work_wires": work_wires, "work_wire_type": "zeroed"})
        _mcx_two_workers(wires, **kwargs)


decompose_mcx_two_zeroed_workers = flip_zero_control(_mcx_two_zeroed_workers)


@register_condition(lambda num_work_wires, **_: not num_work_wires)
@register_condition(lambda num_control_wires, **_: num_control_wires > 2)
@register_resources(
    lambda num_control_wires, **_: _mcx_two_workers_resource(num_control_wires, "borrowed"),
    work_wires=lambda num_control_wires, **_: {"borrowed": 2 - (num_control_wires < 6)},
)
def _mcx_two_borrowed_workers(wires, **kwargs):
    is_small_mcx = (len(wires) - 1) < 6
    with allocation.allocate(2 - is_small_mcx, state="any", restored=True) as work_wires:
        kwargs.update({"work_wires": work_wires, "work_wire_type": "borrowed"})
        _mcx_two_workers(wires, **kwargs)


decompose_mcx_two_borrowed_workers = flip_zero_control(_mcx_two_borrowed_workers)


def _mcx_one_worker_condition(num_control_wires, num_work_wires, **__):
    return num_control_wires > 2 and num_work_wires == 1


def _mcx_one_worker_resource(num_control_wires, work_wire_type, **__):
    if work_wire_type == "zeroed":
        n_ccx = 2 * num_control_wires - 5
        return {
            ops.Toffoli: n_ccx,
            qml.TemporaryAND: 1,
            adjoint_resource_rep(qml.TemporaryAND): 1,
            ops.X: n_ccx - 1,
        }
    # Otherwise, we assume the work wire is borrowed
    n_ccx = 4 * num_control_wires - 8
    return {ops.Toffoli: n_ccx, ops.X: n_ccx - 4}


@register_condition(_mcx_one_worker_condition)
@register_resources(_mcx_one_worker_resource)
def _mcx_one_worker(wires, work_wires, work_wire_type="zeroed", _skip_toggle_detection=False, **__):
    r"""
    Synthesise a multi-controlled X gate with :math:`k` controls using :math:`1` auxiliary qubit. It
    produces a circuit with :math:`2k-3` Toffoli gates and depth :math:`O(k)` if the auxiliary is zeroed
    and :math:`4k-3` Toffoli gates and depth :math:`O(k)` if the auxiliary is borrowed as described in
    Sec. 5.1 of [1].

    .. note::

        The keyword argument ``_skip_toggle_detection`` is only supposed to be used when utilizing
        ``_mcx_one_worker`` as a subroutine within a decomposition rule, but not when using
        it as a decomposition rule itself. This is because ``_mcx_one_worker_resource`` does not
        support/take into account this keyword argument.

    References:
        1. Khattar and Gidney, Rise of conditionally clean ancillae for optimizing quantum circuits
        `arXiv:2407.17966 <https://arxiv.org/abs/2407.17966>`__

    """
    if work_wire_type == "borrowed":
        ops.Toffoli([wires[0], wires[1], work_wires[0]])
    else:
        _skip_toggle_detection = True
        qml.TemporaryAND([wires[0], wires[1], work_wires[0]])

    final_ctrl_index = _build_linear_depth_ladder(wires[:-1])
    ops.Toffoli([work_wires[0], wires[final_ctrl_index], wires[-1]])
    ops.adjoint(_build_linear_depth_ladder, lazy=False)(wires[:-1])

    if work_wire_type == "borrowed":
        ops.Toffoli([wires[0], wires[1], work_wires[0]])
    else:
        ops.adjoint(qml.TemporaryAND([wires[0], wires[1], work_wires[0]]))

    if not _skip_toggle_detection:
        # Perform toggle-detection unless skipped explicitly. By default, toggle detection
        # is skipped for `work_wire_type="zeroed"` but not for `work_wire_type="borrowed"`.
        _build_linear_depth_ladder(wires[:-1])
        ops.Toffoli([work_wires[0], wires[final_ctrl_index], wires[-1]])
        ops.adjoint(_build_linear_depth_ladder, lazy=False)(wires[:-1])


decompose_mcx_one_worker_explicit = flip_zero_control(_mcx_one_worker)


@register_condition(lambda num_work_wires, **_: not num_work_wires)
@register_condition(lambda num_control_wires, **_: num_control_wires > 2)
@register_resources(
    lambda num_control_wires, **_: _mcx_one_worker_resource(num_control_wires, "zeroed"),
    work_wires={"zeroed": 1},
)
def _mcx_one_zeroed_worker(wires, **kwargs):
    with allocation.allocate(1, state="zero", restored=True) as work_wires:
        kwargs.update({"work_wires": work_wires, "work_wire_type": "zeroed"})
        _mcx_one_worker(wires, **kwargs)


decompose_mcx_one_zeroed_worker = flip_zero_control(_mcx_one_zeroed_worker)


@register_condition(lambda num_work_wires, **_: not num_work_wires)
@register_condition(lambda num_control_wires, **_: num_control_wires > 2)
@register_resources(
    lambda num_control_wires, **_: _mcx_one_worker_resource(num_control_wires, "borrowed"),
    work_wires={"borrowed": 1},
)
def _mcx_one_borrowed_worker(wires, **kwargs):
    with allocation.allocate(1, state="any", restored=True) as work_wires:
        kwargs.update({"work_wires": work_wires, "work_wire_type": "borrowed"})
        _mcx_one_worker(wires, **kwargs)


decompose_mcx_one_borrowed_worker = flip_zero_control(_mcx_one_borrowed_worker)


def _decompose_mcx_no_worker_resource(num_control_wires, **__):
    len_k1 = (num_control_wires + 1) // 2
    len_k2 = num_control_wires - len_k1
    if len_k1 == len_k2:
        return {
            ops.Hadamard: 2,
            resource_rep(ops.QubitUnitary, num_wires=1): 2,
            controlled_resource_rep(
                ops.X,
                {},
                num_control_wires=len_k2,
                num_work_wires=len_k1,
                work_wire_type="borrowed",
            ): 4,
            adjoint_resource_rep(ops.QubitUnitary, {"num_wires": 1}): 2,
            controlled_resource_rep(ops.GlobalPhase, {}, num_control_wires=num_control_wires): 1,
        }
    return {
        ops.Hadamard: 2,
        resource_rep(ops.QubitUnitary, num_wires=1): 2,
        controlled_resource_rep(
            ops.X,
            {},
            num_control_wires=len_k2,
            num_work_wires=len_k1,
            work_wire_type="borrowed",
        ): 2,
        controlled_resource_rep(
            ops.X,
            {},
            num_control_wires=len_k1,
            num_work_wires=len_k2,
            work_wire_type="borrowed",
        ): 2,
        adjoint_resource_rep(ops.QubitUnitary, {"num_wires": 1}): 2,
        controlled_resource_rep(ops.GlobalPhase, {}, num_control_wires=num_control_wires): 1,
    }


@register_condition(lambda num_control_wires, **_: num_control_wires > 2)
@register_resources(_decompose_mcx_no_worker_resource)
def _decompose_mcx_with_no_worker(wires, **_):
    """Use ctrl_decomp_bisect_md to decompose a multi-controlled X gate with no work wires."""
    U = ops.RX.compute_matrix(np.pi)
    _ctrl_decomp_bisect_md(U, wires)
    ops.ctrl(ops.GlobalPhase(-np.pi / 2), control=wires[:-1])


decompose_mcx_with_no_worker = flip_zero_control(_decompose_mcx_with_no_worker)

####################
# Helper Functions #
####################


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
    _controlled_x(wires[-1], control=ctrl_k2, work_wires=ctrl_k1, work_wire_type="borrowed")
    ops.adjoint(ops.QubitUnitary(c1, wires[-1]))

    # Cancel the two identity controlled X gates
    _ctrl_decomp_bisect_od(d, wires, skip_initial_cx=True)

    # Adjoint of the component
    _controlled_x(wires[-1], control=ctrl_k1, work_wires=ctrl_k2, work_wire_type="borrowed")
    ops.QubitUnitary(c1, wires[-1])
    _controlled_x(wires[-1], control=ctrl_k2, work_wires=ctrl_k1, work_wire_type="borrowed")
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

    if not skip_initial_cx:
        _controlled_x(wires[-1], control=ctrl_k1, work_wires=ctrl_k2, work_wire_type="borrowed")

    ops.QubitUnitary(a, wires[-1])
    _controlled_x(wires[-1], control=ctrl_k2, work_wires=ctrl_k1, work_wire_type="borrowed")
    ops.adjoint(ops.QubitUnitary(a, wires[-1]))
    _controlled_x(wires[-1], control=ctrl_k1, work_wires=ctrl_k2, work_wire_type="borrowed")
    ops.QubitUnitary(a, wires[-1])
    _controlled_x(wires[-1], control=ctrl_k2, work_wires=ctrl_k1, work_wire_type="borrowed")
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


def _multi_control_zyz(
    phi, theta, omega, wires, work_wires, work_wire_type
):  # pylint: disable=too-many-arguments
    """Implements Lemma 7.9 from https://arxiv.org/pdf/quant-ph/9503016"""

    # Operator A
    ops.cond(_not_zero(phi), ops.CRZ)(phi, wires=wires[-2:])
    ops.cond(_not_zero(theta), ops.CRY)(theta / 2, wires=wires[-2:])

    ops.ctrl(
        ops.X(wires[-1]), control=wires[:-2], work_wires=work_wires, work_wire_type=work_wire_type
    )

    # Operator B
    ops.cond(_not_zero(theta), ops.CRY)(-theta / 2, wires=wires[-2:])
    ops.cond(_not_zero(phi + omega), ops.CRZ)(-(phi + omega) / 2, wires=wires[-2:])

    ops.ctrl(
        ops.X(wires[-1]), control=wires[:-2], work_wires=work_wires, work_wire_type=work_wire_type
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


def _controlled_x(target_wire, control, work_wires, work_wire_type):
    if len(control) == 1:
        ops.CNOT([control[0], target_wire])
    elif len(control) == 2:
        ops.Toffoli([control[0], control[1], target_wire])
    else:
        ops.MultiControlledX(
            control + [target_wire], work_wires=work_wires, work_wire_type=work_wire_type
        )


# pylint: disable=no-value-for-parameter
def _n_parallel_ccx_x(control_wires_x, control_wires_y, target_wires):
    r"""
    Construct a quantum circuit for creating n-condionally zeroed auxiliary qubits using 3n qubits. This
    implements Fig. 4a of [1]. Each wire is of the same size :math:`n`.

    Args:
        control_wires_x: The control wires for register 1.
        control_wires_y: The control wires for register 2.
        target_wires: The wires for target register.

    References:
        1. Khattar and Gidney, Rise of conditionally clean ancillae for optimizing quantum circuits
        `arXiv:2407.17966 <https://arxiv.org/abs/2407.17966>`__
    """

    @control_flow.for_loop(0, len(control_wires_x), 1)
    def loop(i):
        ops.X(target_wires[i])
        ops.Toffoli([control_wires_x[i], control_wires_y[i], target_wires[i]])

    loop()


def _build_linear_depth_ladder(wires) -> int:
    r"""
    Helper function to create linear-depth ladder operations used in Khattar and Gidney's MCX synthesis.
    In particular, this implements Step-1 and Step-2 on Fig. 3 of [1] except for the first and last
    CCX gates.

    Preconditions:
        - The number of wires must be greater than 2.

    Args:
        wires: the list of wires.

    Returns:
        int: the index of the last unmarked wire.

    References:
        1. Khattar and Gidney, Rise of conditionally clean ancillae for optimizing quantum circuits
        `arXiv:2407.17966 <https://arxiv.org/abs/2407.17966>`__

    """

    if len(wires) == 3:
        return 2

    if len(wires) == 4:
        ops.Toffoli([wires[2], wires[3], wires[1]])
        ops.X(wires[1])
        return 1

    i = -1
    while i + 2 < len(wires) - 2:
        i += 2
        ops.Toffoli([wires[i + 1], wires[i + 2], wires[i]])
        ops.X(wires[i])

    x, y = (i - 2, i) if i + 2 == len(wires) - 1 else (i, i + 3)
    k = x - 1

    ops.Toffoli([wires[x], wires[y], wires[k]])
    ops.X(wires[k])

    for i in range(k, 1, -2):
        ops.Toffoli([wires[i - 1], wires[i], wires[i - 2]])
        ops.X(wires[i - 2])

    return 0


def _build_log_n_depth_ccx_ladder(control_wires) -> list:
    r"""
    Helper function to build a log-depth ladder compose of CCX and X gates as shown in Fig. 4b of [1].

    Args:
        control_wires: The control wires.

    Returns:
        list: The list of unmarked wires to use as control wires.

    References:
        1. Khattar and Gidney, Rise of conditionally clean ancillae for optimizing quantum circuits
        `arXiv:2407.17966 <https://arxiv.org/abs/2407.17966>`__
    """

    final_control_wires = []

    # See Section 5.2 of [1] for what the following variables mean
    rightmost_marked = 1
    timestep = 1

    while rightmost_marked < len(control_wires) - 1:

        # At every time step, we aim to flip the next 2^i + 1 unmarked wires, but if
        # there are not enough wires available, we just flip all the remaining wires.
        n_to_flip = min(2**timestep + 1, len(control_wires) - rightmost_marked - 1)
        rightmost_ctrl = rightmost_marked + n_to_flip
        new_rightmost_marked = rightmost_ctrl
        leftmost_unmarked = rightmost_marked + 1

        while n_to_flip > 1:

            ccx_n = n_to_flip // 2
            ccx_t = control_wires[rightmost_marked + 1 - ccx_n : rightmost_marked + 1]
            ccx_y = control_wires[rightmost_ctrl + 1 - ccx_n : rightmost_ctrl + 1]
            ccx_x = control_wires[rightmost_ctrl + 1 - ccx_n * 2 : rightmost_ctrl + 1 - ccx_n]

            leftmost_unmarked = rightmost_marked + 1 - ccx_n
            _n_parallel_ccx_x(ccx_x, ccx_y, ccx_t)

            # The primitive used ccx_n target wires to flip the 2 * ccx_n control wires. The
            # total number of remaining wires to flip in this timestep is given by the original
            # number of wires minus the 2 * ccx_n wires that were flipped, plus the ccx_n
            # target wires that were unmarked as a result of this primitive.
            n_to_flip = n_to_flip - ccx_n
            rightmost_marked -= ccx_n
            rightmost_ctrl -= ccx_n * 2

        final_control_wires.append(leftmost_unmarked)
        rightmost_marked = new_rightmost_marked
        timestep += 1

    return final_control_wires
