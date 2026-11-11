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

"""Built-in decomposition rules for op_math.decompositions.controlled_decompositions."""

from functools import partial

import pennylane as qp
from pennylane import capture, compiler, math, ops
from pennylane.decomposition import register_condition, register_resources
from pennylane.decomposition_rules.op_math.decompositions.unitary_decompositions import (
    two_qubit_decomp_rule,
)
from pennylane.ops.op_math.decompositions.controlled_decompositions import (
    _ctrl_decomp_bisect,
    _ctrl_global_phase,
    _decompose_mcx_many_workers,
    _decompose_mcx_one_worker,
    _decompose_mcx_two_workers,
    _decompose_mcx_with_no_worker,
    _mcx_to_cnot_or_toffoli,
    _multi_control_zyz,
    _not_zero,
    _single_control_zyz,
)
from pennylane.typing import Complex, Float, Wire


def _ctrl_decomp_bisect_condition(U, wires, *_, **__):  # pylint: disable=unused-argument
    num_target_wires = int(qp.math.log2(qp.math.shape(U)[-1]))
    num_control_wires = len(wires) - num_target_wires
    # This decomposition rule is only applicable when the target is a single-qubit unitary.
    # Also, it is not helpful when there's only a single control wire.
    return num_target_wires == 1 and num_control_wires > 1


# pylint: disable-next=unused-argument
def _ctrl_decomp_bisect_resources(U, wires, control_values, *_, **__):

    num_ctrl_wires = len(wires) - 1
    len_k1 = (num_ctrl_wires + 1) // 2
    len_k2 = num_ctrl_wires - len_k1

    # this is a heuristic overestimate based on the resource requirement of the general case.
    mcx_k1 = ops.MultiControlledX(Wire[len_k1 + 1], work_wires=Wire[len_k2])
    mcx_k2 = ops.MultiControlledX(Wire[len_k2 + 1], work_wires=Wire[len_k1])
    mcx_reps = {mcx_k1: 6} if len_k1 == len_k2 else {mcx_k2: 4, mcx_k1: 2}
    return {
        ops.QubitUnitary(Complex[2, 2], wires=Wire[1]): 4,
        qp.adjoint(ops.QubitUnitary(Complex[2, 2], wires=Wire[1])): 4,
        # we only need Hadamard for the main diagonal case (see _ctrl_decomp_bisect_md),
        # but it still needs to be accounted for.
        ops.Hadamard: 2,
        ops.ctrl(ops.GlobalPhase(Float), Wire[num_ctrl_wires], work_wires=Wire[1]): 1,
        qp.X: num_ctrl_wires,  # assume half the control values are 0s
    } | mcx_reps


# Resources are not exact because rotations might be skipped for zero angles
@register_condition(_ctrl_decomp_bisect_condition)
@register_resources(_ctrl_decomp_bisect_resources, exact=False)
def ctrl_decomp_bisect_rule(U, wires, control_values, *_, **__):
    """The decomposition rule for ControlledQubitUnitary from
    `Vale et al. (2023) <https://arxiv.org/abs/2302.06377>`_."""

    return _ctrl_decomp_bisect(U, wires, control_values, *_, **__)


def _single_ctrl_decomp_zyz_condition(U, wires, *_, **__):
    num_target_wires = int(qp.math.log2(qp.math.shape(U)[-1]))
    num_control_wires = len(wires) - num_target_wires
    return num_target_wires == 1 and num_control_wires == 1


# pylint:disable-next=unused-argument
def _single_ctrl_decomp_zyz_resources(U, wires, **_):
    return {
        ops.RZ: 3,
        ops.RY: 2,
        ops.CNOT: 2,
        ops.ctrl(ops.GlobalPhase(Float), Wire[1]): 1,
        ops.X: len(wires) - 1,
    }


# Resources are not exact because rotations might be skipped for zero angles
@register_condition(_single_ctrl_decomp_zyz_condition)
@register_resources(_single_ctrl_decomp_zyz_resources, exact=False)
# pylint: disable-next=unused-argument
def single_ctrl_decomp_zyz_rule(U, wires, control_values, work_wires, **_):
    """The decomposition rule for ControlledQubitUnitary from Lemma 5.1 of
    https://arxiv.org/pdf/quant-ph/9503016"""

    if capture.enabled() or compiler.active():
        wires = qp.math.array(wires, like="jax")
        control_values = math.array(control_values, like="jax")

    @qp.for_loop(len(control_values))
    def _x_flips(i):
        qp.cond(qp.math.logical_not(control_values[i]), qp.X)(wires[i])

    phi, theta, omega, phase = math.decomposition.zyz_rotation_angles(U)

    _x_flips()  # pylint: disable=no-value-for-parameter
    _single_control_zyz(phi, theta, omega, wires=wires)
    ops.cond(_not_zero(phase), _ctrl_global_phase)(phase, wires[:-1])
    _x_flips()  # pylint: disable=no-value-for-parameter


def _multi_ctrl_decomp_zyz_condition(U, wires, *_, **__):
    num_target_wires = int(qp.math.log2(qp.math.shape(U)[-1]))
    num_control_wires = len(wires) - num_target_wires
    return num_target_wires == 1 and num_control_wires > 1


# pylint: disable-next=unused-argument
def _multi_ctrl_decomp_zyz_resources(U, wires, control_values, work_wires, work_wire_type, **_):
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
@register_resources(_multi_ctrl_decomp_zyz_resources, exact=False)
# pylint: disable-next=unused-argument
def multi_control_decomp_zyz_rule(U, wires, control_values, work_wires, work_wire_type, **_):
    """The decomposition rule for ControlledQubitUnitary from Lemma 7.9 of
    https://arxiv.org/pdf/quant-ph/9503016"""

    if compiler.active() or capture.enabled():
        wires = math.array(wires, like="jax")
        control_values = math.array(control_values, like="jax")

    @qp.for_loop(len(control_values))
    def _x_flips(i):
        qp.cond(qp.math.logical_not(control_values[i]), qp.X)(wires[i])

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


# pylint: disable-next=unused-argument
def _ctrl_two_qubit_unitary_resource(U, wires, control_values, work_wires, work_wire_type, **_):
    num_target_wires = int(qp.math.log2(qp.math.shape(U)[-1]))
    num_control_wires = len(wires) - num_target_wires
    num_work_wires = len(work_wires)
    base_resources = two_qubit_decomp_rule.compute_resources(num_wires=num_target_wires)
    gate_counts = {
        qp.ctrl(
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
@register_condition(lambda U, *_, **__: qp.math.ceil_log2(qp.math.shape(U)[0]) == 2)
@register_resources(_ctrl_two_qubit_unitary_resource, exact=False)
def controlled_two_qubit_unitary_rule(U, wires, control_values, work_wires, work_wire_type, **_):
    """A controlled two-qubit unitary is decomposed by applying ctrl to the base decomposition."""

    if compiler.active() or capture.enabled():
        wires = math.array(wires, like="jax")
        control_values = math.array(control_values, like="jax")

    @qp.for_loop(len(control_values))
    def _x_flips(i):
        qp.cond(qp.math.logical_not(control_values[i]), qp.X)(wires[i])

    _x_flips()  # pylint: disable=no-value-for-parameter
    ops.ctrl(
        two_qubit_decomp_rule._impl,  # pylint: disable=protected-access
        control=wires[:-2],
        work_wires=work_wires,
        work_wire_type=work_wire_type,
    )(U, wires=wires[-2:])
    _x_flips()  # pylint: disable=no-value-for-parameter


def _wrap_mcx_rule_w_alloc(base_rule, num_work_wires, work_wire_type, name=""):
    """Given a MCX decomposition rule that takes explicit work wires, populate the same
    decomposition rule that uses dynamic work wire allocation instead."""

    def _resource_fn(wires, control_values, *_, **__):
        # pylint: disable-next=protected-access
        return base_rule._compute_resources(
            wires,
            control_values,
            Wire[num_work_wires],
            work_wire_type=work_wire_type,
        )

    # pylint: disable-next=unused-argument
    def _condition_fn(wires, control_values, work_wires, work_wire_type):
        # The allocation-based rules are only considered if the operator does not
        # come with explicitly specified work wires. We've made the decision last
        # year that passing work wires to an operator is like explicitly saying
        # "use these work wires in the operator's decomposition"
        return len(work_wires) == 0 and base_rule.is_applicable(
            wires,
            control_values,
            Wire[num_work_wires],
            work_wire_type=work_wire_type,
        )

    @register_condition(_condition_fn)
    @register_resources(
        _resource_fn,
        work_wires={work_wire_type: num_work_wires},
        exact=base_rule.exact_resources,
        name=name or f"use_allocation({base_rule.name})",
    )
    def _impl(wires, control_values, *_, **__):
        state = "zero" if work_wire_type == "zeroed" else "any"
        with qp.allocation.allocate(num_work_wires, state, restored=True) as work_wires:
            # pylint: disable-next=protected-access
            base_rule._impl(wires, control_values, work_wires, work_wire_type=work_wire_type)

    return _impl


def _mcx_to_cnot_or_toffoli_resource(wires, *_, **__):
    # The PauliX may be required to flip any non-zero controlled bits
    # Each non-zero control value requires 2 PauliX gates to flip, but we count
    # only one for each control wire as an average case estimate.
    if len(wires) - 1 == 1:
        return {qp.CNOT: 1, qp.X: 1}
    return {qp.Toffoli: 1, qp.X: 2}


@register_condition(lambda wires, *_, **__: len(wires) <= 3)
@register_resources(_mcx_to_cnot_or_toffoli_resource, exact=False)
def mcx_to_cnot_or_toffoli(wires, control_values, *_, **__):
    """A decomposition rule that decomposes an MCX to a CNOT or Toffoli"""

    return _mcx_to_cnot_or_toffoli(wires, control_values, *_, **__)


def _2cx_elbow_explicit_resources(wires, *_, **__):
    num_ctrl_wires = len(wires) - 1
    return {qp.Elbow: 1, qp.CNOT: 1, qp.adjoint(qp.Elbow(Wire[3])): 1, qp.X: num_ctrl_wires}


# pylint: disable-next=unused-argument
def _2cx_elbow_explicit_condition(wires, control_values, work_wires, work_wire_type="borrowed"):
    num_control_wires = len(wires) - 1
    return len(work_wires) >= 1 and num_control_wires == 2 and work_wire_type == "zeroed"


@register_condition(_2cx_elbow_explicit_condition)
@register_resources(_2cx_elbow_explicit_resources, exact=False)
def decompose_mcx_two_controls_elbows(wires, control_values, work_wires, **_):
    """A decomposition rule for MCX that uses controlled Elbow."""

    if compiler.active() or capture.enabled():
        wires = math.array(wires, like="jax")
        control_values = math.array(control_values, like="jax")

    @qp.for_loop(len(control_values))
    def _x_flips(i):
        qp.cond(qp.math.logical_not(control_values[i]), qp.X)(wires[i])

    elbow_wires = [wires[0], wires[1], work_wires[0]]

    _x_flips()  # pylint: disable=no-value-for-parameter
    qp.Elbow(elbow_wires)
    qp.CNOT([work_wires[0], wires[2]])
    qp.adjoint(qp.Elbow)(elbow_wires)
    _x_flips()  # pylint: disable=no-value-for-parameter


# pylint: disable-next=unused-argument
def _mcx_many_workers_condition(wires, control_values, work_wires, **_):
    num_control_wires = len(wires) - 1
    num_work_wires = len(work_wires)
    return num_control_wires > 2 and num_work_wires >= num_control_wires - 2


# pylint: disable-next=unused-argument
def _mcx_many_workers_resource(wires, control_values, work_wires, work_wire_type="borrowed"):

    num_control_wires = len(wires) - 1
    num_used_work_wires = num_control_wires - 2
    num_extra_work_wires = len(work_wires) - num_used_work_wires

    # The middle MCX retains the variable number of extra work wires.
    middle_toffoli = ops.MultiControlledX(
        Wire[3],
        work_wires=Wire[num_extra_work_wires],
        work_wire_type=work_wire_type,
    )

    if work_wire_type == "borrowed":
        return {
            ops.Toffoli: 4 * num_used_work_wires - 1,
            middle_toffoli: 1,
            qp.X: num_control_wires,
        }

    return {
        qp.TemporaryAND: num_used_work_wires,
        qp.adjoint(qp.TemporaryAND(Wire[3])): num_used_work_wires,
        middle_toffoli: 1,
        qp.X: num_control_wires,
    }


@register_condition(_mcx_many_workers_condition)
@register_resources(_mcx_many_workers_resource, exact=False)
# pylint: disable-next=unused-argument
def decompose_mcx_many_workers(wires, control_values, work_wires, work_wire_type="borrowed"):
    """Decomposes the multi-controlled PauliX gate using the approach in Lemma 7.2 of
    https://arxiv.org/abs/quant-ph/9503016, which requires a suitably large register of
    work wires"""

    return _decompose_mcx_many_workers(wires, control_values, work_wires, work_wire_type)


# pylint: disable-next=unused-argument
def _mcx_two_workers_condition(wires, control_values, work_wires, **_):
    num_ctrls = len(wires) - 1
    num_work_wires = len(work_wires)
    return num_ctrls > 2 and (num_work_wires >= 2 or (num_work_wires == 1 and num_ctrls < 6))


# pylint: disable-next=unused-argument
def _mcx_two_workers_resource(wires, control_values, work_wires, work_wire_type="borrowed"):

    num_control_wires = len(wires) - 1
    is_small_mcx = num_control_wires < 6
    n_adjoint_ladder = num_control_wires - 4 + is_small_mcx
    adjoint_toffoli = qp.adjoint(ops.Toffoli(Wire[3]))
    adjoint_x = qp.adjoint(ops.X(Wire[1]))

    if work_wire_type == "zeroed":
        n_ccx = 2 * num_control_wires - 3
        n_temporary_ccx_pairs = 2 - is_small_mcx
        n_x = (n_ccx - 3 if is_small_mcx else n_ccx - 5) + num_control_wires
        return {
            ops.Toffoli: n_ccx - 2 * n_temporary_ccx_pairs - n_adjoint_ladder,
            adjoint_toffoli: n_adjoint_ladder,
            ops.X: n_x - n_adjoint_ladder,
            adjoint_x: n_adjoint_ladder,
            qp.TemporaryAND: n_temporary_ccx_pairs,
            qp.adjoint(qp.TemporaryAND(Wire[3])): n_temporary_ccx_pairs,
        }

    # Otherwise, we assume the work wires are borrowed
    n_ccx = 4 * num_control_wires - 8
    n_x = (n_ccx - 4 if is_small_mcx else n_ccx - 8) + num_control_wires
    return {
        ops.Toffoli: n_ccx - 2 * n_adjoint_ladder,
        adjoint_toffoli: 2 * n_adjoint_ladder,
        ops.X: n_x - 2 * n_adjoint_ladder,
        adjoint_x: 2 * n_adjoint_ladder,
    }


@register_condition(_mcx_two_workers_condition)
@register_resources(_mcx_two_workers_resource, exact=False)
# pylint: disable-next=unused-argument
def decompose_mcx_two_workers(wires, control_values, work_wires, work_wire_type="borrowed"):
    r"""
    Synthesise a multi-controlled X gate with :math:`k` controls using :math:`2` auxiliary qubits.
    It produces a circuit with :math:`2k-3` Toffoli gates and depth :math:`O(\log(k))` if using
    zeroed auxiliary qubits, and :math:`4k-8` Toffoli gates and depth :math:`O(\log(k))` if using borrowed
    auxiliary qubits as described in Sec. 5 of [1].

    References:
        1. Khattar and Gidney, Rise of conditionally clean ancillae for optimizing quantum circuits
        `arXiv:2407.17966 <https://arxiv.org/abs/2407.17966>`__

    """

    return _decompose_mcx_two_workers(wires, control_values, work_wires, work_wire_type)


# pylint: disable-next=unused-argument
def _mcx_one_worker_condition(wires, control_values, work_wires, **_):
    num_control_wires = len(wires) - 1
    num_work_wires = len(work_wires)
    return num_control_wires > 2 and num_work_wires == 1


# pylint: disable-next=unused-argument
def _mcx_one_worker_resource(wires, control_values, work_wires, work_wire_type="borrowed"):
    num_control_wires = len(wires) - 1
    n_adjoint_ladder = num_control_wires - 3
    adjoint_toffoli = qp.adjoint(ops.Toffoli(Wire[3]))
    adjoint_x = qp.adjoint(ops.X(Wire[1]))
    if work_wire_type == "zeroed":
        n_ccx = 2 * num_control_wires - 5
        return {
            ops.Toffoli: n_ccx - n_adjoint_ladder,
            adjoint_toffoli: n_adjoint_ladder,
            qp.TemporaryAND: 1,
            qp.adjoint(qp.TemporaryAND(Wire[3])): 1,
            ops.X: n_ccx - 1 + num_control_wires - n_adjoint_ladder,
            adjoint_x: n_adjoint_ladder,
        }
    # Otherwise, we assume the work wire is borrowed
    n_ccx = 4 * num_control_wires - 8
    return {
        ops.Toffoli: n_ccx - 2 * n_adjoint_ladder,
        adjoint_toffoli: 2 * n_adjoint_ladder,
        ops.X: n_ccx - 4 + num_control_wires - 2 * n_adjoint_ladder,
        adjoint_x: 2 * n_adjoint_ladder,
    }


@register_condition(_mcx_one_worker_condition)
@register_resources(_mcx_one_worker_resource, exact=False)
# pylint: disable-next=unused-argument
def decompose_mcx_one_worker(wires, control_values, work_wires, work_wire_type="zeroed"):
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

    return _decompose_mcx_one_worker(wires, control_values, work_wires, work_wire_type)


def _decompose_mcx_no_worker_resource(wires, *_, **__):
    num_control_wires = len(wires) - 1
    len_k1 = (num_control_wires + 1) // 2
    len_k2 = num_control_wires - len_k1
    mcx_k1 = ops.MultiControlledX(Wire[len_k1 + 1], work_wires=Wire[len_k2])
    mcx_k2 = ops.MultiControlledX(Wire[len_k2 + 1], work_wires=Wire[len_k1])
    mcx_reps = {mcx_k1: 4} if len_k1 == len_k2 else {mcx_k1: 2, mcx_k2: 2}
    return {
        ops.Hadamard: 2,
        ops.QubitUnitary(Complex[2, 2], wires=Wire[1]): 2,
        qp.adjoint(ops.QubitUnitary(Complex[2, 2], wires=Wire[1])): 2,
        ops.ctrl(ops.GlobalPhase(Float), Wire[num_control_wires]): 1,
        ops.X: num_control_wires,
    } | mcx_reps


# pylint: disable-next=unused-argument
def _mcx_no_worker_condition(wires, control_values, work_wires, **_):
    # With work wires present, one of the one-, two- or many-worker rules always applies, which
    # matches ``MultiControlledX.compute_decomposition``. Offering this rule as well pulls
    # ``C(GlobalPhase)`` and ``C(PhaseShift)`` with every number of controls into the
    # decomposition graph, and ``qjit(capture=True)`` traces and compiles all of them.
    return len(wires) > 3 and len(work_wires) == 0


@register_condition(_mcx_no_worker_condition)
@register_resources(_decompose_mcx_no_worker_resource, exact=False)
def decompose_mcx_with_no_worker(wires, control_values, *_, **__):
    """Use ctrl_decomp_bisect_md to decompose a multi-controlled X gate with no work wires."""

    return _decompose_mcx_with_no_worker(wires, control_values, *_, **__)
