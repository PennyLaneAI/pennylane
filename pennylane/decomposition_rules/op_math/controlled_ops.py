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


"""Built-in decomposition rules for op_math.controlled_ops."""

from collections import defaultdict
from functools import partial

import numpy as np
from scipy import sparse

from pennylane import allocation, capture, compiler, control_flow, math, ops, templates
from pennylane.allocation import allocate
from pennylane.core.operator import abstractify
from pennylane.decomposition import add_decomps, register_condition, register_resources
from pennylane.decomposition.decomposition_rule import (
    DecompCollection,
    _list_decomps,
    get_fixed_decomp,
)
from pennylane.decomposition.resources import resolve_work_wire_type
from pennylane.decomposition_rules.op_math import adjoint_negate_angle
from pennylane.decomposition_rules.op_math import pow_involutory, pow_scale_angle
from pennylane.decomposition_rules.op_math import self_adjoint
from pennylane.decomposition_rules.qubit.matrix_ops import qubit_unitary_two_qubit
from pennylane.ops.mid_measure.pauli_measure import pauli_measure
from pennylane.ops.op_math.change_op_basis2 import _change_op_basis_abstract
from pennylane.ops.op_math.controlled import _resolve_ctrl_values
from pennylane.ops.op_math.decompositions.controlled_decompositions import (
    _controlled_unitary_bisect_impl,
    _ctrl_global_phase,
    _mcx_many_work_wires_impl,
    _mcx_no_work_wires_impl,
    _mcx_one_work_wire_impl,
    _mcx_to_cnot_or_toffoli_impl,
    _mcx_two_work_wires_impl,
    _multi_control_zyz,
    _not_zero,
    _single_control_zyz,
)
from pennylane.typing import AbstractWires, Bool, Complex, Float, TensorLike, Wire
from pennylane.wires import WiresLike, concatenate_wires


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
    return _controlled_unitary_bisect_impl(U, wires, control_values, *_, **__)


def _single_ctrl_decomp_zyz_condition(U, wires, *_, **__):
    num_target_wires = int(math.log2(math.shape(U)[-1]))
    num_control_wires = len(wires) - num_target_wires
    return num_target_wires == 1 and num_control_wires == 1


# pylint: disable-next=unused-argument
def _cu_single_control_zyz_resources(U, wires, **_):
    return {
        ops.RZ: 3,
        ops.RY: 2,
        ops.CNOT: 2,
        ops.ctrl(ops.GlobalPhase(Float), Wire[1]): 1,
        ops.X: len(wires) - 1,
    }


# Resources are not exact because rotations might be skipped for zero angles
@register_condition(_single_ctrl_decomp_zyz_condition)
@register_resources(_cu_single_control_zyz_resources, exact=False, name="single_control_zyz")
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


# pylint: disable-next=unused-argument
def _cu_multi_control_zyz_resources(U, wires, control_values, work_wires, work_wire_type, **_):
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
@register_resources(_cu_multi_control_zyz_resources, exact=False, name="multi_control_zyz")
def controlled_unitary_multi_control_zyz(U, wires, control_values, work_wires, work_wire_type, **_):
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


# pylint: disable-next=unused-argument
def _cu_two_qubit_resources(U, wires, control_values, work_wires, work_wire_type, **_):
    num_target_wires = int(math.log2(math.shape(U)[-1]))
    num_control_wires = len(wires) - num_target_wires
    num_work_wires = len(work_wires)
    base_resources = qubit_unitary_two_qubit.compute_resources(num_wires=num_target_wires)
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
@register_resources(_cu_two_qubit_resources, exact=False, name="two_qubit")
def controlled_unitary_two_qubit(U, wires, control_values, work_wires, work_wire_type, **_):
    """A controlled two-qubit unitary is decomposed by applying ctrl to the base decomposition."""

    if compiler.active() or capture.enabled():
        wires = math.array(wires, like="jax")
        control_values = math.array(control_values, like="jax")

    @control_flow.for_loop(len(control_values))
    def _x_flips(i):
        ops.cond(math.logical_not(control_values[i]), ops.X)(wires[i])

    _x_flips()  # pylint: disable=no-value-for-parameter
    ops.ctrl(
        qubit_unitary_two_qubit._impl,  # pylint: disable=protected-access
        control=wires[:-2],
        work_wires=work_wires,
        work_wire_type=work_wire_type,
    )(U, wires=wires[-2:])
    _x_flips()  # pylint: disable=no-value-for-parameter


########################################
# MultiControlledX Decomposition Rules #
########################################


def _wrap_mcx_rule_w_alloc(base_rule, num_work_wires, work_wire_type, name=""):
    """Given a MCX decomposition rule that takes explicit work wires, populate the same
    decomposition rule that uses dynamic work wire allocation instead."""

    def _resource_fn(wires, control_values, *_, **__):
        return base_rule._compute_resources(  # pylint: disable=protected-access
            wires,
            control_values,
            Wire[num_work_wires],
            work_wire_type=work_wire_type,
        )

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
        with allocation.allocate(num_work_wires, state, restored=True) as work_wires:
            # pylint: disable-next=protected-access
            base_rule._impl(wires, control_values, work_wires, work_wire_type=work_wire_type)

    return _impl


def _mcx_to_cnot_or_toffoli_resources(wires, *_, **__):
    # The PauliX may be required to flip any non-zero controlled bits
    # Each non-zero control value requires 2 PauliX gates to flip, but we count
    # only one for each control wire as an average case estimate.
    if len(wires) - 1 == 1:
        return {ops.CNOT: 1, ops.X: 1}
    return {ops.Toffoli: 1, ops.X: 2}


@register_condition(lambda wires, *_, **__: len(wires) <= 3)
@register_resources(_mcx_to_cnot_or_toffoli_resources, exact=False, name="to_cnot_or_toffoli")
def mcx_to_cnot_or_toffoli(wires, control_values, *_, **__):
    """A decomposition rule that decomposes an MCX to a CNOT or Toffoli"""
    return _mcx_to_cnot_or_toffoli_impl(wires, control_values, *_, **__)


def _mcx_two_controls_elbows_resources(wires, *_, **__):
    num_ctrl_wires = len(wires) - 1
    return {
        templates.Elbow: 1,
        ops.CNOT: 1,
        ops.adjoint(templates.Elbow(Wire[3])): 1,
        ops.X: num_ctrl_wires,
    }


# pylint: disable-next=unused-argument
def _2cx_elbow_explicit_condition(wires, control_values, work_wires, work_wire_type="borrowed"):
    num_control_wires = len(wires) - 1
    return len(work_wires) >= 1 and num_control_wires == 2 and work_wire_type == "zeroed"


@register_condition(_2cx_elbow_explicit_condition)
@register_resources(_mcx_two_controls_elbows_resources, exact=False, name="two_controls_elbows")
def mcx_two_controls_elbows(wires, control_values, work_wires, **_):
    """A decomposition rule for MCX that uses controlled Elbow."""

    if compiler.active() or capture.enabled():
        wires = math.array(wires, like="jax")
        control_values = math.array(control_values, like="jax")

    @control_flow.for_loop(len(control_values))
    def _x_flips(i):
        ops.cond(math.logical_not(control_values[i]), ops.X)(wires[i])

    elbow_wires = [wires[0], wires[1], work_wires[0]]

    _x_flips()  # pylint: disable=no-value-for-parameter
    templates.Elbow(elbow_wires)
    ops.CNOT([work_wires[0], wires[2]])
    ops.adjoint(templates.Elbow)(elbow_wires)
    _x_flips()  # pylint: disable=no-value-for-parameter


# pylint: disable-next=unused-argument
def _mcx_many_workers_condition(wires, control_values, work_wires, **_):
    num_control_wires = len(wires) - 1
    num_work_wires = len(work_wires)
    return num_control_wires > 2 and num_work_wires >= num_control_wires - 2


# pylint: disable-next=unused-argument
def _mcx_many_work_wires_resources(wires, control_values, work_wires, work_wire_type="borrowed"):

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
            ops.X: num_control_wires,
        }

    return {
        templates.TemporaryAND: num_used_work_wires,
        ops.adjoint(templates.TemporaryAND(Wire[3])): num_used_work_wires,
        middle_toffoli: 1,
        ops.X: num_control_wires,
    }


@register_condition(_mcx_many_workers_condition)
@register_resources(_mcx_many_work_wires_resources, exact=False, name="many_work_wires")
def mcx_many_work_wires(wires, control_values, work_wires, work_wire_type="borrowed"):
    """Decomposes the multi-controlled PauliX gate using the approach in Lemma 7.2 of
    https://arxiv.org/abs/quant-ph/9503016, which requires a suitably large register of
    work wires"""
    return _mcx_many_work_wires_impl(wires, control_values, work_wires, work_wire_type)


# pylint: disable-next=unused-argument
def _mcx_two_workers_condition(wires, control_values, work_wires, **_):
    num_ctrls = len(wires) - 1
    num_work_wires = len(work_wires)
    return num_ctrls > 2 and (num_work_wires >= 2 or (num_work_wires == 1 and num_ctrls < 6))


# pylint: disable-next=unused-argument
def _mcx_two_work_wires_resources(wires, control_values, work_wires, work_wire_type="borrowed"):

    num_control_wires = len(wires) - 1
    is_small_mcx = num_control_wires < 6
    n_adjoint_ladder = num_control_wires - 4 + is_small_mcx
    adjoint_toffoli = ops.adjoint(ops.Toffoli(Wire[3]))
    adjoint_x = ops.adjoint(ops.X(Wire[1]))

    if work_wire_type == "zeroed":
        n_ccx = 2 * num_control_wires - 3
        n_temporary_ccx_pairs = 2 - is_small_mcx
        n_x = (n_ccx - 3 if is_small_mcx else n_ccx - 5) + num_control_wires
        return {
            ops.Toffoli: n_ccx - 2 * n_temporary_ccx_pairs - n_adjoint_ladder,
            adjoint_toffoli: n_adjoint_ladder,
            ops.X: n_x - n_adjoint_ladder,
            adjoint_x: n_adjoint_ladder,
            templates.TemporaryAND: n_temporary_ccx_pairs,
            ops.adjoint(templates.TemporaryAND(Wire[3])): n_temporary_ccx_pairs,
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
@register_resources(_mcx_two_work_wires_resources, exact=False, name="two_work_wires")
def mcx_two_work_wires(wires, control_values, work_wires, work_wire_type="borrowed"):
    r"""
    Synthesise a multi-controlled X gate with :math:`k` controls using :math:`2` auxiliary qubits.
    It produces a circuit with :math:`2k-3` Toffoli gates and depth :math:`O(\log(k))` if using
    zeroed auxiliary qubits, and :math:`4k-8` Toffoli gates and depth :math:`O(\log(k))` if using borrowed
    auxiliary qubits as described in Sec. 5 of [1].

    References:
        1. Khattar and Gidney, Rise of conditionally clean ancillae for optimizing quantum circuits
        `arXiv:2407.17966 <https://arxiv.org/abs/2407.17966>`__

    """
    return _mcx_two_work_wires_impl(wires, control_values, work_wires, work_wire_type)


# pylint: disable-next=unused-argument
def _mcx_one_worker_condition(wires, control_values, work_wires, **_):
    num_control_wires = len(wires) - 1
    num_work_wires = len(work_wires)
    return num_control_wires > 2 and num_work_wires == 1


# pylint: disable-next=unused-argument
def _mcx_one_work_wire_resources(wires, control_values, work_wires, work_wire_type="borrowed"):
    num_control_wires = len(wires) - 1
    n_adjoint_ladder = num_control_wires - 3
    adjoint_toffoli = ops.adjoint(ops.Toffoli(Wire[3]))
    adjoint_x = ops.adjoint(ops.X(Wire[1]))
    if work_wire_type == "zeroed":
        n_ccx = 2 * num_control_wires - 5
        return {
            ops.Toffoli: n_ccx - n_adjoint_ladder,
            adjoint_toffoli: n_adjoint_ladder,
            templates.TemporaryAND: 1,
            ops.adjoint(templates.TemporaryAND(Wire[3])): 1,
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
@register_resources(_mcx_one_work_wire_resources, exact=False, name="one_work_wire")
def mcx_one_work_wire(wires, control_values, work_wires, work_wire_type="zeroed"):
    r"""
    Synthesise a multi-controlled X gate with :math:`k` controls using :math:`1` auxiliary qubit. It
    produces a circuit with :math:`2k-3` Toffoli gates and depth :math:`O(k)` if the auxiliary is zeroed
    and :math:`4k-3` Toffoli gates and depth :math:`O(k)` if the auxiliary is borrowed as described in
    Sec. 5.1 of [1].

    .. note::

        The keyword argument ``_skip_toggle_detection`` is only supposed to be used when utilizing
        ``_mcx_one_worker`` as a subroutine within a decomposition rule, but not when using
        it as a decomposition rule itself. This is because ``_mcx_one_work_wire_resources`` does not
        support/take into account this keyword argument.

    References:
        1. Khattar and Gidney, Rise of conditionally clean ancillae for optimizing quantum circuits
        `arXiv:2407.17966 <https://arxiv.org/abs/2407.17966>`__

    """
    return _mcx_one_work_wire_impl(wires, control_values, work_wires, work_wire_type)


def _mcx_no_work_wires_resources(wires, *_, **__):
    num_control_wires = len(wires) - 1
    len_k1 = (num_control_wires + 1) // 2
    len_k2 = num_control_wires - len_k1
    mcx_k1 = ops.MultiControlledX(Wire[len_k1 + 1], work_wires=Wire[len_k2])
    mcx_k2 = ops.MultiControlledX(Wire[len_k2 + 1], work_wires=Wire[len_k1])
    mcx_reps = {mcx_k1: 4} if len_k1 == len_k2 else {mcx_k1: 2, mcx_k2: 2}
    return {
        ops.Hadamard: 2,
        ops.QubitUnitary(Complex[2, 2], wires=Wire[1]): 2,
        ops.adjoint(ops.QubitUnitary(Complex[2, 2], wires=Wire[1])): 2,
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
@register_resources(_mcx_no_work_wires_resources, exact=False, name="no_work_wires")
def mcx_no_work_wires(wires, control_values, *_, **__):
    """Use ctrl_decomp_bisect_md to decompose a multi-controlled X gate with no work wires."""
    return _mcx_no_work_wires_impl(wires, control_values, *_, **__)


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
    ops.ControlledQubitUnitary,
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
    ops.ControlledQubitUnitary(
        _conj_transpose(base.U),
        wires=base.wires,
        control_values=base.control_values,
        work_wires=base.work_wires,
        work_wire_type=base.work_wire_type,
    )


add_decomps("Adjoint(ControlledQubitUnitary)", adjoint_controlled_qubit_unitary_standard)


@register_resources({ops.RY: 2, ops.CZ: 1}, name="to_ry_cz_ry")
def ch_to_ry_cz_ry(wires: WiresLike, **__):
    """Decompose ``CH`` using the ``to_ry_cz_ry`` strategy."""
    ops.RY(-np.pi / 4, wires=wires[1])
    ops.CZ(wires=wires)
    ops.RY(+np.pi / 4, wires=wires[1])


add_decomps(ops.CH, ch_to_ry_cz_ry)
add_decomps("Adjoint(CH)", self_adjoint)
add_decomps("Pow(CH)", pow_involutory)


@register_resources({ops.CRY: 1, ops.S: 1}, name="to_ry_cnot")
def cy_to_ry_cnot(wires: WiresLike, **__):
    """Decompose ``CY`` using the ``to_ry_cnot`` strategy."""
    ops.CRY(np.pi, wires=wires)
    ops.S(wires=wires[0])


# pylint: disable-next=unused-argument
def _pauli_ctrl_pauli_ppr_resources(wires: AbstractWires, pauli0, pauli1):
    """Resources for _pauli_ctrl_pauli_ppr."""
    resources = defaultdict(int)
    p0, p1 = pauli0.__name__[-1], pauli1.__name__[-1]
    resources[ops.PPR_4(-1, pauli_word=p0, wires=Wire[len(p0)])] += 1
    resources[ops.PPR_4(-1, pauli_word=p1, wires=Wire[len(p1)])] += 1
    resources[ops.PPR_4(1, pauli_word=p0 + p1, wires=Wire[len(p0 + p1)])] += 1
    resources[ops.GlobalPhase] += 1
    return dict(resources)


def _pauli_ctrl_pauli_ppr(wires: AbstractWires, pauli0, pauli1):
    """Generalized two-qubit Pauli-controlled Pauli gate decomposition to PPRs."""
    p0, p1 = pauli0.__name__[-1], pauli1.__name__[-1]
    ops.PPR_4(-1, p0, wires=wires[0])
    ops.PPR_4(-1, p1, wires=wires[1])
    ops.PPR_4(1, p0 + p1, wires=wires)
    ops.GlobalPhase(np.pi / 4)


_cy_to_ppr_resources = partial(_pauli_ctrl_pauli_ppr_resources, pauli0=ops.Z, pauli1=ops.Y)


@register_resources(_cy_to_ppr_resources, name="to_ppr")
def cy_to_ppr(wires: AbstractWires):
    """Decompose ``CY`` using the ``to_ppr`` strategy."""
    _pauli_ctrl_pauli_ppr(wires, ops.Z, ops.Y)


def _pauli_ctrl_pauli_ppm(wires, pauli0, pauli1):
    """Generalized two-qubit Pauli-controlled Pauli gate decomposition to PPMs."""
    with allocate(1, state="zero", restored=False) as work_wires:
        m0 = pauli_measure(pauli0.__name__[-1] + "X", [wires[0], work_wires[0]])
        m1 = pauli_measure("Z" + pauli1.__name__[-1], [work_wires[0], wires[1]])
        m2 = pauli_measure("X", work_wires)
        ops.cond(m1, pauli0)(wires[0])
        ops.cond(m0 != m2, pauli1)(wires[1])
        ops.cond(m1 & (m0 != m2), ops.GlobalPhase)(np.pi)
        ops.cond(m2, ops.Z)(work_wires[0])  # Reset work wire (to |+>), achieving pure state


@register_resources(
    {
        ops.PauliMeasure("ZX", wires=Wire[2]): 1,
        ops.PauliMeasure("ZY", wires=Wire[2]): 1,
        ops.PauliMeasure("X", wires=Wire[1]): 1,
        ops.Z: 2,
        ops.Y: 1,
        ops.GlobalPhase: 1,
    },
    work_wires={"burnable": 1},
    name="lattice_surgery_ppm",
)
def cy_lattice_surgery_ppm(wires: AbstractWires):
    """Decompose ``CY`` using the ``lattice_surgery_ppm`` strategy."""
    _pauli_ctrl_pauli_ppm(wires, ops.Z, ops.Y)


add_decomps(ops.CY, cy_to_ry_cnot, cy_to_ppr, cy_lattice_surgery_ppm)
add_decomps("Adjoint(CY)", self_adjoint)
add_decomps("Pow(CY)", pow_involutory)


@register_resources({ops.ControlledPhaseShift: 1}, name="to_cps")
def cz_to_cps(wires: WiresLike):
    """Decompose ``CZ`` using the ``to_cps`` strategy."""
    ops.ControlledPhaseShift(np.pi, wires=wires)


@register_resources({ops.H: 2, ops.CNOT: 1}, name="to_cnot")
def cz_to_cnot(wires: WiresLike):
    """Decompose ``CZ`` using the ``to_cnot`` strategy."""
    ops.H(wires=wires[1])
    ops.CNOT(wires=wires)
    ops.H(wires=wires[1])


_cz_to_ppr_resources = partial(_pauli_ctrl_pauli_ppr_resources, pauli0=ops.Z, pauli1=ops.Z)


@register_resources(_cz_to_ppr_resources, name="to_ppr")
def cz_to_ppr(wires: AbstractWires):
    """Decompose ``CZ`` using the ``to_ppr`` strategy."""
    _pauli_ctrl_pauli_ppr(wires, ops.Z, ops.Z)


@register_resources(
    {
        ops.PauliMeasure("ZX", wires=Wire[2]): 1,
        ops.PauliMeasure("ZZ", wires=Wire[2]): 1,
        ops.PauliMeasure("X", wires=Wire[1]): 1,
        ops.Z: 3,
        ops.GlobalPhase: 1,
    },
    work_wires={"burnable": 1},
    name="lattice_surgery_ppm",
)
def cz_lattice_surgery_ppm(wires: WiresLike):
    """Decompose ``CZ`` using the ``lattice_surgery_ppm`` strategy."""
    _pauli_ctrl_pauli_ppm(wires, ops.Z, ops.Z)


add_decomps(ops.CZ, cz_to_cps, cz_to_cnot, cz_to_ppr, cz_lattice_surgery_ppm)
add_decomps("Adjoint(CZ)", self_adjoint)
add_decomps("Pow(CZ)", pow_involutory)


@register_resources({ops.CNOT: 2, ops.Toffoli: 1}, name="to_cnot_toffoli")
def cswap_to_cnot_toffoli(wires: WiresLike, **__):
    """Decompose ``CSWAP`` using the ``to_cnot_toffoli`` strategy."""
    ops.CNOT([wires[2], wires[1]])
    ops.Toffoli(wires=[wires[0], wires[1], wires[2]])
    ops.CNOT([wires[2], wires[1]])


@register_resources(
    {
        ops.PPR_8(-1, pauli_word="ZZZ", wires=Wire[3]): 1,
        ops.PPR_8(-1, pauli_word="ZYY", wires=Wire[3]): 1,
        ops.PPR_8(-1, pauli_word="ZXX", wires=Wire[3]): 1,
        ops.PPR_8(1, pauli_word="ZZ", wires=Wire[2]): 1,
        ops.PPR_8(1, pauli_word="YY", wires=Wire[2]): 1,
        ops.PPR_8(1, pauli_word="XX", wires=Wire[2]): 1,
        ops.PPR_8(1, pauli_word="Z", wires=Wire[1]): 1,
        ops.GlobalPhase: 1,
    },
    name="to_ppr",
)
def cswap_to_ppr(wires: WiresLike, **_):
    """Decompose ``CSWAP`` using the ``to_ppr`` strategy."""
    ops.PPR_8(-1, "ZZZ", wires=wires)
    ops.PPR_8(-1, "ZYY", wires=wires)
    ops.PPR_8(-1, "ZXX", wires=wires)
    ops.PPR_8(1, "ZZ", wires=wires[1:])
    ops.PPR_8(1, "YY", wires=wires[1:])
    ops.PPR_8(1, "XX", wires=wires[1:])
    ops.PPR_8(1, "Z", wires=wires[0])
    ops.GlobalPhase(-np.pi / 8)


add_decomps(ops.CSWAP, cswap_to_cnot_toffoli, cswap_to_ppr)
add_decomps("Adjoint(CSWAP)", self_adjoint)
add_decomps("Pow(CSWAP)", pow_involutory)


@register_resources(
    {ops.CNOT: 6, ops.adjoint(ops.T(Wire[1])): 3, ops.T: 4, ops.Hadamard: 2},
    name="to_clifford_t",
)
def ccz_to_clifford_t(wires: WiresLike, **__):
    """Decompose ``CCZ`` using the ``to_clifford_t`` strategy."""
    ops.CNOT(wires=[wires[1], wires[2]])
    ops.adjoint(ops.T(wires=wires[2]))
    ops.CNOT(wires=[wires[0], wires[2]])
    ops.T(wires=wires[2])
    ops.CNOT(wires=[wires[1], wires[2]])
    ops.adjoint(ops.T(wires=wires[2]))
    ops.CNOT(wires=[wires[0], wires[2]])
    ops.T(wires=wires[2])
    ops.T(wires=wires[1])
    ops.CNOT(wires=[wires[0], wires[1]])
    ops.Hadamard(wires=wires[2])
    ops.T(wires=wires[0])
    ops.adjoint(ops.T(wires=wires[1]))
    ops.CNOT(wires=[wires[0], wires[1]])
    ops.Hadamard(wires=wires[2])


@register_resources({ops.Hadamard: 2, ops.Toffoli: 1}, name="to_toffoli")
def ccz_to_toffoli(wires: WiresLike, **__):
    """Decompose ``CCZ`` using the ``to_toffoli`` strategy."""
    ops.Hadamard(wires[2])
    ops.Toffoli(wires)
    ops.Hadamard(wires[2])


add_decomps(ops.CCZ, ccz_to_clifford_t, ccz_to_toffoli)
add_decomps("Adjoint(CCZ)", self_adjoint)
add_decomps("Pow(CCZ)", pow_involutory)


@register_resources({ops.H: 2, ops.CZ: 1}, name="to_cz_h")
def cnot_to_cz_h(wires: WiresLike):
    """Decompose ``CNOT`` using the ``to_cz_h`` strategy."""
    ops.H(wires[1])
    ops.CZ(wires=wires)
    ops.H(wires[1])


_cnot_to_ppr_resources = partial(_pauli_ctrl_pauli_ppr_resources, pauli0=ops.Z, pauli1=ops.X)


@register_resources(_cnot_to_ppr_resources, name="to_ppr")
def cnot_to_ppr(wires: AbstractWires):
    """Decompose ``CNOT`` using the ``to_ppr`` strategy."""
    _pauli_ctrl_pauli_ppr(wires, ops.Z, ops.X)


@register_resources(
    {
        ops.PauliMeasure("ZX", wires=Wire[2]): 2,
        ops.PauliMeasure("X", wires=Wire[1]): 1,
        ops.Z: 2,
        ops.X: 1,
        ops.GlobalPhase: 1,
    },
    work_wires={"burnable": 1},
    name="lattice_surgery_ppm",
)
def cnot_lattice_surgery_ppm(wires: WiresLike):
    """Decompose ``CNOT`` using the ``lattice_surgery_ppm`` strategy."""
    _pauli_ctrl_pauli_ppm(wires, ops.Z, ops.X)


add_decomps(ops.CNOT, cnot_to_cz_h, cnot_to_ppr, cnot_lattice_surgery_ppm)
add_decomps("Adjoint(CNOT)", self_adjoint)
add_decomps("Pow(CNOT)", pow_involutory)


# pylint: disable-next=unused-argument
def _ctrl_cnot_to_mcx_resources(base, control_wires, control_values, work_wires, work_wire_type):
    ctrl_values = _resolve_ctrl_values(control_values, [True], len(control_wires))
    return {
        ops.MultiControlledX(
            Wire[len(control_wires) + 2],
            control_values=ctrl_values,
            work_wires=work_wires,
            work_wire_type=work_wire_type,
        ): 1
    }


@register_resources(_ctrl_cnot_to_mcx_resources, name="to_mcx")
def ctrl_cnot_to_mcx(base, control_wires, control_values, work_wires, work_wire_type):
    """Decompose ``C(CNOT)`` using the ``to_mcx`` strategy."""
    ctrl_values = _resolve_ctrl_values(control_values, [True], len(control_wires))
    ops.MultiControlledX(
        concatenate_wires(control_wires, base.wires),
        control_values=ctrl_values,
        work_wires=work_wires,
        work_wire_type=work_wire_type,
    )


add_decomps("C(CNOT)", ctrl_cnot_to_mcx)


@register_resources(
    {ops.Hadamard: 2, ops.CNOT: 6, ops.T: 4, ops.adjoint(ops.T(Wire[1])): 3},
    name="to_clifford_t",
)
def toffoli_to_clifford_t(wires: WiresLike):
    """Decompose ``Toffoli`` using the ``to_clifford_t`` strategy."""
    ops.Hadamard(wires=wires[2])
    ops.CNOT(wires=[wires[1], wires[2]])
    ops.adjoint(ops.T(wires=wires[2]))
    ops.CNOT(wires=[wires[0], wires[2]])
    ops.T(wires=wires[2])
    ops.CNOT(wires=[wires[1], wires[2]])
    ops.adjoint(ops.T(wires=wires[2]))
    ops.CNOT(wires=[wires[0], wires[2]])
    ops.T(wires=wires[2])
    ops.T(wires=wires[1])
    ops.CNOT(wires=[wires[0], wires[1]])
    ops.Hadamard(wires=wires[2])
    ops.T(wires=wires[0])
    ops.adjoint(ops.T(wires=wires[1]))
    ops.CNOT(wires=[wires[0], wires[1]])


@register_resources(
    {
        ops.PPR_8(-1, pauli_word="ZZ", wires=Wire[2]): 1,
        ops.PPR_8(-1, pauli_word="ZX", wires=Wire[2]): 2,
        ops.PPR_8(1, pauli_word="ZZX", wires=Wire[3]): 1,
        ops.PPR_8(1, pauli_word="X", wires=Wire[1]): 1,
        ops.PPR_8(1, pauli_word="Z", wires=Wire[1]): 2,
        ops.GlobalPhase: 1,
    },
    name="to_ppr",
)
def toffoli_to_ppr(wires: WiresLike):
    """Decompose ``Toffoli`` using the ``to_ppr`` strategy."""
    ops.PPR_8(-1, "ZZ", wires=wires[:2])
    ops.PPR_8(-1, "ZX", wires=[wires[0], wires[2]])
    ops.PPR_8(-1, "ZX", wires=wires[1:])
    ops.PPR_8(1, "ZZX", wires=wires)
    ops.PPR_8(1, "X", wires=wires[2])
    ops.PPR_8(1, "Z", wires=wires[1])
    ops.PPR_8(1, "Z", wires=wires[0])
    ops.GlobalPhase(-np.pi / 8)


@register_resources(
    {
        _change_op_basis_abstract(
            templates.Elbow,
            ops.CNOT,
            ops.adjoint(templates.Elbow(Wire[3])),
        ): 1
    },
    work_wires={"zeroed": 1},
    name="elbow",
)
def toffoli_elbow(wires: WiresLike):
    """Decompose ``Toffoli`` using the ``elbow`` strategy."""
    with allocate(1, allocation.AllocateState.ZERO, restored=True) as work_wires:
        ops.change_op_basis(
            templates.Elbow([wires[0], wires[1], work_wires[0]]),
            ops.CNOT([work_wires[0], wires[2]]),
        )


add_decomps(ops.Toffoli, toffoli_to_clifford_t, toffoli_to_ppr, toffoli_elbow)
add_decomps("Adjoint(Toffoli)", self_adjoint)
add_decomps("Pow(Toffoli)", pow_involutory)


# pylint: disable-next=unused-argument
def _ctrl_toffoli_to_mcx_resources(base, control_wires, control_values, work_wires, work_wire_type):
    ctrl_values = _resolve_ctrl_values(control_values, [True, True], len(control_wires))
    return {
        ops.MultiControlledX(
            Wire[len(control_wires) + 3],
            control_values=ctrl_values,
            work_wires=work_wires,
            work_wire_type=work_wire_type,
        ): 1
    }


@register_resources(_ctrl_toffoli_to_mcx_resources, name="to_mcx")
def ctrl_toffoli_to_mcx(base, control_wires, control_values, work_wires, work_wire_type):
    """Decompose ``C(Toffoli)`` using the ``to_mcx`` strategy."""
    ctrl_values = _resolve_ctrl_values(control_values, [True, True], len(control_wires))
    ops.MultiControlledX(
        concatenate_wires(control_wires, base.wires),
        control_values=ctrl_values,
        work_wires=work_wires,
        work_wire_type=work_wire_type,
    )


add_decomps("C(Toffoli)", ctrl_toffoli_to_mcx)


@_list_decomps.register
def _list_mcx_decomps(op: ops.MultiControlledX):
    # fixed_decomps should override everything
    if fixed_rule := get_fixed_decomp(op):
        return DecompCollection([fixed_rule])
    if not op.work_wires:
        return DecompCollection(_list_mcx_no_work_wire_decomps(op))
    if len(op.wires) == 2:
        return [mcx_to_cnot_or_toffoli]
    if len(op.wires) == 3:
        return [mcx_to_cnot_or_toffoli, mcx_two_controls_elbows]
    return [mcx_many_work_wires, mcx_two_work_wires, mcx_one_work_wire, mcx_no_work_wires]


def _list_mcx_no_work_wire_decomps(op: ops.MultiControlledX):

    if len(op.wires) == 2:
        return [mcx_to_cnot_or_toffoli]

    if len(op.wires) == 3:
        elbow_rule = _wrap_mcx_rule_w_alloc(
            mcx_two_controls_elbows, 1, "zeroed", name="two_controls_elbows_allocated"
        )
        return [mcx_to_cnot_or_toffoli, elbow_rule]

    capture_compatible_rules = [
        _wrap_mcx_rule_w_alloc(mcx_two_work_wires, 2, "zeroed", "two_zeroed_work_wires"),
        _wrap_mcx_rule_w_alloc(mcx_two_work_wires, 2, "borrowed", "two_borrowed_work_wires"),
        _wrap_mcx_rule_w_alloc(mcx_one_work_wire, 1, "zeroed", "one_zeroed_work_wire"),
        _wrap_mcx_rule_w_alloc(mcx_one_work_wire, 1, "borrowed", "one_borrowed_work_wire"),
        mcx_no_work_wires,
    ]
    if capture.enabled():
        return capture_compatible_rules

    # TODO: the following decomposition rules are not capture compatible [sc-129521]
    return [
        _wrap_mcx_rule_w_alloc(
            mcx_many_work_wires,
            len(op.control_wires) - 2,
            "zeroed",
            "many_zeroed_work_wires",
        ),
        _wrap_mcx_rule_w_alloc(
            mcx_many_work_wires,
            len(op.control_wires) - 2,
            "borrowed",
            "many_borrowed_work_wires",
        ),
    ] + capture_compatible_rules


add_decomps(
    "MultiControlledX",
    mcx_to_cnot_or_toffoli,
    mcx_two_controls_elbows,
    mcx_many_work_wires,
    mcx_two_work_wires,
    mcx_one_work_wire,
    mcx_no_work_wires,
    _wrap_mcx_rule_w_alloc(
        mcx_two_controls_elbows, 1, "zeroed", name="two_controls_elbows_allocated"
    ),
    _wrap_mcx_rule_w_alloc(mcx_two_work_wires, 2, "zeroed", "two_zeroed_work_wires"),
    _wrap_mcx_rule_w_alloc(mcx_two_work_wires, 2, "borrowed", "two_borrowed_work_wires"),
    _wrap_mcx_rule_w_alloc(mcx_one_work_wire, 1, "zeroed", "one_zeroed_work_wire"),
    _wrap_mcx_rule_w_alloc(mcx_one_work_wire, 1, "borrowed", "one_borrowed_work_wire"),
    # TODO: include the allocation-wrapped versions of _mcx_many_work_wires
    # after we support tracer indexing into allocated wires. [sc-129521]
)
add_decomps("Adjoint(MultiControlledX)", self_adjoint)
add_decomps("Pow(MultiControlledX)", pow_involutory)


def _ctrl_mcx_to_mcx_resources(base, control_wires, control_values, work_wires, work_wire_type):
    work_wire_type = resolve_work_wire_type(
        base.work_wires,
        base.work_wire_type,
        work_wires,
        work_wire_type,
    )
    return {
        ops.MultiControlledX(
            control_wires + base.wires,
            control_values=Bool[len(control_values) + len(base.control_values)],
            work_wires=work_wires + base.work_wires,
            work_wire_type=work_wire_type,
        ): 1
    }


@register_resources(_ctrl_mcx_to_mcx_resources, name="to_mcx")
def ctrl_mcx_to_mcx(base, control_wires, control_values, work_wires, work_wire_type):
    """Decompose ``C(MultiControlledX)`` using the ``to_mcx`` strategy."""
    n_ctrl_wires = len(control_wires)
    work_wire_type = resolve_work_wire_type(
        base.work_wires,
        base.work_wire_type,
        work_wires,
        work_wire_type,
    )
    ops.MultiControlledX(
        concatenate_wires(control_wires, base.wires),
        control_values=_resolve_ctrl_values(control_values, base.control_values, n_ctrl_wires),
        work_wires=concatenate_wires(work_wires, base.work_wires),
        work_wire_type=work_wire_type,
    )


add_decomps("C(MultiControlledX)", ctrl_mcx_to_mcx)


@register_resources({ops.RZ: 2, ops.RY: 2, ops.CNOT: 2}, name="to_rz_ry")
def crx_to_rz_ry(phi: TensorLike, wires: WiresLike):
    """Decompose ``CRX`` using the ``to_rz_ry`` strategy."""
    ops.RZ(np.pi / 2, wires=wires[1])
    ops.RY(phi / 2, wires=wires[1])
    ops.CNOT(wires=wires)
    ops.RY(-phi / 2, wires=wires[1])
    ops.CNOT(wires=wires)
    ops.RZ(-np.pi / 2, wires=wires[1])


@register_resources({ops.RX: 2, ops.CZ: 2}, name="to_rx_cz")
def crx_to_rx_cz(phi: TensorLike, wires: WiresLike):
    """Decompose ``CRX`` using the ``to_rx_cz`` strategy."""
    ops.RX(phi / 2, wires=wires[1])
    ops.CZ(wires=wires)
    ops.RX(-phi / 2, wires=wires[1])
    ops.CZ(wires=wires)


@register_resources({ops.Hadamard: 2, ops.CRZ: 1}, name="to_h_crz")
def crx_to_h_crz(phi: TensorLike, wires: WiresLike):
    """Decompose ``CRX`` using the ``to_h_crz`` strategy."""
    ops.Hadamard(wires=wires[1])
    ops.CRZ(phi, wires=wires)
    ops.Hadamard(wires=wires[1])


@register_resources(
    {
        ops.PauliRot(Float, pauli_word="ZX", wires=Wire[2]): 1,
        ops.PauliRot(Float, pauli_word="X", wires=Wire[1]): 1,
    },
    name="to_paulirot",
)
def crx_to_ppr(phi: TensorLike, wires: WiresLike):
    """Decompose ``CRX`` using the ``to_paulirot`` strategy."""
    ops.PauliRot(phi / 2, "X", wires=wires[1])
    ops.PauliRot(-phi / 2, "ZX", wires=wires)


add_decomps(ops.CRX, crx_to_rx_cz, crx_to_rz_ry, crx_to_h_crz, crx_to_ppr)
add_decomps("Adjoint(CRX)", adjoint_negate_angle)
add_decomps("Pow(CRX)", pow_scale_angle)


@register_resources({ops.RY: 2, ops.CNOT: 2}, name="to_ry_cnot")
def cry_to_ry_cnot(phi: TensorLike, wires: WiresLike):
    """Decompose ``CRY`` using the ``to_ry_cnot`` strategy."""
    ops.RY(phi / 2, wires=wires[1])
    ops.CNOT(wires=wires)
    ops.RY(-phi / 2, wires=wires[1])
    ops.CNOT(wires=wires)


@register_resources(
    {
        ops.PauliRot(Float, pauli_word="ZY", wires=Wire[2]): 1,
        ops.PauliRot(Float, pauli_word="Y", wires=Wire[1]): 1,
    },
    name="to_paulirot",
)
def cry_to_ppr(phi: TensorLike, wires: WiresLike):
    """Decompose ``CRY`` using the ``to_paulirot`` strategy."""
    ops.PauliRot(phi / 2, "Y", wires=wires[1])
    ops.PauliRot(-phi / 2, "ZY", wires=wires)


add_decomps(ops.CRY, cry_to_ry_cnot, cry_to_ppr)
add_decomps("Adjoint(CRY)", adjoint_negate_angle)
add_decomps("Pow(CRY)", pow_scale_angle)


@register_resources({ops.RZ: 2, ops.CNOT: 2}, name="to_rz_cnot")
def crz_to_rz_cnot(phi: TensorLike, wires: WiresLike):
    """Decompose ``CRZ`` using the ``to_rz_cnot`` strategy."""
    ops.RZ(phi / 2, wires=wires[1])
    ops.CNOT(wires=wires)
    ops.RZ(-phi / 2, wires=wires[1])
    ops.CNOT(wires=wires)


@register_resources(
    {
        ops.PauliRot(Float, pauli_word="ZZ", wires=Wire[2]): 1,
        ops.PauliRot(Float, pauli_word="Z", wires=Wire[1]): 1,
    },
    name="to_paulirot",
)
def crz_to_ppr(phi: TensorLike, wires: WiresLike):
    """Decompose ``CRZ`` using the ``to_paulirot`` strategy."""
    ops.PauliRot(phi / 2, "Z", wires=wires[1])
    ops.PauliRot(-phi / 2, "ZZ", wires=wires)


add_decomps(ops.CRZ, crz_to_rz_cnot, crz_to_ppr)
add_decomps("Adjoint(CRZ)", adjoint_negate_angle)
add_decomps("Pow(CRZ)", pow_scale_angle)


@register_resources({ops.RZ: 3, ops.CNOT: 2, ops.RY: 2}, name="to_rot_cnot")
def crot_to_rot_cnot(phi: TensorLike, theta: TensorLike, omega: TensorLike, wires: WiresLike):
    """Decompose ``CRot`` using the ``to_rot_cnot`` strategy."""
    ops.RZ((phi - omega) / 2, wires=wires[1])
    ops.CNOT(wires=wires)
    ops.RZ(-(phi + omega) / 2, wires=wires[1])
    ops.RY(-theta / 2, wires=wires[1])
    ops.CNOT(wires=wires)
    ops.RY(theta / 2, wires=wires[1])
    ops.RZ(omega, wires=wires[1])


add_decomps(ops.CRot, crot_to_rot_cnot)


@register_resources({ops.CRot: 1}, name="reverse_angles")
def adjoint_crot_standard(base):
    """Decompose ``Adjoint(CRot)`` using the ``reverse_angles`` strategy."""
    ops.CRot(-base.omega, -base.theta, -base.phi, wires=base.wires)


add_decomps("Adjoint(CRot)", adjoint_crot_standard)


@register_resources({ops.RZ: 3, ops.CNOT: 2, ops.GlobalPhase: 1}, name="to_rz_cnot")
def cphase_to_rz_cnot(phi: TensorLike, wires: WiresLike, **__):
    """Decompose ``ControlledPhaseShift`` using the ``to_rz_cnot`` strategy."""
    ops.RZ(phi / 2, wires=wires[0])
    ops.CNOT(wires=wires)
    ops.RZ(-phi / 2, wires=wires[1])
    ops.CNOT(wires=wires)
    ops.RZ(phi / 2, wires=wires[1])
    ops.GlobalPhase(-phi / 4)


@register_resources(
    {
        ops.GlobalPhase: 1,
        ops.PauliRot(Float, pauli_word="Z", wires=Wire[1]): 2,
        ops.PauliRot(Float, pauli_word="ZZ", wires=Wire[2]): 1,
    },
    name="to_paulirot",
)
def cphase_to_ppr(phi: TensorLike, wires: WiresLike, **__):
    """Decompose ``ControlledPhaseShift`` using the ``to_paulirot`` strategy."""
    ops.PauliRot(-phi / 2, pauli_word="ZZ", wires=wires)
    ops.PauliRot(phi / 2, pauli_word="Z", wires=wires[1])
    ops.PauliRot(phi / 2, pauli_word="Z", wires=wires[0])
    ops.GlobalPhase(-phi / 4)


@register_resources(
    {
        _change_op_basis_abstract(
            templates.TemporaryAND(wires=Wire[3]),
            ops.PhaseShift,
            ops.adjoint(templates.TemporaryAND(wires=Wire[3])),
        ): 1,
    },
    work_wires={"zeroed": 1},
    name="to_single_phase",
)
def cphase_to_single_phase(phi: TensorLike, wires: WiresLike, **__):
    """Decompose ``ControlledPhaseShift`` using the ``to_single_phase`` strategy."""
    with allocate(1, state="zero", restored=True) as work_wire:
        elbow = templates.TemporaryAND([wires[0], wires[1], work_wire[0]])
        ops.change_op_basis(elbow, ops.PhaseShift(phi, work_wire))


add_decomps(ops.ControlledPhaseShift, cphase_to_rz_cnot, cphase_to_ppr, cphase_to_single_phase)
add_decomps("Adjoint(ControlledPhaseShift)", adjoint_negate_angle)
add_decomps("Pow(ControlledPhaseShift)", pow_scale_angle)
