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

# pylint: disable=unused-argument


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
from pennylane.decomposition_rules.op_math.adjoint2 import (
    _adjoint_negate_angle as adjoint_rotation2,
)
from pennylane.decomposition_rules.op_math.pow2 import _pow_scale_angle as pow_rotation2
from pennylane.decomposition_rules.op_math.pow2 import pow_involutory as pow_involutory2
from pennylane.decomposition_rules.op_math.symbolicop2 import self_adjoint
from pennylane.decomposition_rules.qubit.matrix_ops import _qubit_unitary_two_qubit
from pennylane.decomposition_rules.qubit.parametric_ops_single_qubit import (
    _precision_dispatcher,
    validate_phase_gradient_wires,
)
from pennylane.ops.identity import GlobalPhase
from pennylane.ops.mid_measure.pauli_measure import PauliMeasure, pauli_measure
from pennylane.ops.op_math import ChangeOpBasis2
from pennylane.ops.op_math.adjoint2 import _adjoint_abstract
from pennylane.ops.op_math.change_op_basis2 import _change_op_basis_abstract
from pennylane.ops.op_math.controlled import _resolve_ctrl_values
from pennylane.ops.op_math.controlled2 import ControlledOp2
from pennylane.ops.op_math.controlled_ops import (
    CCZ,
    CH,
    CNOT,
    CRX,
    CRY,
    CRZ,
    CSWAP,
    CY,
    CZ,
    ControlledPhaseShift,
    ControlledQubitUnitary,
    CRot,
    MultiControlledX,
    Toffoli,
)
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
from pennylane.ops.qubit import X, Y, Z
from pennylane.ops.subcircuit import subcircuit
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
        X: num_ctrl_wires,  # assume half the control values are 0s
    } | mcx_reps


# Resources are not exact because rotations might be skipped for zero angles
@register_condition(_ctrl_decomp_bisect_condition)
@register_resources(_controlled_unitary_bisect_resources, exact=False, name="bisect")
def _controlled_unitary_bisect(U, wires, control_values, *_, **__):
    """The decomposition rule for ControlledQubitUnitary from
    `Vale et al. (2023) <https://arxiv.org/abs/2302.06377>`_."""
    return _controlled_unitary_bisect_impl(U, wires, control_values, *_, **__)


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
def _controlled_unitary_single_control_zyz(U, wires, control_values, work_wires, **_):
    """The decomposition rule for ControlledQubitUnitary from Lemma 5.1 of
    https://arxiv.org/pdf/quant-ph/9503016"""

    if capture.enabled() or compiler.active():
        wires = math.array(wires, like="jax")
        control_values = math.array(control_values, like="jax")

    @control_flow.for_loop(len(control_values))
    def _x_flips(i):
        ops.cond(math.logical_not(control_values[i]), X)(wires[i])

    phi, theta, omega, phase = math.decomposition.zyz_rotation_angles(U)

    _x_flips()  # pylint: disable=no-value-for-parameter
    _single_control_zyz(phi, theta, omega, wires=wires)
    ops.cond(_not_zero(phase), _ctrl_global_phase)(phase, wires[:-1])
    _x_flips()  # pylint: disable=no-value-for-parameter


def _multi_ctrl_decomp_zyz_condition(U, wires, *_, **__):
    num_target_wires = int(math.log2(math.shape(U)[-1]))
    num_control_wires = len(wires) - num_target_wires
    return num_target_wires == 1 and num_control_wires > 1


def _controlled_unitary_multi_control_zyz_resources(
    U, wires, control_values, work_wires, work_wire_type, **_  # pylint: disable=unused-argument
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
def _controlled_unitary_multi_control_zyz(
    U, wires, control_values, work_wires, work_wire_type, **_
):
    """The decomposition rule for ControlledQubitUnitary from Lemma 7.9 of
    https://arxiv.org/pdf/quant-ph/9503016"""

    if compiler.active() or capture.enabled():
        wires = math.array(wires, like="jax")
        control_values = math.array(control_values, like="jax")

    @control_flow.for_loop(len(control_values))
    def _x_flips(i):
        ops.cond(math.logical_not(control_values[i]), X)(wires[i])

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


def _controlled_unitary_two_qubit_resources(
    U, wires, control_values, work_wires, work_wire_type, **_  # pylint: disable=unused-argument
):
    num_target_wires = int(math.log2(math.shape(U)[-1]))
    num_control_wires = len(wires) - num_target_wires
    num_work_wires = len(work_wires)
    base_resources = _qubit_unitary_two_qubit.compute_resources(num_wires=num_target_wires)
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
def _controlled_unitary_two_qubit(U, wires, control_values, work_wires, work_wire_type, **_):
    """A controlled two-qubit unitary is decomposed by applying ctrl to the base decomposition."""

    if compiler.active() or capture.enabled():
        wires = math.array(wires, like="jax")
        control_values = math.array(control_values, like="jax")

    @control_flow.for_loop(len(control_values))
    def _x_flips(i):
        ops.cond(math.logical_not(control_values[i]), X)(wires[i])

    _x_flips()  # pylint: disable=no-value-for-parameter
    ops.ctrl(
        _qubit_unitary_two_qubit._impl,  # pylint: disable=protected-access
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
        return {CNOT: 1, X: 1}
    return {Toffoli: 1, X: 2}


@register_condition(lambda wires, *_, **__: len(wires) <= 3)
@register_resources(_mcx_to_cnot_or_toffoli_resources, exact=False, name="to_cnot_or_toffoli")
def _mcx_to_cnot_or_toffoli(wires, control_values, *_, **__):
    """A decomposition rule that decomposes an MCX to a CNOT or Toffoli"""
    return _mcx_to_cnot_or_toffoli_impl(wires, control_values, *_, **__)


def _mcx_two_controls_elbows_resources(wires, *_, **__):
    num_ctrl_wires = len(wires) - 1
    return {
        templates.Elbow: 1,
        CNOT: 1,
        ops.adjoint(templates.Elbow(Wire[3])): 1,
        X: num_ctrl_wires,
    }


# pylint: disable-next=unused-argument
def _2cx_elbow_explicit_condition(wires, control_values, work_wires, work_wire_type="borrowed"):
    num_control_wires = len(wires) - 1
    return len(work_wires) >= 1 and num_control_wires == 2 and work_wire_type == "zeroed"


@register_condition(_2cx_elbow_explicit_condition)
@register_resources(_mcx_two_controls_elbows_resources, exact=False, name="two_controls_elbows")
def _mcx_two_controls_elbows(wires, control_values, work_wires, **_):
    """A decomposition rule for MCX that uses controlled Elbow."""

    if compiler.active() or capture.enabled():
        wires = math.array(wires, like="jax")
        control_values = math.array(control_values, like="jax")

    @control_flow.for_loop(len(control_values))
    def _x_flips(i):
        ops.cond(math.logical_not(control_values[i]), X)(wires[i])

    elbow_wires = [wires[0], wires[1], work_wires[0]]

    _x_flips()  # pylint: disable=no-value-for-parameter
    templates.Elbow(elbow_wires)
    CNOT([work_wires[0], wires[2]])
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
            X: num_control_wires,
        }

    return {
        templates.TemporaryAND: num_used_work_wires,
        ops.adjoint(templates.TemporaryAND(Wire[3])): num_used_work_wires,
        middle_toffoli: 1,
        X: num_control_wires,
    }


@register_condition(_mcx_many_workers_condition)
@register_resources(_mcx_many_work_wires_resources, exact=False, name="many_work_wires")
def _mcx_many_work_wires(wires, control_values, work_wires, work_wire_type="borrowed"):
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
def _mcx_two_work_wires(wires, control_values, work_wires, work_wire_type="borrowed"):
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
def _mcx_one_work_wire(wires, control_values, work_wires, work_wire_type="zeroed"):
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
def _mcx_no_work_wires(wires, control_values, *_, **__):
    """Use ctrl_decomp_bisect_md to decompose a multi-controlled X gate with no work wires."""
    return _mcx_no_work_wires_impl(wires, control_values, *_, **__)


def _to_general_c_qu_standard_resources(U, wires, control_values, work_wires, work_wire_type, **_):
    num_target_wires = int(math.log2(math.shape(U)[-1]))
    num_control_wires = len(wires) - num_target_wires
    num_work_wires = len(work_wires)
    u_size = 2**num_target_wires
    return {
        ControlledOp2(
            ops.QubitUnitary(Complex[u_size, u_size], wires=Wire[num_target_wires]),
            control_wires=Wire[num_control_wires],
            control_values=Bool[num_control_wires],
            work_wires=Wire[num_work_wires],
            work_wire_type=work_wire_type,
        ): 1
    }


@register_condition(lambda U, *_, **__: math.ceil_log2(math.shape(U)[0]) > 2)
@register_resources(_to_general_c_qu_standard_resources, name="to_controlled_qubit_unitary")
def _to_general_c_qu_standard(U, wires, control_values, work_wires, work_wire_type, **_):
    """Convert a ControlledQubitUnitary to a general Controlled(QubitUnitary) so that
    the graph finds the general decomposition rule of applying control to the decomposition
    of the base QubitUnitary."""
    num_target_wires = len(wires) - len(control_values)
    ControlledOp2(
        ops.QubitUnitary(U, wires=wires[-num_target_wires:]),
        control_wires=wires[:-num_target_wires],
        control_values=control_values,
        work_wires=work_wires,
        work_wire_type=work_wire_type,
    )


add_decomps(
    ControlledQubitUnitary,
    _controlled_unitary_bisect,
    _controlled_unitary_single_control_zyz,
    _controlled_unitary_multi_control_zyz,
    _controlled_unitary_two_qubit,
    _to_general_c_qu_standard,
)


def _conj_transpose(U):
    return U.conjugate().transpose() if sparse.issparse(U) else math.moveaxis(math.conj(U), -2, -1)


@register_resources(lambda base: {abstractify(base): 1}, name="conjugate_transpose")
def _adjoint_controlled_qubit_unitary_standard(base):
    ControlledQubitUnitary(
        _conj_transpose(base.U),
        wires=base.wires,
        control_values=base.control_values,
        work_wires=base.work_wires,
        work_wire_type=base.work_wire_type,
    )


add_decomps("Adjoint(ControlledQubitUnitary)", _adjoint_controlled_qubit_unitary_standard)


def _ch_to_ry_cz_ry_resources(wires: WiresLike = None):
    return {ops.RY: 2, CZ: 1}


@register_resources(_ch_to_ry_cz_ry_resources, name="to_ry_cz_ry")
def _ch_to_ry_cz_ry(wires: WiresLike, **__):
    ops.RY(-np.pi / 4, wires=wires[1])
    CZ(wires=wires)
    ops.RY(+np.pi / 4, wires=wires[1])


add_decomps(CH, _ch_to_ry_cz_ry)

add_decomps("Adjoint(CH)", self_adjoint)

add_decomps("Pow(CH)", pow_involutory2)


def _cy_to_ry_cnot_resources(wires: AbstractWires):
    return {CRY: 1, ops.S: 1}


@register_resources(_cy_to_ry_cnot_resources, name="to_ry_cnot")
def _cy_to_ry_cnot(wires: WiresLike, **__):
    CRY(np.pi, wires=wires)
    ops.S(wires=wires[0])


def _pauli_ctrl_pauli_ppr_resources(wires: AbstractWires, pauli0, pauli1):
    """Resources for _pauli_ctrl_pauli_ppr."""
    resources = defaultdict(int)
    p0, p1 = pauli0.__name__[-1], pauli1.__name__[-1]
    resources[ops.PPR_4(-1, pauli_word=p0, wires=Wire[len(p0)])] += 1
    resources[ops.PPR_4(-1, pauli_word=p1, wires=Wire[len(p1)])] += 1
    resources[ops.PPR_4(1, pauli_word=p0 + p1, wires=Wire[len(p0 + p1)])] += 1
    resources[GlobalPhase] += 1
    return dict(resources)


def _pauli_ctrl_pauli_ppr(wires: AbstractWires, pauli0, pauli1):
    """Generalized two-qubit Pauli-controlled Pauli gate decomposition to PPRs."""
    p0, p1 = pauli0.__name__[-1], pauli1.__name__[-1]
    ops.PPR_4(-1, p0, wires=wires[0])
    ops.PPR_4(-1, p1, wires=wires[1])
    ops.PPR_4(1, p0 + p1, wires=wires)
    GlobalPhase(np.pi / 4)


@register_resources(partial(_pauli_ctrl_pauli_ppr_resources, pauli0=Z, pauli1=Y), name="to_ppr")
def _cy_to_ppr(wires: AbstractWires):
    _pauli_ctrl_pauli_ppr(wires, Z, Y)


def _pauli_ctrl_pauli_ppm(wires, pauli0, pauli1):
    """Generalized two-qubit Pauli-controlled Pauli gate decomposition to PPMs."""
    with allocate(1, state="zero", restored=False) as work_wires:
        m0 = pauli_measure(pauli0.__name__[-1] + "X", [wires[0], work_wires[0]])
        m1 = pauli_measure("Z" + pauli1.__name__[-1], [work_wires[0], wires[1]])
        m2 = pauli_measure("X", work_wires)
        ops.cond(m1, pauli0)(wires[0])
        ops.cond(m0 != m2, pauli1)(wires[1])
        ops.cond(m1 & (m0 != m2), GlobalPhase)(np.pi)
        ops.cond(m2, Z)(work_wires[0])  # Reset work wire (to |+>), achieving pure state


@register_resources(
    {
        PauliMeasure("ZX", wires=Wire[2]): 1,
        PauliMeasure("ZY", wires=Wire[2]): 1,
        PauliMeasure("X", wires=Wire[1]): 1,
        Z: 2,
        Y: 1,
        GlobalPhase: 1,
    },
    work_wires={"burnable": 1},
    name="lattice_surgery_ppm",
)
def _cy_lattice_surgery_ppm(wires: AbstractWires):
    _pauli_ctrl_pauli_ppm(wires, Z, Y)


add_decomps(CY, _cy_to_ry_cnot, _cy_to_ppr, _cy_lattice_surgery_ppm)

add_decomps("Adjoint(CY)", self_adjoint)

add_decomps("Pow(CY)", pow_involutory2)


def _cz_to_cps_resources(wires: AbstractWires):
    return {ControlledPhaseShift: 1}


@register_resources(_cz_to_cps_resources, name="to_cps")
def _cz_to_cps(wires: WiresLike):
    ControlledPhaseShift(np.pi, wires=wires)


def _cz_to_cnot_resources(wires: AbstractWires):
    return {ops.H: 2, CNOT: 1}


@register_resources(_cz_to_cnot_resources, name="to_cnot")
def _cz_to_cnot(wires: WiresLike):
    ops.H(wires=wires[1])
    CNOT(wires=wires)
    ops.H(wires=wires[1])


@register_resources(partial(_pauli_ctrl_pauli_ppr_resources, pauli0=Z, pauli1=Z), name="to_ppr")
def _cz_to_ppr(wires: AbstractWires):
    _pauli_ctrl_pauli_ppr(wires, Z, Z)


def _cz_lattice_surgery_ppm_resources(wires: AbstractWires):
    return {
        PauliMeasure("ZX", wires=Wire[2]): 1,
        PauliMeasure("ZZ", wires=Wire[2]): 1,
        PauliMeasure("X", wires=Wire[1]): 1,
        Z: 3,
        GlobalPhase: 1,
    }


@register_resources(
    _cz_lattice_surgery_ppm_resources, work_wires={"burnable": 1}, name="lattice_surgery_ppm"
)
def _cz_lattice_surgery_ppm(wires: WiresLike):
    _pauli_ctrl_pauli_ppm(wires, Z, Z)


add_decomps(CZ, _cz_to_cps, _cz_to_cnot, _cz_to_ppr, _cz_lattice_surgery_ppm)

add_decomps("Adjoint(CZ)", self_adjoint)

add_decomps("Pow(CZ)", pow_involutory2)


def _cswap_to_cnot_toffoli_resources(wires: WiresLike = None):
    return {CNOT: 2, Toffoli: 1}


@register_resources(_cswap_to_cnot_toffoli_resources, name="to_cnot_toffoli")
def _cswap_to_cnot_toffoli(wires: WiresLike, **__):
    CNOT([wires[2], wires[1]])
    Toffoli(wires=[wires[0], wires[1], wires[2]])
    CNOT([wires[2], wires[1]])


def _cswap_to_ppr_resources(wires: WiresLike = None):
    return {
        ops.PPR_8(-1, pauli_word="ZZZ", wires=Wire[3]): 1,
        ops.PPR_8(-1, pauli_word="ZYY", wires=Wire[3]): 1,
        ops.PPR_8(-1, pauli_word="ZXX", wires=Wire[3]): 1,
        ops.PPR_8(1, pauli_word="ZZ", wires=Wire[2]): 1,
        ops.PPR_8(1, pauli_word="YY", wires=Wire[2]): 1,
        ops.PPR_8(1, pauli_word="XX", wires=Wire[2]): 1,
        ops.PPR_8(1, pauli_word="Z", wires=Wire[1]): 1,
        GlobalPhase: 1,
    }


@register_resources(_cswap_to_ppr_resources, name="to_ppr")
def _cswap_to_ppr(wires: WiresLike, **_):
    ops.PPR_8(-1, "ZZZ", wires=wires)
    ops.PPR_8(-1, "ZYY", wires=wires)
    ops.PPR_8(-1, "ZXX", wires=wires)
    ops.PPR_8(1, "ZZ", wires=wires[1:])
    ops.PPR_8(1, "YY", wires=wires[1:])
    ops.PPR_8(1, "XX", wires=wires[1:])
    ops.PPR_8(1, "Z", wires=wires[0])
    GlobalPhase(-np.pi / 8)


add_decomps(CSWAP, _cswap_to_cnot_toffoli, _cswap_to_ppr)

add_decomps("Adjoint(CSWAP)", self_adjoint)

add_decomps("Pow(CSWAP)", pow_involutory2)


def _ccz_to_clifford_t_resources(wires: AbstractWires):
    return {
        CNOT: 6,
        _adjoint_abstract(ops.T): 3,
        ops.T: 4,
        ops.Hadamard: 2,
    }


@register_resources(_ccz_to_clifford_t_resources, name="to_clifford_t")
def _ccz_to_clifford_t(wires: WiresLike, **__):
    CNOT(wires=[wires[1], wires[2]])
    ops.adjoint(ops.T(wires=wires[2]))
    CNOT(wires=[wires[0], wires[2]])
    ops.T(wires=wires[2])
    CNOT(wires=[wires[1], wires[2]])
    ops.adjoint(ops.T(wires=wires[2]))
    CNOT(wires=[wires[0], wires[2]])
    ops.T(wires=wires[2])
    ops.T(wires=wires[1])
    CNOT(wires=[wires[0], wires[1]])
    ops.Hadamard(wires=wires[2])
    ops.T(wires=wires[0])
    ops.adjoint(ops.T(wires=wires[1]))
    CNOT(wires=[wires[0], wires[1]])
    ops.Hadamard(wires=wires[2])


def _ccz_to_toffoli_resources(wires: AbstractWires):
    return {ops.Hadamard: 2, Toffoli: 1}


@register_resources(_ccz_to_toffoli_resources, name="to_toffoli")
def _ccz_to_toffoli(wires: WiresLike, **__):
    ops.Hadamard(wires[2])
    Toffoli(wires)
    ops.Hadamard(wires[2])


add_decomps(CCZ, _ccz_to_clifford_t, _ccz_to_toffoli)

add_decomps("Adjoint(CCZ)", self_adjoint)

add_decomps("Pow(CCZ)", pow_involutory2)


def _cnot_to_cz_h_resources(wires: WiresLike):
    return {ops.H: 2, CZ: 1}


@register_resources(_cnot_to_cz_h_resources, name="to_cz_h")
def _cnot_to_cz_h(wires: WiresLike):
    ops.H(wires[1])
    CZ(wires=wires)
    ops.H(wires[1])


@register_resources(partial(_pauli_ctrl_pauli_ppr_resources, pauli0=Z, pauli1=X), name="to_ppr")
def _cnot_to_ppr(wires: AbstractWires):
    _pauli_ctrl_pauli_ppr(wires, Z, X)


def _cnot_lattice_surgery_ppm_resources(wires: WiresLike):
    return {
        PauliMeasure("ZX", wires=Wire[2]): 2,
        PauliMeasure("X", wires=Wire[1]): 1,
        Z: 2,
        X: 1,
        GlobalPhase: 1,
    }


@register_resources(
    _cnot_lattice_surgery_ppm_resources, work_wires={"burnable": 1}, name="lattice_surgery_ppm"
)
def _cnot_lattice_surgery_ppm(wires: WiresLike):
    _pauli_ctrl_pauli_ppm(wires, Z, X)


add_decomps(CNOT, _cnot_to_cz_h, _cnot_to_ppr, _cnot_lattice_surgery_ppm)

add_decomps("Adjoint(CNOT)", self_adjoint)

add_decomps("Pow(CNOT)", pow_involutory2)


def _ctrl_cnot_to_mcx_resources(base, control_wires, control_values, work_wires, work_wire_type):
    ctrl_values = _resolve_ctrl_values(control_values, [True], len(control_wires))
    return {
        MultiControlledX(
            Wire[len(control_wires) + 2],
            control_values=ctrl_values,
            work_wires=work_wires,
            work_wire_type=work_wire_type,
        ): 1
    }


@register_resources(_ctrl_cnot_to_mcx_resources, name="to_mcx")
def _ctrl_cnot_to_mcx(base, control_wires, control_values, work_wires, work_wire_type):
    ctrl_values = _resolve_ctrl_values(control_values, [True], len(control_wires))
    MultiControlledX(
        concatenate_wires(control_wires, base.wires),
        control_values=ctrl_values,
        work_wires=work_wires,
        work_wire_type=work_wire_type,
    )


add_decomps("C(CNOT)", _ctrl_cnot_to_mcx)


def _toffoli_to_clifford_t_resources(wires: WiresLike):
    return {
        ops.Hadamard: 2,
        CNOT: 6,
        ops.T: 4,
        _adjoint_abstract(ops.T): 3,
    }


@register_resources(_toffoli_to_clifford_t_resources, name="to_clifford_t")
def _toffoli_to_clifford_t(wires: WiresLike):
    ops.Hadamard(wires=wires[2])
    CNOT(wires=[wires[1], wires[2]])
    ops.adjoint(ops.T(wires=wires[2]))
    CNOT(wires=[wires[0], wires[2]])
    ops.T(wires=wires[2])
    CNOT(wires=[wires[1], wires[2]])
    ops.adjoint(ops.T(wires=wires[2]))
    CNOT(wires=[wires[0], wires[2]])
    ops.T(wires=wires[2])
    ops.T(wires=wires[1])
    CNOT(wires=[wires[0], wires[1]])
    ops.Hadamard(wires=wires[2])
    ops.T(wires=wires[0])
    ops.adjoint(ops.T(wires=wires[1]))
    CNOT(wires=[wires[0], wires[1]])


def _toffoli_to_ppr_resources(wires: WiresLike):
    return {
        ops.PPR_8(-1, pauli_word="ZZ", wires=Wire[2]): 1,
        ops.PPR_8(-1, pauli_word="ZX", wires=Wire[2]): 2,
        ops.PPR_8(1, pauli_word="ZZX", wires=Wire[3]): 1,
        ops.PPR_8(1, pauli_word="X", wires=Wire[1]): 1,
        ops.PPR_8(1, pauli_word="Z", wires=Wire[1]): 2,
        GlobalPhase: 1,
    }


@register_resources(_toffoli_to_ppr_resources, name="to_ppr")
def _toffoli_to_ppr(wires: WiresLike):
    ops.PPR_8(-1, "ZZ", wires=wires[:2])
    ops.PPR_8(-1, "ZX", wires=[wires[0], wires[2]])
    ops.PPR_8(-1, "ZX", wires=wires[1:])
    ops.PPR_8(1, "ZZX", wires=wires)
    ops.PPR_8(1, "X", wires=wires[2])
    ops.PPR_8(1, "Z", wires=wires[1])
    ops.PPR_8(1, "Z", wires=wires[0])
    GlobalPhase(-np.pi / 8)


def _toffoli_elbow_resources(*_, **__):
    return {
        _change_op_basis_abstract(templates.Elbow, CNOT, ops.adjoint(templates.Elbow(Wire[3]))): 1
    }


@register_resources(_toffoli_elbow_resources, work_wires={"zeroed": 1}, name="elbow")
def _toffoli_elbow(wires: WiresLike):
    with allocate(1, allocation.AllocateState.ZERO, restored=True) as work_wires:
        ops.change_op_basis(
            templates.Elbow([wires[0], wires[1], work_wires[0]]),
            CNOT([work_wires[0], wires[2]]),
        )


add_decomps(Toffoli, _toffoli_to_clifford_t, _toffoli_to_ppr, _toffoli_elbow)

add_decomps("Adjoint(Toffoli)", self_adjoint)

add_decomps("Pow(Toffoli)", pow_involutory2)


def _ctrl_toffoli_to_mcx_resources(base, control_wires, control_values, work_wires, work_wire_type):
    ctrl_values = _resolve_ctrl_values(control_values, [True, True], len(control_wires))
    return {
        MultiControlledX(
            Wire[len(control_wires) + 3],
            control_values=ctrl_values,
            work_wires=work_wires,
            work_wire_type=work_wire_type,
        ): 1
    }


@register_resources(_ctrl_toffoli_to_mcx_resources, name="to_mcx")
def _ctrl_toffoli_to_mcx(base, control_wires, control_values, work_wires, work_wire_type):
    ctrl_values = _resolve_ctrl_values(control_values, [True, True], len(control_wires))
    MultiControlledX(
        concatenate_wires(control_wires, base.wires),
        control_values=ctrl_values,
        work_wires=work_wires,
        work_wire_type=work_wire_type,
    )


add_decomps("C(Toffoli)", _ctrl_toffoli_to_mcx)


@_list_decomps.register
def _list_mcx_decomps(op: MultiControlledX):
    # fixed_decomps should override everything
    if fixed_rule := get_fixed_decomp(op):
        return DecompCollection([fixed_rule])
    if not op.work_wires:
        return DecompCollection(_list_mcx_no_work_wire_decomps(op))
    if len(op.wires) == 2:
        return [_mcx_to_cnot_or_toffoli]
    if len(op.wires) == 3:
        return [_mcx_to_cnot_or_toffoli, _mcx_two_controls_elbows]
    return [
        _mcx_many_work_wires,
        _mcx_two_work_wires,
        _mcx_one_work_wire,
        _mcx_no_work_wires,
    ]


def _list_mcx_no_work_wire_decomps(op: MultiControlledX):

    if len(op.wires) == 2:
        return [_mcx_to_cnot_or_toffoli]

    if len(op.wires) == 3:
        elbow_rule = _wrap_mcx_rule_w_alloc(
            _mcx_two_controls_elbows, 1, "zeroed", name="two_controls_elbows_allocated"
        )
        return [_mcx_to_cnot_or_toffoli, elbow_rule]

    capture_compatible_rules = [
        _wrap_mcx_rule_w_alloc(_mcx_two_work_wires, 2, "zeroed", "two_zeroed_work_wires"),
        _wrap_mcx_rule_w_alloc(_mcx_two_work_wires, 2, "borrowed", "two_borrowed_work_wires"),
        _wrap_mcx_rule_w_alloc(_mcx_one_work_wire, 1, "zeroed", "one_zeroed_work_wire"),
        _wrap_mcx_rule_w_alloc(_mcx_one_work_wire, 1, "borrowed", "one_borrowed_work_wire"),
        _mcx_no_work_wires,
    ]
    if capture.enabled():
        return capture_compatible_rules

    # TODO: the following decomposition rules are not capture compatible [sc-129521]
    return [
        _wrap_mcx_rule_w_alloc(
            _mcx_many_work_wires,
            len(op.control_wires) - 2,
            "zeroed",
            "many_zeroed_work_wires",
        ),
        _wrap_mcx_rule_w_alloc(
            _mcx_many_work_wires,
            len(op.control_wires) - 2,
            "borrowed",
            "many_borrowed_work_wires",
        ),
    ] + capture_compatible_rules


add_decomps(
    "MultiControlledX",
    _mcx_to_cnot_or_toffoli,
    _mcx_two_controls_elbows,
    _mcx_many_work_wires,
    _mcx_two_work_wires,
    _mcx_one_work_wire,
    _mcx_no_work_wires,
    _wrap_mcx_rule_w_alloc(
        _mcx_two_controls_elbows, 1, "zeroed", name="two_controls_elbows_allocated"
    ),
    _wrap_mcx_rule_w_alloc(_mcx_two_work_wires, 2, "zeroed", "two_zeroed_work_wires"),
    _wrap_mcx_rule_w_alloc(_mcx_two_work_wires, 2, "borrowed", "two_borrowed_work_wires"),
    _wrap_mcx_rule_w_alloc(_mcx_one_work_wire, 1, "zeroed", "one_zeroed_work_wire"),
    _wrap_mcx_rule_w_alloc(_mcx_one_work_wire, 1, "borrowed", "one_borrowed_work_wire"),
    # TODO: include the allocation-wrapped versions of _mcx_many_work_wires
    # after we support tracer indexing into allocated wires. [sc-129521]
)

add_decomps("Adjoint(MultiControlledX)", self_adjoint)

add_decomps("Pow(MultiControlledX)", pow_involutory2)


def _ctrl_mcx_to_mcx_resources(base, control_wires, control_values, work_wires, work_wire_type):
    work_wire_type = resolve_work_wire_type(
        base.work_wires,
        base.work_wire_type,
        work_wires,
        work_wire_type,
    )
    return {
        MultiControlledX(
            control_wires + base.wires,
            control_values=Bool[len(control_values) + len(base.control_values)],
            work_wires=work_wires + base.work_wires,
            work_wire_type=work_wire_type,
        ): 1
    }


@register_resources(_ctrl_mcx_to_mcx_resources, name="to_mcx")
def _ctrl_mcx_to_mcx(base, control_wires, control_values, work_wires, work_wire_type):
    n_ctrl_wires = len(control_wires)
    work_wire_type = resolve_work_wire_type(
        base.work_wires,
        base.work_wire_type,
        work_wires,
        work_wire_type,
    )
    MultiControlledX(
        concatenate_wires(control_wires, base.wires),
        control_values=_resolve_ctrl_values(control_values, base.control_values, n_ctrl_wires),
        work_wires=concatenate_wires(work_wires, base.work_wires),
        work_wire_type=work_wire_type,
    )


add_decomps("C(MultiControlledX)", _ctrl_mcx_to_mcx)


def _crx_to_rz_ry_resources(phi, wires):
    return {ops.RZ: 2, ops.RY: 2, CNOT: 2}


@register_resources(_crx_to_rz_ry_resources, name="to_rz_ry")
def _crx_to_rz_ry(phi: TensorLike, wires: WiresLike):
    ops.RZ(np.pi / 2, wires=wires[1])
    ops.RY(phi / 2, wires=wires[1])
    CNOT(wires=wires)
    ops.RY(-phi / 2, wires=wires[1])
    CNOT(wires=wires)
    ops.RZ(-np.pi / 2, wires=wires[1])


def _crx_to_rx_cz_resources(phi, wires):
    return {ops.RX: 2, CZ: 2}


@register_resources(_crx_to_rx_cz_resources, name="to_rx_cz")
def _crx_to_rx_cz(phi: TensorLike, wires: WiresLike):
    ops.RX(phi / 2, wires=wires[1])
    CZ(wires=wires)
    ops.RX(-phi / 2, wires=wires[1])
    CZ(wires=wires)


def _crx_to_h_crz_resources(phi, wires):
    return {ops.Hadamard: 2, CRZ: 1}


@register_resources(_crx_to_h_crz_resources, name="to_h_crz")
def _crx_to_h_crz(phi: TensorLike, wires: WiresLike):
    ops.Hadamard(wires=wires[1])
    CRZ(phi, wires=wires)
    ops.Hadamard(wires=wires[1])


def _crx_to_ppr_resources(phi, wires):
    return {
        ops.PauliRot(Float, pauli_word="ZX", wires=Wire[2]): 1,
        ops.PauliRot(Float, pauli_word="X", wires=Wire[1]): 1,
    }


@register_resources(_crx_to_ppr_resources, name="to_paulirot")
def _crx_to_ppr(phi: TensorLike, wires: WiresLike):
    ops.PauliRot(phi / 2, "X", wires=wires[1])
    ops.PauliRot(-phi / 2, "ZX", wires=wires)


add_decomps(CRX, _crx_to_rx_cz, _crx_to_rz_ry, _crx_to_h_crz, _crx_to_ppr)

add_decomps("Adjoint(CRX)", adjoint_rotation2)

add_decomps("Pow(CRX)", pow_rotation2)


def _cry_to_ry_cnot_resources(phi, wires):
    return {ops.RY: 2, CNOT: 2}


@register_resources(_cry_to_ry_cnot_resources, name="to_ry_cnot")
def _cry_to_ry_cnot(phi: TensorLike, wires: WiresLike):
    ops.RY(phi / 2, wires=wires[1])
    CNOT(wires=wires)
    ops.RY(-phi / 2, wires=wires[1])
    CNOT(wires=wires)


def _cry_to_ppr_resources(phi, wires):
    return {
        ops.PauliRot(Float, pauli_word="ZY", wires=Wire[2]): 1,
        ops.PauliRot(Float, pauli_word="Y", wires=Wire[1]): 1,
    }


@register_resources(_cry_to_ppr_resources, name="to_paulirot")
def _cry_to_ppr(phi: TensorLike, wires: WiresLike):
    ops.PauliRot(phi / 2, "Y", wires=wires[1])
    ops.PauliRot(-phi / 2, "ZY", wires=wires)


add_decomps(CRY, _cry_to_ry_cnot, _cry_to_ppr)

add_decomps("Adjoint(CRY)", adjoint_rotation2)

add_decomps("Pow(CRY)", pow_rotation2)


def _crz_to_rz_cnot_resources(phi, wires):
    return {ops.RZ: 2, CNOT: 2}


@register_resources(_crz_to_rz_cnot_resources, name="to_rz_cnot")
def _crz_to_rz_cnot(phi: TensorLike, wires: WiresLike):
    ops.RZ(phi / 2, wires=wires[1])
    CNOT(wires=wires)
    ops.RZ(-phi / 2, wires=wires[1])
    CNOT(wires=wires)


def _crz_to_ppr_resources(phi, wires):
    return {
        ops.PauliRot(Float, pauli_word="ZZ", wires=Wire[2]): 1,
        ops.PauliRot(Float, pauli_word="Z", wires=Wire[1]): 1,
    }


@register_resources(_crz_to_ppr_resources, name="to_paulirot")
def _crz_to_ppr(phi: TensorLike, wires: WiresLike):
    ops.PauliRot(phi / 2, "Z", wires=wires[1])
    ops.PauliRot(-phi / 2, "ZZ", wires=wires)


add_decomps(CRZ, _crz_to_rz_cnot, _crz_to_ppr)

add_decomps("Adjoint(CRZ)", adjoint_rotation2)

add_decomps("Pow(CRZ)", pow_rotation2)


def _crot_to_rot_cnot_resources(*_, **__):
    return {ops.RZ: 3, CNOT: 2, ops.RY: 2}


@register_resources(_crot_to_rot_cnot_resources, name="to_rot_cnot")
def _crot_to_rot_cnot(phi: TensorLike, theta: TensorLike, omega: TensorLike, wires: WiresLike):
    ops.RZ((phi - omega) / 2, wires=wires[1])
    CNOT(wires=wires)
    ops.RZ(-(phi + omega) / 2, wires=wires[1])
    ops.RY(-theta / 2, wires=wires[1])
    CNOT(wires=wires)
    ops.RY(theta / 2, wires=wires[1])
    ops.RZ(omega, wires=wires[1])


add_decomps(CRot, _crot_to_rot_cnot)


@register_resources({CRot: 1}, name="reverse_angles")
def _adjoint_crot_standard(base):
    CRot(-base.omega, -base.theta, -base.phi, wires=base.wires)


add_decomps("Adjoint(CRot)", _adjoint_crot_standard)


def _cphase_to_rz_cnot_resources(phi, wires):
    return {ops.RZ: 3, CNOT: 2, GlobalPhase: 1}


@register_resources(_cphase_to_rz_cnot_resources, name="to_rz_cnot")
def _cphase_to_rz_cnot(phi: TensorLike, wires: WiresLike, **__):
    ops.RZ(phi / 2, wires=wires[0])
    CNOT(wires=wires)
    ops.RZ(-phi / 2, wires=wires[1])
    CNOT(wires=wires)
    ops.RZ(phi / 2, wires=wires[1])
    GlobalPhase(-phi / 4)


def _cphase_to_ppr_resources(phi, wires):
    return {
        GlobalPhase: 1,
        ops.PauliRot(Float, pauli_word="Z", wires=Wire[1]): 2,
        ops.PauliRot(Float, pauli_word="ZZ", wires=Wire[2]): 1,
    }


@register_resources(_cphase_to_ppr_resources, name="to_paulirot")
def _cphase_to_ppr(phi: TensorLike, wires: WiresLike, **__):
    ops.PauliRot(-phi / 2, pauli_word="ZZ", wires=wires)
    ops.PauliRot(phi / 2, pauli_word="Z", wires=wires[1])
    ops.PauliRot(phi / 2, pauli_word="Z", wires=wires[0])
    GlobalPhase(-phi / 4)


def _cphase_to_single_phase_resources(phi, wires):
    elbow = templates.TemporaryAND(wires=Wire[3])
    return {_change_op_basis_abstract(elbow, ops.PhaseShift, ops.adjoint(elbow)): 1}


@register_resources(
    _cphase_to_single_phase_resources, work_wires={"zeroed": 1}, name="to_single_phase"
)
def _cphase_to_single_phase(phi: TensorLike, wires: WiresLike, **__):
    with allocate(1, state="zero", restored=True) as work_wire:
        elbow = templates.TemporaryAND([wires[0], wires[1], work_wire[0]])
        ops.change_op_basis(elbow, ops.PhaseShift(phi, work_wire))


add_decomps(ControlledPhaseShift, _cphase_to_rz_cnot, _cphase_to_ppr, _cphase_to_single_phase)

add_decomps("Adjoint(ControlledPhaseShift)", adjoint_rotation2)

add_decomps("Pow(ControlledPhaseShift)", pow_rotation2)


def _crz_phase_gradient_compute_resources(bitstrings, wires, angle_wires, phase_grad_wires):
    """Resources for loading a controlled angle and flipping the phase-gradient register."""
    angle_fanout = ops.ctrl(templates.MultiX(bitstrings, angle_wires), control=wires[:1])
    phase_fanout = ops.ctrl(
        templates.MultiX(Bool[len(phase_grad_wires)], phase_grad_wires), control=wires[1:]
    )
    resources = {angle_fanout: 1}
    resources[phase_fanout] = resources.get(phase_fanout, 0) + 1
    return resources


@register_resources(_crz_phase_gradient_compute_resources)
def _CRZPhaseGradientCompute(bitstrings, wires, angle_wires, phase_grad_wires) -> None:
    """Load a controlled angle and flip phase-gradient wires when the target is zero."""
    ops.ctrl(templates.MultiX(bitstrings, angle_wires), control=wires[0])
    ops.ctrl(
        templates.MultiX([1] * len(phase_grad_wires), phase_grad_wires),
        control=wires[1],
        control_values=[0],
    )


def _crz_phase_gradient(phi, wires, angle_wires, phase_grad_wires, work_wires):
    """Apply the phase gradient circuit for CRZ. The precision is implicitly defined by the
    length of ``angle_wires``."""
    binary_int = math.cast(math.binary_decimals(phi, len(angle_wires), unit=4 * np.pi), bool)
    compute_op = _CRZPhaseGradientCompute(binary_int, wires, angle_wires, phase_grad_wires)
    target_op = templates.SemiAdder(angle_wires, phase_grad_wires, work_wires=work_wires)
    ops.change_op_basis(compute_op, target_op, compute_op)


def make_crz_to_phase_gradient_decomp(
    angle_wires, phase_grad_wires, work_wires, precision_stats=None
):
    r"""
    Create a custom decomposition rule for :class:`~.CRZ` gates.

    Here, the necessary auxiliary wires for the `phase gradient decomposition of CRZ
    <https://pennylane.ai/compilation/phase-gradient/c-control-rotations>`__ are taken as inputs
    and bound into the created decomposition rule.
    This way, this function can be used in a workflow context that explicitly uses those wires
    to generate this decomposition rule, which can then be registered with
    :func:`~pennylane.decomposition.add_decomps`.

    Parameters:
        angle_wires (Wires): wires that encode the binary representation of the rotation angle
        phase_grad_wires (Wires): wires that carry a phase gradient state. Should have the same
            length as ``angle_wires``.
        work_wires (Wires): additional work wires for :class:`~.SemiAdder` decomposition.
            At least ``len(angle_wires)-1`` work wires are required.
        precision_stats (dict[int, int] | None): Optional statistics of the number of bits
            required by the rotation angles. Defaults to ``None``.

    Returns:
        func: decomposition rule to be used within :func:`~.pennylane.decompose`.

    .. seealso:: :func:`~.make_rz_to_phase_gradient_decomp`, :func:`~.make_selectpaulirot_to_phase_gradient_decomp`

    **Example**

    In this example we decompose a circuit containing only a single :class:`~.CRZ`
    gate using the custom decomposition rule that we generate from within the context of the
    example, where all auxiliary wires exist.

    .. code-block:: python

        import pennylane as qp
        from pennylane.decomposition_rules.op_math.controlled_ops import (
            make_crz_to_phase_gradient_decomp,
        )
        import numpy as np

        prec = 3
        phi = (1/2 + 1/4 + 1/8) * 4 * np.pi # binary rep is (111)

        angle_wires = qp.wires.Wires([f"aux_{i}" for i in range(prec)])
        phase_grad_wires = qp.wires.Wires([f"phg_{i}" for i in range(prec)])
        work_wires = qp.wires.Wires([f"work_{i}" for i in range(prec - 1)])

        custom_decomp = make_crz_to_phase_gradient_decomp(
            angle_wires, phase_grad_wires, work_wires
        )

        @qp.transforms._tape_decompose(
            gate_set={"CNOT", "SemiAdder", "PauliX"},
            fixed_decomps={qp.CRZ: custom_decomp}
        )
        @qp.qnode(qp.device("null.qubit"))
        def circuit():
            qp.CRZ(phi, [0, 1])
            return qp.state()

    The resulting circuit corresponds to the phase gradient decomposition of ``CRZ``. The compute
    and uncompute fanouts load the binary representation of the angle (111 in this case) onto the
    ``angle_wires`` with a :class:`~.MultiX` controlled by the control wire (rendering as ``CNOT``,
    since the control is always on |1>), and flip all of the ``phase_grad_wires`` with a single
    :class:`~.MultiX` controlled by the target wire on |0> (rendering as a ``PauliX`` flip of that
    control wire, ``CNOT``\\ s to the ``phase_grad_wires``, and a ``PauliX`` flip back), and
    enclose the :class:`~.SemiAdder`.

    >>> wire_order = [0, 1] + angle_wires + phase_grad_wires + work_wires
    >>> print(qp.draw(circuit, wire_order=wire_order)())
         0: ─╭●─╭●─╭●────────────────────────╭●─╭●─╭●────────────────┤ ╭State
         1: ─│──│──│───X─╭●─╭●─╭●──X─────────│──│──│───X─╭●─╭●─╭●──X─┤ ├State
     aux_0: ─╰X─│──│─────│──│──│──╭SemiAdder─╰X─│──│─────│──│──│─────┤ ├State
     aux_1: ────╰X─│─────│──│──│──├SemiAdder────╰X─│─────│──│──│─────┤ ├State
     aux_2: ───────╰X────│──│──│──├SemiAdder───────╰X────│──│──│─────┤ ├State
     phg_0: ─────────────╰X─│──│──├SemiAdder─────────────╰X─│──│─────┤ ├State
     phg_1: ────────────────╰X─│──├SemiAdder────────────────╰X─│─────┤ ├State
     phg_2: ───────────────────╰X─├SemiAdder───────────────────╰X────┤ ├State
    work_0: ──────────────────────├SemiAdder─────────────────────────┤ ├State
    work_1: ──────────────────────╰SemiAdder─────────────────────────┤ ╰State

    .. details::
        :title: Usage details

        **Adaptive precision via ``precision_stats``**

        For angles that do not require the full precision determined by the angle encoding and
        phase gradient wires, we may use smaller fanouts and adder, reducing the cost of the
        phase gradient rotation. However, as rotation angles are not known statically in general,
        this adaptive precision is only supported if additional information about the distribution
        of rotation angles is provided via ``precision_stats``, a dictionary mapping a number
        of bits to the number of rotations requiring that many bits.

        The bits required by an angle :math:`\phi` are the
        position of the last set bit in ``qp.math.binary_decimals(phi, len(angle_wires),
        unit=4 * np.pi)``, i.e., the precision without trailing zeros. If provided,
        the fanouts and the :class:`~.SemiAdder` are truncated to the smallest number of
        bits in ``precision_stats`` that suffices for a given angle, using :func:`~.cond`
        branches that are hinted with the corresponding probabilities via :func:`~.hint`.
        If no number of bits in ``precision_stats`` suffices, the full precision is used.
        Keys must lie between ``0`` and ``len(angle_wires)``. Zero counts are ignored, and
        a dictionary without non-zero counts is treated like ``None``. Use
        :func:`~.phase_gradient_precision_stats` to compute the statistics from the angles.

        **Example**

        .. code-block:: python

            from pennylane.decomposition_rules.qubit.parametric_ops_single_qubit import (
                phase_gradient_precision_stats,
            )

            angles = np.array([1/4, 1/2 + 1/8]) * 4 * np.pi # (010) and (101)
            precision_stats = phase_gradient_precision_stats(angles, prec, qp.CRZ)

            truncating_decomp = make_crz_to_phase_gradient_decomp(
                angle_wires, phase_grad_wires, work_wires, precision_stats=precision_stats
            )

            @qp.transforms._tape_decompose(
                gate_set={"CNOT", "SemiAdder", "PauliX"},
                fixed_decomps={qp.CRZ: truncating_decomp}
            )
            @qp.qnode(qp.device("null.qubit"))
            def truncated_circuit(phi):
                qp.CRZ(phi, [0, 1])
                return qp.state()

        The rotation by the first angle only uses two angle wires and phase gradient wires:

        >>> precision_stats
        {2: 1, 3: 1}
        >>> print(qp.draw(truncated_circuit, wire_order=wire_order)(angles[0]))
             0: ─╭●─────────────────────╭●─────────────┤ ╭State
             1: ─│───X─╭●─╭●──X─────────│───X─╭●─╭●──X─┤ ├State
         aux_0: ─│─────│──│──╭SemiAdder─│─────│──│─────┤ ├State
         aux_1: ─╰X────│──│──├SemiAdder─╰X────│──│─────┤ ├State
         phg_0: ───────╰X─│──├SemiAdder───────╰X─│─────┤ ├State
         phg_1: ──────────╰X─├SemiAdder──────────╰X────┤ ├State
        work_0: ─────────────├SemiAdder────────────────┤ ├State
        work_1: ─────────────╰SemiAdder────────────────┤ ╰State

        Finally, keep the following in mind when using ``precision_stats``:

        * The resource function of the rule adds up the cost of all supported precisions, so
          that the graph-based decomposition system accounts for the operators in all
          conditional branches. This over-reports the cost of the rule, so that the
          decomposition graph may prefer another rule over this one when it is registered with
          :func:`~pennylane.decomposition.add_decomps`. Use ``fixed_decomps`` to enforce this rule.
        * Angles that require more bits than the largest key in ``precision_stats`` fall back
          to the full precision. Unless ``len(angle_wires)`` is a key in ``precision_stats``,
          this fallback branch is hinted with probability ``0``, so that resource estimates
          based on the hints do not account for these angles.

    """
    angle_wires, phase_grad_wires, work_wires = validate_phase_gradient_wires(
        angle_wires, phase_grad_wires, work_wires
    )
    precisions, dispatch = _precision_dispatcher(precision_stats, len(angle_wires), "CRZ")

    def _crz_phase_gradient_standard_resources(phi, wires):  # pylint: disable=unused-argument
        # Cost from the wires in the outer scope, counting each non-zero precision that may be
        # applied once. The angle-load fanout emits one gate per *set* bit of the (concrete) angle,
        # so this is an upper bound (exact=False below).
        resources = {}
        for p in precisions - {0}:
            target_op = templates.SemiAdder(Wire[p], Wire[p], Wire[len(work_wires)])
            compute_op = _CRZPhaseGradientCompute(Bool[p], Wire[2], Wire[p], Wire[p])
            resources[ChangeOpBasis2(compute_op, target_op, compute_op)] = 1
        return resources

    @register_resources(_crz_phase_gradient_standard_resources, exact=False, name="phase_gradient")
    def _crz_phase_gradient_standard(phi, wires):
        dispatch(
            phi,
            lambda p: _crz_phase_gradient(
                phi, wires, angle_wires[:p], phase_grad_wires[:p], work_wires
            ),
        )

    return _crz_phase_gradient_standard


def register_phase_gradient_compute():
    """Register the helper operator at its original point in built-in initialization."""
    if isinstance(_CRZPhaseGradientCompute, type):
        return
    globals()["_CRZPhaseGradientCompute"] = subcircuit(
        dynamic_argnames=("bitstrings",),
        wire_argnames=("wires", "angle_wires", "phase_grad_wires"),
        arg_specs={
            "bitstrings": Bool[-1],
            "wires": Wire[2],
            "angle_wires": Wire[-1],
            "phase_grad_wires": Wire[-1],
        },
        grad_method=None,
    )(_CRZPhaseGradientCompute)
