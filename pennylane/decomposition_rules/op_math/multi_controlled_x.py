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

"""Decomposition rules for multi controlled x."""


import numpy as np

from pennylane import allocation, capture, compiler, control_flow, math, ops, templates
from pennylane.decomposition import add_decomps, register_condition, register_resources
from pennylane.decomposition.decomposition_rule import (
    DecompCollection,
    _list_decomps,
    get_fixed_decomp,
)
from pennylane.decomposition.resources import resolve_work_wire_type
from pennylane.decomposition.symbolic_decomposition import self_adjoint
from pennylane.ops.op_math.pow2 import pow_involutory
from pennylane.ops.op_math.controlled_ops import MultiControlledX
from pennylane.typing import Bool, Complex, Float, Wire
from pennylane.wires import concatenate_wires
from .controlled_qubit_unitary import _ctrl_decomp_bisect_md


def wrap_mcx_rule_w_alloc(base_rule, num_work_wires, work_wire_type, name=""):
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

    if compiler.active() or capture.enabled():
        wires = math.array(wires, like="jax")
        control_values = math.array(control_values, like="jax")

    # Case 1: Decompose to single CNOT
    if len(wires) == 2:
        ops.CNOT(wires=wires)
        ops.cond(math.logical_not(control_values[0]), ops.X)(wires[1])
        return

    @control_flow.for_loop(0, len(wires) - 1)
    def _x_flips(i):
        ops.cond(math.logical_not(control_values[i]), ops.X)(wires[i])

    _x_flips()  # pylint: disable=no-value-for-parameter
    ops.Toffoli(wires=wires)
    _x_flips()  # pylint: disable=no-value-for-parameter


def _mcx_two_controls_elbows_resources(wires, *_, **__):
    num_ctrl_wires = len(wires) - 1
    return {templates.Elbow: 1, ops.CNOT: 1, ops.adjoint(templates.Elbow(Wire[3])): 1, ops.X: num_ctrl_wires}


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
    middle_toffoli = MultiControlledX(
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

    if compiler.active() or capture.enabled():
        wires = math.array(wires, like="jax")
        work_wires = math.array(work_wires, like="jax")
        control_values = math.array(control_values, like="jax")

    @control_flow.for_loop(len(control_values))
    def _x_flips(i):
        ops.cond(math.logical_not(control_values[i]), ops.X)(wires[i])

    target_wire, control_wires = wires[-1], wires[:-1]
    num_work_wires = len(control_wires) - 2
    extra_work_wires = work_wires[num_work_wires:]
    work_wires = work_wires[:num_work_wires]

    if work_wire_type == "borrowed":
        up_gate = down_gate = ops.Toffoli
    else:
        down_gate = templates.TemporaryAND
        up_gate = ops.adjoint(templates.TemporaryAND)

    @control_flow.for_loop(1, len(work_wires), 1)
    def loop_up(i):
        up_gate(wires=[control_wires[i], work_wires[i], work_wires[i - 1]])

    @control_flow.for_loop(num_work_wires - 1, 0, -1)
    def loop_down(i):
        down_gate(wires=[control_wires[i], work_wires[i], work_wires[i - 1]])

    _x_flips()  # pylint: disable=no-value-for-parameter

    if work_wire_type == "borrowed":
        ops.Toffoli(wires=[control_wires[0], work_wires[0], target_wire])
        loop_up()  # pylint: disable=no-value-for-parameter

    down_gate(wires=[control_wires[-1], control_wires[-2], work_wires[-1]])
    loop_down()  # pylint: disable=no-value-for-parameter

    _wires = [control_wires[0], work_wires[0], target_wire]
    MultiControlledX(
        _wires,
        work_wires=extra_work_wires,
        work_wire_type=work_wire_type,
    )

    loop_up()  # pylint: disable=no-value-for-parameter
    up_gate(wires=[control_wires[-1], control_wires[-2], work_wires[-1]])

    if work_wire_type == "borrowed":
        loop_down()  # pylint: disable=no-value-for-parameter

    _x_flips()  # pylint: disable=no-value-for-parameter


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

    if compiler.active() or capture.enabled():
        wires = math.array(wires, like="jax")
        control_values = math.array(control_values, like="jax")

    @control_flow.for_loop(len(control_values))
    def _x_flips(i):
        ops.cond(math.logical_not(control_values[i]), ops.X)(wires[i])

    _x_flips()  # pylint: disable=no-value-for-parameter

    # Unpack work wires for readability. There might just be one of them if it
    # is a "small" MCX (less than 6 controls)
    work0, *work1 = work_wires
    # First use the work wire to prepare the first two control wires as conditionally clean.
    left_elbow = ops.Toffoli if work_wire_type == "borrowed" else templates.TemporaryAND
    left_elbow([wires[0], wires[1], work0])

    middle_ctrl_indices = _build_log_n_depth_ccx_ladder(wires[:-1])

    # Apply the MCX in the middle. This is just a single Toffoli without work wires for "small" MCX
    if len(middle_ctrl_indices) == 1:
        ops.Toffoli([work0, wires[middle_ctrl_indices[0]], wires[-1]])
    else:
        middle_wires = [wires[i] for i in middle_ctrl_indices]
        # No toggle detection needed for the inner MCX decomposition, even for borrowed work wires
        _mcx_one_worker_no_flips(
            [work0] + middle_wires + [wires[-1]],
            work_wires=work1,
            work_wire_type=work_wire_type,
            _skip_toggle_detection=True,
        )

    # Uncompute the first ladder
    ops.adjoint(_build_log_n_depth_ccx_ladder)(wires[:-1])

    right_elbow = ops.Toffoli if work_wire_type == "borrowed" else ops.adjoint(templates.TemporaryAND)
    right_elbow([wires[0], wires[1], work0])

    if work_wire_type == "borrowed":
        # Perform toggle-detection if the work wire is borrowed
        middle_ctrl_indices = _build_log_n_depth_ccx_ladder(wires[:-1])
        if len(middle_ctrl_indices) == 1:
            ops.Toffoli([work0, wires[middle_ctrl_indices[0]], wires[-1]])
        else:
            middle_wires = [wires[i] for i in middle_ctrl_indices]
            _mcx_one_worker_no_flips(
                [work0] + middle_wires + [wires[-1]],
                work_wires=work1,
                work_wire_type=work_wire_type,
                _skip_toggle_detection=True,
            )

        ops.adjoint(_build_log_n_depth_ccx_ladder)(wires[:-1])

    _x_flips()  # pylint: disable=no-value-for-parameter


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


def _mcx_one_worker_no_flips(wires, work_wires, work_wire_type, _skip_toggle_detection=False):

    if work_wire_type == "zeroed":
        _skip_toggle_detection = True

    toffoli_type = templates.TemporaryAND if work_wire_type == "zeroed" else ops.Toffoli
    toffoli_type([wires[0], wires[1], work_wires[0]])

    final_ctrl_index = _build_linear_depth_ladder(wires[:-1])
    ops.Toffoli([work_wires[0], wires[final_ctrl_index], wires[-1]])
    ops.adjoint(_build_linear_depth_ladder)(wires[:-1])

    adj_toffoli_type = ops.adjoint(templates.TemporaryAND) if work_wire_type == "zeroed" else ops.Toffoli
    adj_toffoli_type([wires[0], wires[1], work_wires[0]])

    if not _skip_toggle_detection:
        # Perform toggle-detection unless skipped explicitly. By default, toggle detection
        # is skipped for `work_wire_type="zeroed"` but not for `work_wire_type="borrowed"`.
        _build_linear_depth_ladder(wires[:-1])
        ops.Toffoli([work_wires[0], wires[final_ctrl_index], wires[-1]])
        ops.adjoint(_build_linear_depth_ladder)(wires[:-1])


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

    if compiler.active() or capture.enabled():
        wires = math.array(wires, like="jax")
        control_values = math.array(control_values, like="jax")

    @control_flow.for_loop(len(control_values))
    def _x_flips(i):
        ops.cond(math.logical_not(control_values[i]), ops.X)(wires[i])

    _x_flips()  # pylint: disable=no-value-for-parameter
    _mcx_one_worker_no_flips(wires, work_wires, work_wire_type)
    _x_flips()  # pylint: disable=no-value-for-parameter


def _mcx_no_work_wires_resources(wires, *_, **__):
    num_control_wires = len(wires) - 1
    len_k1 = (num_control_wires + 1) // 2
    len_k2 = num_control_wires - len_k1
    mcx_k1 = MultiControlledX(Wire[len_k1 + 1], work_wires=Wire[len_k2])
    mcx_k2 = MultiControlledX(Wire[len_k2 + 1], work_wires=Wire[len_k1])
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

    U = ops.RX.compute_matrix(np.pi)

    if compiler.active() or capture.enabled():
        wires = math.array(wires, like="jax")
        control_values = math.array(control_values, like="jax")

    @control_flow.for_loop(len(control_values))
    def _x_flips(i):
        ops.cond(math.logical_not(control_values[i]), ops.X)(wires[i])

    _x_flips()  # pylint: disable=no-value-for-parameter
    _ctrl_decomp_bisect_md(U, wires)
    ops.ctrl(ops.GlobalPhase(-np.pi / 2), control=wires[:-1])
    _x_flips()  # pylint: disable=no-value-for-parameter


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

    loop()  # pylint: disable=no-value-for-parameter


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


@_list_decomps.register
def _list_mcx_decomps(op: MultiControlledX):
    # fixed_decomps should override everything
    if fixed_rule := get_fixed_decomp(op):
        return DecompCollection([fixed_rule])
    if not op.work_wires:
        return DecompCollection(_list_mcx_no_work_wire_decomps(op))
    if len(op.wires) == 2:
        return [mcx_to_cnot_or_toffoli]
    if len(op.wires) == 3:
        return [mcx_to_cnot_or_toffoli, mcx_two_controls_elbows]
    return [
        mcx_many_work_wires,
        mcx_two_work_wires,
        mcx_one_work_wire,
        mcx_no_work_wires,
    ]


def _list_mcx_no_work_wire_decomps(op: MultiControlledX):

    if len(op.wires) == 2:
        return [mcx_to_cnot_or_toffoli]

    if len(op.wires) == 3:
        elbow_rule = wrap_mcx_rule_w_alloc(
            mcx_two_controls_elbows, 1, "zeroed", name="two_controls_elbows_allocated"
        )
        return [mcx_to_cnot_or_toffoli, elbow_rule]

    capture_compatible_rules = [
        wrap_mcx_rule_w_alloc(mcx_two_work_wires, 2, "zeroed", "two_zeroed_work_wires"),
        wrap_mcx_rule_w_alloc(mcx_two_work_wires, 2, "borrowed", "two_borrowed_work_wires"),
        wrap_mcx_rule_w_alloc(mcx_one_work_wire, 1, "zeroed", "one_zeroed_work_wire"),
        wrap_mcx_rule_w_alloc(mcx_one_work_wire, 1, "borrowed", "one_borrowed_work_wire"),
        mcx_no_work_wires,
    ]
    if capture.enabled():
        return capture_compatible_rules

    # TODO: the following decomposition rules are not capture compatible [sc-129521]
    return [
        wrap_mcx_rule_w_alloc(
            mcx_many_work_wires,
            len(op.control_wires) - 2,
            "zeroed",
            "many_zeroed_work_wires",
        ),
        wrap_mcx_rule_w_alloc(
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
    wrap_mcx_rule_w_alloc(
        mcx_two_controls_elbows, 1, "zeroed", name="two_controls_elbows_allocated"
    ),
    wrap_mcx_rule_w_alloc(mcx_two_work_wires, 2, "zeroed", "two_zeroed_work_wires"),
    wrap_mcx_rule_w_alloc(mcx_two_work_wires, 2, "borrowed", "two_borrowed_work_wires"),
    wrap_mcx_rule_w_alloc(mcx_one_work_wire, 1, "zeroed", "one_zeroed_work_wire"),
    wrap_mcx_rule_w_alloc(mcx_one_work_wire, 1, "borrowed", "one_borrowed_work_wire"),
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
        MultiControlledX(
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
    MultiControlledX(
        concatenate_wires(control_wires, base.wires),
        control_values=_resolve_ctrl_values(control_values, base.control_values, n_ctrl_wires),
        work_wires=concatenate_wires(work_wires, base.work_wires),
        work_wire_type=work_wire_type,
    )


add_decomps("C(MultiControlledX)", ctrl_mcx_to_mcx)
