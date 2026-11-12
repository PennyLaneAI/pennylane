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

"""Built-in decomposition rules for templates.subroutines.select."""

from collections import Counter, defaultdict
from itertools import product

import numpy as np

from pennylane import capture, math
from pennylane.core.apply import apply
from pennylane.core.queuing import QueuingManager
from pennylane.decomposition import add_decomps, register_condition, register_resources
from pennylane.ops import CNOT, MultiControlledX, adjoint, ctrl
from pennylane.ops.op_math.adjoint2 import _adjoint_abstract
from pennylane.ops.op_math.controlled2 import _ctrl_abstract
from pennylane.templates.subroutines.arithmetic.temporary_and import TemporaryAND
from pennylane.templates.subroutines.select import Select, _add_k_units
from pennylane.typing import Bool, Wire


def _partial_select(K, control):
    r"""Compute the simplified control structure for a partial Select operator.

    Args:
        K (int): The number of operators in ``Select``.
        control (Sequence[hashable]): Control wires.

    Returns:
        list[list[tuple]]: a list of length ``K`` with each entry being a list of length two.
        The first entry in the ``j``\ th inner list is a tuple of control wires that control
        the application of the ``j``\ th operator in the partial ``Select``. The second entry
        in the ``j``\ th inner list is a tuple of control values for these control wires of
        the ``j``\ th operation.

    """
    c = len(control)
    # Here is the logic behind the function below:
    # For the j-th operation, the control values are given by the bit string of j.
    # The r-th control node will only be skipped if
    # a) its control value ``j_r`` is 0, and
    # b) flipping the control node from ``j_r=0`` to ``j_r=1`` yields a bitstring that corresponds
    #    to an integer ``k`` with ``k>=K``.
    # In the loop below, the ``if`` clause decides which nodes to _keep_, not skip, so that we
    # invert the condition "a) and b)" to "not a) or not b)", i.e. "``j_r=1`` or ``k<K``"
    # Finally, note that ``j_r = (j>>(c-1-r)) & 1`` and ``k=j+2**(c-1-r)``.
    controls = [
        [
            (control[r], j_c)
            for r in range(c)
            if (j_c := (j >> (c - 1 - r)) & 1) or j + 2 ** (c - 1 - r) < K
        ]
        for j in range(K)
    ]
    return (list(zip(*ctrl_, strict=True)) for ctrl_ in controls)


def _multi_controlled_rep(target_rep, num_control_wires, ctrl_state, num_work_wires):
    return _ctrl_abstract(
        target_rep,
        Wire[num_control_wires],
        Wire[num_work_wires],
        num_zero_control_values=num_control_wires - sum(ctrl_state),
    )


def _select_resources_multi_control(ops, control, work_wires, partial):
    num_control_wires = len(control)
    num_work_wires = len(work_wires)
    resources = defaultdict(int)
    if partial:
        if len(ops) == 1:
            resources[ops[0]] += 1
        else:
            # Use dummy control values, we will only care about the length of the outputs
            ctrls_and_ctrl_states = _partial_select(len(ops), list(range(num_control_wires)))
            for (ctrl_, ctrl_state), op in zip(ctrls_and_ctrl_states, ops, strict=True):
                resources[_multi_controlled_rep(op, len(ctrl_), ctrl_state, num_work_wires)] += 1
    else:
        state_iterator = product([0, 1], repeat=num_control_wires)

        # state_iterator may be longer than ops, but will not be shorter. So we use strict=False
        for state, op in zip(state_iterator, ops, strict=False):
            resources[_multi_controlled_rep(op, num_control_wires, state, num_work_wires)] += 1
    return dict(resources)


@register_resources(_select_resources_multi_control)
def _select_decomp_multi_control(*_, ops, control, work_wires, partial, **__):

    if partial:
        if len(ops) == 1:
            apply(ops[0])
        else:
            for (ctrl_, ctrl_state), op in zip(
                _partial_select(len(ops), control), ops, strict=True
            ):
                ctrl(op, ctrl_, control_values=ctrl_state, work_wires=work_wires)
    else:
        # ctrl_states may be longer than `ops`, but will never be shorter. So we use strict=False
        for ctrl_state, op in zip(product([0, 1], repeat=len(control)), ops, strict=False):
            ctrl(op, control, control_values=ctrl_state, work_wires=work_wires)


add_decomps(Select, _select_decomp_multi_control)


def _add_first_k_units(ops, controls, work_wires, k):
    """Add all controlled-applied operators within the unary iterator scheme.

    This function is used for the outer-most recursion level, and then calls _add_k_units
    for the inner recursion levels.
    In the documentation, this is the outer level that is being described in detail, and
    ``_add_k_units`` corresponds to the subroutine ``R``.

    """
    assert k == len(ops) > 2

    needed_controls = 2 * math.ceil_log2(k) - 1
    assert len(controls) >= needed_controls, f"{len(controls)=}, {needed_controls=}"
    controls = controls[:needed_controls]

    and_wires = controls[:3]
    new_controls = controls[2:]

    a = math.ceil_log2(k)  # a >= 2 because k>2 by assertion above
    k01 = 2 ** (a - 1)  # First half of circuit will implement 2^(a-1)>=2 operators
    k0 = k1 = 2 ** (a - 2)  # First two quarters of circuit each implement 2^(a-2)>=1 operator(s).
    l = k - k01
    k2 = int(math.ceil(2 ** (math.ceil_log2(l) - 1)))
    k3 = k - k01 - k2

    # Open TemporaryAND (controlled on |00>) + first quarter + CX (controlled on |0>) + second quarter
    first_half = (
        [TemporaryAND(and_wires, control_values=(0, 0))]
        + _add_k_units(ops[:k0], new_controls, work_wires, k0)
        + [MultiControlledX([controls[0], controls[2]], control_values=[0])]
        + _add_k_units(ops[k0:k01], new_controls, work_wires, k1)
    )

    if l == 1:  # first variant

        # Single operation left to apply: Only the third quarter will be needed, and it will not need
        # TemporaryAND gates at all
        and_wires_sec_half = []
        new_controls_sec_half = controls
        # Closing TemporaryAND for first half
        middle_part = [adjoint(TemporaryAND)(and_wires, control_values=(0, 1))]

    else:
        c_bar = 2 * (math.ceil_log2(k) - math.ceil_log2(k - k01) - 1)
        and_wires_sec_half = [controls[0], controls[c_bar + 1], controls[c_bar + 2]]
        new_controls_sec_half = controls[c_bar + 2 :]

        if c_bar > 0:  # second variant
            # Closing TemporaryAND for first half, opening TemporaryAND for second half
            middle_part = [
                adjoint(TemporaryAND)(and_wires, control_values=(0, 1)),
                TemporaryAND(and_wires_sec_half, control_values=(1, 0)),
            ]
        else:  # third variant
            middle_part = [CNOT(and_wires[::2]), CNOT(and_wires[1:])]

    second_half = _add_k_units(ops[k01 : k01 + k2], new_controls_sec_half, work_wires, k2)
    if and_wires_sec_half:
        second_half += (
            [CNOT(and_wires_sec_half[::2])]
            + _add_k_units(ops[k0 + k1 + k2 :], new_controls_sec_half, work_wires, k3)
            + [adjoint(TemporaryAND)(and_wires_sec_half)]
        )

    return first_half + middle_part + second_half


def _select_resources_unary_not_partial(ops, num_control_wires, num_work_wires):
    resources = defaultdict(int)
    c = num_control_wires
    K = len(ops)
    num_work_wires = num_work_wires - (c - 1)

    if c == 1:
        for i, op in enumerate(ops):
            resources[
                _ctrl_abstract(
                    op,
                    Wire[1],
                    Wire[num_work_wires],
                    num_zero_control_values=(1 - i),
                )
            ] += 1
        return dict(resources)

    # The number of elbows required for non-partial unary iteration is given by
    # N(c, K) = c + K - 2 - ‖K-1‖_H - int(K>2^{c-1}),
    # where ‖.‖_H denotes the Hamming weight, or bit count.
    # To see this, note that adding a control node to a given unary iteration is done by using the
    # given iteration, and replacing each "slot" (controlled unitary) by a construction that
    # yields two new "slots" and requires one elbow. Consequently, the addition of a control
    # node uses the given iteration with ⌈K/2⌉ slots, and ⌈K/2⌉ additional elbows, leading to the
    # recursion relation
    # N(c+1, K) = N(c, ⌈K/2⌉) + ⌈K/2⌉
    # In addition, we know that for two control nodes, just a single elbow is required:
    # N(2, K) = 1
    # The formula at the top is the solution to this recursion relation. An alternative expression
    # for the same is
    # N(c,K)=1+∑_{j=1}^{c−2} ⌈K⋅2^{−j}⌉
    num_elbows = c + K - 2 - (K - 1).bit_count() - int(K > 2 ** (c - 1))

    resources[TemporaryAND] += num_elbows
    resources[_adjoint_abstract(TemporaryAND)] += num_elbows
    more_than_a_quarter = int(K > 2 ** (c - 2))
    more_than_a_half = int(K > 2 ** (c - 1))
    resources[CNOT] += K - 1 + more_than_a_half - more_than_a_quarter
    resources[MultiControlledX(Wire[2], control_values=Bool[1])] += more_than_a_quarter
    for op in ops:
        resources[_ctrl_abstract(op, Wire[1], Wire[num_work_wires])] += 1

    return dict(resources)


# pylint: disable=unused-argument
def _select_resources_unary(ops, control, work_wires, partial):
    num_control_wires = len(control)
    num_work_wires = len(work_wires)
    num_ops = len(ops)
    if num_ops == 0:
        return {}
    if not partial:
        return _select_resources_unary_not_partial(ops, num_control_wires, num_work_wires)

    if num_ops == 1:
        return {ops[0]: 1}
    counts = Counter()

    if num_ops == 2:
        for i, op in enumerate(ops):
            counts[
                _ctrl_abstract(
                    op,
                    Wire[1],
                    Wire[num_work_wires],
                    num_zero_control_values=1 - i,
                )
            ] += 1
        return dict(counts)
    if num_ops / 2 ** math.ceil_log2(num_ops) > 3 / 4:
        counts.update(
            {
                TemporaryAND: num_ops - 3,
                _adjoint_abstract(TemporaryAND): num_ops - 3,
                CNOT: num_ops - 1,
                MultiControlledX(Wire[2], control_values=Bool[1]): 1,
            }
        )
    else:
        counts.update(
            {
                TemporaryAND: num_ops - 2,
                _adjoint_abstract(TemporaryAND): num_ops - 2,
                CNOT: num_ops - 3,
                MultiControlledX(Wire[2], control_values=Bool[1]): 1,
            }
        )

    # Due to partial=True, we are allowed to restrict to a subset of the control wires
    unary_control_wires = max(math.ceil_log2(num_ops) - 1, 0)
    num_work_wires = num_work_wires - unary_control_wires
    for op in ops:
        counts[_ctrl_abstract(op, Wire[1], Wire[num_work_wires])] += 1

    return dict(counts)


def _select_decomp_unary_not_partial(ops, control, work_wires):
    """Decompose Select operator into unary iterator, without applying the partial Select
    reduction. The control structure is simpler without the Select reduction, so that we do
    not use the same recursive structure as for ``_select_decomp_unary`` but a simple ``for`` loop
    instead.

    Args:
        ops (Sequence[Operator]): Operators applied by the Select unary iterator.
        control (Sequence[hashable]): Control wires. Should be at least ``ceil(log2(len(ops)))`` many.
        work_wires (Sequence[hashable]): Work wires. Should be at least ``len(control)-1`` many.

    Returns:
        Sequence[Operator]: Decomposition of a non-partial ``Select`` using unary iteration.

    Denote the number of control qubits as ``c`` and the number of operators as ``K``.
    Arrange the control wires and ``c-1`` work wires as ``["c0", "c1", "w0", "c2", "w1", ...]``.

    We begin with a ladder of ``TemporaryAND`` operators:

    ```
    c0: ─╭○──────────
    c1: ─├○──────────
    w0: ─╰──╭●───────
    c2: ────├○───────
    w1: ────╰──╭●────
     :         :
     :            :
    wp: ──────────╰──
    t1: ─────────────
     :
    tn: ─────────────
    ```

    Here, the ``p`` in ``wp`` is the number of work wires, and ``n`` is the number of target
    wires.
    Then we iterate over the target operators and perform the following steps for each, except
    for the last operator.

    1. Apply the operator, controlled on the last wire ``"wp"`` of the control structure, to the
       target wires ``["t1", ... "tn"]``.
    2. For the ``k``-th operator (in 0-based indexing), find the position ``a`` of the most
       significant bit that flips when incrementing ``k`` to ``k+1``. It will be the position of
       the last bit that is ``0`` in the bit string of ``k``.
    3. Apply right elbows, starting from the lower end of the control structure up to ``a`` (exclusive).
    4. Apply gates that result from merging a right with the next left elbow. This is always a
       ``CNOT`` on the first and last gate of the elbows, but might be complemented with
       ``X`` gates or a second ``CNOT`` gate if ``a=1`` or ``a=0``, respectively.
    5. Apply left elbows corresponding to the right elbows applied in step 3, but starting at
       position ``a`` (exclusive).

    As an example, let ``c=4`` and the operator ``SWAP(["t1", "t2"])`` be the ``k=3``-rd operator
    that is applied, the following subcircuit is appended (using that ``k=0011_2`` and
    ``k+1=0100_2``, so that ``a=1`` is the position of the most significant flipped bit)

    ```
    c0: ─────────────╭●──────────
    c1: ─────────────│───────────
    w0: ──────────●╮─╰X──╭●──────
    c2: ──────────●┤─────├○──────
    w1: ───────●╮──╯─────╰──╭●───
    c3: ───────●┤───────────├○───
    w2: ─╭●─────╯───────────╰────
    t1: ─├SWAP───────────────────
    t2: ─╰SWAP───────────────────
         1.      3.  4.   5.
    ```

    To conclude, we control-apply the last target operator, which was excluded from the loop above,
    and apply a ladder of right elbows across the full control structure, starting at the
    low end:

    ```
    c0: ──────────●╮── *
    c1: ──────────○┤── *
    w0: ───────●╮──╯──
    c2: ───────●┤───── *
    w1: ────●╮──╯─────
     :       :
     :    :
    wp: ──╯───────────
    t1: ──────────────
     :
    tn: ──────────────
    ```

    The value of the control node (filled or open) on the control wires (marked with an
    asterisk (*), i.e. without the work wires) depends on the total number ``K`` of operators
    and can be computed as

    ```
    control_values = np.binary_repr(K-1, width=c)
    ```
    """

    c = len(control)
    K = len(ops)
    if c == 1:
        # Don't need unary iterator, just control-apply the one/two operator(s) directly.
        new_ops = [
            ctrl(op, control=control, control_values=[i], work_wires=work_wires)
            for i, op in enumerate(ops)
        ]
        return new_ops

    # Validate work wires
    p = len(work_wires)
    if p < c - 1:
        raise ValueError(
            f"Can't use this decomposition with less than {c - 1} work wires for {c} controls. "
            f"Got {p} work wires: {work_wires}."
        )

    unary_work_wires = work_wires[: c - 1]
    new_work_wires = work_wires[c - 1 :]
    aux_control = [control[0]]
    for ctrl_wire, work_wire in zip(control[1:], unary_work_wires, strict=True):
        aux_control.append(ctrl_wire)
        aux_control.append(work_wire)
    # Create triples of wires to which elbows are applied
    unary_triples = [aux_control[2 * i : 2 * i + 3] for i in range(c - 1)]

    # Apply initial ladder of left elbows
    ops_decomp = [
        TemporaryAND(triple, control_values=((0, 0) if i == 0 else (1, 0)))
        for i, triple in enumerate(unary_triples)
    ]

    first_bit_has_flipped = False
    for k, op in enumerate(ops[:-1]):
        # For all but the last target operator, do the following:
        # 1. apply target operator, always controlled on last unary iteration wire
        ops_decomp.append(ctrl(op, control=aux_control[-1], work_wires=new_work_wires))

        # 2. find the most significant bit ``a`` that flips when incrementing from k to k+1
        first_flip_bit = c - 1 - list(np.binary_repr(k, width=c)[::-1]).index("0")

        # 3. apply the ladder of right elbows up to the most significant flipped bit (exclusive)
        sub_triples = unary_triples[max(first_flip_bit, 1) : c - 1]
        ops_decomp.extend([adjoint(TemporaryAND(triple)) for triple in reversed(sub_triples)])

        # 4. apply gates that result from merging a right and left elbow (``inter_ops``)
        if first_flip_bit == 1:
            c0, c1, c2 = unary_triples[0]
            if first_bit_has_flipped:
                inter_ops = [CNOT([c0, c2])]
            else:
                inter_ops = [MultiControlledX([c0, c2], control_values=[0])]
        elif first_flip_bit == 0:
            c0, c1, c2 = unary_triples[0]
            inter_ops = [CNOT([c0, c2]), CNOT([c1, c2])]
            first_bit_has_flipped = True
        else:
            inter_ops = [CNOT(unary_triples[first_flip_bit - 1][::2])]
        ops_decomp.extend(inter_ops)

        # 5. apply the ladder of elbows starting at the most significant flipped bit (exclusive)
        ops_decomp.extend([TemporaryAND(triple, control_values=(1, 0)) for triple in sub_triples])

    # For the last target operator, apply controlled target op and then the "closing"
    # ladder of right elbows
    closing_ctrl_bits = list(map(int, np.binary_repr(K - 1, width=c)))
    ops_decomp.append(ctrl(ops[-1], control=aux_control[-1:], work_wires=new_work_wires))
    ops_decomp.extend(
        [
            adjoint(TemporaryAND(triple, control_values=(1, val)))
            for val, triple in zip(
                reversed(closing_ctrl_bits[2:]), reversed(unary_triples[1 : c - 1]), strict=True
            )
        ]
    )
    ops_decomp.append(adjoint(TemporaryAND(unary_triples[0], control_values=closing_ctrl_bits[:2])))
    return ops_decomp


# pylint: disable=unused-argument
def _unary_condition(ops, control, work_wires, partial):
    return len(work_wires) >= len(control) - 1


@register_condition(_unary_condition)
@register_resources(_select_resources_unary)
def _select_decomp_unary(*_, ops, control, work_wires, partial, **__):
    r"""This function reproduces the unary iterator behaviour in https://arxiv.org/abs/1805.03662.
    For :math:`K` operators this decomposition requires at least :math:`c=\lceil\log_2 K\rceil`
    control wires (as usual for Select), and :math:`c-1` additional work wires.
    See the documentation of ``Select`` for details.

    The ``partial`` argument controls whether the reduction to partial Select is performed,
    see the documentation of ``Select`` and https://pennylane.ai/compilation/partial-select for
    details.
    """

    K = len(ops)
    if K == 0:
        return []

    # Validate number of control wires
    c = len(control)
    min_num_controls = max(math.ceil_log2(K), 1)
    if c < min_num_controls:
        raise ValueError(
            f"At least {min_num_controls} control wires are required to implement Select of "
            f"{K} operators, but only {c} control wires were provided: {control}."
        )

    if not partial:
        return _select_decomp_unary_not_partial(ops, control, work_wires)

    # Due to partial=True, we are allowed to restrict to a subset of the control wires
    control = control[-min_num_controls:]
    if 1 <= K <= 2:
        if K == 1 and partial:
            # Can skip control for partial Select and a single op
            if QueuingManager.recording() or capture.enabled():
                apply(ops[0])
            return list(ops)
        # Don't need unary iterator, just control-apply the one/two operator(s) directly.
        new_ops = [
            ctrl(op, control=control[-1], control_values=[i], work_wires=work_wires)
            for i, op in enumerate(ops)
        ]
        return new_ops

    # Validate work wires
    p = len(work_wires)
    if p < c - 1:
        raise ValueError(
            f"Can't use this decomposition with less than {c - 1} work wires for {c} controls. "
            f"Got {p} work wires: {work_wires}."
        )

    # Arrange control and work wires into common register
    aux_control = [control[0]]
    for i in range(min_num_controls - 1):
        aux_control.append(control[i + 1])
        aux_control.append(work_wires[i])
    work_wires = work_wires[min_num_controls - 1 :]
    return _add_first_k_units(ops, aux_control, work_wires, len(ops))


add_decomps(Select, _select_decomp_unary)


def _select_multi_control_work_wire_resources(ops, control, work_wires, partial):
    if len(control) == 0:
        return dict(Counter(ops))

    num_control_wires = len(control)
    num_work_wires = len(work_wires)
    resources = defaultdict(int)

    if partial:
        if len(ops) == 1:
            resources[_multi_controlled_rep(ops[0], 1, [1], num_work_wires - 1)] += 1
            resources[
                MultiControlledX(
                    Wire[num_control_wires + 1],
                    control_values=Bool[num_control_wires],
                    work_wires=Wire[num_work_wires - 1],
                )
            ] += 2
        else:
            # Use dummy control values, we will only care about the length of the outputs
            ctrls_and_ctrl_states = _partial_select(len(ops), list(range(num_control_wires)))
            for (ctrl_, _), op in zip(ctrls_and_ctrl_states, ops, strict=True):
                resources[_multi_controlled_rep(op, 1, [1], num_work_wires - 1)] += 1
                resources[
                    MultiControlledX(
                        Wire[len(ctrl_) + 1],
                        control_values=Bool[len(ctrl_)],
                        work_wires=Wire[num_work_wires - 1],
                    )
                ] += 2
    else:
        state_iterator = product([0, 1], repeat=num_control_wires)

        # state_iterator may be longer than ops, but will not be shorter. So we use strict=False
        for _, op in zip(state_iterator, ops, strict=False):

            resources[_multi_controlled_rep(op, 1, [1], num_work_wires - 1)] += 1
            resources[
                MultiControlledX(
                    Wire[num_control_wires + 1],
                    control_values=Bool[num_control_wires],
                    work_wires=Wire[num_work_wires - 1],
                )
            ] += 2

    return dict(resources)


# pylint: disable=unused-argument
def _work_wire_condition(ops, control, work_wires, partial):
    return len(work_wires) >= 1


@register_condition(_work_wire_condition)
@register_resources(_select_multi_control_work_wire_resources)
def _select_decomp_multi_control_work_wire(*_, ops, control, work_wires, partial, **__):
    """
    Multi-controlled gate decomposition, in which, instead of directly controlling the target operator with all control
    wires, an auxiliary qubit is employed to encode whether the control condition is satisfied. The target operator
    is then applied as a single-qubit controlled gate from this auxiliary qubit.
    An example of this decomposition can be found in Figure 1(a):  https://arxiv.org/abs/1812.00954
    """

    if len(ops) == 0:
        return []

    if len(control) == 0:
        for op in ops:
            apply(op)
        return []

    if partial:
        if len(ops) == 1:
            MultiControlledX(
                [*control, work_wires[0]],
                control_values=[0] * len(control),
                work_wires=work_wires[1:],
            )
            ctrl(ops[0], control=work_wires[:1], work_wires=work_wires[1:])
            MultiControlledX(
                [*control, work_wires[0]],
                control_values=[0] * len(control),
                work_wires=work_wires[1:],
            )
            return []

        ctrls_and_ctrl_states = _partial_select(len(ops), control)
        for (ctrl_, ctrl_state), op in zip(ctrls_and_ctrl_states, ops, strict=True):
            mcx_wires = [*ctrl_, work_wires[0]]
            MultiControlledX(mcx_wires, control_values=ctrl_state, work_wires=work_wires[1:])
            ctrl(op, control=work_wires[:1], work_wires=work_wires[1:])
            MultiControlledX(mcx_wires, control_values=ctrl_state, work_wires=work_wires[1:])
        return []

    # ctrl_states may be longer than `ops`, but will never be shorter. So we use strict=False
    for ctrl_state, op in zip(product([0, 1], repeat=len(control)), ops, strict=False):
        mcx_wires = [*control, work_wires[0]]
        MultiControlledX(mcx_wires, control_values=ctrl_state, work_wires=work_wires[1:])
        ctrl(op, control=work_wires[:1], work_wires=work_wires[1:])
        MultiControlledX(mcx_wires, control_values=ctrl_state, work_wires=work_wires[1:])
    return []


add_decomps(Select, _select_decomp_multi_control_work_wire)
