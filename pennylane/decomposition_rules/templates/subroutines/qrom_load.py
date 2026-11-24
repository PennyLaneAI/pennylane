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


"""Built-in decomposition rules for templates.subroutines.qrom_load."""

from collections import defaultdict
from functools import partial
from pennylane import capture, compiler, math
from pennylane import ops as qp_ops
from pennylane.capture import hint
from pennylane.control_flow import for_loop
from pennylane.core.queuing import QueuingManager
from pennylane.decomposition import (
    add_decomps,
    register_condition,
    register_resources,
)
from pennylane.ops import CNOT, SWAP, H, X, cond, ctrl, pauli_measure
from pennylane.ops.mid_measure.pauli_measure import PauliMeasure
from pennylane.typing import Bool, Wire
from pennylane.wires import Wires
from pennylane.templates.subroutines.arithmetic import TemporaryAND
from pennylane.templates.subroutines.qrom import QROM
from pennylane.decomposition_rules.templates.subroutines.qrom import (
    _qrom_unary_iteration_resources,
    _unary_iteration_num_elbows,
)
from pennylane.templates.subroutines.qrom_load import QROMLoad


def _select_ops(
    bitstrings,
    depth,
    target_wires,
    swap_wires,
    select_control_wires,
    select_work_wires,
    zeroed_targets=False,
):  # pylint:disable=too-many-arguments
    num_targets = len(target_wires)
    num_bitstrings = bitstrings.shape[0]
    num_columns = -(-num_bitstrings // depth)
    num_missing = (-num_bitstrings) % depth

    if num_missing > 0:
        bitstrings = math.vstack([bitstrings, math.zeros((num_missing, num_targets), dtype=int)])

    num_targets_select = depth * num_targets
    new_bitstrings = bitstrings.reshape((num_columns, num_targets_select))
    wires = {
        "control_wires": select_control_wires,
        "target_wires": swap_wires[:num_targets_select],
        "work_wires": select_work_wires,
    }
    if zeroed_targets:
        QROMLoad(new_bitstrings, **wires)
    else:
        QROM(new_bitstrings, **wires)


def _swap_ops(swap_control_wires, swap_wires, target_wires, cswap_work_wires, adjoint=False):

    n_targets = len(target_wires)
    if capture.enabled() or compiler.active():
        swap_wires = math.array(swap_wires, like="jax").reshape((-1, n_targets))
        swap_control_wires = math.array(swap_control_wires, like="jax")
    else:
        # Need to work with nested list here in order to not coerce (deprecated) string wire labels
        # to object-dtyped np.array. This forces manual "reshape" here and indexing via [j][k]
        # instead of [j, k] below.
        num_columns = len(swap_wires) // n_targets
        swap_wires = [swap_wires[i * n_targets : (i + 1) * n_targets] for i in range(num_columns)]

    n_iters = len(swap_control_wires)
    outer_loop = for_loop(0, n_iters) if adjoint else for_loop(n_iters - 1, -1, -1)

    def outer(i):

        inner_loop = for_loop(0, 2**i) if adjoint else for_loop(2**i - 1, -1, -1)

        def inner(j):

            layer_loop = for_loop(n_targets - 1, -1, -1) if adjoint else for_loop(0, n_targets)

            def swap_layer(k):
                ctrl(
                    SWAP([swap_wires[j][k], swap_wires[j + 2**i][k]]),
                    control=swap_control_wires[-i - 1],
                    work_wires=cswap_work_wires,
                    work_wire_type="zeroed",
                )

            layer_loop(swap_layer)()

        # The average of 2**i over all i, so that resource estimates count all SWAPs exactly
        hint(inner_loop(inner), {"num-iters": (2**n_iters - 1) / n_iters})()

    outer_loop(outer)()


def _select_swap_cost(num_bitstrings, num_control_wires, num_targets, log_depth, clean=True):
    """Number of Toffoli-like gates, i.e., elbows of the Select component and controlled SWAPs,
    of the Select-SWAP decomposition loading ``2**log_depth`` bitstrings in parallel. For
    ``log_depth=0``, this is the cost of a single plain ``QROM``. With ``clean=False``, the work
    wires are not restored, as in :class:`~.SelectSwap`, so each component only runs once."""
    num_control_wires_select = num_control_wires - log_depth
    num_columns = -(-num_bitstrings // 2**log_depth)
    num_elbows = 0
    if num_control_wires_select >= 2:
        num_elbows = _unary_iteration_num_elbows(num_control_wires_select, num_columns)
    if log_depth == 0:
        # Plain QROM
        return num_elbows
    num_cswaps = num_targets * (2**log_depth - 1)
    if clean:
        return 2 * num_elbows + 4 * num_cswaps
    return num_elbows + num_cswaps


def _calculate_select_swap_sizes(
    num_bitstrings, num_control_wires, num_targets, num_work_wires, clean=True
):
    """Calculates the register sizes for the Select-SWAP decomposition.

    This utility function determines how many bitstrings to load in parallel, and how many
    auxiliary wires from the total pool should be allocated to the Select operation versus the
    SWAP network. Among all depths that fit into the work wires, the one with the lowest cost
    according to ``_select_swap_cost`` is chosen, preferring smaller depths in case of a tie.

    Args:
        num_bitstrings (int): number of bitstrings/entries in the data
        num_control_wires (int): number of control wires
        num_targets (int): number of target wires (bitstring length)
        num_work_wires (int): total number of available work wires
        clean (bool): whether the work wires are restored, which changes the cost of each depth

    Returns:
        tuple[int]: ``(num_control_wires_select, num_work_wires_select, num_work_wires_swap,
        num_work_wires_cswap, depth)`` — control wires of the Select component, work wires left
        for the Select component (of which it uses one fewer than its control wires), work wires
        assigned to the SWAP network, work wires borrowed from the Select component by the
        CSWAPs, and the number of bitstrings loaded in parallel.
    """

    if num_work_wires < num_control_wires - 1:
        return num_control_wires, num_work_wires, 0, 0, 1

    num_bitstrings = int(num_bitstrings)
    # Work wires for the SWAP network, and for the unary iteration of the Select component
    log_depths = [
        log_depth
        for log_depth in range(num_control_wires + 1)
        if num_targets * (2**log_depth - 1) + max(0, num_control_wires - log_depth - 1)
        <= num_work_wires
    ]
    log_depth = min(
        log_depths,
        key=lambda j: (
            _select_swap_cost(num_bitstrings, num_control_wires, num_targets, j, clean),
            j,
        ),
    )
    depth = 2**log_depth
    num_control_wires_select = num_control_wires - log_depth
    num_work_wires_swap = num_targets * (depth - 1)
    num_work_wires_select = num_work_wires - num_work_wires_swap
    num_work_wires_cswap = int(num_work_wires_select >= 1)

    return (
        num_control_wires_select,
        num_work_wires_select,
        num_work_wires_swap,
        num_work_wires_cswap,
        depth,
    )


def _select_swap_condition(bitstrings, control_wires, target_wires, work_wires):
    """We use Select-SWAP only when there are enough work wires for unary iteration on the
    nested select QROM and an actual SWAP network is used on top, i.e. for depth > 1. If the
    cost-optimal depth is 1, the plain QROM fallback is the better decomposition."""
    num_control_wires = len(control_wires)
    num_work_wires = len(work_wires)

    if num_control_wires == 0 or num_work_wires < num_control_wires - 1:
        return False

    *_, depth = _calculate_select_swap_sizes(
        len(bitstrings), num_control_wires, len(target_wires), num_work_wires
    )
    return depth > 1


def _select_swap_resources(bitstrings, control_wires, target_wires, work_wires):
    """Assumes depth computed below satisfies depth > 1, as guaranteed by _select_swap_condition."""
    num_bitstrings = len(bitstrings)
    num_control_wires = len(control_wires)
    num_targets = len(target_wires)
    num_work_wires = len(work_wires)

    num_control_wires_select, _, _, num_work_wires_cswap, depth = _calculate_select_swap_sizes(
        num_bitstrings, num_control_wires, num_targets, num_work_wires
    )
    num_work_wires_select = max(0, num_control_wires_select - 1)
    num_columns = -(-num_bitstrings // depth)
    # Select block (implemented as a nested QROM or QROMLoad over concatenated columns)
    num_targets_select = depth * num_targets
    qrom_rep = QROM(
        Bool[num_columns, num_targets_select],
        Wire[num_control_wires_select],
        Wire[num_targets_select],
        Wire[num_work_wires_select],
    )

    # Swap block
    num_control_wires_swap = num_control_wires - num_control_wires_select
    num_cswaps_per_block = num_targets * (2**num_control_wires_swap - 1)

    cswap_rep = ctrl(
        SWAP(Wire[2]),
        control=Wire[1],
        work_wires=Wire[num_work_wires_cswap],
        work_wire_type="zeroed",
    )
    return {qrom_rep: 2, cswap_rep: 4 * num_cswaps_per_block, H: 2 * num_targets}


@register_condition(_select_swap_condition)
@register_resources(_select_swap_resources, name="select_swap")
def qrom_load_select_swap(bitstrings, control_wires, target_wires, work_wires):
    num_control_wires_select, _, num_work_wires_swap, num_work_wires_cswap, depth = (
        _calculate_select_swap_sizes(
            len(bitstrings), len(control_wires), len(target_wires), len(work_wires)
        )
    )

    swap_work_wires = work_wires[:num_work_wires_swap]
    num_work_wires_select = max(0, num_control_wires_select - 1)
    select_work_wires = work_wires[
        num_work_wires_swap : num_work_wires_swap + num_work_wires_select
    ]
    # The Select block returns its work wires to |0>, so the CSWAPs can borrow one of them.
    cswap_work_wires = work_wires[num_work_wires_swap : num_work_wires_swap + num_work_wires_cswap]
    swap_wires = Wires(target_wires) + Wires(swap_work_wires)

    select_control_wires = control_wires[:num_control_wires_select]
    swap_control_wires = control_wires[num_control_wires_select:]

    if capture.enabled() or compiler.active():
        target_wires = math.array(target_wires, like="jax")

    @for_loop(2)
    def _select_swap_loop(_):

        @for_loop(len(target_wires))
        def apply_h(i):
            H(target_wires[i])

        apply_h()  # pylint: disable=no-value-for-parameter

        cswaps = partial(_swap_ops, swap_control_wires, swap_wires, target_wires, cswap_work_wires)

        cswaps(adjoint=True)
        _select_ops(
            bitstrings, depth, target_wires, swap_wires, select_control_wires, select_work_wires
        )
        cswaps()

    _select_swap_loop()  # pylint: disable=no-value-for-parameter


def _qrom_load_to_qrom_resources(bitstrings, control_wires, target_wires, work_wires):
    return {
        QROM(
            Bool[*bitstrings.shape],
            Wire[len(control_wires)],
            Wire[len(target_wires)],
            Wire[len(work_wires)],
        ): 1
    }


@register_resources(_qrom_load_to_qrom_resources, name="to_qrom")
def qrom_load_to_qrom(bitstrings, control_wires, target_wires, work_wires):
    QROM(bitstrings, control_wires, target_wires, work_wires)


def _adjoint_qrom_load_one_hot_size(num_bitstrings, num_control_wires, num_aux_wires):
    """Number of least-significant address bits ``j`` that are encoded into a one-hot register of
    ``k = 2**j`` auxiliary wires for the phase fixup of ``Adjoint(QROMLoad)``.

    The fixup ``QROM`` uses unary iteration over the remaining ``num_control_wires - j`` address
    bits with the remaining ``num_aux_wires - k`` auxiliary wires as work wires. Each work wire it
    lacks for a full iteration tree adds a control to every load, which costs far more than a
    larger one-hot register saves. Therefore, ``j`` first minimizes this shortfall for unary iteration
    and then the T-count, estimated as 4 per ``TemporaryAND`` and 7 per Toffoli in the loads.
    Without shortfall, this approximately minimizes the Toffoli cost ``ceil(L / k) + k`` from
    Appendix C of `arXiv:1902.02134 <https://arxiv.org/abs/1902.02134>`__.
    """

    def cost(j):
        k = 2**j
        num_columns = -(-num_bitstrings // k)
        num_fixup_work = num_aux_wires - k
        shortfall = max(0, num_control_wires - j - 1 - num_fixup_work)
        fixup = _qrom_unary_iteration_resources(
            Bool[num_columns, k], Wire[num_control_wires - j], Wire[k], Wire[num_fixup_work]
        )
        num_elbows = k - 1 + fixup.get(TemporaryAND, 0)
        return shortfall, 4 * num_elbows + 7 * shortfall * num_columns * k, j

    # The cost estimate creates abstract operators, which must not end up in a captured program.
    with capture.pause(), QueuingManager.stop_recording():
        return min(range(min(num_control_wires, num_aux_wires.bit_length() - 1) + 1), key=cost)


def _one_hot_swap_triples(address_wires, one_hot_wires):
    r"""Wire triples ``(control, x, y)`` of the controlled SWAPs that move a 1 from
    ``one_hot_wires[0]`` to ``one_hot_wires[l]``, where ``l`` is the value stored in
    ``address_wires`` (most-significant bit first), following Figure 8 of
    `arXiv:1902.02134 <https://arxiv.org/abs/1902.02134>`__. The wire ``y`` is in the
    :math:`|0\rangle` state before its controlled SWAP is applied.

    These are the controlled SWAPs of ``qp.adjoint(_swap_ops)`` with a single target column,
    i.e., of the inverted SWAP network of Select-SWAP. They are not obtained from ``_swap_ops``,
    because the latter applies generic controlled SWAPs and cannot make use of ``y`` being in
    the :math:`|0\rangle` state."""
    num_address = len(address_wires)
    return [
        (address_wires[num_address - 1 - p], one_hot_wires[j], one_hot_wires[j + 2**p])
        for p in range(num_address)
        for j in range(2**p)
    ]


def _measure_and_reset_resources(num_wires):
    return {
        PauliMeasure("X", wires=Wire[1]): num_wires,
        PauliMeasure("Z", wires=Wire[1]): num_wires,
        X: num_wires,
    }


def _measure_and_reset(wires):
    r"""Measure each wire in the :math:`X` basis and reset it to :math:`|0\rangle`, returning the
    outcomes of the :math:`X`-basis measurements. It has its independent resource function
    above to allow for reuse in ``SelectSwap``."""
    outcomes = []
    for wire in wires:
        outcome = pauli_measure("X", wire)
        outcomes.append(outcome)
        m_reset = pauli_measure("Z", wire)
        cond(m_reset, X)(wire)
    return outcomes


def _phase_fixup_resources(num_entries, num_control_wires, num_aux_wires):
    num_one_hot_address = _adjoint_qrom_load_one_hot_size(
        num_entries, num_control_wires, num_aux_wires
    )
    k = 2**num_one_hot_address
    num_columns = -(-num_entries // k)

    fixup_qrom = QROM(
        Bool[num_columns, k],
        Wire[num_control_wires - num_one_hot_address],
        Wire[k],
        Wire[num_aux_wires - k],
    )
    return defaultdict(
        int,
        {
            X: 2,
            TemporaryAND: k - 1,
            CNOT: 2 * (k - 1),
            H: 2 * k,
            fixup_qrom: 1,
            qp_ops.adjoint(TemporaryAND(Wire[3])): k - 1,
        },
    )


def _phase_fixup(fixups, control_wires, aux_wires):
    r"""Apply the phase :math:`(-1)^{f_i}` to each address :math:`i` of the control register,
    for the bits :math:`f_i` in ``fixups``, and none to addresses beyond.
     It has its independent resource function above to allow for reuse in ``SelectSwap``.

    A :class:`~.QROM` of the fixup table over the most-significant address bits targets a one-hot
    encoding of the :math:`j` least-significant address bits in :math:`k=2^j` auxiliary wires
    that are prepared in the :math:`|\pm\rangle` states (Figure 6 of
    `arXiv:1902.02134 <https://arxiv.org/abs/1902.02134>`__). The one-hot encoding is computed
    with ``TemporaryAND`` gates and uncomputed with their adjoints (Figures 8 to 10). The first
    :math:`k` auxiliary wires make up the one-hot register, and the remaining ones are passed on
    to the ``QROM``. All auxiliary wires must be in the :math:`|0\rangle` state.
    """
    num_entries, num_control_wires = len(fixups), len(control_wires)
    num_one_hot_address = _adjoint_qrom_load_one_hot_size(
        num_entries, num_control_wires, len(aux_wires)
    )
    k = 2**num_one_hot_address
    num_columns = -(-num_entries // k)

    fixups = math.concatenate(
        [fixups, math.zeros(num_columns * k - num_entries, dtype=int, like=fixups)]
    )
    fixup_table = math.cast(math.reshape(fixups, (num_columns, k)), bool)

    high_wires = control_wires[: num_control_wires - num_one_hot_address]
    low_wires = control_wires[num_control_wires - num_one_hot_address :]

    one_hot_wires = list(aux_wires[:k])
    triples = _one_hot_swap_triples(low_wires, one_hot_wires)

    X(one_hot_wires[0])
    for control, x, y in triples:
        TemporaryAND([control, x, y])
        CNOT([y, x])

    for wire in one_hot_wires:
        H(wire)
    QROM(fixup_table, high_wires, one_hot_wires, aux_wires[k:])
    for wire in one_hot_wires:
        H(wire)

    for control, x, y in reversed(triples):
        CNOT([y, x])
        qp_ops.adjoint(TemporaryAND)(wires=[control, x, y])
    X(one_hot_wires[0])


def _adjoint_qrom_load_measurement_resources(base):
    num_targets = len(base.target_wires)
    resources = _phase_fixup_resources(
        len(base.bitstrings), len(base.control_wires), num_targets + len(base.work_wires)
    )
    for op, count in _measure_and_reset_resources(num_targets).items():
        resources[op] += count
    return resources


@register_condition(lambda base: capture.enabled() and len(base.control_wires) > 0)
@register_resources(_adjoint_qrom_load_measurement_resources, exact=False, name="measurement")
def adjoint_qrom_load_measurement(base):
    r"""Uncomputation of QROMLoad with measurements and a phase fixup, following Appendix C of
    `arXiv:1902.02134 <https://arxiv.org/abs/1902.02134>`__.

    The target wires store the bitstring :math:`b_j` for each address :math:`j`, as promised by
    ``QROMLoad``, and are reset to :math:`|0\rangle`.

    Measuring the target wires in the :math:`X` basis with outcomes :math:`m` leaves the phase
    :math:`(-1)^{m\cdot b_j}` on each address. This phase is undone with a :class:`~.QROM` of the
    measurement-dependent fixup table over the most-significant address bits, targeting a one-hot
    encoding of the :math:`j` least-significant address bits in :math:`k=2^j` auxiliary wires
    that are prepared in the :math:`|\pm\rangle` states (Figure 6). The one-hot encoding is
    computed with ``TemporaryAND`` gates and uncomputed with their adjoints (Figures 8 to 10).
    The auxiliary wires are the reset target wires, followed by the work wires, and the remaining
    ones are passed on to the ``QROM``. All work wires must be in the :math:`|0\rangle` state.

    For example, for 16 bitstrings with four control wires, two target wires and three work
    wires, two address bits are encoded into a one-hot register made of both target wires and two
    work wires:

    .. code-block::

        c0: ────────────────────────────────────────────────────╭◑───────────────────────────────────┤
        c1: ────────────────────────────────────────────────────├◑───────────────────────────────────┤
        c2: ─────────────────────────────────────╭●────╭●───────│────────────────●╮─────●╮───────────┤
        c3: ───────────────────────────────╭●────│─────│────────│─────────────────│──────│─────●╮────┤
        t0: ──┤↗X├──┤↗Z├──X──X─────────────├●─╭X─├●─╭X─│───H────├QROM(M0)──H──────│─╭X──●┤─╭X──●┤──X─┤
        t1: ─────────────────┤↗X├──┤↗Z├──X─╰⊕─╰●─│──│──├●─╭X──H─├QROM(M0)──H─╭X──●┤─│────│─╰●──⊕╯────┤
        w0: ─────────────────────────────────────╰⊕─╰●─│──│───H─├QROM(M0)──H─│────│─╰●──⊕╯───────────┤
        w1: ───────────────────────────────────────────╰⊕─╰●──H─├QROM(M0)──H─╰●──⊕╯──────────────────┤
        w2: ────────────────────────────────────────────────────╰work────────────────────────────────┤

    Here, the first ``X`` after the measurements on each target wire is conditioned on the outcome
    of its :math:`Z` measurement, and ``M0`` is the fixup table computed from the outcomes of the
    :math:`X` measurements.
    """
    bitstrings = math.cast(base.bitstrings, int)
    target_wires = base.target_wires
    outcomes = _measure_and_reset(target_wires)
    fixups = math.dot(bitstrings, math.cast(math.stack(outcomes), int)) % 2
    _phase_fixup(fixups, base.control_wires, Wires(target_wires) + Wires(base.work_wires))


add_decomps(QROMLoad, qrom_load_select_swap, qrom_load_to_qrom)


add_decomps("Adjoint(QROMLoad)", adjoint_qrom_load_measurement)
