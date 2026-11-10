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
"""
This submodule contains the template QROMLoad.
"""

from collections.abc import Sequence
from functools import partial

from pennylane import capture, compiler, math
from pennylane import ops as qp_ops
from pennylane.capture import hint
from pennylane.control_flow import for_loop
from pennylane.core.operator import Operator2
from pennylane.core.queuing import QueuingManager
from pennylane.decomposition import (
    add_decomps,
    register_condition,
    register_resources,
)
from pennylane.ops import CNOT, SWAP, H, X, cond, ctrl, pauli_measure
from pennylane.ops.mid_measure.pauli_measure import PauliMeasure
from pennylane.typing import Bool, TensorLike, Wire
from pennylane.wires import Wires, WiresLike

from .arithmetic import TemporaryAND
from .multix import MultiX
from .qrom import (
    QROM,
    _qrom_unary_iteration_resources,
    _unary_iteration_num_elbows,
    preprocess_qrom_inputs,
)


class QROMLoad(Operator2):
    r"""Load bitstrings into a zeroed target register, depending on the index stored in a control
    register.

    For :math:`L` classical bitstrings :math:`b_0,\dots,b_{L-1}`, ``QROMLoad`` reads the index
    :math:`i` from the control register and writes :math:`b_i` into the target register, which
    has to start out in :math:`|0\rangle`:

    .. math::

        \text{QROMLoad}|i\rangle|0\rangle = \begin{cases}
            |i\rangle|b_i\rangle & \text{if } i < L,\\
            |i\rangle|0\rangle & \text{otherwise.}
        \end{cases}

    So this is :class:`~.QROM` with the promise of a zeroed target, and that promise unlocks
    reduced quantum resources: given enough work wires to hold additional copies of the target
    register, ``QROMLoad`` is decomposed with `Select-SWAP <https://arxiv.org/abs/1812.00954>`__.
    It loads several bitstrings in parallel and then swaps the right one into the target register,
    trading qubits for fewer T gates. With fewer work wires, or if loading in parallel does not pay
    off, it simply falls back to ``QROM``.

    To unload the data again, apply ``qp.adjoint`` of ``QROMLoad``. For ``clean=False``, this
    requires the garbage state of the work wires to be left untouched in the meantime.
    For ``clean=True`` and within :func:`~.qjit` with program capture, the adjoint can be
    decomposed with the measurement-based uncomputation from Appendix C of
    [`arXiv:1902.02134 <https://arxiv.org/abs/1902.02134>`__], which requires at least one
    work wire.

    .. warning::

        If the target register does not start out in :math:`|0\rangle`, the result depends on the
        decomposition and generally is *not* the XOR you would get from :class:`~.QROM`. Use
        :class:`~.QROM` if you cannot assume a zeroed target.

    .. seealso:: :class:`~.QROM`, :class:`~.BBQRAM`, :class:`~.QROMStatePreparation`

    Args:
        bitstrings (TensorLike | Sequence[str]): the :math:`L` bitstrings, either as a 2D array of
            bits or as a sequence of strings like ``"0110"``. ``bitstrings[i]`` is loaded if the
            control register is in the state :math:`|i\rangle`.
        control_wires (WiresLike): the control register holding the index :math:`i`, most
            significant bit first. Requires at least :math:`\lceil\log_2 L\rceil` wires.
        target_wires (WiresLike): the target register, with one wire per bit of the bitstrings.
            Needs to start in :math:`|0\rangle`.
        work_wires (WiresLike): auxiliary wires, which need to start in :math:`|0\rangle`. The more
            work wires, the more bitstrings Select-SWAP can load in parallel. It only uses as many
            as pay off, though, so not all work wires are necessarily used.
        clean (bool): whether to return the ``work_wires`` to :math:`|0\rangle`. With
            ``clean=False``, they are left in a garbage state, which cuts the cost of Select-SWAP
            by more than half. As this garbage state depends on the decomposition, Select-SWAP is
            then always used where it pays off, irrespective of the target gate set. Defaults to
            ``True``.

    **Example**

    Consider four 3-bit strings, store the index :math:`i=2` in the control register, and
    apply ``QROMLoad``:

    .. code-block:: python

        bitstrings = ["010", "111", "110", "000"]

        @qp.qnode(qp.device("default.qubit"), shots=1)
        def circuit():
            qp.BasisState([1, 0], wires=[0, 1])
            qp.QROMLoad(bitstrings, control_wires=[0, 1], target_wires=[2, 3, 4], work_wires=[5])
            return qp.sample(wires=[2, 3, 4])

    >>> print(circuit())
    [[1 1 0]]

    This is exactly ``bitstrings[2]``, loaded into the target register.

    .. details::
        :title: Cheaper decomposition with Select-SWAP

        For the example above, we could just have used :class:`~.QROM` instead, and without work
        wires ``QROMLoad`` even falls back to ``QROM`` itself. To demonstrate the savings we can
        get from additional work space if the target wires are guaranteed to start in the state
        :math:`|0\rangle`, let us compare the two operators on a larger example: 256 bitstrings
        of length 2, with 11 work wires. We let :func:`~.estimate` count the T gates in their
        decompositions into Clifford+T gates:

        .. code-block:: python

            bitstrings = np.random.default_rng(42).integers(0, 2, size=(256, 2))
            reg = qp.registers({"control": 8, "target": 2, "work": 11})
            args = (bitstrings, reg["control"], reg["target"], reg["work"])

            def t_count(op_type, **kwargs):
                @qp.qjit(capture=True)
                @qp.qnode(qp.device("null.qubit", wires=21))
                def circuit():
                    op_type(*args, **kwargs)
                    return qp.probs()

                counts = qp.estimate(circuit, target=qp.gate_sets.CLIFFORD_T)().resources
                return counts.quantum_operations["T"] + counts.quantum_operations["Adjoint(T)"]

        >>> t_count(qp.QROM), t_count(qp.QROMLoad), t_count(qp.QROMLoad, clean=False)
        (1012, 656, 286)

        So ``QROMLoad`` saves more than a third of the T gates, and if we do not mind garbage on the
        work wires, ``clean=False`` saves more than two thirds. To see where the savings come from,
        let us draw the Select-SWAP decomposition with ``clean=False``:

        >>> op = qp.QROMLoad(*args, clean=False)
        >>> rule = qp.list_decomps(qp.QROMLoad)["_select_swap"]
        >>> print(qp.draw(rule, show_matrices=False)(**op.arguments))
         0: ─╭◑────────────────────────────────────────────┤
         1: ─├◑────────────────────────────────────────────┤
         2: ─├◑────────────────────────────────────────────┤
         3: ─├◑────────────────────────────────────────────┤
         4: ─├◑────────────────────────────────────────────┤
         5: ─├◑────────────────────────────────────────────┤
         6: ─│─────────╭●────╭●────╭●────╭●────────────────┤
         7: ─│─────────│─────│─────│─────│─────╭●────╭●────┤
         8: ─├QROM(M0)─│─────│─────├SWAP─│─────├SWAP─│─────┤
         9: ─├QROM(M0)─│─────│─────│─────├SWAP─│─────├SWAP─┤
        10: ─├QROM(M0)─├SWAP─│─────│─────│─────╰SWAP─│─────┤
        11: ─├QROM(M0)─│─────├SWAP─│─────│───────────╰SWAP─┤
        12: ─├QROM(M0)─│─────│─────╰SWAP─│─────────────────┤
        13: ─├QROM(M0)─│─────│───────────╰SWAP─────────────┤
        14: ─├QROM(M0)─╰SWAP─│─────────────────────────────┤
        15: ─├QROM(M0)───────╰SWAP─────────────────────────┤
        16: ─├work─────────────────────────────────────────┤
        17: ─├work─────────────────────────────────────────┤
        18: ─├work─────────────────────────────────────────┤
        19: ─├work─────────────────────────────────────────┤
        20: ─╰work─────────────────────────────────────────┤

        The inner ``QROM`` only reads the first six control wires, so it has 64 instead of 256
        entries. Each entry stacks four bitstrings, which are loaded onto the target wires 8 and 9
        *and* the work wires 10 to 15 at once. The two remaining control wires then pick the
        requested bitstring with controlled ``SWAP`` gates, moving it into the target register.
        In general, loading :math:`k` bitstrings of length :math:`n` in parallel divides the number
        of entries in the ``QROM`` block by :math:`k`, at the price of :math:`n(k-1)` controlled
        ``SWAP`` gates. ``QROMLoad`` weighs the two against each other and picks the :math:`k`
        that minimizes the combined cost, among the powers of two that fit into the work wires.
        Here, eight bitstrings in parallel would not fit, and two would be more expensive than four.

    """

    dynamic_argnames = ("bitstrings",)
    wire_argnames = ("control_wires", "target_wires", "work_wires")
    compilable_argnames = ("clean",)

    arg_specs = {
        "bitstrings": Bool[-1, -1],
        "control_wires": Wire[-1],
        "target_wires": Wire[-1],
        "work_wires": Wire[-1],
    }

    def __init__(
        self,
        bitstrings: TensorLike | Sequence[str],
        control_wires: WiresLike,
        target_wires: WiresLike,
        work_wires: WiresLike,
        clean: bool = True,
    ):  # pylint: disable=too-many-arguments
        args = (bitstrings, control_wires, target_wires, work_wires)
        super().__init__(*preprocess_qrom_inputs(*args), clean)

    @property
    def wires(self):
        """All wires involved in the operation."""
        return self.control_wires + self.target_wires + self.work_wires


def _select_ops(
    bitstrings, depth, target_wires, swap_wires, select_control_wires, select_work_wires
):  # pylint:disable=too-many-arguments
    num_targets = len(target_wires)
    num_bitstrings = bitstrings.shape[0]
    num_columns = -(-num_bitstrings // depth)
    num_missing = (-num_bitstrings) % depth

    if num_missing > 0:
        bitstrings = math.vstack([bitstrings, math.zeros((num_missing, num_targets), dtype=int)])

    num_targets_select = depth * num_targets
    new_bitstrings = bitstrings.reshape((num_columns, num_targets_select))
    QROM(
        new_bitstrings,
        control_wires=select_control_wires,
        target_wires=swap_wires[:num_targets_select],
        work_wires=select_work_wires,
    )


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


def _select_swap_cost(num_bitstrings, num_control_wires, num_targets, log_depth, clean):
    """Number of Toffoli-like gates, i.e., elbows of the Select component and controlled SWAPs,
    of the Select-SWAP decomposition loading ``2**log_depth`` bitstrings in parallel. For
    ``log_depth=0``, this is the cost of a single plain ``QROM``, irrespective of ``clean``."""
    num_control_wires_select = num_control_wires - log_depth
    num_columns = -(-num_bitstrings // 2**log_depth)
    num_elbows = 0
    if num_control_wires_select >= 2:
        num_elbows = _unary_iteration_num_elbows(num_control_wires_select, num_columns)
    num_cswaps = num_targets * (2**log_depth - 1)
    if clean and log_depth > 0:
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
        clean (bool): whether the work wires are returned to the zero state, which changes the
            cost of the decomposition

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


def _select_swap_condition(bitstrings, control_wires, target_wires, work_wires, clean):
    # pylint: disable=unused-argument
    """We use Select-SWAP only when there are enough work wires for unary iteration on the
    nested select QROM and an actual SWAP network is used on top, i.e. for depth > 1. If the
    cost-optimal depth is 1, the plain QROM fallback is the better decomposition."""
    num_control_wires = len(control_wires)
    num_work_wires = len(work_wires)

    if num_control_wires == 0 or num_work_wires < num_control_wires - 1:
        return False

    *_, depth = _calculate_select_swap_sizes(
        len(bitstrings), num_control_wires, len(target_wires), num_work_wires, clean
    )
    return depth > 1


def _select_swap_resources(
    bitstrings, control_wires, target_wires, work_wires, clean
):  # pylint: disable=too-many-branches
    """Assumes depth computed below satisfies depth > 1, as guaranteed by _select_swap_condition."""
    num_bitstrings = len(bitstrings)
    num_control_wires = len(control_wires)
    num_targets = len(target_wires)
    num_work_wires = len(work_wires)

    num_control_wires_select, _, _, num_work_wires_cswap, depth = _calculate_select_swap_sizes(
        num_bitstrings, num_control_wires, num_targets, num_work_wires, clean
    )
    num_work_wires_select = max(0, num_control_wires_select - 1)
    num_columns = -(-num_bitstrings // depth)
    # Select block (implemented as a nested QROM over concatenated columns)
    num_targets_select = depth * num_targets
    bigger_qrom = QROM(
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

    if not clean:
        return {bigger_qrom: 1, cswap_rep: num_cswaps_per_block}

    return {bigger_qrom: 2, cswap_rep: 4 * num_cswaps_per_block, H: 2 * num_targets}


@register_condition(_select_swap_condition)
@register_resources(_select_swap_resources)
def _select_swap(
    bitstrings, control_wires, target_wires, work_wires, clean
):  # pylint: disable=unused-argument, too-many-arguments
    if len(control_wires) == 0:
        MultiX(bitstrings[0, :], wires=target_wires)
        return

    num_control_wires_select, _, num_work_wires_swap, num_work_wires_cswap, depth = (
        _calculate_select_swap_sizes(
            len(bitstrings), len(control_wires), len(target_wires), len(work_wires), clean
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

    if not clean:
        _select_ops(
            bitstrings, depth, target_wires, swap_wires, select_control_wires, select_work_wires
        )
        _swap_ops(swap_control_wires, swap_wires, target_wires, cswap_work_wires)
        return

    if capture.enabled() or compiler.active():
        target_wires = math.array(target_wires, like="jax")

    @for_loop(2)
    def _select_swap_loop(i):

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


def _qrom_load_to_qrom_resources(bitstrings, control_wires, target_wires, work_wires, clean):
    # pylint: disable=unused-argument
    return {
        QROM(
            Bool[*bitstrings.shape],
            Wire[len(control_wires)],
            Wire[len(target_wires)],
            Wire[len(work_wires)],
        ): 1
    }


def _qrom_load_to_qrom_condition(bitstrings, control_wires, target_wires, work_wires, clean):
    """For ``clean=False``, the final state of the work wires depends on the decomposition. Only
    one rule may be applicable, so that loading and unloading with the adjoint always match."""
    return clean or not _select_swap_condition(
        bitstrings, control_wires, target_wires, work_wires, clean
    )


@register_condition(_qrom_load_to_qrom_condition)
@register_resources(_qrom_load_to_qrom_resources)
def _qrom_load_to_qrom(bitstrings, control_wires, target_wires, work_wires, clean):
    # pylint: disable=unused-argument
    QROM(bitstrings, control_wires, target_wires, work_wires)


@QueuingManager.stop_recording()
def _adjoint_qrom_load_one_hot_size(num_bitstrings, num_control_wires, num_work_wires):
    """Number of least-significant address bits ``n`` that are encoded into a one-hot register of
    ``k = 2**n`` work wires for the phase fixup of ``Adjoint(QROMLoad)``.

    The fixup ``QROM`` uses unary iteration over the remaining ``num_control_wires - n`` address
    bits with the remaining ``num_work_wires - k`` work wires. Each work wire it lacks for a full
    iteration tree adds a control to every load, which costs far more than a larger one-hot
    register saves. Therefore, ``n`` first minimizes this shortfall and then the T-count, estimated
    as 4 per ``TemporaryAND`` and 7 per Toffoli in the loads. Without shortfall, this approximately
    minimizes the Toffoli cost ``ceil(L / k) + k`` from Appendix C of
    `arXiv:1902.02134 <https://arxiv.org/abs/1902.02134>`__.
    """

    def cost(n):
        k = 2**n
        num_columns = -(-num_bitstrings // k)
        num_fixup_work = num_work_wires - k
        shortfall = max(0, num_control_wires - n - 1 - num_fixup_work)
        fixup = _qrom_unary_iteration_resources(
            Bool[num_columns, k], Wire[num_control_wires - n], Wire[k], Wire[num_fixup_work]
        )
        num_elbows = k - 1 + fixup.get(TemporaryAND, 0)
        return shortfall, 4 * num_elbows + 7 * shortfall * num_columns * k, n

    return min(range(min(num_control_wires, num_work_wires.bit_length() - 1) + 1), key=cost)


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


def _adjoint_qrom_load_measurement_resources(base):
    num_bitstrings = len(base.bitstrings)
    num_control_wires = len(base.control_wires)
    num_targets = len(base.target_wires)
    num_work_wires = len(base.work_wires)
    num_one_hot_address = _adjoint_qrom_load_one_hot_size(
        num_bitstrings, num_control_wires, num_work_wires
    )
    k = 2**num_one_hot_address
    num_columns = -(-num_bitstrings // k)

    fixup_qrom = QROM(
        Bool[num_columns, k],
        Wire[num_control_wires - num_one_hot_address],
        Wire[k],
        Wire[num_work_wires - k],
    )
    return {
        PauliMeasure("X", wires=Wire[1]): num_targets,
        PauliMeasure("Z", wires=Wire[1]): num_targets,
        X: num_targets + 2,
        TemporaryAND: k - 1,
        CNOT: 2 * (k - 1),
        H: 2 * k,
        fixup_qrom: 1,
        qp_ops.adjoint(TemporaryAND(Wire[3])): k - 1,
    }


@register_condition(
    lambda base: compiler.active()
    and capture.enabled()
    and len(base.control_wires) > 0
    and len(base.work_wires) > 0
    and base.clean
)
@register_resources(_adjoint_qrom_load_measurement_resources, exact=False)
def _adjoint_qrom_load_measurement(base):
    r"""Uncomputation of QROMLoad with measurements and a phase fixup, following Appendix C of
    `arXiv:1902.02134 <https://arxiv.org/abs/1902.02134>`__.

    The target wires store the bitstring :math:`b_j` for each address :math:`j`, as promised by
    ``QROMLoad``, and are reset to :math:`|0\rangle`.

    Measuring the target wires in the :math:`X` basis with outcomes :math:`m` leaves the phase
    :math:`(-1)^{m\cdot b_j}` on each address. This phase is undone with a :class:`~.QROM` of the
    measurement-dependent fixup table over the most-significant address bits, targeting a one-hot
    encoding of the :math:`n` least-significant address bits in :math:`k=2^n` work wires
    that are prepared in the :math:`|\pm\rangle` states (Figure 6). The one-hot encoding is
    computed with ``TemporaryAND`` gates and uncomputed with their adjoints (Figures 8 to 10).
    The remaining work wires are passed on to the ``QROM``. All work wires must be in the
    :math:`|0\rangle` state.

    For example, for 16 bitstrings with four control wires, two target wires and five work
    wires, two address bits are encoded into a one-hot register of :math:`k=4` work wires:

    .. code-block::

        c0: ─────────────────────────────────────╭◑───────────────────────────────────┤
        c1: ─────────────────────────────────────├◑───────────────────────────────────┤
        c2: ──────────────────────╭●────╭●───────│────────────────●╮─────●╮───────────┤
        c3: ────────────────╭●────│─────│────────│─────────────────│──────│─────●╮────┤
        t0: ──┤↗X├──┤↗Z├──X─│─────│─────│────────│─────────────────│──────│──────│────┤
        t1: ──┤↗X├──┤↗Z├──X─│─────│─────│────────│─────────────────│──────│──────│────┤
        w0: ──X─────────────├●─╭X─├●─╭X─│───H────├QROM(M0)──H──────│─╭X──●┤─╭X──●┤──X─┤
        w1: ────────────────╰⊕─╰●─│──│──├●─╭X──H─├QROM(M0)──H─╭X──●┤─│────│─╰●──⊕╯────┤
        w2: ──────────────────────╰⊕─╰●─│──│───H─├QROM(M0)──H─│────│─╰●──⊕╯───────────┤
        w3: ────────────────────────────╰⊕─╰●──H─├QROM(M0)──H─╰●──⊕╯──────────────────┤
        w4: ─────────────────────────────────────╰work────────────────────────────────┤

    Here, the ``X`` on each target wire is conditioned on the outcome of its :math:`Z`
    measurement, and ``M0`` is the fixup table computed from the outcomes of the :math:`X`
    measurements.
    """
    bitstrings = math.cast(base.bitstrings, int)
    control_wires, target_wires, work_wires = base.control_wires, base.target_wires, base.work_wires
    num_bitstrings, num_control_wires = len(bitstrings), len(control_wires)
    num_one_hot_address = _adjoint_qrom_load_one_hot_size(
        num_bitstrings, num_control_wires, len(work_wires)
    )
    k = 2**num_one_hot_address
    num_columns = -(-num_bitstrings // k)

    outcomes = []
    for wire in target_wires:
        outcome = pauli_measure("X", wire)
        outcomes.append(outcome)
        m_reset = pauli_measure("Z", wire)
        cond(m_reset, X)(wire)

    fixups = math.dot(bitstrings, math.cast(math.stack(outcomes), int)) % 2
    fixups = math.concatenate(
        [fixups, math.zeros(num_columns * k - num_bitstrings, dtype=int, like=fixups)]
    )
    fixup_table = math.cast(math.reshape(fixups, (num_columns, k)), bool)

    high_wires = control_wires[: num_control_wires - num_one_hot_address]
    low_wires = control_wires[num_control_wires - num_one_hot_address :]

    one_hot_wires = list(work_wires[:k])
    triples = _one_hot_swap_triples(low_wires, one_hot_wires)

    X(one_hot_wires[0])
    for control, x, y in triples:
        TemporaryAND([control, x, y])
        CNOT([y, x])

    for wire in one_hot_wires:
        H(wire)
    QROM(fixup_table, high_wires, one_hot_wires, work_wires[k:])
    for wire in one_hot_wires:
        H(wire)

    for control, x, y in reversed(triples):
        CNOT([y, x])
        qp_ops.adjoint(TemporaryAND)(wires=[control, x, y])
    X(one_hot_wires[0])


add_decomps(QROMLoad, _select_swap, _qrom_load_to_qrom)
add_decomps("Adjoint(QROMLoad)", _adjoint_qrom_load_measurement)
