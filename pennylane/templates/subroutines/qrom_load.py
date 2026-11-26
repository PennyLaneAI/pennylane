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

from collections import defaultdict
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

    To unload the data again, apply ``qp.adjoint`` of ``QROMLoad``. If you can leave the work
    wires in a garbage state until unloading, use :class:`~.SelectSwap` instead, which skips
    cleaning them up and thus costs less than half. Unloading with ``Adjoint(QROMLoad)`` can be
    decomposed with the measurement-based uncomputation from Appendix C of
    [`arXiv:1902.02134 <https://arxiv.org/abs/1902.02134>`__].

    .. warning::

        If the target register does not start out in :math:`|0\rangle`, the result depends on the
        decomposition and generally is *not* the XOR you would get from :class:`~.QROM`. Use
        :class:`~.QROM` if you cannot assume a zeroed target.

    .. seealso:: :class:`~.QROM`, :class:`~.SelectSwap`, :class:`~.BBQRAM`,
        :class:`~.QROMStatePreparation`

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

            def t_count(op_type):
                @qp.qjit(capture=True)
                @qp.qnode(qp.device("null.qubit", wires=21))
                def circuit():
                    op_type(*args)
                    return qp.probs()

                counts = qp.estimate(circuit, target=qp.gate_sets.CLIFFORD_T)().resources
                return counts.quantum_operations["T"] + counts.quantum_operations["Adjoint(T)"]

        >>> t_count(qp.QROM), t_count(qp.QROMLoad)
        (1012, 656)

        So ``QROMLoad`` saves more than a third of the T gates. To see where the savings come from,
        let us draw the Select-SWAP decomposition:

        >>> op = qp.QROMLoad(*args)
        >>> rule = qp.list_decomps(qp.QROMLoad)["select_swap"]
        >>> print(qp.draw(rule, show_matrices=False)(**op.arguments))
         0: ────────────────────────────────────────╭◑────────────────────────────────────────────── ···
         1: ────────────────────────────────────────├◑────────────────────────────────────────────── ···
         2: ────────────────────────────────────────├◑────────────────────────────────────────────── ···
         3: ────────────────────────────────────────├◑────────────────────────────────────────────── ···
         4: ────────────────────────────────────────├◑────────────────────────────────────────────── ···
         5: ────────────────────────────────────────├◑────────────────────────────────────────────── ···
         6: ────────────────╭●────╭●────╭●────╭●────│─────────╭●────╭●────╭●────╭●────────────────── ···
         7: ────╭●────╭●────│─────│─────│─────│─────│─────────│─────│─────│─────│─────╭●────╭●────── ···
         8: ──H─│─────├SWAP─│─────├SWAP─│─────│─────├QROM(M0)─│─────│─────├SWAP─│─────├SWAP─│──────H ···
         9: ──H─├SWAP─│─────├SWAP─│─────│─────│─────├QROM(M0)─│─────│─────│─────├SWAP─│─────├SWAP──H ···
        10: ────│─────╰SWAP─│─────│─────│─────├SWAP─├QROM(M0)─├SWAP─│─────│─────│─────╰SWAP─│─────── ···
        11: ────╰SWAP───────│─────│─────├SWAP─│─────├QROM(M0)─│─────├SWAP─│─────│───────────╰SWAP─── ···
        12: ────────────────│─────╰SWAP─│─────│─────├QROM(M0)─│─────│─────╰SWAP─│─────────────────── ···
        13: ────────────────╰SWAP───────│─────│─────├QROM(M0)─│─────│───────────╰SWAP─────────────── ···
        14: ────────────────────────────│─────╰SWAP─├QROM(M0)─╰SWAP─│─────────────────────────────── ···
        15: ────────────────────────────╰SWAP───────├QROM(M0)───────╰SWAP─────────────────────────── ···
        16: ────────────────────────────────────────├work─────────────────────────────────────────── ···
        17: ────────────────────────────────────────├work─────────────────────────────────────────── ···
        18: ────────────────────────────────────────├work─────────────────────────────────────────── ···
        19: ────────────────────────────────────────├work─────────────────────────────────────────── ···
        20: ────────────────────────────────────────╰work─────────────────────────────────────────── ···
        <BLANKLINE>
         0: ··· ─────────────────────────────────────╭◑────────────────────────────────────────────┤
         1: ··· ─────────────────────────────────────├◑────────────────────────────────────────────┤
         2: ··· ─────────────────────────────────────├◑────────────────────────────────────────────┤
         3: ··· ─────────────────────────────────────├◑────────────────────────────────────────────┤
         4: ··· ─────────────────────────────────────├◑────────────────────────────────────────────┤
         5: ··· ─────────────────────────────────────├◑────────────────────────────────────────────┤
         6: ··· ─────────────╭●────╭●────╭●────╭●────│─────────╭●────╭●────╭●────╭●────────────────┤
         7: ··· ─╭●────╭●────│─────│─────│─────│─────│─────────│─────│─────│─────│─────╭●────╭●────┤
         8: ··· ─│─────├SWAP─│─────├SWAP─│─────│─────├QROM(M0)─│─────│─────├SWAP─│─────├SWAP─│─────┤
         9: ··· ─├SWAP─│─────├SWAP─│─────│─────│─────├QROM(M0)─│─────│─────│─────├SWAP─│─────├SWAP─┤
        10: ··· ─│─────╰SWAP─│─────│─────│─────├SWAP─├QROM(M0)─├SWAP─│─────│─────│─────╰SWAP─│─────┤
        11: ··· ─╰SWAP───────│─────│─────├SWAP─│─────├QROM(M0)─│─────├SWAP─│─────│───────────╰SWAP─┤
        12: ··· ─────────────│─────╰SWAP─│─────│─────├QROM(M0)─│─────│─────╰SWAP─│─────────────────┤
        13: ··· ─────────────╰SWAP───────│─────│─────├QROM(M0)─│─────│───────────╰SWAP─────────────┤
        14: ··· ─────────────────────────│─────╰SWAP─├QROM(M0)─╰SWAP─│─────────────────────────────┤
        15: ··· ─────────────────────────╰SWAP───────├QROM(M0)───────╰SWAP─────────────────────────┤
        16: ··· ─────────────────────────────────────├work─────────────────────────────────────────┤
        17: ··· ─────────────────────────────────────├work─────────────────────────────────────────┤
        18: ··· ─────────────────────────────────────├work─────────────────────────────────────────┤
        19: ··· ─────────────────────────────────────├work─────────────────────────────────────────┤
        20: ··· ─────────────────────────────────────╰work─────────────────────────────────────────┤

        The inner ``QROM``\ s only read the first six control wires, so they have 64 instead of 256
        entries. Each entry stacks four bitstrings, which are loaded onto the target wires 8 and 9
        *and* the work wires 10 to 15 at once. The two remaining control wires (wires 6 and 7) then
        pick the requested bitstring with controlled ``SWAP`` gates around each ``QROM``, moving it
        into the target register. ``QROMLoad`` weighs the reduced cost of two smaller ``QROM`` calls
        and the four controlled ``SWAP`` blocks against the cost of a single full-sized ``QROM``,
        and picks the number of bitstrings to load in parallel that minimizes the combined cost,
        among the powers of two that fit into the work wires.
        Here, eight bitstrings in parallel would not fit, and two would be more expensive than four.

    .. details::
        :title: Available decompositions

        .. decomposition-table:: qp.QROMLoad
            :symbols: :math:`L` bitstrings of length :math:`n`, :math:`c` control wires and
                :math:`w` work wires
            :example: :math:`L=32`, :math:`n=1`, :math:`c=5` and :math:`w=5`

            .. decomposition-rule:: select_swap
                :image: /_static/decomposition_thumbnails/qrom_load_select_swap.svg
                :lightbox:
                :requires: :math:`w \geq c - 1` work wires, and :math:`k > 1` to be the cheapest
                    choice
                :where: :math:`k = 2^j` bitstrings are loaded in parallel, with :math:`j` chosen to
                    minimize the cost among all :math:`j` with
                    :math:`n(k - 1) + \max(c - j - 1, 0) \leq w`

                :class:`~.QROM` with :math:`\lceil L/k \rceil` bitstrings of length :math:`kn` on :math:`c - j` control wires | :math:`2` | 2
                :class:`~.SWAP` with 1 control | :math:`4n(k - 1)` | 12
                :class:`~.Hadamard` | :math:`2n` | 2

            .. decomposition-rule:: to_qrom
                :image: /_static/decomposition_thumbnails/qrom_load_to_qrom.svg

                :class:`~.QROM` | :math:`1` | 1

            .. decomposition-rule:: select_swap_measure_garbage
                :image: /_static/decomposition_thumbnails/qrom_load_select_swap_measure_garbage.svg
                :lightbox:
                :requires: program capture, as with :func:`qjit(capture=True) <.qjit>`,
                    and :math:`k > 1`
                :where: :math:`k` is the number of bitstrings that :class:`~.SelectSwap` loads in
                    parallel. The :math:`n(k - 1)` work wires holding the bitstrings that were
                    not swapped into the target register are unloaded by the adjoint of a
                    ``QROMLoad`` of these bitstrings for each of the
                    :math:`L' = k \lceil L/k \rceil` addresses, which can be decomposed with
                    measurements and a phase fixup (see the ``measurement`` rule below)

                :class:`~.SelectSwap` | :math:`1` | 1
                :class:`Adjoint(QROMLoad) <.QROMLoad>` with :math:`L'` bitstrings of length :math:`n(k - 1)` on :math:`c` control wires and :math:`w - n(k - 1)` work wires | :math:`1` | 1

            .. decomposition-rule:: measurement
                :image: /_static/decomposition_thumbnails/qrom_load_adjoint_measurement.svg
                :lightbox:
                :region: adjoint
                :requires: program capture, as with :func:`qjit(capture=True) <.qjit>`,
                    and :math:`c \geq 1` control wires
                :where: the :math:`j` least-significant control wires are encoded into a one-hot
                    register of :math:`k = 2^j` auxiliary wires, taken from the target wires
                    after their reset and the work wires, with
                    :math:`j \leq \min(c, \lfloor \log_2 (n + w) \rfloor)` chosen to minimize the
                    estimated T-count. With enough auxiliary wires, this approximately minimizes
                    the Toffoli count :math:`\lceil L/k \rceil + k`

                :class:`~.QROM` with :math:`\lceil L/k \rceil` bitstrings of length :math:`k` on :math:`c - j` control wires and :math:`n + w - k` work wires | :math:`1` | 1
                :func:`PauliMeasure("X") <.pauli_measure>` | :math:`n` | 1
                :func:`PauliMeasure("Z") <.pauli_measure>` | :math:`n` | 1
                :class:`~.PauliX` (conditional) | :math:`n` | 1
                :class:`~.PauliX` | :math:`2` | 2
                :class:`~.TemporaryAND` | :math:`k - 1` | 3
                :class:`Adjoint(TemporaryAND) <.TemporaryAND>` | :math:`k - 1` | 3
                :class:`~.CNOT` | :math:`2(k - 1)` | 6
                :class:`~.Hadamard` | :math:`2k` | 8

    """

    dynamic_argnames = ("bitstrings",)
    wire_argnames = ("control_wires", "target_wires", "work_wires")

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
    ):
        args = (bitstrings, control_wires, target_wires, work_wires)
        super().__init__(*preprocess_qrom_inputs(*args))

    @property
    def wires(self):
        """All wires involved in the operation."""
        return self.control_wires + self.target_wires + self.work_wires


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


def _select_swap_sizes(bitstrings, control_wires, target_wires, work_wires):
    return _calculate_select_swap_sizes(
        len(bitstrings), len(control_wires), len(target_wires), len(work_wires), clean=False
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
def _select_swap(bitstrings, control_wires, target_wires, work_wires):
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
def _qrom_load_to_qrom(bitstrings, control_wires, target_wires, work_wires):
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
def _adjoint_qrom_load_measurement(base):
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


def _swap_permutations(depth):
    """For each value ``l`` of the SWAP control wires, the permutation ``perms[l]`` of a column of
    ``depth`` bitstrings applied by the SWAP network of ``_swap_ops``, which moves bitstring
    ``perms[l, p]`` of the column into position ``p``. The SWAP layer controlled on bit ``i`` of
    ``l`` exchanges the first two blocks of ``2**i`` positions."""
    perms = math.tile(math.arange(depth), (depth, 1))
    for i in reversed(range(depth.bit_length() - 1)):
        swapped = (math.arange(depth) >> i) & 1 == 1
        perms[swapped, : 2 ** (i + 1)] = math.roll(perms[swapped, : 2 ** (i + 1)], 2**i, axis=1)
    return perms


def _select_swap_table(bitstrings, depth):
    r"""Return the bits that ``SelectSwap`` leaves on its target wires followed by the first
    :math:`n(k - 1)` work wires, for each of the :math:`k\lceil L/k \rceil` addresses, with
    :math:`k` = ``depth``.

    Row :math:`i` consists of the :math:`k` bitstrings :math:`b_{\pi_i(p)}` that were loaded in
    parallel, in the order :math:`p` in which the SWAP network leaves them, starting with the
    bitstring in the target register. As ``SelectSwap`` loads nothing beyond these rows, it acts
    on the target and first :math:`n(k - 1)` work wires like a ``QROMLoad`` of this table.

    Args:
        bitstrings (TensorLike): the :math:`L` loaded bitstrings as integers
        depth (int): the number :math:`k` of bitstrings loaded in parallel

    Returns:
        TensorLike: the table of shape ``(depth * ceil(L / depth), depth * n)``
    """
    num_bitstrings, num_targets = math.shape(bitstrings)
    num_columns = -(-num_bitstrings // depth)
    padding = math.zeros(
        (num_columns * depth - num_bitstrings, num_targets), dtype=int, like=bitstrings
    )
    columns = math.concatenate([bitstrings, padding])
    columns = math.reshape(columns, (num_columns, depth, num_targets))
    # loaded[h, l, p] is the bitstring that the SWAP network moves into position p of column h
    loaded = columns[:, _swap_permutations(depth)]
    return math.reshape(loaded, (num_columns * depth, depth * num_targets))


def _select_swap_measure_garbage_condition(bitstrings, control_wires, target_wires, work_wires):
    if not capture.enabled() or len(control_wires) == 0:
        return False
    *_, depth = _select_swap_sizes(bitstrings, control_wires, target_wires, work_wires)
    return depth > 1


def _select_swap_measure_garbage_resources(bitstrings, control_wires, target_wires, work_wires):
    # pylint: disable-next=import-outside-toplevel
    from .select_swap import SelectSwap

    _, _, num_work_wires_swap, _, depth = _select_swap_sizes(
        bitstrings, control_wires, target_wires, work_wires
    )
    num_entries = depth * -(-len(bitstrings) // depth)
    select_swap = SelectSwap(
        Bool[*bitstrings.shape],
        Wire[len(control_wires)],
        Wire[len(target_wires)],
        Wire[len(work_wires)],
    )
    unload_garbage = qp_ops.adjoint(
        QROMLoad(
            Bool[num_entries, num_work_wires_swap],
            Wire[len(control_wires)],
            Wire[num_work_wires_swap],
            Wire[len(work_wires) - num_work_wires_swap],
        )
    )
    return {select_swap: 1, unload_garbage: 1}


@register_condition(_select_swap_measure_garbage_condition)
@register_resources(_select_swap_measure_garbage_resources, name="select_swap_measure_garbage")
def _select_swap_measure_garbage(bitstrings, control_wires, target_wires, work_wires):
    r"""Load with ``SelectSwap`` and unload the garbage on its work wires with
    ``Adjoint(QROMLoad)``, which can be decomposed with measurements and a phase fixup following
    Appendix C of `arXiv:1902.02134 <https://arxiv.org/abs/1902.02134>`__.

    The work wires holding the bitstrings that were loaded in parallel but not swapped into the
    target register are in the state that a ``QROMLoad`` of these bitstrings would prepare, and
    the remaining work wires are in :math:`|0\rangle`. Compared to the ``select_swap`` rule, this
    replaces the second round of Select and SWAPs by the uncomputation of that ``QROMLoad``.
    """
    # pylint: disable-next=import-outside-toplevel
    from .select_swap import SelectSwap

    bitstrings = math.cast(bitstrings, int)
    _, _, num_work_wires_swap, _, depth = _select_swap_sizes(
        bitstrings, control_wires, target_wires, work_wires
    )

    SelectSwap(bitstrings, control_wires, target_wires, work_wires)

    garbage = _select_swap_table(bitstrings, depth)[:, len(target_wires) :]
    qp_ops.adjoint(QROMLoad)(
        garbage,
        control_wires,
        work_wires[:num_work_wires_swap],
        work_wires[num_work_wires_swap:],
    )


add_decomps(QROMLoad, _select_swap, _qrom_load_to_qrom, _select_swap_measure_garbage)
add_decomps("Adjoint(QROMLoad)", _adjoint_qrom_load_measurement)
