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
"""
This submodule contains the template QROMLoad.
"""

from collections.abc import Sequence
from functools import partial

import numpy as np

from pennylane import capture, compiler, math
from pennylane.control_flow import for_loop
from pennylane.core.operator import Operator2
from pennylane.decomposition import (
    add_decomps,
    register_condition,
    register_resources,
)
from pennylane.ops import SWAP, H, ctrl
from pennylane.typing import Bool, TensorLike, Wire
from pennylane.wires import Wires, WiresLike

from .multix import MultiX
from .qrom import QROM, preprocess_qrom_inputs


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
    trading qubits for fewer T gates. With fewer work wires, it simply falls back to ``QROM``.

    To unload the data again, apply ``qp.adjoint`` of ``QROMLoad``.

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
            work wires, the more bitstrings Select-SWAP can load in parallel.
        clean (bool): whether to return the ``work_wires`` to :math:`|0\rangle`. With
            ``clean=False``, they are left in a garbage state, which cuts the cost of Select-SWAP
            by more than half. Defaults to ``True``.

    **Example**

    Consider four 3-bit strings, store the index :math:`i=2` in the control register, and
    apply ``QROMLoad``:

    .. code-block:: python

        bitstrings = ["010", "111", "110", "000"]

        @qp.qnode(qp.device("default.qubit"), shots=1)
        def circuit():
            qp.BasisState([1, 0], wires=[0, 1])
            qp.QROMLoad(bitstrings, control_wires=[0, 1], target_wires=[2, 3, 4])
            return qp.sample(wires=[2, 3, 4])

    >>> print(circuit())
    [[1 1 0]]

    This is exactly ``bitstrings[2]``, loaded into the target register.

    **Cheaper decomposition with Select-SWAP**

    For the example above, we could just have used :class:`~.QROM` instead, and without work
    wires ``QROMLoad`` even falls back to ``QROM`` itself.
    To demonstrate the savings we can get from additional work space if the
    target wires are guaranteed to start in the state :math:`|0\rangle`, let us
    compare the two operators on a larger example:


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
    num_columns = int(np.ceil(num_bitstrings / depth))
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

        inner_loop(inner)()

    outer_loop(outer)()


def _calculate_select_swap_sizes(
    num_bitstrings, num_control_wires, num_targets, num_work_wires, **_
):
    """Calculates the register sizes for the Select-SWAP decomposition.

    This utility function determines how many auxiliary wires from the total pool
    should be allocated to the Select operation versus the SWAP network.

    Args:
        num_bitstrings (int): number of bitstrings/entries in the data
        num_control_wires (int): number of control wires
        num_targets (int): number of target wires (bitstring length)
        num_work_wires (int): total number of available work wires

    Returns:
        tuple[int]: ``(num_control_wires_select, num_work_wires_select, num_work_wires_swap,
        num_work_wires_cswap, depth)`` — control and work wires assigned to the Select component,
        work wires assigned to the SWAP network, and the number of bitstrings loaded in parallel.
    """

    if num_work_wires < num_control_wires - 1:
        return num_control_wires, num_work_wires, 0, 0, 1

    # Initialize available swap space using total work wires
    num_work_wires_swap = num_work_wires
    num_wires_swap = num_targets + num_work_wires_swap

    # Calculate depth: how many bitstrings we can load in parallel (power of 2)
    depth = num_wires_swap // num_targets
    depth = 1 << math.floor_log2(min(depth, num_bitstrings))

    # Recalculate actual wires used by SWAP and the remaining for Select
    num_work_wires_swap = num_targets * depth - num_targets
    num_work_wires_select = num_work_wires - num_work_wires_swap

    # Adjust depth if Select doesn't have enough work wires for the required control logic
    num_control_wires_select = num_control_wires - int(math.floor_log2(depth))
    while num_work_wires_select < num_control_wires_select - 1:
        depth = depth // 2
        num_work_wires_swap = num_targets * (depth - 1)
        num_work_wires_select = num_work_wires - num_work_wires_swap
        num_control_wires_select = num_control_wires - int(math.floor_log2(depth))

    # As soon as there is an excess work wire for Select, reroute it to the CSWAPs themselves.
    num_work_wires_cswap = int(num_work_wires_select - max(0, num_control_wires_select - 1) >= 1)

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
    nested select QROM and an actual SWAP network is used on top, i.e. for depth > 1."""
    num_control_wires = len(control_wires)
    num_work_wires = len(work_wires)

    if num_control_wires == 0 or num_work_wires < num_control_wires - 1:
        return False

    *_, depth = _calculate_select_swap_sizes(
        len(bitstrings), num_control_wires, len(target_wires), num_work_wires
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
        num_bitstrings, num_control_wires, num_targets, num_work_wires
    )
    num_work_wires_select = max(0, num_control_wires_select - 1)
    num_columns = int(np.ceil(num_bitstrings / depth))
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
            len(bitstrings), len(control_wires), len(target_wires), len(work_wires)
        )
    )

    swap_work_wires = work_wires[:num_work_wires_swap]
    select_work_wires = work_wires[num_work_wires_swap : len(work_wires) - num_work_wires_cswap]
    cswap_work_wires = work_wires[len(work_wires) - num_work_wires_cswap :]
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


@register_resources(_qrom_load_to_qrom_resources)
def _qrom_load_to_qrom(bitstrings, control_wires, target_wires, work_wires, clean):
    # pylint: disable=unused-argument
    QROM(bitstrings, control_wires, target_wires, work_wires)


add_decomps(QROMLoad, _select_swap, _qrom_load_to_qrom)
