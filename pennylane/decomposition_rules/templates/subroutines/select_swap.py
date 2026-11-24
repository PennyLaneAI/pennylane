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


"""Built-in decomposition rules for templates.subroutines.select_swap."""

from pennylane import capture, math
from pennylane.decomposition import add_decomps, register_condition, register_resources
from pennylane.decomposition_rules.templates.subroutines.qrom_load import (
    _calculate_select_swap_sizes,
    _measure_and_reset,
    _measure_and_reset_resources,
    _phase_fixup,
    _phase_fixup_resources,
    _select_ops,
    _swap_ops,
)
from pennylane.ops import SWAP, ctrl
from pennylane.templates.subroutines.qrom_load import QROMLoad
from pennylane.templates.subroutines.select_swap import SelectSwap
from pennylane.typing import Bool, Wire
from pennylane.wires import Wires


def _select_swap_sizes(bitstrings, control_wires, target_wires, work_wires):
    return _calculate_select_swap_sizes(
        len(bitstrings), len(control_wires), len(target_wires), len(work_wires), clean=False
    )


def _select_swap_resources(bitstrings, control_wires, target_wires, work_wires):
    num_control_wires_select, num_work_wires_select, _, num_work_wires_cswap, depth = (
        _select_swap_sizes(bitstrings, control_wires, target_wires, work_wires)
    )
    num_targets = len(target_wires)
    num_columns = -(-len(bitstrings) // depth)
    select_rep = QROMLoad(
        Bool[num_columns, depth * num_targets],
        Wire[num_control_wires_select],
        Wire[depth * num_targets],
        Wire[min(num_work_wires_select, max(0, num_control_wires_select - 1))],
    )
    resources = {select_rep: 1}
    if depth > 1:
        cswap_rep = ctrl(
            SWAP(Wire[2]),
            control=Wire[1],
            work_wires=Wire[num_work_wires_cswap],
            work_wire_type="zeroed",
        )
        resources[cswap_rep] = num_targets * (depth - 1)
    return resources


@register_resources(_select_swap_resources, name="select_swap")
def select_swap(bitstrings, control_wires, target_wires, work_wires):
    num_control_wires_select, _, num_work_wires_swap, num_work_wires_cswap, depth = (
        _select_swap_sizes(bitstrings, control_wires, target_wires, work_wires)
    )

    swap_work_wires = work_wires[:num_work_wires_swap]
    num_work_wires_select = max(0, num_control_wires_select - 1)
    select_work_wires = work_wires[
        num_work_wires_swap : num_work_wires_swap + num_work_wires_select
    ]
    # The Select block returns its work wires to |0>, so the CSWAPs can borrow one of them.
    cswap_work_wires = work_wires[num_work_wires_swap : num_work_wires_swap + num_work_wires_cswap]
    swap_wires = Wires(target_wires) + Wires(swap_work_wires)

    _select_ops(
        bitstrings,
        depth,
        target_wires,
        swap_wires,
        control_wires[:num_control_wires_select],
        select_work_wires,
        zeroed_targets=True,
    )
    _swap_ops(control_wires[num_control_wires_select:], swap_wires, target_wires, cswap_work_wires)


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


def _adjoint_select_swap_measurement_resources(base):
    *_, depth = _select_swap_sizes(
        base.bitstrings, base.control_wires, base.target_wires, base.work_wires
    )
    num_targets = len(base.target_wires)
    num_entries = depth * -(-len(base.bitstrings) // depth)
    resources = _phase_fixup_resources(
        num_entries, len(base.control_wires), num_targets + len(base.work_wires)
    )
    for op, count in _measure_and_reset_resources(depth * num_targets).items():
        resources[op] += count
    return resources


@register_condition(lambda base: capture.enabled() and len(base.control_wires) > 0)
@register_resources(_adjoint_select_swap_measurement_resources, exact=False, name="measurement")
def adjoint_select_swap_measurement(base):
    r"""Uncomputation of SelectSwap with measurements and a phase fixup, following Appendix C of
    `arXiv:1902.02134 <https://arxiv.org/abs/1902.02134>`__.

    After ``SelectSwap``, the :math:`k` columns of :math:`n` wires made up of the target wires and
    the first :math:`n(k-1)` work wires hold the :math:`k` bitstrings that were loaded in
    parallel, permuted by the SWAP network depending on the least-significant address bits. All
    other work wires are in the :math:`|0\rangle` state.

    These :math:`nk` wires are measured in the :math:`X` basis and reset to :math:`|0\rangle`.
    With outcomes :math:`m_p` on column :math:`p`, this leaves the phase
    :math:`(-1)^{\sum_p m_p\cdot b_{\pi_i(p)}}` on each address :math:`i`, where
    :math:`b_{\pi_i(p)}` is the bitstring that ``SelectSwap`` moved into column :math:`p`.
    Unlike when inverting the SWAP network first, this phase depends on all address bits, and it
    is undone with the same phase fixup as in the measurement-based uncomputation of
    :class:`~.QROMLoad`, with all target and work wires as auxiliary wires.
    """
    bitstrings = math.cast(base.bitstrings, int)
    target_wires, work_wires = base.target_wires, base.work_wires
    num_bitstrings, num_targets = len(bitstrings), len(target_wires)
    _, _, num_work_wires_swap, _, depth = _select_swap_sizes(
        bitstrings, base.control_wires, target_wires, work_wires
    )
    num_columns = -(-num_bitstrings // depth)

    outcomes = _measure_and_reset(Wires(target_wires) + Wires(work_wires[:num_work_wires_swap]))
    outcomes = math.reshape(math.cast(math.stack(outcomes), int), (depth, num_targets))

    # parities[h, q, p] is the parity of bitstring q of column h when it is moved into position p
    parities = math.dot(bitstrings, math.transpose(outcomes))
    padding = math.zeros((num_columns * depth - num_bitstrings, depth), dtype=int, like=parities)
    parities = math.reshape(math.concatenate([parities, padding]), (num_columns, depth, depth))
    perms = _swap_permutations(depth)
    fixups = math.sum(parities[:, perms, math.arange(depth)], axis=-1) % 2

    _phase_fixup(
        math.reshape(fixups, (-1,)), base.control_wires, Wires(target_wires) + Wires(work_wires)
    )


add_decomps(SelectSwap, select_swap)


add_decomps("Adjoint(SelectSwap)", adjoint_select_swap_measurement)
