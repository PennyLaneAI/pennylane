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
Private helpers for computing the Hamming weight of a register in place, with
:class:`~.TemporaryAND` and :class:`~.CNOT` gates.
"""

from pennylane.decomposition import register_resources
from pennylane.ops import CNOT, subcircuit

from .semi_adder import _left_block, _left_block_zeroed
from .temporary_and import TemporaryAND


def _hamming_weight_resources(wires, **_):
    # Level j adds floor(n / 2^j) bits, with a half adder if this number is even, i.e., if bit j
    # of n is zero. Summing over all levels gives n - popcount(n) adders, ell(n) - popcount(n) of
    # which are half adders, with ell(n) the number of bits of n.
    n = len(wires)
    num_full_adders = n - n.bit_length()
    num_half_adders = n.bit_length() - n.bit_count()
    return {
        TemporaryAND: num_full_adders + num_half_adders,
        CNOT: 5 * num_full_adders + num_half_adders,
    }


@subcircuit(wire_argnames=("wires", "work_wires"))
@register_resources(_hamming_weight_resources)
def _HammingWeight(wires, work_wires):
    r"""Compute the Hamming weight of ``wires`` in place.

    The bits are summed by their weight :math:`2^j`, starting with all inputs at weight
    :math:`1`. Each full adder adds two bits of weight :math:`2^j` onto the last one and writes
    their carry, of weight :math:`2^{j+1}`, onto a zeroed work wire. For an even number of bits
    of weight :math:`2^j`, a half adder adds the remaining one. Each adder uses one
    :class:`~.TemporaryAND`, for a total of ``n - popcount(n)`` work wires and ANDs, where
    ``n = len(wires)``. For example, for 6 wires we get:

    .. code-block:: text

                  ┌─── full adder ──┐ ┌─── full adder ──┐ ┌ half adder ┐ ┌─── full adder ──┐
        wires[0]: ─╭X────╭●───────╭●─────────────────────────────────────────────────────────┤
        wires[1]: ─╰●─╭●─│──╭●─╭●─│──────────────────────────────────────────────────────────┤
        wires[2]: ────│──│──│──│──│────╭X────╭●───────╭●─────────────────────────────────────┤
        wires[3]: ────│──│──│──│──│────╰●─╭●─│──╭●─╭●─│──────────────────────────────────────┤
        wires[4]: ────│──│──│──│──│───────│──│──│──│──│────╭●─╭●─────────────────────────────┤
        wires[5]: ────╰X─├●─│──╰X─╰X──────╰X─├●─│──╰X─╰X───├●─╰X─────────────────────────────┤ bit 0
         work[0]: ───────│──│────────────────│──│──────────╰⊕────────────────╭X─╭●────╭X─╭X──┤ bit 1
         work[1]: ───────│──│────────────────│──│────────────────────────────│──├⊕─╭X─│──│───┤ bit 2
         work[2]: ───────╰⊕─╰X───────────────│──│─────────────────────────╭X─│──╰●─│──│──╰●──┤
         work[3]: ───────────────────────────╰⊕─╰X────────────────────────╰●─╰●────╰●─╰●─────┤

    The gates with a ``⊕`` target are :class:`~.TemporaryAND` gates, all others are
    :class:`~.CNOT` gates.

    Bit :math:`0` of the Hamming weight ends up on ``wires[-1]``, and bit :math:`j>0` on
    ``work_wires[j - 1]``. All other wires are left in garbage states, so that this helper
    method must be used in conjunction with its adjoint to return the system wires to their previous
    state, and only specific operations (e.g. phases) are allowed between ``_HammingWeight``
    and its uncomputation.

    Args:
        wires (Sequence): the ``n`` input wires
        work_wires (Sequence): at least ``n - popcount(n)`` work wires in the zero state
    """
    # Each bit of the Hamming weight above bit 0 needs one level of adders. The last carry of
    # level j is bit j + 1, so it goes onto out_wires[j], all other carries go onto spare wires
    log_n = len(wires).bit_length() - 1
    out_wires, spare_wires = work_wires[:log_n], work_wires[log_n:]
    for j in range(log_n):
        # Add all bits of weight 2^j onto the last one, and collect their carries, which are
        # the bits of weight 2^(j+1) for the next level
        *addends, total = wires
        num_adders = len(wires) // 2
        carry_wires = [*spare_wires[: num_adders - 1], out_wires[j]]
        spare_wires = spare_wires[num_adders - 1 :]
        for i, carry in enumerate(carry_wires):
            a = addends[2 * i]
            if 2 * i + 1 < len(addends):
                # Full adder: compute the carry of a + c + total, and leave the sum bit on total
                c = addends[2 * i + 1]
                _left_block([c, a, total, carry])
                CNOT([c, total])
            else:
                # For an even number of bits, a is left over and only needs a half adder
                _left_block_zeroed([a, total, carry])
            CNOT([a, total])
        wires = carry_wires
