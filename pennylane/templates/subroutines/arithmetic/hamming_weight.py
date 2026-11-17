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

from pennylane import capture
from pennylane.core.operator import abstractify
from pennylane.core.queuing import AnnotatedQueue
from pennylane.ops import CNOT
from pennylane.ops.op_math import prod

from .semi_adder import _left_block, _left_block_zeroed


def _hamming_weight_resource_rep(num_wires):
    """Abstract product of the gates of :func:`~._hamming_weight` on ``num_wires`` input wires,
    e.g., to describe its use as the compute block of :func:`~.change_op_basis` in resource
    functions."""
    num_work_wires = num_wires - num_wires.bit_count()
    with capture.pause(), AnnotatedQueue() as q:
        _hamming_weight(range(num_wires), range(num_wires, num_wires + num_work_wires))
    # The factors of a product are in matrix order, the reverse of the order of application
    return prod(*(abstractify(op) for op in reversed(q.queue)))


def _hamming_weight(wires, work_wires):
    r"""Compute the Hamming weight of ``wires`` in place.

    The bits are summed by their weight :math:`2^j`, starting with all inputs at weight
    :math:`1`. Each full adder adds two bits of weight :math:`2^j` onto the last one and writes
    their carry, of weight :math:`2^{j+1}`, onto a zeroed work wire. For an even number of bits
    of weight :math:`2^j`, a half adder adds the remaining one. Each adder uses one
    :class:`~.TemporaryAND`, for a total of ``n - popcount(n)`` work wires and ANDs, where
    ``n = len(wires)``.

    Bit :math:`0` of the Hamming weight ends up on ``wires[-1]``, and bit :math:`j>0` on
    ``work_wires[j - 1]``. All other wires are left in garbage states, so that callers uncompute
    the circuit, e.g., by using it as the compute block of :func:`~.change_op_basis`.

    Args:
        wires (Sequence): the input wires
        work_wires (Sequence): at least ``n - popcount(n)`` work wires in the zero state

    **Example**

    For three bits, a single full adder leaves the sum on wire ``2`` and the carry on the work
    wire ``3``:

    >>> from pennylane.templates.subroutines.arithmetic.hamming_weight import _hamming_weight
    >>> qp.tape.make_qscript(_hamming_weight)([0, 1, 2], [3]).operations
    [CNOT(wires=[1, 0]), CNOT(wires=[1, 2]), TemporaryAND(wires=[0, 2, 3]), CNOT(wires=[1, 3]),
     CNOT(wires=[1, 2]), CNOT(wires=[0, 2])]
    """
    num_bits = len(wires).bit_length()
    out_wires, spare_wires = work_wires[: num_bits - 1], work_wires[num_bits - 1 :]
    for j in range(num_bits - 1):
        *addends, total = wires
        num_adders = len(wires) // 2
        carry_wires = [*spare_wires[: num_adders - 1], out_wires[j]]
        spare_wires = spare_wires[num_adders - 1 :]
        for i, carry in enumerate(carry_wires):
            a = addends[2 * i]
            if 2 * i + 1 < len(addends):
                c = addends[2 * i + 1]
                _left_block([c, a, total, carry])
                CNOT([c, total])
            else:
                _left_block_zeroed([a, total, carry])
            CNOT([a, total])
        wires = carry_wires
