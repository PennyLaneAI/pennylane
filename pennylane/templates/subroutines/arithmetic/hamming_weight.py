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

from pennylane.ops import CNOT

from .temporary_and import TemporaryAND


def _full_adder(a, b, c, t):
    """Gates of a full adder on the positions ``a``, ``b`` and ``c``, with the zeroed work
    position ``t``. Afterwards, ``b`` holds the sum and ``t`` the carry, ``a`` holds garbage and
    ``c`` is unchanged."""
    return [
        (CNOT, (c, a)),
        (CNOT, (c, b)),
        (TemporaryAND, (a, b, t)),
        (CNOT, (c, t)),
        (CNOT, (a, b)),
        (CNOT, (c, b)),
    ]


def _half_adder(a, b, t):
    """Gates of a half adder on the positions ``a`` and ``b``, with the zeroed work position
    ``t``. Afterwards, ``b`` holds the sum and ``t`` the carry, and ``a`` is unchanged."""
    return [(TemporaryAND, (a, b, t)), (CNOT, (a, b))]


def _hamming_weight_plan(n):
    r"""Plan a circuit that computes the Hamming weight of ``n`` bits in place.

    The bits are sorted into buckets by their weight :math:`2^j`, starting with all inputs in
    bucket :math:`0`. While a bucket holds at least three entries, a full adder replaces three of
    them by their sum, and moves their carry to the next bucket. A half adder does the same for
    the last two entries. Each adder uses one :class:`~.TemporaryAND` with a fresh work wire as
    target, for a total of ``n - popcount(n)`` ANDs and work wires. The remaining entry of bucket
    :math:`j` is bit :math:`j` of the Hamming weight. All other wires are left in garbage states,
    so that callers uncompute the circuit, e.g., by using it as the compute block of
    :func:`~.change_op_basis`.

    Args:
        n (int): number of input bits

    Returns:
        tuple[list[tuple[type, tuple[int]]], list[int], int]: ``(gates, out_positions,
        num_work)``. ``gates`` is the list of gates in order of application, each given as a
        gate type (:class:`~.TemporaryAND` or :class:`~.CNOT`) and the positions of its wires.
        ``out_positions`` are the positions holding the bits of the Hamming weight, least
        significant bit first, and ``num_work`` is the number of work wires. The positions
        refer to the register ``input_wires + work_wires``, i.e., ``0, ..., n-1`` are the inputs
        and ``n, ..., n+num_work-1`` are the work wires.

    **Example**

    For three bits, a single full adder leaves the sum on position ``1`` and the carry on the
    work position ``3``:

    >>> from pennylane.templates.subroutines.arithmetic.hamming_weight import _hamming_weight_plan
    >>> gates, out_positions, num_work = _hamming_weight_plan(3)
    >>> [(op.__name__, positions) for op, positions in gates]
    [('CNOT', (2, 0)), ('CNOT', (2, 1)), ('TemporaryAND', (0, 1, 3)),
     ('CNOT', (2, 3)), ('CNOT', (0, 1)), ('CNOT', (2, 1))]
    >>> out_positions, num_work
    ([1, 3], 1)
    """
    gates, out_positions, num_work = [], [], 0
    bucket = list(range(n))
    while bucket:
        carries = []
        while len(bucket) > 1:
            t = n + num_work
            num_work += 1
            if len(bucket) > 2:
                (a, b, c), bucket = bucket[:3], bucket[3:]
                gates += _full_adder(a, b, c, t)
            else:
                (a, b), bucket = bucket, []
                gates += _half_adder(a, b, t)
            bucket.append(b)
            carries.append(t)
        out_positions.append(bucket[0])
        bucket = carries
    return gates, out_positions, num_work


def _apply_hamming_weight(plan, wires):
    """Apply the gates of a plan from :func:`~._hamming_weight_plan`.

    Args:
        plan (tuple): the output of :func:`~._hamming_weight_plan`
        wires (Sequence): the register that the positions of the plan refer to, i.e., the input
            wires followed by at least ``num_work`` work wires in the zero state
    """
    for op, positions in plan[0]:
        op([wires[p] for p in positions])
