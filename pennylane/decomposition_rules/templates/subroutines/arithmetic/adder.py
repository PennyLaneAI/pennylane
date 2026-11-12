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

"""Built-in decomposition rules for templates.subroutines.arithmetic.adder."""

from collections import defaultdict

from pennylane.decomposition import add_decomps, register_resources
from pennylane.ops import CNOT, ChangeOpBasis2, MultiControlledX, PauliX
from pennylane.ops.op_math import change_op_basis
from pennylane.ops.op_math.adjoint import adjoint
from pennylane.templates.subroutines.arithmetic.adder import Adder
from pennylane.templates.subroutines.arithmetic.phase_adder import PhaseAdder
from pennylane.templates.subroutines.qft import QFT
from pennylane.typing import Int, Wire
from pennylane.wires import Wires, WiresLike, concatenate_wires


def _increment(wires, control=()):
    """Carry-ripple ``+1`` (mod ``2**len(wires)``) on big-endian ``wires``, optionally controlled.

    The bits are flipped from most to least significant so each control still sees the original
    values of the lower bits. Every multi-controlled ``X`` borrows the higher-significance wires
    (already processed in this ripple, hence uninvolved) as dirty work wires, which are left unchanged.
    """
    wires = Wires(wires)
    control = Wires(control)
    n = len(wires)
    for i in range(n):
        controls = wires[i + 1 :] + control
        if len(controls) == 0:
            PauliX(wires[i])
        else:
            MultiControlledX(
                wires=controls + wires[i : i + 1],
                work_wires=wires[:i],
                work_wire_type="borrowed",
            )


def _add_constant(k, wires, control=()):
    """Add the constant ``k`` (mod ``2**len(wires)``) to big-endian ``wires``, optionally controlled.

    Each set bit of ``k`` at position ``j`` is a ``+1`` acting on the top ``len(wires) - j`` wires.
    A subtraction is requested by passing a negative ``k`` (added as its two's complement).
    """
    wires = Wires(wires)
    control = Wires(control)
    n = len(wires)
    k %= 2**n
    for j in range(n):
        if (k >> j) & 1:
            _increment(wires[: n - j], control)


def _adder_decomposition_resources(
    k: int, x_wires: WiresLike, mod: int, work_wires: WiresLike
) -> dict:
    # pylint: disable=unused-argument
    num_x_wires = len(x_wires)
    num_qft_wires = num_x_wires if mod == 2**num_x_wires else 1 + num_x_wires
    _compute_op = QFT(Wire[num_qft_wires])
    resources = {
        ChangeOpBasis2(
            _compute_op,
            PhaseAdder(Int, Wire[num_qft_wires], mod, Wire[int(mod != 2**num_qft_wires)]),
            adjoint(_compute_op),
        ): 1,
    }
    return resources


@register_resources(_adder_decomposition_resources)
def _adder_decomposition(k: int, x_wires: WiresLike, mod: int, work_wires: WiresLike):
    if mod == 2 ** len(x_wires):
        change_op_basis(QFT(x_wires), PhaseAdder(k, x_wires, mod))
        return

    qft_wires = concatenate_wires(work_wires[:1], x_wires)
    change_op_basis(QFT(qft_wires), PhaseAdder(k, qft_wires, mod, work_wires[1:2]))


def _increment_resources(num_wires: int, num_control: int = 0) -> dict:
    """Gate counts for :func:`_increment` acting on ``num_wires`` wires."""
    counts = defaultdict(int)
    for i in range(num_wires):
        num_controls = (num_wires - 1 - i) + num_control
        if num_controls == 0:
            counts[PauliX] += 1
        else:
            counts[MultiControlledX(Wire[num_controls + 1], work_wires=Wire[i])] += 1
    return dict(counts)


def _add_constant_resources(num_wires: int, num_control: int = 0) -> dict:
    """Upper-bound gate counts for an ``_add_constant`` on ``num_wires`` wires.

    The estimate is taken at the worst case where every bit is set, so it is independent of the
    actual constant and covers every gate type that any constant could emit.
    """
    counts = defaultdict(int)
    for size in range(1, num_wires + 1):
        for rep, count in _increment_resources(size, num_control).items():
            counts[rep] += count
    return dict(counts)


def _adder_arithmetic_resources(
    k: int, x_wires: WiresLike, mod: int, work_wires: WiresLike
) -> dict:
    # pylint: disable=unused-argument
    num_x_wires = len(x_wires)
    counts = defaultdict(int)
    if mod == 2**num_x_wires:
        return dict(_add_constant_resources(num_x_wires))

    # The general-modulus circuit augments the register with one extra high bit and uses four
    # uncontrolled add/subtract blocks, one flag-controlled addition of ``mod``, and a little
    # flag bookkeeping (see _adder_arithmetic_decomposition).
    aug = num_x_wires + 1
    for rep, count in _add_constant_resources(aug).items():
        counts[rep] += 4 * count
    for rep, count in _add_constant_resources(aug, num_control=1).items():
        counts[rep] += count
    counts[CNOT] += 2
    counts[PauliX] += 2
    return dict(counts)


@register_resources(_adder_arithmetic_resources, exact=False)
def _adder_arithmetic_decomposition(k: int, x_wires: WiresLike, mod: int, work_wires: WiresLike):
    x_wires = Wires(x_wires)
    work_wires = Wires(work_wires)
    n = len(x_wires)

    if mod == 2**n:
        _add_constant(k, x_wires)
        return

    # Modular reduction follows the modular adder of Beauregard, sec. 2.2 (arXiv:quant-ph/0205095),
    # with the Fourier additions replaced by computational-basis incrementers: augment the register
    # with a spare high bit so x + k cannot wrap, then use a flag wire to conditionally undo an
    # over-subtraction of ``mod`` (the flag is uncomputed via his identity (a+k) mod m >= k <=> a+k < m).
    aug = work_wires[:1] + x_wires
    flag = work_wires[1:2]
    msb = aug[:1]
    k %= mod

    _add_constant(k, aug)  # aug <- x + k
    _add_constant(-mod, aug)  # aug <- x + k - mod (msb is the sign bit)
    CNOT(wires=msb + flag)  # flag = 1 iff x + k < mod
    _add_constant(mod, aug, control=flag)  # add mod back when flag is set
    _add_constant(-k, aug)  # re-expose the branch in msb ...
    PauliX(msb[0])
    CNOT(wires=msb + flag)  # ... to reset flag to 0
    PauliX(msb[0])
    _add_constant(k, aug)  # aug <- (x + k) mod mod, work wires restored


add_decomps(Adder, _adder_decomposition, _adder_arithmetic_decomposition)
