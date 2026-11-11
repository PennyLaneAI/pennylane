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

"""Built-in decomposition rules for templates.subroutines.arithmetic.out_square."""

from collections import defaultdict

from pennylane.decomposition import add_decomps, register_condition, register_resources
from pennylane.decomposition_rules.templates.subroutines.arithmetic.out_multiplier import (
    _c_add_sub,
    _c_add_sub_resources,
)
from pennylane.decomposition_rules.templates.subroutines.arithmetic.semi_adder import (
    _semi_adder,
    _semi_adder_resources,
)
from pennylane.ops import CNOT
from pennylane.templates.subroutines.arithmetic.out_square import OutSquare
from pennylane.templates.subroutines.arithmetic.semi_adder import (
    SemiAdder,
    _self_ctrl_one_sparse_add,
    _self_ctrl_one_sparse_add_resources,
)
from pennylane.templates.subroutines.arithmetic.temporary_and import TemporaryAND
from pennylane.templates.subroutines.multix import MultiX
from pennylane.typing import Bool, Wire
from pennylane.wires import Wires, WiresLike


def _out_square_with_adder_zeroed_condition(
    x_wires, output_wires, work_wires, output_wires_zeroed=False
) -> bool:
    if not output_wires_zeroed:
        return False
    n = len(x_wires)
    m = len(output_wires)
    # pylint: disable-next=protected-access
    return len(work_wires) >= OutSquare._min_work_wires(n, m, output_wires_zeroed)


def _out_square_with_adder_zeroed_resources(
    x_wires, output_wires, work_wires, output_wires_zeroed=False
) -> dict:
    # pylint: disable=unused-argument
    n = len(x_wires)
    m = len(output_wires)
    num_work_wires = len(work_wires)
    resources = defaultdict(int)
    # Copying of first bit is a CNOT, all other bits require a TemporaryAND
    resources[CNOT] += 1
    resources[TemporaryAND] = min(n - 1, m - 2)
    num_work_wires += 2  # Using the 1s and 2s output bits as work wires
    p = min(n, m // 2 + m % 2) - 1
    for i in range(1, p + 1):
        x_size = n - i
        y_size = min(m - 2 * i, n + 2 - i)
        # First self-controlled 1-sparse adder can copy the first input bit instead of recomputing
        # a temporary AND (Improvement #4 from Sec IIIA).
        for k, val in _self_ctrl_one_sparse_add_resources(
            x_size, y_size, num_work_wires, i == 1
        ).items():
            resources[k] += val

    return dict(resources)


@register_condition(_out_square_with_adder_zeroed_condition)
@register_resources(_out_square_with_adder_zeroed_resources, name="out_square_with_adder")
def _out_square_with_adder_zeroed(
    x_wires: WiresLike,
    output_wires: WiresLike,
    work_wires: WiresLike,
    output_wires_zeroed: bool = False,
):
    # pylint: disable=unused-argument
    n = len(x_wires)
    m = len(output_wires)
    x_wires = x_wires[::-1]
    output_wires = output_wires[::-1]
    work_wires = Wires.all_wires([work_wires, output_wires[:2]])

    # Copy x, controlled on the least significant bit (LSB) of x, to the output register,
    # which is in |0>. This can be reduced to a CNOT for the LSB and TemporaryANDs for
    # the other bits. The CNOT is performed at the very end, to use the 1s bit of the output as
    # work wire until then.
    num_elbows = min(n - 1, m - 2)
    copy_input = x_wires[1 : num_elbows + 1]
    copy_output = output_wires[2 : num_elbows + 2]
    for x_wire, out_wire in zip(copy_input, copy_output, strict=True):
        TemporaryAND([x_wires[0], x_wire, out_wire])

    p = min(n, m // 2 + (m % 2)) - 1
    for i in range(1, p + 1):
        # Perform specialized "self"-controlled addition with zeroed 2s input bit, using
        # sliced x_wires and output_wires.
        # First self-controlled 1-sparse adder can copy the first input bit instead of recomputing
        # a temporary AND (Improvement #4 from Sec IIIA).
        _self_ctrl_one_sparse_add(
            x_wires[i:], output_wires[2 * i : min(m, n + 2 + i)], work_wires, i == 1
        )

    CNOT([x_wires[0], output_wires[0]])  # First control-copy, delayed until end of decomp.


def _out_square_with_caddsub_condition(
    x_wires, output_wires, work_wires, output_wires_zeroed=False
) -> bool:
    # pylint: disable=unused-argument
    n = len(x_wires)
    m = len(output_wires)
    if output_wires_zeroed and (n == 1 or m == 1):
        # Just a single CNOT in the decomposition.
        return True
    return len(work_wires) >= m - 1


def _out_square_with_caddsub_resources(
    x_wires, output_wires, work_wires, output_wires_zeroed=False
) -> dict:
    # pylint: disable=unused-argument
    n = len(x_wires)
    m = len(output_wires)
    num_work_wires = len(work_wires)
    p = min(n - 1, m // 2)

    resources = defaultdict(int)

    # Controlled add-subtract loop
    for i in range(p):
        size = min(n - i, m - 2 * i - 1) if output_wires_zeroed else m - 2 * i - 1
        for key, value in _c_add_sub_resources(n - i - 1, size).items():
            resources[key] += value

    if output_wires_zeroed and p == 0:
        resources[CNOT] += 1
    else:
        skips = [1] + [2 * j for j in range(1, n)]
        sparse_adder_res = _semi_adder_resources(x_wires, output_wires, skip_input_pos=skips)
        for key, value in sparse_adder_res.items():
            resources[key] += value

    if n > 1 and m > 1:
        # Subtract 2 x_{[1:]}
        resources[MultiX(Bool[m - 1], Wire[m - 1])] += 2
        resources[SemiAdder(Wire[n - 1], Wire[m - 1], Wire[num_work_wires])] += 1

        if m > n:
            # Shifted addition: MultiX spans the respective output_wires (m-n) and x_wires (n-1)
            resources[MultiX(Bool[m - 1], Wire[m - 1])] += 2
            resources[SemiAdder(Wire[n - 1], Wire[m - n], Wire[num_work_wires])] += 1

    return dict(resources)


def _shifted_adder(x_wires, output_wires, work_wires):
    """Perform shifted addition y -> y + x - 2^n + 1.

    Wires are in PennyLane (big-endian) order.
    """
    all_ones = [True] * (len(x_wires) + len(output_wires))
    all_target_wires = Wires(x_wires) + Wires(output_wires)
    MultiX(all_ones, all_target_wires)
    SemiAdder(x_wires, output_wires, work_wires)
    MultiX(all_ones, all_target_wires)


@register_condition(_out_square_with_caddsub_condition)
@register_resources(_out_square_with_caddsub_resources)
def _out_square_with_caddsub(
    x_wires: WiresLike,
    output_wires: WiresLike,
    work_wires: WiresLike,
    output_wires_zeroed: bool = False,
):
    r"""This decomposition uses controlled add-subtract blocks, and three correction
    steps. See Sec. II for details.
    """
    n = len(x_wires)
    m = len(output_wires)
    p = min(n - 1, m // 2)

    for i in range(p):
        # ``i`` indexes from the LSB. In big-endian layout the LSB is the last wire.
        x_wire = x_wires[n - 1 - i]
        x_rest = x_wires[: n - 1 - i]
        if output_wires_zeroed:
            # Equivalent to little-endian slice `[2*i+1 : n+1+i]`
            _out_reg = output_wires[max(0, m - (n + 1 + i)) : m - (2 * i + 1)]
        else:
            _out_reg = output_wires[: m - (2 * i + 1)]
        _c_add_sub(x_wire, x_rest, _out_reg, work_wires)

    if output_wires_zeroed and p == 0:
        # output register is still zeroed, no need for a full adder. p=0 holds for n=1 or m=1
        # in both cases we just need a CNOT to copy the LSB of the input into the zeroed output.
        CNOT([x_wires[-1], output_wires[-1]])
    else:
        zeroed = [1] + [2 * j for j in range(1, n)]
        _semi_adder(x_wires, output_wires, work_wires, skip_input_pos=zeroed)

    if n > 1 and m > 1:
        _output = output_wires[:-1]
        output_ones = [True] * len(_output)

        MultiX(output_ones, _output)
        SemiAdder(x_wires[:-1], _output, work_wires)
        MultiX(output_ones, _output)

        # shifted addition
        if m > n:
            _shifted_adder(x_wires[1:], output_wires[: m - n], work_wires)


add_decomps(
    OutSquare,
    _out_square_with_adder_zeroed,
    _out_square_with_caddsub,
)
