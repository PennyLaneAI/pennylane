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

"""Built-in decomposition rules for templates.subroutines.arithmetic.signed_out_square."""

from collections import defaultdict
from types import SimpleNamespace

from pennylane.decomposition import add_decomps, register_resources
from pennylane.decomposition_rules.templates.subroutines.arithmetic.semi_adder import (
    _controlled_semi_adder,
    _ctrl_semi_adder_resource,
)
from pennylane.ops import X
from pennylane.templates.subroutines.arithmetic import OutSquare, SemiAdder
from pennylane.templates.subroutines.arithmetic.signed_out_square import SignedOutSquare
from pennylane.templates.subroutines.multix import MultiX
from pennylane.typing import Bool, Wire
from pennylane.wires import WiresLike


def _c_subtract_then_add_one_resources(n, m, num_work_wires, output_wires_zeroed):
    size = min(m - n, n) if output_wires_zeroed else m - n
    add_base = SemiAdder(Wire[n - 1], Wire[size], Wire[size - 1])
    cadd_resources = _ctrl_semi_adder_resource(
        add_base, Wire[1], None, Wire[num_work_wires - size + 1], "zeroed"
    )

    # Bit flips on input register
    if n - 1 > 1:
        basis_rep = MultiX(Bool[n - 2], Wire[n - 2])
        cadd_resources[basis_rep] = cadd_resources.get(basis_rep, 0) + 2

    # Bit flips on output and work registers. The work-wire pair only occurs if there is a ladder
    # elbow to inject it around, i.e. if the underlying adder has more than one ``y`` wire.
    cadd_resources[X] = cadd_resources.get(X, 0) + (2 + 2 * (num_work_wires > 0 and size > 1))
    return cadd_resources


def _c_subtract_then_add_one(c_wire, x_wires, y_wires, work_wires):
    """Subtract x from y, controlled on c_wire."""
    # Flip input bits (except for the LSB, which would be flipped back by the input carry set)
    if len(x_wires) > 1:
        MultiX([1] * (len(x_wires) - 1), x_wires[:-1])
    # Flip LSB of output register, due to input carry being set
    X(y_wires[-1])

    m = len(y_wires)
    # `_controlled_semi_adder` only reads x/y/work_wires from `base`
    # using SimpleNamespace to avoid having to use capture.pause() and stop_recording()
    base = SimpleNamespace(x_wires=x_wires, y_wires=y_wires, work_wires=work_wires[: m - 1])

    # Inject a work wire bit flip where a carry-in qubit would cause one, i.e. after the very
    # first left elbow and before the last right elbow.
    carry_flip = X if len(work_wires) > 0 else None
    _controlled_semi_adder(
        base,
        control_wires=[c_wire],
        work_wires=work_wires[m - 1 :],
        work_wire_type="zeroed",
        carry_flip=carry_flip,
    )

    # Flip LSB of output register, due to input carry being set
    X(y_wires[-1])
    # Flip input bits (except for the LSB, which would be flipped back by the input carry set)
    if len(x_wires) > 1:
        MultiX([1] * (len(x_wires) - 1), x_wires[:-1])


def _signed_out_square_resources(
    x_wires, output_wires, work_wires, output_wires_zeroed=False
) -> dict:
    # pylint: disable=unused-argument
    n = len(x_wires)
    m = len(output_wires)
    num_work_wires = len(work_wires)
    resources = defaultdict(int)

    square_op = OutSquare(
        Wire[n - 1], Wire[m], Wire[num_work_wires], output_wires_zeroed=output_wires_zeroed
    )
    resources[square_op] += 1

    if n < m:
        # Add x_s 2^n (2^{n-1}-x)
        _res = _c_subtract_then_add_one_resources(n, m, num_work_wires, output_wires_zeroed)
        for key, value in _res.items():
            resources[key] += value

        if m >= 2 * n - 1:
            # Subtract x_s 2^{2n-2}
            size = min(m - (2 * n - 2), 2) if output_wires_zeroed else m - (2 * n - 2)
            resources[X] += 2 * size
            add_rep = SemiAdder(Wire[1], Wire[size], Wire[num_work_wires])
            resources[add_rep] += 1

    return dict(resources)


@register_resources(_signed_out_square_resources, name="signed_square_from_unsigned_square")
def _signed_out_square(
    x_wires: WiresLike,
    output_wires: WiresLike,
    work_wires: WiresLike,
    output_wires_zeroed: bool,
):
    """Implement signed squaring in three steps: Unsigned squaring, controlled subtraction
    (with input carry), and a single-bit subtraction.
    See the documentation of SignedOutSquare for details."""
    n = len(x_wires)
    m = len(output_wires)
    # Compute (x_u)^2 into output register
    OutSquare(x_wires[1:], output_wires, work_wires, output_wires_zeroed=output_wires_zeroed)
    if m > n:
        # Add x_s * 2^n (2^(n-1) - x_u)
        # For output_wires_zeroed=True, note that there are n-1 bits in x_u, so the squaring
        # above can only have produced numbers smaller than 2^(2n-2). We here add a number that
        # is at most 2^(2n-1), so that the output is guaranteed to be smaller than 2^(2n) and
        # we can restrict the subroutine to the 2n LSB of the output.
        # As the wire ordering is big-endian, slicing off the n LSBs (to multiply by 2^n) is done
        # by setting the upper limit of the output wires to `m-n`.
        output_msb = max(0, m - 2 * n) if output_wires_zeroed else 0
        output = output_wires[output_msb : m - n]
        _c_subtract_then_add_one(x_wires[0], x_wires[1:], output, work_wires)

        if m >= 2 * n - 1:
            # Subtract x_s * 2^(2n-2).
            # As the wire ordering is big-endian, slicing off the n LSBs (to multiply by 2^(2n-2))
            # is done by setting the upper limit of the output wires to `m-(2n-2)`.
            output = output_wires[output_msb : m - (2 * n - 2)]
            _ = [X(w) for w in output]
            SemiAdder(x_wires[:1], output, work_wires)
            _ = [X(w) for w in output]


add_decomps(SignedOutSquare, _signed_out_square)
