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

"""Built-in decomposition rules for templates.subroutines.arithmetic.left_classical_comparator."""

from pennylane import capture, compiler, math
from pennylane.control_flow import for_loop
from pennylane.decomposition import add_decomps, register_resources
from pennylane.ops import CNOT, X, cond
from pennylane.templates.subroutines.arithmetic.left_classical_comparator import (
    LeftClassicalComparator,
)
from pennylane.templates.subroutines.arithmetic.temporary_and import Elbow
from pennylane.wires import Wires


def _get_specific_bit(value, i):
    # returns the i-th bit of the binary representation of value in little endian convention.
    return (value >> i) & 1


def _left_classical_comparator_resources(x_wires, value, target_wire, work_wires, comparator, **_):
    """Return conservative worst-case gate counts without knowing the runtime threshold."""
    # pylint: disable=unused-argument
    n = len(x_wires)
    # Count every conditional gate: two Xs for bit 0, four per remaining low bit,
    # and one for bit n. The final negation depends only on the comparator.
    return {
        Elbow: n - 1,
        CNOT: n,
        X: 4 * n - 1 + int(comparator in [">", ">="]),
    }


@register_resources(_left_classical_comparator_resources, exact=False)
def _left_classical_comparator(x_wires, value, target_wire, work_wires, comparator, **_):
    x_wires = x_wires[::-1]

    def _negate_output():
        X(wires=target_wire)

    if comparator in ["<=", ">"]:
        value += 1

    used_work_wires = Wires.all_wires([work_wires[: len(x_wires) - 1], target_wire])

    bit = _get_specific_bit(value, 0)
    cond(bit, X)(wires=[x_wires[0]])
    cond(bit, CNOT)(wires=[x_wires[0], used_work_wires[0]])
    cond(bit, X)(wires=[x_wires[0]])

    if compiler.active() or capture.enabled():
        x_wires = math.array(x_wires, like="jax")
        used_work_wires = math.array(used_work_wires, like="jax")

    @for_loop(1, len(x_wires))
    def _loop(i):
        bit = _get_specific_bit(value, i)
        cond(bit, X)(wires=[x_wires[i]])
        cond(bit, X)(wires=[used_work_wires[i - 1]])
        Elbow(wires=[used_work_wires[i - 1], x_wires[i], used_work_wires[i]])
        cond(bit, X)(wires=[used_work_wires[i - 1]])
        CNOT(wires=[used_work_wires[i - 1], used_work_wires[i]])
        cond(bit, X)(wires=[x_wires[i]])

    _loop()  # pylint: disable=no-value-for-parameter

    # Degenerate bound: when the effective ``value`` equals ``2 ** len(x_wires)`` its
    # low ``n`` bits are all zero, so the bitwise core above computes the constant
    # ``x < 0`` (always False) rather than the intended "always True". Bit ``n`` of
    # ``value`` is set exactly in this case, so a single conditional ``X`` fixes the
    # target wire.
    bit_n = _get_specific_bit(value, len(x_wires))
    cond(bit_n, X)(wires=[used_work_wires[len(x_wires) - 1]])

    cond(comparator.startswith(">"), _negate_output)()


add_decomps(LeftClassicalComparator, _left_classical_comparator)
