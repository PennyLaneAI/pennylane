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


"""Built-in decomposition rules for qubit.arithmetic_ops."""

from collections import Counter

from pennylane import ops
from pennylane.decomposition import (
    add_decomps,
    register_condition,
    register_resources,
    resource_rep,
)
from pennylane.decomposition.symbolic_decomposition import pow_involutory
from pennylane.decomposition.symbolic_decomposition import self_adjoint_legacy
from pennylane.ops.qubit.arithmetic_ops import (
    IntegerComparator,
    QubitCarry,
    QubitSum,
)
from pennylane.typing import Wire
from pennylane.wires import WiresLike


@register_resources({ops.CNOT: 1, ops.Toffoli: 2}, name="to_cnot_toffolis")
def qubitcarry_to_cnot_toffolis(wires: WiresLike, **__):
    """Decompose ``QubitCarry`` using the ``to_cnot_toffolis`` strategy."""
    ops.Toffoli(wires=wires[1:])
    ops.CNOT(wires=[wires[1], wires[2]])
    ops.Toffoli(wires=[wires[0], wires[2], wires[3]])


add_decomps(QubitCarry, qubitcarry_to_cnot_toffolis)


@register_resources({ops.CNOT: 2}, name="to_cnots")
def qubitsum_to_cnots(wires: WiresLike, **__):
    """Decompose ``QubitSum`` using the ``to_cnots`` strategy."""
    ops.CNOT(wires=[wires[1], wires[2]])
    ops.CNOT(wires=[wires[0], wires[2]])


add_decomps(QubitSum, qubitsum_to_cnots)
add_decomps("Adjoint(QubitSum)", self_adjoint_legacy)
add_decomps("Pow(QubitSum)", pow_involutory)


def _integer_comparator_less_than_resources(num_wires, value, num_work_wires, **_):

    if value == 0:
        return {}

    if value > 2 ** (num_wires - 1) - 1:
        return {ops.X: 1}

    num_controls = num_wires - 1
    binary_str = format(value, f"0{num_controls}b")
    last_significant = binary_str.rfind("1")
    gate_counts = {ops.X: (last_significant + 1) * 2}

    first_significant = binary_str.find("1")
    gate_counts[
        ops.MultiControlledX(
            Wire[first_significant + 2],
            work_wires=Wire[num_work_wires + num_wires - 2 - first_significant],
        )
    ] = 1

    while (first_significant := binary_str.find("1", first_significant + 1)) != -1:
        gate_counts[
            ops.MultiControlledX(
                Wire[first_significant + 2],
                work_wires=Wire[num_work_wires + num_wires - 2 - first_significant],
            )
        ] = 1

    return gate_counts


@register_condition(lambda geq, **_: not geq)
@register_resources(_integer_comparator_less_than_resources, name="less_than")
def integer_comparator_less_than(wires, value, work_wires, **_):
    """Decompose the ``IntegerComparator`` for when the flipping condition is ``n < value``.

    This decomposition uses the minimum number of ``MultiControlledX`` gates. For a given value,
    we first convert it to binary, and iteratively look for the significant bits. For example,
    with 6 control wires, if the value is 22, which is 010110 in 6-bit binary, we observe
    that all 6-bit numbers that start with 00 will satisfy the flipping condition, so we apply
    a ``MultiControlledX`` with only the first two wires as controls, and 00 as the control values.
    Then we look for the next significant bit, and observe that 22 starts with 0101. Therefore,
    all 6-bit numbers that start with 0100 will also satisfy the flipping condition, so we apply
    a ``MultiControlledX`` with the first four wires as controls, and 0100 as the control values.
    This continues until we add a ``MultiControlledX`` for every significant bit in the value.

    .. code-block:: pycon

        0: ─╭○─╭○─╭○─┤
        1: ─├○─├●─├●─┤
        2: ─│──├○─├○─┤
        3: ─│──├○─├●─┤
        4: ─│──│──├○─┤
        6: ─╰X─╰X─╰X─┤

    If we decompose this circuit one level further, we get

    .. code-block:: pycon

        0: ──X─╭●──X──X─╭●──X──X─╭●──X─┤
        1: ──X─├●──X────├●───────├●────┤
        2: ────│───X────├●──X──X─├●──X─┤
        3: ────│───X────├●──X────├●────┤
        4: ────│────────│───X────├●──X─┤
        6: ────╰X───────╰X───────╰X────┤

    And we observe that the ``PauliX`` gates used to flip the control values can be merged:

    .. code-block:: pycon

        0: ──X─╭●────╭●────╭●──X─┤
        1: ──X─├●──X─├●────├●────┤
        2: ──X─│─────├●────├●──X─┤
        3: ──X─│─────├●──X─├●────┤
        4: ──X─│─────│─────├●──X─┤
        6: ────╰X────╰X────╰X────┤

    """
    if value == 0:
        return

    num_controls = len(wires) - 1

    # If the value is larger than the maximum value that can be represented by
    # the number of control bits, the flipping condition is always satisfied, in
    # which case we apply an X gate to the target wire and terminate.
    if value > 2**num_controls - 1:
        ops.X(wires[-1])
        return

    # Track which control bits have been flipped back
    control_value_tracker = [0] * num_controls

    # First apply X to all wires until the last significant bit to flip control values to 1.
    binary_str = format(value, f"0{num_controls}b")
    last_significant = binary_str.rfind("1")
    for i in range(last_significant + 1):
        ops.X(wires[i])

    # The flipping condition is satisfied if all bits from the first bit to the
    # first non-zero bit of the value are zeroes.
    first_significant = binary_str.find("1")
    ops.MultiControlledX(
        wires=wires[: first_significant + 1] + wires[-1:],
        work_wires=wires[first_significant + 1 : -1] + work_wires,
    )
    control_value_tracker[first_significant] = 1
    ops.X(wires[first_significant])

    # If the wire corresponding to the first significant bit of the value is 1, then we
    # iteratively look for the next significant bit, and apply a flip conditioned on all
    # bits from the last significant bit to this next significant bit being zeroes.
    while (first_significant := binary_str.find("1", first_significant + 1)) != -1:
        ops.MultiControlledX(
            wires=wires[: first_significant + 1] + wires[-1:],
            work_wires=wires[first_significant + 1 : -1] + work_wires,
        )
        control_value_tracker[first_significant] = 1
        ops.X(wires[first_significant])

    for i in range(last_significant + 1):
        if control_value_tracker[i] == 0:
            ops.X(wires[i])


def _integer_comparator_greater_equal_resources(num_wires, value, num_work_wires, **_):

    # If the value is 0, the flipping condition is always satisfied.
    if value == 0:
        return {ops.X: 1}

    num_controls = num_wires - 1

    if value > 2**num_controls - 1:
        return {}

    binary_str = format(value, f"0{num_controls}b")
    first_zero = binary_str.find("0")

    if first_zero == -1:
        return {ops.MultiControlledX(Wire[num_controls + 1], work_wires=Wire[num_work_wires]): 1}

    gate_set = Counter()

    gate_set[
        ops.MultiControlledX(
            Wire[first_zero + 2], work_wires=Wire[num_work_wires + num_wires - 2 - first_zero]
        )
    ] = 1
    gate_set[ops.X] = 2

    while (first_zero := binary_str.find("0", first_zero + 1)) != -1:
        gate_set[
            ops.MultiControlledX(
                Wire[first_zero + 2], work_wires=Wire[num_work_wires + num_wires - 2 - first_zero]
            )
        ] = 1
        gate_set[ops.X] += 2

    gate_set[ops.MultiControlledX(Wire[num_controls + 1], work_wires=Wire[num_work_wires])] += 1

    return dict(gate_set)


@register_condition(lambda geq, **_: geq)
@register_resources(_integer_comparator_greater_equal_resources, name="greater_equal")
def integer_comparator_greater_equal(wires, value, work_wires, **_):
    """Decompose the ``IntegerComparator`` for when the flipping condition is ``n >= value``.

    This decomposition rule mirrors the implementation for the ``n < value`` case.

    """
    if value == 0:
        ops.X(wires[-1])
        return

    num_controls = len(wires) - 1

    # If the value is larger than the maximum value that can be represented by
    # the number of control bits, the flipping condition is never satisfied,
    if value > 2**num_controls - 1:
        return

    # Track which control bits have been flipped
    control_value_tracker = [0] * num_controls

    binary_str = format(value, f"0{num_controls}b")
    first_zero = binary_str.find("0")

    if first_zero == -1:
        # If the value happens to be the all-one state, then we apply a single MCX
        ops.MultiControlledX(wires=wires, work_wires=work_wires)
        return

    ops.MultiControlledX(
        wires=wires[: first_zero + 1] + wires[-1:],
        work_wires=wires[first_zero + 1 : -1] + work_wires,
    )
    control_value_tracker[first_zero] = 1
    ops.X(wires[first_zero])

    while (first_zero := binary_str.find("0", first_zero + 1)) != -1:
        ops.MultiControlledX(
            wires=wires[: first_zero + 1] + wires[-1:],
            work_wires=wires[first_zero + 1 : -1] + work_wires,
        )
        control_value_tracker[first_zero] = 1
        ops.X(wires[first_zero])

    # The last MCX corresponds to the equal case.
    ops.MultiControlledX(wires=wires, work_wires=work_wires)

    for i in range(num_controls):
        if control_value_tracker[i]:
            ops.X(wires[i])


def _integer_comparator_flip_greater_equal_resources(num_wires, value, num_work_wires, geq, **_):
    """Resource estimation for flipping the geq condition."""
    return {
        ops.X: 1,
        resource_rep(
            IntegerComparator,
            num_wires=num_wires,
            value=value,
            geq=not geq,
            num_work_wires=num_work_wires,
        ): 1,
    }


@register_resources(_integer_comparator_flip_greater_equal_resources, name="flip_greater_equal")
def integer_comparator_flip_greater_equal(value, geq, wires, work_wires, **_):
    """Decompose the IntegerComparator by flipping geq to lt or vice versa."""
    ops.X(wires[-1])
    IntegerComparator(value, wires, geq=not geq, work_wires=work_wires)


add_decomps(
    IntegerComparator,
    integer_comparator_less_than,
    integer_comparator_greater_equal,
    integer_comparator_flip_greater_equal,
)
add_decomps("Adjoint(IntegerComparator)", self_adjoint_legacy)
