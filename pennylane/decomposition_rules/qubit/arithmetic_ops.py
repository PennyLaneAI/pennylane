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

# pylint: disable=arguments-differ

from collections import Counter

import pennylane as qp
from pennylane.decomposition import (
    add_decomps,
    register_condition,
    register_resources,
    resource_rep,
)
from pennylane.decomposition_rules.op_math.pow import pow_involutory
from pennylane.decomposition_rules.op_math.symbolicop import self_adjoint_legacy
from pennylane.ops.qubit.arithmetic_ops import (
    IntegerComparator,
    QubitCarry,
    QubitSum,
    _integer_comparator_ge_decomposition_impl,
    _integer_comparator_lt_decomposition_impl,
)
from pennylane.typing import Wire
from pennylane.wires import WiresLike


def _qubitcarry_to_cnot_toffoli_resources():
    return {qp.CNOT: 1, qp.Toffoli: 2}


@register_resources(_qubitcarry_to_cnot_toffoli_resources)
def _qubitcarry_to_cnot_toffolis(wires: WiresLike, **__):
    qp.Toffoli(wires=wires[1:])
    qp.CNOT(wires=[wires[1], wires[2]])
    qp.Toffoli(wires=[wires[0], wires[2], wires[3]])


add_decomps(QubitCarry, _qubitcarry_to_cnot_toffolis)


def _qubitsum_to_cnots_resources():
    return {qp.CNOT: 2}


@register_resources(_qubitsum_to_cnots_resources)
def _qubitsum_to_cnots(wires: WiresLike, **__):
    qp.CNOT(wires=[wires[1], wires[2]])
    qp.CNOT(wires=[wires[0], wires[2]])


add_decomps(QubitSum, _qubitsum_to_cnots)

add_decomps("Adjoint(QubitSum)", self_adjoint_legacy)

add_decomps("Pow(QubitSum)", pow_involutory)


def _integer_comparator_lt_resource(num_wires, value, num_work_wires, **_):

    if value == 0:
        return {}

    if value > 2 ** (num_wires - 1) - 1:
        return {qp.X: 1}

    num_controls = num_wires - 1
    binary_str = format(value, f"0{num_controls}b")
    last_significant = binary_str.rfind("1")
    gate_counts = {qp.X: (last_significant + 1) * 2}

    first_significant = binary_str.find("1")
    gate_counts[
        qp.MultiControlledX(
            Wire[first_significant + 2],
            work_wires=Wire[num_work_wires + num_wires - 2 - first_significant],
        )
    ] = 1

    while (first_significant := binary_str.find("1", first_significant + 1)) != -1:
        gate_counts[
            qp.MultiControlledX(
                Wire[first_significant + 2],
                work_wires=Wire[num_work_wires + num_wires - 2 - first_significant],
            )
        ] = 1

    return gate_counts


@register_condition(lambda geq, **_: not geq)
@register_resources(_integer_comparator_lt_resource)
def _integer_comparator_lt_decomposition(wires, value, work_wires, **_):
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

    return _integer_comparator_lt_decomposition_impl(wires, value, work_wires, **_)


def _integer_comparator_ge_resource(num_wires, value, num_work_wires, **_):

    # If the value is 0, the flipping condition is always satisfied.
    if value == 0:
        return {qp.X: 1}

    num_controls = num_wires - 1

    if value > 2**num_controls - 1:
        return {}

    binary_str = format(value, f"0{num_controls}b")
    first_zero = binary_str.find("0")

    if first_zero == -1:
        return {qp.MultiControlledX(Wire[num_controls + 1], work_wires=Wire[num_work_wires]): 1}

    gate_set = Counter()

    gate_set[
        qp.MultiControlledX(
            Wire[first_zero + 2], work_wires=Wire[num_work_wires + num_wires - 2 - first_zero]
        )
    ] = 1
    gate_set[qp.X] = 2

    while (first_zero := binary_str.find("0", first_zero + 1)) != -1:
        gate_set[
            qp.MultiControlledX(
                Wire[first_zero + 2], work_wires=Wire[num_work_wires + num_wires - 2 - first_zero]
            )
        ] = 1
        gate_set[qp.X] += 2

    gate_set[qp.MultiControlledX(Wire[num_controls + 1], work_wires=Wire[num_work_wires])] += 1

    return dict(gate_set)


@register_condition(lambda geq, **_: geq)
@register_resources(_integer_comparator_ge_resource)
def _integer_comparator_ge_decomposition(wires, value, work_wires, **_):
    """Decompose the ``IntegerComparator`` for when the flipping condition is ``n >= value``.

    This decomposition rule mirrors the implementation for the ``n < value`` case.

    """

    return _integer_comparator_ge_decomposition_impl(wires, value, work_wires, **_)


def _integer_comparator_flip_geq_resource(num_wires, value, num_work_wires, geq, **_):
    """Resource estimation for flipping the geq condition."""
    return {
        qp.X: 1,
        resource_rep(
            qp.IntegerComparator,
            num_wires=num_wires,
            value=value,
            geq=not geq,
            num_work_wires=num_work_wires,
        ): 1,
    }


@register_resources(_integer_comparator_flip_geq_resource)
def _integer_comparator_flip_geq(value, geq, wires, work_wires, **_):
    """Decompose the IntegerComparator by flipping geq to lt or vice versa."""
    qp.X(wires[-1])
    IntegerComparator(value, wires, geq=not geq, work_wires=work_wires)


add_decomps(
    IntegerComparator,
    _integer_comparator_lt_decomposition,
    _integer_comparator_ge_decomposition,
    _integer_comparator_flip_geq,
)

add_decomps("Adjoint(IntegerComparator)", self_adjoint_legacy)
