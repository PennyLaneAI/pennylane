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


"""Built-in decomposition rules for op_math.prod2."""

from collections import Counter

from pennylane import ops, templates
from pennylane.core.apply import apply
from pennylane.core.operator import abstractify
from pennylane.decomposition import add_decomps, register_condition, register_resources
from pennylane.decomposition_rules.op_math.controlled2 import (
    _multi_temporary_and_all_ones,
    flip_zero_control,
)
from pennylane.ops.op_math.prod2 import Prod2
from pennylane.typing import Wire


def _prod2_standard_resources(operands, _init_pauli_rep=None):
    return dict(Counter(abstractify(op) for op in operands))


@register_resources(_prod2_standard_resources, name="expand_factors")
def prod2_standard(operands, _init_pauli_rep=None):
    """Decompose ``Prod2`` using the ``expand_factors`` strategy."""
    for op in reversed(operands):
        apply(op)


def _controlled_prod2_with_work_wires_resources(
    base, control_wires, work_wires, work_wire_type, **_  # pylint: disable=unused-argument
):

    num_control_wires = len(control_wires)
    resources = Counter()
    resources[templates.TemporaryAND] += num_control_wires - 1
    resources[ops.adjoint(abstractify(templates.TemporaryAND))] += num_control_wires - 1

    # Per-factor single-control fan-out from the single aux qubit
    for op in base.operands:
        resources[ops.ctrl(abstractify(op), Wire[1])] += 1

    return dict(resources)


@register_condition(
    lambda control_wires, work_wires, work_wire_type, **_: len(control_wires) >= 2
    and len(work_wires) >= len(control_wires) - 1
    and work_wire_type == "zeroed"
)
@register_resources(_controlled_prod2_with_work_wires_resources, name="with_work_wires")
# pylint: disable-next=unused-argument
def controlled_prod2_with_work_wires(base, control_wires, control_values, work_wires, **_):
    """Decomposition of ``C(Prod2)`` with at least ``num_control_wires - 1`` zeroed work wires.

    ``Operator2`` port of :func:`~._controlled_product_with_work_wires`. Assumes all
    ``control_values`` are 1; zero-control flipping is handled by ``flip_zero_control``.
    """
    target_wire = _multi_temporary_and_all_ones(control_wires, work_wires)
    for op in base.operands[::-1]:
        ops.ctrl(op, control=[target_wire])
    ops.adjoint(_multi_temporary_and_all_ones)(control_wires, work_wires)


# pylint: disable-next=unused-argument
def _controlled_prod2_with_one_work_wire_resources(base, control_wires, work_wires, **_):

    resources = Counter()
    resources[ops.MultiControlledX(Wire[len(control_wires) + 1])] += 2

    # Per-factor single-control fan-out from the single aux qubit
    for op in base.operands:
        resources[ops.ctrl(abstractify(op), Wire[1])] += 1

    return dict(resources)


@register_condition(
    lambda control_wires, work_wires, work_wire_type, **_: len(control_wires) >= 2
    and len(work_wires) >= 1
    and work_wire_type == "zeroed"
)
@register_resources(_controlled_prod2_with_one_work_wire_resources, name="with_one_work_wire")
# pylint: disable-next=unused-argument
def controlled_prod2_with_one_work_wire(base, control_wires, control_values, work_wires, **_):
    """Decomposition of ``C(Prod2)`` with a single zeroed work wire.

    ``Operator2`` port of :func:`~._controlled_product_with_one_work_wire`. Assumes all
    ``control_values`` are 1; zero-control flipping is handled by ``flip_zero_control``.
    """
    ops.MultiControlledX([*control_wires, work_wires[0]])
    for op in base.operands[::-1]:
        ops.ctrl(op, control=work_wires[:1])
    ops.MultiControlledX([*control_wires, work_wires[0]])


add_decomps(Prod2, prod2_standard)

add_decomps(
    "C(Prod2)",
    flip_zero_control(controlled_prod2_with_work_wires, name="with_work_wires"),
    flip_zero_control(controlled_prod2_with_one_work_wire, name="with_one_work_wire"),
)
