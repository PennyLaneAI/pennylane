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

import pennylane as qp
from pennylane import compiler, control_flow, math
from pennylane.core.apply import apply
from pennylane.core.operator import abstractify
from pennylane.decomposition import add_decomps, register_condition, register_resources
from pennylane.decomposition_rules.op_math.controlled2 import flip_zero_control
from pennylane.ops.op_math.prod2 import Prod2
from pennylane.typing import Wire
from pennylane.wires import DynamicWire, is_abstract_qubit


def _prod2_resources(operands, _init_pauli_rep=None):  # pylint: disable=unused-argument
    return dict(Counter(abstractify(op) for op in operands))


@register_resources(_prod2_resources)
def _prod2_decomp(operands, _init_pauli_rep=None):  # pylint: disable=unused-argument
    for op in reversed(operands):
        apply(op)


# pylint: disable=unused-argument
def _ctrl_prod2_resources(base, control_wires, work_wires, work_wire_type, **_):

    num_control_wires = len(control_wires)
    resources = Counter()
    resources[qp.TemporaryAND] += num_control_wires - 1
    resources[qp.adjoint(abstractify(qp.TemporaryAND))] += num_control_wires - 1

    # Per-factor single-control fan-out from the single aux qubit
    for op in base.operands:
        resources[qp.ctrl(abstractify(op), Wire[1])] += 1

    return dict(resources)


def _multi_temporary_and_all_ones(
    control,
    work_wires,
):
    """Controlled decomposition using a ``TemporaryAND`` ladder.

    Assumes all control values are 1 and returns the last ancilla as the
    effective control target. Compatible with QJIT: the ladder is expressed
    via ``control_flow.for_loop`` when running under tracing.
    """
    num_needed = len(control) - 1

    if any(is_abstract_qubit(w) or isinstance(w, DynamicWire) for w in (*control, *work_wires)):
        # Dynamically allocated wires can't be cast to a JAX array for the traced indexing below.
        # The ladder length is static either way, so just unroll it directly.
        qp.TemporaryAND(wires=[control[0], control[1], work_wires[0]])
        for i in range(1, num_needed):
            qp.TemporaryAND(wires=[work_wires[i - 1], control[i + 1], work_wires[i]])
        return work_wires[num_needed - 1]

    if compiler.active() or qp.capture.enabled():
        control = math.array(control, like="jax")
        work_wires = math.array(work_wires, like="jax")

    qp.TemporaryAND(wires=[control[0], control[1], work_wires[0]])

    @control_flow.for_loop(1, num_needed, 1)
    def _ladder(i):
        qp.TemporaryAND(wires=[work_wires[i - 1], control[i + 1], work_wires[i]])

    _ladder()  # pylint: disable = no-value-for-parameter

    return work_wires[num_needed - 1]


# pylint: disable=unused-argument
@register_condition(
    lambda control_wires, work_wires, work_wire_type, **_: len(control_wires) >= 2
    and len(work_wires) >= len(control_wires) - 1
    and work_wire_type == "zeroed"
)
@register_resources(_ctrl_prod2_resources)
def _controlled_prod2_with_work_wires(base, control_wires, control_values, work_wires, **_):
    """Decomposition of ``C(Prod2)`` with at least ``num_control_wires - 1`` zeroed work wires.

    ``Operator2`` port of :func:`~._controlled_product_with_work_wires`. Assumes all
    ``control_values`` are 1; zero-control flipping is handled by ``flip_zero_control``.
    """
    target_wire = _multi_temporary_and_all_ones(control_wires, work_wires)
    for op in base.operands[::-1]:
        qp.ctrl(op, control=[target_wire])
    qp.adjoint(_multi_temporary_and_all_ones)(control_wires, work_wires)


def _ctrl_prod2_resources_with_one_work_wire(base, control_wires, work_wires, **_):

    resources = Counter()
    resources[qp.MultiControlledX(Wire[len(control_wires) + 1])] += 2

    # Per-factor single-control fan-out from the single aux qubit
    for op in base.operands:
        resources[qp.ctrl(abstractify(op), Wire[1])] += 1

    return dict(resources)


# pylint: disable=unused-argument
@register_condition(
    lambda control_wires, work_wires, work_wire_type, **_: len(control_wires) >= 2
    and len(work_wires) >= 1
    and work_wire_type == "zeroed"
)
@register_resources(_ctrl_prod2_resources_with_one_work_wire)
def _controlled_prod2_with_one_work_wire(base, control_wires, control_values, work_wires, **_):
    """Decomposition of ``C(Prod2)`` with a single zeroed work wire.

    ``Operator2`` port of :func:`~._controlled_product_with_one_work_wire`. Assumes all
    ``control_values`` are 1; zero-control flipping is handled by ``flip_zero_control``.
    """
    qp.MultiControlledX([*control_wires, work_wires[0]])
    for op in base.operands[::-1]:
        qp.ctrl(op, control=work_wires[:1])
    qp.MultiControlledX([*control_wires, work_wires[0]])


add_decomps(Prod2, _prod2_decomp)

add_decomps(
    "C(Prod2)",
    flip_zero_control(_controlled_prod2_with_work_wires),
    flip_zero_control(_controlled_prod2_with_one_work_wire),
)
