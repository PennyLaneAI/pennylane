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

"""Built-in decomposition rules for op_math.prod."""

from collections import Counter

import pennylane as qp
from pennylane.decomposition_rules.op_math.controlled import flip_zero_control
from pennylane.decomposition_rules.op_math.prod2 import _multi_temporary_and_all_ones
from pennylane.ops.op_math.adjoint2 import _adjoint_abstract
from pennylane.ops.op_math.controlled2 import _ctrl_abstract
from pennylane.ops.op_math.prod import Prod
from pennylane.typing import Wire


def _prod_resources(resources):
    return resources


# pylint: disable=unused-argument
@qp.register_resources(_prod_resources)
def _prod_decomp(*_, wires=None, operands, **__):
    for op in reversed(operands):
        qp.apply(op)


qp.add_decomps(Prod, _prod_decomp)


def _ctrl_prod_resources(
    num_control_wires,
    base_params,
    **_,
):
    factor_reps = base_params["resources"]

    resources = Counter()
    resources[qp.TemporaryAND] += num_control_wires - 1
    resources[_adjoint_abstract(qp.TemporaryAND)] += num_control_wires - 1

    # Per-factor single-control fan-out from the single aux qubit
    for rep, count in factor_reps.items():
        resources[_ctrl_abstract(rep, Wire[1])] += count

    return dict(resources)


@qp.register_condition(
    lambda num_control_wires, num_work_wires, work_wire_type, **_: num_control_wires >= 2
    and num_work_wires >= num_control_wires - 1
    and work_wire_type == "zeroed"
)
@qp.register_resources(_ctrl_prod_resources)
def _controlled_product_with_work_wires(*_, control_wires, work_wires, base, **__):
    """Decomposition of ``C(Prod)`` with at least ``num_control_wires - 1`` work wires.

    Assumes that all ``control_values`` are 1. Zero-control flipping is handled
    by wrapping this rule with ``flip_zero_control``.
    """
    target_wire = _multi_temporary_and_all_ones(control_wires, work_wires)
    for op in base.operands[::-1]:
        qp.ctrl(op, control=[target_wire])
    qp.adjoint(_multi_temporary_and_all_ones)(control_wires, work_wires)


def _ctrl_prod_resources_with_one_work_wire(
    num_control_wires,
    base_params,
    **_,
):
    factor_reps = base_params["resources"]  # {rep: count} from Prod
    multicx_rep = qp.MultiControlledX(Wire[num_control_wires + 1])

    resources = Counter()
    resources[multicx_rep] += 2

    # Per-factor single-control fan-out from the single aux qubit
    for rep, count in factor_reps.items():
        resources[_ctrl_abstract(rep, Wire[1])] += count

    return dict(resources)


@qp.register_condition(
    lambda num_control_wires, num_work_wires, work_wire_type, **_: num_control_wires >= 2
    and num_work_wires >= 1
    and work_wire_type == "zeroed"
)
@qp.register_resources(_ctrl_prod_resources_with_one_work_wire)
def _controlled_product_with_one_work_wire(*_, control_wires, work_wires, base, **__):
    """Decomposition of ``C(Prod)`` with a single zeroed work wire.

    Assumes that all ``control_values`` are 1. Zero-control flipping is handled
    by wrapping this rule with ``flip_zero_control``.
    """
    qp.MultiControlledX([*control_wires, work_wires[0]])
    for op in base.operands[::-1]:
        qp.ctrl(op, control=work_wires[:1])
    qp.MultiControlledX([*control_wires, work_wires[0]])


qp.add_decomps(
    "C(Prod)",
    flip_zero_control(_controlled_product_with_work_wires),
    flip_zero_control(_controlled_product_with_one_work_wire),
)
