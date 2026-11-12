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

"""Built-in decomposition rules for op_math.change_op_basis."""

from collections import Counter, defaultdict

from pennylane.core.apply import apply
from pennylane.decomposition import add_decomps, register_resources
from pennylane.ops.op_math import adjoint, ctrl
from pennylane.ops.op_math.adjoint2 import _adjoint_abstract
from pennylane.ops.op_math.change_op_basis import ChangeOpBasis
from pennylane.ops.op_math.controlled2 import _ctrl_abstract
from pennylane.typing import Wire


def _change_op_basis_resources(compute_op, target_op, uncompute_op):
    resources = Counter()

    resources[compute_op] += 1
    resources[target_op] += 1
    resources[uncompute_op] += 1

    return resources


def _adjoint_change_op_basis_resources(base_params, **_):
    resources = defaultdict(int)
    resources[base_params["compute_op"]] += 1
    resources[base_params["uncompute_op"]] += 1
    target_op = base_params["target_op"]
    resources[_adjoint_abstract(target_op)] += 1
    return resources


# pylint: disable=protected-access
@register_resources(_adjoint_change_op_basis_resources)
def _adjoint_change_op_basis_decomp(*_, base, **__):
    apply(base.operands[2])
    adjoint(apply(base.operands[1]))
    apply(base.operands[0])


add_decomps("Adjoint(ChangeOpBasis)", _adjoint_change_op_basis_decomp)


def _controlled_change_op_basis_resources(
    *_,
    num_control_wires,
    num_zero_control_values,
    num_work_wires,
    work_wire_type,
    base_class,
    base_params,
    **__,
):  # pylint: disable=unused-argument, too-many-arguments
    resources = defaultdict(int)
    resources[base_params["compute_op"]] += 1
    resources[
        _ctrl_abstract(
            base_params["target_op"],
            Wire[num_control_wires],
            Wire[num_work_wires],
            work_wire_type,
            num_zero_control_values,
        )
    ] += 1
    resources[base_params["uncompute_op"]] += 1
    return resources


@register_resources(_controlled_change_op_basis_resources)
def _controlled_change_op_basis_decomposition(
    *_,
    control_wires,
    control_values,
    work_wires,
    work_wire_type,
    base,
    **__,
):
    apply(base.operands[2])
    ctrl(
        apply(base.operands[1]),
        control=control_wires,
        control_values=control_values,
        work_wires=work_wires,
        work_wire_type=work_wire_type,
    )
    apply(base.operands[0])


# pylint: disable=unused-argument
@register_resources(_change_op_basis_resources)
def _change_op_basis_decomp(*_, wires=None, operands, **__):
    for op in operands[::-1]:
        apply(op)


add_decomps(ChangeOpBasis, _change_op_basis_decomp)

add_decomps("C(ChangeOpBasis)", _controlled_change_op_basis_decomposition)
