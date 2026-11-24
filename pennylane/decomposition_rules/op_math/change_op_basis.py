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
from pennylane.decomposition.resources import CompressedResourceOp, resource_rep
from pennylane.ops.op_math import adjoint, ctrl
from pennylane.ops.op_math.adjoint import Adjoint
from pennylane.ops.op_math.adjoint2 import Adjoint2, _adjoint_abstract
from pennylane.ops.op_math.change_op_basis import ChangeOpBasis
from pennylane.ops.op_math.controlled2 import _ctrl_abstract
from pennylane.typing import Wire


def _change_op_basis_standard_resources(compute_op, target_op, uncompute_op):
    resources = Counter()

    resources[compute_op] += 1
    resources[target_op] += 1
    resources[uncompute_op] += 1

    return resources


def _inverse_abstract(op):
    """Abstract inverse of an operand, unwrapping an existing adjoint instead of nesting another one."""
    if isinstance(op, CompressedResourceOp) and issubclass(op.op_type, Adjoint):
        return resource_rep(op.params["base_class"], **op.params["base_params"])
    if isinstance(op, Adjoint2):
        return op.base
    return _adjoint_abstract(op)


def _adjoint_change_op_basis_standard_resources(base_params, **_):
    resources = defaultdict(int)
    for name in ("uncompute_op", "target_op", "compute_op"):
        resources[_inverse_abstract(base_params[name])] += 1
    return resources


@register_resources(_adjoint_change_op_basis_standard_resources, name="adjoint_target")
def adjoint_change_op_basis_standard(*_, base, **__):
    """Decompose ``Adjoint(ChangeOpBasis)`` using the ``adjoint_target`` strategy."""
    for op in base.operands:
        if isinstance(op, (Adjoint, Adjoint2)):
            apply(op.base)
        else:
            adjoint(apply(op))


add_decomps("Adjoint(ChangeOpBasis)", adjoint_change_op_basis_standard)


def _controlled_change_op_basis_standard_resources(
    *_,
    num_control_wires,
    num_zero_control_values,
    num_work_wires,
    work_wire_type,
    base_class,
    base_params,
    **__,
):  # pylint: disable=too-many-arguments,unused-argument
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


@register_resources(_controlled_change_op_basis_standard_resources, name="control_target")
def controlled_change_op_basis_standard(
    *_,
    control_wires,
    control_values,
    work_wires,
    work_wire_type,
    base,
    **__,
):
    """Decompose ``C(ChangeOpBasis)`` using the ``control_target`` strategy."""
    apply(base.operands[2])
    ctrl(
        apply(base.operands[1]),
        control=control_wires,
        control_values=control_values,
        work_wires=work_wires,
        work_wire_type=work_wire_type,
    )
    apply(base.operands[0])


@register_resources(_change_op_basis_standard_resources, name="expand_basis_change")
def change_op_basis_standard(*_, wires=None, operands, **__):  # pylint: disable=unused-argument
    """Decompose ``ChangeOpBasis`` using the ``expand_basis_change`` strategy."""
    for op in operands[::-1]:
        apply(op)


add_decomps(ChangeOpBasis, change_op_basis_standard)

add_decomps("C(ChangeOpBasis)", controlled_change_op_basis_standard)
