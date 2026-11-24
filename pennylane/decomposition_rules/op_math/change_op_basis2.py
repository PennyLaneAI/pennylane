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


"""Built-in decomposition rules for op_math.change_op_basis2."""

from collections import Counter, defaultdict

from pennylane.core.apply import apply
from pennylane.decomposition import add_decomps, register_resources
from pennylane.decomposition_rules.op_math import flip_zero_control
from pennylane.ops.op_math import adjoint, ctrl
from pennylane.ops.op_math.adjoint import Adjoint
from pennylane.ops.op_math.adjoint2 import Adjoint2
from pennylane.ops.op_math.change_op_basis2 import ChangeOpBasis2, _change_op_basis_abstract
from pennylane.typing import Bool


def _controlled_change_op_basis_standard_resources(  # pylint: disable=unused-argument
    base,
    control_wires,
    control_values,
    work_wires,
    work_wire_type,
):
    resources = defaultdict(int)
    resources[base.compute_op] += 1
    resources[
        ctrl(
            base.target_op,
            control=control_wires,
            control_values=Bool[len(control_wires)],
            work_wires=work_wires,
            work_wire_type=work_wire_type,
        )
    ] += 1
    resources[base.uncompute_op] += 1
    return resources


@register_resources(_controlled_change_op_basis_standard_resources, name="control_target")
def controlled_change_op_basis_standard(
    base,
    control_wires,
    control_values,
    work_wires,
    work_wire_type,
):
    """Decompose ``C(ChangeOpBasis2)`` using the ``control_target`` strategy."""
    apply(base.compute_op)
    ctrl(
        base.target_op,
        control=control_wires,
        control_values=control_values,
        work_wires=work_wires,
        work_wire_type=work_wire_type,
    )
    apply(base.uncompute_op)


def _change_op_basis_standard_resources(compute_op, target_op, uncompute_op):
    resources = Counter()

    resources[compute_op] += 1
    resources[target_op] += 1
    resources[uncompute_op] += 1

    return resources


@register_resources(_change_op_basis_standard_resources, name="expand_basis_change")
def change_op_basis_standard(compute_op, target_op, uncompute_op):
    """Decompose ``ChangeOpBasis2`` using the ``expand_basis_change`` strategy."""
    apply(compute_op)
    apply(target_op)
    apply(uncompute_op)


add_decomps(ChangeOpBasis2, change_op_basis_standard)

add_decomps(
    "C(ChangeOpBasis2)",
    flip_zero_control(controlled_change_op_basis_standard, name="control_target"),
)


def _inverse(op):
    """Invert an operand, unwrapping an existing adjoint instead of nesting another one."""
    if isinstance(op, (Adjoint, Adjoint2)):
        return op.base
    return adjoint(op)


def _adjoint_change_op_basis_standard_resources(base):
    return {
        _change_op_basis_abstract(
            _inverse(base.uncompute_op), adjoint(base.target_op), _inverse(base.compute_op)
        ): 1
    }


@register_resources(_adjoint_change_op_basis_standard_resources, name="adjoint_target")
def adjoint_change_op_basis_standard(base):
    """Decompose ``Adjoint(ChangeOpBasis2)`` using the ``adjoint_target`` strategy."""
    ChangeOpBasis2(_inverse(base.uncompute_op), adjoint(base.target_op), _inverse(base.compute_op))


add_decomps("Adjoint(ChangeOpBasis2)", adjoint_change_op_basis_standard)
