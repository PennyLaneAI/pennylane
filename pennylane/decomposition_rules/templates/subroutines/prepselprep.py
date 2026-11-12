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

"""Built-in decomposition rules for templates.subroutines.prepselprep."""

# pylint: disable=arguments-differ

from pennylane import math
from pennylane.core.operator import abstractify
from pennylane.decomposition import add_decomps, register_resources, resource_rep
from pennylane.ops import GlobalPhase, StatePrep, change_op_basis
from pennylane.ops.op_math.adjoint2 import _adjoint_abstract
from pennylane.ops.op_math.change_op_basis2 import _change_op_basis_abstract
from pennylane.ops.op_math.prod import prod
from pennylane.templates.subroutines.prepselprep import PrepSelPrep, _get_new_terms
from pennylane.templates.subroutines.select import Select
from pennylane.typing import Wire


def _prepselprep_resources(op_reps, num_control):
    select_lcu = Select(list(op_reps), control=Wire[num_control], work_wires=Wire[0], partial=True)
    select_phases = Select(
        [abstractify(GlobalPhase)] * len(op_reps),
        control=Wire[num_control],
        work_wires=Wire[0],
        partial=True,
    )

    _compute_op = resource_rep(StatePrep, num_wires=num_control)
    _target_op = prod(select_lcu, select_phases)
    resources = {
        _change_op_basis_abstract(
            _compute_op,
            _target_op,
            _adjoint_abstract(_compute_op),
        ): 1,
    }
    return resources


# pylint: disable=unused-argument
@register_resources(_prepselprep_resources)
def _prepselprep_decomp(*_, wires, lcu, control, target_wires):
    coeffs, ops, phases = _get_new_terms(lcu)
    sqrt_coeffs = math.sqrt(coeffs)
    change_op_basis(
        StatePrep(sqrt_coeffs, normalize=True, pad_with=0, wires=control),
        prod(Select(ops, control, partial=True), Select(phases, control, partial=True)),
    )


add_decomps(PrepSelPrep, _prepselprep_decomp)
