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

"""Built-in decomposition rules for templates.subroutines.qsvt."""

from collections import defaultdict

from pennylane import ops
from pennylane.core.apply import apply
from pennylane.core.operator import abstractify
from pennylane.decomposition import add_decomps, register_resources
from pennylane.ops.op_math.adjoint2 import _adjoint_abstract
from pennylane.ops.op_math.change_op_basis2 import _change_op_basis_abstract
from pennylane.templates.subroutines.qsvt import QSVT


def _QSVT_resources(UA, projectors):
    resources = defaultdict(int)
    resources[abstractify(projectors[0])] = 1
    for i in range(1, len(projectors) - 1, 2):
        resources[_change_op_basis_abstract(UA, projectors[i], _adjoint_abstract(UA))] += 1
        resources[abstractify(projectors[i + 1])] += 1
    if len(projectors) % 2 == 0:
        resources[abstractify(UA)] += 1
        resources[abstractify(projectors[-1])] += 1
    return dict(resources)


@register_resources(_QSVT_resources)
def _QSVT_decomposition(UA, projectors):

    apply(projectors[0])

    for i in range(1, len(projectors) - 1, 2):
        ops.change_op_basis(UA, projectors[i])
        apply(projectors[i + 1])

    if len(projectors) % 2 == 0:
        apply(UA)
        apply(projectors[-1])


add_decomps(QSVT, _QSVT_decomposition)
