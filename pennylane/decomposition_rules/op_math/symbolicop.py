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

"""Built-in decomposition rules for op_math.symbolicop."""

from __future__ import annotations

import pennylane as qp
from pennylane.decomposition.decomposition_rule import DecompositionRule, register_resources
from pennylane.decomposition.resources import resource_rep


def _decomp_to_base_legacy_res(base_class, base_params, **__):
    return {resource_rep(base_class, **base_params): 1}


# pylint: disable=protected-access,unused-argument
@register_resources(_decomp_to_base_legacy_res)
def decompose_to_base_legacy(*params, wires, base, **__):
    """Decompose a symbolic operator to its base."""
    qp.apply(base)


self_adjoint_legacy: DecompositionRule = decompose_to_base_legacy
