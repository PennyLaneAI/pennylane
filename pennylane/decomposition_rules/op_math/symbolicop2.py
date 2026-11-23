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


"""Built-in decomposition rules for op_math.symbolicop2."""

from __future__ import annotations

from pennylane.core.apply import apply
from pennylane.core.operator import abstractify
from pennylane.decomposition.decomposition_rule import DecompositionRule, register_resources


def _operator_to_base_resources(base):
    return {abstractify(base): 1}


@register_resources(_operator_to_base_resources, name="to_base")
def _operator_to_base(base):
    """Decompose a symbolic operator to its base."""
    apply(base)


self_adjoint: DecompositionRule = _operator_to_base
