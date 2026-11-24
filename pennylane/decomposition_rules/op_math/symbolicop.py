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

from pennylane.core.apply import apply
from pennylane.decomposition.decomposition_rule import DecompositionRule, register_resources
from pennylane.decomposition.resources import resource_rep


def _operator1_to_base_resources(base_class, base_params, **__):
    return {resource_rep(base_class, **base_params): 1}


@register_resources(_operator1_to_base_resources, name="to_base")
def operator1_to_base(*params, wires, base, **__):  # pylint: disable=unused-argument
    """Decompose a symbolic operator to its base."""
    apply(base)


self_adjoint_legacy: DecompositionRule = operator1_to_base
