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

"""Built-in decomposition rules for templates.core."""

from pennylane.core.apply import apply
from pennylane.decomposition import add_decomps, register_resources
from pennylane.pytrees import unflatten
from pennylane.templates.core import SubroutineOp


def _calculate_resources(subroutine: "Subroutine", signature_key):
    sig = subroutine.signature.bind(*signature_key)
    for arg in subroutine.dynamic_argnames:
        struct, shapes = sig.arguments[arg]
        sig.arguments[arg] = unflatten(shapes, struct)
    return subroutine.compute_resources(**sig.arguments)


# pylint: disable=unused-argument
@register_resources(_calculate_resources)
def _subroutine_decomp(*data, wires, decomposition):
    _ = [apply(op) for op in decomposition]


add_decomps(SubroutineOp, _subroutine_decomp)
