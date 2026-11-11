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

"""Built-in decomposition rules for templates.subroutines.arithmetic.mod_exp."""

from pennylane.decomposition import add_decomps, register_resources, resource_rep
from pennylane.templates.subroutines.arithmetic.mod_exp import ModExp
from pennylane.templates.subroutines.arithmetic.multiplier import Multiplier
from pennylane.templates.subroutines.controlled_sequence import ControlledSequence
from pennylane.typing import Wire
from pennylane.wires import WiresLike


def _mod_exp_decomposition_resources(num_x_wires, num_output_wires, mod, num_work_wires) -> dict:
    return {
        ControlledSequence(
            base=resource_rep(
                Multiplier, num_x_wires=num_output_wires, num_work_wires=num_work_wires, mod=mod
            ),
            control=Wire[num_x_wires],
        ): 1,
    }


@register_resources(_mod_exp_decomposition_resources)
def _mod_exp_decomposition(
    x_wires, output_wires: WiresLike, base, mod, work_wires: WiresLike, **__
):
    ControlledSequence(Multiplier(base, output_wires, mod, work_wires), control=x_wires)


add_decomps(ModExp, _mod_exp_decomposition)
