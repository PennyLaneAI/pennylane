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

"""Built-in decomposition rules for templates.subroutines.flip_sign."""

from pennylane.decomposition import add_decomps, register_resources
from pennylane.ops import CZ, X, Z, ctrl
from pennylane.templates.subroutines.flip_sign import FlipSign
from pennylane.typing import Bool, Wire
from pennylane.wires import WiresLike


def _flip_sign_resources(state: tuple[int], wires: WiresLike, work_wires: WiresLike):
    num_wires = len(wires)
    num_ctrl_wires = num_wires - 1
    if num_ctrl_wires == 0:
        res = {Z: 1}
    elif num_ctrl_wires == 1 and state[0]:
        res = {CZ: 1}
    else:
        res = {
            ctrl(
                Z(Wire[1]),
                control=Wire[num_ctrl_wires],
                control_values=Bool[num_ctrl_wires],
                work_wires=Wire[len(work_wires)],
                work_wire_type="zeroed",
            ): 1
        }

    if state[-1] == 0:
        res[X] = 2
    return res


@register_resources(_flip_sign_resources)
def _flip_sign_decomposition(state: tuple[int], wires: WiresLike, work_wires: WiresLike):
    if state[-1] == 0:
        X(wires[-1])

    if len(wires) == 1:
        Z(wires)
    elif len(wires) == 2 and state[0]:
        CZ(wires)
    else:
        ctrl(
            Z(wires[-1]),
            control=wires[:-1],
            work_wire_type="zeroed",
            work_wires=work_wires,
            control_values=state[:-1],
        )

    if state[-1] == 0:
        X(wires[-1])


add_decomps(FlipSign, _flip_sign_decomposition)
