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

"""Built-in decomposition rules for templates.subroutines.aqft."""

import numpy as np

from pennylane import capture, compiler, math
from pennylane.capture import hint
from pennylane.control_flow import for_loop
from pennylane.decomposition import add_decomps, register_resources
from pennylane.ops import SWAP, ControlledPhaseShift, Hadamard
from pennylane.templates.subroutines.aqft import AQFT


def _AQFT_resources(order, wires):
    num_wires = len(wires)
    num_ctrl_ps = sum(min(num_wires - 1 - i, order) for i in range(num_wires))
    return {
        Hadamard: num_wires,
        ControlledPhaseShift: num_ctrl_ps,
        SWAP: num_wires // 2,
    }


@register_resources(_AQFT_resources)
def _AQFT_decomposition(wires, order):
    num_wires = len(wires)
    shifts = [2 * np.pi / (2**i) for i in range(2, num_wires + 1)]

    if compiler.active() or capture.enabled():
        shifts = math.array(shifts, like="jax")
        wires = math.array(wires, like="jax")

    total_inner_iters = sum(min(num_wires - 1 - i, order) for i in range(num_wires))
    average_inner_iters = total_inner_iters / num_wires

    @for_loop(num_wires)
    def wire_loop(i):
        wire = wires[i]
        Hadamard(wire)

        @hint({"num-iters": average_inner_iters})
        @for_loop(math.min(math.array([num_wires - 1 - i, order], like=math.get_interface(i))))
        def cphase_loop(j):
            shift = shifts[j]
            control_wire = wires[i + 1 + j]
            ControlledPhaseShift(shift, wires=[control_wire, wire])

        cphase_loop()  # pylint: disable=no-value-for-parameter

    wire_loop()  # pylint: disable=no-value-for-parameter

    @for_loop(num_wires // 2)
    def half_wire_loop(k):
        wire1 = wires[k]
        wire2 = wires[-k - 1]
        SWAP(wires=[wire1, wire2])

    half_wire_loop()  # pylint: disable=no-value-for-parameter


add_decomps(AQFT, _AQFT_decomposition)
