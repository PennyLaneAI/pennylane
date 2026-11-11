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

"""Built-in decomposition rules for templates.subroutines.qft."""

import numpy as np

from pennylane import capture, compiler, math
from pennylane.capture import hint
from pennylane.control_flow import for_loop
from pennylane.decomposition import add_decomps, register_resources
from pennylane.ops import SWAP, ControlledPhaseShift, Hadamard
from pennylane.templates.subroutines.qft import QFT
from pennylane.typing import AbstractWires
from pennylane.wires import WiresLike


def _qft_decomposition_resources(wires: AbstractWires):
    num_wires = len(wires)
    return {
        Hadamard: num_wires,
        SWAP: num_wires // 2,
        ControlledPhaseShift: num_wires * (num_wires - 1) // 2,
    }


@register_resources(_qft_decomposition_resources)
def _qft_decomposition(wires: WiresLike):
    num_wires = len(wires)
    shifts = [2 * np.pi * 2**-i for i in range(2, num_wires + 1)]
    if compiler.active() or capture.enabled():
        shifts = math.array(shifts, like="jax")
        wires = math.array(wires, like="jax")

    @for_loop(num_wires)
    def outer_loop(i):
        Hadamard(wires[i])

        if num_wires > 1:

            @hint({"num-iters": (num_wires - 1) / 2})
            @for_loop(num_wires - 1 - i)
            def cphaseshift_loop(j):
                ControlledPhaseShift(shifts[j], wires=[wires[i + j + 1], wires[i]])

            cphaseshift_loop()  # pylint: disable=no-value-for-parameter

    outer_loop()  # pylint: disable=no-value-for-parameter

    @for_loop(num_wires // 2)
    def swaps(i):
        SWAP(wires=[wires[i], wires[num_wires - i - 1]])

    swaps()  # pylint: disable=no-value-for-parameter


add_decomps(QFT, _qft_decomposition)
