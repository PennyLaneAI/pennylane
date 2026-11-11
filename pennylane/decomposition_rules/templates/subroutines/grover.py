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

"""Built-in decomposition rules for templates.subroutines.grover."""

import numpy as np

from pennylane import capture, math
from pennylane.control_flow import for_loop
from pennylane.decomposition import add_decomps, register_resources
from pennylane.ops import GlobalPhase, Hadamard, MultiControlledX, PauliZ
from pennylane.templates.subroutines.grover import GroverOperator
from pennylane.typing import Wire


def _grover_operator_resources(num_wires, num_work_wires):
    return {
        Hadamard: (num_wires - 1) * 2,
        PauliZ: 2,
        GlobalPhase: 1,
        MultiControlledX(Wire[num_wires], work_wires=Wire[num_work_wires]): 1,
    }


@register_resources(_grover_operator_resources)
def _grover_decomposition(wires, work_wires, n_wires):
    ctrl_values = [0] * (n_wires - 1)

    if capture.enabled():
        wires = math.array(wires, like="jax")

    @for_loop(len(wires) - 1)
    def apply_hadamards(i):
        Hadamard(wires[i])

    apply_hadamards()  # pylint: disable=no-value-for-parameter

    PauliZ(wires[-1])
    MultiControlledX(
        control_values=ctrl_values,
        wires=wires,
        work_wires=work_wires,
    )
    PauliZ(wires[-1])

    apply_hadamards()  # pylint: disable=no-value-for-parameter

    GlobalPhase(np.pi)


add_decomps(GroverOperator, _grover_decomposition)
