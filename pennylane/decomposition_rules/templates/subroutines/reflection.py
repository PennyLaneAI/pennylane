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

"""Built-in decomposition rules for templates.subroutines.reflection."""

import numpy as np

from pennylane import ops
from pennylane.core.apply import apply
from pennylane.decomposition import add_decomps, register_resources
from pennylane.ops.op_math.adjoint2 import _adjoint_abstract
from pennylane.templates.subroutines.reflection import Reflection
from pennylane.typing import Bool, Float, Wire
from pennylane.wires import Wires


def _reflection_decomposition_resources(base_rep, num_wires, num_reflection_wires=None) -> dict:

    num_wires = num_reflection_wires if num_reflection_wires is not None else num_wires

    resources = {
        ops.GlobalPhase: 1,
        _adjoint_abstract(base_rep): 1,
        ops.PauliX: 2,
    }

    if num_wires > 1:
        resources[
            ops.ctrl(
                ops.PhaseShift(Float, Wire[1]),
                control=Wire[num_wires - 1],
                control_values=Bool[num_wires - 1],
            )
        ] = 1
    else:
        resources[ops.PhaseShift] = 1

    resources[base_rep] = 1

    return resources


@register_resources(_reflection_decomposition_resources)
def _reflection_decomposition(*parameters, wires=None, **hyperparameters):
    alpha = parameters[0]
    U = hyperparameters["base"]
    reflection_wires = hyperparameters["reflection_wires"]

    wires = Wires(reflection_wires) if reflection_wires is not None else wires

    ops.GlobalPhase(np.pi)
    ops.adjoint(U)

    if len(wires) > 1:
        ops.PauliX(wires=wires[-1])

        ops.ctrl(
            ops.PhaseShift(alpha, wires=wires[-1]),
            control=wires[:-1],
            control_values=[0] * (len(wires) - 1),
        )

        ops.PauliX(wires=wires[-1])

    else:
        ops.PauliX(wires=wires)
        ops.PhaseShift(alpha, wires=wires)
        ops.PauliX(wires=wires)

    apply(U)


add_decomps(Reflection, _reflection_decomposition)
