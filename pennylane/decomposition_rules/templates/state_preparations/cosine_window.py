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


"""Built-in decomposition rules for templates.state_preparations.cosine_window."""

import numpy as np

from pennylane import capture, math, ops, templates
from pennylane.control_flow import for_loop
from pennylane.decomposition import add_decomps, register_resources
from pennylane.ops.op_math.adjoint2 import _adjoint_abstract
from pennylane.templates.state_preparations.cosine_window import CosineWindow
from pennylane.typing import Wire


def _cosine_window_standard_resources(num_wires):
    return {
        ops.Hadamard: 1,
        ops.RZ: 1,
        _adjoint_abstract(templates.QFT(Wire[num_wires])): 1,
        ops.PhaseShift: num_wires,
    }


@register_resources(_cosine_window_standard_resources, name="inverse_qft")
def _cosine_window_standard(wires, **_):
    ops.Hadamard(wires=wires[-1])
    ops.RZ(np.pi, wires=wires[-1])
    ops.adjoint(templates.QFT)(wires=wires)

    if capture.enabled():
        wires = math.array(wires, like="jax")

    @for_loop(len(wires))
    def wires_loop(ind):
        wire = wires[ind]
        ops.PhaseShift(np.pi * 2 ** (-ind - 1), wires=wire)

    wires_loop()  # pylint: disable=no-value-for-parameter


add_decomps(CosineWindow, _cosine_window_standard)
