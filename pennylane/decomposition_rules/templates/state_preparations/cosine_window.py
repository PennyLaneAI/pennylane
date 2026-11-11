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

import pennylane as qp
from pennylane import capture, register_resources
from pennylane.control_flow import for_loop
from pennylane.decomposition import add_decomps
from pennylane.ops.op_math.adjoint2 import _adjoint_abstract
from pennylane.templates.state_preparations.cosine_window import CosineWindow
from pennylane.typing import Wire


def _cosine_window_resources(num_wires):
    return {
        qp.Hadamard: 1,
        qp.RZ: 1,
        _adjoint_abstract(qp.QFT(Wire[num_wires])): 1,
        qp.PhaseShift: num_wires,
    }


@register_resources(_cosine_window_resources)
def _cosine_window_decomposition(wires, **_):
    qp.Hadamard(wires=wires[-1])
    qp.RZ(np.pi, wires=wires[-1])
    qp.adjoint(qp.QFT)(wires=wires)

    if capture.enabled():
        wires = qp.math.array(wires, like="jax")

    @for_loop(len(wires))
    def wires_loop(ind):
        wire = wires[ind]
        qp.PhaseShift(np.pi * 2 ** (-ind - 1), wires=wire)

    wires_loop()  # pylint: disable=no-value-for-parameter


add_decomps(CosineWindow, _cosine_window_decomposition)
