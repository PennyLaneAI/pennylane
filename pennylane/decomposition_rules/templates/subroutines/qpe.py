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

"""Built-in decomposition rules for templates.subroutines.qpe."""

from pennylane import capture, compiler, math, ops
from pennylane.control_flow import for_loop
from pennylane.decomposition import add_decomps, register_resources
from pennylane.ops import adjoint
from pennylane.ops import pow as qp_pow
from pennylane.ops.op_math.controlled2 import _ctrl_abstract
from pennylane.ops.op_math.pow2 import _pow_abstract
from pennylane.templates.subroutines.qft import QFT
from pennylane.templates.subroutines.qpe import QuantumPhaseEstimation
from pennylane.typing import Wire


def _qpe_decomp_resource(unitary, estimation_wires):
    num_estimation_wires = len(estimation_wires)
    gate_count = {
        ops.Hadamard: num_estimation_wires,
        adjoint(QFT(Wire[num_estimation_wires])): 1,
    }

    # NOTE: Need abstract resource representations just in case
    # the unitary is an operator1.
    for i in range(num_estimation_wires):
        pow_rep = _pow_abstract(
            unitary,
            2**i,
        )
        gate_count[_ctrl_abstract(pow_rep, control_wires=Wire[1])] = 1
    return gate_count


@register_resources(_qpe_decomp_resource)
def _qpe_decomp(unitary, estimation_wires):
    if compiler.active() or capture.enabled():
        estimation_wires = math.array(estimation_wires, like="jax")

    num_estimation_wires = len(estimation_wires)

    @for_loop(0, num_estimation_wires)
    def _apply_h(i):
        ops.Hadamard(estimation_wires[i])

    # pylint: disable=no-value-for-parameter
    _apply_h()

    # NOTE: Must be pythonic for loop as 'z' argument in 'pow'
    # is a static argument.
    for i, w in enumerate(estimation_wires):
        ops.ctrl(qp_pow(unitary, 2 ** (len(estimation_wires) - 1 - i)), w)

    ops.adjoint(QFT(wires=estimation_wires))


add_decomps(QuantumPhaseEstimation, _qpe_decomp)
