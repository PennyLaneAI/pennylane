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

"""Built-in decomposition rules for templates.subroutines.arithmetic.phase_adder."""

import numpy as np

from pennylane import ops
from pennylane.decomposition import add_decomps, register_resources
from pennylane.ops.op_math.change_op_basis2 import _change_op_basis_abstract
from pennylane.templates.subroutines.arithmetic.phase_adder import PhaseAdder
from pennylane.templates.subroutines.qft import QFT
from pennylane.typing import Float, Wire
from pennylane.wires import WiresLike


def _add_k_fourier(k, wires: WiresLike):
    """Adds k in the Fourier basis"""
    op_list = []
    for j, wire in enumerate(wires):
        op_list.append(ops.PhaseShift(k * np.pi / (2**j), wires=wire))
    return op_list


# pylint: disable-next=unused-argument
def _phase_adder_decomposition_resources(k, x_wires, mod, work_wire) -> dict:
    num_x_wires = len(x_wires)

    if mod == 2**num_x_wires:
        return {ops.PhaseShift: num_x_wires}

    basis_op1 = ops.prod(
        ops.X(Wire[1]),
        ops.adjoint(QFT(Wire[num_x_wires])),
        *(ops.PhaseShift(Float, Wire[1]) for _ in range(num_x_wires)),
    )
    basis_op2 = ops.prod(
        *(ops.PhaseShift(Float, Wire[1]) for _ in range(num_x_wires)),
        QFT(Wire[num_x_wires]),
        ops.X(Wire[1]),
    )

    resources = {
        ops.PhaseShift: 2 * num_x_wires,
        _change_op_basis_abstract(
            ops.adjoint(QFT(Wire[num_x_wires])),
            ops.CNOT,
            QFT(Wire[num_x_wires]),
        ): 1,
        ops.ctrl(ops.PhaseShift(Float, Wire[1]), control=Wire[1]): num_x_wires,
        _change_op_basis_abstract(
            basis_op1,
            ops.CNOT,
            basis_op2,
        ): 1,
    }
    return resources


# pylint: disable=no-value-for-parameter
@register_resources(_phase_adder_decomposition_resources)
def _phase_adder_decomposition(k, x_wires: WiresLike, mod, work_wire):

    n_wires = len(x_wires)

    _add_k_fourier(k, x_wires)
    if mod == 2**n_wires:
        return

    aux_k = x_wires[0]
    _add_k_fourier(-mod, x_wires)
    ops.change_op_basis(
        ops.adjoint(QFT(wires=x_wires)),
        ops.CNOT(wires=[aux_k, work_wire[0]]),
        QFT(wires=x_wires),
    )
    for op in _add_k_fourier(mod, x_wires):
        ops.ctrl(op, control=work_wire)

    def compute():
        _add_k_fourier(-k, x_wires)
        ops.adjoint(QFT(wires=x_wires))
        ops.X(aux_k)

    def uncompute():
        ops.X(aux_k)
        QFT(wires=x_wires)
        _add_k_fourier(k, x_wires)

    ops.change_op_basis(compute, ops.CNOT(wires=[aux_k, work_wire[0]]), uncompute)


add_decomps(PhaseAdder, _phase_adder_decomposition)
