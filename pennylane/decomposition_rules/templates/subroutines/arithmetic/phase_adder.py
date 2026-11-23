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
from pennylane.decomposition import (
    add_decomps,
    register_resources,
)
from pennylane.ops.op_math.change_op_basis2 import _change_op_basis_abstract
from pennylane.ops.subcircuit import subcircuit
from pennylane.templates.subroutines.arithmetic.phase_adder import PhaseAdder
from pennylane.templates.subroutines.qft import QFT
from pennylane.typing import Float, Int, Wire
from pennylane.wires import WiresLike


def _add_k_fourier(k, wires: WiresLike):
    """Adds k in the Fourier basis"""
    op_list = []
    for j, wire in enumerate(wires):
        op_list.append(ops.PhaseShift(k * np.pi / (2**j), wires=wire))
    return op_list


def _phase_adder_basis_resources(wires, reverse, **_):
    qft = QFT(Wire[len(wires)])
    return {ops.PhaseShift: len(wires), ops.X: 1, qft if reverse else ops.adjoint(qft): 1}


@subcircuit(
    dynamic_argnames=("k",),
    static_argnames=("reverse",),
    arg_specs={"k": Int, "wires": Wire[-1]},
)
@register_resources(_phase_adder_basis_resources)
def _PhaseAdderBasis(k: int, wires: WiresLike, reverse: bool):
    if reverse:
        ops.X(wires[0])
        QFT(wires=wires)
        _add_k_fourier(k, wires)
    else:
        _add_k_fourier(-k, wires)
        ops.adjoint(QFT(wires=wires))
        ops.X(wires[0])


# pylint: disable-next=unused-argument
def _phase_adder_standard_resources(k, x_wires, mod, work_wire) -> dict:
    num_x_wires = len(x_wires)

    if mod == 2**num_x_wires:
        return {ops.PhaseShift: num_x_wires}

    basis_op1 = _PhaseAdderBasis(Int, wires=Wire[num_x_wires], reverse=False)
    basis_op2 = _PhaseAdderBasis(Int, wires=Wire[num_x_wires], reverse=True)

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


@register_resources(_phase_adder_standard_resources, name="fourier_phase_shifts")
def _phase_adder_standard(k, x_wires: WiresLike, mod, work_wire):

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

    ops.change_op_basis(
        _PhaseAdderBasis(k, wires=x_wires, reverse=False),
        ops.CNOT(wires=[aux_k, work_wire[0]]),
        _PhaseAdderBasis(k, wires=x_wires, reverse=True),
    )


add_decomps(PhaseAdder, _phase_adder_standard)
