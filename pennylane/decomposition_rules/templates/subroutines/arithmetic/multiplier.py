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

"""Built-in decomposition rules for templates.subroutines.arithmetic.multiplier."""

from pennylane.decomposition import add_decomps, register_resources
from pennylane.ops import SWAP, adjoint, change_op_basis, prod
from pennylane.ops.op_math.adjoint2 import _adjoint_abstract
from pennylane.ops.op_math.change_op_basis2 import _change_op_basis_abstract
from pennylane.templates.subroutines.arithmetic.multiplier import Multiplier
from pennylane.templates.subroutines.arithmetic.phase_adder import PhaseAdder
from pennylane.templates.subroutines.controlled_sequence import ControlledSequence
from pennylane.templates.subroutines.qft import QFT
from pennylane.typing import Int, Wire
from pennylane.wires import WiresLike


def _multiplier_decomposition_resources(
    num_x_wires,
    mod,
    num_work_wires,
) -> dict:
    num_wires_aux = num_work_wires - 1 if mod != 2**num_x_wires else num_x_wires
    controlled_sequence_rep = ControlledSequence(
        base=PhaseAdder(Int, Wire[num_wires_aux], mod, Wire[int(mod != 2**num_wires_aux)]),
        control=Wire[num_x_wires],
    )
    target_op_rep = prod(*[SWAP(Wire[2]) for _ in range(num_x_wires)])
    _compute_op = QFT(Wire[num_wires_aux])
    resources = {
        _change_op_basis_abstract(
            _compute_op,
            controlled_sequence_rep,
            adjoint(_compute_op),
        ): 1,
        target_op_rep: 1,
        _change_op_basis_abstract(
            _compute_op,
            adjoint(controlled_sequence_rep),
            _adjoint_abstract(_compute_op),
        ): 1,
    }
    return resources


@register_resources(_multiplier_decomposition_resources)
def _multiplier_decomposition(k, x_wires: WiresLike, mod, work_wires: WiresLike, **__):
    if mod != 2 ** len(x_wires):
        work_wire_aux = work_wires[:1]
        wires_aux = work_wires[1:]
        wires_aux_swap = wires_aux[1:]
    else:
        work_wire_aux = ()
        wires_aux = work_wires[: len(x_wires)]
        wires_aux_swap = wires_aux

    inv_k = pow(k, -1, mod)
    change_op_basis(
        QFT(wires=wires_aux),
        ControlledSequence(PhaseAdder(k, wires_aux, mod, work_wire_aux), control=x_wires),
    )
    prod(
        *list(
            reversed(
                [
                    SWAP(wires=[x_wire, aux_wire])
                    for x_wire, aux_wire in zip(x_wires, wires_aux_swap, strict=True)
                ]
            )
        )
    )
    change_op_basis(
        QFT(wires=wires_aux),
        adjoint(
            ControlledSequence(PhaseAdder(inv_k, wires_aux, mod, work_wire_aux), control=x_wires)
        ),
    )


add_decomps(Multiplier, _multiplier_decomposition)
