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


"""Built-in decomposition rules for templates.subroutines.arithmetic.out_adder."""

from collections import defaultdict

from pennylane.decomposition import (
    add_decomps,
    register_resources,
)
from pennylane.ops import change_op_basis
from pennylane.ops.op_math.adjoint2 import _adjoint_abstract
from pennylane.ops.op_math.change_op_basis2 import _change_op_basis_abstract
from pennylane.ops.subcircuit import subcircuit
from pennylane.templates.subroutines.arithmetic.out_adder import OutAdder
from pennylane.templates.subroutines.arithmetic.phase_adder import PhaseAdder
from pennylane.templates.subroutines.controlled_sequence import ControlledSequence
from pennylane.templates.subroutines.qft import QFT
from pennylane.typing import Int, Wire
from pennylane.wires import WiresLike


def _out_adder_target_resources(x_wires, y_wires, output_wires, mod, work_wires) -> dict:
    target_resources = defaultdict(int)
    for control_wires in (x_wires, y_wires):
        target_resources[
            ControlledSequence(
                base=PhaseAdder(Int, Wire[len(output_wires)], mod, Wire[len(work_wires)]),
                control=Wire[len(control_wires)],
            )
        ] += 1
    return dict(target_resources)


@subcircuit(
    wire_argnames=("x_wires", "y_wires", "output_wires", "work_wires"),
    compilable_argnames=("mod",),
)
@register_resources(_out_adder_target_resources)
def _OutAdderTarget(
    x_wires: WiresLike,
    y_wires: WiresLike,
    output_wires: WiresLike,
    mod: int,
    work_wires: WiresLike,
):
    """Add both input registers into an output register in the Fourier basis."""
    ControlledSequence(PhaseAdder(1, output_wires, mod, work_wires), control=x_wires)
    ControlledSequence(PhaseAdder(1, output_wires, mod, work_wires), control=y_wires)


def _out_adder_standard_resources(num_output_wires, num_x_wires, num_y_wires, mod) -> dict:
    num_qft_wires = num_output_wires if mod == 2**num_output_wires else num_output_wires + 1
    num_work_wires = int(mod != 2**num_output_wires)
    _compute_op = QFT(Wire[num_qft_wires])
    resources = {
        _change_op_basis_abstract(
            _compute_op,
            _OutAdderTarget(
                Wire[num_x_wires], Wire[num_y_wires], Wire[num_qft_wires], mod, Wire[num_work_wires]
            ),
            _adjoint_abstract(_compute_op),
        ): 1
    }
    return resources


@register_resources(_out_adder_standard_resources, name="fourier_phase_adders")
def _out_adder_standard(x_wires, y_wires, output_wires, mod, work_wires, **__):
    if mod != 2 ** len(output_wires) and mod is not None:
        qft_new_output_wires = work_wires[:1] + output_wires
        work_wire = work_wires[1:]
    else:
        qft_new_output_wires = output_wires
        work_wire = ()

    target = _OutAdderTarget(x_wires, y_wires, qft_new_output_wires, mod, work_wire)
    change_op_basis(QFT(wires=qft_new_output_wires), target)


add_decomps(OutAdder, _out_adder_standard)
