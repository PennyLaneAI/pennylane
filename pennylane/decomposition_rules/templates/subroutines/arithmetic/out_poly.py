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

"""Built-in decomposition rules for templates.subroutines.arithmetic.out_poly."""

from collections import Counter

from pennylane.decomposition import add_decomps, register_resources
from pennylane.ops import adjoint, ctrl
from pennylane.ops.op_math.adjoint2 import _adjoint_abstract
from pennylane.templates.subroutines.arithmetic.out_poly import OutPoly
from pennylane.templates.subroutines.arithmetic.phase_adder import PhaseAdder
from pennylane.templates.subroutines.qft import QFT
from pennylane.typing import Int, Wire
from pennylane.wires import WiresLike


def _out_poly_decomposition_resources(num_output_wires, num_work_wires, mod, coeffs_list) -> dict:
    num_output_adder_mod = num_output_wires + 1 if num_work_wires else num_output_wires

    resources = Counter({QFT(Wire[num_output_adder_mod]): 1})

    coeffs_dic = dict(coeffs_list)
    phase_rep = PhaseAdder(
        Int, Wire[num_output_adder_mod], mod, Wire[int(mod != 2**num_output_adder_mod)]
    )

    for item in coeffs_dic:

        if 1 not in item:
            resources[phase_rep] += 1
        else:
            num_controls = sum(1 for bit in item if bit == 1)

            ctrl_phase_rep = ctrl(
                phase_rep,
                control=Wire[num_controls],
                work_wires=Wire[int(num_work_wires > 0)],
                work_wire_type="borrowed",
            )
            resources[ctrl_phase_rep] += 1

    resources[_adjoint_abstract(QFT(Wire[num_output_adder_mod]))] = 1

    return dict(resources)


@register_resources(_out_poly_decomposition_resources)
def _out_poly_decomposition(
    polynomial_function,
    input_registers,
    output_wires: WiresLike,
    mod=None,
    work_wires: WiresLike = (),
    **kwargs,
):  # pylint: disable=unused-argument
    registers_wires = [*input_registers, output_wires]

    if len(work_wires) == 0:
        work_wires = [(), ()]

    output_adder_mod = (
        [work_wires[0]] + registers_wires[-1] if work_wires[0] else registers_wires[-1]
    )

    QFT(wires=output_adder_mod)

    coeffs_dic = dict(kwargs["coeffs_list"])

    all_wires_input = sum([*registers_wires[:-1]], start=[])

    for item, coeff in coeffs_dic.items():

        if 1 not in item:
            # Add the constant term
            PhaseAdder(int(coeff), output_adder_mod)
        else:
            controls = [all_wires_input[i] for i, bit in enumerate(item) if bit == 1]

            ctrl(
                PhaseAdder(
                    int(coeff) % mod,
                    output_adder_mod,
                    work_wire=work_wires[1],
                    mod=mod,
                ),
                control=controls,
            )

    adjoint(QFT(wires=output_adder_mod))


add_decomps(OutPoly, _out_poly_decomposition)
