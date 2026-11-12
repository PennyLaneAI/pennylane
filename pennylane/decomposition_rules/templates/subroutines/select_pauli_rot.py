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

"""Built-in decomposition rules for templates.subroutines.select_pauli_rot."""

from functools import partial

from pennylane import capture, math
from pennylane.core.operator import abstractify
from pennylane.decomposition import add_decomps, register_resources
from pennylane.ops import CNOT, RZ, Hadamard, S, adjoint, change_op_basis, prod
from pennylane.ops.op_math.adjoint2 import _adjoint_abstract
from pennylane.ops.op_math.change_op_basis2 import _change_op_basis_abstract
from pennylane.templates.state_preparations.mottonen import _apply_uniform_rotation_dagger
from pennylane.templates.subroutines.select_pauli_rot import SelectPauliRot
from pennylane.wires import Wires


# pylint: disable-next=unused-argument
def _select_pauli_rot_resource(angles, control_wires, target_wire, rot_axis):

    num_wires = len(control_wires) + 1
    num_rotations = 2 ** (num_wires - 1)

    rz_rep = abstractify(RZ)
    cnot_rep = abstractify(CNOT)

    prod_res = {
        RZ: num_rotations,
        CNOT: num_rotations if num_wires > 1 else 0,
    }
    if rot_axis == "Z":
        return prod_res

    target_rep = prod(*((cnot_rep, rz_rep) * num_rotations)) if num_wires > 1 else rz_rep

    if rot_axis == "X":
        return {_change_op_basis_abstract(Hadamard, target_rep, Hadamard): 1}

    prod_rep1 = prod(abstractify(Hadamard), _adjoint_abstract(S))
    prod_rep2 = prod(abstractify(S), abstractify(Hadamard))

    return {_change_op_basis_abstract(prod_rep1, target_rep, prod_rep2): 1}


# Not exact resources because rotations might be skipped based on angles
@register_resources(_select_pauli_rot_resource, exact=False)
def decompose_select_pauli_rot(angles, control_wires, target_wire, rot_axis):
    r"""Decomposes the SelectPauliRot"""

    wires = Wires(control_wires) + Wires(target_wire)

    if capture.enabled():
        wires = math.array(wires, like="jax")

    match rot_axis:
        case "X":
            change_op_basis(
                Hadamard(wires[-1]),
                partial(
                    _apply_uniform_rotation_dagger,
                    gate=RZ,
                    alpha=angles,
                    control_wires=wires[-2::-1],
                    target_wire=wires[-1],
                ),
                Hadamard(wires[-1]),
            )
        case "Y":

            def compute(wires):
                adjoint(S(wires[-1]))
                Hadamard(wires[-1])

            def uncompute(wires):
                Hadamard(wires[-1])
                S(wires[-1])

            change_op_basis(
                partial(compute, wires=wires),
                partial(
                    _apply_uniform_rotation_dagger,
                    gate=RZ,
                    alpha=angles,
                    control_wires=wires[-2::-1],
                    target_wire=wires[-1],
                ),
                partial(uncompute, wires=wires),
            )
        case "Z":
            _apply_uniform_rotation_dagger(RZ, angles, wires[-2::-1], wires[-1])


add_decomps(SelectPauliRot, decompose_select_pauli_rot)
