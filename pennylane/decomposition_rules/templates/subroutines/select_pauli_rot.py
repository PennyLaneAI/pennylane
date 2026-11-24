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


from pennylane import capture, math
from pennylane.decomposition import (
    add_decomps,
    register_resources,
)
from pennylane.ops import CNOT, RZ, Hadamard, S, adjoint, change_op_basis
from pennylane.ops.op_math.adjoint2 import _adjoint_abstract
from pennylane.ops.op_math.change_op_basis2 import _change_op_basis_abstract
from pennylane.ops.subcircuit import subcircuit
from pennylane.templates.state_preparations.mottonen import _apply_uniform_rotation_dagger
from pennylane.templates.subroutines.select_pauli_rot import SelectPauliRot
from pennylane.typing import Float, Wire
from pennylane.wires import Wires


def _uniform_rz_resources(control_wires, **_):
    """Resources for a uniformly controlled Z rotation."""
    num_rotations = 2 ** len(control_wires)
    return {
        RZ: num_rotations,
        CNOT: num_rotations if len(control_wires) else 0,
    }


@subcircuit(
    dynamic_argnames=("angles",),
    wire_argnames=("control_wires", "target_wire"),
    arg_specs={"angles": Float[-1], "control_wires": Wire[-1], "target_wire": Wire[1]},
    ndim_params=(1,),
    grad_method=None,
)
@register_resources(_uniform_rz_resources, exact=False)
def _UniformRZ(angles, control_wires, target_wire) -> None:
    """Apply uniformly controlled Z rotations using the reversed control register."""
    _apply_uniform_rotation_dagger(RZ, angles, control_wires[::-1], target_wire[0])


@subcircuit(static_argnames=("reverse",), arg_specs={"wires": Wire[1]})
@register_resources(lambda reverse, **_: {Hadamard: 1, S if reverse else _adjoint_abstract(S): 1})
def _SelectPauliRotYBasis(wires, reverse: bool) -> None:
    """Change between the Y and Z bases."""
    if reverse:
        Hadamard(wires)
        S(wires)
    else:
        adjoint(S(wires))
        Hadamard(wires)


def _select_pauli_rot_standard_resources(angles, control_wires, target_wire, rot_axis):
    if rot_axis == "Z":
        return _uniform_rz_resources(control_wires)

    target_rep = _UniformRZ(angles, control_wires, target_wire)

    if rot_axis == "X":
        return {_change_op_basis_abstract(Hadamard, target_rep, Hadamard): 1}

    return {
        _change_op_basis_abstract(
            _SelectPauliRotYBasis(Wire[1], reverse=False),
            target_rep,
            _SelectPauliRotYBasis(Wire[1], reverse=True),
        ): 1
    }


# Not exact resources because rotations might be skipped based on angles
@register_resources(_select_pauli_rot_standard_resources, exact=False, name="gray_code")
def select_pauli_rot_standard(angles, control_wires, target_wire, rot_axis):
    r"""Decomposes the SelectPauliRot"""

    wires = Wires(control_wires) + Wires(target_wire)

    if capture.enabled():
        wires = math.array(wires, like="jax")

    match rot_axis:
        case "X":
            change_op_basis(
                Hadamard(wires[-1]),
                _UniformRZ(angles, control_wires, target_wire),
                Hadamard(wires[-1]),
            )
        case "Y":
            change_op_basis(
                _SelectPauliRotYBasis(target_wire, reverse=False),
                _UniformRZ(angles, control_wires, target_wire),
                _SelectPauliRotYBasis(target_wire, reverse=True),
            )
        case "Z":
            _apply_uniform_rotation_dagger(RZ, angles, wires[-2::-1], wires[-1])


add_decomps(SelectPauliRot, select_pauli_rot_standard)
