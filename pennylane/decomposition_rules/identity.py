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

"""Built-in decomposition rules for identity."""

import pennylane as qp
from pennylane.core.operator import abstractify
from pennylane.decomposition import add_decomps, register_resources
from pennylane.decomposition_rules._common import null_decomp
from pennylane.decomposition_rules.op_math.adjoint2 import adjoint_rotation as adjoint_rotation2
from pennylane.decomposition_rules.op_math.pow2 import pow_rotation as pow_rotation2
from pennylane.ops.identity import Identity
from pennylane.typing import Wire

add_decomps(Identity, null_decomp)

add_decomps("Adjoint(Identity)", null_decomp)

add_decomps("C(Identity)", null_decomp)

add_decomps("Pow(Identity)", null_decomp)


def _controlled_g_phase_resource(
    base, control_wires, control_values, work_wires, work_wire_type
):  # pylint: disable=unused-argument
    num_control_wires = len(control_wires)
    num_work_wires = len(work_wires)

    resources = {}

    if num_control_wires == 1:  # Worst-case we need a GlobalPhase if control on zero
        resources[qp.PhaseShift] = 1
        resources[qp.GlobalPhase] = 1
        return resources

    # NOTE: Take average case scenario for number of X required for zero control values
    resources[qp.X] = num_control_wires

    resources[
        qp.ctrl(
            abstractify(qp.PhaseShift),
            control=Wire[num_control_wires - 1],
            work_wires=Wire[num_work_wires],
            work_wire_type=work_wire_type,
        )
    ] = 1

    return resources


@register_resources(_controlled_g_phase_resource, exact=False)
def _controlled_g_phase_decomp(
    base,
    control_wires,
    control_values,
    work_wires,
    work_wire_type,
):
    """The decomposition rule for a controlled global phase."""

    if qp.compiler.active() or qp.capture.enabled():
        control_wires = qp.math.array(control_wires, like="jax")
        control_values = qp.math.array(control_values, like="jax")

    if len(control_wires) == 1:

        def _true():
            qp.PhaseShift(-base.phi, wires=control_wires[-1])

        def _false():
            qp.PhaseShift(base.phi, wires=control_wires[-1])
            qp.GlobalPhase(base.phi)

        qp.cond(control_values[0], _true, _false)()
        return

    @qp.for_loop(0, len(control_values))
    def _x_flips(i):
        qp.cond(qp.math.logical_not(control_values[i]), qp.X)(control_wires[i])

    _x_flips()  # pylint: disable=no-value-for-parameter
    qp.ctrl(
        qp.PhaseShift(-base.phi, wires=control_wires[-1]),
        control=control_wires[:-1],
        work_wires=work_wires,
        work_wire_type=work_wire_type,
    )
    _x_flips()  # pylint: disable=no-value-for-parameter


add_decomps("Adjoint(GlobalPhase)", adjoint_rotation2)

add_decomps("Pow(GlobalPhase)", pow_rotation2)

add_decomps("C(GlobalPhase)", _controlled_g_phase_decomp)
