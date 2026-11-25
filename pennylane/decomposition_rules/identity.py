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

from pennylane import capture, compiler, control_flow, math, ops
from pennylane.core.operator import abstractify
from pennylane.decomposition import add_decomps, register_resources
from pennylane.decomposition_rules.common import null_decomp
from pennylane.ops.identity import GlobalPhase, Identity
from pennylane.ops.op_math.adjoint2 import _adjoint_negate_angle
from pennylane.ops.op_math.pow2 import _pow_scale_angle
from pennylane.typing import Wire

add_decomps(Identity, null_decomp)
add_decomps("Adjoint(Identity)", null_decomp)
add_decomps("C(Identity)", null_decomp)
add_decomps("Pow(Identity)", null_decomp)


def _controlled_g_phase_standard_resources(  # pylint: disable=unused-argument
    base,
    control_wires,
    control_values,
    work_wires,
    work_wire_type,
):
    num_control_wires = len(control_wires)
    num_work_wires = len(work_wires)

    resources = {}

    if num_control_wires == 1:  # Worst-case we need a GlobalPhase if control on zero
        resources[ops.PhaseShift] = 1
        resources[GlobalPhase] = 1
        return resources

    # NOTE: Take average case scenario for number of X required for zero control values
    resources[ops.X] = num_control_wires

    resources[
        ops.ctrl(
            abstractify(ops.PhaseShift),
            control=Wire[num_control_wires - 1],
            work_wires=Wire[num_work_wires],
            work_wire_type=work_wire_type,
        )
    ] = 1

    return resources


@register_resources(
    _controlled_g_phase_standard_resources, exact=False, name="to_controlled_phaseshift"
)
def controlled_g_phase_standard(base, control_wires, control_values, work_wires, work_wire_type):
    """The decomposition rule for a controlled global phase."""

    if compiler.active() or capture.enabled():
        control_wires = math.array(control_wires, like="jax")
        control_values = math.array(control_values, like="jax")

    if len(control_wires) == 1:

        def _true():
            ops.PhaseShift(-base.phi, wires=control_wires[-1])

        def _false():
            ops.PhaseShift(base.phi, wires=control_wires[-1])
            GlobalPhase(base.phi)

        ops.cond(control_values[0], _true, _false)()
        return

    @control_flow.for_loop(0, len(control_values))
    def _x_flips(i):
        ops.cond(math.logical_not(control_values[i]), ops.X)(control_wires[i])

    _x_flips()  # pylint: disable=no-value-for-parameter
    ops.ctrl(
        ops.PhaseShift(-base.phi, wires=control_wires[-1]),
        control=control_wires[:-1],
        work_wires=work_wires,
        work_wire_type=work_wire_type,
    )
    _x_flips()  # pylint: disable=no-value-for-parameter


add_decomps("Adjoint(GlobalPhase)", _adjoint_negate_angle)
add_decomps("Pow(GlobalPhase)", _pow_scale_angle)
add_decomps("C(GlobalPhase)", controlled_g_phase_standard)
