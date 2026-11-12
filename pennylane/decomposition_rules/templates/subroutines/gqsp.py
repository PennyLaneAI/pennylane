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

"""Built-in decomposition rules for templates.subroutines.gqsp."""

from pennylane import capture, compiler, math, ops
from pennylane.control_flow import for_loop
from pennylane.core.operator import abstractify
from pennylane.decomposition import add_decomps, register_resources
from pennylane.ops.op_math.adjoint2 import _adjoint_abstract
from pennylane.ops.op_math.controlled2 import _ctrl_abstract
from pennylane.ops.op_math.pow2 import _pow_abstract
from pennylane.templates.subroutines.gqsp import GQSP
from pennylane.typing import Wire


def _num_shifted_ctrl_ops(shift, num_ctrl_ops):
    """Number of controlled unitaries that absorb a negative shift, and the remaining powers
    of the unitary (positive: ``U``, negative: ``adjoint(U)``) applied outside of them."""
    if shift >= 0:
        return 0, shift
    num_absorbed = min(-shift, num_ctrl_ops)
    return num_absorbed, shift + num_absorbed


def _GQSP_resources(
    unitary, angles, control, work_wires, work_wire_type, shift
):  # pylint: disable=unused-argument,too-many-arguments
    num_iters = angles.shape[1]
    num_absorbed, remaining = _num_shifted_ctrl_ops(shift, num_iters - 1)
    unitary = abstractify(unitary)
    ctrl_kwargs = {"work_wires": Wire[len(work_wires)], "work_wire_type": work_wire_type}

    resources = {
        ops.X: 2 + 2 * (num_iters - 1),
        ops.U3: num_iters,
        ops.Z: num_iters,
    }
    if num_iters - 1 - num_absorbed > 0:
        resources[_ctrl_abstract(unitary, Wire[1], num_zero_control_values=1, **ctrl_kwargs)] = (
            num_iters - 1 - num_absorbed
        )
    if num_absorbed > 0:
        resources[_ctrl_abstract(_adjoint_abstract(unitary), Wire[1], **ctrl_kwargs)] = num_absorbed
    if remaining > 0:
        resources[_pow_abstract(unitary, remaining)] = 1
    elif remaining < 0:
        resources[_pow_abstract(_adjoint_abstract(unitary), -remaining)] = 1
    return resources


@register_resources(_GQSP_resources)
def _GQSP_decomposition(
    unitary, angles, control, work_wires, work_wire_type, shift
):  # pylint: disable=too-many-arguments
    if compiler.active() or capture.enabled():
        angles = math.array(angles, like="jax")

    thetas, phis, lambdas = angles[0], angles[1], angles[2]
    num_absorbed, remaining = _num_shifted_ctrl_ops(shift, len(thetas) - 1)

    # Powers of the unitary that cannot be absorbed into the controlled unitaries. These
    # commute with every other gate of the circuit since they are powers of the same operator.
    if remaining > 0:
        ops.pow(unitary, remaining)
    elif remaining < 0:
        ops.pow(ops.adjoint(unitary), -remaining)

    # These four gates adapt PennyLane's ops.U3 to the chosen U3 format in the GQSP paper.
    ops.X(control)
    ops.U3(2 * thetas[0], phis[0], lambdas[0], wires=control)
    ops.X(control)
    ops.Z(control)

    def _signal_processing_rotation(i):
        ops.X(control)
        ops.U3(2 * thetas[i], phis[i], lambdas[i], wires=control)
        ops.X(control)
        ops.Z(control)

    # |0><0| x I + |1><1| x U^dagger = (I x U^dagger) (|0><0| x U + |1><1| x I), so each of
    # these iterations shifts the polynomial by -1 (Theorem 6 of arXiv:2308.01501).
    @for_loop(1, 1 + num_absorbed)
    def shifted_gqsp_loop(i):
        ops.ctrl(
            ops.adjoint(unitary),
            control=control,
            control_values=[1],
            work_wires=work_wires,
            work_wire_type=work_wire_type,
        )
        _signal_processing_rotation(i)

    @for_loop(1 + num_absorbed, len(thetas))
    def gqsp_loop(i):
        ops.ctrl(
            unitary,
            control=control,
            control_values=[0],
            work_wires=work_wires,
            work_wire_type=work_wire_type,
        )
        _signal_processing_rotation(i)

    shifted_gqsp_loop()  # pylint:disable= no-value-for-parameter
    gqsp_loop()  # pylint:disable= no-value-for-parameter


add_decomps(GQSP, _GQSP_decomposition)
