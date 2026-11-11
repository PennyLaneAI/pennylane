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

"""Built-in decomposition rules for templates.subroutines.amplitude_amplification."""

# pylint: disable-msg=too-many-arguments,too-many-positional-arguments

import numpy as np

from pennylane.control_flow import for_loop
from pennylane.core.apply import apply
from pennylane.core.operator import abstractify
from pennylane.decomposition import add_decomps, register_resources, resource_rep
from pennylane.ops import Hadamard, PhaseShift
from pennylane.ops.op_math import ctrl
from pennylane.ops.op_math.controlled2 import _ctrl_abstract
from pennylane.templates.subroutines.amplitude_amplification import AmplitudeAmplification
from pennylane.templates.subroutines.reflection import Reflection
from pennylane.typing import Wire


def _amplitude_amplification_resources(fixed_point, O, iters, num_reflection_wires, U):
    resources = {}

    if fixed_point and iters // 2 > 0:

        resources[Hadamard] = 4 * (iters // 2)
        resources[_ctrl_abstract(O, Wire[1])] = 2 * (iters // 2)
        resources[PhaseShift] = iters // 2
        resources[
            resource_rep(
                Reflection,
                base_rep=abstractify(U),
                num_wires=len(U.wires),
                num_reflection_wires=num_reflection_wires,
            )
        ] = (
            iters // 2
        )
    elif not fixed_point and iters > 0:
        resources[abstractify(O)] = iters
        resources[
            resource_rep(
                Reflection,
                base_rep=abstractify(U),
                num_wires=len(U.wires),
                num_reflection_wires=num_reflection_wires,
            )
        ] = iters

    return resources


@register_resources(_amplitude_amplification_resources)
def _amplitude_amplification_decomposition(
    *_, U, O, iters, fixed_point, work_wire, p_min, reflection_wires, **__
):

    delta = np.sqrt(1 - p_min)
    gamma = np.cos(np.arccos(1 / delta, dtype=np.complex128) / iters, dtype=np.complex128) ** -1

    if fixed_point:

        def alpha(it):
            return np.real(
                2 * np.arctan(1 / (np.tan(2 * np.pi * it / iters) * np.sqrt(1 - gamma**2)))
            )

        def beta(it):
            return -alpha(iters // 2 + 1 - it)

        @for_loop(1, iters // 2 + 1)
        def half_iter_loop(it):
            Hadamard(wires=work_wire)
            ctrl(O, control=work_wire)
            Hadamard(wires=work_wire)
            PhaseShift(beta(it), wires=work_wire)
            Hadamard(wires=work_wire)
            ctrl(O, control=work_wire)
            Hadamard(wires=work_wire)

            Reflection(U, -alpha(it), reflection_wires=reflection_wires)

        half_iter_loop()  # pylint: disable=no-value-for-parameter
    else:

        @for_loop(iters)
        def iter_loop(_):
            apply(O)
            Reflection(U, np.pi, reflection_wires=reflection_wires)

        iter_loop()  # pylint: disable=no-value-for-parameter


add_decomps(AmplitudeAmplification, _amplitude_amplification_decomposition)
