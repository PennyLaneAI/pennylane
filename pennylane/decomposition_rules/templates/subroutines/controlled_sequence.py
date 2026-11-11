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

"""Built-in decomposition rules for templates.subroutines.controlled_sequence."""

from pennylane import capture, compiler, math
from pennylane.control_flow import for_loop
from pennylane.core.operator import abstractify
from pennylane.decomposition import add_decomps, register_resources
from pennylane.ops.op_math import ctrl
from pennylane.ops.op_math.controlled2 import _ctrl_abstract
from pennylane.templates.subroutines.controlled_sequence import ControlledSequence
from pennylane.typing import Wire
from pennylane.wires import Wires


def _ctrl_seq_decomposition_resources(base, control) -> dict:
    control = Wires(control)
    base_rep = abstractify(base)

    if not control:
        return {}
    return {_ctrl_abstract(base_rep, Wire[1]): 2 ** len(control) - 1}


# pylint: disable=no-value-for-parameter
@register_resources(_ctrl_seq_decomposition_resources)
def _ctrl_seq_decomposition(*_, base=None, control=None, **__):
    if compiler.active() or capture.enabled():
        control = math.array(control, like="jax")

    num_controls = len(control)

    @for_loop(num_controls - 1, -1, -1)
    def _powers_loop(i):
        j = num_controls - 1 - i
        ctrl_wire = control[j]

        # Note that we cannot use both Pow here and a for_loop on line 144
        # since the arg i which becomes z will be traced, and Pow expects a static
        # z. Opted for an inner for_loop since it is still compact, rather than
        # an unrolled outer python loop and Pows.
        @for_loop(2**i)
        def _repeat_loop(_):
            ctrl(base, control=ctrl_wire)

        _repeat_loop()

    _powers_loop()


add_decomps(ControlledSequence, _ctrl_seq_decomposition)
