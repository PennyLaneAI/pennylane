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

from collections import Counter

from pennylane import capture, compiler, math
from pennylane.capture.op_eqns import pop_op_eqns
from pennylane.control_flow import for_loop
from pennylane.core.operator import Operator2, abstractify
from pennylane.decomposition import add_decomps, register_resources
from pennylane.decomposition.resources import CompressedResourceOp
from pennylane.ops.op_math import ctrl, pow
from pennylane.ops.op_math.controlled2 import _ctrl_abstract
from pennylane.ops.op_math.pow2 import _pow_abstract
from pennylane.templates.subroutines.controlled_sequence import ControlledSequence
from pennylane.typing import Wire
from pennylane.wires import Wires


def _ctrl_seq_execution_optimized_decomposition_resources(base, control) -> dict:
    control = Wires(control)
    base_rep = abstractify(base)

    if not control:
        return {}
    controlled_base = _ctrl_abstract(base_rep, Wire[1])
    if isinstance(controlled_base, CompressedResourceOp):
        return dict(Counter(_pow_abstract(controlled_base, 2**i) for i in range(len(control))))
    return dict(
        Counter(abstractify(pow(controlled_base, 2**i, lazy=False)) for i in range(len(control)))
    )


@register_resources(_ctrl_seq_execution_optimized_decomposition_resources, name="controlled_powers")
def _ctrl_seq_execution_optimized_decomposition(*_, base=None, control=None, **__):
    """
    A decomposition optimized for execution time, over IR size and compilation time.

    Using Pows with different z grows the IR but the execution is faster because the powers
    can be simplified eagerly to a single rotation each. Additionally, the Pows have more
    decomposition variants which increases compile time.
    """
    if compiler.active() or capture.enabled():
        control = math.array(control, like="jax")

    num_controls = len(control)
    # To match legacy behaviour
    lazy = not isinstance(base, Operator2)

    for i in range(num_controls - 1, -1, -1):
        j = num_controls - 1 - i
        ctrl_wire = control[j]

        controlled_base = ctrl(base, control=ctrl_wire)
        if capture.enabled():
            pop_op_eqns((controlled_base,))
        pow(controlled_base, 2**i, lazy=lazy)


def _ctrl_seq_ir_optimized_resources(base, control) -> dict:
    control = Wires(control)
    if not control:
        return {}
    controlled_base = _ctrl_abstract(abstractify(base), Wire[1])
    return {controlled_base: 2 ** len(control) - 1}


@register_resources(_ctrl_seq_ir_optimized_resources, name="looped_controlled_repetitions")
def _ctrl_seq_ir_optimized_decomposition(*_, base=None, control=None, **__):
    """
    A decomposition optimized for IR size and compilation time, over execution time.

    Using the nested @for_loop keeps the IR compact, but the runtime increases since
    there is no simplification of repeated operators. They are all executed individually.
    """
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

        _repeat_loop()  # pylint: disable=no-value-for-parameter

    _powers_loop()  # pylint: disable=no-value-for-parameter


# TODO: worth exploring how Catalyst can handle these situations (execution versus compilation optimization)
add_decomps(
    ControlledSequence,
    _ctrl_seq_execution_optimized_decomposition,
    _ctrl_seq_ir_optimized_decomposition,
)
