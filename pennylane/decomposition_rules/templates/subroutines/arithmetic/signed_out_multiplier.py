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

"""Built-in decomposition rules for templates.subroutines.arithmetic.signed_out_multiplier."""

from collections import defaultdict

from jax import numpy as jnp

from pennylane import capture, compiler, math
from pennylane.control_flow import for_loop
from pennylane.decomposition import add_decomps, register_condition, register_resources
from pennylane.ops import CNOT, ctrl
from pennylane.templates.subroutines.arithmetic.incrementer import Incrementer
from pennylane.templates.subroutines.arithmetic.out_multiplier import OutMultiplier
from pennylane.templates.subroutines.arithmetic.semi_adder import SemiAdder
from pennylane.templates.subroutines.arithmetic.signed_out_multiplier import SignedOutMultiplier
from pennylane.typing import Wire
from pennylane.wires import Wires, WiresLike, is_abstract_qubit


def _zeroed_signed_out_multiplier_resources(
    x_wires, y_wires, output_wires, work_wires, output_wires_zeroed=False
):  # pylint: disable=unused-argument
    """
    Computes the resources for the SignedOutMultiplier.
    """
    num_x_wires = len(x_wires)
    num_y_wires = len(y_wires)
    num_output_wires = len(output_wires)
    num_work_wires = len(work_wires)
    resources = defaultdict(int)
    resources[
        ctrl(Incrementer(Wire[num_x_wires], work_wires=Wire[num_work_wires - 2]), Wire[1])
    ] += 2
    resources[
        OutMultiplier(
            Wire[num_x_wires],
            Wire[num_y_wires],
            Wire[num_output_wires - 1],
            mod=2 ** (num_output_wires - 1),
            work_wires=Wire[num_work_wires - 2],
            output_wires_zeroed=True,
        )
    ] += 1
    resources[
        ctrl(Incrementer(Wire[num_output_wires - 1], work_wires=Wire[num_work_wires - 2]), Wire[1])
    ] += 1
    resources[
        ctrl(Incrementer(Wire[num_y_wires], work_wires=Wire[num_work_wires - 2]), Wire[1])
    ] += 2
    resources[CNOT] = 6 + (num_x_wires + num_y_wires) * 2 + (num_output_wires - 1)
    # Convert to a builtin dict so downstream lookups of missing gates
    # don't accidentally insert them with a value of 0
    return dict(resources)


def _not_zeroed_signed_out_multiplier_resources(
    x_wires, y_wires, output_wires, work_wires, output_wires_zeroed=False
):  # pylint: disable=unused-argument
    """
    Computes the resources for the SignedOutMultiplier.
    """
    num_x_wires = len(x_wires)
    num_y_wires = len(y_wires)
    num_output_wires = len(output_wires)
    num_work_wires = len(work_wires)
    resources = {}

    resources[
        SignedOutMultiplier(
            Wire[num_x_wires],
            Wire[num_y_wires],
            Wire[num_output_wires],
            work_wires=Wire[2 + num_work_wires - (2 * num_output_wires + 1)],
            output_wires_zeroed=True,
        )
    ] = 1

    resources[
        SemiAdder(Wire[num_output_wires], Wire[num_output_wires], Wire[num_output_wires - 1])
    ] = 1

    return resources


def _twos_complement_helper(input_reg, aux_wire, work_wires):
    # Dynamically-allocated wires (``AbstractQubit``) cannot be
    # stacked into a numeric array for ``for_loop``-based dynamic indexing, so unroll instead.
    # Remove this fallback once dynamic-wire indexing is supported (shortcut.com/story/129521).
    if any(is_abstract_qubit(w) for w in input_reg):
        for wire in input_reg:
            CNOT([aux_wire, wire])
    else:
        if compiler.active() or capture.enabled():
            input_reg_arr = math.array(input_reg, like="jax")
        else:
            input_reg_arr = input_reg

        # Invert all bits
        @for_loop(len(input_reg_arr))
        def invert(w):
            # sign bit of 1 indicates a negative value
            CNOT([aux_wire, input_reg_arr[w]])

        invert()  # pylint: disable=no-value-for-parameter

    # Add one
    ctrl(
        Incrementer(
            wires=Wires(input_reg),
            work_wires=Wires(
                work_wires
            ),  # we can use the work wires since they are returned in a clean state
        ),
        control=aux_wire,
        control_values=(1,),
    )


def _not_zeroed_work_wire_condition(
    x_wires, y_wires, output_wires, work_wires, output_wires_zeroed=False
):  # pylint: disable=unused-argument
    return len(work_wires) >= 2 * len(output_wires) + 1


def _zeroed_work_wire_condition(
    x_wires, y_wires, output_wires, work_wires, output_wires_zeroed=False
):  # pylint: disable=unused-argument
    return len(work_wires) >= 2


def _zeroed_condition(
    x_wires, y_wires, output_wires, work_wires, output_wires_zeroed=False
):  # pylint: disable=unused-argument
    return output_wires_zeroed


def _not_zeroed_condition(
    x_wires, y_wires, output_wires, work_wires, output_wires_zeroed=False
):  # pylint: disable=unused-argument
    return not output_wires_zeroed


@register_condition(_zeroed_condition)
@register_condition(_zeroed_work_wire_condition)
@register_resources(_zeroed_signed_out_multiplier_resources)
def _signed_out_multiplier_decomposition_zeroed(
    x_wires: WiresLike,
    y_wires: WiresLike,
    output_wires: WiresLike,
    work_wires: WiresLike,
    output_wires_zeroed=False,  # pylint: disable=unused-argument
):
    """Computes the decomposition of the operator as a product of other operators when the output wires are zeroed."""
    if capture.enabled():
        x_wires, y_wires, work_wires, output_wires = (
            math.array(x_wires, like="jax"),
            math.array(y_wires, like="jax"),
            math.array(work_wires, like="jax"),
            math.array(output_wires, like="jax"),
        )

    x_aux = work_wires[0]
    y_aux = work_wires[1]

    # Sign extension
    CNOT([x_wires[0], x_aux])
    CNOT([y_wires[0], y_aux])

    # Take 2s complements if necessary
    _twos_complement_helper(x_wires, x_aux, work_wires[2:])
    _twos_complement_helper(y_wires, y_aux, work_wires[2:])

    # at this point the sign is only kept in the auxiliary qubits' states

    # Multiply the magnitudes
    OutMultiplier(
        x_wires,
        y_wires,
        output_wires[1:],
        work_wires=work_wires[2:],
        output_wires_zeroed=True,
    )

    # Compute the sign
    CNOT([x_aux, output_wires[0]])
    CNOT([y_aux, output_wires[0]])

    # Encode the output
    _twos_complement_helper(output_wires[1:], output_wires[0], work_wires[2:])

    # Return inputs to original state
    _twos_complement_helper(x_wires, x_aux, work_wires[2:])
    _twos_complement_helper(y_wires, y_aux, work_wires[2:])

    # Uncompute sign extension
    CNOT([x_wires[0], x_aux])
    CNOT([y_wires[0], y_aux])


@register_condition(_not_zeroed_condition)
@register_condition(_not_zeroed_work_wire_condition)
@register_resources(_not_zeroed_signed_out_multiplier_resources)
def _signed_out_multiplier_decomposition_not_zeroed(
    x_wires: WiresLike,
    y_wires: WiresLike,
    output_wires: WiresLike,
    work_wires: WiresLike,
    output_wires_zeroed=False,  # pylint: disable=unused-argument
):
    """Computes the decomposition of the operator as a product of other operators."""

    if capture.enabled():
        x_wires, y_wires, work_wires, output_wires = (
            math.array(x_wires, like="jax"),
            math.array(y_wires, like="jax"),
            math.array(work_wires, like="jax"),
            math.array(output_wires, like="jax"),
        )

    x_aux = work_wires[0]
    y_aux = work_wires[1]

    # Temp output register for multiplication output
    mult_temp = work_wires[2 : len(output_wires) + 2]

    if capture.enabled():
        signed_work_wires = work_wires[2 * len(output_wires) + 1 :]
        signed_work_wires = jnp.concatenate([jnp.atleast_1d(y_aux), signed_work_wires])
        signed_work_wires = jnp.concatenate([jnp.atleast_1d(x_aux), signed_work_wires])
    else:
        signed_work_wires = [x_aux] + [y_aux] + work_wires[2 * len(output_wires) + 1 :]

    SignedOutMultiplier(x_wires, y_wires, mult_temp, signed_work_wires, True)

    # Add any initial value in the output register
    SemiAdder(
        mult_temp,
        output_wires,
        work_wires=work_wires[len(output_wires) + 2 : 2 * len(output_wires) + 1],
    )


add_decomps(SignedOutMultiplier, _signed_out_multiplier_decomposition_not_zeroed)

add_decomps(SignedOutMultiplier, _signed_out_multiplier_decomposition_zeroed)
