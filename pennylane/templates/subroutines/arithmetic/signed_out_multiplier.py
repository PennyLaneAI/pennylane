# Copyright 2018-2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Contains the SignedOutMultiplier template.
"""

from collections import defaultdict

from .adder import Adder

from pennylane import capture, math
from pennylane.control_flow import for_loop
from pennylane.decomposition import (
    add_decomps,
    controlled_resource_rep,
    register_condition,
    register_resources,
    resource_rep,
)
from pennylane.operation import Operator
from pennylane.ops import CNOT, Controlled, PauliX
from .incrementer import Incrementer
from .out_multiplier import OutMultiplier
from pennylane.wires import Wires, WiresLike


class SignedOutMultiplier(Operator):
    """
    Implements the SignedOutMultiplier template.

    The SignedOutMultiplier simply makes use of the :class:`~.OutMultiplier` template to multiply
    the magnitudes of the encoded inputs, and a quantum comparator on their sign bits to determine
    the final sign of the result.

    The inputs and output are given in 2s complement.

    Args:
        x_wires (Sequence[int]): wires that store the signed integer :math:`x`
        y_wires (Sequence[int]): wires that store the signed integer :math:`y`
        output_wires (Sequence[int]): wires that store the multiplication result. If the
            register is in a non-zero state :math:`z`, the solution will be added to this value
        work_wires (Sequence[int]): auxiliary wires to use for the multiplication. The needed
            number of work wires depends on the decomposition, the register sizes and
            ``output_wires_zeroed``. Defaults to an empty tuple, i.e., no work wires.
    """

    def __init__(
        self,
        x_wires: WiresLike,
        y_wires: WiresLike,
        output_wires: WiresLike,
        work_wires: WiresLike = (),
    ):

        x_wires = Wires(x_wires)
        y_wires = Wires(y_wires)
        output_wires = Wires(output_wires)
        work_wires = Wires(() if work_wires is None else work_wires)

        if len(work_wires) != 0:
            if any(wire in work_wires for wire in x_wires):
                raise ValueError("None of the wires in work_wires should be included in x_wires.")
            if any(wire in work_wires for wire in y_wires):
                raise ValueError("None of the wires in work_wires should be included in y_wires.")

        if any(wire in y_wires for wire in x_wires):
            raise ValueError("None of the wires in y_wires should be included in x_wires.")
        if any(wire in x_wires for wire in output_wires):
            raise ValueError("None of the wires in x_wires should be included in output_wires.")
        if any(wire in y_wires for wire in output_wires):
            raise ValueError("None of the wires in y_wires should be included in output_wires.")

        wires_list = [x_wires, y_wires, output_wires, work_wires]
        wires_name = ["x_wires", "y_wires", "output_wires", "work_wires"]

        for name, wires in zip(wires_name, wires_list):
            self.hyperparameters[name] = Wires(wires)

        # pylint: disable=consider-using-generator
        all_wires = sum([self.hyperparameters[name] for name in wires_name], start=[])
        super().__init__(wires=all_wires)

    @property
    def resource_params(self) -> dict:
        return {
            "num_x_wires": len(self.hyperparameters["x_wires"]),
            "num_y_wires": len(self.hyperparameters["y_wires"]),
            "num_output_wires": len(self.hyperparameters["output_wires"]),
            "num_work_wires": len(self.hyperparameters["work_wires"]),
        }


def _signed_out_multiplier_resources(
    num_x_wires, num_y_wires, num_output_wires, num_work_wires, mod, output_wires_zeroed
):
    """
    Computes the resources for the SignedOutMultiplier.
    Assumes the worst case that both numbers are negative.
    """
    resources = defaultdict(int)
    resources[controlled_resource_rep(PauliX, {}, 1, 0)] = (
        (num_x_wires - 1 + num_y_wires - 1) * 2 + num_output_wires - 1
    )
    resources[
        controlled_resource_rep(Incrementer, {"num_wires": num_x_wires - 1}, num_control_wires=1)
    ] += 2
    resources[
        resource_rep(
            OutMultiplier,
            num_output_wires=num_output_wires - 1,
            num_x_wires=num_x_wires - 1,
            num_y_wires=num_y_wires - 1,
            num_work_wires=num_work_wires,
            mod=mod,
            output_wires_zeroed=output_wires_zeroed,
        )
    ] = 1
    resources[
        controlled_resource_rep(
            Incrementer, {"num_wires": num_output_wires - 1}, num_control_wires=1
        )
    ] += 1
    resources[
        controlled_resource_rep(Incrementer, {"num_wires": num_y_wires - 1}, num_control_wires=1)
    ] += 2
    resources[resource_rep(CNOT)] = 2

    return resources


def _twos_complement_helper(input_reg, aux_wire, work_wires):

    # Invert all bits
    @for_loop(len(input_reg))
    def invert(w):
        # sign bit of 1 indicates a negative value
        CNOT([aux_wire, input_reg[w]])

    invert()  # pylint: disable=no-value-for-parameter

    # Add one
    Controlled(
        Adder(
            k=1,
            x_wires=input_reg,
            work_wires=work_wires,  # we can use the work wires since they are returned in a clean state
        ),
        control_wires=(aux_wire,),
        control_values=(1,),
    )


def _work_wire_condition(num_work_wires, **_):
    return (
        num_work_wires >= 2
    )  # or max(len(x_wires), len(y_wires)) + 1 to use incrementer decomp with work wires


@register_condition(_work_wire_condition)
@register_resources(_signed_out_multiplier_resources, exact=False)
def _signed_out_multiplier_decomposition(
    x_wires: WiresLike,
    y_wires: WiresLike,
    output_wires: WiresLike,
    work_wires: WiresLike,
    **_
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


add_decomps(SignedOutMultiplier, _signed_out_multiplier_decomposition)
