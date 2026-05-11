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
from pennylane import measure
from pennylane.ops import PauliX, Toffoli, Controlled, CNOT

from pennylane.templates import OutMultiplier, Adder
from pennylane.templates.subroutines.arithmetic.out_multiplier import _increment

from pennylane.wires import Wires, WiresLike

from pennylane.operation import Operator


class SignedOutMultiplier(Operator):
    """
    Implements the SignedOutMultiplier template.

    The SignedOutMultiplier simply makes use of the :class:`~.OutMultiplier` template to multiply
    the magnitudes of the encoded inputs, and a quantum comparator on their sign bits to determine
    the final sign of the result.

    If the inputs are given in 2s complement, the sign bits are recorded before their 2s complements
    are taken, and they can be multiplied by the approach specified above.

    Args:
        x_wires (Sequence[int]): wires that store the signed integer :math:`x`
        y_wires (Sequence[int]): wires that store the signed integer :math:`y`
        output_wires (Sequence[int]): wires that store the multiplication result. If the
            register is in a non-zero state :math:`z`, the solution will be added to this value
        mod (int): the modulo for performing the multiplication. If not provided, it will be set
            to its maximum value, :math:`2^{\text{len(output_wires)}}`
        work_wires (Sequence[int]): auxiliary wires to use for the multiplication. The needed
            number of work wires depends on the decomposition, the register sizes and
            ``output_wires_zeroed``. Defaults to an empty tuple, i.e., no work wires.
        output_wires_zeroed (bool): Whether the ``output_wires`` are guaranteed to be in state
            :math:`|0\rangle` initially. Setting this argument to ``True`` reduces the cost of
            the operation.
        twos_complement (bool): If ``True``, the inputs are taken to be encoded in two's complement
    """

    def __init__(
        self,
        x_wires: WiresLike,
        y_wires: WiresLike,
        output_wires: WiresLike,
        mod=None,
        work_wires: WiresLike = (),
        output_wires_zeroed: bool = False,
        twos_complement: bool = False,
        id=None,
    ):

        x_wires = Wires(x_wires)
        y_wires = Wires(y_wires)
        output_wires = Wires(output_wires)
        work_wires = Wires(() if work_wires is None else work_wires)

        num_work_wires = len(work_wires)

        if mod is None:
            mod = 2 ** len(output_wires)
        if mod != 2 ** len(output_wires):
            if num_work_wires < 2:
                raise ValueError(
                    f"If mod is not 2^{len(output_wires)}, at least two work wires should be provided."
                )
            work_wires = work_wires[:2]
        if mod > 2 ** (len(output_wires)):
            raise ValueError(
                "OutMultiplier must have enough wires to represent mod. The maximum mod "
                f"with len(output_wires)={len(output_wires)} is {2 ** len(output_wires)}, but received {mod}."
            )

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
        self.hyperparameters["mod"] = mod
        self.hyperparameters["output_wires_zeroed"] = output_wires_zeroed
        self.hyperparameters["twos_complement"] = twos_complement

        # pylint: disable=consider-using-generator
        all_wires = sum([self.hyperparameters[name] for name in wires_name], start=[])
        super().__init__(wires=all_wires, id=id)

    @property
    def resource_params(self) -> dict:
        return {
            "num_output_wires": len(self.hyperparameters["output_wires"]),
            "num_x_wires": len(self.hyperparameters["x_wires"]),
            "num_y_wires": len(self.hyperparameters["y_wires"]),
            "num_work_wires": len(self.hyperparameters["work_wires"]),
            "mod": self.hyperparameters["mod"],
            "output_wires_zeroed": self.hyperparameters["output_wires_zeroed"],
            "twos_complement": self.hyperparameters["twos_complement"],
        }

    @staticmethod
    def compute_decomposition(
        x_wires: WiresLike,
        y_wires: WiresLike,
        output_wires: WiresLike,
        mod,
        work_wires: WiresLike,
        output_wires_zeroed: bool = False,
        twos_complement: bool = False,
    ):  # pylint: disable=arguments-differ, too-many-arguments, unused-argument
        """Computes the decomposition of the operator as a product of other operators."""

        def twos_complement_helper(input_reg):
            # Invert all bits
            for w in input_reg[1:]:
                # sign bit of 1 indicates a negative value
                Controlled(PauliX(w), control_wires=(input_reg[0],), control_values=(1,))

            # Add one
            Controlled(
                # TODO: make Incrementer a Template
                _increment(
                    wires=input_reg[1:],
                    work_wires=work_wires,
                ),
                control_wires=(input_reg[0],),
                control_values=(1,),
            )

        # Compute the sign
        CNOT([x_wires[0], output_wires[0]])
        CNOT([y_wires[0], output_wires[0]])

        # Take 2s complements if necessary
        for input_reg in [x_wires, y_wires]:
            twos_complement_helper(input_reg)

            # we would reset the sign bit here to complete the two's complement,
            # but we do not, so that we can remember the sign

        # Multiply the magnitudes
        OutMultiplier(
            x_wires[1:],
            y_wires[1:],
            output_wires[1:],
            mod=mod,
            work_wires=work_wires,
            output_wires_zeroed=output_wires_zeroed,
        )

        # Encode the output
        twos_complement_helper(output_wires)

        # Return inputs to original state
        for input_reg in [x_wires, y_wires]:
            twos_complement_helper(input_reg)