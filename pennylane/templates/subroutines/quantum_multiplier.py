# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

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
Contains the QuantumMultiplier template.
"""
import pennylane as qml
from pennylane.operation import AnyWires, Operation


class QuantumMultiplier(Operation):
    r"""
    Quantum multiplier circuit.

    This performs the transformation:

    .. math::
        |a_0,...,a_n\rangle |b_0,...,b_n\rangle |0\rangle ^{\oplus n} 0\rangle ^{\oplus 2n+1)}\rightarrow |a_0,...,a_n\rangle |b_0,...,b_n\rangle |(ab)_n,...,(ab)_{2n-1}\rangle |(ab)_0,...,(ab)_{n-1}\rangle |0\rangle^{\otimes n+1}

    .. figure:: ../../_static/templates/subroutines/quantum_multiplier.svg
        :align: center
        :width: 60%
        :target: javascript:void(0);

    See `here <https://academicworks.cuny.edu/cgi/viewcontent.cgi?article=1245&context=gc_cs_tr>`__ for more information.
    """
    num_params = 0
    num_wires = AnyWires
    par_domain = None

    def __init__(
        self,
        multiplicand_wires,
        multiplier_wires,
        accumulator_wires,
        carry_wires,
        work_wire,
        do_queue=True,
    ):

        self.multiplicand_wires = list(multiplicand_wires)
        self.multiplier_wires = list(multiplier_wires)
        self.accumulator_wires = list(accumulator_wires)
        self.carry_wires = list(carry_wires)
        self.work_wire = work_wire

        wires = (
            self.multiplicand_wires
            + self.multiplier_wires
            + self.accumulator_wires
            + self.carry_wires
            + [self.work_wire]
        )

        super().__init__(wires=wires, do_queue=do_queue)

    def expand(self):
        with qml.tape.QuantumTape() as tape:
            # have to perform controlled addition for each control wire
            # start from least significant digit in controlled wires
            # as we increase the significance, we have to add |0> digits to the end of multiplicand

            for ctrl in self.multiplier_wires[
                ::-1
            ]:  # we go backwards because the last wire is the least significant
                # every control step adds two bits to the bitstring, okay
                # the total number of carry wires should reflect this
                # accumulator + multiplicand
                qml.templates.ControlledQuantumAdder(
                    a_wires=self.multiplicand_wires,
                    b_wires=self.accumulator_wires,
                    carry_wires=self.carry_wires[0 : len(self.accumulator_wires) + 1],
                    work_wire=self.work_wire,
                    control_wire=ctrl,
                )
                # update accumulator and carry lists
                self.accumulator_wires = [self.carry_wires[0]] + self.accumulator_wires
                del self.carry_wires[0]

                # we can remove the last accumulator wire, (to account for adding 0's to end of multiplicand)
                del self.accumulator_wires[-1]
        return tape
