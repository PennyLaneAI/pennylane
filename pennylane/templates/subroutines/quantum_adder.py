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
Contains the QuantumAdder template.
"""
import pennylane as qml
from pennylane.operation import AnyWires, Operation
from pennylane import numpy as np

class QuantumAdder(Operation):
    r"""

    """

    def __init__(self, a_wires, b_wires, carry_wires, do_queue=True):
        self.a_wires = list(a_wires)
        self.b_wires = list(b_wires)
        self.carry_wires = list(carry_wires)

        wires = self.a_wires + self.b_wires + self.carry_wires

        if any(set(a_wires) & set(b_wires) & set(carry_wires)):
            raise qml.QuantumFunctionError(
                "The value a wires, value b wires, and carry wires must be different"
            )

        if len(self.a_wires)!=len(self.b_wires):
            raise qml.QuantumFunctionError(
                "The value a wires and value b wires must be of the same length"
            )

        if len(self.carry_wires)!=(len(a_wires)+1):
            raise qml.QuantumFunctionError(
                "The carry wires must have one more wire than the a and b wires"
            )

        super().__init__(wires=wires, do_queue=do_queue)


    #support different size input summands
    def expand(self):
        temp = [self.carry_wires[0]]+list(self.b_wires)
        # need to know if b_wires or a_wires is larger
        ab_wires = [len(self.a_wires),len(self.b_wires)]
        ab_wires.sort()
        loga = ab_wires[1]
        
        # if they're equal, run normally
        # if one is larger, use carry_0 as replacement for smaller one in later instances
        with qml.tape.QuantumTape as tape:
            #carry operations
            for i in range(ab_wires[1]-1,-1,-1):
                qml.QubitCarry(wires=[self.carry_wires[i+1],self.a_wires[i],self.b_wires[i],self.carry_wires[i]])
            #CNOT and Sum in middle
            qml.CNOT(wires=[self.a_wires[0],self.b_wires[0]])
            qml.QubitSum(wires=[self.carry_wires[1],self.a_wires[0],self.b_wires[0]])

            for i in range(1,ab_wires[1]):
                qml.QubitCarry(wires=[self.carry_wires[i+1],self.a_wires[i],self.b_wires[i],self.carry_wires[i]]).inv()
                qml.QubitSum(wires=[self.carry_wires[i+1],self.a_wires[i],self.b_wires[i]])

        return tape
