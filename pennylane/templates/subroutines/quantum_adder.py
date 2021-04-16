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
    Quantum Plain Adder circuit <https://arxiv.org/abs/quant-ph/0008033v1>

    Given two sets of wires for input values and a third set of wires to act as ancillas for carry operations,
    this template applies the circuit for the quantum plain adder.

    .. figure:: ../../_static/templates/subroutines/qpe.svg
        :align: center
        :width: 60%
        :target: javascript:void(0);
    """
    num_params = 0
    num_wires = AnyWires
    par_domain = None

    def __init__(self, a_wires, b_wires, carry_wires, do_queue=True):
        self.a_wires = list(a_wires)
        self.b_wires = list(b_wires)
        self.carry_wires = list(carry_wires)

        wires = self.a_wires + self.b_wires + self.carry_wires

        if any(set(a_wires) & set(b_wires) & set(carry_wires)):
            raise qml.QuantumFunctionError(
                "The value a wires, value b wires, and carry wires must be different"
            )

        if len(self.a_wires)>len(self.b_wires):
            raise qml.QuantumFunctionError(
                "The longer bit string should go in b_wires"
            )

        if len(self.carry_wires)!=(max(len(self.a_wires),len(self.b_wires))+1):
            raise qml.QuantumFunctionError(
                "The carry wires must have one more wire than the a and b wires"
            )

        super().__init__(wires=wires, do_queue=do_queue)


    #support different size input summands
    def expand(self):
        # if they're equal, run normally
        # if one is larger, use carry_0 as replacement for smaller one in later instances
        temp = len(self.b_wires) - len(self.a_wires)
        with qml.tape.QuantumTape() as tape:
            #Initial carry operations
            for i in range(len(self.b_wires)-1,-1,-1):
                if i<temp:
                    #different length bit strings means we don't need to use full qubitcarry
                    qml.Toffoli(wires=[self.carry_wires[i+1],self.b_wires[i],self.carry_wires[i]])
                else:
                    qml.QubitCarry(wires=[self.carry_wires[i+1],self.a_wires[i-temp],self.b_wires[i],self.carry_wires[i]])

            #CNOT and Sum in the middle
            #CNOT is between b and 0 value carry if len(b)!=len(a)
            if temp>0:
                #don't need the CNOT, it will never activate
                #sum becomes a single CNOT
                qml.CNOT(wires=[self.carry_wires[1],self.b_wires[0]])
            else:
            #CNOT is between a and b if they are the same length bit strings
                qml.CNOT(wires=[self.a_wires[0],self.b_wires[0]])
                qml.QubitSum(wires=[self.carry_wires[1],self.a_wires[0],self.b_wires[0]])
                
            

            #Final carry and sum cascade
            for i in range(1,len(self.b_wires)):
                if i<temp:
                    #here summing most significant bits, a doesn't contribute
                    #carry becomes a toffoli
                    #Sum becomes CNOT
                    qml.Toffoli(wires=[self.carry_wires[i+1],self.b_wires[i],self.carry_wires[i]])
                    qml.CNOT(wires=[self.carry_wires[i+1],self.b_wires[i]])
                else:
                    qml.QubitCarry(wires=[self.carry_wires[i+1],self.a_wires[i-temp],self.b_wires[i],self.carry_wires[i]]).inv()
                    qml.QubitSum(wires=[self.carry_wires[i+1],self.a_wires[i-temp],self.b_wires[i]])

        return tape
        