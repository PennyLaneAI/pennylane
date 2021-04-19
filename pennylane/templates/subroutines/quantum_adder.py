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

class QuantumAdder(Operation):
    r"""
    Quantum Plain Adder circuit.

    This performs the transformation: 

    .. math::
        |a_0...a_n\rangle |b_0...b_n\rangle |0\rangle ^{\oplus (n+1)}\rightarrow |a_0...a_n\rangle |a_1+b_1...a_n+b_n\rangle |a_0+b_0\rangle |0\rangle ^{n}

    .. figure:: ../../_static/templates/subroutines/QuantumAdder.svg
        :align: center
        :width: 60%
        :target: javascript:void(0);

    See `here <https://arxiv.org/abs/quant-ph/0008033v1>`__ for more information.

    Args:
        a_wires (Sequence[int]): wires containing the first value to be added
        b_wires (Sequence[int]): wires containing the second value to be added
        carry_wires (Sequence[int]): wires containing ancilla carry wires, must have one more carry wire than a or b wires

    Raises:
        ValueError: if `a_wires`, `b_wires`, and `carry_wires` share wires, `b_wires` contains less wires than `a_wires`, or
            there are not enough carry wires.

    .. UsageDetails::

        Consider the addition of :math:`a = 2 = '10'` and :math:`b = 7 = '111'`. We can do this using the ``QuantumAdder`` template as follows:

        .. code-block:: python

            a_wires = [0, 1]
            a_value = np.array([1, 0])
            b_wires = [2, 3, 4]
            b_value = np.array([1, 1, 1])
            carry_wires = [5, 6, 7, 8]

            dev = qml.device('default.qubit',wires = (len(a_wires)+len(b_wires)+len(carry_wires)))
            @qml.qnode(dev)
            def circuit():
                qml.BasisState(np.append(a_value, b_value), wires = (a_wires + b_wires))
                qml.templates.QuantumAdder(a_wires, b_wires, carry_wires)
                return qml.state()

            result = np.argmax(circuit())

        The most significant bit goes into `carry_wires[0]` and the rest into `b_wires`. We can read the result of the sum as:

        .. code-block:: python

            result = format(result, '09b')
            sum_result = result[carry_wires[0]] + result[b_wires[0]] + result[b_wires[1]] + result[b_wires[2]]

        >>> sum_result
        '1001'
        >>> int(sum_result,2)
        9
    
    



    """
    num_params = 0
    num_wires = AnyWires
    par_domain = None

    def __init__(self, a_wires, b_wires, carry_wires, do_queue=True):
        self.a_wires = list(a_wires)
        self.b_wires = list(b_wires)
        self.carry_wires = list(carry_wires)

        wires = self.a_wires + self.b_wires + self.carry_wires

        if len(self.a_wires)>len(self.b_wires):
            raise ValueError(
                "The longer bit string must be in b_wires"
            )

        if len(self.carry_wires)!=(max(len(self.a_wires),len(self.b_wires))+1):
            raise ValueError(
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
        