# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

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
Contains the OutMultiplier template.
"""

import numpy as np
import pennylane as qml
from pennylane.operation import Operation

class OutMultiplier(Operation):
    r"""Performs the Outplace Multiplication operation in the computational basis.
    
    This operator multiplies the integers :math:`x` and :math:`y` modulo :math:`mod` in the computational basis:

    .. math::
        OutMultiplier(mod) |x \rangle |y \rangle |0 \rangle = |x \rangle |y \rangle |x*y \textrm{mod} mod \rangle,

    The quantum circuit that represents the OutMultiplier operator is:


    Args: 
        x_wires (Sequence[int]): the wires that stores the integer :math:`x`.
        y_wires (Sequence[int]): the wires that stores the integer :math:`y`.
        output_wires (Sequence[int]): the wires that stores the multiplication modulo mod :math:`x*y \textrm{mod} mod`. 
        mod (int): modulo with respect to which the multiplication is performed, default value will be ``2^len(output_wires)``
        work_wires (Sequence[int]): the auxiliary wires to use for the multiplication modulo :math:`mod` when :math:`mod \neq 2^{\textrm{len(output_wires)}}`

    **Example**

    Multiplication of two integers :math:`x=2` and :math:`y=7` modulo :math:`mod=12`. Note that to perform this multiplication using qml.OutMultiplier we need that :math:`m,k < mod`.
    
    .. code-block::
        x=2
        y=7
        mod=12
        x_wires=[0,1]
        y_wires=[2,3,4]
        output_wires=[6,7,8,9]
        work_wires=[5,10]
        dev = qml.device("default.qubit", shots=1)
        @qml.qnode(dev)
        def circuit_OutMultiplier():
            qml.BasisEmbedding(x, wires=x_wires)
            qml.BasisEmbedding(y, wires=y_wires)
            qml.OutMultiplier(x_wires, y_wires, output_wires, mod, work_wires)
            return qml.sample(wires=output_wires)

    .. code-block:: pycon

        >>> print(f"The ket representation of {x} * {y} mod {mod} is {circuit_OutMultiplier()}")
        The ket representation of 2 * 7 mod 12 is [0 0 1 0]
    
    We can see that the result [0 0 1 0] corresponds to 2, which comes from :math:`2*7=14 \longrightarrow 14 mod 12 = 2`.
    """

    grad_method = None
    
    def __init__(self, x_wires, y_wires, output_wires, mod=None, work_wires=None, id=None):

        if mod==None: 
            mod=2**(len(output_wires))
        if (not hasattr(output_wires, "__len__")) or (mod > 2**(len(output_wires))):
            raise ValueError("OutAdder must have at least enough wires to represent mod.")
        if work_wires!=None:
            if any(wire in work_wires for wire in x_wires):
                raise ValueError("Any wire in work_wires should not be included in x_wires.")
            if any(wire in work_wires for wire in y_wires):
                raise ValueError("Any wire in work_wires should not be included in y_wires.")
        else:
            max_wire=max(max(x_wires),max(y_wires),max(output_wires))
            work_wires = [max_wire + 1, max_wire + 2]
        x_wires=qml.wires.Wires(x_wires)
        y_wires=qml.wires.Wires(y_wires)
        output_wires=qml.wires.Wires(output_wires)
        work_wires=qml.wires.Wires(work_wires)
        self.hyperparameters["x_wires"] = x_wires
        self.hyperparameters["y_wires"] = y_wires 
        self.hyperparameters["output_wires"] = output_wires 
        self.hyperparameters["mod"] = mod
        self.hyperparameters["work_wires"] = work_wires 
        all_wires=x_wires+y_wires+output_wires+work_wires
        super().__init__(wires=all_wires,id=id)

    @property
    def num_params(self):
        return 0

    @staticmethod
    def compute_decomposition(x_wires, y_wires, output_wires, mod, work_wires,**kwargs):
        r"""Representation of the operator as a product of other operators.
        Args:
            x_wires (Sequence[int]): the wires that stores the integer :math:`x`.
            y_wires (Sequence[int]): the wires that stores the integer :math:`y`.
            output_wires (Sequence[int]): the wires that stores the sum modulo mod :math:`x+y mod mod`. 
            mod (int): modulo with respect to which the sum is performed, default value will be ``2^len(output_wires)``
            work_wires (Sequence[int]): the auxiliary wires to use for the sum modulo :math:`mod` when :math:`mod \neq 2^{\textrm{len(output_wires)}}`
        Returns:
            list[.Operator]: Decomposition of the operator

        **Example**

        >>> qml.OutMultiplier.compute_decomposition(x_wires=[0,1], y_wires=[2,3], output_wires=[5,6], mod=4, work_wires=[4,7])
        [QFT(wires=[5, 6]),
        ControlledSequence(ControlledSequence(PhaseAdder(wires=[5, 6]), control=[0, 1]), control=[2, 3]),
        Adjoint(QFT(wires=[5, 6]))]
        """
        op_list = []
        if (mod!=2**(len(output_wires))):
            qft_output_wires= work_wires[:1] + output_wires
        else:
            qft_output_wires = output_wires
        op_list.append(qml.QFT(wires=qft_output_wires))
        op_list.append(qml.ControlledSequence(qml.ControlledSequence(qml.PhaseAdder(1, output_wires,mod,work_wires), control = x_wires),control=y_wires))
        op_list.append(qml.adjoint(qml.QFT)(wires=qft_output_wires))
        
        return op_list