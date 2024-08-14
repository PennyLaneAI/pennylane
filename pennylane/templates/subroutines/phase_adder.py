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
Contains the PhaseAdder template.
"""
import numpy as np
import pennylane as qml
from pennylane.operation import Operation

def _add_k_fourier(k, wires):
    """Adds k in the Fourier basis"""
    op_list = []
    for j in range(len(wires)):
        op_list.append(qml.RZ(k * np.pi / (2**j), wires=wires[j]))
    return op_list

class PhaseAdder(Operation):
    r"""Performs the Phase Addition operation.
    
    This operator adds the integer :math:`k` modulo :math:`mod` in the Fourier basis:

    .. math::
        PhaseAdder(k,mod) |\phi (m) \rangle = |\phi (m+k mod mod) \rangle,

    where :math:`|\phi (m) \rangle` represents the :math:`| m \rangle`: state in the Fourier basis such:

    .. math::
        QFT |m \rangle = |\phi (m) \rangle.

    The quantum circuit that represents the PhaseAdder operator is:


    Args:
        k (int): number that wants to be added 
        wires (Sequence[int]): the wires the operation acts on. There are needed at least enough wires to represent k and mod.
        mod (int): modulo with respect to which the sum is performed, default value will be ``2^len(wires)``
        work_wires (Sequence[int]): the auxiliary wire to use for the sum modulo :math:`mod` when :math:`mod \neq 2^{\textrm{len(wires)}}`

    **Example**

    Sum of two integers :math:`m=5` and :math:`k=4` modulo :math:`mod=7`. Note that to perform this sum using qml.PhaseAdder we need that :math:`m,k < mod`
    
    .. code-block::
        m = 5
        k = 4
        mod = 7
        wires_m =[1,2,3]
        work_wires=[0,4]
        dev = qml.device("default.qubit", shots=1)
        @qml.qnode(dev)
        def adder_modulo(m, k, mod, wires_m, work_wires):
            # Function that performs m + k modulo mod in the computational basis
            qml.BasisEmbedding(m, wires=wires_m) 
            qml.QFT(wires=work_wires[:1]+ wires_m)
            PhaseAdder(k, wires_m, mod, work_wires)
            qml.adjoint(qml.QFT)(wires=work_wires[:1]+wires_m)
            return qml.sample(wires=wires_m)

    .. code-block:: pycon

        >>> print(f"The ket representation of {m} + {k} mod {mod} is {adder_modulo(m, k, mod,wires_m,work_wires)}")
        The ket representation of 5 + 4 mod 7 is [0 1 0]
    
    We can see that the result [0 1 0] corresponds to 2, which comes from :math:`5+4=9 \longrightarrow 9 mod 7 = 2`.
    """

    grad_method = None
    
    def __init__(self, k, wires, mod=None, work_wires=None, id=None):
        if mod==None: 
            mod=2**(len(wires))
        if (k>=mod):
            raise ValueError("The module mod must be larger than k.")
        if (not hasattr(wires, "__len__")) or (mod > 2**(len(wires))):
            raise ValueError("PhaseAdder must have at least enough wires to represent mod.")
        if work_wires!=None:
            if any(wire in work_wires for wire in wires):
                raise ValueError("Any wire in work_wires should not be included in wires.")
        else:
            work_wires = [wires[-1] + 1, wires[-1] + 2]
          
        self.hyperparameters["k"] = k
        self.hyperparameters["mod"] = mod
        self.hyperparameters["work_wires"] = qml.wires.Wires(work_wires)   
        super().__init__(wires=wires,id=id)

    @property
    def num_params(self):
        return 0

    @staticmethod
    def compute_decomposition(k, mod, work_wires, wires):
        r"""Representation of the operator as a product of other operators.
        Args:
            k (int): number that wants to be added 
            mod (int): modulo of the sum
            work_wires (Sequence[int]): the auxiliary wires to use for the sum modulo :math:`mod` when :math:`mod \neq 2^{\textrm{len(wires)}}`
            wires (Sequence[int]): the wires the operation acts on
        Returns:
            list[.Operator]: Decomposition of the operator

        **Example**

        >>> qml.PhaseAdder.compute_decomposition(k=2,mod=8,wires=[1,2,3],work_wires=[0,4])
        [RZ(6.283185307179586, wires=[1]),
        RZ(3.141592653589793, wires=[2]),
        RZ(1.5707963267948966, wires=[3])]
        """
        op_list = []
        
        if (mod==2**(len(wires))):
            # we perform m+k modulo 2^len(wires)
            op_list.extend(_add_k_fourier(k, wires))
        else:
            new_wires = work_wires[:1] + wires
            work_wire=work_wires[1]
            aux_k = new_wires[0]
            op_list.extend(_add_k_fourier(k, new_wires))
            # we implement this operators to make m+k modulo mod
            op_list.extend(qml.adjoint(_add_k_fourier)(mod, new_wires)) 
            op_list.append(qml.adjoint(qml.QFT)(wires=new_wires))
            op_list.append(qml.CNOT(wires=[aux_k, work_wire]))
            op_list.append(qml.QFT(wires=new_wires))
            op_list.extend(qml.ctrl(op, control=work_wire) for op in _add_k_fourier(mod,new_wires))
            op_list.extend(qml.adjoint(_add_k_fourier)(k, new_wires))
            op_list.append(qml.adjoint(qml.QFT)(wires=new_wires))
            op_list.append(qml.ctrl(qml.PauliX(work_wire), control=aux_k, control_values=0))
            op_list.append(qml.QFT(wires=new_wires))
            op_list.extend(_add_k_fourier(k, new_wires))
        
        return op_list