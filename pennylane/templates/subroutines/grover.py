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
Contains the Grover Operation template.
"""
import pennylane as qml
from pennylane.operation import AnyWires, Operation
from pennylane.ops import Hadamard, PauliX, PauliZ, MultiControlledX

class GroverDiffusionOperator(Operation):
    r"""Performs the Grover Diffusion Operator.

    .. math::

        G = 2 |s \rangle \langle s | - I \\
        = H^{\bigotimes n} \left( 2 |0\rangle \langle 0| - I ) H^{\bigotimes n}

    For this template, the operator is implemented with 

    .. figure:: ../../_static/templates/subroutines/grover.png
        :align: center
        :width: 60%
        :target: javascript:void(0);

    Args:
        wires (Union[Wires, Sequence[int], or int]): the wires to apply to

    **Example**

    For this example, we will be using three wires and ``"default.qubit"``:

    .. code-block:: python

        n_wires = 3
        wires = list(range(n_wires))
        dev = qml.device('default.qubit', wires=wires)
    
    The Grover Diffusion Operator amplifies the magnitude of the component with 
    a negative phase.  For example, if we wanted to select out the :math:`|111\rangle`
    state, we could use an oracle implementing a `CCZ` gate:

    .. code-block:: python

        def oracle():
            qml.Hadamard(wires[-1])
            qml.Toffoli(wires=wires)
            qml.Hadamard(wires[-1])
    
    We can then implement the entire Grover Search Algorithm for ``n`` iterations:

    .. code-block:: python

        @qml.qnode(dev)
        def GroverSearch(n=1):
            for wire in wires:
                qml.Hadamard(wire)

            for _ in range(n):
                oracle()
                qml.templates.GroverDiffusionOperator(wires=wires)
            return qml.probs(wires)

    >>> GroverSearch(n=1)  
    tensor([0.03125, 0.03125, 0.03125, 0.03125, 0.03125, 0.03125, 0.03125,
            0.78125], requires_grad=True)
    >>> GroverSearch(n=2)
    tensor([0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125,
        0.0078125, 0.9453125], requires_grad=True)

    """
    num_params = 0
    num_wires = AnyWires
    par_domain = None

    def expand(self):
        cntrl_str = "0" * (len(self.wires) - 1)
    
        with qml.tape.QuantumTape() as tape:
            for wire in self.wires[:-1]:
                Hadamard(wire)
            
            PauliZ(self.wires[-1])
            MultiControlledX(control_values = cntrl_str, control_wires=self.wires[:-1], wires=self.wires[-1])
            PauliZ(self.wires[-1])
            
            for wire in self.wires[:-1]:
                Hadamard(wire)
                
        return tape