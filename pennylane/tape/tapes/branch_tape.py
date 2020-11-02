# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

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
This module contains a branching quantum tape.
"""
import pennylane as qml

class BranchTape(qml.tape.QuantumTape):
    """A quantum tape recorder whose queue records elements which are interpreted as alternatives to each other.

       A QuantumTape containing BranchTape(s) or of the class BranchTape is interpreted as a compact representation
       of multiple quantum tapes.


        **Example**

       .. code-block:: python

           with QuantumTape() as tape:
               qml.CNOT(wires=['a', 0])

               with BranchTape() as batch:
                   qml.RY(0.2, wires='a')
                   qml.RX(0.2, wires='a')

               probs(wires=0), probs(wires='a')

       The ``tape`` now contains a ``BranchTape`` object:

       >>> tape.operations
       [CNOT(wires=[0, 'a']), <BranchTape: wires=[0], n=2>]

       We can also branch measurements:

        .. code-block:: python

           with QuantumTape() as tape:

               qml.CNOT(wires=['a', 0])

               with BranchTape() as branches:
                   probs(wires=0)
                   probs(wires='a')

       >>> tape.measurements
       [<BranchTape: wires=[0, 'a'], n=2>]


       """

    def __init__(self, name=None):
        # Todo: make uniquely identifyable name
        super().__init__(name=name)

    def __repr__(self):
        return (
            f"<{self.__class__.__name__}: wires={self.wires.tolist()} "
            f"n={self.num_branches}, name={self.name}>"
        )

    @property
    def num_branches(self):
        return len(self.operations) + len(self.observables)
