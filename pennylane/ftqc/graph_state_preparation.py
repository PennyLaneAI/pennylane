# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This module contains the classes and functions for creating cluster state for measurement based quantum computing (MBQC)."""

import pennylane as qml
from pennylane import Operation
from .lattice import Lattice
from .qubit_graph import QubitGraph

    
class GraphStatePreparation(Operation):
    r"""
    This class represents cluster state used in the MBQC formalism.

    Args:
        lattice: Lattice representation of qubits connectivity
        qubit_ops: Operations to prepare the initial state of each qubit
        entanglement_ops: Operations to entangle nearest qubits
        wires: QubitGraph object maps qubit to wires 
    """
    def __init__(lattice: Lattice, qubit_ops: Operation, entanglement_ops: Operation, wires: QubitGraph):
        self._lattice = lattice
        self._qubit_ops = qubit_ops
        self._entanglement_ops = entanglement_ops
        self._wires = wires
    
    def decompose(self):
        # Add qubit_ops to the queue


        # Add entanglement_ops to the queue
        
        
