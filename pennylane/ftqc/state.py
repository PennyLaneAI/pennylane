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

from abc import ABC, abstractmethod
from typing import List, Optional

from pennylane import Operations
from .lattice import Lattice


class State(ABC):
    r"""
    This class is an abstract class for the state used in MBQC formalism.
    
    Args:
        lattice: A Lattice object represent the underlying connectivity of qubits.
        init_state: Initial state of each qubit of the system. #TODO (str?)
        init_entangle_ops: Ops to entangle qubits #TODO: (TBD: qml.operations? or str?)
    """
    def __init__(self, lattice : Lattice, init_states: Optional[List] = None, init_entangle_ops: Optional[List] = None):
        self._lattice = lattice
        self._init_states = init_states
        self._init_entangle_ops =  init_entangle_ops

    @property
    def get_lattice(self):
        return self._lattice
    
    @property
    def get_init_states(self):
        return self._init_states

    @property
    def get_init_entangle_ops(self):
        return self._init_entangle_ops
    

class ClusterState(State):
    r"""
    This class represents cluster state used in the MBQC formalism.

    Args:
        ops: Gate operations or gate names? #TODO: determine the interface that cluster state accept.
    """
    def __init__(ops: Operations):
        #determine the 
    
    def _init_1d_cluster_state(self):
        # for 1 qubit gate
    
    def _init_2d_cluster_state(self):
        # for CNOT gate 
