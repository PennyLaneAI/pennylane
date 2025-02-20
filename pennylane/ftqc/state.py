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
from .lattice import Lattice


class State(ABC):
    r"""
    This class is an abstract class for the state used in MQBC formalism.
    
    Args:
        lattice: A Lattice object represent the underlying connectivity of qubits.
        init_state: Initial state of each qubit of the system.


    """
    def __init__(self, lattice : Lattice, init_states: Optional[List] = None, entangle_ops = Optional[List] = None):
        self._lattice = lattice
        self._init_states = init_states
        self._init_entanglements =  entangle_ops

