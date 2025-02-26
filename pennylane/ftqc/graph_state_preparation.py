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

from typing import List, Optional

from pennylane import Operations
from .lattice import Lattice

    
class GraphStatePreparation(Operations):
    r"""
    This class represents cluster state used in the MBQC formalism.

    Args:
        ops: Gate operations
    """
    def __init__(ops: Operations):
        if len(ops.wires) == 1:
            self._init_1d_cluster_state()
        elif ops.name == 'CNOT':
            self._init_2d_cluster_state()
        else:
            raise(f"The {ops.name} gate is not supported.")
    
    def _init_1d_cluster_state(self):
        # for 1 qubit gate
    
    def _init_2d_cluster_state(self):
        # for CNOT gate 
