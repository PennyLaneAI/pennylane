# Copyright 2021 Xanadu Quantum Technologies Inc.

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
This module contains the CommutationDAG class which is used to generate a DAG (directed acyclic graph)
representation of a quantum circuit from an Operator queue.
"""

from collections import Counter, OrderedDict, namedtuple

import networkx as nx

import pennylane as qml
import numpy as np

class CommutationDAG:
    """Represents a quantum circuit as a directed acyclic graph.
    """

    def __init__(self, ops, obs, wires):
        self._operations = ops
        self._observables = obs

        queue = ops + obs

        self.wires = wires
        self.num_wires = len(wires)

        for k, op in enumerate(queue):
            op.queue_idx = k  # store the queue index in the Operator

        self._graph = nx.DiGraph()

    def print_contents(self):
        print("Operations")


    @property
    def observables(self):
        return self._observables

    @property
    def operations(self):
        return self._operations

    @property
    def graph(self):
        return self._graph
