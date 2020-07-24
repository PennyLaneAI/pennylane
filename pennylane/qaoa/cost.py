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
r"""
This file contains functions that generate cost Hamiltonians corresponding to
different optimization problems, for use in QAOA workflows.
"""

import networkx
from pennylane.wires import Wires
import pennylane as qml


def MaxCut(graph):
    r"""A method that builds a QAOA cost Hamiltonian corresponding to the MaxCut problem, for a given graph.

    The goal of the MaxCut problem for a particular graph is to find the cut of t

    This problem is of some practical relevance in physics and circuit design, but within quantum computing,
    MaxCut is an easy problem to implement and solve with QAOA, making it useful for benchmarking NISQ quantum devices.

    Args:
         graph (networkx.Graph) A graph defining the pairs of indices of wires on which each term of the Hamiltonian acts.

    Returns:
        ~.Hamiltonian:
    """

    ##############
    # Input checks

    if isinstance(graph, networkx.Graph):
        graph = graph.edges

    else:
        raise ValueError(
            "Inputted graph must be a networkx.Graph object, got {}".format(type(graph).__name__)
        )

    ##############

    coeffs = []
    obs = []

    for g in graph:

        obs.append(qml.Identity(Wires(g[0])) @ qml.Identity(Wires(g[1])))
        coeffs.append(0.5)

        obs.append(qml.PauliZ(Wires(g[0])) @ qml.PauliZ(Wires(g[1])))
        coeffs.append(-0.5)

    return qml.Hamiltonian(coeffs, obs)
