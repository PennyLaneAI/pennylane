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
This file contains functions that generate mixer Hamiltonians for use
in QAOA workflows.
"""
import pennylane as qml

def x_mixer(wires):
    r""""Creates the basic Pauli-X mixer Hamiltonian used in the original `QAOA paper <https://arxiv.org/abs/1411.4028>`__,
    defined as:

    .. math:: H_M \ = \ \displaystyle\sum_{i} X_{i}

    where :math:`i` ranges over all qubits, and :math:`X_i` denotes the Pauli-X on the :math:`i`-th qubit
    Args:
        qubits (Iterable or Wires): The collection of wires on which the observables in the Hamiltonian are defined
    """

    ##############
    #Input checks

    if len(set(wires)) != len(wires):
        raise ValueError("Wires indices must be unique")

    ##############

    coeffs = [1 for i in wires]
    obs = [qml.PauliX(i) for i in wires]

    return qml.Hamiltonian(coeffs, obs)

def xy_mixer(graph):
    r""""Creates the generalized SWAP/XY mixer outlined in `this paper <https://arxiv.org/abs/1709.03489>`__, defined 
        as:

        .. math:: H_M \ = \ \frac{1}{2} \displaystyle\sum_{(i, j) \in E(G)} X_i X_j \ + \ Y_i Y_j

        For some graph :math:`G`, and where :math:`X_i` and :math:`Y_i` denote the Pauli-X and Pauli-Y on the :math:`i`-th 
        qubit respectively.
        Args:
            graph (networkx.Graph) A graph defining the pairs of qubits on which each term of the Hamiltonian acts.
        """

    ##############
    # Input checks

    ##############

    coeffs = 2 * [0.5 for i in graph.edges]

    obs = []
    for e in graph.edges:
        obs.append(qml.PauliX(e[0]) @ qml.PauliX(e[1]))
        obs.append(qml.PauliY(e[0]) @ qml.PauliY(e[1]))

    return qml.Hamiltonian(coeffs, obs)