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
This file contains functions that generate cost and mixer layers for use
in QAOA workflows.
"""
import pennylane as qml
from pennylane.wires import Wires
import networkx
from .utils import check_iterable_graph
from collections.abc import Iterable

from ..templates.subroutines import ApproxTimeEvolution

def cost_layer(hamiltonian):

    ##############
    # Input checks

    if not isinstance(hamiltonian, qml.Hamiltonian):
        raise ValueError("`hamiltonian` must be of type pennylane.Hamiltonian, got {}".format(type(hamiltonian).__name__))

    for i in hamiltonian.ops:

        if isinstance(i.name, str):
            if i.name not in diagonals:
                raise ValueError("Each term of the cost Hamiltonian must be diagonal (products of Identity and PauliZ operations), got {}".format(i.name))

        if isinstance(i.name, list):
            for j in i.name:
                if j not in diagonals:
                    raise ValueError("Each term of the cost Hamiltonian must be diagonal (products of Identity and PauliZ operations), got {}".format(j))

    ##############

    return lambda gamma : ApproxTimeEvolution(hamiltonian, gamma, N=1)

def mixer_layer(hamiltonian):

    ##############
    # Input checks

    if not isinstance(hamiltonian, qml.Hamiltonian):
        raise ValueError("`hamiltonian` must be of type pennylane.Hamiltonian, got {}".format(type(hamiltonian).__name__))

    ##############

    return lambda alpha : ApproxTimeEvolution(hamiltonian, alpha, N=1)
