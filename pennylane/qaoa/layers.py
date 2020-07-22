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

import pennylane as qml
from pennylane.templates import ApproxTimeEvolution


def cost_layer(hamiltonian):

    ##############
    # Input checks

    if not isinstance(hamiltonian, qml.Hamiltonian):
        raise ValueError("`hamiltonian` must be of type pennylane.Hamiltonian, got {}".format(type(hamiltonian).__name__))

    if not hamiltonian.is_diagonal():
        raise ValueError("`hamiltonian` must be diagonal in the computational basis")

    ##############

    return lambda gamma, wires : ApproxTimeEvolution(hamiltonian, wires, gamma, n=1)

def mixer_layer(hamiltonian):

    ##############
    # Input checks

    if not isinstance(hamiltonian, qml.Hamiltonian):
        raise ValueError("`hamiltonian` must be of type pennylane.Hamiltonian, got {}".format(type(hamiltonian).__name__))

    ##############

    return lambda alpha, wires : ApproxTimeEvolution(hamiltonian, wires, alpha, n=1)

