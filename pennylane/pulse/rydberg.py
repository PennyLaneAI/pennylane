# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This module contains the classes/functions needed to simulate the evolution of ensembles of
individual (trapped) rydberg atoms under the excitation of several laser fields."""
import numpy as np

import pennylane as qml
from pennylane.ops import SProd, Sum


def rydberg_interaction(qubit_positions: list, wires: list) -> Sum:
    r"""Returns the hamiltonian of the interaction of an ensemble of Rydberg atoms located in the
    given qubit positions:

    .. math::

        \sum_{i<j} U_{ij} n_i n_j

    where :math:`n_i` corresponds to the projector on the Rydberg state of the atom :math:`i`, and
    :math:`U_{ij}` is the van der Waals potential, which is proportional to:

    .. math::

        U_{ij} \propto R_{ij}^{-6}

    where :math:`R_{ij}` is the distance between the atoms :math:`i` and :math:`j`.

    Args:
        qubit_positions (list): list of coordinates of the Rydberg atoms
        separation (float): separation between the Rydberg atoms (in meters)
        wires (list): List of wires containing the wire values for all the atoms. This list should
            have the same length as ``qubit_positions``.

    Returns:
        Sum: hamiltonian of the interaction
    """

    def rydberg_projector(wire: int) -> SProd:
        """Returns the projector into the Rydberg state for the given wire.

        Args:
            wire (int): _description_

        Returns:
            SProd: projector into the Rydberg state
        """
        return (1 - qml.PauliZ(wire)) / 2

    H = 0
    for idx, (pos1, wire1) in enumerate(zip(qubit_positions[1:], wires[1:])):
        for pos2, wire2 in zip(qubit_positions[: idx + 1], wires[: idx + 1]):
            atom_distance = np.linalg.norm(qml.math.array(pos2) - pos1)
            Vij = 1 / (abs(atom_distance) ** 6)  # van der Waals potential
            H += qml.s_prod(Vij, qml.prod(rydberg_projector(wire1), rydberg_projector(wire2)))

    return H
