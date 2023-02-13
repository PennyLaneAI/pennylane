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

from .parametrized_hamiltonian import ParametrizedHamiltonian


class RydbergEnsemble:
    """Class representing the interaction of an ensemble of Rydberg atoms.

    Args:
        qubit_positions (list): list of coordinates of each atom in the ensemble
        wires (list): List of wires for each atom in the ensemble. If ``None``, the wire values
            correspond to the index of the atom in the list of qubit positions. Defaults to ``None``.

    Returns:
        RydbergEnsemble: class representing an ensemble of Rydberg atoms
    """

    def __init__(self, qubit_positions: list, wires: list = None) -> None:
        self.qubit_positions = qubit_positions
        self._rydberg_interaction = None
        self.wires = wires or range(len(qubit_positions))

    @property
    def rydberg_interaction(self) -> Sum:
        r"""Returns the hamiltonian of the interaction of the ensemble of atoms due to the Rydberg
        blockade:

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
        if self._rydberg_interaction is not None:
            return self._rydberg_interaction

        def rydberg_projector(wire: int) -> SProd:
            """Returns the projector into the Rydberg state for the given wire.

            Args:
                wire (int): _description_

            Returns:
                SProd: projector into the Rydberg state
            """
            return (1 - qml.PauliZ(wire)) / 2

        coeffs = []
        ops = []
        for idx, (pos1, wire1) in enumerate(zip(self.qubit_positions[1:], self.wires[1:])):
            for pos2, wire2 in zip(self.qubit_positions[: idx + 1], self.wires[: idx + 1]):
                atom_distance = np.linalg.norm(qml.math.array(pos2) - pos1)
                Vij = 1 / (abs(atom_distance) ** 6)  # van der Waals potential
                coeffs.append(Vij)
                ops.append(qml.prod(rydberg_projector(wire1), rydberg_projector(wire2)))

        self._rydberg_interaction = qml.dot(coeffs, ops)
        return self._rydberg_interaction

    def drive(
        self, amplitudes: list, detunings: list, phases: list, wires: list
    ) -> ParametrizedHamiltonian:
        r"""Returns a :class:`ParametrizedHamiltonian` describing the evolution of the Rydberg ensemble
        when driving the given atoms (``wires``) with lasers with the corresponding ``amplitude``
        and ``detuning``:

        .. math::

            H = \frac{\hbar}{2} \sum_i  \Omega_i(t) (\cos(\phi)\sigma_i^x - \sin(\phi)\sigma_i^y) -
            \frac{\hbar}{2} \sum_i \delta_i(t) \sigma_i^z + \sum_{i<j} U_{ij} n_i n_j

        where :math:`\Omega_i` and :math:`\delta_i` correspond to the amplitude and detuning of the
        laser applied to atom :math:`i`, and :math:`\sigma^\alpha` for :math:`\alpha = x,y,z` are
        the Pauli matrices.

        The last term of the sum corresponds to the interaction between the atoms of the ensemble
        due to the Rydberg blockade. For more info, check :seealso:`rydberg_interaction`.

        .. note::

            :math:`\hbar` is set to 1.

        Args:
            amplitudes (list): list of laser amplitudes
            detunings (list): list of laser detuning parameters
            wires (list): list of wires to drive

        Returns:
            ParametrizedHamiltonian: hamiltonian describing the laser driving
        """
        ops = [
            qml.math.cos(p) * qml.PauliX(w) - qml.math.sin(p) * qml.PauliY(w)
            for p, w in zip(phases, wires)
        ]
        H1 = (1 / 2) * qml.dot(amplitudes, ops)
        H2 = -1 / 2 * qml.dot(detunings, [qml.PauliZ(w) for w in wires])
        return H1 + H2 + self.rydberg_interaction
