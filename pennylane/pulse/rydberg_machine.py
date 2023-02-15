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
from pennylane.wires import Wires

from .parametrized_hamiltonian import ParametrizedHamiltonian


class RydbergMachine:
    r"""Class representing the interaction of an ensemble of Rydberg atoms.

    Args:
        coordinates (list): List of coordinates (in micrometers) of each atom in the ensemble.
        wires (list): List of wires for each atom in the ensemble. If ``None``, the wire values
            correspond to the index of the atom in the list of qubit positions. Defaults to ``None``.
        interaction (float): Rydberg interaction constant in units: :math:`MHz \times \mu m^6`.
            Defaults to :math:`862690 \times 2\pi MHz \times \mu m^6`.

    Returns:
        RydbergEnsemble: class representing an ensemble of Rydberg atoms
    """

    def __init__(
        self, coordinates: list, wires: list = None, interaction_coeff: float = 862690 * np.pi
    ) -> None:
        if len(coordinates) != len(wires):
            raise ValueError("Coordinates and wires must have the same length.")
        self.coordinates = coordinates
        self.interaction_coeff = interaction_coeff
        self.wires = Wires(wires) or Wires(range(len(coordinates)))
        self._rydberg_interaction = None
        self._driving_interaction = ParametrizedHamiltonian([], [])
        # The following 2 dictionaries are only needed to be able to run these laser drivings in hardware
        # Dictionary containing the information about the local driving fields
        self._local_drives = {"rabi": [], "detunings": [], "phases": [], "wires": []}
        # List of tuples containing the information about the global driving field
        self._global_drive = None  # (rabi frequency, detuning, phase)

    @property
    def rydberg_interaction(self) -> Sum:
        r"""Returns the hamiltonian of the interaction of the ensemble of atoms due to the Rydberg
        blockade:

        .. math::

            \sum_{i<j} V_{ij} n_i n_j

        where :math:`n_i` corresponds to the projector on the Rydberg state of the atom :math:`i`, and
        :math:`V_{ij}` is the van der Waals potential:

        .. math::

            V_{ij} = \frac{C_6}{R_{ij}^6}

        where :math:`R_{ij}` is the distance between the atoms :math:`i` and :math:`j`, and :math:`C_6`
        is the Rydberg interaction constant, which can be set during initialization, and defaults
        to :math:`862690 \times 2\pi MHz \times \mu m^6`.

        Args:
            qubit_positions (list): list of coordinates of the Rydberg atoms
            separation (float): separation between the Rydberg atoms (in meters)
            wires (list): List of wires containing the wire values for all the atoms. This list should
                have the same length as ``qubit_positions``.

        Returns:
            Sum: hamiltonian of the interaction
        """
        if self._rydberg_interaction is None:

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
            for idx, (pos1, wire1) in enumerate(zip(self.coordinates[1:], self.wires[1:])):
                for pos2, wire2 in zip(self.coordinates[: idx + 1], self.wires[: idx + 1]):
                    atom_distance = np.linalg.norm(qml.math.array(pos2) - pos1)
                    Vij = self.interaction_coeff / (
                        abs(atom_distance) ** 6
                    )  # van der Waals potential
                    coeffs.append(Vij)
                    ops.append(qml.prod(rydberg_projector(wire1), rydberg_projector(wire2)))

            self._rydberg_interaction = qml.dot(coeffs, ops)

        return self._rydberg_interaction

    def local_drive(self, rabi: list, detunings: list, phases: list, wires: list):
        """Apply ``N = len(rabi)``  local driving laser fields with the given rabi frequencies,
        detunings and phases acting on the given wires.

        Args:
            rabi (list): list of Rabi frequency values (in MHz) of each driving laser field
            detunings (list): list of detuning values (in MHz) of each driving laser field
            phases (list): list of phases (in radiants) of each driving laser field
            wires (list): list of wire values that each laser field acts on
        """
        lengths = [len(rabi), len(detunings), len(phases), len(wires)]
        if len(set(lengths)) > 1:
            raise ValueError(
                "The lists containing the driving parameters must all have the same"
                f"length. Got lengths: {lengths}"
            )
        if any(wire not in self.wires for wire in wires):
            raise ValueError(
                "The wires list contains a wire value that is not present in the RydbergMachine."
            )
        # Update `_driving_interaction` Hamiltonian
        ops = [
            qml.math.cos(p) * qml.PauliX(w) - qml.math.sin(p) * qml.PauliY(w)
            for p, w in zip(phases, wires)
        ]
        H1 = qml.dot(rabi, ops)
        H2 = qml.dot(detunings, [qml.PauliZ(w) for w in wires])
        self._driving_interaction += 1 / 2 * H1 + 1 / 2 * H2
        # Update dictionaries of applied pulses to allow translation into hardware
        self._local_drives["rabi"].extend(rabi)
        self._local_drives["detunings"].extend(detunings)
        self._local_drives["phases"].extend(phases)
        self._local_drives["wires"].extend(wires)

    def global_drive(self, rabi: float, detuning: float, phase: float):
        """Apply ``N = len(rabi)``  global driving laser fields with the given rabi frequencies,
        detunings and phases acting on all wires.

        Args:
            rabi (float): rabi frequency (in MHz) of the global driving laser field
            detuning (float): detuning (in MHz) of the global driving laser field
            phase (float): phase (in radiants) of the global driving laser field
        """
        # Update `_driving_interaction` Hamiltonian
        ops = qml.sum(
            *[
                qml.math.cos(phase) * qml.PauliX(w) - qml.math.sin(phase) * qml.PauliY(w)
                for w in self.wires
            ]
        )
        H1 = rabi * ops
        H2 = detuning * qml.sum(*[qml.PauliZ(w) for w in self.wires])
        self._driving_interaction += 1 / 2 * H1 + 1 / 2 * H2
        # Update dictionaries of applied pulses to allow translation into hardware
        self._global_drive = (rabi, detuning, phase)

    @property
    def driving_interaction(self) -> ParametrizedHamiltonian:
        r"""Returns a :class:`ParametrizedHamiltonian` describing the evolution of the Rydberg ensemble
        when driving the given atoms (``wires``) with lasers with the corresponding ``amplitude``
        and ``detuning``:

        .. math::

            H = \hbar \frac{1}{2} \sum_i  \Omega_i(t) (\cos(\phi)\sigma_i^x - \sin(\phi)\sigma_i^y) -
            \frac{1}{2} \sum_i \delta_i(t) \sigma_i^z

        where :math:`\Omega_i` and :math:`\delta_i` correspond to the amplitude and detuning of the
        laser applied to atom :math:`i`, and :math:`\sigma^\alpha` for :math:`\alpha = x,y,z` are
        the Pauli matrices.

        .. note::

            :math:`\hbar` is set to 1.

        Returns:
            ParametrizedHamiltonian: hamiltonian describing the laser driving
        """
        return self._driving_interaction

    @property
    def hamiltonian(self) -> ParametrizedHamiltonian:
        r"""Returns a :class:`ParametrizedHamiltonian` describing the Hamiltonian of the array
        of Rydberg atoms:

        .. math::

            H = \hbar \frac{1}{2} \sum_i  \Omega_i(t) (\cos(\phi)\sigma_i^x - \sin(\phi)\sigma_i^y) -
            \frac{1}{2} \sum_i \delta_i(t) \sigma_i^z + \sum_{i<j} V_{ij} n_i n_j

        where :math:`\Omega_i` and :math:`\delta_i` correspond to the amplitude and detuning of the
        laser applied to atom :math:`i`, and :math:`\sigma^\alpha` for :math:`\alpha = x,y,z` are
        the Pauli matrices.

        The last term of the sum corresponds to the interaction between the atoms of the ensemble
        due to the Rydberg blockade. For more info, check :seealso:`rydberg_interaction`.

        .. note::

            :math:`\hbar` is set to 1.

        Returns:
            ParametrizedHamiltonian: hamiltonian describing the evolution of the array of Rydberg atoms
        """
        return self.rydberg_interaction + self.driving_interaction

    @property
    def ops(self):
        """Return the operators of the Hamiltonian.

        Returns:
            Iterable[Observable]: observables in the Hamiltonian expression
        """
        return self.hamiltonian.ops

    @property
    def coeffs(self):
        """Return the coefficients of the Hamiltonian.

        Returns:
            Iterable[float]: coefficients in the Hamiltonian expression
        """
        return self.hamiltonian.coeffs

    def __call__(self, params, t):
        return self.hamiltonian(params, t)
