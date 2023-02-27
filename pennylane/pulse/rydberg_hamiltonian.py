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
import warnings
from dataclasses import dataclass
from typing import Callable, List, Union

import numpy as np

import pennylane as qml
from pennylane.ops import SProd
from pennylane.wires import Wires

from .parametrized_hamiltonian import ParametrizedHamiltonian


def rydberg_interaction(register: list, wires=None, interaction_coeff: float = 862690 * np.pi):
    r"""Returns a :class:`ParametrizedHamiltonian` representing the interaction of an ensemble of
    Rydberg atoms due to the Rydberg blockade:

    .. math::

        \sum_{i<j} V_{ij} n_i n_j

    where :math:`n_i` corresponds to the projector on the Rydberg state of the atom :math:`i`, and
    :math:`V_{ij}` is the van der Waals potential:

    .. math::

        V_{ij} = \frac{C_6}{R_{ij}^6}

    where :math:`R_{ij}` is the distance between the atoms :math:`i` and :math:`j`, and :math:`C_6`
    is the Rydberg interaction constant, which defaults to :math:`862690 \times 2\pi MHz \times \mu m^6`.

    Args:
        register (list): list of coordinates of the Rydberg atoms
        wires (list): List of wires containing the wire values for all the atoms. This list should
            have the same length as ``qubit_positions``. If ``None``, each atom's wire value will
            correspond to its index in the ``register`` list.
        interaction_coeff (float): Rydberg interaction constant in units: :math:`MHz \times \mu m^6`.
            Defaults to :math:`862690 \times 2\pi MHz \times \mu m^6`.

    Returns:
        RydbergHamiltonian: Hamiltonian representing the atom interaction
    """
    wires = wires or list(range(len(register)))

    def rydberg_projector(wire: int) -> SProd:
        """Returns the projector into the Rydberg state for the given wire.

        Args:
            wire (int): _description_

        Returns:
            SProd: projector into the Rydberg state
        """
        return qml.s_prod(1 / 2, (1 - qml.PauliZ(wire)))

    coeffs = []
    observables = []
    for idx, (pos1, wire1) in enumerate(zip(register[:-1], wires[:-1])):
        for pos2, wire2 in zip(register[(idx + 1) :], wires[(idx + 1) :]):
            atom_distance = np.linalg.norm(qml.math.array(pos2) - pos1)
            Vij = interaction_coeff / (abs(atom_distance) ** 6)  # van der Waals potential
            coeffs.append(Vij)
            observables.append(qml.prod(rydberg_projector(wire1), rydberg_projector(wire2)))

    return RydbergHamiltonian(
        coeffs, observables, register=register, interaction_coeff=interaction_coeff
    )


def rydberg_transition(rabi, detuning, phase, wires):
    r"""Returns a :class:`ParametrizedHamiltonian` representing the action of a driving laser
    field with the given rabi frequency, detuning and phase acting on the given wires:

    .. math::

        \hbar \frac{1}{2} \Omega(t) \sum_i (\cos(\phi)\sigma_i^x - \sin(\phi)\sigma_i^y) -
        \frac{1}{2} \delta(t) \sum_i \sigma_i^z

    where :math:`\Omega` and :math:`\delta` correspond to the rabi and detuning of the
    laser, :math:`i` correspond to the wire index, and :math:`\sigma^\alpha` for
    :math:`\alpha = x,y,z` are the Pauli matrices.

    Args:
        rabi (Union[float, Callable]): float or callable returning the frequency (in MHz) of a laser
            field
        detuning (Union[float, Callable]): float or callable returning the detuning (in MHz) of a
            laser field
        phase (float): float containing the phase (in radiants) of the laser field
        wires (Union[int, List[int]]): integer or list containing wire values that the laser field
            acts on

    Returns:
        RydbergHamiltonian: Hamiltonian representing the action of the laser field on the
            Rydberg atoms
    """
    if isinstance(wires, int):
        wires = [wires]

    # We compute the `coeffs` and `observables` of the laser field
    coeffs = [rabi, detuning]
    rabi_observable = sum(
        qml.math.cos(phase) * qml.PauliX(wire) - qml.math.sin(phase) * qml.PauliY(wire)
        for wire in wires
    )
    detuning_observable = sum(qml.PauliZ(wire) for wire in wires)
    observables = [rabi_observable, detuning_observable]

    # We convert the pulse data into a list of ``RydbergPulse`` objects
    pulses = [RydbergPulse(rabi, phase, detuning, wires)]
    return RydbergHamiltonian(coeffs, observables, pulses=pulses)


class RydbergHamiltonian(ParametrizedHamiltonian):
    r"""Class representing the Hamiltonian of an ensemble of Rydberg atoms under the action of
    local and global laser fields:

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

    .. warning::

        When adding a ``RydbergHamiltonian`` with a :class:`ParametrizedHamiltonian` all the
        information needed to translate this class into hardware will be lost.

    .. seealso:: :class:`rydberg_interaction`, :class:`rydberg_transition`


    Args:
        coeffs (Union[float, callable]): coefficients of the Hamiltonian expression, which may be
            constants or parametrized functions. All functions passed as ``coeffs`` must have two
            arguments, the first one being the trainable parameters and the second one being time.
        observables (Iterable[Observable]): observables in the Hamiltonian expression, of same
            length as ``coeffs``

    Keyword Args:
        register (list): list of coordinates (in micrometers) of each atom in the ensemble
        transition (list): information about the amplitude, phase, detuning and wires of the pulses
        interaction_coeff (float): Rydberg interaction constant in units: :math:`MHz \times \mu m^6`.
            Defaults to :math:`862690 \times 2\pi MHz \times \mu m^6`.

    Returns:
        RydbergHamiltonian: class representing the Hamiltonian of an ensemble of Rydberg atoms
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        coeffs,
        observables,
        register: list = None,
        pulses: List["RydbergPulse"] = None,
        interaction_coeff: float = 862690 * np.pi,
    ):
        self.register = register
        self.pulses = [] if pulses is None else pulses
        self.interaction_coeff = interaction_coeff
        super().__init__(coeffs, observables)

    def __add__(self, other):
        if not isinstance(other, RydbergHamiltonian):
            return super().__add__(other)

        # Update coeffs, obs and hardware attributes
        if self.register is not None:
            if other.register is not None:
                raise ValueError("We cannot add two Hamiltonians with an interaction term!")
            if not other.wires.contains_wires(self.wires):
                warnings.warn(
                    "The wires of the laser fields are not present in the Rydberg ensemble."
                )
        elif other.register is not None and not self.wires.contains_wires(other.wires):
            warnings.warn("The wires of the laser fields are not present in the Rydberg ensemble.")

        new_register = self.register or other.register
        new_pulses = self.pulses + other.pulses
        new_ops = self.ops + other.ops
        new_coeffs = self.coeffs + other.coeffs
        return RydbergHamiltonian(new_coeffs, new_ops, register=new_register, pulses=new_pulses)


@dataclass
class RydbergPulse:
    """Dataclass that contains the information of a single Rydberg pulse. This class is used
    internally in PL to group into a single object all the data related to a single laser field.

    Args:
        rabi (Union[float, Callable]): float or callable returning the frequency (in MHz) of a laser
            field
        detuning (Union[float, Callable]): float or callable returning the detuning (in MHz) of a
            laser field
        phase (float): float containing the phase (in radiants) of the laser field
        wires (Union[int, List[int]]): integer or list containing wire values that the laser field
            acts on
    """

    rabi: Union[float, Callable]
    phase: Union[float, Callable]
    detuning: float
    wires: List[Wires]

    def __post_init__(self):
        self.wires = Wires(self.wires)

    def __eq__(self, other):
        return (
            self.rabi == other.rabi
            and self.phase == other.phase
            and self.detuning == other.detuning
            and self.wires == other.wires
        )
