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
from pennylane.operation import Operator
from pennylane.ops.qubit.hamiltonian import Hamiltonian

from .parametrized_hamiltonian import ParametrizedHamiltonian


def rydberg_interaction(register: list, wires=None, interaction_coeff: float = 862690 * 2 * np.pi):
    r"""Returns a :class:`ParametrizedHamiltonian` representing the interaction of an ensemble of
    Rydberg atoms due to the Rydberg blockade

    .. math::

        \sum_{i<j} V_{ij} n_i n_j

    where :math:`n_i` corresponds to the projector on the Rydberg state of the atom :math:`i`, and
    :math:`V_{ij}` is the van der Waals potential:

    .. math::

        V_{ij} = \frac{C_6}{R_{ij}^6}

    where :math:`R_{ij}` is the distance between the atoms :math:`i` and :math:`j`, and :math:`C_6`
    is the Rydberg interaction constant, which defaults to :math:`862690 \times 2\pi MHz \times \mu m^6`.

    Args:
        register (list): list of coordinates of the Rydberg atoms (in micrometers)
        wires (list): List of wires containing the wire values for all the atoms. This list should
            have the same length as ``register``. If ``None``, each atom's wire value will
            correspond to its index in the ``register`` list.
        interaction_coeff (float): Rydberg interaction constant in units: :math:`\text{MHz} \times \mu \text{m}^6`.
            Defaults to :math:`862690 \times 2\pi \text{ MHz} \times \mu \text{m}^6`.

    Returns:
        RydbergHamiltonian: a :class:`~.ParametrizedHamiltonian` representing the atom interaction
    """
    if wires is None:
        wires = list(range(len(register)))
    elif len(wires) != len(register):
        raise ValueError("The length of the wires and the register must match.")

    def rydberg_projector(wire: int) -> SProd:
        """Returns the projector into the Rydberg state for the given wire.

        Args:
            wire (int): wire value

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


def rydberg_drive(rabi, phase, detuning, wires):
    r"""Returns a :class:`ParametrizedHamiltonian` representing the action of a driving laser
    field with the given rabi frequency, detuning and phase acting on the given wires

    .. math::

        \frac{1}{2} \Omega(t) \sum_i (\cos(\phi)\sigma_i^x - \sin(\phi)\sigma_i^y) -
        \frac{1}{2} \delta(t) \sum_i \sigma_i^z

    where :math:`\Omega`, :math:`\delta` and :math:`\phi` correspond to the rabi frequency, detuning
    and phase of the laser, :math:`i` correspond to the wire index, and :math:`\sigma^\alpha` for
    :math:`\alpha = x,y,z` are the Pauli matrices.

    Args:
        rabi (Union[float, Callable]): float or callable returning the frequency (in MHz) of a laser
            field
        phase (float): float containing the phase (in radians) of the laser field
        detuning (Union[float, Callable]): float or callable returning the detuning (in MHz) of a
            laser field
        wires (Union[int, List[int]]): integer or list containing wire values that the laser field
            acts on

    Returns:
        RydbergHamiltonian: Hamiltonian representing the action of the laser field on the Rydberg atoms
    """
    if isinstance(wires, int):
        wires = [wires]

    # We compute the `coeffs` and `observables` of the laser field
    coeffs = [
        amplitude_and_phase(qml.math.cos, rabi, phase),
        amplitude_and_phase(qml.math.sin, rabi, phase),
        detuning,
    ]

    drive_terms_1 = sum(qml.PauliX(wire) for wire in wires)
    drive_terms_2 = sum(-qml.PauliY(wire) for wire in wires)
    drive_terms_3 = sum(qml.PauliZ(wire) for wire in wires)

    observables = [drive_terms_1, drive_terms_2, drive_terms_3]

    # We convert the pulse data into a list of ``RydbergPulse`` objects
    pulses = [RydbergPulse(rabi, detuning, phase, wires)]

    return RydbergHamiltonian(coeffs, observables, pulses=pulses)


def amplitude_and_phase(trig_fn, amp, phase):
    """Wrapper function for combining amplitude and phase into a single callalbe
    (or constant if neither amplitude nor phase are callalbe)."""

    def callable_amp_and_phase(params, t):
        return amp(params[0], t) * trig_fn(phase(params[1], t))

    def callable_amp(params, t):
        return amp(params, t) * trig_fn(phase)

    def callable_phase(params, t):
        return amp * trig_fn(phase(params, t))

    if callable(amp):
        if callable(phase):
            return callable_amp_and_phase
        return callable_amp

    if callable(phase):
        return callable_phase

    return amp * trig_fn(phase)


class RydbergHamiltonian(ParametrizedHamiltonian):
    r"""Internal class used to keep track of the needed information to translate a `RydbergHamiltonian`
    into hardware.

    This class contains the ``coeffs`` and the ``observables`` that represent one or more
    terms of the Hamiltonian of an ensemble of Rydberg atoms under the action of local and global
    laser fields:

    .. math::

        H = \frac{1}{2} \sum_i  \Omega_i(t) (\cos(\phi_i)\sigma_i^x - \sin(\phi_i)\sigma_i^y) -
        \frac{1}{2} \sum_i \delta_i(t) \sigma_i^z + \sum_{i<j} V_{ij} n_i n_j

    Additionally, it also contains two more attributes (``register`` and ``pulses``) that contain
    the information that the hardware needs to execute this Hamiltonian.

    .. warning::

        This class should NEVER be initialized directly! Please use the functions
        :func:`rydberg_interaction` and :func:`rydberg_drive` instead.

    .. seealso:: :func:`rydberg_interaction`, :func:`rydberg_drive`, :class:`ParametrizedHamiltonian`

    Args:
        coeffs (Union[float, callable]): coefficients of the Hamiltonian expression, which may be
            constants or parametrized functions. All functions passed as ``coeffs`` must have two
            arguments, the first one being the trainable parameters and the second one being time.
        observables (Iterable[Observable]): observables in the Hamiltonian expression, of same
            length as ``coeffs``

    Keyword Args:
        register (list): list of coordinates (in micrometers) of each atom in the ensemble
        pulses (list): list of ``RydbergPulse`` classes containing the information about the
            amplitude, phase, detuning and wires of each pulse
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
        interaction_coeff: float = 862690 * 2 * np.pi,
    ):
        self.register = register
        self.pulses = [] if pulses is None else pulses
        self.interaction_coeff = interaction_coeff
        super().__init__(coeffs, observables)

    def __call__(self, params, t):
        params = self._reorder_parameters(params)
        return super().__call__(params, t)

    def _reorder_parameters(self, params):
        """Takes `params`, and reorganizes it based on whether the Hamiltonian has
        callable phase and/or callable amplitude.

        Consolidates phase and amplitude parameters in the case that both are callable,
        and duplicates phase and/or amplitude parameters if either are callables, since
        they will be passed to two operators in the Hamiltonian"""

        reordered_params = []

        coeff_idx = 0
        params_idx = 0

        for i, coeff in enumerate(self.coeffs_parametrized):
            if i == coeff_idx:
                if coeff.__name__ == "callable_amp_and_phase":
                    # add the joined parameters twice, and skip an index
                    reordered_params.append([params[params_idx], params[params_idx + 1]])
                    reordered_params.append([params[params_idx], params[params_idx + 1]])
                    coeff_idx += 2
                    params_idx += 2
                elif coeff.__name__ in ["callable_amp", "callable_phase"]:
                    reordered_params.append(params[params_idx])
                    reordered_params.append(params[params_idx])
                    coeff_idx += 2
                    params_idx += 1
                else:
                    reordered_params.append(params[params_idx])
                    coeff_idx += 1
                    params_idx += 1

        return reordered_params

    def __add__(self, other):
        if isinstance(other, RydbergHamiltonian):
            # Update coeffs, obs and hardware attributes
            if self.register is not None:
                if other.register is not None:
                    raise ValueError("We cannot add two Hamiltonians with an interaction term!")
                if not self.wires.contains_wires(other.wires):
                    warnings.warn(
                        "The wires of the laser fields are not present in the Rydberg ensemble."
                    )
            elif other.register is not None and not other.wires.contains_wires(self.wires):
                warnings.warn(
                    "The wires of the laser fields are not present in the Rydberg ensemble."
                )

            new_register = self.register or other.register
            new_pulses = self.pulses + other.pulses
            new_ops = self.ops + other.ops
            new_coeffs = self.coeffs + other.coeffs
            return RydbergHamiltonian(new_coeffs, new_ops, register=new_register, pulses=new_pulses)

        ops = self.ops.copy()
        coeffs = self.coeffs.copy()
        register = self.register
        pulses = self.pulses

        if isinstance(other, (Hamiltonian, ParametrizedHamiltonian)):
            new_coeffs = coeffs + other.coeffs.copy()
            new_ops = ops + other.ops.copy()
            return RydbergHamiltonian(new_coeffs, new_ops, register=register, pulses=pulses)

        if isinstance(other, qml.ops.SProd):  # pylint: disable=no-member
            new_coeffs = coeffs + [other.scalar]
            new_ops = ops + [other.base]
            return RydbergHamiltonian(new_coeffs, new_ops, register=register, pulses=pulses)

        if isinstance(other, Operator):
            new_coeffs = coeffs + [1]
            new_ops = ops + [other]
            return RydbergHamiltonian(new_coeffs, new_ops, register=register, pulses=pulses)

        return NotImplemented


@dataclass
class RydbergPulse:
    """Dataclass that contains the information of a single Rydberg pulse. This class is used
    internally in PL to group into a single object all the data related to a single laser field.

    Args:
        rabi (Union[float, Callable]): float or callable returning the frequency (in MHz) of a laser
            field
        detuning (Union[float, Callable]): float or callable returning the detuning (in MHz) of a
            laser field
        phase (float): float containing the phase (in radians) of the laser field
        wires (Union[int, List[int]]): integer or list containing wire values that the laser field
            acts on
    """

    rabi: Union[float, Callable]
    detuning: Union[float, Callable]
    phase: float
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
